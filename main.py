import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Use the second GPU (index 1)
import time 
import logging
from typing import Dict, Any, Union
import torch
from src.core.graph_builder.main_graph import MainGraphBuilder
from src.tools.knowledge_tools import arxiv, wikipedia
from src.tools.vqa_tool import vqa_tool, lm_knowledge, dam_caption_image_tool
from src.utils.text_processing import normalize_answer
from PIL import Image
from tqdm import tqdm
from src.evaluation.metrics_x import VQAXEvaluator
from dotenv import load_dotenv
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_tools_registry() -> Dict[str, Any]:
    return {
        "vqa_tool": vqa_tool,
        "arxiv": arxiv,
        "wikipedia": wikipedia,
        "lm_knowledge": lm_knowledge,
        "analyze_image_object": dam_caption_image_tool,
    }


evaluator = VQAXEvaluator()

# 1) Load ViVQA-X dataset
json_path = "/mnt/VLAI_data/ViVQA-X/ViVQA-X_val.json"
coco_img_dir = "/mnt/VLAI_data/COCO_Images/val2014/"

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

samples = []
for item in data:
    img_path = os.path.join(coco_img_dir, item["image_name"])
    image = Image.open(img_path).convert("RGB")
    sample = {
        "question": item["question"],
        "image": image,
        "image_path": img_path,
        "explanation": item["explanation"],
        "answer": item["answer"],
        "question_id": item["question_id"]
    }
    samples.append(sample)

# Limit samples for testing
sampled = samples[:300]

def run_visual_qa(question: str, image: Union[str, Image.Image], graph, sample_id: str = None):
    """
    Run visual QA with comprehensive error handling
    Returns tuple: (full_state, success_flag, error_message)
    """
    try:
        initial_state = {"question": question, "image": image}
        
        result = graph.invoke(initial_state)
        caption = result['image_caption']
        rationales = result['rationales']
        answer = result["final_answer"]
        explanation = result["explanation"]

        full_state = {
            "question": question,
            "image_caption": caption,
            "rationales": rationales,
            "final_answer": answer,
            "explanation": explanation,
        }
        print("messages: ", result["messages"])
        print("full_state: ", full_state)
        return full_state, True, None
        
    except Exception as e:
        error_message = str(e)
        sample_info = f" (Sample ID: {sample_id})" if sample_id else ""

        logger.error(f"General error{sample_info}: {error_message}")
        return {}, False, error_message

def main():
    tools_registry = setup_tools_registry()
    builder = MainGraphBuilder(tools_registry)
    graph = builder.create_main_workflow()

    predicted_answers = []
    ground_truth_answers = []
    predicted_explanations = {}
    ground_truth_explanations = {}
    
    # Track errors for reporting
    error_samples = []
    successful_samples = 0
    detailed_results = []
    for i, sample in enumerate(tqdm(sampled, desc="Processing samples")):
        q = sample["question"]
        img = sample["image"]
        gold_answer = sample["answer"]
        gold_explanation = sample["explanation"]
        sample_id = str(sample["question_id"])

        print(f"\n--- Sample {i+1}/{len(sampled)} (ID: {sample_id}) ---")
        
        start_time = time.time()
        print("Invoking graph...")
        
        # Use the enhanced run_visual_qa with error handling
        full_state, success, error_msg = run_visual_qa(
            question=q, image=img, graph=graph, sample_id=str(sample_id)
        )
        
        end_time = time.time()
        print(f"Graph invocation finished in {end_time - start_time:.2f} seconds.")
        
        # Track success/failure
        if success:
            successful_samples += 1
            print("✅ Sample processed successfully")
        else:
            error_samples.append({
                "sample_id": sample_id,
                "question": q,
                "error": error_msg
            })
            print(f"❌ Sample failed: {error_msg}")

        # Answers
        predicted_answer = normalize_answer(full_state.get("final_answer", ""))
        ground_truth_answer = normalize_answer(gold_answer)
        
        predicted_answers.append(predicted_answer)
        ground_truth_answers.append(ground_truth_answer)
        
        # Explanations
        predicted_explanations[sample_id] = [full_state.get("explanation", "")]
        ground_truth_explanations[sample_id] = gold_explanation

        # Detailed results
        full_state["gold_answer"] = gold_answer
        detailed_results.append(full_state)

    # Print error summary
    print(f"\n--- Processing Summary ---")
    print(f"Total samples: {len(sampled)}")
    print(f"Successful: {successful_samples}")
    print(f"Failed: {len(error_samples)}")
    


    # Create a vocabulary for answers to convert them to integer indices
    # as required by compute_answer_metrics
    all_answers = sorted(list(set(predicted_answers + ground_truth_answers)))
    answer_to_idx = {ans: i for i, ans in enumerate(all_answers)}
    
    predicted_answer_indices = [answer_to_idx[ans] for ans in predicted_answers]
    ground_truth_answer_indices = [answer_to_idx[ans] for ans in ground_truth_answers]

    # Release memory from the main graph and models before explanation evaluation
    print("\nReleasing main graph and associated models from memory...")
    del graph
    del builder
    del tools_registry
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Memory released.")

    # Compute metrics using the evaluator
    print("\nComputing evaluation metrics...")
    try:
        answer_metrics = evaluator.compute_answer_metrics(
            predicted_answer_indices, ground_truth_answer_indices
        )
        
        explanation_metrics = evaluator.compute_explanation_metrics(
            predicted_explanations, ground_truth_explanations
        )

        # Combine all metrics
        metrics = {**answer_metrics, **explanation_metrics}
        
    except Exception as e:
        logger.error(f"Error computing metrics: {e}")
        metrics = {"error": f"Failed to compute metrics: {e}"}
    
    # Prepare final results object
    num_samples = len(sampled)
    results_to_save = {
        "num_samples": num_samples,
        "successful_samples": successful_samples,
        "failed_samples": len(error_samples),
        "error_details": error_samples,
        "metrics": metrics,
        "detailed_results": detailed_results
    }

    # Save results to a JSON file
    output_filename = f"evaluation_results_0_to_{num_samples}_samples.json"
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(results_to_save, f, ensure_ascii=False, indent=4)
    
    print(f"\nResults saved to {output_filename}")

    print("\n--- Evaluation Results ---")
    if isinstance(metrics, dict) and "error" not in metrics:
        for metric, value in metrics.items():
            print(f"{metric.replace('_', ' ').title()}: {value:.4f}")
    else:
        print("Metrics computation failed - check error details in output file")
    print("--------------------------")

if __name__ == "__main__":
    main()