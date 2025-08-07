import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # Use the second GPU (index 2)
import time 
import logging
from typing import Dict, Any, Union
import torch
from src.core.graph_builder.main_graph import MainGraphBuilder
from src.tools.knowledge_tools import arxiv, wikipedia
from src.tools.vqa_tool import vqa_tool, lm_knowledge
from PIL import Image
from tqdm import tqdm
from src.evaluation.metrics_x import VQAXEvaluator

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_tools_registry() -> Dict[str, Any]:
    return {
        "vqa_tool": vqa_tool,
        "arxiv": arxiv,
        "wikipedia": wikipedia,
        "lm_knowledge": lm_knowledge,
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
    Returns tuple: (answer, explanation, success_flag, error_message)
    """
    try:
        initial_state = {"question": question, "image": image}
        
        result = graph.invoke(initial_state)
        answer = result["final_answer"]
        explanation = result["explanation"]
        
        return answer, explanation, True, None
        
    except Exception as e:
        error_message = str(e)
        sample_info = f" (Sample ID: {sample_id})" if sample_id else ""
        
        # Check for common VLLM/context length errors
        if any(keyword in error_message.lower() for keyword in [
            "context length", "maximum context", "input is too long", 
            "sequence length", "token limit", "context window"
        ]):
            logger.error(f"Context length error{sample_info}: {error_message}")
            fallback_answer = "Error: Context length exceeded"
            fallback_explanation = "Unable to process due to context length limitation"
            return fallback_answer, fallback_explanation, False, f"Context length error: {error_message}"
        
        # Check for other common errors
        elif any(keyword in error_message.lower() for keyword in [
            "cuda out of memory", "out of memory", "oom"
        ]):
            logger.error(f"Memory error{sample_info}: {error_message}")
            # Try to free up some memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            fallback_answer = "Error: Out of memory"
            fallback_explanation = "Unable to process due to memory limitation"
            return fallback_answer, fallback_explanation, False, f"Memory error: {error_message}"
        
        # Handle other general errors
        else:
            logger.error(f"General error{sample_info}: {error_message}")
            fallback_answer = "Error: Processing failed"
            fallback_explanation = "Unable to process due to unexpected error"
            return fallback_answer, fallback_explanation, False, f"General error: {error_message}"

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

    for i, sample in enumerate(tqdm(sampled, desc="Processing samples")):
        q = sample["question"]
        img = sample["image"]
        gold_answer = sample["answer"]
        gold_explanation = sample["explanation"]
        sample_id = sample["question_id"]

        print(f"\n--- Sample {i+1}/{len(sampled)} (ID: {sample_id}) ---")
        
        start_time = time.time()
        print("Invoking graph...")
        
        # Use the enhanced run_visual_qa with error handling
        pred_answer, pred_explanation, success, error_msg = run_visual_qa(
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

        # Always add results to lists (even if they're error fallbacks)
        predicted_answers.append(pred_answer)
        ground_truth_answers.append(gold_answer)
        
        sample_id_str = str(sample["question_id"])
        predicted_explanations[sample_id_str] = [pred_explanation]
        ground_truth_explanations[sample_id_str] = gold_explanation

    # Print error summary
    print(f"\n--- Processing Summary ---")
    print(f"Total samples: {len(sampled)}")
    print(f"Successful: {successful_samples}")
    print(f"Failed: {len(error_samples)}")
    
    if error_samples:
        print(f"\n--- Error Details ---")
        for error_sample in error_samples:
            print(f"Sample {error_sample['sample_id']}: {error_sample['error']}")

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
        "metrics": metrics
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
