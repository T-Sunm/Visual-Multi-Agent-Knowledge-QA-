import json
import os
from typing import Dict, Any, Union
from src.core.graph_builder.main_graph import MainGraphBuilder
from src.tools.knowledge_tools import arxiv, wikipedia
from src.tools.vqa_tool import vqa_tool, lm_knowledge
from PIL import Image
from tqdm import tqdm
from src.evaluation.metrics_x import VQAXEvaluator
from src.utils.text_processing import extract_explanation

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
sampled = samples[:650]

def run_visual_qa(question: str, image: Union[str, Image.Image], graph):
    initial_state = {"question": question, "image": image}

    result = graph.invoke(initial_state)
    answer = result["final_answer"]
    explanation = result["explanation"]
    return answer, explanation

def main():
    tools_registry = setup_tools_registry()
    builder = MainGraphBuilder(tools_registry)
    graph = builder.create_main_workflow()

    predicted_answers = []
    ground_truth_answers = []
    predicted_explanations = {}
    ground_truth_explanations = {}

    for sample in tqdm(sampled, desc="Processing samples"):
        q = sample["question"]
        img = sample["image"]
        gold_answer = sample["answer"]
        gold_explanation = sample["explanation"]

        pred_answer, pred_explanation = run_visual_qa(question=q, image=img, graph=graph)

        pred_explanation = extract_explanation(pred_explanation)

        predicted_answers.append(pred_answer)
        ground_truth_answers.append(gold_answer)
        
        sample_id = str(sample["question_id"])
        predicted_explanations[sample_id] = [pred_explanation]
        # The 'explanation' from the JSON is likely already a list of strings
        ground_truth_explanations[sample_id] = gold_explanation

    # Create a vocabulary for answers to convert them to integer indices
    # as required by compute_answer_metrics
    all_answers = sorted(list(set(predicted_answers + ground_truth_answers)))
    answer_to_idx = {ans: i for i, ans in enumerate(all_answers)}
    
    predicted_answer_indices = [answer_to_idx[ans] for ans in predicted_answers]
    ground_truth_answer_indices = [answer_to_idx[ans] for ans in ground_truth_answers]

    # Compute metrics using the evaluator
    print("Computing evaluation metrics...")
    answer_metrics = evaluator.compute_answer_metrics(
        predicted_answer_indices, ground_truth_answer_indices
    )
    
    explanation_metrics = evaluator.compute_explanation_metrics(
        predicted_explanations, ground_truth_explanations
    )

    # Combine all metrics
    metrics = {**answer_metrics, **explanation_metrics}
    
    # Prepare final results object
    num_samples = len(sampled)
    results_to_save = {
        "num_samples": num_samples,
        "metrics": metrics
    }

    # Save results to a JSON file
    output_filename = f"evaluation_results_{num_samples}_samples.json"
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(results_to_save, f, ensure_ascii=False, indent=4)
    
    print(f"\nResults saved to {output_filename}")

    print("\n--- Evaluation Results ---")
    for metric, value in metrics.items():
        print(f"{metric.replace('_', ' ').title()}: {value:.4f}")
    print("--------------------------")

if __name__ == "__main__":
    main()
