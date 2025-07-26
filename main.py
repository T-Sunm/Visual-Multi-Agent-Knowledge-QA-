import json
import os
from typing import Dict, Any, Union
from src.core.graph_builder.main_graph import MainGraphBuilder
from src.tools.knowledge_tools import arxiv, wikipedia
from src.tools.vqa_tool import vqa_tool, lm_knowledge
from PIL import Image
from src.evaluation.evaluator.accuracy import evaluate_accuracy
from src.utils.text_processing import extract_explanation

def setup_tools_registry() -> Dict[str, Any]:
    return {
        "vqa_tool": vqa_tool,
        "arxiv": arxiv,
        "wikipedia": wikipedia,
        "lm_knowledge": lm_knowledge,
    }

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
sampled = samples[:2]

def run_visual_qa(question: str, image: Union[str, Image.Image]):
    tools_registry = setup_tools_registry()
    builder = MainGraphBuilder(tools_registry)
    graph = builder.create_main_workflow()

    initial_state = {"question": question, "image": image}

    print(f"Q: {question}")
    print(f"Image: {image}")
    print("-" * 50)

    result = graph.invoke(initial_state)
    answer = result["final_answer"]
    explanation = result["explanation"]
    return answer, explanation

def main():
    predictions = []
    references = []

    for sample in sampled:
        q = sample["question"]
        img = sample["image"]
        gold = sample["answer"]

        pred, explanation = run_visual_qa(question=q, image=img)

        print(f"Pred: {pred}")
        explanation = extract_explanation(explanation)
        print(f"Explanation: {explanation}")
        print("-" * 50)
        predictions.append(pred)
        references.append(gold)

    accuracy = evaluate_accuracy(predictions, references)
    print(f"Accuracy: {accuracy}")

if __name__ == "__main__":
    main()
