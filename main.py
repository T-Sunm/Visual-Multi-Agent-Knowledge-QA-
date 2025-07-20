from typing import Dict, Any, Union
from src.core.graph_builder.main_graph import MainGraphBuilder
from src.tools.knowledge_tools import arxiv, wikipedia
from src.tools.vqa_tool import vqa_tool
from datasets import load_dataset
from PIL import Image
from src.evaluation.evaluator.accuracy import evaluate_accuracy

def setup_tools_registry() -> Dict[str, Any]:
    return {
        "vqa_tool": vqa_tool,
        "arxiv": arxiv,
        "wikipedia": wikipedia, 
    }

# 1) Tải về và sample 100 item
dataset = load_dataset(
    "Erland/VQAv2-sample",
    split="test",
    cache_dir="./DATA"
)
sampled = dataset.shuffle(seed=42).select(range(2))
sampled = sampled.select([1])

def run_visual_qa(question: str, image: Union[str, Image.Image]):
    tools_registry = setup_tools_registry()
    builder = MainGraphBuilder(tools_registry)
    graph = builder.create_main_workflow()

    initial_state = {"question": question, "image": image}

    print(f"Q: {question}")
    print(f"Image: {image}")
    print("-" * 50)

    result = graph.invoke(initial_state)
    
    return result["final_answer"]

def main():
    predictions = []
    references = []

    for sample in sampled:
        q = sample["question"]
        img = sample["image"]
        gold = sample["multiple_choice_answer"]

        pred = run_visual_qa(question=q, image=img)

        predictions.append(pred)
        references.append(gold)

    accuracy = evaluate_accuracy(predictions, references)
    print(f"Accuracy: {accuracy}")

if __name__ == "__main__":
    main()
