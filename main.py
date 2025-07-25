from typing import Dict, Any, Union
from src.core.graph_builder.main_graph import MainGraphBuilder
from src.tools.knowledge_tools import arxiv, wikipedia
from src.tools.vqa_tool import vqa_tool, lm_knowledge
from datasets import load_dataset
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
    answer = result["final_answer"]
    explanation = result["explanation"]
    return answer, explanation

def main():
    predictions = []
    references = []

    for sample in sampled:
        q = sample["question"]
        img = sample["image"]
        gold = sample["multiple_choice_answer"]

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
