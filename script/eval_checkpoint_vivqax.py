
import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any

import torch
from torch.utils.data import DataLoader
from huggingface_hub import hf_hub_download

import random

# Add ViVQA-X baseline paths for model + dataset
PROJ_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ_ROOT)) # For import from src.evaluation.metrics_x
BASELINE_DIR = PROJ_ROOT / "ViVQA-X" / "src" / "models" / "baseline_model"
sys.path.append(str(BASELINE_DIR))
sys.path.append(str(BASELINE_DIR / "dataloaders"))

from vivqax_model import ViVQAX_Model
from dataset import VQA_X_Dataset  # uses underthesea + torchvision transforms
from src.evaluation.metrics_x import VQAXEvaluator


def load_checkpoint(ckpt_path: str, device: str):
    state = torch.load(ckpt_path, map_location=device)
    cfg = state.get("config", None)
    if cfg is None:
        raise ValueError("Checkpoint missing 'config'. Please provide a checkpoint saved with config.")
    word2idx = state["word2idx"]
    answer2idx = state["answer2idx"]
    model_state = state["model_state_dict"]
    return cfg, word2idx, answer2idx, model_state


def invert_vocab(d: Dict[Any, Any]) -> Dict[Any, Any]:
    return {v: k for k, v in d.items()}


def load_and_adapt_json(json_path: str, limit: int = None, random_subset: bool = False, seed: int = 42) -> Dict[str, Any]:
    """
    Accept both:
    - list[ {question_id, question, answer, explanation, image_name} ]
    - dict[str -> item with 'answers' list]
    Returns dict[str_id] for VQA_X_Dataset.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        items = list(data.items())
        if limit and limit > 0:
            if random_subset:
                random.seed(seed); random.shuffle(items)
            items = items[:limit]
        return {k: v for k, v in items}

    if not isinstance(data, list):
        raise ValueError("Unsupported JSON format. Expect list or dict.")

    if limit and limit > 0:
        if random_subset:
            random.seed(seed)
            data = random.sample(data, k=min(limit, len(data)))
        else:
            data = data[:limit]

    out = {}
    for i, item in enumerate(data):
        qid = str(item.get("question_id", i))
        out[qid] = {
            "question": item.get("question", ""),
            "answers": [{"answer": str(item.get("answer", "")).lower()}] if item.get("answer", "") else [],
            "explanation": item.get("explanation", ""),
            "image_name": item.get("image_name", "")
        }
    return out


def parse_args():
    p = argparse.ArgumentParser()
    # Checkpoint: local or HF
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to local best_model.pth")
    p.add_argument("--hf_repo", type=str, default="VLAI-AIVN/ViVQA-X_LSTM-Generative",
                   help="HuggingFace repo id if --checkpoint not provided")
    p.add_argument("--hf_filename", type=str, default="best_model.pth",
                   help="HuggingFace filename if --checkpoint not provided")

    p.add_argument("--json_path", type=str, required=True,
                   help="Path to ViVQA-X split JSON (e.g., ViVQA-X_val.json)")
    p.add_argument("--image_dir", type=str, required=True,
                   help="Image directory for the split (e.g., COCO val2014)")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=0, help="Limit samples for quick test; 0 = full")
    p.add_argument("--random_subset", action="store_true", help="Sample ngẫu nhiên thay vì lấy head")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--gpu", type=str, default=None, help="CUDA_VISIBLE_DEVICES, e.g., 2")
    p.add_argument("--out", type=str, default=None, help="Output metrics JSON path")
    return p.parse_args()


def main():
    args = parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    device = args.device

    # Resolve checkpoint path (local or HF)
    ckpt_path = args.checkpoint or hf_hub_download(
        repo_id=args.hf_repo,
        filename=args.hf_filename
    )

    cfg, word2idx, answer2idx, model_state = load_checkpoint(ckpt_path, device)

    # Build idx2 mappings
    idx2word = invert_vocab(word2idx)
    idx2answer = invert_vocab(answer2idx)

    # Load and adapt data
    data = load_and_adapt_json(args.json_path, limit=(args.limit if args.limit > 0 else None),
                           random_subset=getattr(args, "random_subset", False),
                           seed=getattr(args, "seed", 42))

    # Dataset with fixed vocabs from checkpoint to avoid mismatch
    ds = VQA_X_Dataset(
        data=data,
        image_dir=args.image_dir,
        word2idx=word2idx,
        idx2word=idx2word,
        answer2idx=answer2idx,
        idx2answer=idx2answer,
        max_explanation_length=cfg["model"]["max_explanation_length"],
        max_vocab_size=len(word2idx)
    )

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.startswith("cuda"),
    )

    # Model from checkpoint config sizes
    model = ViVQAX_Model(
        vocab_size=len(word2idx),
        embed_size=cfg["model"]["embed_size"],
        hidden_size=cfg["model"]["hidden_size"],
        num_layers=cfg["model"]["num_layers"],
        num_answers=len(answer2idx),
        max_explanation_length=cfg["model"]["max_explanation_length"],
        word2idx=word2idx
    ).to(device)
    model.load_state_dict(model_state)
    model.eval()

    evaluator = VQAXEvaluator(device=device)
    metrics = evaluator.evaluate(model, loader, idx2word)

    out_file = args.out or f"eval_metrics_{Path(args.json_path).stem}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"\nSaved metrics to {out_file}")


if __name__ == "__main__":
    main()