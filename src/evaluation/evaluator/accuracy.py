import re

def normalize_answer(ans: str) -> str:
    """Simple normalization for VQA answers"""
    # Convert to lowercase and remove punctuation
    ans = ans.lower().strip()
    ans = re.sub(r'[^\w\s]', '', ans)
    ans = re.sub(r'\s+', ' ', ans)
    return ans.strip()

def evaluate_accuracy(predictions, references):
    """Evaluate accuracy using exact match after normalization"""
    scores = []
    for pred, ref in zip(predictions, references):
        pred_norm = normalize_answer(str(pred))
        ref_norm = normalize_answer(str(ref))
        score = 1.0 if pred_norm == ref_norm else 0.0
        scores.append(score)
    
    accuracy = sum(scores) / len(scores) if scores else 0.0
    return {"accuracy": accuracy}