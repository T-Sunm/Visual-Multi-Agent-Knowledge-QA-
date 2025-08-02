import re
from typing import Tuple


TAG_PATTERN = re.compile(
    r"<(?:think)>(.*?)</(?:think)>",  # khớp <think>…</think>
    re.DOTALL | re.IGNORECASE,
)
ANSWER_PATTERN = re.compile(r"Answer:\s*(.*?)(?:\n\n|</think>)", re.IGNORECASE | re.DOTALL)

def extract_answer_from_result(result: str) -> Tuple[str, str]:
    """
    Trả về (answer, evidence) từ chuỗi kết quả của agent.
    - answer  : nội dung sau 'Answer:' (đã chuẩn hoá về lower case & bỏ dấu câu đầu/cuối)
    - evidence: nội dung bên trong thẻ <think> 
    """
    if not result:
        return "", ""

    # --- Evidence ----------------------------------------------------------
    evidence_match = TAG_PATTERN.search(result)
    evidence = evidence_match.group(1).strip() if evidence_match else ""

    # --- Answer ------------------------------------------------------------
    answer_match = ANSWER_PATTERN.search(result)
    if answer_match:
        answer_raw = answer_match.group(1).strip()
    else:  # fallback: câu cuối cùng
        sentences = [s.strip() for s in result.strip().split('.') if s.strip()]
        answer_raw = sentences[-1] if sentences else ""

    # chuẩn hoá answer: bỏ ký tự thừa và về lower-case
    answer = re.sub(r'^[\W_]+|[\W_]+$', '', answer_raw).lower()

    return answer, evidence


def extract_explanation(result: str) -> str:
    """
    Extract the explanation from the agent's result text,
    and remove the <think>...</think> block.
    """
    if not result:
        return ""
    
    explanation_part = result
    # First, check if "Explanation:" marker exists and split from there
    if "Explanation:" in result:
        explanation_part = result.split("Explanation:", 1)[-1]
    
    # Use regex to remove the <think>...</think> block
    # re.DOTALL makes '.' match newlines as well
    cleaned_explanation = re.sub(r'<think>.*?</think>', '', explanation_part, flags=re.DOTALL)
    
    # Return the cleaned explanation, stripped of leading/trailing whitespace
    return cleaned_explanation.strip()
