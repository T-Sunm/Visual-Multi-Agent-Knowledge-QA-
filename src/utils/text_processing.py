import re
from typing import Tuple


# Khớp một dòng: Answer: … | Evidence: …
INLINE_PATTERN = re.compile(
    r"Answer:\s*([^|]+?)\s*\|\s*Evidence\s*:\s*(.*)",  # group1 = answer, group2 = evidence
    re.IGNORECASE | re.DOTALL,
)

def extract_answer_from_result(result: str) -> Tuple[str, str]:
    """
    Trả về (answer, evidence) từ chuỗi kết quả của agent
    - answer  : nội dung sau 'Answer:' (chuẩn hoá lower-case, bỏ ký tự thừa)
    - evidence: nội dung sau 'Evidence:'
    """
    if not result:
        return "", ""

    match = INLINE_PATTERN.search(result)
    if not match:
        return "", ""

    answer_raw, evidence = match.group(1).strip(), match.group(2).strip()

    # Chuẩn hoá answer: loại ký tự đầu/cuối không phải chữ số-chữ cái & về lower-case
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


def remove_think_block(text: str) -> str:
    """Removes the <think>...</think> block from a string."""
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
