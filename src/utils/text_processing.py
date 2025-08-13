import re
from typing import Tuple


# Khớp một dòng: Answer: … | Evidence: …
INLINE_PATTERN = re.compile(
    r"Answer:\s*([^|]+?)\s*\|\s*Evidence\s*:\s*(.*)",  # group1 = answer, group2 = evidence
    re.IGNORECASE | re.DOTALL,
)

MULTILINE_PATTERN = re.compile(
    r"Answer:\s*(.+?)(?:\n|\r\n|\r)\s*Evidence:\s*(.*)",  # group1 = answer, group2 = evidence
    re.IGNORECASE | re.DOTALL,
)

def extract_answer_from_result(result: str) -> Tuple[str, str]:
    """
    Trả về (answer, evidence) từ chuỗi kết quả của agent
    - answer  : nội dung sau 'Answer:' 
    - evidence: nội dung sau 'Evidence:'
    
    Hỗ trợ cả 2 format:
    1. Inline: "Answer: abc | Evidence: xyz"
    2. Multiline: "Answer: abc\nEvidence: xyz"
    """

    # Thử format mới trước (multiline)
    match = MULTILINE_PATTERN.search(result)
    if match:
        answer, evidence = match.group(1).strip(), match.group(2).strip()
    else:
        # Fallback về format cũ (inline)
        match = INLINE_PATTERN.search(result)
        if not match:
            return "", ""
        answer, evidence = match.group(1).strip(), match.group(2).strip()

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


def normalize_answer(text: str) -> str:
    text = text.lower().strip()
    
    # 1. Chuẩn hóa các câu trả lời Có/Không
    if text in ["có", "đúng", "yes", "true", "correct"]:
        return "có"
    if text in ["không", "sai", "no", "false", "incorrect"]:
        return "không"
        
    # 2. Loại bỏ các tiền tố/hậu tố phổ biến trong tiếng Việt
    # Ví dụ: "con bò" -> "bò", "cái cây" -> "cây"
    prefixes_to_remove = ["con ", "cái ", "chiếc ", "quả ", "hoa ", "màu ", "bên "]
    for prefix in prefixes_to_remove:
        if text.startswith(prefix):
            text = text[len(prefix):]
            
    # 3. Loại bỏ các ký tự đặc biệt và khoảng trắng thừa
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text