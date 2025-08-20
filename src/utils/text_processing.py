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


def extract_answer(result: str) -> str:
    """
    Trích xuất câu trả lời từ chuỗi kết quả của agent.
    - Trả về nội dung sau 'Answer:'
    """
    match = re.search(r'Answer:\s*(.*)', result, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return result.strip() if result else ""

def extract_rationale(result: str) -> str:
    """
    Trích xuất rationale từ kết quả trả về của agent.
    Hàm sẽ tìm marker "Rationale:" và trả về nội dung theo sau.
    Nếu không tìm thấy, hàm sẽ trả về toàn bộ chuỗi kết quả (sau khi strip).
    """
    if not result:
        return ""

    # Tách chuỗi bằng regex để không phân biệt hoa thường
    parts = re.split(r'Rationale:', result, maxsplit=1, flags=re.IGNORECASE)

    # Nếu tìm thấy "Rationale:", phần thứ 2 chính là nội dung cần lấy
    if len(parts) > 1:
        return parts[1].strip()
    
    # Nếu không, trả về toàn bộ chuỗi
    return result.strip()

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
    
    # 1. Chuẩn hóa các câu trả lời Có/Không (giữ nguyên)
    if text in ["có", "đúng", "yes", "true", "correct"]:
        return "có"
    if text in ["không", "sai", "no", "false", "incorrect"]:
        return "không"
        
    # 2. Chuẩn hóa từ/cụm từ đồng nghĩa
    # Ví dụ: "bay diều", "diều bay" đều có thể được hiểu là "thả diều".
    synonym_map = {
        "bay diều": "thả diều",
        "diều bay": "thả diều",
    }
    if text in synonym_map:
        text = synonym_map[text]
        
    # 3. Loại bỏ các tiền tố/hậu tố phổ biến (giữ nguyên)
    prefixes_to_remove = ["con ", "cái ", "chiếc ", "quả ", "hoa ", "màu ", "bên ", "phía "]
    for prefix in prefixes_to_remove:
        if text.startswith(prefix):
            text = text[len(prefix):]
            
    # 4. Loại bỏ các ký tự đặc biệt và khoảng trắng thừa (giữ nguyên)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 5. Chuẩn hóa thứ tự từ
    # Ví dụ: "bò sữa" và "sữa bò" sau khi sắp xếp đều trở thành "bò sữa".
    words = sorted(text.split())
    text = " ".join(words)
    
    return text
