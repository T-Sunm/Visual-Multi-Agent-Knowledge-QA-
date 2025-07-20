def extract_answer_from_result(result: str) -> str:
    """
    Extract the actual answer from agent result text.
    Assumes the answer is the last word/phrase after parsing.
    """
    if not result:
        return ""
    
    # Try to extract answer after "Answer:" pattern
    if "Answer:" in result:
        answer_part = result.split("Answer:")[-1].strip()
        # Get the first word/phrase as the answer
        answer = answer_part.split()[0] if answer_part.split() else ""
        return answer.lower().strip('.,!?;:"')
    
    # If no "Answer:" pattern, use the last sentence as answer
    sentences = result.strip().split('.')
    if sentences:
        answer = sentences[-1].strip()
        # Get the first word as answer if it's a short phrase
        if len(answer.split()) <= 3:
            return answer.lower().strip('.,!?;:"')
    
    # Fallback: use the whole result if it's short enough
    if len(result.split()) <= 3:
        return result.lower().strip('.,!?;:"')
    
    return ""