from PIL import Image
import requests
from langchain_core.tools import tool
from typing import Union
from src.tools.dam_tools import dam_candidate_answers, dam_caption_image, dam_extract_knowledge, describe_object_with_prompt

VQA_API_URL = "http://localhost:1235/vqa/predict_base64"


@tool
def vqa_tool(image: str, question: str) -> str:
    """return the candidate answer with probability of the question"""
    try:
        payload = {
            "image_base64": image,
            "question": question,
            "top_k": 5
        }
        
        response = requests.post(VQA_API_URL, data=payload, timeout=30)
        
        response.raise_for_status()
        
        result = response.json()
        
        predictions = result.get("predictions", "")
        if not predictions:
            return "API trả về thành công nhưng không có dự đoán nào."    
        return predictions
            
    except Exception as e:
        return f"Error in vqa_tool: {e}"

@tool
def dam_caption_image_tool(image: str) -> str:
    """Identifies and provides a detailed description of a specific object in an image based on a text prompt."""
    return describe_object_with_prompt(image)

@tool
def lm_knowledge(image: str) -> str:
    """Extract 2–3 background-knowledge facts about the scene to help reason toward an answer."""
    return dam_extract_knowledge(image)


def vqa_tool_dam(image: str, question: str) -> str:
    """return the candidate answer with probability of the question"""
    return dam_candidate_answers(image, question)