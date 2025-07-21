from PIL import Image
from langchain_core.tools import tool
from typing import Union
from src.tools.dam_tools import dam_candidate_answers, dam_caption_image, dam_extract_knowledge


@tool
def vqa_tool(image: Union[str, Image.Image], question: str) -> str:
    """return the candidate answer with probability of the question"""
    return dam_candidate_answers(image, question)

@tool
def dam_caption_image_tool(image: Union[str, Image.Image]) -> str:
    """return the caption of the image"""
    return dam_caption_image(image)

@tool
def lm_knowledge(image: Union[str, Image.Image], question: str, context: str = "") -> str:
    """return the knowledge of the image"""
    return dam_extract_knowledge(image, question, context)