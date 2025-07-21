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
    """Generate a short caption describing the image."""
    return dam_caption_image(image)

@tool
def lm_knowledge(image: Union[str, Image.Image]) -> str:
    """Extract 2–3 background-knowledge facts about the scene to help reason toward an answer."""
    return dam_extract_knowledge(image)