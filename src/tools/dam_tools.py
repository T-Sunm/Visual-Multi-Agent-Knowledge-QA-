from PIL import Image
from typing import Union
from src.utils.image_processing import load_image
from transformers import AutoModel, AutoProcessor

model = AutoModel.from_pretrained(
    'nvidia/DAM-3B-Self-Contained',
    trust_remote_code=True,
    torch_dtype='torch.float16'
).to("cuda")
dam = model.init_dam(conv_mode='v1', prompt_mode='full+focal_crop')


def dam_candidate_answers(image: Union[str, Image.Image], question: str) -> str:
    img = load_image(image)
    full_mask = Image.new("L", img.size, 255)
    prompt = f"""<image>
    You are a professional Visual Question Answering (VQA) system.
    Task  
    1. Rely *only* on information clearly visible in the image to produce 5 candidate answers, each with a confidence score (0.00–1.00, two decimal places).  
    2. Replace the placeholders “answer1”, “answer2”… with the actual answers; do **NOT** repeat the template text.  
    3. If the answer cannot be determined, write `unanswerable(0.00)` in the first position, followed by the next best guesses.  
    4. Return **exactly one single line**, with no line breaks, in this format:

    Candidates: <answer_1>(<score_1>), <answer_2>(<score_2>), <answer_3>(<score_3>), <answer_4>(<score_4>), <answer_5>(<score_5>)

    Example (for illustration — do not copy verbatim):  
    Question: What color is the car?  
    Candidates: red(0.98), orange(0.75), yellow(0.15), white(0.07), unanswerable(0.05)

    Now apply to the following image and question:

    Question: {question}  
    Answer:
    """
    result = dam.get_description(
        img,
        full_mask,
        prompt,
        streaming=False,
        temperature=0.2,
        top_p=0.5,
        num_beams=1,
        max_new_tokens=512
    )
    return result

def dam_caption_image(image: Union[str, Image.Image]) -> str:
    img = load_image(image)
    full_mask = Image.new("L", img.size, 255)
    prompt = """<image>
    You are an image captioning system.
    Given an image, describe it in one concise and factual sentence.
    – Only describe what is clearly visible in the image.
    – Do not make guesses or add subjective opinions.
    – Output only the caption on a single line, no extra text.

    Example 1:
    Caption: A man riding a bicycle on a city street.

    Example 2:
    Caption: A cat sitting on a windowsill looking outside.

    Now apply to the new image:

    Caption:"""
    result = dam.get_description(
        img,
        full_mask,
        prompt,
        streaming=False,
        temperature=0.2,
        top_p=0.5,
        num_beams=1,
        max_new_tokens=512
    )
    return result

def dam_extract_knowledge(image: Union[str, Image.Image]) -> str:
    img = load_image(image)
    full_mask = Image.new("L", img.size, 255)
    prompt = f"""
        <image>
        Provide a highly detailed description of the image.
        """.strip()
    result = dam.get_description(
        img,
        full_mask,
        prompt,
        streaming=False,
        temperature=0.2,
        top_p=0.5,
        num_beams=1,
        max_new_tokens=256
    )
    return result