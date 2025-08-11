import torch
from PIL import Image
from src.utils.image_processing import load_image
from transformers import AutoModel, AutoProcessor
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor as DinoProcessor
from transformers import SamModel, AutoProcessor as SamProcessor
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DAM (Describe Anything Model)
model = AutoModel.from_pretrained(
    'nvidia/DAM-3B-Self-Contained',
    trust_remote_code=True,
    torch_dtype='torch.float16'
).to(device)
dam = model.init_dam(conv_mode='v1', prompt_mode='full+focal_crop')

# Grounding DINO
gd_model_id = "IDEA-Research/grounding-dino-tiny"
gd_processor = DinoProcessor.from_pretrained(gd_model_id)
gd_model = AutoModelForZeroShotObjectDetection.from_pretrained(gd_model_id).to(device)
print("Grounding DINO model loaded.")

# SAM (Segment Anything Model)
sam_model_id = "facebook/sam-vit-base"
sam_processor = SamProcessor.from_pretrained(sam_model_id)
sam_model = SamModel.from_pretrained(sam_model_id).to(device)
print("SAM model loaded.")



def dam_candidate_answers(image: str, question: str) -> str:
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
    with torch.no_grad():
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

def dam_caption_image(image: str) -> str:
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
    with torch.no_grad():
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

def dam_extract_knowledge(image: str) -> str:
    img = load_image(image)
    full_mask = Image.new("L", img.size, 255)
    prompt = f"""
        <image>
        Provide a highly detailed description of the image.
        """.strip()
    with torch.no_grad():
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


def get_bbox_from_prompt(image: Image.Image, text_prompt: str) -> list | None:
    """Sử dụng Grounding DINO để lấy bbox từ prompt."""
    inputs = gd_processor(images=image, text=[[text_prompt]], return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = gd_model(**inputs)
    
    results = gd_processor.post_process_grounded_object_detection(
        outputs, inputs.input_ids, box_threshold=0.35, target_sizes=[image.size[::-1]]
    )
    
    result = results[0]
    if len(result["scores"]) == 0:
        return None
    
    best_idx = torch.argmax(result["scores"])
    return result["boxes"][best_idx].tolist()


def get_mask_from_bbox(image: Image.Image, bbox: list) -> np.ndarray:
    """Sử dụng SAM để lấy mask từ bbox."""
    inputs = sam_processor(image, input_boxes=[[bbox]], return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = sam_model(**inputs)
    
    masks = sam_processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
    )[0][0]
    
    scores = outputs.iou_scores[0, 0]
    best_mask_idx = scores.argmax()
    return masks[best_mask_idx].numpy()

### --- Hàm CÔNG CỤ CHÍNH đã được đóng gói ---

def describe_object_with_prompt(image: str, text_prompt: str) -> str:
    try:
        img = load_image(image)
    except Exception as e:
        return f"[Error] Failed to load image: {e}"

    # Bước 1: Tìm đối tượng bằng Grounding DINO
    bbox = get_bbox_from_prompt(img, text_prompt)
    if bbox is None:
        return f"[Error] Could not find '{text_prompt}' in the image."

    # Bước 2: Phân đoạn đối tượng bằng SAM
    mask_np = get_mask_from_bbox(img, bbox)
    mask = Image.fromarray((mask_np * 255).astype(np.uint8))

    # Bước 3: Mô tả đối tượng bằng DAM
    dam_prompt = (
        "<image>\n"
        "Provide a summary of the highlighted object by listing its key characteristics. "
        "Include both visual details from the image and relevant general knowledge.\n\n"
        "- **Object Identity:** [Name of the object]\n"
        "- **Visual Description:** [Describe visible features such as color, shape, texture...]\n"
        "- **Image Context:** [Describe the context of the object in the photo, for example, lying on the floor, being held by a person...]\n"
        "- **General Knowledge:** [An interesting fact, common use, or relevant information about this object]"
    )
    
    with torch.no_grad():
        result = dam.get_description(
            img,
            mask,
            dam_prompt,
            streaming=False,
            temperature=0.2,
            top_p=0.5,
            num_beams=1,
            max_new_tokens=512
        )
    return result
