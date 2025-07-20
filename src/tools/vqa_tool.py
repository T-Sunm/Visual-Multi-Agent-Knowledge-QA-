import torch
import numpy as np
from PIL import Image
from transformers import SamModel, SamProcessor, AutoModel
import cv2
import requests
from io import BytesIO
from langchain_core.tools import tool
from typing import Union
import base64

device = torch.device("cpu")
sam_model = SamModel.from_pretrained("facebook/sam-vit-base").to(device)
sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

model = AutoModel.from_pretrained(
    'nvidia/DAM-3B-Self-Contained',
    trust_remote_code=True,
    torch_dtype='torch.float16'
).to("cuda")
dam = model.init_dam(conv_mode='v1', prompt_mode='full+focal_crop')


# def apply_sam(image, input_points=None, input_boxes=None, input_labels=None):
#   inputs = sam_processor(image, input_points=input_points, input_boxes=input_boxes,
#                          input_labels=input_labels, return_tensors="pt").to(device)

#   with torch.no_grad():
#     outputs = sam_model(**inputs)

#   masks = sam_processor.image_processor.post_process_masks(
#       outputs.pred_masks.cpu(),
#       inputs["original_sizes"].cpu(),
#       inputs["reshaped_input_sizes"].cpu()
#   )[0][0]
#   scores = outputs.iou_scores[0, 0]

#   mask_selection_index = scores.argmax()
#   mask_np = masks[mask_selection_index].numpy()
#   return mask_np


def add_contour(img, mask, input_points=None, input_boxes=None):
  img = img.copy()
  mask = mask.astype(np.uint8) * 255
  contours, _ = cv2.findContours(
      mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  cv2.drawContours(img, contours, -1, (1.0, 1.0, 1.0), thickness=6)

  if input_points is not None:
    for points in input_points:
      for x, y in points:
        cv2.circle(img, (int(x), int(y)), radius=10,
                   color=(1.0, 0.0, 0.0), thickness=-1)
        cv2.circle(img, (int(x), int(y)), radius=10,
                   color=(1.0, 1.0, 1.0), thickness=2)

  if input_boxes is not None:
    for box_batch in input_boxes:
      for box in box_batch:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2),
                      color=(1.0, 1.0, 1.0), thickness=4)
        cv2.rectangle(img, (x1, y1), (x2, y2),
                      color=(1.0, 0.0, 0.0), thickness=2)

  return img

def print_streaming(text):
  print(text, end="", flush=True)


def candidate_answers(
    image: Union[str, Image.Image],
    question: str,
) -> str:
    # Xử lý hình ảnh từ base64 hoặc URL hoặc PIL Image
  if isinstance(image, str):
      if image.startswith('http'):
            # Nếu là URL
            resp = requests.get(image)
            resp.raise_for_status()
            img = Image.open(BytesIO(resp.content)).convert('RGB')
      else:
            # Nếu là base64 string
            img_data = base64.b64decode(image)
            img = Image.open(BytesIO(img_data)).convert('RGB')
  else:
        # Nếu là PIL Image
        img = image

  # 2. Tạo full‐mask
  full_mask = Image.new("L", img.size, 255)

  # 4. Tạo prompt
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



def caption_image(
    image: Union[str, Image.Image],
) -> str:

    if isinstance(image, str):
        resp = requests.get(image)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content)).convert("RGB")
    else:
        img = image

    # 2. Tạo full‐mask (toàn ảnh)
    full_mask = Image.new("L", img.size, 255)

    # 3. Xây prompt
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

    # 4. Gọi DAM để sinh caption
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


# if __name__ == "__main__":
#     # VD dùng hàm từ dòng lệnh
#   URL = "https://github.com/NVlabs/describe-anything/blob/main/images/1.jpg?raw=true"
#   Q = "What color is the dog's fur?"
#   result = caption_image(URL)
#   print(result)
@tool
def vqa_tool(image: Union[str, Image.Image], question: str) -> str:
    """return the candidate answer with probability of the question"""
    return candidate_answers(image, question)