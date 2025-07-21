import base64
from io import BytesIO
from PIL import Image
from typing import Union
import requests

def pil_to_base64(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def load_image(image: Union[str, Image.Image]) -> Image.Image:
    if isinstance(image, str):
        if image.startswith('http'):
            resp = requests.get(image)
            resp.raise_for_status()
            img = Image.open(BytesIO(resp.content)).convert('RGB')
        else:
            img_data = base64.b64decode(image)
            img = Image.open(BytesIO(img_data)).convert('RGB')
    else:
        img = image
    return img