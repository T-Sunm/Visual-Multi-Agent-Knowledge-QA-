from transformers import AutoModel
from transformers import AutoTokenizer

from PIL import Image
import torch
import os
import json
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ViVQA.beit3.HCMUS.processor import Processor

device = "cuda" if torch.cuda.is_available() else "cpu"

# Đường dẫn
json_path = "/mnt/VLAI_data/ViVQA-X/ViVQA-X_val.json"
coco_img_dir = "/mnt/VLAI_data/COCO_Images/val2014/"

# Đọc file JSON
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

samples = []
for item in data:
    img_path = os.path.join(coco_img_dir, item["image_name"])
    # Mở ảnh thành PIL Image
    image = Image.open(img_path).convert("RGB")
    sample = {
        "question": item["question"],
        "image": image,  # <-- PIL Image object
        "image_path": img_path,
        "explanation": item["explanation"],  # list
        "answer": item["answer"],
        "question_id": item["question_id"]
    }
    samples.append(sample)

sample_test = samples[2]
print(sample_test)
model = AutoModel.from_pretrained("ngocson2002/vivqa-model", trust_remote_code=True).to(device)
processor = Processor()

image = Image.open(sample_test["image_path"]).convert('RGB')
question = sample_test["question"]

inputs = processor(image, question, return_tensors='pt')
inputs["image"] = inputs["image"].unsqueeze(0)

model.eval()
with torch.no_grad():
    output = model(**inputs)
    logits = output.logits
    # 1. Chuyển đổi logits thành xác suất bằng softmax
    # Lấy phần tử [0] vì batch size của chúng ta là 1
    probabilities = torch.nn.functional.softmax(logits, dim=-1)[0]

    # 2. Lấy 5 xác suất cao nhất và chỉ số tương ứng
    top5_probs, top5_indices = torch.topk(probabilities, 5)
    
    # 3. In ra 5 câu trả lời có xác suất cao nhất
    print("Top 5 predictions:")
    candidates_answer = ""
    for i in range(5):
        candidates_answer += f"{i+1}. {model.config.id2label[top5_indices[i].item()]} (Probability: {top5_probs[i]:.4f})\n"
    print(candidates_answer)