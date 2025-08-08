import evaluate
import numpy as np
import json
import os
from PIL import Image
import itertools
from processor import Processor


metric = evaluate.load("accuracy")

# 1) Load ViVQA-X dataset
json_path = "/mnt/VLAI_data/ViVQA-X/ViVQA-X_val.json"
coco_img_dir = "/mnt/VLAI_data/COCO_Images/val2014/"

# Load the dataset
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

dataset  = []
for item in data:
    img_path = os.path.join(coco_img_dir, item["image_name"])
    image = Image.open(img_path).convert("RGB")
    sample = {
        "question": item["question"],
        "image": image,
        "image_path": img_path,
        "explanation": item["explanation"],
        "answer": item["answer"],
        "question_id": item["question_id"]
    }
    dataset .append(sample)


labels = [item['answer'] for item in dataset]
flattened_labels = list(itertools.chain(*labels))
unique_labels = list(set(flattened_labels))

label2id = {label: idx for idx, label in enumerate(unique_labels)}
id2label = {idx: label for label, idx in label2id.items()}

# Replace the ids with the labels
def replace_ids(inputs):
  inputs["answer"] = [label2id[x] for x in inputs["answer"]]
  return inputs

dataset = dataset.map(replace_ids)



# Preprocessing data

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # convert the logits to their predicted class
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


