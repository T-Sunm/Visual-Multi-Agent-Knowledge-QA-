# A knowledge-based multi-agent approach for Vietnamese VQA with rationale explanations

<!-- Badges: Add badges here, e.g., License, Python version, etc. -->
[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)

This project introduces a multi-agent system for Vietnamese Visual Question Answering (VQA). It leverages external knowledge sources to provide answers with detailed, rationale-based explanations.

---

## 📖 Introduction

Traditional Visual Question Answering (VQA) models often operate as "black boxes," providing answers without explaining their reasoning. This limitation reduces trust and utility, especially in complex scenarios that require external knowledge. This project addresses the challenge by implementing a knowledge-based multi-agent system designed to provide not just answers, but also the reasoning behind them, specifically for Vietnamese VQA.

Our approach mimics a human analytical workflow using three distinct agents orchestrated by LangGraph:
- **Junior Analyst**: Performs the initial visual analysis using a foundational VQA model to generate a basic description and answer.
- **Senior Analyst**: Enriches the initial findings by querying external knowledge bases (e.g., Wikipedia, ArXiv, DuckDuckGo) to gather relevant context and verify information.
- **Manager Analyst**: Synthesizes all gathered information—visual descriptions, initial answers, and external knowledge—to produce a final, comprehensive answer complete with a step-by-step rationale.

This hierarchical process ensures that the final output is robust, verifiable, and explainable, bridging the gap between simple VQA and true visual understanding.

## 🚀 Quick Start

This section provides a complete, step-by-step guide to setting up a clean environment and running the project.

### 1. Prerequisites
- **Conda**: To manage dependencies in an isolated environment.
- **Python 3.10+**
- **API Keys**: For any external services you wish to use (e.g., OpenAI).

### 2. Installation

#### For ViVqa tool
To ensure a conflict-free setup, please follow these steps precisely.

**Step A: Create and Activate a New Conda Environment**
```bash
# Create a new conda environment named 'vivqa-minh' with Python 3.10
conda create -n vivqa-minh python=3.10 -y

# Activate the new environment
conda activate vivqa-minh
```

**Step B: Clone the Repository**
```bash
# Clone the repository and navigate into the directory
git clone https://github.com/T-Sunm/Visual-Multi-Agent-Knowledge-QA-.git
cd Visual-Multi-Agent-Knowledge-QA-
```

**Step C: Manual Package Installation**

Due to conflicting dependencies when using vivqa tool, you **must** install packages in the following specific order.
```bash
1. pip install timm underthesea efficientnet_pytorch transformers
2. pip install salesforce-lavis
3. pip install torchscale
```
**Step D: Fastapi Package Installation**
```bash
pip install fastapi==0.115.12 uvicorn[standard]==0.34.2 python-multipart
```
### 3. Usage

The system can be run with a local or remote LLM.

#### Step 3a: Run the LLM Server (for Local Models)
If you are using a local model with VLLM, open a new terminal, activate the `vivqa-minh` conda environment, and start the server. The following is an example command for the Qwen model.

```bash
# Command to serve a local LLM with VLLM
CUDA_VISIBLE_DEVICES=2 \
vllm serve Qwen/Qwen3-1.7B \
  --port 1234 \
  --dtype auto \
  --gpu-memory-utilization 0.4 \
  --max-model-len 4096 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --trust-remote-code
```

#### Step 3b: Run a Sample Query
Once the environment is set up (and the local LLM server is running, if applicable), you can run a query from the command line.

```bash
python src/main.py \
  --question "What color is the dog's fur in the image?" \
  --image "path/to/your/image.jpg"
```

## 📁 Repository Structure

The project is organized as follows:

```
.
├── src/
│   ├── agents/          # Contains the logic for each agent (Junior, Senior, Manager).
│   ├── core/            # Implements the core multi-agent graph using LangGraph.
│   ├── models/          # Defines Pydantic models for state management.
│   ├── tools/           # Houses tools for VQA and external knowledge retrieval.
│   ├── utils/           # Includes utility functions and helper scripts.
│   ├── evaluation/      # Scripts for evaluating model and agent performance.
│   └── main.py          # Entry point for running the application.
│
├── ViVQA/               # Submodule containing the underlying Vietnamese VQA model implementation.
│
├── notebooks/           # Jupyter notebooks for experimentation and analysis.
│
├── .env.example         # Example environment file.
└── requirements.txt     # Project dependencies and detailed installation guide.
```

## 📊 Dataset

This system is designed to work with Vietnamese VQA datasets. The integrated `ViVQA` model was trained on **ViVQA** and **ViVQA-X**, which uses images from the MS COCO dataset.

### Data Aquisition
- For information on the dataset and the underlying model, please refer to the original repository: [ngocson1042002/ViVQA](https://github.com/ngocson1042002/ViVQA).
- You will need to acquire the MS COCO image dataset and the corresponding Vietnamese question/answer files.

### Expected Structure
The scripts expect the data to be organized in a specific structure. The code often references a root directory like `/mnt/VLAI_data/`. A typical structure would be:
```
/path/to/your/data/
├── COCO_Images/
│   ├── train2014/
│   └── val2014/
│
└── ViVQA-X/
    ├── ViVQA-X_train.json
    └── ViVQA-X_val.json
```

## 🔧 Training & Evaluation

This section provides instructions for training the model and evaluating system performance.

### Training
```bash
# Command to start the training process
python ViVQA/beit3/HCMUS/main.py --train-path ... --val-path ...
```

### Evaluation
```bash
# Command to run evaluation tests
python -m pytest src/evaluation/
```

## 📜 Citation

If this project is based on a research paper, add the BibTeX citation information here.

## 📝 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details. 