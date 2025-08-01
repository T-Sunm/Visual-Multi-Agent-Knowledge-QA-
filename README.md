# Visual Multi-Agent QA System

Hệ thống đa tác nhân (multi-agent) cho trả lời câu hỏi về hình ảnh, kết hợp computer vision với khả năng tìm kiếm tri thức.

## 🚀 Tính năng

- **Multi-Agent Architecture**: Junior, Senior, và Manager analysts
- **Visual Question Answering**: Sử dụng Describe Anything Model (DAM)
- **Knowledge Retrieval**: Tích hợp ArXiv, Wikipedia, DuckDuckGo
- **LangGraph Orchestration**: Workflow management và state tracking
- **Session Memory**: Lưu trữ trạng thái conversation

## 📁 Cấu trúc dự án

```
src/
├── agents/          # Agent definitions
├── core/           # Core workflow & orchestration  
├── models/         # Data models & state definitions
├── tools/          # VQA & knowledge tools
├── utils/          # Utility functions
├── evaluation/     # Evaluation & testing
└── main.py        # Entry point
```

CUDA_VISIBLE_DEVICES=2 vllm serve Qwen/Qwen3-1.7B \
--port 1234 \
--dtype auto \
--gpu-memory-utilization 0.6 \
--max-model-len 4096 \
--enable-auto-tool-choice \
--tool-call-parser hermes \
--trust-remote-code

vllm serve Qwen/Qwen3-1.7B \
  --port 1234 \
  --dtype auto \
  --gpu-memory-utilization 0.5 \
  --max-model-len 65536 \
  --rope-scaling '{"rope_type":"yarn","factor":2.0,"original_max_position_embeddings":32768}' \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --trust-remote-code


## 🛠️ Cài đặt

```bash
# Clone repository
git clone <repo-url>
cd visual-multi-agent-qa

# Install dependencies  
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env with your API keys
```

## 🎯 Sử dụng

### 1. Command Line

```bash
python src/main.py
```

### 2. Programmatic

```python
from src.main import run_visual_qa

result = run_visual_qa(
    question="What color is the dog's fur?",
    image_url="https://example.com/dog.jpg",
    thread_id="session_1"
)
print(result)
```

### 3. Jupyter Notebook

```python
import os
os.environ['JUPYTER_RUNNING'] = '1'

from src import run_visual_qa

# Analyze image
result = run_visual_qa(
    "What breed is this dog?", 
    "path/to/image.jpg"
)
```

## 🔧 Cấu hình

### API Configuration

```python
# .env file
OPENAI_API_KEY=your_openai_key
OPENAI_BASE_URL=http://127.0.0.1:1234/v1  # Local LLM
```

### Agent Configuration

Chỉnh sửa `src/main.py` để tùy chỉnh analysts:

```python
def create_analysts():
    # Customize agent roles, tools, prompts
    pass
```

## 📊 Ví dụ

```python
# Basic visual QA
result = run_visual_qa(
    "What's in this image?",
    "https://example.com/image.jpg"
)

# Multi-perspective analysis  
result = run_visual_qa(
    "Analyze the technical details of this architecture",
    "building.jpg",
    thread_id="architecture_analysis"
)
```

## 🧪 Testing

```bash
# Run tests
python -m pytest src/evaluation/

# Run specific test
python src/evaluation/test_agents.py
```

## 📈 Performance

- **Junior Agent**: Basic VQA (~1-2s)
- **Senior Agent**: VQA + Knowledge (~3-5s)  
- **Manager Agent**: Comprehensive analysis (~5-10s)

## 🔍 Debugging

Bật debug mode:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

result = run_visual_qa(question, image, thread_id="debug")
```

## 📝 License

MIT License

## 🤝 Contributing

1. Fork repository
2. Create feature branch
3. Submit pull request

## 📞 Support

- Issues: [GitHub Issues](link)
- Documentation: [Wiki](link)
- Email: support@example.com 