import os
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI
from huggingface_hub import hf_hub_download
import torch
import logging

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Add ViVQA-X/src to Python path to avoid hyphen import issue
vivqax_src_path = os.path.join(project_root, 'ViVQA-X', 'src')
if vivqax_src_path not in sys.path:
    sys.path.insert(0, vivqax_src_path)

from api.vqa_router import router as vqa_router
from models.baseline_model.vivqax_model import ViVQAX_Model

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and processor on startup, cleanup on shutdown"""
    try:
        # Load device
        app.state.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {app.state.device}")
        
        # Load model
        logger.info("Loading ViVQA model...")
        checkpoint_path = hf_hub_download(
            repo_id="VLAI-AIVN/ViVQA-X_LSTM-Generative",
            filename="best_model.pth"
        )

        # Load checkpoint first
        state = torch.load(checkpoint_path, map_location=app.state.device)
        app.state.ckpt = state

        cfg = state['config']
        word2idx = state['word2idx']
        answer2idx = state['answer2idx']

        # Build model using sizes from checkpoint (per HF docs)
        app.state.model = ViVQAX_Model(
            vocab_size=len(word2idx),
            embed_size=cfg['model']['embed_size'],
            hidden_size=cfg['model']['hidden_size'],
            num_layers=cfg['model']['num_layers'],
            num_answers=len(answer2idx),
            max_explanation_length=cfg['model']['max_explanation_length'],
            word2idx=word2idx
        ).to(app.state.device)

        app.state.model.load_state_dict(state["model_state_dict"])
        app.state.model.eval()
        
        
        logger.info("Model loaded successfully!")
        yield
        
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise
    finally:
        logger.info("Shutting down...")


# Create FastAPI app
app = FastAPI(
    title="Vietnamese VQA API",
    description="API for Vietnamese Visual Question Answering",
    version="1.0.0",
    lifespan=lifespan
)


# Include routers
app.include_router(vqa_router, prefix="/vqa", tags=["VQA"])


@app.get("/")
async def health_check():
    """Simple health check endpoint"""
    return {
        "status": "ok",
        "message": "Vietnamese VQA API is running",
        "model_loaded": hasattr(app.state, 'model') and app.state.model is not None
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=1235) 