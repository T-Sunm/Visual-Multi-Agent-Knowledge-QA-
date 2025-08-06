import os
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import torch
import logging
from transformers import AutoModel

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ViVQA.beit3.HCMUS.processor import Processor
from api.vqa_router import router as vqa_router

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
        app.state.model = AutoModel.from_pretrained(
            "ngocson2002/vivqa-model", 
            trust_remote_code=True
        ).to(app.state.device)
        app.state.model.eval()
        
        # Load processor
        logger.info("Loading processor...")
        app.state.processor = Processor()
        
        logger.info("Model and processor loaded successfully!")
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