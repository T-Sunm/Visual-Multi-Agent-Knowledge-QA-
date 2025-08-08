from fastapi import APIRouter, HTTPException, Request, Form, UploadFile, File
from pydantic import BaseModel, Field
from typing import List, Dict
from PIL import Image
import torch
import base64
import io
import logging

from api.utils.processor import Processor

# Setup logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()


# Response models
class PredictionItem(BaseModel):
    rank: int
    answer: str
    confidence: float


class VQAResponse(BaseModel):
    success: bool
    question: str
    predictions: str
    device_used: str


def predict_vqa(model, device, ckpt, image: Image.Image, question: str, top_k: int = 5) -> Dict:
    """
    Core VQA prediction function
    
    Args:
        model: The loaded VQA model
        device: Computing device (CPU/GPU)
        image: PIL Image object
        question: Vietnamese question string
        top_k: Number of top predictions to return
    
    Returns:
        Dictionary containing predictions and metadata
    """
    config = ckpt['config']
    word2idx = ckpt['word2idx']
    idx2answer = ckpt['idx2answer']

    processor = Processor(
        word2idx=word2idx,
        max_question_length=config.get('max_question_length', 20)
    )

    inputs = processor(image, question)
    image_tensor = inputs["image"].to(device)
    question_tensor = inputs["question"].to(device)

    with torch.no_grad():
        answer_logits, _ = model.generate_explanation(
            image=image_tensor,
            question=question_tensor,
            beam_size=3
        )
        probabilities = torch.nn.functional.softmax(answer_logits, dim=-1)[0]
        topk_probs, topk_indices = torch.topk(probabilities, top_k)
        candidates_answer = ""
        for i in range(top_k):
            idx = topk_indices[i].item()
            prob = topk_probs[i].item()
            answer = idx2answer.get(idx, "Unknown")
            candidates_answer += f"{answer} ({prob:.4f}) "
        return {
            "success": True,
            "question": question,
            "predictions": candidates_answer,
            "device_used": str(device)
        }


@router.post("/predict", response_model=VQAResponse)
async def predict_with_file(
    request: Request,
    image: UploadFile = File(..., description="Image file (jpg, png, etc.)"),
    question: str = Form(..., description="Vietnamese question about the image"),
    top_k: int = Form(5, ge=1, le=10, description="Number of top predictions (1-10)")
):
    """
    Predict VQA answer from uploaded image file
    """
    # Validate inputs
    if not question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    # Check if model is loaded
    if not hasattr(request.app.state, 'model') or request.app.state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Read and process image
        image_data = await image.read()
        pil_image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # Make prediction
        result = predict_vqa(
            request.app.state.model,
            request.app.state.device,
            request.app.state.ckpt,
            pil_image,
            question,
            top_k
        )
        
        if result["success"]:
            return VQAResponse(**result)
        else:
            raise HTTPException(status_code=500, detail=result["error"])
            
    except Exception as e:
        logger.error(f"File upload endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


@router.post("/predict_base64", response_model=VQAResponse)
async def predict_with_base64(
    request: Request,
    image_base64: str = Form(..., description="Base64 encoded image"),
    question: str = Form(..., description="Vietnamese question about the image"),
    top_k: int = Form(5, ge=1, le=10, description="Number of top predictions (1-10)")
):
    """
    Predict VQA answer from base64 encoded image
    """
    # Validate inputs
    if not question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    # Check if model is loaded
    if not hasattr(request.app.state, 'model') or request.app.state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Decode base64 image
        image_data = base64.b64decode(image_base64)
        pil_image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # Make prediction
        result = predict_vqa(
            request.app.state.model,
            request.app.state.device,
            request.app.state.ckpt,
            pil_image,
            question,
            top_k
        )
        
        if result["success"]:
            return VQAResponse(**result)
        else:
            raise HTTPException(status_code=500, detail=result["error"])
            
    except Exception as e:
        logger.error(f"Base64 endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


@router.get("/health")
async def vqa_health_check(request: Request):
    """Check VQA model health"""
    return {
        "model_loaded": hasattr(request.app.state, 'model') and request.app.state.model is not None,
        "processor_loaded": hasattr(request.app.state, 'processor') and request.app.state.processor is not None,
        "device": getattr(request.app.state, 'device', 'unknown')
    } 