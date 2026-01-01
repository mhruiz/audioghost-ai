"""
Models API - Model Preloading and Status
"""
import os
from typing import Optional, Literal
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from workers.celery_app import celery_app
from workers.tasks import preload_model_task, get_model_status

router = APIRouter()


class ModelStatusResponse(BaseModel):
    current_model: Optional[str] = None  # e.g., "facebook/sam-audio-base"
    status: Literal["offline", "loading", "ready"]
    model_size: Optional[str] = None  # e.g., "base"
    gpu_memory_gb: Optional[float] = None


class PreloadRequest(BaseModel):
    model_size: str = "base"  # small, base, large
    use_float32: bool = False


class PreloadResponse(BaseModel):
    success: bool
    message: str
    task_id: Optional[str] = None


@router.get("/status", response_model=ModelStatusResponse)
async def get_status():
    """
    Get current model loading status.
    
    Returns which model is loaded (if any) and its state:
    - offline: No model loaded
    - loading: Model is being loaded
    - ready: Model is loaded and ready for inference
    """
    status_info = get_model_status()
    return ModelStatusResponse(**status_info)


@router.post("/preload", response_model=PreloadResponse)
async def preload_model(request: PreloadRequest = PreloadRequest()):
    """
    Preload a model in the background.
    
    This will load the specified model into GPU memory so it's ready
    for immediate use when processing requests.
    
    - **model_size**: small, base, or large (default: base)
    - **use_float32**: Use float32 precision instead of bfloat16
    """
    if request.model_size not in ["small", "base", "large"]:
        raise HTTPException(status_code=400, detail="Invalid model_size. Use: small, base, large")
    
    # Check current status
    current_status = get_model_status()
    
    if current_status["status"] == "loading":
        return PreloadResponse(
            success=False,
            message=f"A model is already being loaded. Please wait."
        )
    
    # Submit preload task
    task = preload_model_task.delay(request.model_size, request.use_float32)
    
    return PreloadResponse(
        success=True,
        message=f"Preloading {request.model_size} model in background",
        task_id=task.id
    )


@router.post("/unload")
async def unload_model():
    """
    Unload current model from GPU memory.
    
    This frees up GPU memory but will require reloading the model
    before the next inference.
    """
    from workers.tasks import unload_current_model
    
    result = unload_current_model()
    return {
        "success": True,
        "message": "Model unloaded",
        "freed_memory_gb": result.get("freed_memory_gb", 0)
    }
