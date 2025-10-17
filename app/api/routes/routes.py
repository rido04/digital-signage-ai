from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import torch
from loguru import logger
from typing import Optional
import os
import uuid

from app.models.schemas import (
    GenerationRequest,
    GenerationResponse,
    TaskStatusResponse,
    HealthResponse
)
from app.utils.config import settings
from app.utils.helpers import save_image, check_gpu_available
from app.services.ai_generator import get_generator
from app.services.face_swapper import get_face_swapper
from tasks.generation_task import celery_app, generate_personalized_image

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        # Check GPU
        gpu_available, gpu_name = check_gpu_available()
        
        # Check models
        generator = get_generator()
        face_swapper = get_face_swapper()
        models_loaded = generator.model_loaded and face_swapper.model_loaded
        
        # Check Redis
        redis_connected = False
        try:
            celery_app.broker_connection().connect()
            redis_connected = True
        except Exception as e:
            logger.warning(f"Redis connection check failed: {e}")
        
        return HealthResponse(
            status="healthy",
            gpu_available=gpu_available,
            gpu_name=gpu_name,
            models_loaded=models_loaded,
            redis_connected=redis_connected
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/process-generation", response_model=GenerationResponse)
async def process_generation(
    generation_id: str = Form(...),
    face_image: UploadFile = File(...),
    campaign_data: str = Form(...)
):
    """Process generation request from Laravel"""
    try:
        logger.info(f"Received generation request: {generation_id}")
        
        # Save uploaded face image
        face_filename = f"{generation_id}_face.jpg"
        face_path = os.path.join(settings.temp_dir, face_filename)
        
        with open(face_path, "wb") as f:
            content = await face_image.read()
            f.write(content)
        
        logger.info(f"Face image saved to {face_path}")
        
        # Parse campaign data
        import json
        campaign_dict = json.loads(campaign_data)
        
        # ❌ HAPUS INI - jangan pakai URL lagi
        # face_image_url = f"file://{face_path}"
        
        # ✅ LANGSUNG PASS FILE PATH
        task = generate_personalized_image.delay(
            generation_id=generation_id,
            face_image_path=face_path,  # ← GANTI parameter name
            campaign_data=campaign_dict
        )
        
        logger.info(f"Task queued with ID: {task.id}")
        
        return GenerationResponse(
            status="queued",
            task_id=task.id,
            generation_id=generation_id
        )
        
    except Exception as e:
        logger.error(f"Failed to process generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/task-status/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    """
    Get status of a generation task
    
    Args:
        task_id: Celery task ID
        
    Returns:
        Task status response
    """
    try:
        task = celery_app.AsyncResult(task_id)
        
        if task.state == 'PENDING':
            response = TaskStatusResponse(
                task_id=task_id,
                status="pending",
                progress=0
            )
        elif task.state == 'PROGRESS':
            response = TaskStatusResponse(
                task_id=task_id,
                status="processing",
                progress=task.info.get('progress', 0),
                result=task.info
            )
        elif task.state == 'SUCCESS':
            response = TaskStatusResponse(
                task_id=task_id,
                status="completed",
                progress=100,
                result=task.result
            )
        elif task.state == 'FAILURE':
            response = TaskStatusResponse(
                task_id=task_id,
                status="failed",
                error=str(task.info)
            )
        else:
            response = TaskStatusResponse(
                task_id=task_id,
                status=task.state.lower(),
                progress=50
            )
        
        return response
        
    except Exception as e:
        logger.error(f"Failed to get task status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cancel-task/{task_id}")
async def cancel_task(task_id: str):
    """
    Cancel a running task
    
    Args:
        task_id: Celery task ID
        
    Returns:
        Success response
    """
    try:
        celery_app.control.revoke(task_id, terminate=True)
        logger.info(f"Task {task_id} cancelled")
        
        return JSONResponse(
            status_code=200,
            content={"status": "cancelled", "task_id": task_id}
        )
        
    except Exception as e:
        logger.error(f"Failed to cancel task: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/info")
async def get_models_info():
    """Get information about loaded models"""
    try:
        generator = get_generator()
        face_swapper = get_face_swapper()
        
        return {
            "stable_diffusion": generator.get_model_info(),
            "face_swapper": face_swapper.get_model_info(),
            "device": str(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        }
    except Exception as e:
        logger.error(f"Failed to get models info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/test/generate-person")
async def test_generate_person(
    template_name: str = Form("chair_empty"),
    pose: str = Form("sitting_upright"),
    width: int = Form(512),
    height: int = Form(768)
):
    """
    Test endpoint to generate a person (for debugging)
    
    Args:
        template_name: Template name
        pose: Pose to generate
        width: Image width
        height: Image height
        
    Returns:
        Success response with image path
    """
    try:
        logger.info(f"Test generation: {pose}")
        
        generator = get_generator()
        person_image = generator.generate_person(
            template_name=template_name,
            pose=pose,
            width=width,
            height=height
        )
        
        if person_image is None:
            raise HTTPException(status_code=500, detail="Generation failed")
        
        # Save test image
        test_filename = f"test_{uuid.uuid4()}.jpg"
        test_path = os.path.join(settings.output_dir, test_filename)
        save_image(person_image, test_path)
        
        return {
            "status": "success",
            "image_path": test_path,
            "size": f"{person_image.width}x{person_image.height}"
        }
        
    except Exception as e:
        logger.error(f"Test generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))