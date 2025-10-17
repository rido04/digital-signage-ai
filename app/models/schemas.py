from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any
from enum import Enum


class GenerationStatus(str, Enum):
    """Generation status enum"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ObjectTemplate(BaseModel):
    """Object template schema"""
    name: str
    slug: str
    category: str
    default_pose: str
    pose_variants: List[str] = []
    generation_params: Dict[str, Any] = {}
    base_prompt: Optional[str] = None
    negative_prompt: Optional[str] = None
    min_width: Optional[int] = 768
    min_height: Optional[int] = 1024
    guidance_scale: Optional[float] = 7.5
    num_inference_steps: Optional[int] = 30
    seed: Optional[int] = None
    prompt_strategy: Optional[str] = 'auto'


class InsertionArea(BaseModel):
    """Insertion area coordinates"""
    x: float
    y: float
    width: float
    height: float
    type: str = "rectangle"


class CampaignData(BaseModel):
    """Campaign data from Laravel"""
    design_image_url: str
    object_template: ObjectTemplate
    insertion_area: InsertionArea
    generation_settings: Optional[Dict[str, Any]] = {}


class GenerationRequest(BaseModel):
    """Generation request schema"""
    generation_id: str = Field(..., description="UUID from Laravel")
    campaign_data: CampaignData
    
    class Config:
        json_schema_extra = {
            "example": {
                "generation_id": "550e8400-e29b-41d4-a716-446655440000",
                "campaign_data": {
                    "design_image_url": "https://s3.amazonaws.com/bucket/design.jpg",
                    "object_template": {
                        "name": "Chair Empty",
                        "slug": "chair_empty",
                        "category": "furniture",
                        "default_pose": "sitting_upright",
                        "pose_variants": ["sitting_relaxed"],
                        "generation_params": {}
                    },
                    "insertion_area": {
                        "x": 100,
                        "y": 200,
                        "width": 300,
                        "height": 400,
                        "type": "rectangle"
                    },
                    "generation_settings": {}
                }
            }
        }


class GenerationResponse(BaseModel):
    """Generation response schema"""
    status: str
    task_id: str
    generation_id: str
    message: str = "Generation queued successfully"


class TaskStatusResponse(BaseModel):
    """Task status response"""
    task_id: str
    status: GenerationStatus
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    progress: Optional[float] = None


class CompletionCallback(BaseModel):
    """Callback data for Laravel when generation completes"""
    result_image_url: str
    processing_time_ms: int
    pose_used: str
    ai_parameters: Dict[str, Any] = {}


class FailureCallback(BaseModel):
    """Callback data for Laravel when generation fails"""
    error: str
    error_details: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    gpu_available: bool
    gpu_name: Optional[str] = None
    models_loaded: bool
    redis_connected: bool