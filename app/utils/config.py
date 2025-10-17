from pydantic_settings import BaseSettings
from functools import lru_cache
import os


class Settings(BaseSettings):
    """Application Settings"""
    
    # Application
    app_name: str = "Digital Signage AI Service"
    env: str = "development"
    debug: bool = True
    
    # Server
    host: str = "0.0.0.0"
    port: int = 8001
    
    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: str = ""
    
    # Celery
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/0"
    
    # Laravel API
    laravel_api_url: str = "http://localhost:8000"
    laravel_internal_secret: str = ""
    
    # AI Models
    sd_model_id: str = "runwayml/stable-diffusion-v1-5"
    use_xformers: bool = True
    enable_cpu_offload: bool = False
    face_swap_model: str = "inswapper_128.onnx"
    
    # Image Generation
    default_image_width: int = 512
    default_image_height: int = 768
    num_inference_steps: int = 30
    guidance_scale: float = 7.5
    
    # Processing
    max_batch_size: int = 4
    gpu_memory_fraction: float = 0.8
    
    # Directories
    temp_dir: str = "./temp"
    models_dir: str = "./models"
    output_dir: str = "./output"
    
    # Logging
    log_level: str = "INFO"
    log_file: str = "./logs/ai_service.log"
    
    class Config:
        env_file = ".env"
        case_sensitive = False
    
    @property
    def redis_url(self) -> str:
        """Get Redis URL"""
        if self.redis_password:
            return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}/{self.redis_db}"
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"
    
    def ensure_directories(self):
        """Ensure required directories exist"""
        for directory in [self.temp_dir, self.models_dir, self.output_dir, os.path.dirname(self.log_file)]:
            os.makedirs(directory, exist_ok=True)


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    settings = Settings()
    settings.ensure_directories()
    return settings


# Global settings instance
settings = get_settings()