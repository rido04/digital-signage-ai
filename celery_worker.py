"""
Celery worker for processing AI generation tasks
"""
from loguru import logger
import sys

from tasks.generation_task import celery_app
from app.utils.config import settings

# Configure logging
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - <level>{message}</level>",
    level=settings.log_level
)
logger.add(
    settings.log_file.replace('.log', '_worker.log'),
    rotation="100 MB",
    retention="30 days",
    level=settings.log_level
)

# Pre-load models when worker starts (optional but recommended)
@celery_app.task(bind=True)
def warmup_models(self):
    """Warmup task to load models"""
    try:
        logger.info("Warming up AI models...")
        
        from app.services.ai_generator import get_generator
        from app.services.face_swapper import get_face_swapper
        
        generator = get_generator()
        face_swapper = get_face_swapper()
        
        logger.info("AI models loaded and ready")
        return {"status": "ready"}
        
    except Exception as e:
        logger.error(f"Model warmup failed: {e}")
        raise


if __name__ == '__main__':
    logger.info("Starting Celery worker")
    
    # Start worker
    celery_app.worker_main([
        'worker',
        '--loglevel=info',
        '--concurrency=1',  # Process one task at a time (GPU limitation)
        '--pool=solo',      # Solo pool for better compatibility
        '--task-events',
        '--without-gossip',
        '--without-mingle',
    ])