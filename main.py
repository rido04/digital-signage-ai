import sys, os
sys.path.append(os.path.dirname(__file__))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from app.api.routes.routes import router
from app.utils.config import settings


# Configure logging
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level=settings.log_level
)
logger.add(
    settings.log_file,
    rotation="100 MB",
    retention="30 days",
    level=settings.log_level
)

# Initialize FastAPI app
app = FastAPI(
    title=settings.app_name,
    description="AI Service for Digital Signage - Face Swapping & Image Generation",
    version="1.0.0",
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify Laravel URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api", tags=["AI Service"])


@app.on_event("startup")
async def startup_event():
    """Run on application startup"""
    logger.info(f"Starting {settings.app_name}")
    logger.info(f"Environment: {settings.env}")
    logger.info(f"Debug mode: {settings.debug}")
    
    # Ensure directories exist
    settings.ensure_directories()
    logger.info("Directories checked")
    
    # Pre-load AI models on startup (optional, can be slow)
    # Uncomment if you want models loaded immediately
    # try:
    #     from app.services.ai_generator import get_generator
    #     from app.services.face_swapper import get_face_swapper
    #     
    #     logger.info("Pre-loading AI models...")
    #     generator = get_generator()
    #     face_swapper = get_face_swapper()
    #     logger.info("AI models loaded successfully")
    # except Exception as e:
    #     logger.error(f"Failed to pre-load models: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    """Run on application shutdown"""
    logger.info("Shutting down application")
    
    # Cleanup
    try:
        from app.services.ai_generator import get_generator
        generator = get_generator()
        generator.unload_model()
        logger.info("Models unloaded")
    except Exception as e:
        logger.warning(f"Model cleanup error: {e}")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": settings.app_name,
        "status": "running",
        "version": "1.0.0",
        "docs": "/docs" if settings.debug else "disabled"
    }


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if settings.debug else "An error occurred"
        }
    )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="info"
    )