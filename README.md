# Digital Signage AI Service

AI-powered image generation and face swapping service for interactive digital signage.

## 🚀 Features

- **Stable Diffusion** - Generate photorealistic person images
- **Face Swapping** - InsightFace for high-quality face replacement
- **Image Composition** - Intelligent blending and color matching
- **Async Processing** - Celery for background task processing
- **FastAPI** - Modern async Python web framework
- **GPU Accelerated** - CUDA support for fast generation

## 📋 Prerequisites

- Python 3.10+
- CUDA 11.8+ (for GPU acceleration)
- Redis (for Celery)
- 8GB+ RAM
- 6GB+ VRAM (GPU) recommended

## 🛠️ Installation

### 1. Clone Repository

```bash
git clone <repository>
cd digital-signage-ai-service
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt
```

### 4. Download AI Models

Models will be downloaded automatically on first run, but you can pre-download:

```bash
python -c "from app.services.ai_generator import get_generator; get_generator()"
python -c "from app.services.face_swapper import get_face_swapper; get_face_swapper()"
```

### 5. Configure Environment

```bash
cp .env.example .env
# Edit .env with your settings
```

## 🚦 Running the Service

### Option 1: Local Development

**Terminal 1 - Redis:**

```bash
redis-server
```

**Terminal 2 - FastAPI:**

```bash
python main.py
# or
uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```

**Terminal 3 - Celery Worker:**

```bash
python celery_worker.py
# or
celery -A tasks.generation_tasks worker --loglevel=info --concurrency=1 --pool=solo
```

### Option 2: Docker (Recommended for Production)

```bash
docker-compose up -d
```

Services will be available at:

- FastAPI: http://localhost:8001
- API Docs: http://localhost:8001/docs
- Flower (monitoring): http://localhost:5555

## 📡 API Endpoints

### Health Check

```bash
GET /health
```

### Process Generation

```bash
POST /api/process-generation
Content-Type: multipart/form-data

generation_id: <uuid>
face_image: <file>
campaign_data: <json>
```

### Task Status

```bash
GET /api/task-status/{task_id}
```

### Models Info

```bash
GET /api/models/info
```

## 🔧 Configuration

Key environment variables in `.env`:

```env
# AI Models
SD_MODEL_ID=runwayml/stable-diffusion-v1-5
FACE_SWAP_MODEL=inswapper_128.onnx

# Performance
USE_XFORMERS=true
ENABLE_CPU_OFFLOAD=false
NUM_INFERENCE_STEPS=30

# Laravel Integration
LARAVEL_API_URL=http://localhost:8000
LARAVEL_INTERNAL_SECRET=your-secret-key
```

## 🎨 How It Works

```
1. Receive Request
   ↓
2. Download Face Image + Design
   ↓
3. Generate Person with Stable Diffusion
   - Based on object template (chair, sofa, etc)
   - With specified pose
   ↓
4. Face Swap with InsightFace
   - Extract face from user photo
   - Swap onto generated body
   ↓
5. Composite into Design
   - Place at insertion coordinates
   - Adjust lighting & colors
   - Blend edges
   ↓
6. Save & Send Callback to Laravel
```

## 📊 Performance

**Generation Times (NVIDIA RTX 3090):**

- AI Generation: ~3-5 seconds
- Face Swap: ~1-2 seconds
- Composition: <1 second
- **Total: ~5-8 seconds**

**CPU Only (not recommended):**

- Total: ~60-120 seconds

## 🐛 Troubleshooting

### CUDA Out of Memory

```env
ENABLE_CPU_OFFLOAD=true
DEFAULT_IMAGE_WIDTH=512
DEFAULT_IMAGE_HEIGHT=768
```

### Face Not Detected

- Ensure face photo has clear frontal face
- Good lighting
- Face size > 100x100 pixels

### Slow Generation

- Enable xformers: `USE_XFORMERS=true`
- Reduce steps: `NUM_INFERENCE_STEPS=20`
- Use smaller model: `SD_MODEL_ID=runwayml/stable-diffusion-v1-5`

### Celery Worker Not Processing

```bash
# Check Redis connection
redis-cli ping

# Check Celery worker logs
tail -f logs/ai_service_worker.log

# Restart worker
celery -A tasks.generation_tasks worker --purge
```

## 📝 Project Structure

```
digital-signage-ai-service/
├── app/
│   ├── api/
│   │   └── routes.py          # FastAPI endpoints
│   ├── services/
│   │   ├── ai_generator.py    # Stable Diffusion
│   │   ├── face_swapper.py    # InsightFace
│   │   └── image_composer.py  # Image composition
│   ├── models/
│   │   └── schemas.py         # Pydantic models
│   └── utils/
│       ├── config.py           # Configuration
│       └── helpers.py          # Helper functions
├── tasks/
│   └── generation_tasks.py    # Celery tasks
├── main.py                     # FastAPI app
├── celery_worker.py           # Celery worker
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## 🔐 Security Notes

- Use strong `LARAVEL_INTERNAL_SECRET`
- Implement rate limiting in production
- Validate file uploads
- Clean up temporary files regularly
- Use HTTPS in production

## 📈 Scaling

For high traffic:

1. **Multiple Workers**

```yaml
# docker-compose.yml
worker:
  deploy:
    replicas: 3
```

2. **Load Balancer** for FastAPI instances

3. **Separate GPU Servers** for different models

4. **Cache Results** (Redis)

## 🧪 Testing

```bash
# Test API
curl http://localhost:8001/health

# Test generation (with test endpoint)
curl -X POST http://localhost:8001/api/test/generate-person \
  -F "template_name=chair_empty" \
  -F "pose=sitting_upright"
```

## 📦 Model Files

Required model files (auto-downloaded):

- Stable Diffusion: ~4GB
- InsightFace buffalo_l: ~400MB
- Face swapper: ~250MB

Total: ~5GB

## 🤝 Integration with Laravel

Laravel sends:

```json
{
  "generation_id": "uuid",
  "face_image": "file",
  "campaign_data": {
    "design_image_url": "...",
    "object_template": {...},
    "insertion_area": {...}
  }
}
```

Python AI Service responds:

```json
{
  "status": "queued",
  "task_id": "celery-task-id",
  "generation_id": "uuid"
}
```

On completion, callback to Laravel:

```
POST /api/internal/generations/{id}/complete
{
  "result_image_url": "...",
  "processing_time_ms": 5432,
  "pose_used": "sitting_upright"
}
```

## 📄 License

MIT License

## 👥 Support

For issues or questions, please contact the development team.

---

**Ready for Production!** 🚀
