import requests
import httpx
from PIL import Image
from io import BytesIO
import torch
import os
from loguru import logger
from typing import Optional, Dict, Any
from .config import settings


def download_image(url: str) -> Optional[Image.Image]:
    """Download image from URL"""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
        return image.convert('RGB')
    except Exception as e:
        logger.error(f"Failed to download image from {url}: {e}")
        return None


def save_image(image: Image.Image, path: str) -> bool:
    """Save PIL Image to file"""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        image.save(path, 'JPEG', quality=95)
        logger.info(f"Image saved to {path}")
        return True
    except Exception as e:
        logger.error(f"Failed to save image to {path}: {e}")
        return False


def check_gpu_available() -> tuple[bool, Optional[str]]:
    """Check if GPU is available"""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"GPU available: {gpu_name}")
        return True, gpu_name
    else:
        logger.warning("GPU not available, will use CPU")
        return False, None


# def get_device() -> torch.device:
#     """Get PyTorch device (cuda or cpu)"""
#     if torch.cuda.is_available():
#         return torch.device("cuda")
#     return torch.device("cpu")


async def send_callback_to_laravel(
    endpoint: str,
    data: dict,
    timeout: int = 30
) -> bool:
    """Send callback to Laravel API"""
    url = f"{settings.laravel_api_url}/api/internal{endpoint}"
    headers = {
        "X-Internal-Secret": settings.laravel_internal_secret,
        "Content-Type": "application/json"
    }
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=data, headers=headers, timeout=timeout)
            response.raise_for_status()
            logger.info(f"Callback sent successfully to {url}")
            return True
    except Exception as e:
        logger.error(f"Failed to send callback to {url}: {e}")
        return False


def resize_image_to_fit(
    image: Image.Image,
    target_width: int,
    target_height: int,
    maintain_aspect: bool = True
) -> Image.Image:
    """Resize image to fit target dimensions"""
    if maintain_aspect:
        image.thumbnail((target_width, target_height), Image.Resampling.LANCZOS)
    else:
        image = image.resize((target_width, target_height), Image.Resampling.LANCZOS)
    
    return image

def substitute_prompt_placeholders(
    prompt_template: str,
    pose: str = "standing",  # ✅ ADD PARAMETER
    detected_gender: str = "neutral",
    generation_settings: Dict[str, Any] = None
) -> str:
    """
    Replace placeholders in prompt template
    
    Args:
        prompt_template: Template with {gender}, {clothing}, {pose} placeholders
        pose: Pose name (e.g., "sitting", "standing")
        detected_gender: Detected gender from face
        generation_settings: Settings containing clothing_styles
        
    Returns:
        Prompt with substituted values
    """
    prompt = prompt_template
    
    # Replace {gender}
    gender_map = {
        "male": "man",
        "female": "woman",
        "neutral": "person"
    }
    prompt = prompt.replace('{gender}', gender_map.get(detected_gender, "person"))
    
    # ✅ ADD: Replace {pose}
    pose_readable = pose.replace('_', ' ')  # "sitting_upright" → "sitting upright"
    prompt = prompt.replace('{pose}', pose_readable)
    
    # Replace {clothing}
    clothing = "casual clothing"
    if generation_settings and generation_settings.get('clothing_styles'):
        clothing_list = generation_settings['clothing_styles']
        if isinstance(clothing_list, list) and clothing_list:
            clothing = clothing_list[0]
    prompt = prompt.replace('{clothing}', clothing)
    
    return prompt

def create_pose_prompt(
    template_name: str, 
    pose: str, 
    generation_settings: Dict[str, Any] = None,
    detected_gender: str = "neutral",
    detected_age: int = None,
    template_base_prompt: Optional[str] = None
) -> str:
    """Create prompt for generating isolated person without background"""
    
    settings = generation_settings or {}
    
    # ✅ PRIORITIZE TEMPLATE PROMPT (with modifications for isolation)
    if template_base_prompt:
        # ✅ ADD ISOLATION KEYWORDS to template prompt
        prompt = substitute_prompt_placeholders(
            template_base_prompt,
            pose=pose,
            detected_gender=detected_gender,
            generation_settings=settings
        )
        
        # ✅ FORCE ISOLATED PERSON (critical addition)
        # isolation_keywords = (
        #     "plain white background, isolated subject, no background elements, "
        #     "studio portrait style, single person only, transparent background, "
        #     "clean edges, cutout style"
        # )
        
        # prompt = f"{prompt}"
        logger.info(f"🎯 Using TEMPLATE prompt with isolation: {prompt[:200]}...")
        return prompt
    
    # ✅ FALLBACK: Auto-generate with ISOLATION
    logger.info("🔧 Using AUTO-GENERATED prompt with isolation")
    
    gender_terms = {
        "male": "handsome Indonesian man, masculine features",
        "female": "beautiful Indonesian woman, feminine features",
        "neutral": "person"
    }
    
    gender_desc = gender_terms.get(detected_gender, "person")
    
    # Age range
    age_desc = ""
    if detected_age:
        if detected_age < 25:
            age_desc = "young adult"
        elif detected_age < 40:
            age_desc = "adult"
        elif detected_age < 60:
            age_desc = "middle-aged"
        else:
            age_desc = "mature"
    
    # Pose descriptions (enhanced)
    pose_map = {
        "sitting": "sitting upright on a chair with good posture, formal seated position",
        "sitting_upright": "sitting upright on a chair, formal posture, straight back",
        "sitting_relaxed": "sitting relaxed in comfortable posture, casual seated pose",
        "sitting_crossed_legs": "sitting with crossed legs, elegant sophisticated pose",
        "standing": "standing naturally with confident posture, full body visible",
        "standing_casual": "standing casually in natural relaxed pose, arms at sides",
        "standing_confident": "standing confidently with professional posture, shoulders back",
        "lying_down": "lying down comfortably in relaxed horizontal position",
        "lying_relaxed": "lying down comfortably, relaxed horizontal pose",
        "leaning_casual": "leaning casually against surface, relaxed informal pose",
        "sitting_driving_position": "sitting in driving position, hands on steering wheel",
        "sitting_bar_stool": "sitting on a bar stool, casual elevated seated position",
        "walking": "walking naturally with confident stride, full body in motion",
        "running": "running dynamically with athletic form, action pose",
        "arms_crossed": "standing with arms crossed, confident assertive pose",
        "waving": "waving hand in friendly greeting gesture",
        "pointing": "pointing gesture with extended arm",
    }
    
    pose_desc = pose_map.get(pose, f"{pose} pose")
    
    # ✅ BUILD ISOLATED PERSON PROMPT
    prompt_parts = [
        # ✅ CRITICAL: Isolation first
        "single person only",
        "isolated on plain white background",
        "no background elements",
        "studio portrait",
        
        # Person description
        f"{age_desc} {gender_desc}" if age_desc else gender_desc,
        pose_desc,
        
        # Clothing
        f"wearing {settings.get('clothing_styles', ['casual'])[0]} clothing" if settings.get('clothing_styles') else "",
        
        # Quality
        "full body visible",
        "professional photography",
        "high quality",
        "sharp focus",
        "clean edges",
        "8k uhd",
        "photorealistic"
    ]
    
    full_prompt = ", ".join([p for p in prompt_parts if p])
    logger.info(f"Generated isolated person prompt: {full_prompt[:200]}...")
    return full_prompt


def create_negative_prompt(
    detected_gender: str = "neutral",
    template_negative_prompt: Optional[str] = None  # ✅ NEW PARAMETER
) -> str:
    """
    Create gender-aware negative prompt for Stable Diffusion
    
    Args:
        detected_gender: Detected gender to avoid opposite gender terms
        template_negative_prompt: Negative prompt from template (if exists)  # ✅ NEW
        
    Returns:
        Negative prompt string
    """
    
    # ✅ PRIORITIZE TEMPLATE NEGATIVE PROMPT
    if template_negative_prompt:
        logger.info("🎯 Using TEMPLATE negative prompt")
        return template_negative_prompt
    
    # ✅ FALLBACK: Auto-generate negative prompt (EXISTING LOGIC - UNCHANGED)
    logger.info("🔧 Using AUTO-GENERATED negative prompt")
    base_negative = [
        # ✅ CRITICAL: Anatomy issues (MORE SPECIFIC)
        "extra limbs", "duplicate arms", "duplicate legs", 
        "multiple hands", "multiple feet", "third arm", "third leg",
        "four arms", "four legs", "six fingers", "fused limbs",
        "conjoined limbs", "twisted limbs", "malformed hands", 
        "malformed feet", "extra fingers", "missing fingers",
        "bad anatomy", "bad proportions", "deformed", "disfigured",
        "twisted body", "elongated body", "unnatural pose",
        "floating limbs", "disconnected body parts",
        
        # Quality issues
        "low quality", "blurry", "out of focus", "distorted",
        "pixelated", "grainy", "noisy", "jpeg artifacts",
        
        # Multiple people
        "multiple people", "crowd", "two people", "duplicate person",
        "clone", "extra person", "group photo",
        
        # Face issues
        "disfigured face", "asymmetric face", "extra eyes", 
        "missing eyes", "malformed eyes",
        
        # Unwanted elements
        "watermark", "text", "signature", "logo", "border",
        "frame", "timestamp",
        
        # Style issues
        "cartoon", "anime", "drawing", "illustration", "painting",
        "3d render", "cgi", "unrealistic",
    ]
    
    # ✅ GENDER-SPECIFIC NEGATIVE PROMPTS
    if detected_gender == "male":
        base_negative.extend([
            "feminine features", "female", "woman", "girl",
            "feminine clothing", "dress", "skirt", "makeup",
            "long hair", "feminine accessories"
        ])
    elif detected_gender == "female":
        base_negative.extend([
            "masculine features", "male", "man", "boy",
            "masculine clothing", "beard", "mustache", "facial hair",
            "masculine accessories", "very short hair"
        ])
    
    return ", ".join(base_negative)


def calculate_processing_time(start_time: float, end_time: float) -> int:
    """Calculate processing time in milliseconds"""
    return int((end_time - start_time) * 1000)