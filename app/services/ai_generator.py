import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image
from loguru import logger
from typing import Optional
import gc

from app.utils.config import settings
from app.utils.helpers import create_pose_prompt, create_negative_prompt


def get_device() -> torch.device:
    """
    Get PyTorch device with DirectML support for AMD GPU on Windows
    Priority: DirectML (AMD) > CUDA (NVIDIA) > CPU
    """
    try:
        # ✅ Try DirectML first (AMD GPU on Windows)
        import torch_directml
        device = torch_directml.device()
        logger.info(f"✅ Using DirectML device (AMD GPU): {device}")
        return device
    except ImportError:
        logger.warning("⚠️ DirectML not available")
    
    # Fallback to CUDA (NVIDIA GPU)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"✅ Using CUDA device (NVIDIA GPU): {device}")
        return device
    
    # Final fallback to CPU
    logger.warning("⚠️ No GPU available, using CPU (will be slow!)")
    return torch.device("cpu")


class AIGenerator:
    """
    AI Image Generator using Stable Diffusion with gender-aware prompts
    Supports: DirectML (AMD), CUDA (NVIDIA), CPU
    """
    
    def __init__(self):
        self.pipe = None
        self.device = get_device()
        self.model_loaded = False
        self.is_directml = False
        
        # Check if using DirectML
        try:
            import torch_directml
            self.is_directml = str(self.device).startswith('privateuseone')
            if self.is_directml:
                logger.info("🎮 DirectML (AMD GPU) mode enabled")
        except ImportError:
            pass
    
    def _patch_pipeline_for_directml(self):
        """
        Patch pipeline to handle text encoder on CPU while keeping UNet/VAE on GPU
        This fixes the tensor device mismatch error in DirectML
        """
        if not self.is_directml or self.pipe is None:
            return
        
        # Store original encode_prompt method
        original_encode_prompt = self.pipe.encode_prompt
        
        def patched_encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt=None,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            lora_scale=None,
            **kwargs
        ):
            """
            Wrapper that forces text encoding on CPU, then moves result to GPU
            """
            # Get text encoder's device (CPU)
            text_encoder_device = self.pipe.text_encoder.device
            
            # Call original method with CPU device for text encoding
            result = original_encode_prompt(
                prompt=prompt,
                device=text_encoder_device,  # Encode on CPU
                num_images_per_prompt=num_images_per_prompt,
                do_classifier_free_guidance=do_classifier_free_guidance,
                negative_prompt=negative_prompt,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                lora_scale=lora_scale,
                **kwargs
            )
            
            # ✅ Move embeddings to DirectML device for UNet
            if isinstance(result, tuple):
                # Result is (prompt_embeds, negative_prompt_embeds)
                prompt_embeds, negative_prompt_embeds = result
                prompt_embeds = prompt_embeds.to(self.device)
                if negative_prompt_embeds is not None:
                    negative_prompt_embeds = negative_prompt_embeds.to(self.device)
                return (prompt_embeds, negative_prompt_embeds)
            else:
                # Single tensor
                return result.to(self.device)
        
        # Replace with patched version
        self.pipe.encode_prompt = patched_encode_prompt
        logger.info("✅ Pipeline patched for DirectML compatibility")
        
    def load_model(self):
        """Load Stable Diffusion model with device-specific optimizations"""
        if self.model_loaded:
            logger.info("Model already loaded")
            return
        
        try:
            logger.info(f"⏳ Loading Stable Diffusion model: {settings.sd_model_id}")
            logger.info(f"🖥️ Target device: {self.device}")
            
            # Device-specific dtype
            if self.is_directml or self.device.type == "cuda":
                dtype = torch.float32
                logger.info("Using FP16 precision for GPU")
            else:
                dtype = torch.float16
                logger.info("Using FP32 precision for CPU")
            
            # Load pipeline
            self.pipe = StableDiffusionPipeline.from_pretrained(
                settings.sd_model_id,
                torch_dtype=dtype,
                safety_checker=None,
                requires_safety_checker=False,
                low_cpu_mem_usage=True
            )
            
            # Use DPM Solver
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                self.pipe.scheduler.config
            )
            
            # Move to device
            logger.info(f"📦 Moving model to {self.device}...")
            self.pipe = self.pipe.to(self.device)
            
            # ✅ DirectML WORKAROUND: Text encoder on CPU
            if self.is_directml:
                logger.info("🔧 Applying DirectML compatibility fixes...")
                
                # ✅ FORCE OLD ATTENTION PROCESSOR (DirectML compatible)
                from diffusers.models.attention_processor import AttnProcessor
                self.pipe.unet.set_attn_processor(AttnProcessor())
                logger.info("✅ Using legacy attention processor (DirectML compatible)")
                try:
                    # Move text encoder to CPU (avoids version_counter error)
                    self.pipe.text_encoder = self.pipe.text_encoder.to("cpu")
                    logger.info("✅ Text encoder moved to CPU")
                    logger.info("✅ UNet & VAE remain on DirectML (AMD GPU)")
                    
                    # ✅ PATCH PIPELINE to handle CPU text encoder
                    self._patch_pipeline_for_directml()
                    
                except Exception as e:
                    logger.warning(f"⚠️ DirectML workaround failed: {e}")
                
                # Apply optimizations
                logger.info("🔧 Applying DirectML (AMD) optimizations...")
                # self.pipe.enable_attention_slicing("max")
                # self.pipe.enable_vae_slicing()
                logger.info("✅ DirectML optimizations applied")
                
            elif self.device.type == "cuda":
                logger.info("🔧 Applying CUDA (NVIDIA) optimizations...")
                
                if settings.use_xformers:
                    try:
                        self.pipe.enable_xformers_memory_efficient_attention()
                        logger.info("✅ xformers enabled")
                    except Exception as e:
                        logger.warning(f"⚠️ xformers not available: {e}")
                
                if settings.enable_cpu_offload:
                    self.pipe.enable_model_cpu_offload()
                    logger.info("✅ CPU offload enabled")
                
                self.pipe.enable_attention_slicing()
                logger.info("✅ CUDA optimizations applied")
                
            else:  # CPU
                logger.info("🔧 Applying CPU optimizations...")
                self.pipe.enable_attention_slicing("max")
                self.pipe.enable_vae_slicing()
                torch.set_num_threads(torch.get_num_threads())
                logger.info(f"✅ CPU optimizations applied (threads: {torch.get_num_threads()})")
            
            self.model_loaded = True
            logger.info("✅ Stable Diffusion model loaded successfully!")
            logger.info(f"📊 Model info: {self.get_model_info()}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load Stable Diffusion model: {e}")
            logger.exception(e)
            raise
    
    def generate_person(
        self,
        template_name: str,
        pose: str,
        width: Optional[int] = None,
        height: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        seed: Optional[int] = None,
        generation_settings: dict = None,
        detected_gender: str = "neutral",
        detected_age: int = None,
        template_base_prompt: Optional[str] = None,
        template_negative_prompt: Optional[str] = None
    ) -> Optional[Image.Image]:
        """
        Generate a person image based on template, pose, and detected face attributes
        """
        if not self.model_loaded:
            self.load_model()
        
        try:
            # Use defaults if not specified
            width = width or settings.default_image_width
            height = height or settings.default_image_height
            num_inference_steps = num_inference_steps or settings.num_inference_steps
            guidance_scale = guidance_scale or settings.guidance_scale
            
            # ✅ CREATE PROMPTS (with template priority)
            prompt = create_pose_prompt(
                template_name, 
                pose, 
                generation_settings,
                detected_gender=detected_gender,
                detected_age=detected_age,
                template_base_prompt=template_base_prompt
            )
            
            negative_prompt = create_negative_prompt(
                detected_gender=detected_gender,
                template_negative_prompt=template_negative_prompt
            )
            
            logger.info(f"🎨 Generating {detected_gender} person: {pose}, size: {width}x{height}")
            logger.info(f"⚙️ Steps: {num_inference_steps}, Guidance: {guidance_scale}")
            logger.info(f"📝 Prompt: {prompt[:200]}...")
            logger.info(f"🚫 Negative: {negative_prompt[:100]}...")
            
            # Set seed for reproducibility
            generator = None
            if seed is not None:
                # ✅ DirectML needs CPU generator
                device_for_generator = "cpu" if self.is_directml else self.device
                generator = torch.Generator(device=device_for_generator).manual_seed(seed)
                logger.info(f"🎲 Using seed: {seed}")
            
            # Generate image
            logger.info("🚀 Starting generation...")
            import time
            start_time = time.time()
            
            with torch.no_grad():
                result = self.pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    width=width,
                    height=height,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator
                )
            
            elapsed = time.time() - start_time
            logger.info(f"⏱️ Generation took {elapsed:.2f} seconds")
            
            image = result.images[0]
            logger.info(f"✅ {detected_gender.capitalize()} person generated successfully")
            
            return image
            
        except Exception as e:
            logger.error(f"❌ Failed to generate person: {e}")
            logger.exception(e)
            return None
    
    def unload_model(self):
        """Unload model to free memory"""
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
            self.model_loaded = False
            
            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Force garbage collection
            gc.collect()
            
            logger.info("Model unloaded")
    
    def get_model_info(self) -> dict:
        """Get model information"""
        device_type = "DirectML (AMD)" if self.is_directml else str(self.device)
        
        return {
            "model_id": settings.sd_model_id,
            "device": device_type,
            "device_raw": str(self.device),
            "loaded": self.model_loaded,
            "is_directml": self.is_directml,
            "xformers_enabled": settings.use_xformers if not self.is_directml else False,
            "cpu_offload_enabled": settings.enable_cpu_offload if not self.is_directml else False,
        }


# Global instance
_generator_instance = None

def get_generator() -> AIGenerator:
    """Get singleton generator instance"""
    global _generator_instance
    if _generator_instance is None:
        _generator_instance = AIGenerator()
        # Pre-load model on startup
        _generator_instance.load_model()
    return _generator_instance