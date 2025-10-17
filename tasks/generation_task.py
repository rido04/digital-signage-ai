from celery import Celery, Task
from loguru import logger
from rembg import remove
import time
import os
from typing import Dict, Any

from app.utils.config import settings
from app.utils.helpers import (
    download_image,
    save_image,
    send_callback_to_laravel,
    calculate_processing_time
)
from app.services.ai_generator import get_generator
from app.services.face_swapper import get_face_swapper
from app.services.image_composer import get_image_composer
from app.models.schemas import CampaignData, CompletionCallback, FailureCallback

# Initialize Celery
celery_app = Celery(
    'digital_signage_tasks',
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend
)

# Celery configuration
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=600,  # 10 minutes max
    task_soft_time_limit=540,  # 9 minutes soft limit
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=50,
)


class CallbackTask(Task):
    """Base task with callback support"""
    
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Handle task failure"""
        logger.error(f"Task {task_id} failed: {exc}")
        
        # Send failure callback to Laravel
        generation_id = kwargs.get('generation_id')
        if generation_id:
            callback_data = FailureCallback(
                error=str(exc),
                error_details={"traceback": str(einfo)}
            )
            
            # Send async callback
            import asyncio
            asyncio.run(send_callback_to_laravel(
                f"/generations/{generation_id}/failed",
                callback_data.dict()
            ))

def round_to_multiple_of_8(value: int) -> int:
    """Round value to nearest multiple of 8"""
    return max(8, (value // 8) * 8)

@celery_app.task(bind=True, base=CallbackTask, name='tasks.generate_personalized_image')
def generate_personalized_image(
    self,
    generation_id: str,
    face_image_path: str,
    campaign_data: Dict[str, Any]
) -> Dict[str, Any]:
    """Main task to generate personalized image with gender-aware AI"""
    start_time = time.time()
    
    try:
        logger.info(f"🚀 Starting generation for {generation_id}")
        
        # Parse campaign data
        campaign = CampaignData(**campaign_data)
        template = campaign.object_template
        
        # Update progress
        self.update_state(state='PROGRESS', meta={'progress': 5, 'status': 'Loading face image'})
        
        # ✅ STEP 0: Load face image and detect attributes
        logger.info("Step 0: Analyzing source face")
        from PIL import Image
        face_image = Image.open(face_image_path).convert('RGB')
        if face_image is None:
            raise Exception("Failed to load face image")
        
        # ✅ DETECT GENDER & AGE from source face
        face_swapper = get_face_swapper()
        face_attributes = face_swapper.get_face_attributes(face_image)
        
        detected_gender = face_attributes.get('gender', 'neutral')
        detected_age = face_attributes.get('age', None)
        
        logger.info(f"✅ Face Analysis Results:")
        logger.info(f"   - Gender: {detected_gender}")
        logger.info(f"   - Age: {detected_age}")
        logger.info(f"   - Detection Score: {face_attributes.get('det_score', 'N/A')}")
        
        if not face_attributes.get('detected'):
            raise Exception("❌ No face detected in source image!")
        
        # Update progress
        self.update_state(state='PROGRESS', meta={
            'progress': 10, 
            'status': 'Loading design image',
            'face_gender': detected_gender,
            'face_age': detected_age
        })
        
        # Step 1: Load design image
        logger.info("Step 1: Loading design image")
        design_image = download_image(campaign.design_image_url)
        if design_image is None:
            raise Exception("Failed to download design image")
        
        logger.info("✅ Images loaded successfully")
        self.update_state(state='PROGRESS', meta={
            'progress': 25, 
            'status': f'Generating {detected_gender} person with AI'
        })
        
        # Step 2: Generate person with AI (GENDER-AWARE)
        logger.info(f"Step 2: Generating {detected_gender} person with pose: {template.default_pose}")
        generator = get_generator()

        # Determine pose to use
        pose = template.default_pose
        generation_settings = campaign.generation_settings or {}

        # ✅ EXTRACT TEMPLATE CONFIGS
        template_base_prompt = template.base_prompt
        template_negative_prompt = template.negative_prompt  # Langsung akses, default None
        template_min_width = template.min_width  # Default 768
        template_min_height = template.min_height  # Default 1024
        template_guidance_scale = template.guidance_scale  # Default 7.5
        template_num_steps = template.num_inference_steps  # Default 30
        template_seed = template.seed  # Default None

        logger.info(f"📋 Template Config:")
        logger.info(f"   - Has base_prompt: {bool(template_base_prompt)}")
        logger.info(f"   - Has negative_prompt: {bool(template_negative_prompt)}")
        logger.info(f"   - Min size: {template_min_width}x{template_min_height}")
        logger.info(f"   - Guidance: {template_guidance_scale}, Steps: {template_num_steps}")

        # ✅ FIX: Enforce TEMPLATE minimum size (not hardcoded)
        MIN_WIDTH = template_min_width
        MIN_HEIGHT = template_min_height

        # Get original dimensions
        orig_width = int(campaign.insertion_area.width)
        orig_height = int(campaign.insertion_area.height)

        # Calculate aspect ratio
        aspect_ratio = orig_width / orig_height if orig_height > 0 else 1.0

        # Scale UP to meet minimum requirements while maintaining aspect ratio
        if orig_width < MIN_WIDTH or orig_height < MIN_HEIGHT:
            if aspect_ratio > 1:  # Landscape
                target_width = max(MIN_WIDTH, orig_width)
                target_height = int(target_width / aspect_ratio)
            else:  # Portrait
                target_height = max(MIN_HEIGHT, orig_height)
                target_width = int(target_height * aspect_ratio)
        else:
            target_width = orig_width
            target_height = orig_height

        # Round to multiple of 8 (SD requirement)
        target_width = round_to_multiple_of_8(target_width)
        target_height = round_to_multiple_of_8(target_height)
        

        logger.info(f"Original size: {orig_width}x{orig_height}")
        logger.info(f"Generation size (scaled up): {target_width}x{target_height}")

        # ✅ GENERATE with template configs
        person_image = generator.generate_person(
            template_name=template.name,
            pose=pose,
            width=target_width,
            height=target_height,
            generation_settings=generation_settings,
            detected_gender=detected_gender,
            detected_age=detected_age,
            template_base_prompt=template_base_prompt,
            template_negative_prompt=template_negative_prompt,
            guidance_scale=template_guidance_scale,
            num_inference_steps=template_num_steps,
            seed=template_seed
        )

        if person_image is None:
            raise Exception("Failed to generate person image")

        face_count = face_swapper.detect_faces(person_image)

        if face_count == 0:
            logger.error("❌ No face detected in generated image - possible anatomy failure")
            raise Exception("Generated image has no detectable face - rejecting")

        if face_count > 1:
            logger.error(f"❌ Multiple faces detected ({face_count}) - possible duplicate body parts")
            raise Exception(f"Generated image has {face_count} faces - anatomy error detected")

        logger.info(f"✅ Validation passed: 1 face detected")

        logger.info(f"✅ {detected_gender.capitalize()} person generated at {person_image.width}x{person_image.height}")
        logger.info("Step 2.5: Removing background from generated person")
        try:
            person_image_no_bg = remove(person_image)
            logger.info("✅ Background removed successfully")
            person_image = person_image_no_bg
        except Exception as e:
            logger.warning(f"⚠️ Background removal failed: {e}, using original image")
        self.update_state(state='PROGRESS', meta={'progress': 50, 'status': 'Swapping face'})
        
        # Step 3: Face swap
        logger.info("Step 3: Performing face swap")
        
        swapped_image = face_swapper.swap_face(
            source_face_image=face_image,
            target_body_image=person_image
        )
        
        if swapped_image is None:
            logger.warning("⚠️ Face swap failed - possible gender mismatch!")
            logger.warning("Using generated person image without face swap")
            swapped_image = person_image 
        else:
            # Optional: Enhance face after swap
            swapped_image = face_swapper.enhance_face(swapped_image)
            logger.info("✅ Face swap completed successfully")
        
        self.update_state(state='PROGRESS', meta={'progress': 75, 'status': 'Compositing image'})
        
        # Step 4: Composite into design
        logger.info("Step 4: Compositing into design")
        composer = get_image_composer()
        
        final_image = composer.composite_person_into_design(
            design_image=design_image,
            person_image=swapped_image,
            insertion_area=campaign.insertion_area,
            blend=True
        )
        
        if final_image is None:
            raise Exception("Failed to composite image")
        
        logger.info("✅ Composition completed successfully")
        self.update_state(state='PROGRESS', meta={'progress': 90, 'status': 'Saving result'})
        
        # Step 5: Save result
        logger.info("Step 5: Saving result")
        output_filename = f"{generation_id}.jpg"
        output_path = os.path.join(settings.output_dir, output_filename)
        
        if not save_image(final_image, output_path):
            raise Exception("Failed to save result image")
        
        # Calculate processing time
        end_time = time.time()
        processing_time_ms = calculate_processing_time(start_time, end_time)
        
        logger.info(f"✅ Generation completed in {processing_time_ms}ms")
        
        # Step 6: Send callback to Laravel
        logger.info("Step 6: Sending completion callback")
        
        # In production, you would upload to S3 and get URL
        # For now, we'll use a placeholder
        result_image_url = f"{settings.laravel_api_url}/storage/results/{output_filename}"
        
        callback_data = CompletionCallback(
            result_image_url=result_image_url,
            processing_time_ms=processing_time_ms,
            pose_used=pose,
            ai_parameters={
                "model": settings.sd_model_id,
                "steps": template_num_steps,  # ✅ Use template value
                "guidance_scale": template_guidance_scale,  # ✅ Use template value
                "detected_gender": detected_gender,
                "detected_age": detected_age,
                "used_template_prompt": bool(template_base_prompt),  # ✅ NEW INFO
                "template_min_size": f"{template_min_width}x{template_min_height}"  # ✅ NEW INFO
            }
        )
        
        # Send callback
        import asyncio
        callback_success = asyncio.run(send_callback_to_laravel(
            f"/generations/{generation_id}/complete",
            callback_data.dict()
        ))
        
        if not callback_success:
            logger.warning("Failed to send completion callback")
        
        return {
            "status": "completed",
            "generation_id": generation_id,
            "result_path": output_path,
            "processing_time_ms": processing_time_ms,
            "detected_gender": detected_gender,
            "detected_age": detected_age
        }
        
    except Exception as e:
        logger.error(f"❌ Generation failed for {generation_id}: {e}")
        
        # Send failure callback
        callback_data = FailureCallback(
            error=str(e),
            error_details={"step": "generation", "traceback": str(e)}
        )
        
        import asyncio
        asyncio.run(send_callback_to_laravel(
            f"/generations/{generation_id}/failed",
            callback_data.dict()
        ))
        
        raise


@celery_app.task(name='tasks.cleanup_old_files')
def cleanup_old_files(days_old: int = 7):
    """
    Cleanup old temporary and output files
    
    Args:
        days_old: Delete files older than this many days
    """
    import glob
    from datetime import datetime, timedelta
    
    cutoff_time = time.time() - (days_old * 86400)
    deleted_count = 0
    
    for directory in [settings.temp_dir, settings.output_dir]:
        for filepath in glob.glob(f"{directory}/*"):
            try:
                if os.path.getmtime(filepath) < cutoff_time:
                    os.remove(filepath)
                    deleted_count += 1
            except Exception as e:
                logger.warning(f"Failed to delete {filepath}: {e}")
    
    logger.info(f"Cleaned up {deleted_count} old files")
    return {"deleted_count": deleted_count}