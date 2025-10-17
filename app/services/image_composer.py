from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
from typing import Tuple, Optional
from loguru import logger

from app.models.schemas import InsertionArea


class ImageComposer:
    """
    Image composition service for placing generated person into design
    """
    
    def __init__(self):
        pass
    
    def composite_person_into_design(
        self,
        design_image: Image.Image,
        person_image: Image.Image,
        insertion_area: InsertionArea,
        blend: bool = True
    ) -> Optional[Image.Image]:
        try:
            logger.info(f"Compositing person into design at {insertion_area}")
            logger.info(f"Design size: {design_image.size}")
            logger.info(f"Person size (before resize): {person_image.size}")  # ✅ ADD THIS
            
            # Create a copy of design
            result = design_image.copy()
            
            # Resize person to fit insertion area
            target_size = (int(insertion_area.width), int(insertion_area.height))
            logger.info(f"Resizing person to: {target_size}")  # ✅ ADD THIS
            
            person_resized = person_image.resize(target_size, Image.Resampling.LANCZOS)
            
            # Apply adjustments if blend is enabled
            if blend:
                person_resized = self._adjust_for_scene(person_resized, design_image, insertion_area)
            
            # Create alpha mask for smooth edges
            mask = self._create_soft_mask(person_resized.size)
            
            # Paste person into design
            position = (int(insertion_area.x), int(insertion_area.y))
            result.paste(person_resized, position, mask)
            
            logger.info("Composition completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Composition failed: {e}")
            return None
    
    def _adjust_for_scene(
        self,
        person_image: Image.Image,
        design_image: Image.Image,
        insertion_area: InsertionArea
    ) -> Image.Image:
        """
        Adjust person image to match scene lighting and color
        
        Args:
            person_image: Person image to adjust
            design_image: Background design
            insertion_area: Where person will be placed
            
        Returns:
            Adjusted person image
        """
        try:
            # Sample background color around insertion area
            bg_sample = self._sample_background_color(design_image, insertion_area)
            
            # Adjust color temperature
            person_adjusted = self._match_color_temperature(person_image, bg_sample)
            
            # Match brightness
            person_adjusted = self._match_brightness(person_adjusted, design_image, insertion_area)
            
            return person_adjusted
            
        except Exception as e:
            logger.warning(f"Scene adjustment failed: {e}, using original")
            return person_image
    
    def _sample_background_color(
        self,
        design_image: Image.Image,
        insertion_area: InsertionArea
    ) -> Tuple[int, int, int]:
        """
        Sample average background color around insertion area
        
        Args:
            design_image: Design image
            insertion_area: Insertion area
            
        Returns:
            RGB tuple
        """
        try:
            # Sample area around the insertion (edges)
            x, y = int(insertion_area.x), int(insertion_area.y)
            w, h = int(insertion_area.width), int(insertion_area.height)
            
            # Get samples from borders
            border_width = 20
            samples = []
            
            # Top border
            if y - border_width > 0:
                crop = design_image.crop((x, y - border_width, x + w, y))
                samples.append(np.array(crop).mean(axis=(0, 1)))
            
            # Bottom border
            if y + h + border_width < design_image.height:
                crop = design_image.crop((x, y + h, x + w, y + h + border_width))
                samples.append(np.array(crop).mean(axis=(0, 1)))
            
            # Left border
            if x - border_width > 0:
                crop = design_image.crop((x - border_width, y, x, y + h))
                samples.append(np.array(crop).mean(axis=(0, 1)))
            
            # Right border
            if x + w + border_width < design_image.width:
                crop = design_image.crop((x + w, y, x + w + border_width, y + h))
                samples.append(np.array(crop).mean(axis=(0, 1)))
            
            if samples:
                avg_color = np.mean(samples, axis=0)
                return tuple(avg_color.astype(int))
            
            return (128, 128, 128)  # Default gray
            
        except Exception as e:
            logger.warning(f"Background color sampling failed: {e}")
            return (128, 128, 128)
    
    def _match_color_temperature(
        self,
        person_image: Image.Image,
        bg_color: Tuple[int, int, int]
    ) -> Image.Image:
        """
        Match color temperature of person to background
        
        Args:
            person_image: Person image
            bg_color: Background average color
            
        Returns:
            Adjusted image
        """
        try:
            # Calculate color temperature shift
            r, g, b = bg_color
            
            # Simple color temperature adjustment
            # If background is warm (more red), add warmth
            # If background is cool (more blue), add coolness
            
            if r > b + 20:  # Warm background
                enhancer = ImageEnhance.Color(person_image)
                person_image = enhancer.enhance(1.1)  # Slight saturation boost
            elif b > r + 20:  # Cool background
                enhancer = ImageEnhance.Color(person_image)
                person_image = enhancer.enhance(0.95)  # Slight saturation reduction
            
            return person_image
            
        except Exception as e:
            logger.warning(f"Color temperature matching failed: {e}")
            return person_image
    
    def _match_brightness(
        self,
        person_image: Image.Image,
        design_image: Image.Image,
        insertion_area: InsertionArea
    ) -> Image.Image:
        """
        Match brightness of person to design
        
        Args:
            person_image: Person image
            design_image: Design image
            insertion_area: Insertion area
            
        Returns:
            Adjusted image
        """
        try:
            # Sample brightness around insertion area
            bg_sample = self._sample_background_color(design_image, insertion_area)
            bg_brightness = sum(bg_sample) / 3
            
            # Calculate person brightness
            person_array = np.array(person_image)
            person_brightness = person_array.mean()
            
            # Adjust if difference is significant
            if abs(bg_brightness - person_brightness) > 20:
                factor = bg_brightness / person_brightness
                factor = max(0.7, min(1.3, factor))  # Clamp adjustment
                
                enhancer = ImageEnhance.Brightness(person_image)
                person_image = enhancer.enhance(factor)
            
            return person_image
            
        except Exception as e:
            logger.warning(f"Brightness matching failed: {e}")
            return person_image
    
    def _create_soft_mask(self, size: Tuple[int, int]) -> Image.Image:
        """
        Create soft-edged alpha mask for smooth blending
        
        Args:
            size: Mask size (width, height)
            
        Returns:
            Alpha mask image
        """
        try:
            # Create white mask
            mask = Image.new('L', size, 255)
            
            # Apply Gaussian blur to soften edges
            mask = mask.filter(ImageFilter.GaussianBlur(radius=3))
            
            # Create gradient fade on edges
            mask_array = np.array(mask)
            h, w = mask_array.shape
            
            # Fade edges (10% of dimension)
            fade_h = int(h * 0.05)
            fade_w = int(w * 0.05)
            
            # Top and bottom fade
            for i in range(fade_h):
                mask_array[i, :] = mask_array[i, :] * (i / fade_h)
                mask_array[-(i+1), :] = mask_array[-(i+1), :] * (i / fade_h)
            
            # Left and right fade
            for i in range(fade_w):
                mask_array[:, i] = mask_array[:, i] * (i / fade_w)
                mask_array[:, -(i+1)] = mask_array[:, -(i+1)] * (i / fade_w)
            
            return Image.fromarray(mask_array.astype('uint8'))
            
        except Exception as e:
            logger.warning(f"Soft mask creation failed: {e}, using full mask")
            return Image.new('L', size, 255)
    
    def add_shadow(
        self,
        image: Image.Image,
        insertion_area: InsertionArea,
        shadow_opacity: float = 0.3
    ) -> Image.Image:
        """
        Add subtle shadow under person (optional enhancement)
        
        Args:
            image: Composited image
            insertion_area: Person's insertion area
            shadow_opacity: Shadow opacity (0-1)
            
        Returns:
            Image with shadow
        """
        try:
            # Create shadow layer
            shadow = Image.new('RGBA', image.size, (0, 0, 0, 0))
            
            # Shadow position (slightly below person)
            shadow_y = int(insertion_area.y + insertion_area.height - 20)
            shadow_height = 30
            
            # Create shadow shape (ellipse)
            from PIL import ImageDraw
            draw = ImageDraw.Draw(shadow)
            
            shadow_bbox = [
                int(insertion_area.x + insertion_area.width * 0.2),
                shadow_y,
                int(insertion_area.x + insertion_area.width * 0.8),
                shadow_y + shadow_height
            ]
            
            draw.ellipse(shadow_bbox, fill=(0, 0, 0, int(255 * shadow_opacity)))
            
            # Blur shadow
            shadow = shadow.filter(ImageFilter.GaussianBlur(radius=10))
            
            # Composite shadow
            result = Image.alpha_composite(image.convert('RGBA'), shadow)
            
            return result.convert('RGB')
            
        except Exception as e:
            logger.warning(f"Shadow addition failed: {e}")
            return image


# Global instance
def get_image_composer() -> ImageComposer:
    """Get image composer instance"""
    return ImageComposer()