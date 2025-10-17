import insightface
from insightface.app import FaceAnalysis
import cv2
import numpy as np
from PIL import Image
from loguru import logger
from typing import Optional, List, Tuple, Dict
import os

from app.utils.config import settings


class FaceSwapper:
    """Face swapping service using InsightFace with gender detection"""
    
    def __init__(self):
        self.app = None
        self.swapper = None
        self.model_loaded = False
    
    def load_model(self):
        """Load InsightFace models"""
        if self.model_loaded:
            logger.info("Face swap model already loaded")
            return
        
        try:
            logger.info("Loading InsightFace models...")
            
            # Initialize face analysis
            self.app = FaceAnalysis(name='buffalo_l')
            ctx_id = -1 if settings.enable_cpu_offload else 0
            self.app.prepare(ctx_id=ctx_id, det_size=(640, 640))
            
            # Load face swapper model
            model_path = os.path.join(settings.models_dir, settings.face_swap_model)
            
            if not os.path.exists(model_path):
                logger.warning(f"Face swap model not found at {model_path}")
                logger.info("Downloading face swap model...")
                from insightface.model_zoo import get_model
                self.swapper = get_model('inswapper_128.onnx', download=True, download_zip=True)
            else:
                logger.info(f"Loading face swap model from {model_path}")
                from insightface.model_zoo import get_model
                self.swapper = get_model(model_path)
            
            self.model_loaded = True
            logger.info("Face swap models loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load face swap models: {e}")
            raise
    
    def detect_gender(self, image: Image.Image) -> str:
        """
        Detect gender from face image
        
        Args:
            image: PIL Image containing face
            
        Returns:
            Gender string: 'male', 'female', or 'neutral'
        """
        if not self.model_loaded:
            self.load_model()
        
        try:
            # Convert PIL to OpenCV
            image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Detect faces
            faces = self.app.get(image_cv)
            
            if not faces:
                logger.warning("No face detected for gender detection")
                return "neutral"
            
            # Get largest face
            face = sorted(
                faces,
                key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]),
                reverse=True
            )[0]
            
            # InsightFace gender attribute: 0 = female, 1 = male
            if hasattr(face, 'gender'):
                gender = "male" if face.gender == 1 else "female"
                logger.info(f"✅ Detected gender: {gender}")
                return gender
            else:
                logger.warning("Gender attribute not available")
                return "neutral"
            
        except Exception as e:
            logger.error(f"Gender detection failed: {e}")
            return "neutral"
    
    def detect_age(self, image: Image.Image) -> Optional[int]:
        """
        Detect age from face image
        
        Args:
            image: PIL Image containing face
            
        Returns:
            Estimated age or None
        """
        if not self.model_loaded:
            self.load_model()
        
        try:
            # Convert PIL to OpenCV
            image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Detect faces
            faces = self.app.get(image_cv)
            
            if not faces:
                logger.warning("No face detected for age detection")
                return None
            
            # Get largest face
            face = sorted(
                faces,
                key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]),
                reverse=True
            )[0]
            
            # Get age
            if hasattr(face, 'age'):
                age = int(face.age)
                logger.info(f"✅ Detected age: {age}")
                return age
            else:
                logger.warning("Age attribute not available")
                return None
            
        except Exception as e:
            logger.error(f"Age detection failed: {e}")
            return None
    
    def get_face_attributes(self, image: Image.Image) -> Dict[str, any]:
        """
        Get comprehensive face attributes (gender, age, bbox, etc.)
        
        Args:
            image: PIL Image containing face
            
        Returns:
            Dictionary with face attributes
        """
        if not self.model_loaded:
            self.load_model()
        
        try:
            # Convert PIL to OpenCV
            image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Detect faces
            faces = self.app.get(image_cv)
            
            if not faces:
                logger.warning("No face detected")
                return {
                    "detected": False,
                    "gender": "neutral",
                    "age": None,
                    "bbox": None
                }
            
            # Get largest face
            face = sorted(
                faces,
                key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]),
                reverse=True
            )[0]
            
            # Extract attributes
            gender = "male" if (hasattr(face, 'gender') and face.gender == 1) else "female"
            age = int(face.age) if hasattr(face, 'age') else None
            
            attributes = {
                "detected": True,
                "gender": gender,
                "age": age,
                "bbox": face.bbox.tolist() if hasattr(face, 'bbox') else None,
                "det_score": float(face.det_score) if hasattr(face, 'det_score') else None
            }
            
            logger.info(f"✅ Face attributes: gender={gender}, age={age}")
            return attributes
            
        except Exception as e:
            logger.error(f"Face attribute detection failed: {e}")
            return {
                "detected": False,
                "gender": "neutral",
                "age": None,
                "bbox": None,
                "error": str(e)
            }
    
    def swap_face(
        self,
        source_face_image: Image.Image,
        target_body_image: Image.Image,
        face_index: int = 0
    ) -> Optional[Image.Image]:
        """
        Swap face from source image to target image
        
        Args:
            source_face_image: Image containing the face to extract
            target_body_image: Image containing the body where face will be placed
            face_index: Index of face to swap (0 = largest face)
            
        Returns:
            Swapped image or None if failed
        """
        if not self.model_loaded:
            self.load_model()
        
        try:
            logger.info("Starting face swap process")
            
            # Convert PIL to OpenCV format
            source_cv = cv2.cvtColor(np.array(source_face_image), cv2.COLOR_RGB2BGR)
            target_cv = cv2.cvtColor(np.array(target_body_image), cv2.COLOR_RGB2BGR)
            
            # Detect faces in source image
            source_faces = self.app.get(source_cv)
            if not source_faces:
                logger.error("❌ No face detected in source image")
                return None
            
            # Get the largest face (assume it's the main face)
            source_faces_sorted = sorted(
                source_faces, 
                key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]),
                reverse=True
            )
            
            if face_index >= len(source_faces_sorted):
                logger.warning(f"Face index {face_index} out of range, using largest face")
                face_index = 0
            
            source_face = source_faces_sorted[face_index]
            logger.info(f"✅ Source face detected at bbox: {source_face.bbox}")
            
            # Detect faces in target image
            target_faces = self.app.get(target_cv)
            if not target_faces:
                logger.error("❌ No face detected in target AI-generated image")
                logger.warning("This might indicate wrong gender was generated!")
                return None
            
            # Get the largest face in target
            target_face = sorted(
                target_faces, 
                key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]),
                reverse=True
            )[0]
            logger.info(f"✅ Target face detected at bbox: {target_face.bbox}")
            
            # Perform face swap
            result = target_cv.copy()
            result = self.swapper.get(result, target_face, source_face, paste_back=True)
            
            # Convert back to PIL
            result_pil = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
            
            logger.info("✅ Face swap completed successfully")
            return result_pil
            
        except Exception as e:
            logger.error(f"❌ Face swap failed: {e}")
            return None
    
    def swap_multiple_faces(
        self,
        source_face_image: Image.Image,
        target_body_image: Image.Image
    ) -> Optional[Image.Image]:
        """
        Swap all faces from source image to all faces in target image
        
        Args:
            source_face_image: Image containing the face to extract
            target_body_image: Image containing multiple bodies where face will be placed
            
        Returns:
            Swapped image or None if failed
        """
        if not self.model_loaded:
            self.load_model()
        
        try:
            logger.info("Starting multiple face swap process")
            
            # Convert PIL to OpenCV format
            source_cv = cv2.cvtColor(np.array(source_face_image), cv2.COLOR_RGB2BGR)
            target_cv = cv2.cvtColor(np.array(target_body_image), cv2.COLOR_RGB2BGR)
            
            # Detect faces in source image
            source_faces = self.app.get(source_cv)
            if not source_faces:
                logger.error("No face detected in source image")
                return None
            
            # Get the largest face from source
            source_face = sorted(
                source_faces, 
                key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]),
                reverse=True
            )[0]
            logger.info(f"Source face detected")
            
            # Detect all faces in target image
            target_faces = self.app.get(target_cv)
            if not target_faces:
                logger.error("No face detected in target image")
                return None
            
            logger.info(f"Found {len(target_faces)} faces in target image")
            
            # Swap each face
            result = target_cv.copy()
            for i, target_face in enumerate(target_faces):
                result = self.swapper.get(result, target_face, source_face, paste_back=True)
                logger.info(f"Swapped face {i+1}/{len(target_faces)}")
            
            # Convert back to PIL
            result_pil = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
            
            logger.info("Multiple face swap completed successfully")
            return result_pil
            
        except Exception as e:
            logger.error(f"Multiple face swap failed: {e}")
            return None
    
    def detect_faces(self, image: Image.Image) -> int:
        """
        Detect number of faces in image
        
        Args:
            image: Input image
            
        Returns:
            Number of faces detected
        """
        if not self.model_loaded:
            self.load_model()
        
        try:
            image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            faces = self.app.get(image_cv)
            logger.info(f"Detected {len(faces)} face(s) in image")
            return len(faces)
        except Exception as e:
            logger.error(f"Face detection failed: {e}")
            return 0
    
    def get_face_info(self, image: Image.Image) -> List[dict]:
        """
        Get detailed information about detected faces
        
        Args:
            image: Input image
            
        Returns:
            List of face information dictionaries
        """
        if not self.model_loaded:
            self.load_model()
        
        try:
            image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            faces = self.app.get(image_cv)
            
            face_info = []
            for i, face in enumerate(faces):
                info = {
                    "index": i,
                    "bbox": face.bbox.tolist(),
                    "score": float(face.det_score),
                    "age": int(face.age) if hasattr(face, 'age') else None,
                    "gender": int(face.gender) if hasattr(face, 'gender') else None,
                    "embedding_shape": face.embedding.shape if hasattr(face, 'embedding') else None
                }
                face_info.append(info)
            
            logger.info(f"Retrieved info for {len(face_info)} face(s)")
            return face_info
            
        except Exception as e:
            logger.error(f"Get face info failed: {e}")
            return []
    
    def enhance_face(self, image: Image.Image, strength: float = 0.3) -> Image.Image:
        """
        Enhance face in image (optional post-processing)
        
        Args:
            image: Input image
            strength: Enhancement strength (0.0 to 1.0)
            
        Returns:
            Enhanced image or original if failed
        """
        try:
            # Convert to OpenCV
            image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Apply face enhancement techniques
            # 1. Slight sharpening
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            sharpened = cv2.filter2D(image_cv, -1, kernel)
            
            # 2. Blend original and sharpened based on strength
            strength = np.clip(strength, 0.0, 1.0)
            enhanced = cv2.addWeighted(image_cv, 1.0 - strength, sharpened, strength, 0)
            
            # 3. Optional: slight color enhancement
            lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            
            # Convert back to PIL
            result = Image.fromarray(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
            
            logger.info("Face enhanced successfully")
            return result
            
        except Exception as e:
            logger.warning(f"Face enhancement failed: {e}")
            return image
    
    def get_model_info(self) -> dict:
        """Get model information"""
        return {
            "model_name": settings.face_swap_model,
            "loaded": self.model_loaded,
            "face_analysis": "buffalo_l",
            "ctx_id": -1 if settings.enable_cpu_offload else 0,
            "detection_size": (640, 640)
        }
    
    def unload_model(self):
        """Unload models to free memory"""
        try:
            self.app = None
            self.swapper = None
            self.model_loaded = False
            logger.info("Face swap models unloaded")
        except Exception as e:
            logger.error(f"Failed to unload models: {e}")


# Global instance
_swapper_instance = None


def get_face_swapper() -> FaceSwapper:
    """Get singleton face swapper instance"""
    global _swapper_instance
    if _swapper_instance is None:
        _swapper_instance = FaceSwapper()
        # Pre-load model on startup
        try:
            _swapper_instance.load_model()
        except Exception as e:
            logger.warning(f"Failed to pre-load face swapper model: {e}")
    return _swapper_instance