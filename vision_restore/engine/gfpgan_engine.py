"""
Vision-Restore AI - GFPGAN Face Restoration Engine

Face detection and restoration using GFPGAN for high-quality face enhancement.
"""

from pathlib import Path
from typing import Optional, Tuple, List
import numpy as np

from vision_restore.core.config import Config, get_config
from vision_restore.core.logger import get_logger

logger = get_logger(__name__)


class GFPGANEngine:
    """GFPGAN face restoration engine.
    
    Provides high-quality face restoration with:
    - Automatic face detection
    - Individual face enhancement
    - Seamless blending with background
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        upscale: int = 4,
        model_name: str = "GFPGANv1.4",
        gpu_id: int = 0,
    ):
        """Initialize the GFPGAN engine.
        
        Args:
            config: Application configuration
            upscale: Face upscale factor
            model_name: GFPGAN model to use
            gpu_id: GPU device ID (-1 for CPU)
        """
        self.config = config or get_config()
        self.upscale = upscale
        self.model_name = model_name
        self.gpu_id = gpu_id
        
        # Engine instance (lazy loaded)
        self._restorer = None
        self._face_helper = None

    def _initialize(self) -> bool:
        """Initialize the GFPGAN engine."""
        try:
            import torch
            from gfpgan import GFPGANer
            
            # Get model path
            model_path = self.config.models_dir / "gfpgan" / f"{self.model_name}.pth"
            
            if not model_path.exists():
                logger.warning(f"GFPGAN model not found: {model_path}")
                return False
            
            # Unified GPU Backend Selection
            from vision_restore.engine.gpu_backend import gpu_backend
            device = gpu_backend.get_torch_device(self.gpu_id)
            
            # Initialize restorer
            self._restorer = GFPGANer(
                model_path=str(model_path),
                upscale=self.upscale,
                arch="clean",
                channel_multiplier=2,
                bg_upsampler=None,  # We handle background separately
                device=device,
            )
            
            logger.info(f"GFPGAN initialized successfully (device: {device})")
            return True
            
        except ImportError as e:
            logger.warning(f"GFPGAN not available: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize GFPGAN: {e}")
            return False

    @property
    def is_initialized(self) -> bool:
        """Check if engine is initialized."""
        return self._restorer is not None

    def initialize(self) -> bool:
        """Explicitly initialize the engine.
        
        Returns:
            True if initialization was successful
        """
        if self.is_initialized:
            return True
        return self._initialize()

    def detect_faces(self, image: np.ndarray) -> List[dict]:
        """Detect faces in an image.
        
        Args:
            image: Input image (BGR, uint8)
            
        Returns:
            List of face detection results
        """
        if not self.is_initialized:
            if not self.initialize():
                return []
        
        try:
            # Use face helper from GFPGAN
            if hasattr(self._restorer, 'face_helper'):
                self._restorer.face_helper.clean_all()
                self._restorer.face_helper.read_image(image)
                self._restorer.face_helper.get_face_landmarks_5(
                    only_center_face=False,
                    resize=640,
                    eye_dist_threshold=5,
                )
                
                faces = []
                for i, (face, landmark) in enumerate(zip(
                    self._restorer.face_helper.cropped_faces,
                    self._restorer.face_helper.all_landmarks_5
                )):
                    faces.append({
                        "index": i,
                        "bbox": None,  # Will be available after de-transform
                        "landmark": landmark,
                    })
                
                return faces
            
            return []
            
        except Exception as e:
            logger.warning(f"Face detection failed: {e}")
            return []

    def restore_faces(
        self,
        image: np.ndarray,
        only_center_face: bool = False,
        paste_back: bool = True,
        weight: float = 0.5,
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Restore faces in an image.
        
        Args:
            image: Input image (BGR, uint8)
            only_center_face: Only restore the center/largest face
            paste_back: Paste restored faces back to original image
            weight: Blending weight for face restoration (0-1)
            
        Returns:
            Tuple of (restored_image, list_of_restored_faces)
        """
        if not self.is_initialized:
            if not self.initialize():
                logger.error("GFPGAN not initialized")
                return image, []
        
        try:
            # Enhance with GFPGAN
            _, restored_faces, restored_img = self._restorer.enhance(
                image,
                has_aligned=False,
                only_center_face=only_center_face,
                paste_back=paste_back,
                weight=weight,
            )
            
            if restored_img is None:
                logger.warning("No faces detected or restoration failed")
                return image, []
            
            logger.debug(f"Restored {len(restored_faces)} face(s)")
            return restored_img, restored_faces
            
        except Exception as e:
            logger.error(f"Face restoration failed: {e}")
            return image, []

    def restore_with_progress(
        self,
        image: np.ndarray,
        progress_callback: Optional[callable] = None,
        only_center_face: bool = False,
        paste_back: bool = True,
    ) -> np.ndarray:
        """Restore faces with progress reporting.
        
        Args:
            image: Input image
            progress_callback: Callback function (progress: float, message: str)
            only_center_face: Only restore center face
            paste_back: Paste restored faces back
            
        Returns:
            Image with restored faces
        """
        if progress_callback:
            progress_callback(0.0, "Initializing face restoration...")
        
        if not self.is_initialized:
            self.initialize()
        
        if progress_callback:
            progress_callback(0.2, "Detecting faces...")
        
        faces = self.detect_faces(image)
        num_faces = len(faces)
        
        if num_faces == 0:
            if progress_callback:
                progress_callback(1.0, "No faces detected")
            return image
        
        if progress_callback:
            progress_callback(0.4, f"Restoring {num_faces} face(s)...")
        
        restored_img, _ = self.restore_faces(
            image,
            only_center_face=only_center_face,
            paste_back=paste_back,
        )
        
        if progress_callback:
            progress_callback(1.0, f"Restored {num_faces} face(s)")
        
        return restored_img

    def get_face_count(self, image: np.ndarray) -> int:
        """Get number of faces in an image.
        
        Args:
            image: Input image
            
        Returns:
            Number of detected faces
        """
        faces = self.detect_faces(image)
        return len(faces)

    def cleanup(self) -> None:
        """Release resources."""
        if self._restorer is not None:
            del self._restorer
            self._restorer = None
            
            # Clear CUDA cache
            try:
                import torch
                torch.cuda.empty_cache()
            except ImportError:
                pass
            
            logger.debug("GFPGAN engine resources released")


class FaceEnhancementSettings:
    """Settings for face enhancement."""
    
    def __init__(
        self,
        enabled: bool = True,
        only_center_face: bool = False,
        weight: float = 0.5,
        upscale: int = 4,
    ):
        """Initialize face enhancement settings.
        
        Args:
            enabled: Enable face enhancement
            only_center_face: Only enhance the center/largest face
            weight: Blending weight (0=original, 1=fully restored)
            upscale: Face upscale factor
        """
        self.enabled = enabled
        self.only_center_face = only_center_face
        self.weight = weight
        self.upscale = upscale
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "enabled": self.enabled,
            "only_center_face": self.only_center_face,
            "weight": self.weight,
            "upscale": self.upscale,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "FaceEnhancementSettings":
        """Create from dictionary."""
        return cls(**data)
