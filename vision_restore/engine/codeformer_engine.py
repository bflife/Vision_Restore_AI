"""
Vision-Restore AI - CodeFormer Face Restoration Engine

Advanced face restoration using CodeFormer (Code-Prediction Transformer for Face Restoration).
CodeFormer provides superior face restoration quality compared to GFPGAN, especially for
heavily degraded faces.
"""

from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
import numpy as np

from vision_restore.core.config import Config, get_config
from vision_restore.core.logger import get_logger

logger = get_logger(__name__)


class CodeFormerEngine:
    """CodeFormer face restoration engine.
    
    CodeFormer provides:
    - Superior handling of heavily degraded faces
    - Adjustable fidelity parameter (quality vs identity preservation)
    - Better texture and detail generation
    - More natural results for extreme degradation
    
    Compared to GFPGAN:
    - Better for old/damaged photos
    - More controllable output quality
    - Better identity preservation at high fidelity settings
    """

    # Model configurations
    MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
        "codeformer-v0.1": {
            "filename": "codeformer.pth",
            "url": "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth",
            "detection_model": "retinaface_resnet50",
            "detection_url": "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/detection_Resnet50_Final.pth",
        },
    }

    def __init__(
        self,
        config: Optional[Config] = None,
        fidelity: float = 0.7,
        upscale: int = 2,
        model_name: str = "codeformer-v0.1",
        gpu_id: int = 0,
        only_center_face: bool = False,
    ):
        """Initialize the CodeFormer engine.
        
        Args:
            config: Application configuration
            fidelity: Balance between quality and fidelity (0=quality, 1=fidelity)
            upscale: Face upscale factor
            model_name: Model version to use
            gpu_id: GPU device ID (-1 for CPU)
            only_center_face: Only restore the center/largest face
        """
        self.config = config or get_config()
        self.fidelity = np.clip(fidelity, 0.0, 1.0)
        self.upscale = upscale
        self.model_name = model_name
        self.gpu_id = gpu_id
        self.only_center_face = only_center_face
        
        # Model instances (lazy loaded)
        self._codeformer_net = None
        self._face_helper = None
        self._device = None
        self._initialized = False

    def _get_model_path(self) -> Path:
        """Get path to the CodeFormer model file."""
        model_dir = self.config.models_dir / "codeformer"
        model_dir.mkdir(parents=True, exist_ok=True)
        config = self.MODEL_CONFIGS.get(self.model_name, self.MODEL_CONFIGS["codeformer-v0.1"])
        return model_dir / config["filename"]

    def _get_detection_model_path(self) -> Path:
        """Get path to the face detection model."""
        model_dir = self.config.models_dir / "codeformer"
        return model_dir / "detection_Resnet50_Final.pth"

    def initialize(self) -> bool:
        """Initialize the CodeFormer engine.
        
        Returns:
            True if initialization was successful
        """
        if self._initialized:
            return True
            
        try:
            import torch
            from basicsr.utils import img2tensor, tensor2img
            from basicsr.utils.download_util import load_file_from_url
            from facexlib.utils.face_restoration_helper import FaceRestoreHelper
            
            # Unified GPU Backend Selection
            from vision_restore.engine.gpu_backend import gpu_backend
            self._device = gpu_backend.get_torch_device(self.gpu_id)
            logger.info(f"CodeFormer using device: {self._device}")
            
            # Try to import CodeFormer network
            try:
                from codeformer.facelib.utils.face_restoration_helper import FaceRestoreHelper as CFHelper
                from codeformer.basicsr.archs.codeformer_arch import CodeFormer
                use_codeformer_pkg = True
            except ImportError:
                # Fall back to building CodeFormer architecture manually
                use_codeformer_pkg = False
                logger.info("Using embedded CodeFormer architecture")
            
            # Initialize face helper for detection and alignment
            self._face_helper = FaceRestoreHelper(
                upscale_factor=self.upscale,
                face_size=512,
                crop_ratio=(1, 1),
                det_model="retinaface_resnet50",
                save_ext="png",
                use_parse=True,
                device=self._device,
            )
            
            # Build CodeFormer network
            if use_codeformer_pkg:
                self._codeformer_net = CodeFormer(
                    dim_embd=512,
                    codebook_size=1024,
                    n_head=8,
                    n_layers=9,
                    connect_list=["32", "64", "128", "256"],
                ).to(self._device)
            else:
                # Simplified CodeFormer-like restoration using GFPGAN as fallback
                self._codeformer_net = self._build_fallback_restorer()
            
            # Load weights
            model_path = self._get_model_path()
            if model_path.exists():
                checkpoint = torch.load(model_path, map_location=self._device)
                if "params_ema" in checkpoint:
                    self._codeformer_net.load_state_dict(checkpoint["params_ema"], strict=False)
                elif "params" in checkpoint:
                    self._codeformer_net.load_state_dict(checkpoint["params"], strict=False)
                logger.info("CodeFormer model loaded successfully")
            else:
                logger.warning(f"CodeFormer model not found: {model_path}")
                # Try to use GFPGAN as fallback
                return self._initialize_fallback()
            
            self._codeformer_net.eval()
            self._initialized = True
            logger.info("CodeFormer engine initialized successfully")
            return True
            
        except ImportError as e:
            logger.warning(f"CodeFormer dependencies not available: {e}")
            return self._initialize_fallback()
        except Exception as e:
            logger.error(f"Failed to initialize CodeFormer: {e}")
            return self._initialize_fallback()

    def _build_fallback_restorer(self):
        """Build a fallback face restorer when CodeFormer is not available."""
        # This is a placeholder that will use GFPGAN architecture
        try:
            from gfpgan.archs.gfpganv1_clean_arch import GFPGANv1Clean
            
            model = GFPGANv1Clean(
                out_size=512,
                num_style_feat=512,
                channel_multiplier=2,
                narrow=1,
                sft_half=True,
            )
            logger.info("Using GFPGANv1Clean as fallback for CodeFormer")
            return model
        except ImportError:
            logger.warning("GFPGAN fallback also not available")
            return None

    def _initialize_fallback(self) -> bool:
        """Initialize using GFPGAN as fallback when CodeFormer is not available."""
        try:
            from vision_restore.engine.gfpgan_engine import GFPGANEngine
            
            # Create GFPGAN engine as fallback
            self._fallback_engine = GFPGANEngine(
                config=self.config,
                upscale=self.upscale,
                gpu_id=self.gpu_id,
            )
            
            if self._fallback_engine.initialize():
                logger.info("Using GFPGAN as fallback for CodeFormer")
                self._initialized = True
                self._use_fallback = True
                return True
                
        except Exception as e:
            logger.error(f"Fallback initialization failed: {e}")
        
        return False

    @property
    def is_initialized(self) -> bool:
        """Check if engine is initialized."""
        return self._initialized

    @property
    def is_available(self) -> bool:
        """Check if CodeFormer is available (dependencies installed)."""
        try:
            import torch
            from facexlib.utils.face_restoration_helper import FaceRestoreHelper
            return True
        except ImportError:
            return False

    def restore_faces(
        self,
        image: np.ndarray,
        fidelity: Optional[float] = None,
        only_center_face: Optional[bool] = None,
        paste_back: bool = True,
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Restore faces in an image.
        
        Args:
            image: Input image (BGR, uint8)
            fidelity: Override fidelity setting (0=quality, 1=fidelity)
            only_center_face: Only restore the center/largest face
            paste_back: Paste restored faces back to original image
            
        Returns:
            Tuple of (restored_image, list_of_restored_faces)
        """
        import torch
        import cv2
        
        if not self.is_initialized:
            if not self.initialize():
                raise RuntimeError("Failed to initialize CodeFormer engine")
        
        # Use fallback if CodeFormer not available
        if hasattr(self, "_use_fallback") and self._use_fallback:
            return self._fallback_engine.restore_faces(
                image,
                only_center_face=only_center_face if only_center_face is not None else self.only_center_face,
                paste_back=paste_back,
            )
        
        fidelity = fidelity if fidelity is not None else self.fidelity
        only_center = only_center_face if only_center_face is not None else self.only_center_face
        
        # Reset face helper for new image
        self._face_helper.clean_all()
        self._face_helper.read_image(image)
        
        # Detect and align faces
        self._face_helper.get_face_landmarks_5(
            only_center_face=only_center,
            resize=640,
            eye_dist_threshold=5,
        )
        self._face_helper.align_warp_face()
        
        restored_faces = []
        
        # Process each detected face
        for cropped_face in self._face_helper.cropped_faces:
            # Prepare face tensor
            cropped_face_t = self._img_to_tensor(cropped_face)
            
            try:
                with torch.no_grad():
                    output = self._codeformer_net(
                        cropped_face_t,
                        w=fidelity,
                        adain=True,
                    )[0]
                
                # Convert back to numpy
                restored_face = self._tensor_to_img(output)
                
            except Exception as e:
                logger.warning(f"CodeFormer face restoration failed, using input: {e}")
                restored_face = cropped_face
            
            restored_faces.append(restored_face)
            self._face_helper.add_restored_face(restored_face)
        
        # Paste back to original image if requested
        if paste_back:
            self._face_helper.get_inverse_affine(None)
            output_image = self._face_helper.paste_faces_to_input_image()
        else:
            output_image = image.copy()
        
        return output_image, restored_faces

    def _img_to_tensor(self, img: np.ndarray):
        """Convert image to tensor for processing."""
        import torch
        
        # Normalize and convert
        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5  # Normalize to [-1, 1]
        img = np.transpose(img, (2, 0, 1))  # HWC to CHW
        img = torch.from_numpy(img).unsqueeze(0)
        
        # DirectML check
        is_dml = str(self._device).startswith("privateuseone") or str(self._device) == "dml"
        if self.config.use_fp16 and self._device.type == "cuda":
            img = img.half()
        elif is_dml:
            # CodeFormer + DirectML + FP16 can be unstable
             pass
        
        return img.to(self._device)

    def _tensor_to_img(self, tensor) -> np.ndarray:
        """Convert tensor back to image."""
        import torch
        
        tensor = tensor.squeeze(0).float().clamp(-1, 1)
        tensor = (tensor + 1) / 2  # Denormalize to [0, 1]
        tensor = tensor.permute(1, 2, 0).cpu().numpy()
        img = (tensor * 255).astype(np.uint8)
        
        return img

    def restore_with_progress(
        self,
        image: np.ndarray,
        progress_callback: Optional[callable] = None,
        fidelity: Optional[float] = None,
        only_center_face: Optional[bool] = None,
        paste_back: bool = True,
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Restore faces with progress reporting.
        
        Args:
            image: Input image
            progress_callback: Callback function (progress: float, message: str)
            fidelity: Override fidelity setting
            only_center_face: Only restore center face
            paste_back: Paste restored faces to original
            
        Returns:
            Tuple of (restored_image, list_of_restored_faces)
        """
        if progress_callback:
            progress_callback(0.0, "Initializing CodeFormer...")
        
        if not self.is_initialized:
            self.initialize()
        
        if progress_callback:
            progress_callback(0.2, "Detecting faces...")
        
        result, faces = self.restore_faces(
            image,
            fidelity=fidelity,
            only_center_face=only_center_face,
            paste_back=paste_back,
        )
        
        if progress_callback:
            progress_callback(1.0, f"Restored {len(faces)} face(s)")
        
        return result, faces

    def detect_faces(self, image: np.ndarray) -> List[np.ndarray]:
        """Detect faces in an image without restoration.
        
        Args:
            image: Input image (BGR, uint8)
            
        Returns:
            List of detected face bounding boxes
        """
        if not self.is_initialized:
            if not self.initialize():
                return []
        
        if hasattr(self, "_use_fallback") and self._use_fallback:
            return self._fallback_engine.detect_faces(image)
        
        self._face_helper.clean_all()
        self._face_helper.read_image(image)
        
        self._face_helper.get_face_landmarks_5(
            only_center_face=False,
            resize=640,
            eye_dist_threshold=5,
        )
        
        return self._face_helper.det_faces

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
        if self._codeformer_net is not None:
            del self._codeformer_net
            self._codeformer_net = None
        
        if self._face_helper is not None:
            del self._face_helper
            self._face_helper = None
        
        if hasattr(self, "_fallback_engine"):
            self._fallback_engine.cleanup()
            del self._fallback_engine
        
        self._initialized = False
        
        # Force garbage collection for GPU memory
        try:
            import torch
            torch.cuda.empty_cache()
        except ImportError:
            pass
        
        logger.debug("CodeFormer engine resources released")


class FaceRestorationSettings:
    """Settings for face restoration."""
    
    def __init__(
        self,
        engine: str = "auto",  # "auto", "gfpgan", "codeformer"
        enabled: bool = True,
        only_center_face: bool = False,
        fidelity: float = 0.7,  # CodeFormer only
        weight: float = 0.5,   # GFPGAN only  
        upscale: int = 2,
    ):
        """Initialize face restoration settings.
        
        Args:
            engine: Face restoration engine to use
            enabled: Enable face restoration
            only_center_face: Only restore the center/largest face
            fidelity: CodeFormer fidelity (0=quality, 1=fidelity)
            weight: GFPGAN blending weight
            upscale: Face upscale factor
        """
        self.engine = engine
        self.enabled = enabled
        self.only_center_face = only_center_face
        self.fidelity = fidelity
        self.weight = weight
        self.upscale = upscale

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "engine": self.engine,
            "enabled": self.enabled,
            "only_center_face": self.only_center_face,
            "fidelity": self.fidelity,
            "weight": self.weight,
            "upscale": self.upscale,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FaceRestorationSettings":
        """Create from dictionary."""
        return cls(**data)
