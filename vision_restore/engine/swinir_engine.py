"""
Vision-Restore AI - SwinIR Engine

High-quality image super-resolution using SwinIR (Swin Transformer for Image Restoration).
SwinIR provides excellent results for both classical and real-world image degradation.
"""

from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import numpy as np

from vision_restore.core.config import Config, get_config
from vision_restore.core.logger import get_logger

logger = get_logger(__name__)


class SwinIREngine:
    """SwinIR image upscaling engine.
    
    SwinIR (Image Restoration Using Swin Transformer) provides:
    - Superior detail preservation compared to CNN-based methods
    - Excellent handling of various degradation types
    - Multiple model variants for different use cases
    
    Model variants:
    - 'classical': Clean image super-resolution (bicubic downscale)
    - 'real': Real-world degradation (noise, blur, compression)
    - 'lightweight': Faster processing with slightly lower quality
    """

    # Model configurations
    MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
        "swinir-classical-x2": {
            "scale": 2,
            "variant": "classical",
            "filename": "001_classicalSR_DIV2K_s48w8_SwinIR-M_x2.pth",
            "url": "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/001_classicalSR_DIV2K_s48w8_SwinIR-M_x2.pth",
        },
        "swinir-classical-x4": {
            "scale": 4,
            "variant": "classical",
            "filename": "001_classicalSR_DIV2K_s48w8_SwinIR-M_x4.pth",
            "url": "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/001_classicalSR_DIV2K_s48w8_SwinIR-M_x4.pth",
        },
        "swinir-real-x4": {
            "scale": 4,
            "variant": "real",
            "filename": "003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth",
            "url": "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth",
        },
        "swinir-lightweight-x4": {
            "scale": 4,
            "variant": "lightweight",
            "filename": "002_lightweightSR_DIV2K_s64w8_SwinIR-S_x4.pth",
            "url": "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x4.pth",
        },
    }

    def __init__(
        self,
        config: Optional[Config] = None,
        model_name: str = "swinir-real-x4",
        scale: int = 4,
        tile_size: int = 256,
        tile_overlap: int = 32,
        gpu_id: int = 0,
    ):
        """Initialize the SwinIR engine.
        
        Args:
            config: Application configuration
            model_name: Model variant to use
            scale: Upscale factor (2 or 4)
            tile_size: Tile size for memory-efficient processing
            tile_overlap: Overlap between tiles to prevent seams
            gpu_id: GPU device ID (-1 for CPU)
        """
        self.config = config or get_config()
        self.model_name = model_name
        self.scale = scale
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        self.gpu_id = gpu_id
        
        # Model instance (lazy loaded)
        self._model = None
        self._device = None
        self._initialized = False

    def _get_model_config(self) -> Dict[str, Any]:
        """Get configuration for the current model."""
        if self.model_name in self.MODEL_CONFIGS:
            return self.MODEL_CONFIGS[self.model_name]
        # Default to real-x4 for unknown models
        return self.MODEL_CONFIGS["swinir-real-x4"]

    def _get_model_path(self) -> Path:
        """Get path to the model file."""
        config = self._get_model_config()
        model_dir = self.config.models_dir / "swinir"
        model_dir.mkdir(parents=True, exist_ok=True)
        return model_dir / config["filename"]

    def _build_model(self):
        """Build the SwinIR model architecture."""
        try:
            import torch
            from basicsr.archs.swinir_arch import SwinIR
            
            config = self._get_model_config()
            variant = config["variant"]
            scale = config["scale"]
            
            # Configure model based on variant
            if variant == "classical":
                model = SwinIR(
                    upscale=scale,
                    in_chans=3,
                    img_size=48,
                    window_size=8,
                    img_range=1.0,
                    depths=[6, 6, 6, 6, 6, 6],
                    embed_dim=180,
                    num_heads=[6, 6, 6, 6, 6, 6],
                    mlp_ratio=2,
                    upsampler="pixelshuffle",
                    resi_connection="1conv",
                )
            elif variant == "lightweight":
                model = SwinIR(
                    upscale=scale,
                    in_chans=3,
                    img_size=64,
                    window_size=8,
                    img_range=1.0,
                    depths=[6, 6, 6, 6],
                    embed_dim=60,
                    num_heads=[6, 6, 6, 6],
                    mlp_ratio=2,
                    upsampler="pixelshuffledirect",
                    resi_connection="1conv",
                )
            else:  # real-world (default)
                model = SwinIR(
                    upscale=scale,
                    in_chans=3,
                    img_size=64,
                    window_size=8,
                    img_range=1.0,
                    depths=[6, 6, 6, 6, 6, 6, 6, 6, 6],
                    embed_dim=240,
                    num_heads=[8, 8, 8, 8, 8, 8, 8, 8, 8],
                    mlp_ratio=2,
                    upsampler="nearest+conv",
                    resi_connection="3conv",
                )
            
            return model
            
        except ImportError as e:
            logger.warning(f"SwinIR architecture not available: {e}")
            return None

    def initialize(self) -> bool:
        """Initialize the SwinIR engine.
        
        Returns:
            True if initialization was successful
        """
        if self._initialized:
            return True
            
        try:
            import torch
            
            from vision_restore.engine.gpu_backend import gpu_backend
            
            # Use unified backend for device selection (Supports CUDA and DirectML)
            self._device = gpu_backend.get_torch_device_for_model("swinir", self.gpu_id)
            logger.info(f"SwinIR using device: {self._device}")
            
            # Build model
            self._model = self._build_model()
            if self._model is None:
                return False
            
            # Load weights
            model_path = self._get_model_path()
            if not model_path.exists():
                logger.warning(f"SwinIR model not found: {model_path}")
                return False
            
            pretrained = torch.load(model_path, map_location=self._device)
            
            # Handle different checkpoint formats
            if "params_ema" in pretrained:
                self._model.load_state_dict(pretrained["params_ema"], strict=True)
            elif "params" in pretrained:
                self._model.load_state_dict(pretrained["params"], strict=True)
            else:
                self._model.load_state_dict(pretrained, strict=True)
            
            self._model.eval()
            self._model = self._model.to(self._device)
            
            # Use half precision for GPU if configured
            is_dml = str(self._device).startswith("privateuseone") or str(self._device) == "dml"
            
            if self.config.use_fp16 and self._device.type == "cuda":
                self._model = self._model.half()
            elif is_dml:
                 logger.info("DirectML detected (AMD/Intel), enforcing FP32 for stability.")
                 
                 # SAFETY OVERRIDE: DirectML on low-end hardware often crashes/shuts down on large tiles.
                 if self.tile_size > 192:
                     logger.warning(f"DirectML Safety: Reducing tile size from {self.tile_size} to 192 to prevent thermal shutdown.")
                     self.tile_size = 192
                     self.tile_pad = 16
            
            self._initialized = True
            logger.info(f"SwinIR engine initialized: {self.model_name}")
            return True
            
        except ImportError as e:
            logger.warning(f"PyTorch not available for SwinIR: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize SwinIR: {e}")
            return False

    @property
    def is_initialized(self) -> bool:
        """Check if engine is initialized."""
        return self._initialized

    @property
    def is_available(self) -> bool:
        """Check if SwinIR is available (dependencies installed)."""
        try:
            import torch
            from basicsr.archs.swinir_arch import SwinIR
            return True
        except ImportError:
            return False

    def _pad_image(self, image: np.ndarray, window_size: int = 8) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Pad image to be divisible by window size.
        
        Args:
            image: Input image (H, W, C)
            window_size: Window size for transformer
            
        Returns:
            Tuple of (padded_image, original_size)
        """
        h, w = image.shape[:2]
        pad_h = (window_size - h % window_size) % window_size
        pad_w = (window_size - w % window_size) % window_size
        
        if pad_h > 0 or pad_w > 0:
            image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
        
        return image, (h, w)

    def _tile_process(self, image: np.ndarray) -> np.ndarray:
        """Process large image using tiled approach.
        
        Args:
            image: Input image as numpy array (H, W, C), float32 [0, 1]
            
        Returns:
            Upscaled image
        """
        import torch
        import time
        
        h, w, c = image.shape
        scale = self._get_model_config()["scale"]
        tile = self.tile_size
        overlap = self.tile_overlap
        
        # Output image
        out_h, out_w = h * scale, w * scale
        output = np.zeros((out_h, out_w, c), dtype=np.float32)
        weight = np.zeros((out_h, out_w, c), dtype=np.float32)
        
        # Process tiles
        for y in range(0, h, tile - overlap):
            for x in range(0, w, tile - overlap):
                # Extract tile
                y_end = min(y + tile, h)
                x_end = min(x + tile, w)
                y_start = max(0, y_end - tile)
                x_start = max(0, x_end - tile)
                
                tile_img = image[y_start:y_end, x_start:x_end]
                
                # Pad to window size
                tile_img, orig_size = self._pad_image(tile_img)
                
                # Process tile
                with torch.no_grad():
                    tile_tensor = torch.from_numpy(tile_img).permute(2, 0, 1).unsqueeze(0)
                    tile_tensor = tile_tensor.to(self._device)
                    
                    if self.config.use_fp16 and self._device.type == "cuda":
                        tile_tensor = tile_tensor.half()
                    
                    out_tile = self._model(tile_tensor)
                    out_tile = out_tile.squeeze(0).permute(1, 2, 0).float().cpu().numpy()
                
                # Crop to original size (scaled)
                oh, ow = orig_size
                out_tile = out_tile[:oh * scale, :ow * scale]
                
                # Place in output with blending weight
                out_y = y_start * scale
                out_x = x_start * scale
                out_h_tile = out_tile.shape[0]
                out_w_tile = out_tile.shape[1]
                
                output[out_y:out_y + out_h_tile, out_x:out_x + out_w_tile] += out_tile
                weight[out_y:out_y + out_h_tile, out_x:out_x + out_w_tile] += 1
                
                # PULSE MODE: Add slight delay for DirectML thermal safety
                # This prevents continuous max-load which trips laptop power limits
                is_dml = str(self._device).startswith("privateuseone") or str(self._device) == "dml"
                if is_dml:
                    time.sleep(0.1)
        
        # Average overlapping regions
        output = np.divide(output, weight, where=weight > 0)
        return output

    def upscale(
        self,
        image: np.ndarray,
        outscale: Optional[float] = None,
    ) -> np.ndarray:
        """Upscale an image using SwinIR.
        
        Args:
            image: Input image as numpy array (BGR format, uint8)
            outscale: Output scale (uses model scale if not specified)
            
        Returns:
            Upscaled image as numpy array (BGR, uint8)
        """
        import torch
        import cv2
        
        if not self.is_initialized:
            if not self.initialize():
                raise RuntimeError("Failed to initialize SwinIR engine")
        
        config = self._get_model_config()
        model_scale = config["scale"]
        
        # Convert to float32 [0, 1], RGB
        img = image.astype(np.float32) / 255.0
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        h, w = img.shape[:2]
        
        # Use tiling for large images
        if h * w > self.tile_size * self.tile_size:
            output = self._tile_process(img)
        else:
            # Pad and process whole image
            img_padded, orig_size = self._pad_image(img)
            
            with torch.no_grad():
                img_tensor = torch.from_numpy(img_padded).permute(2, 0, 1).unsqueeze(0)
                img_tensor = img_tensor.to(self._device)
                
                if self.config.use_fp16 and self._device.type == "cuda":
                    img_tensor = img_tensor.half()
                
                output = self._model(img_tensor)
                output = output.squeeze(0).permute(1, 2, 0).float().cpu().numpy()
            
            # Crop to original size (scaled)
            output = output[:orig_size[0] * model_scale, :orig_size[1] * model_scale]
        
        # Handle scale mismatch
        if outscale and outscale != model_scale:
            target_h = int(h * outscale)
            target_w = int(w * outscale)
            output = cv2.resize(output, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
        
        # Convert back to BGR uint8
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        output = (np.clip(output, 0, 1) * 255).astype(np.uint8)
        
        return output

    def upscale_with_progress(
        self,
        image: np.ndarray,
        progress_callback: Optional[callable] = None,
        outscale: Optional[float] = None,
    ) -> np.ndarray:
        """Upscale an image with progress reporting.
        
        Args:
            image: Input image
            progress_callback: Callback function (progress: float, message: str)
            outscale: Output scale
            
        Returns:
            Upscaled image
        """
        if progress_callback:
            progress_callback(0.0, "Initializing SwinIR...")
        
        if not self.is_initialized:
            self.initialize()
        
        if progress_callback:
            progress_callback(0.1, f"Processing with SwinIR ({self.model_name})...")
        
        result = self.upscale(image, outscale)
        
        if progress_callback:
            progress_callback(1.0, "SwinIR upscaling complete")
        
        return result

    def get_output_size(
        self,
        input_size: Tuple[int, int],
        outscale: Optional[float] = None,
    ) -> Tuple[int, int]:
        """Calculate output size for given input.
        
        Args:
            input_size: Input (width, height)
            outscale: Output scale
            
        Returns:
            Output (width, height)
        """
        scale = outscale or self._get_model_config()["scale"]
        return (int(input_size[0] * scale), int(input_size[1] * scale))

    def cleanup(self) -> None:
        """Release resources."""
        if self._model is not None:
            del self._model
            self._model = None
            self._initialized = False
            
            # Force garbage collection for GPU memory
            try:
                import torch
                torch.cuda.empty_cache()
            except ImportError:
                pass
            
            logger.debug("SwinIR engine resources released")


def get_available_swinir_models() -> list:
    """Get list of available SwinIR model names."""
    return list(SwinIREngine.MODEL_CONFIGS.keys())
