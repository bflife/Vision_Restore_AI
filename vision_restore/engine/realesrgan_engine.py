"""
Vision-Restore AI - Real-ESRGAN Engine

High-performance image upscaling using Real-ESRGAN with automatic backend selection.
Supports both PyTorch and ncnn backends for maximum compatibility.
"""

import os
from pathlib import Path
from typing import Optional, Tuple
import numpy as np

from vision_restore.core.config import Config, get_config
from vision_restore.core.logger import get_logger

logger = get_logger(__name__)


class RealESRGANEngine:
    """Real-ESRGAN image upscaling engine.
    
    Supports multiple backends:
    - PyTorch (GPU/CPU): Full feature support, requires torch
    - ncnn (Vulkan/CPU): Lighter weight, uses ncnn bindings
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        model_name: Optional[str] = None,
        scale: int = 4,
        tile_size: int = 512,
        tile_pad: int = 32,
        gpu_id: int = 0,
    ):
        """Initialize the Real-ESRGAN engine.
        
        Args:
            config: Application configuration
            model_name: Model to use (e.g., 'realesrgan-x4plus')
            scale: Upscale factor (2, 4, or 8)
            tile_size: Tile size for processing
            tile_pad: Tile padding for overlap
            gpu_id: GPU device ID (-1 for CPU)
        """
        self.config = config or get_config()
        self.scale = scale
        self.tile_size = tile_size
        self.tile_pad = tile_pad
        self.gpu_id = gpu_id
        
        # Select model based on scale
        if model_name:
            self.model_name = model_name
        elif scale == 2:
            self.model_name = "realesrgan-x2plus"
        else:
            self.model_name = "realesrgan-x4plus"
        
        # Engine instance (lazy loaded)
        self._upsampler = None
        self._backend = None

    def _initialize_pytorch_backend(self) -> bool:
        """Initialize PyTorch-based Real-ESRGAN backend."""
        try:
            import torch
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
            
            # Determine model path
            model_dir = self.config.models_dir / "realesrgan"
            
            if self.model_name == "realesrgan-x2plus":
                model_path = model_dir / "RealESRGAN_x2plus.pth"
                model = RRDBNet(
                    num_in_ch=3,
                    num_out_ch=3,
                    num_feat=64,
                    num_block=23,
                    num_grow_ch=32,
                    scale=2,
                )
                netscale = 2
            elif self.model_name == "realesrgan-x4plus-anime":
                model_path = model_dir / "RealESRGAN_x4plus_anime_6B.pth"
                model = RRDBNet(
                    num_in_ch=3,
                    num_out_ch=3,
                    num_feat=64,
                    num_block=6,
                    num_grow_ch=32,
                    scale=4,
                )
                netscale = 4
            else:  # Default to x4plus
                model_path = model_dir / "RealESRGAN_x4plus.pth"
                model = RRDBNet(
                    num_in_ch=3,
                    num_out_ch=3,
                    num_feat=64,
                    num_block=23,
                    num_grow_ch=32,
                    scale=4,
                )
                netscale = 4
            
            # Check if model exists
            if not model_path.exists():
                logger.warning(f"Model not found: {model_path}")
                return False
            
            # Unified GPU Backend Selection (CUDA / DirectML / CPU)
            from vision_restore.engine.gpu_backend import gpu_backend
            
            # Get the optimal device (CUDA, DirectML, or CPU)
            device = gpu_backend.get_torch_device(self.gpu_id)
            logger.info(f"Real-ESRGAN using device: {device}")

            # Check for FP16 compatibility
            # DirectML often has issues with FP16, so enforce FP32 if using DirectML
            is_dml = str(device).startswith("privateuseone") or str(device) == "dml"
            is_cuda = device.type == "cuda"
            
            use_half = self.config.use_fp16 and is_cuda
            
            # SAFETY OVERRIDE: DirectML on low-end hardware often crashes/shuts down on large tiles.
            # We strictly clamp the tile size to prevent thermal trips.
            if is_dml and self.tile_size > 192:
                 logger.warning(f"DirectML Safety: Reducing tile size from {self.tile_size} to 192 to prevent system instability.")
                 self.tile_size = 192
                 self.tile_pad = 16 # Reduce padding too
            
            if is_dml and self.config.use_fp16:
                 logger.info("DirectML detected, enforcing FP32 for stability in Real-ESRGAN.")
                 use_half = False

            # Initialize upsampler with specific device
            # Note: We modify how we call RealESRGANer to pass 'device' instead of 'gpu_id' if possible
            # or we rely on the fact that RealESRGANer checks for device.
            # Official RealESRGANer supports a 'device' argument in newer versions, 
            # but usually takes gpu_id. If we pass gpu_id=None, it uses CPU/device set in model.
            
            # For this integration, we will manually move the model to the device 
            # and pass gpu_id=None to preventing it from trying to move it to a CUDA index.
            
            model = model.to(device)
            
            self._upsampler = RealESRGANer(
                scale=netscale,
                model_path=str(model_path),
                dni_weight=None,
                model=model,
                tile=self.tile_size,
                tile_pad=self.tile_pad,
                pre_pad=0,
                half=use_half,
                gpu_id=None, # We handle device placement manually above
                device=device # Pass explicit device for internal operations if supported
            )
            
            self._backend = "pytorch"
            logger.info(f"Real-ESRGAN initialized with PyTorch backend ({device}, FP16: {use_half})")
            return True
            
        except ImportError as e:
            logger.warning(f"PyTorch backend not available: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize PyTorch backend: {e}")
            return False

    def _initialize_ncnn_backend(self) -> bool:
        """Initialize ncnn-based Real-ESRGAN backend (Vulkan GPU)."""
        try:
            from realesrgan_ncnn_py import Realesrgan
            
            # Initialize ncnn upsampler
            self._upsampler = Realesrgan(
                gpuid=self.gpu_id if self.gpu_id >= 0 else -1,
                tta_mode=False,
                tilesize=self.tile_size,
            )
            
            # Load model
            model_name = "realesrgan-x4plus" if self.scale >= 4 else "realesrgan-x2plus"
            self._upsampler.load(model_name)
            
            self._backend = "ncnn"
            logger.info(f"Real-ESRGAN initialized with ncnn backend (GPU: {self.gpu_id})")
            return True
            
        except ImportError as e:
            logger.warning(f"ncnn backend not available: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize ncnn backend: {e}")
            return False

    def initialize(self) -> bool:
        """Initialize the engine with the best available backend.
        
        Returns:
            True if initialization was successful
        """
        # Try PyTorch first (more features), then ncnn
        if self._initialize_pytorch_backend():
            return True
        
        if self._initialize_ncnn_backend():
            return True
        
        logger.error("No Real-ESRGAN backend available. Please install dependencies.")
        return False

    @property
    def is_initialized(self) -> bool:
        """Check if engine is initialized."""
        return self._upsampler is not None

    @property
    def backend(self) -> Optional[str]:
        """Get current backend name."""
        return self._backend

    def _tile_process_pytorch(self, image: np.ndarray, outscale: float) -> np.ndarray:
        """Manual tiled processing for PyTorch backend with thermal throttling."""
        import torch
        import time
        import math
        
        # Image is BGR uint8 (H, W, C)
        h, w, c = image.shape
        tile = self.tile_size
        tile_pad = self.tile_pad
        
        # Output setup
        out_h, out_w = int(h * outscale), int(w * outscale)
        output = np.zeros((out_h, out_w, c), dtype=np.float32)
        weight = np.zeros((out_h, out_w, c), dtype=np.float32)
        
        # Process tiles
        for y in range(0, h, tile):
            for x in range(0, w, tile):
                # Calculate coordinates with padding
                y_pad = max(y - tile_pad, 0)
                x_pad = max(x - tile_pad, 0)
                y_end = min(y + tile + tile_pad, h)
                x_end = min(x + tile + tile_pad, w)
                
                # Extract tile
                tile_img = image[y_pad:y_end, x_pad:x_end, :]
                
                # Process tile (returns uint8 BGR)
                try:
                    # self._upsampler.process() expects raw image
                    # Use internal 'process' method or 'enhance' on small tile
                    tile_out, _ = self._upsampler.enhance(tile_img, outscale=outscale)
                    tile_out = tile_out.astype(np.float32)
                except Exception as e:
                    logger.error(f"Tile processing failed at {x},{y}: {e}")
                    continue

                # Calculate output positions
                out_y_pad = int(y_pad * outscale)
                out_x_pad = int(x_pad * outscale)
                out_h_tile, out_w_tile = tile_out.shape[:2]
                
                # Add to output buffer (simple varying weight blending not easily possible without alpha)
                # For Real-ESRGAN basics, we can just replace or average overlaps. 
                # basicsr uses complex weight maps. We will use a simplified averaging.
                
                # Ensure dimensions match
                target_h = min(out_h_tile, out_h - out_y_pad)
                target_w = min(out_w_tile, out_w - out_x_pad)
                
                if target_h <= 0 or target_w <= 0:
                    continue

                output[out_y_pad:out_y_pad+target_h, out_x_pad:out_x_pad+target_w] += tile_out[:target_h, :target_w]
                weight[out_y_pad:out_y_pad+target_h, out_x_pad:out_x_pad+target_w] += 1
                
                # PULSE DELAY
                time.sleep(0.1)
        
        # Normalize
        output = np.divide(output, weight, where=weight>0)
        return output.astype(np.uint8)

    def upscale(
        self,
        image: np.ndarray,
        outscale: Optional[float] = None,
    ) -> np.ndarray:
        """Upscale an image.
        
        Args:
            image: Input image as numpy array (BGR format, uint8)
            outscale: Output scale (default: use configured scale)
            
        Returns:
            Upscaled image as numpy array
        """
        if not self.is_initialized:
            if not self.initialize():
                raise RuntimeError("Failed to initialize Real-ESRGAN engine")
        
        if outscale is None:
            outscale = self.scale
        
        try:
            if self._backend == "pytorch":
                # PyTorch backend
                # PULSE MODE: Manual tiling with sleep for DirectML to prevent thermal shutdown
                # We bypass self._upsampler.enhance() and do it ourselves if DML is active or safety is needed.
                # Since we already clamped tile_size for DML in init, we can just check backend/device.
                
                device = getattr(self._upsampler, 'device', None)
                is_dml = device and (str(device).startswith("privateuseone") or str(device) == "dml")
                
                if is_dml:
                    logger.info("DirectML Pulse Mode: Using manual tiling with thermal throttling")
                    output = self._tile_process_pytorch(image, outscale)
                else:
                    output, _ = self._upsampler.enhance(image, outscale=outscale)
            else:
                # ncnn backend
                output = self._upsampler.process(image)
                
                # Handle scale mismatch for ncnn (always 4x, need to resize)
                if outscale != 4:
                    import cv2
                    h, w = image.shape[:2]
                    target_h, target_w = int(h * outscale), int(w * outscale)
                    output = cv2.resize(
                        output, 
                        (target_w, target_h), 
                        interpolation=cv2.INTER_LANCZOS4
                    )
            
            return output
            
        except Exception as e:
            logger.error(f"Upscaling failed: {e}")
            raise

    def upscale_with_progress(
        self,
        image: np.ndarray,
        progress_callback: Optional[callable] = None,
        outscale: Optional[float] = None,
    ) -> np.ndarray:
        """Upscale an image with progress reporting.
        
        This method is useful for large images that take time to process.
        
        Args:
            image: Input image
            progress_callback: Callback function (progress: float, message: str)
            outscale: Output scale
            
        Returns:
            Upscaled image
        """
        if progress_callback:
            progress_callback(0.0, "Initializing upscaler...")
        
        if not self.is_initialized:
            self.initialize()
        
        if progress_callback:
            progress_callback(0.1, f"Upscaling with {self._backend} backend...")
        
        result = self.upscale(image, outscale)
        
        if progress_callback:
            progress_callback(1.0, "Upscaling complete")
        
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
        scale = outscale or self.scale
        return (int(input_size[0] * scale), int(input_size[1] * scale))

    def cleanup(self) -> None:
        """Release resources."""
        if self._upsampler is not None:
            del self._upsampler
            self._upsampler = None
            self._backend = None
            
            # Force garbage collection for GPU memory
            try:
                import torch
                torch.cuda.empty_cache()
            except ImportError:
                pass
            
            logger.debug("Real-ESRGAN engine resources released")
