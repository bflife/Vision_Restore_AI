
"""
Vision-Restore AI - HAT Engine

High-quality image super-resolution using HAT (Hybrid Attention Transformer).
HAT provides superior performance by combining channel attention and window adherence.
"""

from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import numpy as np

from vision_restore.core.config import Config, get_config
from vision_restore.core.logger import get_logger

logger = get_logger(__name__)


class HATEngine:
    """HAT image upscaling engine."""

    # Model configurations
    MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
        "hat-l-real-x4": {
            "scale": 4,
            "filename": "HAT_L_Real_Image_x4.pth",
            "url": "https://huggingface.co/vladmandic/sdnext-upscalers/resolve/main/HAT-L-4x.pth",
            "args": {
                "upscale": 4,
                "in_chans": 3,
                "img_size": 64,
                "window_size": 16,
                "compress_ratio": 3,
                "squeeze_factor": 30,
                "conv_scale": 0.01,
                "overlap_ratio": 0.5,
                "img_range": 1.,
                "depths": [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
                "embed_dim": 180,
                "num_heads": [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
                "mlp_ratio": 2,
                "upsampler": "pixelshuffle",
                "resi_connection": "1conv",
            }
        },
    }

    def __init__(
        self,
        config: Optional[Config] = None,
        model_name: str = "hat-l-real-x4",
        scale: int = 4,
        tile_size: int = 256,
        tile_overlap: int = 32,
        gpu_id: int = 0,
    ):
        """Initialize the HAT engine."""
        self.config = config or get_config()
        self.model_name = model_name
        self.scale = scale
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        self.gpu_id = gpu_id
        
        self._model = None
        self._device = None
        self._initialized = False

    def _get_model_config(self) -> Dict[str, Any]:
        """Get configuration for the current model."""
        if self.model_name in self.MODEL_CONFIGS:
            return self.MODEL_CONFIGS[self.model_name]
        return self.MODEL_CONFIGS["hat-l-real-x4"]

    def _get_model_path(self) -> Path:
        """Get path to the model file."""
        config = self._get_model_config()
        model_dir = self.config.models_dir / "hat"
        model_dir.mkdir(parents=True, exist_ok=True)
        return model_dir / config["filename"]

    def _build_model(self):
        """Build the HAT model architecture."""
        try:
            import torch
            from vision_restore.archs.hat_arch import HAT
            
            config = self._get_model_config()
            args = config.get("args", {})
            model = HAT(**args)
            return model
            
        except ImportError as e:
            logger.warning(f"HAT architecture not available: {e}")
            return None

    def initialize(self) -> bool:
        """Initialize the HAT engine."""
        if self._initialized:
            return True
            
        try:
            import torch
            from vision_restore.engine.gpu_backend import gpu_backend
            
            # Use GPU backend manager for device selection
            self._device = gpu_backend.get_torch_device_for_model("hat", self.gpu_id)
            logger.info(f"HAT using device: {self._device}")
            
            # Build model
            self._model = self._build_model()
            if self._model is None:
                return False
            
            # Load weights
            model_path = self._get_model_path()
            if not model_path.exists():
                logger.warning(f"HAT model not found: {model_path}")
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
            # Note: DirectML (dml) often struggles with FP16, so check backend
            is_dml = str(self._device).startswith("privateuseone") or str(self._device) == "dml"
            
            if self.config.use_fp16 and self._device.type == "cuda":
                self._model = self._model.half()
                # Optimize for fixed input sizes (tiling)
                torch.backends.cudnn.benchmark = True
            elif is_dml:
                 logger.info("DirectML detected, enforcing FP32 for stability.")
                 
                 # SAFETY OVERRIDE: DirectML on low-end hardware often crashes/shuts down on large tiles.
                 if self.tile_size > 192:
                     logger.warning(f"DirectML Safety: Reducing tile size from {self.tile_size} to 192 to prevent thermal shutdown.")
                     self.tile_size = 192
                     self.tile_pad = 16
            
            self._initialized = True
            logger.info(f"HAT engine initialized: {self.model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize HAT: {e}")
            return False

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def _pad_image(self, image: np.ndarray, window_size: int = 16) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Pad image to be divisible by window size."""
        h, w = image.shape[:2]
        pad_h = (window_size - h % window_size) % window_size
        pad_w = (window_size - w % window_size) % window_size
        
        if pad_h > 0 or pad_w > 0:
            image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
        
        return image, (h, w)

    def _tile_process(self, image: np.ndarray) -> np.ndarray:
        """Process large image using tiled approach."""
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
        # Prepare batching
        start_points = []
        for y in range(0, h, tile - overlap):
            for x in range(0, w, tile - overlap):
                y_end = min(y + tile, h)
                x_end = min(x + tile, w)
                y_start = max(0, y_end - tile)
                x_start = max(0, x_end - tile)
                start_points.append((y_start, y_end, x_start, x_end))
        
        # Batch size strategy: 
        # For small tiles (<=256) and CUDA, we can speed up by batching.
        # CRITICAL: Disable batching for DirectML (AMD/Intel) to prevent thermal shutdown/driver crashes on laptops.
        is_directml = str(self._device).startswith("privateuseone") or str(self._device) == "dml"
        
        batch_size = 1
        if self.gpu_id >= 0 and self._device.type == "cuda" and not is_directml:
             if self.tile_size <= 256:
                 batch_size = 4
        
        if str(self._device) == "cpu":
            batch_size = 1 # No benefit on CPU usually and hurts RAM
            
        for i in range(0, len(start_points), batch_size):
            batch_indices = start_points[i:i+batch_size]
            batch_tensors = []
            metadata = []
            
            # Prepare batch
            for (y_s, y_e, x_s, x_e) in batch_indices:
                tile_img = image[y_s:y_e, x_s:x_e]
                tile_img, orig_size = self._pad_image(tile_img, window_size=16)
                tensor = torch.from_numpy(tile_img).permute(2, 0, 1)
                batch_tensors.append(tensor)
                metadata.append(((y_s, y_e, x_s, x_e), orig_size))
                
            with torch.no_grad():
                # Stack: (B, C, H, W)
                input_batch = torch.stack(batch_tensors).to(self._device)
                
                if self.config.use_fp16 and self._device.type == "cuda":
                    input_batch = input_batch.half()
                
                # Inference
                output_batch = self._model(input_batch)
                # Output: (B, C, H*s, W*s)
                
                output_batch = output_batch.float().cpu().numpy()
            
            # Place results
            for idx, ((y_s, y_e, x_s, x_e), orig_size) in enumerate(metadata):
                out_tile = output_batch[idx].transpose(1, 2, 0)
                
                oh, ow = orig_size
                out_tile = out_tile[:oh * scale, :ow * scale]
                
                out_y = y_s * scale
                out_x = x_s * scale
                out_h_tile = out_tile.shape[0]
                out_w_tile = out_tile.shape[1]
                
                output[out_y:out_y + out_h_tile, out_x:out_x + out_w_tile] += out_tile
                output[out_y:out_y + out_h_tile, out_x:out_x + out_w_tile] += out_tile
                weight[out_y:out_y + out_h_tile, out_x:out_x + out_w_tile] += 1
                
                # PULSE MODE: Add slight delay to allow Heat/pwer dissipation
                # This mimics "gaming" load (variable) vs "AI" load (constant 100%)
                if is_directml:
                    time.sleep(0.1) # 100ms cool-down buffer per tile/batch

        output = np.divide(output, weight, where=weight > 0)
        return output

    def _try_tile_process(self, image: np.ndarray) -> np.ndarray:
        """Attempt tile processing with auto-retries for smaller tile sizes on OOM."""
        current_tile_size = self.tile_size
        min_tile_size = 128
        
        while current_tile_size >= min_tile_size:
            try:
                # Temporarily set tile size for this attempt
                original_tile_size = self.tile_size
                self.tile_size = current_tile_size
                
                logger.info(f"Attempting tiled processing with tile_size={current_tile_size}")
                result = self._tile_process(image)
                
                # Restore original config
                self.tile_size = original_tile_size
                return result
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.warning(f"OOM with tile_size={current_tile_size}. Retrying with smaller tiles...")
                    try:
                        import torch
                        torch.cuda.empty_cache()
                    except: pass
                    current_tile_size //= 2 # Halve the tile size
                else:
                    self.tile_size = original_tile_size # Restore before raising
                    raise e
        
        raise RuntimeError("Image too large for device memory even with minimum tile size.")

    def upscale(
        self,
        image: np.ndarray,
        outscale: Optional[float] = None,
    ) -> np.ndarray:
        """Upscale an image using HAT."""
        import torch
        import cv2
        
        if not self.is_initialized:
            if not self.initialize():
                raise RuntimeError("Failed to initialize HAT engine")
        
        config = self._get_model_config()
        model_scale = config["scale"]
        
        img = image.astype(np.float32) / 255.0
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        h, w = img.shape[:2]
        
        if h * w > self.tile_size * self.tile_size:
            output = self._try_tile_process(img)
        else:
            try:
                img_padded, orig_size = self._pad_image(img, window_size=16)
                
                with torch.no_grad():
                    img_tensor = torch.from_numpy(img_padded).permute(2, 0, 1).unsqueeze(0)
                    img_tensor = img_tensor.to(self._device)
                    
                    if self.config.use_fp16 and self._device.type == "cuda":
                        img_tensor = img_tensor.half()
                    
                    output = self._model(img_tensor)
                    output = output.squeeze(0).permute(1, 2, 0).float().cpu().numpy()
                
                output = output[:orig_size[0] * model_scale, :orig_size[1] * model_scale]
            except RuntimeError as e:
                # If OOM on full image, fallback to tiling
                if "out of memory" in str(e).lower():
                    logger.warning("OOM detected on full image inference. Switching to tiled processing.")
                    try:
                        import torch
                        torch.cuda.empty_cache()
                    except: pass
                    output = self._try_tile_process(img)
                else:
                    raise e
        
        if outscale and outscale != model_scale:
            target_h = int(h * outscale)
            target_w = int(w * outscale)
            output = cv2.resize(output, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
        
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        output = (np.clip(output, 0, 1) * 255).astype(np.uint8)
        
        return output

    def upscale_with_progress(
        self,
        image: np.ndarray,
        progress_callback: Optional[callable] = None,
        outscale: Optional[float] = None,
    ) -> np.ndarray:
        if progress_callback:
            progress_callback(0.0, "Initializing HAT...")
        
        if not self.is_initialized:
            self.initialize()
        
        if progress_callback:
            progress_callback(0.1, f"Processing with HAT ({self.model_name})...")
        
        result = self.upscale(image, outscale)
        
        if progress_callback:
            progress_callback(1.0, "HAT upscaling complete")
        
        return result

    def cleanup(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
            self._initialized = False
            try:
                import torch
                torch.cuda.empty_cache()
            except ImportError:
                pass
