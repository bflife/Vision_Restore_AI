"""
Vision-Restore AI - Model Manager

Handles model downloading, verification, and loading.
"""

import shutil
from pathlib import Path
from typing import Callable, Dict, Optional

from vision_restore.core.config import Config, get_config
from vision_restore.core.logger import get_logger

logger = get_logger(__name__)


# Model information with download URLs and checksums
MODEL_INFO: Dict[str, Dict[str, str]] = {
    # Real-ESRGAN models
    "realesrgan-x2plus": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
        "filename": "RealESRGAN_x2plus.pth",
        "size_mb": 64,
    },
    "realesrgan-x4plus": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "filename": "RealESRGAN_x4plus.pth",
        "size_mb": 64,
    },
    "realesrgan-x4plus-anime": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
        "filename": "RealESRGAN_x4plus_anime_6B.pth",
        "size_mb": 17,
    },
    # Face detection/parsing models
    "GFPGANv1.4": {
        "url": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
        "filename": "GFPGANv1.4.pth",
        "size_mb": 348,
    },
    "detection_Resnet50": {
        "url": "https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth",
        "filename": "detection_Resnet50_Final.pth",
        "size_mb": 109,
    },
    "parsing_parsenet": {
        "url": "https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth",
        "filename": "parsing_parsenet.pth",
        "size_mb": 81,
    },
    # v2.0: SwinIR models
    "swinir-classical-x2": {
        "url": "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/001_classicalSR_DIV2K_s48w8_SwinIR-M_x2.pth",
        "filename": "001_classicalSR_DIV2K_s48w8_SwinIR-M_x2.pth",
        "size_mb": 47,
    },
    "swinir-classical-x4": {
        "url": "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/001_classicalSR_DIV2K_s48w8_SwinIR-M_x4.pth",
        "filename": "001_classicalSR_DIV2K_s48w8_SwinIR-M_x4.pth",
        "size_mb": 47,
    },
    "swinir-real-x4": {
        "url": "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth",
        "filename": "003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth",
        "size_mb": 136,
    },
    "swinir-lightweight-x4": {
        "url": "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x4.pth",
        "filename": "002_lightweightSR_DIV2K_s64w8_SwinIR-S_x4.pth",
        "size_mb": 3,
    },
    # v2.0: CodeFormer model
    "codeformer": {
        "url": "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth",
        "filename": "codeformer.pth",
        "size_mb": 376,
    },
    # v3.0: HAT models
    "hat-l-real-x4": {
        "url": "https://huggingface.co/vladmandic/sdnext-upscalers/resolve/main/HAT-L-4x.pth",
        "filename": "HAT_L_Real_Image_x4.pth",
        "size_mb": 350,
        "hash": "0e599b53", # Partial SHA256/MD5 if full is not known, or use file size check at least
    },
}


class ModelManager:
    """Manages AI model downloading, caching, and loading."""

    def __init__(self, config: Optional[Config] = None):
        """Initialize the model manager.
        
        Args:
            config: Application configuration (uses global config if not provided)
        """
        self.config = config or get_config()
        self.models_dir = self.config.models_dir
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for different model types
        self.realesrgan_dir = self.models_dir / "realesrgan"
        self.gfpgan_dir = self.models_dir / "gfpgan"
        self.facexlib_dir = self.models_dir / "facexlib"
        self.swinir_dir = self.models_dir / "swinir"
        self.codeformer_dir = self.models_dir / "codeformer"
        self.hat_dir = self.models_dir / "hat"
        
        for dir_path in [self.realesrgan_dir, self.gfpgan_dir, self.facexlib_dir, 
                         self.swinir_dir, self.codeformer_dir, self.hat_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def get_model_path(self, model_name: str) -> Path:
        """Get the local path for a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Path to the model file
        """
        if model_name not in MODEL_INFO:
            raise ValueError(f"Unknown model: {model_name}")
        
        info = MODEL_INFO[model_name]
        
        # Determine target directory based on model type
        if "realesrgan" in model_name.lower():
            target_dir = self.realesrgan_dir
        elif "gfpgan" in model_name.lower():
            target_dir = self.gfpgan_dir
        elif "swinir" in model_name.lower():
            target_dir = self.swinir_dir
        elif "codeformer" in model_name.lower():
            target_dir = self.codeformer_dir
        elif "hat" in model_name.lower():
            target_dir = self.hat_dir
        else:
            target_dir = self.facexlib_dir
        
        return target_dir / info["filename"]

    def is_model_available(self, model_name: str) -> bool:
        """Check if a model is downloaded and available.
        
        Args:
            model_name: Name of the model
            
        Returns:
            True if model is available locally
        """
        try:
            model_path = self.get_model_path(model_name)
            return model_path.exists() and model_path.stat().st_size > 0
        except ValueError:
            return False

    def get_required_models(self, face_enhance: bool = True) -> list:
        """Get list of required models based on configuration.
        
        Args:
            face_enhance: Whether face enhancement is enabled
            
        Returns:
            List of required model names
        """
        models = []
        
        # Get upscale model from config
        upscale_model = getattr(self.config, 'upscale_model', 'auto')
        scale = self.config.scale
        
        # Real-ESRGAN models (always needed for realesrgan or auto mode)
        if upscale_model in ['auto', 'realesrgan']:
            if scale == 2:
                models.append("realesrgan-x2plus")
            else:
                models.append("realesrgan-x4plus")
        
        # v2.0: SwinIR models
        if upscale_model in ['auto', 'swinir']:
            if scale == 2:
                models.append("swinir-classical-x2")
            else:
                models.append("swinir-real-x4")

        # v3.0: HAT models
        if upscale_model == 'hat':
            models.append("hat-l-real-x4")
        
        # Face enhancement models
        if face_enhance:
            face_model = getattr(self.config, 'face_model', 'auto')
            
            # GFPGAN + detection models (always needed for gfpgan or auto)
            if face_model in ['auto', 'gfpgan']:
                models.extend([
                    "GFPGANv1.4",
                    "detection_Resnet50",
                    "parsing_parsenet",
                ])
            
            # v2.0: CodeFormer model
            if face_model in ['auto', 'codeformer']:
                models.append("codeformer")
                # CodeFormer also needs face detection
                if "detection_Resnet50" not in models:
                    models.append("detection_Resnet50")
                if "parsing_parsenet" not in models:
                    models.append("parsing_parsenet")
        
        return models

    def get_missing_models(self, face_enhance: bool = True) -> list:
        """Get list of models that need to be downloaded.
        
        Args:
            face_enhance: Whether face enhancement is enabled
            
        Returns:
            List of missing model names
        """
        required = self.get_required_models(face_enhance)
        return [m for m in required if not self.is_model_available(m)]

    def download_model(
        self,
        model_name: str,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        max_retries: int = 3,
    ) -> bool:
        """Download a model from the web.
        
        Args:
            model_name: Name of the model to download
            progress_callback: Optional callback for progress updates
            max_retries: Maximum number of retry attempts
            
        Returns:
            True if download was successful
        """
        if model_name not in MODEL_INFO:
            logger.error(f"Unknown model: {model_name}")
            return False
        
        info = MODEL_INFO[model_name]
        url = info["url"]
        target_path = self.get_model_path(model_name)
        
        logger.info(f"Downloading {model_name} ({info['size_mb']}MB)...")
        
        if progress_callback:
            progress_callback(0, f"Downloading {model_name}...")
        
        for attempt in range(max_retries):
            try:
                import requests
                
                # Download with streaming - NO timeout for large model files
                headers = {"User-Agent": "Vision-Restore-AI/2.0"}
                response = requests.get(
                    url, 
                    headers=headers, 
                    stream=True,
                    timeout=30 # Connect timeout
                )
                response.raise_for_status()
                
                total_size = int(response.headers.get("content-length", 0))
                downloaded = 0
                chunk_size = 1024 * 1024  # 1MB chunks
                
                temp_path = target_path.with_suffix(".tmp")
                
                with open(temp_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            
                            if total_size > 0 and progress_callback:
                                progress = downloaded / total_size
                                progress_callback(
                                    progress,
                                    f"Downloading {model_name}: {downloaded // (1024*1024)}MB / {total_size // (1024*1024)}MB"
                                )
                
                # Move temp file to final location
                shutil.move(str(temp_path), str(target_path))
                
                logger.info(f"Successfully downloaded {model_name}")
                
                if progress_callback:
                    progress_callback(1.0, f"Downloaded {model_name}")
                
                return True
                
            except Exception as e:
                logger.warning(f"Download attempt {attempt + 1}/{max_retries} failed for {model_name}: {e}")
                
                # Clean up temp file if it exists
                temp_path = target_path.with_suffix(".tmp")
                if temp_path.exists():
                    temp_path.unlink()
                
                if attempt < max_retries - 1:
                    if progress_callback:
                        progress_callback(0, f"Retrying {model_name} ({attempt + 2}/{max_retries})...")
                    import time
                    time.sleep(2)  # Wait before retry
        
        logger.error(f"Failed to download {model_name} after {max_retries} attempts")
        return False

    def download_all_required(
        self,
        face_enhance: bool = True,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> bool:
        """Download all required models.
        
        Args:
            face_enhance: Whether to include face enhancement models
            progress_callback: Optional callback for progress updates
            
        Returns:
            True if all downloads were successful
        """
        missing = self.get_missing_models(face_enhance)
        
        if not missing:
            logger.info("All required models are already downloaded")
            return True
        
        logger.info(f"Need to download {len(missing)} model(s)")
        
        total_models = len(missing)
        for i, model_name in enumerate(missing):
            # Create per-model progress callback
            def model_progress(p: float, msg: str):
                if progress_callback:
                    overall = (i + p) / total_models
                    progress_callback(overall, msg)
            
            if not self.download_model(model_name, model_progress):
                logger.error(f"Failed to download {model_name}")
                return False
        
        logger.info("All required models downloaded successfully")
        return True

    def get_total_download_size(self, face_enhance: bool = True) -> int:
        """Get total download size for missing models in MB.
        
        Args:
            face_enhance: Whether to include face enhancement models
            
        Returns:
            Total size in megabytes
        """
        missing = self.get_missing_models(face_enhance)
        return sum(MODEL_INFO[m]["size_mb"] for m in missing)

    def cleanup_models(self) -> None:
        """Remove all downloaded models to free disk space."""
        for model_dir in [self.realesrgan_dir, self.gfpgan_dir, self.facexlib_dir]:
            if model_dir.exists():
                shutil.rmtree(model_dir)
                model_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("All models have been removed")
