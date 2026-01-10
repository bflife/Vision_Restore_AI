"""
Vision-Restore AI - Configuration Management

Centralized configuration with sensible defaults and user preferences.
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class Config:
    """Application configuration with all settings."""

    # ═══════════════════════════════════════════════════════════════════════════
    # Model Settings
    # ═══════════════════════════════════════════════════════════════════════════
    scale: int = 4
    available_scales: List[int] = field(default_factory=lambda: [2, 4, 8])
    
    # Model selection
    realesrgan_model: str = "realesrgan-x4plus"
    gfpgan_model: str = "GFPGANv1.4"
    
    # v2.0 Model selection (auto, realesrgan, swinir)
    upscale_model: str = "auto"
    face_model: str = "auto"  # auto, gfpgan, codeformer
    swinir_model: str = "swinir-real-x4"
    codeformer_fidelity: float = 0.7  # 0=quality, 1=fidelity
    
    # Model paths (auto-populated on first run)
    models_dir: Path = field(default_factory=lambda: Path.home() / ".vision-restore" / "models")

    # ═══════════════════════════════════════════════════════════════════════════
    # Processing Settings
    # ═══════════════════════════════════════════════════════════════════════════
    
    # Tiling (memory optimization)
    tile_size: int = 512
    tile_pad: int = 32  # Overlap to prevent seams
    
    # Feature toggles
    enable_face_enhance: bool = True
    enable_denoise: bool = True
    enable_color_correct: bool = True
    enable_sharpen: bool = True
    enable_quality_metrics: bool = True  # v2.0: Enable quality analysis
    
    # Denoise settings
    denoise_strength: float = 0.5  # 0.0 to 1.0
    
    # Sharpen settings
    sharpen_amount: float = 0.3  # 0.0 to 1.0
    sharpen_radius: int = 1

    # ═══════════════════════════════════════════════════════════════════════════
    # GPU Settings
    # ═══════════════════════════════════════════════════════════════════════════
    gpu_id: int = 0  # -1 for CPU
    use_fp16: bool = True  # Half precision for faster processing
    
    # ═══════════════════════════════════════════════════════════════════════════
    # Output Settings
    # ═══════════════════════════════════════════════════════════════════════════
    output_format: str = "png"
    output_quality: int = 95  # For JPEG/WebP
    
    # ═══════════════════════════════════════════════════════════════════════════
    # UI Settings
    # ═══════════════════════════════════════════════════════════════════════════
    theme: str = "dark"
    accent_color: str = "blue"
    window_width: int = 1400
    window_height: int = 900
    
    # ═══════════════════════════════════════════════════════════════════════════
    # Quality Presets
    # ═══════════════════════════════════════════════════════════════════════════
    presets: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "fast": {
            "tile_size": 256,
            "enable_denoise": False,
            "enable_sharpen": False,
            "use_fp16": True,
        },
        "balanced": {
            "tile_size": 512,
            "enable_denoise": True,
            "enable_sharpen": True,
            "denoise_strength": 0.3,
            "sharpen_amount": 0.2,
        },
        "quality": {
            "tile_size": 768,
            "enable_denoise": True,
            "enable_sharpen": True,
            "denoise_strength": 0.5,
            "sharpen_amount": 0.4,
        },
    })

    # ═══════════════════════════════════════════════════════════════════════════
    # Model URLs (for auto-download)
    # ═══════════════════════════════════════════════════════════════════════════
    model_urls: Dict[str, str] = field(default_factory=lambda: {
        "realesrgan-x2plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
        "realesrgan-x4plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "realesrgan-x4plus-anime": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
        "GFPGANv1.4": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
        # v2.0: SwinIR models
        "swinir-classical-x2": "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/001_classicalSR_DIV2K_s48w8_SwinIR-M_x2.pth",
        "swinir-classical-x4": "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/001_classicalSR_DIV2K_s48w8_SwinIR-M_x4.pth",
        "swinir-real-x4": "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth",
        "swinir-lightweight-x4": "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x4.pth",
        # v2.0: CodeFormer model
        "codeformer": "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth",
        # v3.0: HAT models
        "hat-l-real-x4": "https://huggingface.co/vladmandic/sdnext-upscalers/resolve/main/HAT-L-4x.pth",
    })

    def __post_init__(self):
        """Initialize paths and create directories."""
        self.models_dir = Path(self.models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Config file path
        self.config_file = Path.home() / ".vision-restore" / "config.json"

    def apply_preset(self, preset_name: str) -> None:
        """Apply a quality preset to the configuration."""
        if preset_name not in self.presets:
            raise ValueError(f"Unknown preset: {preset_name}")
        
        preset = self.presets[preset_name]
        for key, value in preset.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def get_model_path(self, model_name: str) -> Path:
        """Get the local path for a model file."""
        return self.models_dir / f"{model_name}.pth"

    def is_model_downloaded(self, model_name: str) -> bool:
        """Check if a model is already downloaded."""
        return self.get_model_path(model_name).exists()

    def save(self) -> None:
        """Save configuration to file."""
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to serializable dict
        config_dict = {
            "scale": self.scale,
            "tile_size": self.tile_size,
            "tile_pad": self.tile_pad,
            "enable_face_enhance": self.enable_face_enhance,
            "enable_denoise": self.enable_denoise,
            "enable_color_correct": self.enable_color_correct,
            "enable_sharpen": self.enable_sharpen,
            "denoise_strength": self.denoise_strength,
            "sharpen_amount": self.sharpen_amount,
            "sharpen_radius": self.sharpen_radius,
            "gpu_id": self.gpu_id,
            "use_fp16": self.use_fp16,
            "output_format": self.output_format,
            "output_quality": self.output_quality,
            "theme": self.theme,
            "accent_color": self.accent_color,
            "window_width": self.window_width,
            "window_height": self.window_height,
            "upscale_model": self.upscale_model,
            "face_model": self.face_model,
            "swinir_model": self.swinir_model,
            "codeformer_fidelity": self.codeformer_fidelity,
            "enable_quality_metrics": self.enable_quality_metrics,
        }
        
        with open(self.config_file, "w") as f:
            json.dump(config_dict, f, indent=2)

    def load(self) -> None:
        """Load configuration from file."""
        if not self.config_file.exists():
            return
        
        try:
            with open(self.config_file, "r") as f:
                config_dict = json.load(f)
            
            for key, value in config_dict.items():
                if hasattr(self, key):
                    setattr(self, key, value)
        except (json.JSONDecodeError, IOError):
            pass  # Use defaults if config is invalid

    def get_optimal_tile_size(self, image_width: int, image_height: int) -> int:
        """Calculate optimal tile size based on image dimensions and available memory."""
        # For very small images, use the full image
        if image_width <= 512 and image_height <= 512:
            return max(image_width, image_height)
        
        # For larger images, use configured tile size
        return self.tile_size

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "scale": self.scale,
            "tile_size": self.tile_size,
            "tile_pad": self.tile_pad,
            "enable_face_enhance": self.enable_face_enhance,
            "enable_denoise": self.enable_denoise,
            "enable_color_correct": self.enable_color_correct,
            "enable_sharpen": self.enable_sharpen,
            "denoise_strength": self.denoise_strength,
            "sharpen_amount": self.sharpen_amount,
            "gpu_id": self.gpu_id,
            "use_fp16": self.use_fp16,
            "output_format": self.output_format,
            "output_quality": self.output_quality,
            "upscale_model": self.upscale_model,
            "face_model": self.face_model,
            "swinir_model": self.swinir_model,
            "codeformer_fidelity": self.codeformer_fidelity,
            "enable_quality_metrics": self.enable_quality_metrics,
        }


# Global config instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = Config()
        _config.load()
    return _config
