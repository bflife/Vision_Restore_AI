"""
Vision-Restore AI - Preset Manager

Handles saving, loading, and management of processing presets.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict

from vision_restore.core.logger import get_logger

logger = get_logger(__name__)

PRESETS_FILE = Path("presets.json")

DEFAULT_PRESETS = {
    "Balanced (Default)": {
        "scale": 4,
        "model_name": "realesrgan",
        "face_enhance": True,
        "denoise": True,
        "sharpen": False,
        "output_format": "png",
        "output_quality": 95
    },
    "Old Photo Restoration": {
        "scale": 2,
        "model_name": "swinir_classical",
        "face_enhance": True,
        "denoise": True,
        "sharpen": True,
        "desc": "Best for old, grainy, or damaged photos"
    },
    "Anime / Cartoon 4x": {
        "scale": 4,
        "model_name": "realesrgan",
        "face_enhance": False,
        "denoise": False,
        "sharpen": False,
        "desc": "Sharp upscaling for drawn content"
    },
    "Face Fix Only": {
        "scale": 1,
        "model_name": "none",
        "face_enhance": True,
        "denoise": False,
        "sharpen": False,
        "desc": "Enhance faces without upscaling"
    },
    "Maximum Quality": {
        "scale": 4,
        "model_name": "swinir_real",
        "face_enhance": True,
        "denoise": True,
        "sharpen": True,
        "desc": "Slow but highest fidelity"
    }
}

class PresetManager:
    """Manages application presets."""
    
    def __init__(self):
        self.presets: Dict[str, Dict[str, Any]] = DEFAULT_PRESETS.copy()
        self.user_presets_path = Path.home() / ".vision_restore" / "presets.json"
        self._load_user_presets()

    def _load_user_presets(self):
        """Load user defined presets."""
        if self.user_presets_path.exists():
            try:
                with open(self.user_presets_path, 'r') as f:
                    user_presets = json.load(f)
                    self.presets.update(user_presets)
            except Exception as e:
                logger.error(f"Failed to load presets: {e}")

    def save_user_preset(self, name: str, settings: Dict[str, Any]):
        """Save a new user preset."""
        # Clean settings to only include serializable/relevant keys
        valid_keys = {"scale", "model_name", "face_enhance", "denoise", "sharpen", "output_format", "output_quality"}
        clean_settings = {k: v for k, v in settings.items() if k in valid_keys}
        
        self.presets[name] = clean_settings
        self._persist_presets()
        logger.info(f"Saved preset: {name}")

    def delete_preset(self, name: str) -> bool:
        """Delete a preset (if not default)."""
        if name in DEFAULT_PRESETS:
            logger.warning("Cannot delete default preset")
            return False
            
        if name in self.presets:
            del self.presets[name]
            self._persist_presets()
            return True
        return False

    def _persist_presets(self):
        """Save user presets to disk."""
        # Only save non-default presets
        user_only = {k: v for k, v in self.presets.items() if k not in DEFAULT_PRESETS}
        
        try:
            self.user_presets_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.user_presets_path, 'w') as f:
                json.dump(user_only, f, indent=4)
        except Exception as e:
            logger.error(f"Failed to save presets file: {e}")

    def get_preset_names(self) -> List[str]:
        """Get sorted list of preset names."""
        return sorted(list(self.presets.keys()))

    def get_preset(self, name: str) -> Optional[Dict[str, Any]]:
        """Get settings for a preset."""
        return self.presets.get(name)

# Global instance
preset_manager = PresetManager()
