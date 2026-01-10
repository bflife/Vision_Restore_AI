"""
Vision-Restore AI - ML Engine Layer v2.0

Image enhancement engines including Real-ESRGAN, SwinIR, GFPGAN, CodeFormer,
quality metrics, and processing utilities.
"""

from vision_restore.engine.orchestrator import (
    EnhancementOrchestrator,
    create_orchestrator,
    get_available_models,
)
from vision_restore.engine.tiling import TilingEngine
from vision_restore.engine.realesrgan_engine import RealESRGANEngine
from vision_restore.engine.gfpgan_engine import GFPGANEngine

# Optional v2.0 engines (may not be available depending on dependencies)
try:
    from vision_restore.engine.swinir_engine import SwinIREngine, get_available_swinir_models
except ImportError:
    SwinIREngine = None
    get_available_swinir_models = None

try:
    from vision_restore.engine.codeformer_engine import CodeFormerEngine, FaceRestorationSettings
except ImportError:
    CodeFormerEngine = None
    FaceRestorationSettings = None

try:
    from vision_restore.engine.quality_metrics import QualityMetrics, recommend_processing
except ImportError:
    QualityMetrics = None
    recommend_processing = None

__all__ = [
    "EnhancementOrchestrator",
    "create_orchestrator",
    "get_available_models",
    "TilingEngine",
    "RealESRGANEngine",
    "GFPGANEngine",
    "SwinIREngine",
    "CodeFormerEngine",
    "FaceRestorationSettings",
    "QualityMetrics",
    "recommend_processing",
    "get_available_swinir_models",
]
