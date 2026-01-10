"""
Vision-Restore AI v2.0 - Professional Image Upscaling & Restoration

A free, privacy-focused desktop application for image enhancement
using Real-ESRGAN, SwinIR, GFPGAN, and CodeFormer.

v2.0 Features:
- Multi-model support (Real-ESRGAN, SwinIR)
- Advanced face restoration (GFPGAN, CodeFormer)
- Intelligent auto-selection based on image analysis
- Quality metrics and recommendations

Homepage: https://github.com/Rav-xyl/Vision_Restore_AI
"""

# --- Compatibility Patch for BasicSR ---
try:
    import torchvision.transforms.functional as F
    import sys
    # Patch for basicsr which expects 'functional_tensor'
    if 'torchvision.transforms.functional_tensor' not in sys.modules:
        sys.modules['torchvision.transforms.functional_tensor'] = F
except ImportError:
    pass
# ---------------------------------------

__version__ = "2.1.0"
__author__ = "Vision-Restore AI Team"
__license__ = "MIT"

from vision_restore.core.config import Config, get_config
from vision_restore.core.logger import get_logger

__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__license__",
    # Core
    "Config",
    "get_config",
    "get_logger",
]
