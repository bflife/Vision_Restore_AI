"""
Vision-Restore AI - Input Validation

Validation utilities for paths and image files.
"""

from pathlib import Path
from typing import Optional, Tuple, Union
import os

from vision_restore.core.logger import get_logger

logger = get_logger(__name__)

# Supported image formats
SUPPORTED_INPUT_FORMATS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}
SUPPORTED_OUTPUT_FORMATS = {".jpg", ".jpeg", ".png", ".webp"}


def validate_image_path(path: Union[str, Path]) -> Tuple[bool, str]:
    """Validate an input image path.
    
    Args:
        path: Path to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    path = Path(path)
    
    # Check if file exists
    if not path.exists():
        return False, f"File not found: {path}"
    
    # Check if it's a file
    if not path.is_file():
        return False, f"Not a file: {path}"
    
    # Check file extension
    if path.suffix.lower() not in SUPPORTED_INPUT_FORMATS:
        return False, f"Unsupported format: {path.suffix}. Supported: {', '.join(SUPPORTED_INPUT_FORMATS)}"
    
    # Check file size (warn if very large)
    file_size_mb = path.stat().st_size / (1024 * 1024)
    if file_size_mb > 100:
        logger.warning(f"Large file ({file_size_mb:.1f}MB): {path}")
    
    return True, ""


def validate_output_path(path: Union[str, Path]) -> Tuple[bool, str]:
    """Validate an output path.
    
    Args:
        path: Output path to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    path = Path(path)
    
    # Check file extension
    if path.suffix.lower() not in SUPPORTED_OUTPUT_FORMATS:
        return False, f"Unsupported output format: {path.suffix}. Supported: {', '.join(SUPPORTED_OUTPUT_FORMATS)}"
    
    # Check if parent directory exists or can be created
    parent = path.parent
    if not parent.exists():
        try:
            parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            return False, f"Cannot create directory: {parent}"
    
    # Check if path is writable
    if not os.access(parent, os.W_OK):
        return False, f"Directory not writable: {parent}"
    
    # Warn if file exists
    if path.exists():
        logger.warning(f"Output file exists and will be overwritten: {path}")
    
    return True, ""


def validate_directory(path: Union[str, Path]) -> Tuple[bool, str]:
    """Validate a directory path.
    
    Args:
        path: Directory path to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    path = Path(path)
    
    if not path.exists():
        return False, f"Directory not found: {path}"
    
    if not path.is_dir():
        return False, f"Not a directory: {path}"
    
    if not os.access(path, os.R_OK):
        return False, f"Directory not readable: {path}"
    
    return True, ""


def is_valid_image(path: Union[str, Path]) -> bool:
    """Check if a path points to a valid image file.
    
    Args:
        path: Path to check
        
    Returns:
        True if valid image file
    """
    valid, _ = validate_image_path(path)
    return valid


def get_image_files(directory: Union[str, Path]) -> list[Path]:
    """Get all image files in a directory.
    
    Args:
        directory: Directory to scan
        
    Returns:
        List of image file paths
    """
    directory = Path(directory)
    
    if not directory.is_dir():
        return []
    
    files = []
    for file in directory.iterdir():
        if file.is_file() and file.suffix.lower() in SUPPORTED_INPUT_FORMATS:
            files.append(file)
    
    return sorted(files)


def validate_scale(scale: int) -> Tuple[bool, str]:
    """Validate upscale factor.
    
    Args:
        scale: Scale factor to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    valid_scales = {2, 4, 8}
    
    if scale not in valid_scales:
        return False, f"Invalid scale: {scale}. Supported: {valid_scales}"
    
    return True, ""


def validate_quality(quality: int) -> Tuple[bool, str]:
    """Validate output quality setting.
    
    Args:
        quality: Quality value to validate (1-100)
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(quality, int):
        return False, "Quality must be an integer"
    
    if quality < 1 or quality > 100:
        return False, f"Quality must be between 1 and 100, got {quality}"
    
    return True, ""


def estimate_output_size(
    input_path: Union[str, Path],
    scale: int,
) -> Optional[dict]:
    """Estimate the output size for an upscaled image.
    
    Args:
        input_path: Path to input image
        scale: Upscale factor
        
    Returns:
        Dictionary with estimated sizes, or None if failed
    """
    try:
        import cv2
        
        path = Path(input_path)
        if not path.exists():
            return None
        
        # Read image to get dimensions
        img = cv2.imread(str(path))
        if img is None:
            return None
        
        h, w = img.shape[:2]
        
        # Calculate output dimensions
        out_w = w * scale
        out_h = h * scale
        
        # Estimate file size (rough approximation)
        # Assuming 24-bit color, compressed
        input_size = path.stat().st_size
        compression_ratio = input_size / (w * h * 3)
        estimated_output_size = out_w * out_h * 3 * compression_ratio
        
        return {
            "input_width": w,
            "input_height": h,
            "output_width": out_w,
            "output_height": out_h,
            "scale": scale,
            "input_megapixels": (w * h) / 1_000_000,
            "output_megapixels": (out_w * out_h) / 1_000_000,
            "input_size_mb": input_size / (1024 * 1024),
            "estimated_output_size_mb": estimated_output_size / (1024 * 1024),
        }
        
    except Exception as e:
        logger.error(f"Failed to estimate output size: {e}")
        return None
