"""
Vision-Restore AI - Image I/O Utilities

Functions for loading, saving, and converting images.
"""

from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np
import cv2
from PIL import Image

from vision_restore.core.logger import get_logger

logger = get_logger(__name__)

# Supported image formats
SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}


def load_image(
    path: Union[str, Path],
    convert_rgb: bool = True,
) -> Optional[np.ndarray]:
    """Load an image from disk.
    
    Args:
        path: Path to image file
        convert_rgb: Convert to RGB if True (default), else keep BGR
        
    Returns:
        Image as numpy array (BGR format by default), or None if failed
    """
    path = Path(path)
    
    if not path.exists():
        logger.error(f"Image not found: {path}")
        return None
    
    if path.suffix.lower() not in SUPPORTED_FORMATS:
        logger.warning(f"Unsupported format: {path.suffix}")
    
    try:
        # Try OpenCV first (faster, handles more formats)
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        
        if image is None:
            # Fallback to PIL for some formats
            pil_image = Image.open(path)
            image = pil_to_numpy(pil_image)
        
        if image is None:
            logger.error(f"Failed to load image: {path}")
            return None
        
        logger.debug(f"Loaded image: {path} ({image.shape[1]}x{image.shape[0]})")
        return image
        
    except Exception as e:
        logger.error(f"Error loading image {path}: {e}")
        return None


def save_image(
    image: np.ndarray,
    path: Union[str, Path],
    quality: int = 95,
) -> bool:
    """Save an image to disk.
    
    Args:
        image: Image as numpy array (BGR format)
        path: Output path
        quality: JPEG/WebP quality (1-100)
        
    Returns:
        True if saved successfully
    """
    path = Path(path)
    
    # Create parent directories
    path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        suffix = path.suffix.lower()
        
        if suffix in {".jpg", ".jpeg"}:
            params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        elif suffix == ".png":
            # PNG compression level (0-9, where 9 is max compression)
            compression = int((100 - quality) / 100 * 9)
            params = [cv2.IMWRITE_PNG_COMPRESSION, compression]
        elif suffix == ".webp":
            params = [cv2.IMWRITE_WEBP_QUALITY, quality]
        else:
            params = []
        
        success = cv2.imwrite(str(path), image, params)
        
        if success:
            logger.debug(f"Saved image: {path}")
        else:
            logger.error(f"Failed to save image: {path}")
        
        return success
        
    except Exception as e:
        logger.error(f"Error saving image {path}: {e}")
        return False


def numpy_to_pil(image: np.ndarray) -> Image.Image:
    """Convert numpy array to PIL Image.
    
    Args:
        image: Numpy array (BGR or RGB format)
        
    Returns:
        PIL Image in RGB format
    """
    if len(image.shape) == 2:
        # Grayscale
        return Image.fromarray(image, mode="L")
    elif image.shape[2] == 3:
        # BGR to RGB
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb, mode="RGB")
    elif image.shape[2] == 4:
        # BGRA to RGBA
        rgba = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
        return Image.fromarray(rgba, mode="RGBA")
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")


def pil_to_numpy(image: Image.Image) -> np.ndarray:
    """Convert PIL Image to numpy array.
    
    Args:
        image: PIL Image
        
    Returns:
        Numpy array in BGR format
    """
    # Convert to RGB if necessary
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Convert to numpy and BGR
    arr = np.array(image)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def resize_image(
    image: np.ndarray,
    max_size: int = 4096,
    min_size: int = 32,
) -> np.ndarray:
    """Resize image to fit within max size while maintaining aspect ratio.
    
    Args:
        image: Input image
        max_size: Maximum dimension
        min_size: Minimum dimension
        
    Returns:
        Resized image
    """
    h, w = image.shape[:2]
    
    # Check if resize is needed
    if max(h, w) <= max_size and min(h, w) >= min_size:
        return image
    
    # Calculate new dimensions
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
    else:
        scale = min_size / min(h, w)
    
    new_w = max(min_size, int(w * scale))
    new_h = max(min_size, int(h * scale))
    
    # Use LANCZOS for downscaling, CUBIC for upscaling
    if scale < 1:
        interpolation = cv2.INTER_LANCZOS4
    else:
        interpolation = cv2.INTER_CUBIC
    
    return cv2.resize(image, (new_w, new_h), interpolation=interpolation)


def get_image_info(image: np.ndarray) -> dict:
    """Get information about an image.
    
    Args:
        image: Input image
        
    Returns:
        Dictionary with image information
    """
    h, w = image.shape[:2]
    channels = image.shape[2] if len(image.shape) == 3 else 1
    
    return {
        "width": w,
        "height": h,
        "channels": channels,
        "dtype": str(image.dtype),
        "size_bytes": image.nbytes,
        "size_mb": image.nbytes / (1024 * 1024),
        "megapixels": (w * h) / 1_000_000,
    }


def create_thumbnail(
    image: np.ndarray,
    size: Tuple[int, int] = (256, 256),
) -> np.ndarray:
    """Create a thumbnail of an image.
    
    Args:
        image: Input image
        size: Target thumbnail size (width, height)
        
    Returns:
        Thumbnail image
    """
    h, w = image.shape[:2]
    target_w, target_h = size
    
    # Calculate scale to fit
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize
    thumbnail = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    return thumbnail


def compare_images(
    image1: np.ndarray,
    image2: np.ndarray,
) -> dict:
    """Compare two images and return metrics.
    
    Args:
        image1: First image
        image2: Second image
        
    Returns:
        Dictionary with comparison metrics
    """
    # Ensure same size for comparison
    if image1.shape != image2.shape:
        # Resize image2 to match image1
        image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
    
    # Convert to grayscale for some metrics
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    
    # Mean absolute difference
    mae = np.mean(np.abs(image1.astype(float) - image2.astype(float)))
    
    # Peak Signal-to-Noise Ratio (PSNR)
    mse = np.mean((image1.astype(float) - image2.astype(float)) ** 2)
    if mse == 0:
        psnr = float("inf")
    else:
        psnr = 20 * np.log10(255.0 / np.sqrt(mse))
    
    return {
        "mae": mae,
        "psnr": psnr,
        "size_ratio": image2.size / image1.size if image1.size > 0 else 0,
    }
