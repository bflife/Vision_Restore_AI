"""
Vision-Restore AI - Image Pre-Processing

Advanced pre-processing pipeline for optimal upscaling results.
Includes denoising, color correction, and artifact reduction.
"""

from typing import Optional, Tuple
import numpy as np
import cv2

from vision_restore.core.config import Config, get_config
from vision_restore.core.logger import get_logger

logger = get_logger(__name__)


class PreProcessor:
    """Image pre-processing for optimal upscaling results.
    
    Techniques:
    - Adaptive denoising (Bilateral + Non-local Means)
    - Color correction (white balance, histogram equalization)
    - JPEG artifact reduction
    - Edge-preserving smoothing
    """

    def __init__(self, config: Optional[Config] = None):
        """Initialize the pre-processor.
        
        Args:
            config: Application configuration
        """
        self.config = config or get_config()

    def denoise(
        self,
        image: np.ndarray,
        strength: float = 0.5,
        preserve_detail: bool = True,
    ) -> np.ndarray:
        """Apply intelligent denoising while preserving detail.
        
        Uses a hybrid approach combining bilateral filtering and
        non-local means for optimal results.
        
        Args:
            image: Input image (BGR, uint8)
            strength: Denoising strength (0.0 to 1.0)
            preserve_detail: Use edge-preserving techniques
            
        Returns:
            Denoised image
        """
        if strength <= 0:
            return image
        
        # Clip strength to valid range
        strength = min(1.0, max(0.0, strength))
        
        # Calculate parameters based on strength
        bilateral_d = int(5 + strength * 4)  # 5-9
        bilateral_sigma = 50 + strength * 50  # 50-100
        nlm_h = 3 + strength * 7  # 3-10
        
        if preserve_detail:
            # Step 1: Bilateral filter for edge-preserving smoothing
            bilateral = cv2.bilateralFilter(
                image,
                d=bilateral_d,
                sigmaColor=bilateral_sigma,
                sigmaSpace=bilateral_sigma,
            )
            
            # Step 2: Non-local means for texture-aware denoising
            if len(image.shape) == 3:
                # Use positional args for OpenCV compatibility across versions
                nlm = cv2.fastNlMeansDenoisingColored(
                    bilateral,
                    None,
                    float(nlm_h),  # h
                    float(nlm_h),  # hColor
                    7,  # templateWindowSize
                    21,  # searchWindowSize
                )
            else:
                nlm = cv2.fastNlMeansDenoising(
                    bilateral,
                    None,
                    float(nlm_h),
                    7,
                    21,
                )
            
            # Blend based on strength
            result = cv2.addWeighted(
                image, 1.0 - strength * 0.7,
                nlm, strength * 0.7,
                0,
            )
        else:
            # Simple Gaussian blur for fast denoising
            kernel_size = int(3 + strength * 4) | 1  # Ensure odd
            result = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
            result = cv2.addWeighted(image, 1.0 - strength, result, strength, 0)
        
        logger.debug(f"Applied denoising (strength={strength:.2f})")
        return result

    def reduce_jpeg_artifacts(
        self,
        image: np.ndarray,
        strength: float = 0.5,
    ) -> np.ndarray:
        """Reduce JPEG compression artifacts.
        
        Uses bilateral filtering to smooth block artifacts while
        preserving edges.
        
        Args:
            image: Input image
            strength: Artifact reduction strength (0.0 to 1.0)
            
        Returns:
            Image with reduced artifacts
        """
        if strength <= 0:
            return image
        
        # Bilateral filter is effective for block artifact removal
        d = int(5 + strength * 4)
        sigma = 30 + strength * 40
        
        filtered = cv2.bilateralFilter(image, d, sigma, sigma)
        
        # Blend with original to control strength
        result = cv2.addWeighted(image, 1.0 - strength, filtered, strength, 0)
        
        logger.debug(f"Reduced JPEG artifacts (strength={strength:.2f})")
        return result

    def auto_white_balance(self, image: np.ndarray) -> np.ndarray:
        """Apply automatic white balance correction.
        
        Uses the Gray World assumption for color correction.
        
        Args:
            image: Input image (BGR, uint8)
            
        Returns:
            White-balanced image
        """
        if len(image.shape) != 3 or image.shape[2] != 3:
            return image
        
        # Convert to float
        img_float = image.astype(np.float32)
        
        # Calculate mean of each channel
        b_mean = np.mean(img_float[:, :, 0])
        g_mean = np.mean(img_float[:, :, 1])
        r_mean = np.mean(img_float[:, :, 2])
        
        # Calculate overall mean
        gray_mean = (b_mean + g_mean + r_mean) / 3
        
        # Avoid division by zero
        if b_mean < 1 or g_mean < 1 or r_mean < 1:
            return image
        
        # Calculate scaling factors
        b_scale = gray_mean / b_mean
        g_scale = gray_mean / g_mean
        r_scale = gray_mean / r_mean
        
        # Apply correction
        result = img_float.copy()
        result[:, :, 0] *= b_scale
        result[:, :, 1] *= g_scale
        result[:, :, 2] *= r_scale
        
        # Clip and convert back to uint8
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        logger.debug("Applied auto white balance")
        return result

    def enhance_contrast(
        self,
        image: np.ndarray,
        clip_limit: float = 2.0,
        tile_size: int = 8,
    ) -> np.ndarray:
        """Enhance image contrast using CLAHE.
        
        Contrast Limited Adaptive Histogram Equalization provides
        local contrast enhancement without over-amplifying noise.
        
        Args:
            image: Input image
            clip_limit: CLAHE clip limit
            tile_size: CLAHE tile grid size
            
        Returns:
            Contrast-enhanced image
        """
        if len(image.shape) == 3:
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(
                clipLimit=clip_limit,
                tileGridSize=(tile_size, tile_size),
            )
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            
            # Convert back to BGR
            result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            # Grayscale image
            clahe = cv2.createCLAHE(
                clipLimit=clip_limit,
                tileGridSize=(tile_size, tile_size),
            )
            result = clahe.apply(image)
        
        logger.debug(f"Enhanced contrast (clip_limit={clip_limit})")
        return result

    def auto_levels(
        self,
        image: np.ndarray,
        clip_percent: float = 1.0,
    ) -> np.ndarray:
        """Apply automatic levels adjustment.
        
        Stretches the histogram to use the full dynamic range.
        
        Args:
            image: Input image
            clip_percent: Percentage of pixels to clip at ends
            
        Returns:
            Levels-adjusted image
        """
        if len(image.shape) == 3:
            # Process each channel independently
            result = np.zeros_like(image)
            for c in range(3):
                result[:, :, c] = self._apply_levels_channel(
                    image[:, :, c], clip_percent
                )
        else:
            result = self._apply_levels_channel(image, clip_percent)
        
        logger.debug(f"Applied auto levels (clip={clip_percent}%)")
        return result

    def _apply_levels_channel(
        self,
        channel: np.ndarray,
        clip_percent: float,
    ) -> np.ndarray:
        """Apply levels adjustment to a single channel."""
        # Calculate histogram
        hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
        hist = hist.flatten()
        
        # Find clip points
        total_pixels = channel.size
        clip_pixels = int(total_pixels * clip_percent / 100)
        
        # Find low clip point
        cumsum = 0
        low_clip = 0
        for i in range(256):
            cumsum += hist[i]
            if cumsum > clip_pixels:
                low_clip = i
                break
        
        # Find high clip point
        cumsum = 0
        high_clip = 255
        for i in range(255, -1, -1):
            cumsum += hist[i]
            if cumsum > clip_pixels:
                high_clip = i
                break
        
        # Avoid division by zero
        if high_clip <= low_clip:
            return channel
        
        # Apply levels
        result = channel.astype(np.float32)
        result = (result - low_clip) * 255.0 / (high_clip - low_clip)
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        return result

    def process(
        self,
        image: np.ndarray,
        denoise: bool = True,
        color_correct: bool = True,
        reduce_artifacts: bool = True,
        enhance_contrast: bool = False,
        denoise_strength: Optional[float] = None,
    ) -> np.ndarray:
        """Apply the full pre-processing pipeline.
        
        Args:
            image: Input image
            denoise: Apply denoising
            color_correct: Apply color correction
            reduce_artifacts: Apply JPEG artifact reduction
            enhance_contrast: Apply contrast enhancement
            denoise_strength: Override config denoise strength
            
        Returns:
            Pre-processed image
        """
        result = image.copy()
        
        # Step 1: JPEG artifact reduction (before other processing)
        if reduce_artifacts:
            result = self.reduce_jpeg_artifacts(result, strength=0.3)
        
        # Step 2: Denoising
        if denoise:
            strength = denoise_strength or self.config.denoise_strength
            result = self.denoise(result, strength=strength)
        
        # Step 3: Color correction
        if color_correct and self.config.enable_color_correct:
            result = self.auto_white_balance(result)
            result = self.auto_levels(result, clip_percent=0.5)
        
        # Step 4: Contrast enhancement (optional, can be too aggressive)
        if enhance_contrast:
            result = self.enhance_contrast(result, clip_limit=1.5)
        
        logger.debug("Pre-processing pipeline completed")
        return result


class AdaptivePreProcessor(PreProcessor):
    """Adaptive pre-processor that analyzes the image to choose settings."""

    def analyze_image(self, image: np.ndarray) -> dict:
        """Analyze image characteristics.
        
        Args:
            image: Input image
            
        Returns:
            Dictionary of image characteristics
        """
        # Calculate noise level
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        noise_level = laplacian.var()
        
        # Calculate contrast
        contrast = gray.std()
        
        # Estimate JPEG artifacts (look for block patterns)
        # High-frequency energy in 8x8 block boundaries
        h, w = gray.shape
        block_edges = np.abs(np.diff(gray[:, ::8], axis=1)).mean()
        
        # Calculate brightness
        brightness = gray.mean()
        
        return {
            "noise_level": noise_level,
            "contrast": contrast,
            "block_artifacts": block_edges,
            "brightness": brightness,
            "is_noisy": noise_level < 100,  # Low variance = noisy (paradoxically)
            "is_low_contrast": contrast < 50,
            "has_jpeg_artifacts": block_edges > 5,
            "is_dark": brightness < 100,
            "is_bright": brightness > 200,
        }

    def auto_process(self, image: np.ndarray) -> np.ndarray:
        """Automatically process image based on analysis.
        
        Args:
            image: Input image
            
        Returns:
            Processed image
        """
        analysis = self.analyze_image(image)
        
        logger.debug(f"Image analysis: {analysis}")
        
        # Determine processing settings
        denoise_strength = 0.0
        if analysis["is_noisy"]:
            denoise_strength = 0.5
        
        reduce_artifacts = analysis["has_jpeg_artifacts"]
        enhance_contrast = analysis["is_low_contrast"]
        
        return self.process(
            image,
            denoise=denoise_strength > 0,
            color_correct=True,
            reduce_artifacts=reduce_artifacts,
            enhance_contrast=enhance_contrast,
            denoise_strength=denoise_strength,
        )
