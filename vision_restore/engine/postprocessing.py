"""
Vision-Restore AI - Image Post-Processing

Final optimization pipeline including sharpening, detail enhancement,
and color optimization.
"""

from typing import Optional, Tuple
import numpy as np
import cv2

from vision_restore.core.config import Config, get_config
from vision_restore.core.logger import get_logger

logger = get_logger(__name__)


class PostProcessor:
    """Post-processing for final image optimization.
    
    Techniques:
    - Adaptive unsharp mask sharpening
    - Detail enhancement (high-frequency boost)
    - Color vibrance enhancement
    - Final noise cleanup
    """

    def __init__(self, config: Optional[Config] = None):
        """Initialize the post-processor.
        
        Args:
            config: Application configuration
        """
        self.config = config or get_config()

    def sharpen(
        self,
        image: np.ndarray,
        amount: float = 0.5,
        radius: float = 1.0,
        threshold: int = 0,
    ) -> np.ndarray:
        """Apply adaptive unsharp mask sharpening.
        
        This method sharpens edges while minimizing halo artifacts.
        
        Args:
            image: Input image (BGR, uint8)
            amount: Sharpening amount (0.0 to 2.0)
            radius: Blur radius for unsharp mask
            threshold: Minimum difference threshold
            
        Returns:
            Sharpened image
        """
        if amount <= 0:
            return image
        
        # Clip amount to reasonable range
        amount = min(2.0, max(0.0, amount))
        
        # Calculate sigma for Gaussian blur
        sigma = radius * 2
        
        # Create blurred version
        blurred = cv2.GaussianBlur(image, (0, 0), sigma)
        
        # Calculate unsharp mask
        if threshold > 0:
            # Apply threshold to reduce sharpening in smooth areas
            diff = cv2.absdiff(image, blurred)
            mask = (diff > threshold).astype(np.float32)
            if len(mask.shape) == 3:
                mask = mask.max(axis=2)
            mask = cv2.GaussianBlur(mask, (3, 3), 0)
            
            if len(image.shape) == 3:
                mask = np.stack([mask] * 3, axis=2)
            
            sharpened = cv2.addWeighted(
                image.astype(np.float32), 1.0 + amount,
                blurred.astype(np.float32), -amount,
                0,
            )
            result = image.astype(np.float32) * (1 - mask) + sharpened * mask
        else:
            # Simple unsharp mask
            result = cv2.addWeighted(
                image.astype(np.float32), 1.0 + amount,
                blurred.astype(np.float32), -amount,
                0,
            )
        
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        logger.debug(f"Applied sharpening (amount={amount:.2f}, radius={radius:.1f})")
        return result

    def edge_aware_sharpen(
        self,
        image: np.ndarray,
        amount: float = 0.5,
    ) -> np.ndarray:
        """Apply edge-aware sharpening that avoids halos.
        
        Uses edge detection to limit sharpening to actual edges.
        
        Args:
            image: Input image
            amount: Sharpening amount (0.0 to 1.0)
            
        Returns:
            Sharpened image
        """
        if amount <= 0:
            return image
        
        # Convert to grayscale for edge detection
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Detect edges
        edges = cv2.Canny(gray, 50, 150)
        edge_mask = cv2.dilate(edges, None, iterations=1)
        edge_mask = cv2.GaussianBlur(edge_mask, (5, 5), 0)
        edge_mask = edge_mask.astype(np.float32) / 255.0
        
        # Apply strong sharpening
        sharpened = self.sharpen(image, amount=amount * 1.5, radius=0.5)
        
        # Blend based on edge mask
        if len(image.shape) == 3:
            edge_mask = np.stack([edge_mask] * 3, axis=2)
        
        result = image.astype(np.float32) * (1 - edge_mask) + \
                 sharpened.astype(np.float32) * edge_mask
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        logger.debug(f"Applied edge-aware sharpening (amount={amount:.2f})")
        return result

    def enhance_details(
        self,
        image: np.ndarray,
        amount: float = 0.3,
    ) -> np.ndarray:
        """Enhance fine details using high-frequency boost.
        
        Args:
            image: Input image
            amount: Enhancement amount (0.0 to 1.0)
            
        Returns:
            Detail-enhanced image
        """
        if amount <= 0:
            return image
        
        # Create a high-pass filter
        blurred = cv2.GaussianBlur(image, (0, 0), 3)
        high_freq = cv2.subtract(image, blurred)
        
        # Add boosted high frequencies back
        result = cv2.addWeighted(
            image, 1.0,
            high_freq, amount,
            0,
        )
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        logger.debug(f"Enhanced details (amount={amount:.2f})")
        return result

    def enhance_vibrance(
        self,
        image: np.ndarray,
        amount: float = 0.2,
    ) -> np.ndarray:
        """Enhance color vibrance selectively.
        
        Increases saturation of less-saturated colors while
        protecting already-saturated colors.
        
        Args:
            image: Input image (BGR, uint8)
            amount: Vibrance amount (-1.0 to 1.0)
            
        Returns:
            Vibrance-enhanced image
        """
        if abs(amount) < 0.01 or len(image.shape) != 3:
            return image
        
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        
        # Get saturation channel
        saturation = hsv[:, :, 1]
        
        # Calculate adjustment factor (less saturated = more boost)
        # Normalized saturation (0-1)
        norm_sat = saturation / 255.0
        
        # Inverse relationship: boost low saturation more
        adjustment = (1.0 - norm_sat) * amount * 255.0
        
        # Apply adjustment
        hsv[:, :, 1] = np.clip(saturation + adjustment, 0, 255)
        
        # Convert back to BGR
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        logger.debug(f"Enhanced vibrance (amount={amount:.2f})")
        return result

    def final_denoise(
        self,
        image: np.ndarray,
        strength: float = 0.2,
    ) -> np.ndarray:
        """Apply light final denoising pass.
        
        Very mild denoising to clean up any artifacts from processing.
        
        Args:
            image: Input image
            strength: Denoise strength (0.0 to 1.0)
            
        Returns:
            Cleaned image
        """
        if strength <= 0:
            return image
        
        # Use bilateral filter for edge-preserving smoothing
        d = 5
        sigma = 20 + strength * 30
        
        filtered = cv2.bilateralFilter(image, d, sigma, sigma)
        
        # Gentle blend
        result = cv2.addWeighted(image, 1.0 - strength * 0.5, filtered, strength * 0.5, 0)
        
        logger.debug(f"Applied final denoise (strength={strength:.2f})")
        return result

    def process(
        self,
        image: np.ndarray,
        sharpen: bool = True,
        enhance_details: bool = True,
        enhance_vibrance: bool = False,
        final_denoise: bool = False,
        sharpen_amount: Optional[float] = None,
    ) -> np.ndarray:
        """Apply the full post-processing pipeline.
        
        Args:
            image: Input image
            sharpen: Apply sharpening
            enhance_details: Apply detail enhancement
            enhance_vibrance: Apply vibrance enhancement
            final_denoise: Apply final denoising
            sharpen_amount: Override config sharpen amount
            
        Returns:
            Post-processed image
        """
        result = image.copy()
        
        # Step 1: Sharpening
        if sharpen and self.config.enable_sharpen:
            amount = sharpen_amount or self.config.sharpen_amount
            result = self.edge_aware_sharpen(result, amount=amount)
        
        # Step 2: Detail enhancement
        if enhance_details:
            result = self.enhance_details(result, amount=0.2)
        
        # Step 3: Vibrance
        if enhance_vibrance:
            result = self.enhance_vibrance(result, amount=0.15)
        
        # Step 4: Final cleanup
        if final_denoise:
            result = self.final_denoise(result, strength=0.2)
        
        logger.debug("Post-processing pipeline completed")
        return result


class AdaptivePostProcessor(PostProcessor):
    """Adaptive post-processor that adjusts settings based on image analysis."""

    def analyze_sharpness(self, image: np.ndarray) -> float:
        """Analyze image sharpness.
        
        Args:
            image: Input image
            
        Returns:
            Sharpness score (higher = sharper)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return laplacian.var()

    def auto_process(self, image: np.ndarray) -> np.ndarray:
        """Automatically process image based on analysis.
        
        Args:
            image: Input image
            
        Returns:
            Processed image
        """
        sharpness = self.analyze_sharpness(image)
        
        # Adjust sharpening based on current sharpness
        if sharpness > 1000:
            # Image is already sharp, minimal sharpening
            sharpen_amount = 0.1
        elif sharpness > 500:
            sharpen_amount = 0.2
        else:
            # Image is soft, more sharpening
            sharpen_amount = 0.4
        
        logger.debug(f"Auto postprocess: sharpness={sharpness:.0f}, sharpen_amount={sharpen_amount:.2f}")
        
        return self.process(
            image,
            sharpen=True,
            enhance_details=True,
            enhance_vibrance=False,
            final_denoise=False,
            sharpen_amount=sharpen_amount,
        )
