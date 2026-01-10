"""
Vision-Restore AI - Image Quality Metrics

No-reference and full-reference image quality assessment for:
- Intelligent model selection
- Quality improvement verification
- Processing feedback
"""

from typing import Optional, Dict, Any, Tuple
import numpy as np
import cv2

from vision_restore.core.logger import get_logger

logger = get_logger(__name__)


class QualityMetrics:
    """Image quality assessment metrics.
    
    Provides various image quality measurements:
    - BRISQUE (no-reference, requires scipy)
    - Laplacian variance (sharpness)
    - Noise estimation
    - Color analysis
    - Detail measurement
    """

    def __init__(self):
        """Initialize the quality metrics calculator."""
        self._brisque_available = self._check_brisque_availability()

    def _check_brisque_availability(self) -> bool:
        """Check if BRISQUE dependencies are available."""
        try:
            from scipy import ndimage
            from scipy.special import gamma
            return True
        except ImportError:
            return False

    # =========================================================================
    # Sharpness Metrics
    # =========================================================================

    def laplacian_variance(self, image: np.ndarray) -> float:
        """Calculate Laplacian variance as sharpness metric.
        
        Higher values indicate sharper images.
        
        Args:
            image: Input image (BGR or grayscale)
            
        Returns:
            Laplacian variance (higher = sharper)
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return float(np.var(laplacian))

    def tenengrad_variance(self, image: np.ndarray) -> float:
        """Calculate Tenengrad variance for sharpness.
        
        Uses Sobel gradients for focus measurement.
        
        Args:
            image: Input image
            
        Returns:
            Tenengrad variance (higher = sharper)
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        fm = gx**2 + gy**2
        return float(np.mean(fm))

    def sharpness_score(self, image: np.ndarray) -> float:
        """Combined sharpness score (0-100).
        
        Args:
            image: Input image
            
        Returns:
            Sharpness score (0-100, higher = sharper)
        """
        lap_var = self.laplacian_variance(image)
        
        # Normalize to 0-100 range (empirical thresholds)
        # < 100: very blurry, > 1000: very sharp
        score = min(100, max(0, (lap_var - 50) / 10))
        return score

    # =========================================================================
    # Noise Estimation
    # =========================================================================

    def estimate_noise_sigma(self, image: np.ndarray) -> float:
        """Estimate image noise level using Laplacian method.
        
        Based on Immerkær's method for noise estimation.
        
        Args:
            image: Input image
            
        Returns:
            Estimated noise standard deviation
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        gray = gray.astype(np.float64)
        h, w = gray.shape
        
        # Laplacian kernel
        M = np.array([
            [1, -2, 1],
            [-2, 4, -2],
            [1, -2, 1]
        ])
        
        # Apply kernel and calculate sigma
        sigma = np.sum(np.abs(cv2.filter2D(gray, -1, M)))
        sigma = sigma * np.sqrt(0.5 * np.pi) / (6 * (w - 2) * (h - 2))
        
        return sigma

    def noise_level(self, image: np.ndarray) -> str:
        """Categorize noise level.
        
        Args:
            image: Input image
            
        Returns:
            Noise category: 'low', 'medium', 'high'
        """
        sigma = self.estimate_noise_sigma(image)
        
        if sigma < 3:
            return "low"
        elif sigma < 10:
            return "medium"
        else:
            return "high"

    # =========================================================================
    # JPEG Artifact Detection
    # =========================================================================

    def detect_jpeg_quality(self, image: np.ndarray) -> int:
        """Estimate original JPEG quality from image characteristics.
        
        Args:
            image: Input image
            
        Returns:
            Estimated JPEG quality (1-100)
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Block artifact detection using 8x8 DCT blocks
        h, w = gray.shape
        block_size = 8
        
        # Calculate horizontal and vertical gradients at block boundaries
        h_blocks = h // block_size
        w_blocks = w // block_size
        
        if h_blocks < 2 or w_blocks < 2:
            return 95  # Too small to estimate
        
        # Measure blockiness at 8x8 boundaries
        boundary_diff = 0
        internal_diff = 0
        count_boundary = 0
        count_internal = 0
        
        for i in range(1, h_blocks):
            y = i * block_size
            for x in range(w):
                boundary_diff += abs(float(gray[y, x]) - float(gray[y-1, x]))
                count_boundary += 1
            for x in range(w):
                if (y + 1) < h:
                    internal_diff += abs(float(gray[y+1, x]) - float(gray[y, x]))
                    count_internal += 1
        
        if count_boundary == 0 or count_internal == 0:
            return 95
        
        boundary_avg = boundary_diff / count_boundary
        internal_avg = internal_diff / count_internal
        
        # Ratio indicates blockiness (higher = more blocky)
        if internal_avg > 0:
            block_ratio = boundary_avg / internal_avg
        else:
            block_ratio = 1.0
        
        # Convert to estimated quality (empirical mapping)
        # block_ratio close to 1 = high quality, > 1.5 = low quality
        quality = min(100, max(1, int(100 - (block_ratio - 1) * 50)))
        
        return quality

    def has_jpeg_artifacts(self, image: np.ndarray) -> bool:
        """Check if image has significant JPEG artifacts.
        
        Args:
            image: Input image
            
        Returns:
            True if significant artifacts detected
        """
        quality = self.detect_jpeg_quality(image)
        return quality < 70

    # =========================================================================
    # Color Analysis
    # =========================================================================

    def calculate_histogram(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate color histogram for each channel.
        
        Args:
            image: Input image (BGR)
            
        Returns:
            Dictionary with 'b', 'g', 'r' histograms
        """
        histograms = {}
        
        for i, color in enumerate(['b', 'g', 'r']):
            hist = cv2.calcHist([image], [i], None, [256], [0, 256])
            histograms[color] = hist.flatten()
        
        return histograms

    def detect_color_cast(self, image: np.ndarray) -> Optional[str]:
        """Detect if image has a color cast.
        
        Args:
            image: Input image (BGR)
            
        Returns:
            Color cast type or None
        """
        b, g, r = cv2.split(image.astype(np.float32))
        
        b_mean, g_mean, r_mean = np.mean(b), np.mean(g), np.mean(r)
        overall_mean = (b_mean + g_mean + r_mean) / 3
        
        # Calculate deviation from gray
        threshold = 15  # Threshold for color cast detection
        
        if r_mean - overall_mean > threshold:
            if g_mean - overall_mean > threshold * 0.5:
                return "yellow"
            return "red"
        elif b_mean - overall_mean > threshold:
            if g_mean - overall_mean > threshold * 0.5:
                return "cyan"
            return "blue"
        elif g_mean - overall_mean > threshold:
            return "green"
        elif r_mean - overall_mean < -threshold and g_mean - overall_mean < -threshold:
            return "blue"
        elif b_mean - overall_mean < -threshold and g_mean - overall_mean < -threshold:
            return "red"
        
        return None

    def color_variety_score(self, image: np.ndarray) -> float:
        """Calculate color variety/richness score.
        
        Args:
            image: Input image
            
        Returns:
            Color variety score (0-100)
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # Calculate saturation and hue distribution
        sat_mean = np.mean(s) / 255.0 * 100
        sat_std = np.std(s) / 255.0 * 100
        
        # Hue distribution (circular mean is complex, use std as proxy)
        hue_std = np.std(h)
        
        # Combined score
        score = (sat_mean * 0.4 + sat_std * 0.3 + min(hue_std / 180 * 100, 100) * 0.3)
        
        return min(100, max(0, score))

    # =========================================================================
    # Content Analysis
    # =========================================================================

    def detect_faces(self, image: np.ndarray) -> int:
        """Detect number of faces in image.
        
        Args:
            image: Input image
            
        Returns:
            Number of detected faces
        """
        try:
            # Use OpenCV's Haar cascade for basic face detection
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            face_cascade = cv2.CascadeClassifier(cascade_path)
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
            
            return len(faces)
        except Exception:
            return 0

    def is_anime_style(self, image: np.ndarray) -> bool:
        """Heuristic detection if image appears to be anime/cartoon style.
        
        Args:
            image: Input image
            
        Returns:
            True if image appears to be anime-style
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Anime typically has: strong edges, flat color regions, limited color palette
        
        # Edge detection
        edges = cv2.Canny(gray, 100, 200)
        edge_ratio = np.count_nonzero(edges) / edges.size
        
        # Color quantization check (anime has fewer unique colors)
        h, w = image.shape[:2]
        small = cv2.resize(image, (min(200, w), min(200, h)))
        unique_colors = len(np.unique(small.reshape(-1, 3), axis=0))
        color_ratio = unique_colors / (small.shape[0] * small.shape[1])
        
        # Heuristics: high edges + low unique colors = likely anime
        return edge_ratio > 0.03 and color_ratio < 0.1

    def analyze_detail_level(self, image: np.ndarray) -> str:
        """Analyze detail level of image.
        
        Args:
            image: Input image
            
        Returns:
            Detail level: 'low', 'medium', 'high'
        """
        sharpness = self.laplacian_variance(image)
        
        if sharpness < 200:
            return "low"
        elif sharpness < 1000:
            return "medium"
        else:
            return "high"

    # =========================================================================
    # BRISQUE Score
    # =========================================================================

    def brisque_score(self, image: np.ndarray) -> Optional[float]:
        """Calculate BRISQUE no-reference quality score.
        
        Lower scores indicate better quality.
        
        Args:
            image: Input image
            
        Returns:
            BRISQUE score or None if not available
        """
        if not self._brisque_available:
            logger.debug("BRISQUE not available (scipy not installed)")
            return None
        
        try:
            from scipy import ndimage
            from scipy.special import gamma
            
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            gray = gray.astype(np.float64)
            
            # Calculate MSCN (Mean Subtracted Contrast Normalized) coefficients
            mu = cv2.GaussianBlur(gray, (7, 7), 1.166)
            mu_sq = mu * mu
            sigma = np.sqrt(abs(cv2.GaussianBlur(gray * gray, (7, 7), 1.166) - mu_sq))
            
            mscn = (gray - mu) / (sigma + 1.0)
            
            # Fit to generalized Gaussian distribution
            # Simplified: use statistics as proxy for BRISQUE features
            mscn_flat = mscn.flatten()
            
            # Shape parameter estimation (simplified)
            mean_abs = np.mean(np.abs(mscn_flat))
            std = np.std(mscn_flat)
            
            if std > 0:
                shape = mean_abs / std
            else:
                shape = 1.0
            
            # Convert to score (simplified mapping)
            # Real BRISQUE uses SVR model trained on MOS data
            # This is an approximation
            score = 50 - (shape - 1) * 20 + (1 - np.clip(std / 0.5, 0, 1)) * 30
            
            return max(0, min(100, score))
            
        except Exception as e:
            logger.debug(f"BRISQUE calculation failed: {e}")
            return None

    # =========================================================================
    # Comprehensive Analysis
    # =========================================================================

    def analyze(self, image: np.ndarray) -> Dict[str, Any]:
        """Perform comprehensive quality analysis.
        
        Args:
            image: Input image
            
        Returns:
            Dictionary with all quality metrics
        """
        analysis = {
            "sharpness": {
                "laplacian_variance": self.laplacian_variance(image),
                "score": self.sharpness_score(image),
                "level": self.analyze_detail_level(image),
            },
            "noise": {
                "sigma": self.estimate_noise_sigma(image),
                "level": self.noise_level(image),
            },
            "compression": {
                "jpeg_quality_estimate": self.detect_jpeg_quality(image),
                "has_artifacts": self.has_jpeg_artifacts(image),
            },
            "color": {
                "color_cast": self.detect_color_cast(image),
                "variety_score": self.color_variety_score(image),
            },
            "content": {
                "face_count": self.detect_faces(image),
                "is_anime": self.is_anime_style(image),
                "detail_level": self.analyze_detail_level(image),
            },
        }
        
        # Add BRISQUE if available
        brisque = self.brisque_score(image)
        if brisque is not None:
            analysis["brisque"] = brisque
        
        return analysis

    def compare(
        self,
        original: np.ndarray,
        enhanced: np.ndarray,
    ) -> Dict[str, Any]:
        """Compare quality between original and enhanced image.
        
        Args:
            original: Original image
            enhanced: Enhanced image
            
        Returns:
            Comparison metrics
        """
        orig_analysis = self.analyze(original)
        enh_analysis = self.analyze(enhanced)
        
        comparison = {
            "sharpness_improvement": (
                enh_analysis["sharpness"]["score"] - 
                orig_analysis["sharpness"]["score"]
            ),
            "noise_change": (
                orig_analysis["noise"]["sigma"] - 
                enh_analysis["noise"]["sigma"]
            ),
            "color_variety_change": (
                enh_analysis["color"]["variety_score"] - 
                orig_analysis["color"]["variety_score"]
            ),
            "original": orig_analysis,
            "enhanced": enh_analysis,
        }
        
        # PSNR if same dimensions
        if original.shape == enhanced.shape:
            comparison["psnr"] = self._calculate_psnr(original, enhanced)
        
        return comparison

    def _calculate_psnr(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate PSNR between two images."""
        mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
        if mse == 0:
            return float('inf')
        return 20 * np.log10(255.0 / np.sqrt(mse))


def recommend_processing(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Recommend processing settings based on quality analysis.
    
    Args:
        analysis: Quality analysis from QualityMetrics.analyze()
        
    Returns:
        Processing recommendations
    """
    recommendations = {
        "model": "swinir",  # Default to SwinIR (better quality than RealESRGAN)
        "face_enhance": False,
        "denoise": False,
        "denoise_strength": 0.3,
        "color_correct": False,
        "sharpen": True,
        "sharpen_amount": 0.3,
        "preset": "balanced",
    }
    
    # Face enhancement recommendation
    if analysis["content"]["face_count"] > 0:
        recommendations["face_enhance"] = True
        recommendations["preset"] = "portrait"
    
    # Model selection based on content
    if analysis["content"]["is_anime"]:
        recommendations["model"] = "realesrgan-anime"
        recommendations["preset"] = "anime"
    elif analysis["noise"]["level"] == "high":
        recommendations["model"] = "swinir"
        recommendations["denoise"] = True
        recommendations["denoise_strength"] = 0.7
    elif analysis["sharpness"]["level"] == "high" and analysis["noise"]["level"] == "low":
        # HAT is excellent for preserving high-frequency details in clean images
        recommendations["model"] = "hat"
    elif analysis["content"]["detail_level"] == "medium":
        # SwinIR is a good general purpose balance
        recommendations["model"] = "swinir"
    
    # Denoising recommendation
    if analysis["noise"]["level"] != "low":
        recommendations["denoise"] = True
        if analysis["noise"]["level"] == "high":
            recommendations["denoise_strength"] = 0.7
        else:
            recommendations["denoise_strength"] = 0.4
    
    # Color correction recommendation
    if analysis["color"]["color_cast"] is not None:
        recommendations["color_correct"] = True
    
    # Sharpening recommendation (reduce for already sharp images)
    if analysis["sharpness"]["level"] == "high":
        recommendations["sharpen_amount"] = 0.1
    elif analysis["sharpness"]["level"] == "low":
        recommendations["sharpen_amount"] = 0.5
    
    return recommendations
