"""
Vision-Restore AI - Enhancement Orchestrator v2.0

Master pipeline that orchestrates the complete image enhancement workflow.
Coordinates pre-processing, upscaling, face restoration, and post-processing.

v2.0 Features:
- Multi-model support (Real-ESRGAN, SwinIR)
- Multi-face-restorer support (GFPGAN, CodeFormer)
- Intelligent auto-selection based on quality metrics
- New presets (portrait, anime, restore)
"""

from typing import Callable, Optional, Tuple, Dict, Any
import numpy as np

from vision_restore.core.config import Config, get_config
from vision_restore.core.logger import get_logger
from vision_restore.core.model_manager import ModelManager
from vision_restore.engine.preprocessing import PreProcessor, AdaptivePreProcessor
from vision_restore.engine.postprocessing import PostProcessor, AdaptivePostProcessor
from vision_restore.engine.tiling import TilingEngine
from vision_restore.engine.realesrgan_engine import RealESRGANEngine
from vision_restore.engine.gfpgan_engine import GFPGANEngine

logger = get_logger(__name__)

# Lazy imports for new engines (may not be available)
def _get_swinir_engine():
    try:
        from vision_restore.engine.swinir_engine import SwinIREngine
        return SwinIREngine
    except ImportError:
        return None

def _get_codeformer_engine():
    try:
        from vision_restore.engine.codeformer_engine import CodeFormerEngine
        return CodeFormerEngine
    except ImportError:
        return None

def _get_quality_metrics():
    try:
        from vision_restore.engine.quality_metrics import QualityMetrics, recommend_processing
        return QualityMetrics, recommend_processing
    except ImportError:
        return None, None

def _get_hat_engine():
    try:
        from vision_restore.engine.hat_engine import HATEngine
        return HATEngine
    except ImportError:
        return None


class EnhancementOrchestrator:
    """Orchestrates the complete image enhancement pipeline.
    
    Pipeline flow:
    1. Quality Analysis: Analyze image to recommend settings
    2. Pre-processing: Denoise, color correct, artifact removal
    3. Tiling: Split large images into manageable tiles
    4. Upscaling: Real-ESRGAN or SwinIR super-resolution
    5. Face Enhancement: GFPGAN or CodeFormer face restoration
    6. Tile Merging: Seamless stitching
    7. Post-processing: Sharpen, enhance, optimize
    
    v2.0 Features:
    - Multiple upscaling models (realesrgan, swinir, auto)
    - Multiple face restorers (gfpgan, codeformer, auto)
    - Intelligent model selection based on content analysis
    - Quality metrics for before/after comparison
    """

    # Available upscale models
    UPSCALE_MODELS = ["auto", "realesrgan", "swinir", "hat"]
    
    # Available face restoration models
    FACE_MODELS = ["auto", "gfpgan", "codeformer"]
    
    # Enhanced presets for v2.0
    PRESETS_V2 = {
        "fast": {
            "upscale_model": "realesrgan",
            "face_model": "gfpgan",
            "enable_denoise": False,
            "enable_sharpen": False,
            "enable_quality_metrics": False,
        },
        "balanced": {
            "upscale_model": "realesrgan",
            "face_model": "gfpgan",
            "enable_denoise": True,
            "enable_sharpen": True,
            "denoise_strength": 0.3,
            "sharpen_amount": 0.2,
        },
        "quality": {
            "upscale_model": "auto",
            "face_model": "auto",
            "enable_denoise": True,
            "enable_sharpen": True,
            "denoise_strength": 0.5,
            "sharpen_amount": 0.4,
            "enable_quality_metrics": True,
        },
        "portrait": {
            "upscale_model": "realesrgan",
            "face_model": "codeformer",
            "codeformer_fidelity": 0.7,
            "enable_denoise": True,
            "enable_sharpen": True,
            "enable_face_enhance": True,
        },
        "anime": {
            "upscale_model": "realesrgan",
            "realesrgan_model": "realesrgan-x4plus-anime",
            "face_model": "none",
            "enable_face_enhance": False,
            "enable_sharpen": True,
            "sharpen_amount": 0.3,
        },
        "restore": {
            "upscale_model": "swinir",
            "swinir_model": "swinir-real-x4",
            "face_model": "codeformer",
            "codeformer_fidelity": 0.5,
            "enable_denoise": True,
            "denoise_strength": 0.7,
            "enable_color_correct": True,
            "enable_quality_metrics": True,
        },
    }

    def __init__(self, config: Optional[Config] = None):
        """Initialize the enhancement orchestrator.
        
        Args:
            config: Application configuration
        """
        self.config = config or get_config()
        
        # v2.0 settings: Removed static attributes in favor of properties
        # self.upscale_model = getattr(self.config, 'upscale_model', 'auto')
        # self.face_model = getattr(self.config, 'face_model', 'auto')
        # self.codeformer_fidelity = getattr(self.config, 'codeformer_fidelity', 0.7)
        # self.enable_quality_metrics = getattr(self.config, 'enable_quality_metrics', True)
        
        # Initialize components (lazy loading for engines)
        self.model_manager = ModelManager(self.config)
        self.preprocessor = AdaptivePreProcessor(self.config)
        self.postprocessor = AdaptivePostProcessor(self.config)
        self.tiling = TilingEngine(
            tile_size=self.config.tile_size,
            tile_pad=self.config.tile_pad,
        )
        
        # Quality metrics (optional)
        QualityMetrics, _ = _get_quality_metrics()
        self._quality_metrics = QualityMetrics() if QualityMetrics else None
        
        # Engines (lazy initialized)
        self._realesrgan: Optional[RealESRGANEngine] = None
        self._swinir = None
        self._gfpgan: Optional[GFPGANEngine] = None
        self._codeformer = None
        self._hat = None
        
        # Last analysis results (for inspection)
        self.last_analysis: Optional[Dict[str, Any]] = None

    @property
    def upscale_model(self):
        """Dynamic property for upscale model from config."""
        return getattr(self.config, 'upscale_model', 'auto')

    @upscale_model.setter
    def upscale_model(self, value):
        self.config.upscale_model = value

    @property
    def face_model(self):
        """Dynamic property for face model from config."""
        return getattr(self.config, 'face_model', 'auto')

    @face_model.setter
    def face_model(self, value):
        self.config.face_model = value

    @property
    def codeformer_fidelity(self):
        """Dynamic property for fidelty from config."""
        return getattr(self.config, 'codeformer_fidelity', 0.7)

    @codeformer_fidelity.setter
    def codeformer_fidelity(self, value):
        self.config.codeformer_fidelity = value

    @property
    def enable_quality_metrics(self):
        """Dynamic property for quality metrics from config."""
        return getattr(self.config, 'enable_quality_metrics', True)

    @property
    def realesrgan(self) -> RealESRGANEngine:
        """Get or create Real-ESRGAN engine."""
        if self._realesrgan is None:
            self._realesrgan = RealESRGANEngine(
                config=self.config,
                scale=self.config.scale,
                tile_size=self.config.tile_size,
                tile_pad=self.config.tile_pad,
                gpu_id=self.config.gpu_id,
            )
        return self._realesrgan

    @property
    def swinir(self):
        """Get or create SwinIR engine."""
        if self._swinir is None:
            SwinIREngine = _get_swinir_engine()
            if SwinIREngine:
                model_name = getattr(self.config, 'swinir_model', 'swinir-real-x4')
                self._swinir = SwinIREngine(
                    config=self.config,
                    model_name=model_name,
                    scale=self.config.scale,
                    gpu_id=self.config.gpu_id,
                )
        return self._swinir

    @property
    def hat(self):
        """Get or create HAT engine."""
        if self._hat is None:
            HATEngine = _get_hat_engine()
            if HATEngine:
                self._hat = HATEngine(
                    config=self.config,
                    scale=self.config.scale,
                    gpu_id=self.config.gpu_id,
                )
        return self._hat

    @property
    def gfpgan(self) -> GFPGANEngine:
        """Get or create GFPGAN engine."""
        if self._gfpgan is None:
            self._gfpgan = GFPGANEngine(
                config=self.config,
                upscale=self.config.scale,
                gpu_id=self.config.gpu_id,
            )
        return self._gfpgan

    @property
    def codeformer(self):
        """Get or create CodeFormer engine."""
        if self._codeformer is None:
            CodeFormerEngine = _get_codeformer_engine()
            if CodeFormerEngine:
                self._codeformer = CodeFormerEngine(
                    config=self.config,
                    fidelity=self.codeformer_fidelity,
                    upscale=self.config.scale,
                    gpu_id=self.config.gpu_id,
                )
        return self._codeformer

    def apply_preset_v2(self, preset: str) -> None:
        """Apply a v2.0 preset to the orchestrator.
        
        Args:
            preset: Preset name (fast, balanced, quality, portrait, anime, restore)
        """
        if preset not in self.PRESETS_V2:
            # Try legacy preset through config
            self.config.apply_preset(preset)
            return
        
        settings = self.PRESETS_V2[preset]
        for key, value in settings.items():
            if hasattr(self, key):
                setattr(self, key, value)
            elif hasattr(self.config, key):
                setattr(self.config, key, value)
        
        logger.info(f"Applied v2.0 preset: {preset}")

    def analyze_image(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze image quality and recommend processing settings.
        
        Args:
            image: Input image
            
        Returns:
            Analysis results with recommendations
        """
        if self._quality_metrics is None:
            return {"available": False}
        
        analysis = self._quality_metrics.analyze(image)
        
        # Get recommendations
        _, recommend_processing = _get_quality_metrics()
        if recommend_processing:
            analysis["recommendations"] = recommend_processing(analysis)
        
        self.last_analysis = analysis
        return analysis

    def _select_upscale_model(self, image: np.ndarray, analysis: Optional[Dict] = None) -> str:
        """Auto-select the best upscaling model based on image analysis.
        
        Args:
            image: Input image
            analysis: Pre-computed quality analysis
            
        Returns:
            Model name to use
        """
        if self.upscale_model != "auto":
            return self.upscale_model
        
        # Analyze if not provided
        if analysis is None:
            analysis = self.analyze_image(image)
        
        # Get recommendation or default
        if "recommendations" in analysis:
            rec_model = analysis["recommendations"].get("model", "realesrgan")
            # Map recommendation to our model names
            if "swinir" in rec_model.lower():
                return "swinir"
            elif "hat" in rec_model.lower():
                return "hat"
            elif "anime" in rec_model.lower():
                return "realesrgan"  # Use anime model variant
        
        # Default to swinir (better quality than realesrgan for general photos)
        return "swinir"

    def _select_face_model(self, image: np.ndarray, analysis: Optional[Dict] = None) -> str:
        """Auto-select the best face restoration model.
        
        Args:
            image: Input image
            analysis: Pre-computed quality analysis
            
        Returns:
            Model name to use ('gfpgan', 'codeformer', or 'none')
        """
        if self.face_model != "auto":
            return self.face_model
        
        # Analyze if not provided
        if analysis is None:
            analysis = self.analyze_image(image)
        
        # Check if faces present
        face_count = analysis.get("content", {}).get("face_count", 0)
        if face_count == 0:
            return "none"
        
        # Check image quality to choose restorer
        noise_level = analysis.get("noise", {}).get("level", "low")
        
        # Use CodeFormer for noisy/degraded images (better reconstruction)
        if noise_level in ["medium", "high"]:
            if self.codeformer is not None:
                return "codeformer"
        
        # Default to GFPGAN (faster, good for clean images)
        return "gfpgan"

    def check_models(self, face_enhance: bool = True) -> Tuple[bool, list]:
        """Check if all required models are available.
        
        Args:
            face_enhance: Whether face enhancement will be used
            
        Returns:
            Tuple of (all_available, missing_models)
        """
        missing = self.model_manager.get_missing_models(face_enhance)
        return len(missing) == 0, missing

    def download_models(
        self,
        face_enhance: bool = True,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> bool:
        """Download all required models.
        
        Args:
            face_enhance: Whether to include face enhancement models
            progress_callback: Progress callback
            
        Returns:
            True if all downloads succeeded
        """
        return self.model_manager.download_all_required(face_enhance, progress_callback)

    def enhance(
        self,
        image: np.ndarray,
        scale: int = 4,
        face_enhance: bool = True,
        preprocess: bool = True,
        postprocess: bool = True,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        upscale_model: Optional[str] = None,
        face_model: Optional[str] = None,
    ) -> np.ndarray:
        """Run the complete enhancement pipeline.
        
        Args:
            image: Input image (BGR, uint8)
            scale: Upscale factor (2, 4, or 8)
            face_enhance: Enable face enhancement
            preprocess: Enable pre-processing
            postprocess: Enable post-processing
            progress_callback: Progress callback (progress: float, message: str)
            upscale_model: Override upscale model selection
            face_model: Override face model selection
            
        Returns:
            Enhanced image
        """
        if progress_callback is None:
            progress_callback = lambda p, m: None
        
        # Use provided models or instance settings
        upscale_model = upscale_model or self.upscale_model
        face_model = face_model or self.face_model
        
        logger.info(f"Starting v2.0 enhancement pipeline (scale={scale}, upscale={upscale_model}, face={face_model})")
        
        h, w = image.shape[:2]
        output_h, output_w = h * scale, w * scale
        
        progress_callback(0.0, "Initializing...")
        
        # ═══════════════════════════════════════════════════════════════════════
        # Step 0: Quality Analysis (if enabled)
        # ═══════════════════════════════════════════════════════════════════════
        analysis = None
        if self.enable_quality_metrics and self._quality_metrics:
            progress_callback(0.02, "Analyzing image quality...")
            analysis = self.analyze_image(image)
            logger.debug(f"Quality analysis: noise={analysis.get('noise', {}).get('level')}, faces={analysis.get('content', {}).get('face_count')}")
        
        # ═══════════════════════════════════════════════════════════════════════
        # Smart Caching Check
        # ═══════════════════════════════════════════════════════════════════════
        # Parameters that require re-running the heavy AI models
        heavy_params = {
            "image_id": id(image), # Simple object identity check
            "scale": scale,
            "upscale_model": upscale_model,
            "face_model": face_model,
            "face_enhance": face_enhance,
            "codeformer_fidelity": self.codeformer_fidelity if face_model == "codeformer" else None,
            "preprocess": preprocess, # Preprocessing happens before upscaling
            "denoise_pre": self.config.enable_denoise if preprocess else None,
        }
        
        # Check if we can reuse cached upscaled image
        cached_upscaled = None
        if hasattr(self, "_cache_params") and self._cache_params == heavy_params:
            if hasattr(self, "_cache_upscaled") and self._cache_upscaled is not None:
                logger.info("⚡ Smart Cache Hit! Skipping heavy AI processing.")
                progress_callback(0.85, "Using cached AI result...")
                cached_upscaled = self._cache_upscaled.copy()
        
        if cached_upscaled is not None:
            upscaled = cached_upscaled
        else:
            # Run the heavy pipeline
            progress_callback(0.05, f"Using {upscale_model} for upscaling")
            
            # ═══════════════════════════════════════════════════════════════════════
            # Step 1: Pre-processing
            # ═══════════════════════════════════════════════════════════════════════
            if preprocess and self.config.enable_denoise:
                progress_callback(0.08, "Pre-processing image...")
                processed = self.preprocessor.auto_process(image)
                logger.debug("Pre-processing completed")
            else:
                processed = image
            
            progress_callback(0.10, "Pre-processing complete")
            
            # ═══════════════════════════════════════════════════════════════════════
            # Step 2: Upscaling with selected model
            # ═══════════════════════════════════════════════════════════════════════
            progress_callback(0.15, f"Initializing {upscale_model}...")
            
            # Select and initialize upscaling engine
            if upscale_model == "swinir" and self.swinir is not None:
                engine = self.swinir
                if not engine.is_initialized:
                    if not engine.initialize():
                        logger.warning("SwinIR initialization failed, falling back to Real-ESRGAN")
                        engine = self.realesrgan
                        if not engine.is_initialized:
                            engine.initialize()
                        if not engine.is_initialized:
                            engine.initialize()
            elif upscale_model == "hat" and self.hat is not None:
                engine = self.hat
                if not engine.is_initialized:
                    if not engine.initialize():
                        logger.warning("HAT initialization failed, falling back to SwinIR/Real-ESRGAN")
                        # Fallback chain: HAT -> SwinIR -> Real-ESRGAN
                        if self.swinir:
                            engine = self.swinir
                            if not engine.initialize():
                                engine = self.realesrgan
                        else:
                            engine = self.realesrgan
                        if not engine.is_initialized:
                            engine.initialize()
            else:
                engine = self.realesrgan
                if not engine.is_initialized:
                    engine.initialize()
                engine.scale = scale
            
            # Create upscaling function
            def upscale_tile(tile: np.ndarray) -> np.ndarray:
                return engine.upscale(tile, outscale=scale)
            
            # Process with tiling
            def upscale_progress(p: float, m: str):
                progress_callback(0.15 + p * 0.55, m)
            
            if self.tiling.should_tile(processed):
                upscaled = self.tiling.process_image(
                    processed,
                    upscale_tile,
                    scale=scale,
                    progress_callback=upscale_progress,
                )
            else:
                progress_callback(0.40, f"Upscaling with {upscale_model}...")
                upscaled = upscale_tile(processed)
            
            progress_callback(0.70, "Upscaling complete")
            logger.debug(f"Upscaled from {w}x{h} to {output_w}x{output_h}")
            
            # ═══════════════════════════════════════════════════════════════════════
            # Step 3: Face enhancement with selected model
            # ═══════════════════════════════════════════════════════════════════════
            if face_enhance and self.config.enable_face_enhance and face_model != "none":
                progress_callback(0.75, "Detecting faces...")
                
                try:
                    # Select face restoration engine
                    if face_model == "codeformer" and self.codeformer is not None:
                        face_engine = self.codeformer
                        face_engine_name = "CodeFormer"
                    else:
                        face_engine = self.gfpgan
                        face_engine_name = "GFPGAN"
                    
                    # Initialize if needed
                    if not face_engine.is_initialized:
                        face_engine.initialize()
                    
                    # Check for faces
                    face_count = face_engine.get_face_count(upscaled)
                    
                    if face_count > 0:
                        progress_callback(0.80, f"Restoring {face_count} face(s) with {face_engine_name}...")
                        upscaled, _ = face_engine.restore_faces(upscaled)
                        logger.debug(f"Enhanced {face_count} face(s) with {face_engine_name}")
                    else:
                        logger.debug("No faces detected, skipping face enhancement")
                        
                except Exception as e:
                    logger.warning(f"Face enhancement failed: {e}")
            
            progress_callback(0.85, "Face enhancement complete")
            
            # Update Cache
            self._cache_params = heavy_params
            self._cache_upscaled = upscaled.copy()
            logger.info("⚡ Cached new AI result")
        
        # ═══════════════════════════════════════════════════════════════════════
        # Step 4: Post-processing
        # ═══════════════════════════════════════════════════════════════════════
        if postprocess and self.config.enable_sharpen:
            progress_callback(0.90, "Applying final optimizations...")
            if self.config.enable_quality_metrics:
                result = self.postprocessor.auto_process(upscaled)
            else:
                result = self.postprocessor.process(upscaled)
            logger.debug("Post-processing completed")
        else:
            result = upscaled
        
        progress_callback(0.95, "Post-processing complete")
        
        # ═══════════════════════════════════════════════════════════════════════
        # Complete
        # ═══════════════════════════════════════════════════════════════════════
        progress_callback(1.0, "Enhancement complete!")
        logger.info(f"Enhancement completed: {w}x{h} -> {output_w}x{output_h}")
        
        return result

    def enhance_quick(
        self,
        image: np.ndarray,
        scale: int = 4,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> np.ndarray:
        """Quick enhancement without face restoration.
        
        Faster processing for images without faces or when speed is priority.
        
        Args:
            image: Input image
            scale: Upscale factor
            progress_callback: Progress callback
            
        Returns:
            Enhanced image
        """
        return self.enhance(
            image,
            scale=scale,
            face_enhance=False,
            preprocess=False,
            postprocess=True,
            progress_callback=progress_callback,
            upscale_model="realesrgan",  # Fastest
        )

    def enhance_maximum(
        self,
        image: np.ndarray,
        scale: int = 4,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> np.ndarray:
        """Maximum quality enhancement with all features enabled.
        
        Slowest but highest quality output.
        
        Args:
            image: Input image
            scale: Upscale factor
            progress_callback: Progress callback
            
        Returns:
            Enhanced image
        """
        # Enable all processing
        original_denoise = self.config.enable_denoise
        original_sharpen = self.config.enable_sharpen
        original_face = self.config.enable_face_enhance
        
        self.config.enable_denoise = True
        self.config.enable_sharpen = True
        self.config.enable_face_enhance = True
        self.config.denoise_strength = 0.6
        self.config.sharpen_amount = 0.5
        
        try:
            result = self.enhance(
                image,
                scale=scale,
                face_enhance=True,
                preprocess=True,
                postprocess=True,
                progress_callback=progress_callback,
                upscale_model="auto",
                face_model="auto",
            )
        finally:
            # Restore settings
            self.config.enable_denoise = original_denoise
            self.config.enable_sharpen = original_sharpen
            self.config.enable_face_enhance = original_face
        
        return result

    def enhance_portrait(
        self,
        image: np.ndarray,
        scale: int = 4,
        fidelity: float = 0.7,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> np.ndarray:
        """Enhanced portrait processing with CodeFormer.
        
        Optimized for photos with faces.
        
        Args:
            image: Input image
            scale: Upscale factor
            fidelity: CodeFormer fidelity (0=quality, 1=fidelity)
            progress_callback: Progress callback
            
        Returns:
            Enhanced image
        """
        original_fidelity = self.codeformer_fidelity
        self.codeformer_fidelity = fidelity
        
        try:
            return self.enhance(
                image,
                scale=scale,
                face_enhance=True,
                preprocess=True,
                postprocess=True,
                progress_callback=progress_callback,
                upscale_model="realesrgan",
                face_model="codeformer",
            )
        finally:
            self.codeformer_fidelity = original_fidelity

    def enhance_restore(
        self,
        image: np.ndarray,
        scale: int = 4,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> np.ndarray:
        """Photo restoration mode using SwinIR + CodeFormer.
        
        Best for old, damaged, or heavily degraded photos.
        
        Args:
            image: Input image
            scale: Upscale factor
            progress_callback: Progress callback
            
        Returns:
            Enhanced image
        """
        # Apply restoration preset
        self.apply_preset_v2("restore")
        
        return self.enhance(
            image,
            scale=scale,
            face_enhance=True,
            preprocess=True,
            postprocess=True,
            progress_callback=progress_callback,
        )

    def get_quality_comparison(
        self,
        original: np.ndarray,
        enhanced: np.ndarray,
    ) -> Optional[Dict[str, Any]]:
        """Get quality comparison between original and enhanced image.
        
        Args:
            original: Original image
            enhanced: Enhanced image
            
        Returns:
            Comparison metrics or None if quality metrics unavailable
        """
        if self._quality_metrics is None:
            return None
        
        return self._quality_metrics.compare(original, enhanced)

    def get_output_size(
        self,
        input_size: Tuple[int, int],
        scale: int = 4,
    ) -> Tuple[int, int]:
        """Calculate output size for given input.
        
        Args:
            input_size: Input (width, height)
            scale: Upscale factor
            
        Returns:
            Output (width, height)
        """
        return (input_size[0] * scale, input_size[1] * scale)

    def estimate_memory(
        self,
        input_size: Tuple[int, int],
        scale: int = 4,
    ) -> int:
        """Estimate memory required for processing.
        
        Args:
            input_size: Input (width, height)
            scale: Upscale factor
            
        Returns:
            Estimated memory in megabytes
        """
        w, h = input_size
        bytes_per_pixel = 4 * 3  # float32 RGB
        
        # Input image
        input_mem = w * h * bytes_per_pixel
        
        # Output image
        output_mem = (w * scale) * (h * scale) * bytes_per_pixel
        
        # Intermediate buffers (estimate 3x for processing)
        buffer_mem = (input_mem + output_mem) * 3
        
        total_bytes = input_mem + output_mem + buffer_mem
        return total_bytes // (1024 * 1024)

    def cleanup(self) -> None:
        """Release all resources."""
        if self._realesrgan is not None:
            self._realesrgan.cleanup()
            self._realesrgan = None
        
        if self._swinir is not None:
            self._swinir.cleanup()
            self._swinir = None
        
        if self._gfpgan is not None:
            self._gfpgan.cleanup()
            self._gfpgan = None
        
        if self._codeformer is not None:
            self._codeformer.cleanup()
            self._codeformer = None

        if self._hat is not None:
            self._hat.cleanup()
            self._hat = None
        
        logger.debug("Orchestrator resources released")


def create_orchestrator(preset: str = "balanced") -> EnhancementOrchestrator:
    """Create an orchestrator with a quality preset.
    
    Args:
        preset: Quality preset ('fast', 'balanced', 'quality', 'portrait', 'anime', 'restore')
        
    Returns:
        Configured orchestrator
    """
    config = get_config()
    
    # Try legacy preset first
    if preset in ["fast", "balanced", "quality"]:
        config.apply_preset(preset)
    
    orchestrator = EnhancementOrchestrator(config)
    
    # Apply v2 preset if applicable
    if preset in EnhancementOrchestrator.PRESETS_V2:
        orchestrator.apply_preset_v2(preset)
    
    return orchestrator


def get_available_models() -> Dict[str, list]:
    """Get list of available models.
    
    Returns:
        Dictionary with upscale_models and face_models lists
    """
    upscale = ["realesrgan"]
    face = ["gfpgan"]
    
    # Check for optional models
    if _get_swinir_engine():
        upscale.append("swinir")
    
    if _get_codeformer_engine():
        face.append("codeformer")
    
    return {
        "upscale_models": upscale,
        "face_models": face,
    }
