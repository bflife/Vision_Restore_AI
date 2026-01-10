"""
Vision-Restore AI - Main Entry Point v2.0

Command-line interface for image upscaling and restoration.

v2.0 Features:
- Multi-model support (--model realesrgan/swinir/auto)
- Multi-face-restorer (--face-model gfpgan/codeformer/auto)
- CodeFormer fidelity control (--fidelity)
- New presets (--preset portrait/anime/restore)
- Quality analysis mode (--analyze)
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from vision_restore import __version__
from vision_restore.core.config import Config
from vision_restore.core.logger import get_logger

logger = get_logger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="vision-restore",
        description="Vision-Restore AI v2.0 - Professional Image Upscaling & Restoration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  vision-restore --input photo.jpg --output photo_4x.png
  vision-restore --input ./photos/ --output ./upscaled/ --scale 4
  vision-restore --input portrait.jpg --face-enhance --face-model codeformer --fidelity 0.7
  vision-restore --input old_photo.jpg --preset restore --output restored.png
  vision-restore --input photo.jpg --analyze  # Analyze quality without processing
        """,
    )

    parser.add_argument(
        "--version", "-v", action="version", version=f"Vision-Restore AI v{__version__}"
    )

    parser.add_argument(
        "--input", "-i", type=str, required=True, help="Input image or directory path"
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output path (default: input_path_upscaled)",
    )

    parser.add_argument(
        "--scale",
        "-s",
        type=int,
        choices=[2, 4, 8],
        default=4,
        help="Upscale factor (default: 4)",
    )

    # v2.0: Model selection
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        choices=["auto", "realesrgan", "swinir"],
        default="auto",
        help="Upscaling model (default: auto)",
    )

    parser.add_argument(
        "--face-enhance",
        "-f",
        action="store_true",
        help="Enable face enhancement",
    )

    # v2.0: Face model selection
    parser.add_argument(
        "--face-model",
        type=str,
        choices=["auto", "gfpgan", "codeformer", "none"],
        default="auto",
        help="Face restoration model (default: auto)",
    )

    # v2.0: CodeFormer fidelity
    parser.add_argument(
        "--fidelity",
        type=float,
        default=0.7,
        help="CodeFormer fidelity (0=quality, 1=fidelity, default: 0.7)",
    )

    # v2.0: Preset mode
    parser.add_argument(
        "--preset",
        "-p",
        type=str,
        choices=["fast", "balanced", "quality", "portrait", "anime", "restore"],
        default=None,
        help="Quality preset (overrides other settings)",
    )

    parser.add_argument(
        "--denoise",
        "-d",
        action="store_true",
        help="Apply denoising before upscaling",
    )

    parser.add_argument(
        "--tile-size",
        "-t",
        type=int,
        default=512,
        help="Tile size for processing (default: 512)",
    )

    parser.add_argument(
        "--gpu-id",
        "-g",
        type=int,
        default=0,
        help="GPU device ID, -1 for CPU (default: 0)",
    )

    parser.add_argument(
        "--format",
        type=str,
        choices=["png", "jpg", "webp"],
        default="png",
        help="Output format (default: png)",
    )

    parser.add_argument(
        "--quality",
        "-q",
        type=int,
        default=95,
        help="Output quality for JPEG/WebP (default: 95)",
    )

    # v2.0: Analyze mode
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Analyze image quality without processing",
    )

    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    return parser.parse_args()


def analyze_image(input_path: Path) -> int:
    """Analyze image quality and print report."""
    try:
        from vision_restore.engine.quality_metrics import QualityMetrics
        from vision_restore.utils.image_io import load_image
    except ImportError:
        logger.error("Quality metrics not available. Install scipy for full analysis.")
        return 1
    
    image = load_image(input_path)
    if image is None:
        logger.error(f"Failed to load image: {input_path}")
        return 1
    
    metrics = QualityMetrics()
    analysis = metrics.analyze(image)
    
    print(f"\n{'='*60}")
    print(f" Image Quality Analysis: {input_path.name}")
    print(f"{'='*60}\n")
    
    print(f"📊 Sharpness:")
    print(f"   Score: {analysis['sharpness']['score']:.1f}/100")
    print(f"   Level: {analysis['sharpness']['level']}")
    
    print(f"\n🔊 Noise:")
    print(f"   Sigma: {analysis['noise']['sigma']:.2f}")
    print(f"   Level: {analysis['noise']['level']}")
    
    print(f"\n📦 Compression:")
    print(f"   Est. JPEG Quality: {analysis['compression']['jpeg_quality_estimate']}")
    print(f"   Has Artifacts: {'Yes' if analysis['compression']['has_artifacts'] else 'No'}")
    
    print(f"\n🎨 Color:")
    print(f"   Color Cast: {analysis['color']['color_cast'] or 'None'}")
    print(f"   Variety Score: {analysis['color']['variety_score']:.1f}/100")
    
    print(f"\n📸 Content:")
    print(f"   Faces Detected: {analysis['content']['face_count']}")
    print(f"   Is Anime Style: {'Yes' if analysis['content']['is_anime'] else 'No'}")
    
    if "brisque" in analysis:
        print(f"\n🏆 BRISQUE Score: {analysis['brisque']:.1f} (lower is better)")
    
    # Recommendations
    if "recommendations" in analysis:
        rec = analysis["recommendations"]
        print(f"\n💡 Recommendations:")
        print(f"   Model: {rec.get('model', 'realesrgan')}")
        print(f"   Face Enhance: {'Yes' if rec.get('face_enhance') else 'No'}")
        print(f"   Denoise: {'Yes' if rec.get('denoise') else 'No'} (strength: {rec.get('denoise_strength', 0.3):.1f})")
        print(f"   Preset: {rec.get('preset', 'balanced')}")
    
    print(f"\n{'='*60}\n")
    
    return 0


def process_single_image(
    input_path: Path,
    output_path: Path,
    config: Config,
    face_enhance: bool = False,
    upscale_model: str = "auto",
    face_model: str = "auto",
) -> bool:
    """Process a single image through the enhancement pipeline."""
    from vision_restore.engine.orchestrator import EnhancementOrchestrator
    from vision_restore.utils.image_io import load_image, save_image

    try:
        logger.info(f"Processing: {input_path}")

        # Load image
        image = load_image(input_path)
        if image is None:
            logger.error(f"Failed to load image: {input_path}")
            return False

        # Create orchestrator and process
        orchestrator = EnhancementOrchestrator(config)
        orchestrator.upscale_model = upscale_model
        orchestrator.face_model = face_model
        orchestrator.codeformer_fidelity = config.codeformer_fidelity

        def progress_callback(progress: float, message: str):
            logger.info(f"  [{progress*100:.0f}%] {message}")

        result = orchestrator.enhance(
            image=image,
            scale=config.scale,
            face_enhance=face_enhance,
            progress_callback=progress_callback,
            upscale_model=upscale_model,
            face_model=face_model,
        )

        # Save result
        save_image(result, output_path, quality=config.output_quality)
        logger.info(f"Saved: {output_path}")

        return True

    except Exception as e:
        logger.error(f"Error processing {input_path}: {e}")
        return False


def main() -> int:
    """Main entry point for CLI."""
    args = parse_arguments()

    # v2.0: Analyze mode
    if args.analyze:
        input_path = Path(args.input)
        if not input_path.exists():
            logger.error(f"Input path does not exist: {input_path}")
            return 1
        return analyze_image(input_path)

    # Setup configuration
    config = Config()
    
    # Apply preset if specified
    if args.preset:
        try:
            config.apply_preset(args.preset)
            logger.info(f"Applied preset: {args.preset}")
        except ValueError:
            # v2.0 preset will be applied via orchestrator
            pass
    
    config.scale = args.scale
    config.tile_size = args.tile_size
    config.gpu_id = args.gpu_id
    config.enable_denoise = args.denoise or (args.preset in ["quality", "restore"])
    config.output_format = args.format
    config.output_quality = args.quality
    config.codeformer_fidelity = args.fidelity

    input_path = Path(args.input)
    
    if not input_path.exists():
        logger.error(f"Input path does not exist: {input_path}")
        return 1

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        if input_path.is_file():
            output_path = input_path.parent / f"{input_path.stem}_{args.scale}x.{args.format}"
        else:
            output_path = input_path.parent / f"{input_path.name}_upscaled"

    # Model selection
    upscale_model = args.model
    face_model = args.face_model if args.face_enhance else "none"
    
    # Preset overrides
    if args.preset == "portrait":
        face_model = "codeformer"
        args.face_enhance = True
    elif args.preset == "anime":
        face_model = "none"
    elif args.preset == "restore":
        upscale_model = "swinir"
        face_model = "codeformer"
        args.face_enhance = True

    # Process single image or directory
    if input_path.is_file():
        success = process_single_image(
            input_path, output_path, config, args.face_enhance, upscale_model, face_model
        )
        return 0 if success else 1
    else:
        # Batch processing
        output_path.mkdir(parents=True, exist_ok=True)
        
        image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
        images = [
            f for f in input_path.iterdir()
            if f.suffix.lower() in image_extensions
        ]

        if not images:
            logger.error(f"No images found in: {input_path}")
            return 1

        logger.info(f"Found {len(images)} images to process")

        success_count = 0
        for img_path in images:
            out_path = output_path / f"{img_path.stem}_{args.scale}x.{args.format}"
            if process_single_image(img_path, out_path, config, args.face_enhance, upscale_model, face_model):
                success_count += 1

        logger.info(f"Processed {success_count}/{len(images)} images successfully")
        return 0 if success_count == len(images) else 1


if __name__ == "__main__":
    sys.exit(main())

