# Changelog

All notable changes to Vision-Restore AI will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.1.0] - 2026-01-10

### Major Features
- **Full AMD/Intel Support**: First-class support for non-NVIDIA GPUs via `torch-directml`.
- **Pulse Mode™**: Revolutionary thermal management system for laptops. Prevents shutdowns by injecting micro-cooling pauses derived from variable gaming workloads.
- **Recursive Enhancement**: Added "Use as Input" button to instantly re-process results.
- **Safe Mode**: CLI and GUI support for forced CPU execution (`--gpu-id -1`) via `run_safe_mode.bat`.

### Improvements
- **HAT Engine**: Added support for DirectML with optimized tiling strategies.
- **Stability**: Implemented Hard Tile Clamping (192px) for DirectML devices to prevent VRAM/Power spikes.
- **Unified Backend**: All AI engines now use a central `gpu_backend.py` for device negotiation.

### Fixed
- Fixed critical power-off crashes on laptops using DirectML/AMD GPUs.
- Fixed `AttributeError` in CLI execution flow.

---

## [2.0.0] - 2026-01-09

### Pro Features (Full Scale Mode)
- **Interactive Comparison Slider**: Professional split-view for Before/After inspection with draggable slider.
- **Zoom & Pan Controls**: Mouse-centered zooming (wheel) and panning (drag) for pixel-perfect analysis.
- **Save Comparison**: New functionality to export clean side-by-side or slider comparisons.
- **Preset Manager**: Save and load custom setting combinations (e.g., "Old Photo Fix", "Anime 4x").
- **Manual Post-Processing**: Direct control over post-processing strength when Quality Analysis is off.

### Fixes & Improvements
- **Startup Stability**: Fixed critical `NameError` and dependency checks (`basicsr` compatibility).
- **GUI Fixes**: Resolved Tkinter color errors in canvas.
- **Performance**: Improved rendering for large comparisons.

### Changed
- Reverted AMD DirectML support to ensure maximum stability on CPU/CUDA.
- Updated default presets.

## [2.0.0] - 2026-01-09

### Added
- **SwinIR Model Support**
  - Superior image quality for certain degradation types
  - 4 model variants: classical-x2, classical-x4, real-x4, lightweight-x4
  - Automatic tiled processing for memory efficiency

- **CodeFormer Face Restoration**
  - Superior face restoration compared to GFPGAN
  - Adjustable fidelity parameter (quality vs identity balance)
  - Better handling of heavily degraded/old photos
  - Automatic fallback to GFPGAN if unavailable

- **Quality Metrics Engine**
  - BRISQUE no-reference image quality assessment
  - Sharpness analysis (Laplacian/Tenengrad variance)
  - Noise level estimation (Immerkær method)
  - JPEG artifact detection
  - Color cast and variety analysis
  - Anime/cartoon content detection
  - Face detection for smart processing

- **Intelligent Auto-Selection**
  - Automatic model selection based on image content
  - Recommended processing settings from quality analysis
  - Smart face model selection (GFPGAN vs CodeFormer)

- **New CLI Features**
  - `--model` flag: auto, realesrgan, swinir
  - `--face-model` flag: auto, gfpgan, codeformer, none
  - `--fidelity` flag for CodeFormer (0.0-1.0)
  - `--preset` flag: fast, balanced, quality, portrait, anime, restore
  - `--analyze` mode: analyze image quality without processing

- **New Enhancement Methods**
  - `enhance_portrait()`: Optimized for photos with faces
  - `enhance_restore()`: Best for old/damaged photos
  - `get_quality_comparison()`: Compare before/after quality

- **New Presets**
  - Portrait: CodeFormer face restoration optimized
  - Anime: Anime-specific model without face enhancement
  - Restore: SwinIR + CodeFormer for damaged photos

### Changed
- Orchestrator upgraded to v2.0 with multi-model pipeline
- Config updated with new model selection fields
- Engine __init__.py exports all new components

### Technical Details
- Backward compatible with v1.0 processing
- All 34 existing tests pass
- New optional dependencies: scipy (for BRISQUE)

---

## [1.0.0] - 2024-01-08

### Added
- **Core Features**
  - Real-ESRGAN integration for 2x/4x/8x image upscaling
  - GFPGAN integration for AI-powered face restoration
  - Intelligent tiling engine for memory-efficient processing
  - Pre-processing pipeline (denoising, color correction, artifact removal)
  - Post-processing pipeline (sharpening, detail enhancement)

- **User Interface**
  - Modern CustomTkinter-based GUI with dark theme
  - Interactive Before/After comparison slider
  - Comprehensive settings panel with quality presets
  - Batch processing panel with queue management
  - Progress indicators with real-time status updates

- **Performance**
  - GPU acceleration with CUDA/Vulkan support
  - Dual backend: PyTorch (full features) + ncnn (lightweight)
  - Automatic tile size optimization based on available VRAM
  - Seamless tile blending for large images

- **Developer Features**
  - CLI interface for automation and scripting
  - Comprehensive logging with colored console output
  - Configuration persistence in JSON format
  - Automated model download with progress tracking

- **Documentation**
  - README with installation and usage instructions
  - Contributing guidelines
  - API documentation
  - Security policy

### Technical Details
- Python 3.9+ support
- Cross-platform: Windows, Linux, macOS
- ~4,500 lines of code
- 30 Python modules
- Comprehensive test suite

---

## Version History Format

### Types of Changes

- **Added** - New features
- **Changed** - Changes in existing functionality
- **Deprecated** - Soon-to-be removed features
- **Removed** - Removed features
- **Fixed** - Bug fixes
- **Security** - Vulnerability fixes

---

[Unreleased]: https://github.com/Rav-xyl/vision-restore-ai/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/Rav-xyl/vision-restore-ai/releases/tag/v1.0.0
