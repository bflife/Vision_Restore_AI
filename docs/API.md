# Vision-Restore AI API Reference

This document provides detailed API documentation for developers integrating with or extending Vision-Restore AI.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Core API](#core-api)
- [Engine API](#engine-api)
- [Utilities](#utilities)
- [Configuration](#configuration)

---

## Quick Start

### Basic Usage

```python
from vision_restore.engine.orchestrator import EnhancementOrchestrator
from vision_restore.utils.image_io import load_image, save_image

# Load image
image = load_image("photo.jpg")

# Create orchestrator
orchestrator = EnhancementOrchestrator()

# Enhance image
result = orchestrator.enhance(
    image,
    scale=4,
    face_enhance=True,
)

# Save result
save_image(result, "photo_4x.png")

# Clean up
orchestrator.cleanup()
```

### With Progress Callback

```python
def on_progress(progress: float, message: str):
    print(f"{progress*100:.0f}% - {message}")

result = orchestrator.enhance(
    image,
    scale=4,
    progress_callback=on_progress,
)
```

---

## Core API

### Configuration

#### `vision_restore.core.config.Config`

Application configuration dataclass.

```python
from vision_restore.core.config import Config, get_config

# Get global config
config = get_config()

# Create custom config
config = Config()
config.scale = 4
config.enable_face_enhance = True
config.tile_size = 512
```

**Attributes:**

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `scale` | int | 4 | Upscale factor (2, 4, 8) |
| `tile_size` | int | 512 | Tile size for processing |
| `tile_pad` | int | 32 | Tile overlap |
| `enable_face_enhance` | bool | True | Enable GFPGAN |
| `enable_denoise` | bool | True | Enable denoising |
| `enable_sharpen` | bool | True | Enable sharpening |
| `denoise_strength` | float | 0.5 | Denoise intensity (0-1) |
| `sharpen_amount` | float | 0.3 | Sharpen intensity (0-1) |
| `gpu_id` | int | 0 | GPU ID (-1 for CPU) |
| `output_format` | str | "png" | Output format |
| `output_quality` | int | 95 | JPEG/WebP quality |

**Methods:**

```python
# Apply quality preset
config.apply_preset("quality")  # fast, balanced, quality

# Save/load configuration
config.save()
config.load()

# Get model path
path = config.get_model_path("realesrgan-x4plus")

# Convert to dictionary
data = config.to_dict()
```

---

### Logger

#### `vision_restore.core.logger.get_logger`

Get a configured logger instance.

```python
from vision_restore.core.logger import get_logger

logger = get_logger(__name__)
logger.info("Processing started")
logger.debug("Debug information")
logger.warning("Warning message")
logger.error("Error occurred")
```

---

### Model Manager

#### `vision_restore.core.model_manager.ModelManager`

Manages AI model downloads and verification.

```python
from vision_restore.core.model_manager import ModelManager

manager = ModelManager(config)

# Check for missing models
missing = manager.get_missing_models(face_enhance=True)

# Download size estimate
size_mb = manager.get_total_download_size(face_enhance=True)

# Download models
success = manager.download_all_required(
    face_enhance=True,
    progress_callback=lambda p, m: print(f"{p*100:.0f}%"),
)

# Get required models list
models = manager.get_required_models(face_enhance=True)
```

---

## Engine API

### Enhancement Orchestrator

#### `vision_restore.engine.orchestrator.EnhancementOrchestrator`

Main pipeline coordinator.

```python
from vision_restore.engine.orchestrator import EnhancementOrchestrator

orchestrator = EnhancementOrchestrator(config)
```

**Methods:**

##### `enhance()`

Run the complete enhancement pipeline.

```python
result = orchestrator.enhance(
    image: np.ndarray,           # Input image (BGR, uint8)
    scale: int = 4,              # Upscale factor
    face_enhance: bool = True,   # Enable face restoration
    preprocess: bool = True,     # Enable pre-processing
    postprocess: bool = True,    # Enable post-processing
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> np.ndarray
```

##### `enhance_quick()`

Fast enhancement without face restoration.

```python
result = orchestrator.enhance_quick(image, scale=4)
```

##### `enhance_maximum()`

Maximum quality with all features enabled.

```python
result = orchestrator.enhance_maximum(image, scale=4)
```

##### `check_models()`

Check if required models are available.

```python
available, missing = orchestrator.check_models(face_enhance=True)
```

##### `download_models()`

Download required models.

```python
success = orchestrator.download_models(face_enhance=True, progress_callback=...)
```

##### `estimate_memory()`

Estimate memory required for processing.

```python
mb = orchestrator.estimate_memory(input_size=(1920, 1080), scale=4)
```

##### `cleanup()`

Release resources.

```python
orchestrator.cleanup()
```

---

### Real-ESRGAN Engine

#### `vision_restore.engine.realesrgan_engine.RealESRGANEngine`

Upscaling engine wrapper.

```python
from vision_restore.engine.realesrgan_engine import RealESRGANEngine

engine = RealESRGANEngine(
    config=config,
    scale=4,
    tile_size=512,
    tile_pad=32,
    gpu_id=0,
)
```

**Methods:**

```python
# Initialize engine
engine.initialize()

# Check initialization status
if engine.is_initialized:
    ...

# Upscale image
result = engine.upscale(image, outscale=4)

# Clean up
engine.cleanup()
```

---

### GFPGAN Engine

#### `vision_restore.engine.gfpgan_engine.GFPGANEngine`

Face restoration engine wrapper.

```python
from vision_restore.engine.gfpgan_engine import GFPGANEngine

engine = GFPGANEngine(
    config=config,
    upscale=4,
    gpu_id=0,
)
```

**Methods:**

```python
# Initialize engine
engine.initialize()

# Check for faces
face_count = engine.get_face_count(image)

# Restore faces
restored_image, cropped_faces = engine.restore_faces(image)

# Clean up
engine.cleanup()
```

---

### Tiling Engine

#### `vision_restore.engine.tiling.TilingEngine`

Memory-efficient tile processing.

```python
from vision_restore.engine.tiling import TilingEngine

engine = TilingEngine(
    tile_size=512,
    tile_pad=32,
    min_tile_size=64,
)
```

**Methods:**

##### `should_tile()`

Check if image needs tiling.

```python
needs_tiling = engine.should_tile(image)
```

##### `process_image()`

Process image with automatic tiling.

```python
result = engine.process_image(
    image: np.ndarray,
    processor: Callable[[np.ndarray], np.ndarray],
    scale: int = 1,
    progress_callback: Optional[Callable] = None,
) -> np.ndarray
```

##### `split_into_tiles()`

Split image into tiles.

```python
tiles = engine.split_into_tiles(image)
```

##### `merge_tiles()`

Merge tiles back into image.

```python
result = engine.merge_tiles(tiles, output_width, output_height)
```

#### `calculate_optimal_tile_size()`

Calculate optimal tile size based on VRAM.

```python
from vision_restore.engine.tiling import calculate_optimal_tile_size

tile_size = calculate_optimal_tile_size(
    image_width=1920,
    image_height=1080,
    available_vram_mb=4000,
    scale=4,
)
```

---

### Pre-Processor

#### `vision_restore.engine.preprocessing.PreProcessor`

Image pre-processing.

```python
from vision_restore.engine.preprocessing import PreProcessor, AdaptivePreProcessor

processor = PreProcessor(config)
# or
processor = AdaptivePreProcessor(config)  # Auto-adjusts settings
```

**Methods:**

```python
# Full pipeline
result = processor.process(
    image,
    denoise=True,
    color_correct=True,
    reduce_artifacts=True,
)

# Individual operations
denoised = processor.denoise(image, strength=0.5)
balanced = processor.auto_white_balance(image)
enhanced = processor.enhance_contrast(image)

# Adaptive processing
result = processor.auto_process(image)  # AdaptivePreProcessor only
```

---

### Post-Processor

#### `vision_restore.engine.postprocessing.PostProcessor`

Image post-processing.

```python
from vision_restore.engine.postprocessing import PostProcessor, AdaptivePostProcessor

processor = PostProcessor(config)
# or
processor = AdaptivePostProcessor(config)
```

**Methods:**

```python
# Full pipeline
result = processor.process(
    image,
    sharpen=True,
    enhance_details=True,
    enhance_vibrance=False,
)

# Individual operations
sharpened = processor.sharpen(image, amount=0.5)
detailed = processor.enhance_details(image, amount=0.3)

# Adaptive processing
result = processor.auto_process(image)
```

---

## Utilities

### Image I/O

#### `vision_restore.utils.image_io`

Image loading and saving utilities.

```python
from vision_restore.utils.image_io import (
    load_image,
    save_image,
    numpy_to_pil,
    pil_to_numpy,
    get_image_info,
    resize_image,
)
```

**Functions:**

##### `load_image()`

Load image from file.

```python
image = load_image(path, convert_rgb=True)  # Returns BGR numpy array or None
```

##### `save_image()`

Save image to file.

```python
success = save_image(image, path, quality=95)  # Returns bool
```

##### `numpy_to_pil()`

Convert numpy array to PIL Image.

```python
pil_image = numpy_to_pil(numpy_image)  # BGR to RGB conversion
```

##### `pil_to_numpy()`

Convert PIL Image to numpy array.

```python
numpy_image = pil_to_numpy(pil_image)  # RGB to BGR conversion
```

##### `get_image_info()`

Get image information.

```python
info = get_image_info(image)
# Returns: {'width': 1920, 'height': 1080, 'channels': 3, 'megapixels': 2.07, ...}
```

##### `resize_image()`

Resize image with aspect ratio preservation.

```python
resized = resize_image(image, max_size=4096, min_size=32)
```

---

### Validators

#### `vision_restore.utils.validators`

Input validation utilities.

```python
from vision_restore.utils.validators import (
    validate_image_path,
    validate_output_path,
    validate_scale,
    validate_quality,
    is_valid_image,
    get_image_files,
)
```

**Functions:**

```python
# Validate paths
valid, error = validate_image_path(path)
valid, error = validate_output_path(path)

# Validate settings
valid, error = validate_scale(4)
valid, error = validate_quality(95)

# Check if valid image
is_valid = is_valid_image(path)

# Get all image files in directory
files = get_image_files(directory_path)
```

---

## Error Handling

```python
from vision_restore.engine.orchestrator import EnhancementOrchestrator

try:
    orchestrator = EnhancementOrchestrator()
    result = orchestrator.enhance(image, scale=4)
except RuntimeError as e:
    print(f"Engine initialization failed: {e}")
except ValueError as e:
    print(f"Invalid parameter: {e}")
except Exception as e:
    print(f"Processing failed: {e}")
finally:
    orchestrator.cleanup()
```

---

## Thread Safety

The orchestrator and engines are **not thread-safe**. Create separate instances for concurrent processing:

```python
from concurrent.futures import ThreadPoolExecutor

def process_image(path):
    orchestrator = EnhancementOrchestrator()
    try:
        image = load_image(path)
        result = orchestrator.enhance(image)
        save_image(result, f"{path.stem}_enhanced.png")
    finally:
        orchestrator.cleanup()

with ThreadPoolExecutor(max_workers=2) as executor:
    executor.map(process_image, image_paths)
```

---

*Last updated: January 2024*
