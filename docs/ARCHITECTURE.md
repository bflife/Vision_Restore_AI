# Vision-Restore AI Architecture

This document provides a technical overview of the Vision-Restore AI architecture for developers and contributors.

---

## Table of Contents

- [Overview](#overview)
- [Layer Architecture](#layer-architecture)
- [Core Components](#core-components)
- [ML Engine](#ml-engine)
- [GUI Layer](#gui-layer)
- [Data Flow](#data-flow)
- [Design Decisions](#design-decisions)

---

## Overview

Vision-Restore AI follows a **three-layer architecture**:

```
┌─────────────────────────────────────────────────────────────┐
│                      GUI Layer                              │
│  CustomTkinter • Components • Event Handling               │
├─────────────────────────────────────────────────────────────┤
│                     ML Engine Layer                         │
│  Real-ESRGAN • GFPGAN • Tiling • Pre/Post Processing       │
├─────────────────────────────────────────────────────────────┤
│                   Core Infrastructure                       │
│  Config • Logger • Model Manager • Thread Pool             │
└─────────────────────────────────────────────────────────────┘
```

---

## Layer Architecture

### 1. Core Infrastructure (`vision_restore/core/`)

Provides foundational services used by all other layers.

| Module | Purpose |
|--------|---------|
| `config.py` | Configuration management with persistence |
| `logger.py` | Colored console and file logging |
| `model_manager.py` | Model downloading and verification |
| `thread_pool.py` | Background task execution |

### 2. ML Engine (`vision_restore/engine/`)

Contains all image processing and AI inference logic.

| Module | Purpose |
|--------|---------|
| `realesrgan_engine.py` | Real-ESRGAN upscaling wrapper |
| `gfpgan_engine.py` | GFPGAN face restoration wrapper |
| `tiling.py` | Memory-efficient tile processing |
| `preprocessing.py` | Image preparation and cleanup |
| `postprocessing.py` | Final image optimization |
| `orchestrator.py` | Pipeline coordination |

### 3. GUI Layer (`vision_restore/gui/`)

User interface built with CustomTkinter.

| Module | Purpose |
|--------|---------|
| `app.py` | Main application window |
| `components/` | Reusable UI widgets |

---

## Core Components

### Configuration (`config.py`)

```python
@dataclass
class Config:
    # Model settings
    scale: int = 4
    realesrgan_model: str = "realesrgan-x4plus"
    
    # Processing settings
    tile_size: int = 512
    tile_pad: int = 32
    
    # Feature toggles
    enable_face_enhance: bool = True
    enable_denoise: bool = True
```

**Key Features:**
- Dataclass-based configuration
- JSON persistence to `~/.vision-restore/config.json`
- Quality presets (fast, balanced, quality)
- Model URL management

### Logger (`logger.py`)

```python
logger = get_logger(__name__)
logger.info("Processing started")
logger.debug("Tile 1/4 complete")
```

**Features:**
- Colored console output (via colorama)
- File logging with rotation
- Module-specific loggers

### Model Manager (`model_manager.py`)

```python
manager = ModelManager(config)
missing = manager.get_missing_models(face_enhance=True)
manager.download_all_required(face_enhance=True, progress_callback=...)
```

**Responsibilities:**
- Check for required models
- Download from official GitHub releases
- Progress reporting
- File integrity verification

---

## ML Engine

### Enhancement Orchestrator

The `EnhancementOrchestrator` coordinates the complete pipeline:

```python
orchestrator = EnhancementOrchestrator(config)
result = orchestrator.enhance(
    image,
    scale=4,
    face_enhance=True,
    preprocess=True,
    postprocess=True,
    progress_callback=callback,
)
```

**Pipeline Stages:**

1. **Pre-processing** (0-10%)
   - JPEG artifact reduction
   - Adaptive denoising
   - Color correction

2. **Upscaling** (10-70%)
   - Tile splitting
   - Real-ESRGAN inference
   - Tile merging

3. **Face Enhancement** (70-85%)
   - Face detection
   - GFPGAN restoration
   - Blending

4. **Post-processing** (85-100%)
   - Edge-aware sharpening
   - Detail enhancement

### Tiling Engine

The `TilingEngine` enables processing of arbitrarily large images:

```python
engine = TilingEngine(tile_size=512, tile_pad=32)

# Check if tiling needed
if engine.should_tile(image):
    result = engine.process_image(image, processor_fn, scale=4)
```

**Algorithm:**
1. Calculate tile grid with overlap
2. Split image into overlapping tiles
3. Process each tile through AI model
4. Merge with weighted blending

### Real-ESRGAN Engine

```python
engine = RealESRGANEngine(config, scale=4, gpu_id=0)
engine.initialize()
upscaled = engine.upscale(image, outscale=4)
```

**Backend Selection:**
1. Try PyTorch backend (more features)
2. Fall back to ncnn backend (lighter weight)
3. Fall back to CPU if no GPU

### GFPGAN Engine

```python
engine = GFPGANEngine(config, upscale=4, gpu_id=0)
engine.initialize()
restored, faces = engine.restore_faces(image)
```

**Process:**
1. Detect faces using FaceXLib
2. Restore each face with GFPGAN
3. Paste restored faces back with blending

---

## GUI Layer

### Main Application (`app.py`)

```python
class VisionRestoreApp(ctk.CTk):
    def __init__(self):
        # Initialize config, components, orchestrator
        
    def _enhancement_thread(self):
        # Background processing
        
    def _on_enhancement_complete(self, result):
        # Update UI with result
```

**Threading Model:**
- GUI runs on main thread
- Enhancement runs on background thread
- Progress updates via `after()` callback

### Components

| Component | Description |
|-----------|-------------|
| `BeforeAfterCanvas` | Interactive slider comparison |
| `ImageCanvas` | Zoomable/pannable image display |
| `SettingsPanel` | Configuration controls |
| `BatchPanel` | Queue management |
| `ProcessingProgress` | Status and progress bar |

---

## Data Flow

```
┌──────────────┐
│  User Input  │
│  (Image)     │
└──────┬───────┘
       │
       ▼
┌──────────────┐     ┌──────────────┐
│  Load Image  │────▶│  Display in  │
│  (image_io)  │     │  Canvas      │
└──────┬───────┘     └──────────────┘
       │
       ▼
┌──────────────┐
│ Enhancement  │◀──── Progress Callback
│ Orchestrator │
└──────┬───────┘
       │
       ├──▶ PreProcessor
       │
       ├──▶ TilingEngine
       │      │
       │      └──▶ RealESRGANEngine
       │
       ├──▶ GFPGANEngine
       │
       └──▶ PostProcessor
              │
              ▼
       ┌──────────────┐
       │ Update UI    │
       │ (After Image)│
       └──────────────┘
```

---

## Design Decisions

### 1. Lazy Engine Initialization

Engines are created on first use to minimize startup time:

```python
@property
def realesrgan(self) -> RealESRGANEngine:
    if self._realesrgan is None:
        self._realesrgan = RealESRGANEngine(...)
    return self._realesrgan
```

### 2. Backend Abstraction

The Real-ESRGAN engine abstracts the backend (PyTorch vs ncnn):

```python
def initialize(self):
    if self._try_pytorch_backend():
        return True
    if self._try_ncnn_backend():
        return True
    raise RuntimeError("No backend available")
```

### 3. Callback-Based Progress

All long-running operations accept progress callbacks:

```python
def enhance(self, image, progress_callback=None):
    progress_callback(0.1, "Starting...")
    # ... processing ...
    progress_callback(1.0, "Complete!")
```

### 4. Configuration as Data Class

Using `@dataclass` for configuration provides:
- Type hints
- Default values
- Easy serialization
- IDE autocomplete

### 5. Separation of Concerns

- **Engine**: Pure image processing, no UI knowledge
- **GUI**: Event handling, no processing logic
- **Core**: Shared services

---

## Extension Points

### Adding New Models

1. Add model URL to `config.py`:
   ```python
   model_urls["new-model"] = "https://..."
   ```

2. Create engine wrapper in `engine/`

3. Integrate into orchestrator pipeline

### Adding New Processing Steps

1. Create processor class in `engine/`
2. Add to orchestrator pipeline
3. Add config options as needed

### Adding GUI Features

1. Create component in `gui/components/`
2. Add to main app layout
3. Connect to existing callbacks

---

## File Structure

```
vision_restore/
├── __init__.py          # Package exports
├── main.py              # CLI entry point
│
├── core/                # Infrastructure
│   ├── __init__.py
│   ├── config.py        # Configuration
│   ├── logger.py        # Logging
│   ├── model_manager.py # Model downloads
│   └── thread_pool.py   # Background tasks
│
├── engine/              # ML Engine
│   ├── __init__.py
│   ├── orchestrator.py  # Pipeline coordinator
│   ├── realesrgan_engine.py
│   ├── gfpgan_engine.py
│   ├── tiling.py        # Tile processing
│   ├── preprocessing.py
│   └── postprocessing.py
│
├── gui/                 # User Interface
│   ├── __init__.py
│   ├── app.py           # Main window
│   └── components/
│       ├── before_after.py
│       ├── image_canvas.py
│       ├── settings_panel.py
│       ├── batch_panel.py
│       └── progress_bar.py
│
└── utils/               # Utilities
    ├── __init__.py
    ├── image_io.py      # Load/save images
    └── validators.py    # Input validation
```

---

## Testing Strategy

| Test Type | Location | Purpose |
|-----------|----------|---------|
| Unit | `tests/test_engine.py` | Individual components |
| Tiling | `tests/test_tiling.py` | Tile operations |
| Integration | `tests/test_gui.py` | Component interaction |

Run tests:
```bash
pytest tests/ -v
pytest tests/ --cov=vision_restore --cov-report=html
```

---

*Last updated: January 2024*
