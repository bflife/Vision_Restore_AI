"""
Vision-Restore AI - Test Configuration

Shared fixtures and configuration for pytest.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil


@pytest.fixture
def sample_image():
    """Create a sample test image."""
    # Create a simple 100x100 RGB image with a gradient
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    for i in range(100):
        for j in range(100):
            image[i, j] = [i % 256, j % 256, (i + j) % 256]
    return image


@pytest.fixture
def large_image():
    """Create a larger test image for tiling tests."""
    return np.random.randint(0, 255, (500, 600, 3), dtype=np.uint8)


@pytest.fixture
def grayscale_image():
    """Create a grayscale test image."""
    return np.random.randint(0, 255, (100, 100), dtype=np.uint8)


@pytest.fixture
def noisy_image():
    """Create a noisy test image for denoising tests."""
    base = np.ones((100, 100, 3), dtype=np.uint8) * 128
    noise = np.random.randint(-30, 30, (100, 100, 3), dtype=np.int16)
    noisy = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return noisy


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def sample_image_file(sample_image, temp_dir):
    """Create a temporary image file."""
    import cv2
    
    image_path = temp_dir / "test_image.png"
    cv2.imwrite(str(image_path), sample_image)
    return image_path


@pytest.fixture
def test_config():
    """Create a test configuration."""
    from vision_restore.core.config import Config
    
    config = Config()
    config.scale = 2  # Use smaller scale for faster tests
    config.tile_size = 128
    config.tile_pad = 16
    config.gpu_id = -1  # Force CPU for tests
    config.enable_face_enhance = False  # Disable for faster tests
    return config


@pytest.fixture
def mock_processor():
    """Create a mock image processor for tiling tests."""
    def processor(image):
        # Simple 2x upscale using nearest neighbor
        import cv2
        h, w = image.shape[:2]
        return cv2.resize(image, (w * 2, h * 2), interpolation=cv2.INTER_NEAREST)
    return processor


@pytest.fixture
def identity_processor():
    """Create an identity processor (returns input unchanged)."""
    def processor(image):
        return image.copy()
    return processor


# Markers for slow tests
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU (deselect with '-m \"not gpu\"')"
    )


# Skip GPU tests if CUDA not available
def pytest_collection_modifyitems(config, items):
    """Modify test collection based on available resources."""
    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except ImportError:
        has_cuda = False
    
    skip_gpu = pytest.mark.skip(reason="GPU not available")
    
    for item in items:
        if "gpu" in item.keywords and not has_cuda:
            item.add_marker(skip_gpu)
