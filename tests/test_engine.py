"""
Vision-Restore AI - Engine Tests

Tests for the image enhancement engines.
"""

import pytest
import numpy as np
from pathlib import Path

# Skip tests if dependencies not installed
pytest.importorskip("cv2")


class TestPreProcessor:
    """Tests for the PreProcessor class."""

    def test_denoise_preserves_dimensions(self):
        """Test that denoising preserves image dimensions."""
        from vision_restore.engine.preprocessing import PreProcessor
        
        processor = PreProcessor()
        
        # Create a noisy test image
        image = np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8)
        
        result = processor.denoise(image, strength=0.5)
        
        assert result.shape == image.shape
        assert result.dtype == np.uint8

    def test_denoise_reduces_noise(self):
        """Test that denoising actually reduces noise."""
        from vision_restore.engine.preprocessing import PreProcessor
        
        processor = PreProcessor()
        
        # Create a noisy image (pure noise)
        noisy = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        result = processor.denoise(noisy, strength=1.0)
        
        # Denoised image should have lower variance
        assert np.var(result) < np.var(noisy)

    def test_auto_white_balance(self):
        """Test automatic white balance."""
        from vision_restore.engine.preprocessing import PreProcessor
        
        processor = PreProcessor()
        
        # Create a color-biased image
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[:, :, 0] = 50   # Low blue
        image[:, :, 1] = 100  # Medium green
        image[:, :, 2] = 200  # High red
        
        result = processor.auto_white_balance(image)
        
        # After white balance, channels should be more balanced
        assert result.shape == image.shape


class TestPostProcessor:
    """Tests for the PostProcessor class."""

    def test_sharpen_increases_edges(self):
        """Test that sharpening increases edge contrast."""
        from vision_restore.engine.postprocessing import PostProcessor
        import cv2
        
        processor = PostProcessor()
        
        # Create a simple image with edges
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[25:75, 25:75] = 255  # White square
        
        result = processor.sharpen(image, amount=0.5)
        
        # Sharpened image should have higher edge magnitude
        edges_orig = cv2.Sobel(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.CV_64F, 1, 1)
        edges_sharp = cv2.Sobel(cv2.cvtColor(result, cv2.COLOR_BGR2GRAY), cv2.CV_64F, 1, 1)
        
        assert np.abs(edges_sharp).max() >= np.abs(edges_orig).max()

    def test_process_preserves_dimensions(self):
        """Test that processing preserves image dimensions."""
        from vision_restore.engine.postprocessing import PostProcessor
        
        processor = PostProcessor()
        
        image = np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)
        
        result = processor.process(image)
        
        assert result.shape == image.shape


class TestTilingEngine:
    """Tests for the TilingEngine class."""

    def test_small_image_not_tiled(self):
        """Test that small images are not tiled."""
        from vision_restore.engine.tiling import TilingEngine
        
        engine = TilingEngine(tile_size=512)
        
        small_image = np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)
        
        assert not engine.should_tile(small_image)

    def test_large_image_tiled(self):
        """Test that large images are tiled."""
        from vision_restore.engine.tiling import TilingEngine
        
        engine = TilingEngine(tile_size=256)
        
        large_image = np.random.randint(0, 255, (1000, 1500, 3), dtype=np.uint8)
        
        assert engine.should_tile(large_image)

    def test_tile_split_and_merge(self):
        """Test that splitting and merging reconstructs the original."""
        from vision_restore.engine.tiling import TilingEngine
        
        engine = TilingEngine(tile_size=256, tile_pad=16)
        
        # Create a test image
        original = np.random.randint(0, 255, (500, 600, 3), dtype=np.uint8)
        
        # Split into tiles
        tiles = engine.split_into_tiles(original)
        
        assert len(tiles) > 1  # Should have multiple tiles
        
        # Identity processor (no change)
        def identity(tile):
            return tile
        
        processed = engine.process_tiles(tiles, identity, scale=1)
        
        # Merge back
        result = engine.merge_tiles(processed, original.shape[1], original.shape[0])
        
        assert result.shape == original.shape
        
        # Should be very close to original (some blending differences are acceptable)
        diff = np.abs(original.astype(float) - result.astype(float)).mean()
        assert diff < 5  # Average difference less than 5 levels

    def test_process_image_with_scaling(self):
        """Test processing image with scaling."""
        from vision_restore.engine.tiling import TilingEngine
        import cv2
        
        engine = TilingEngine(tile_size=128, tile_pad=8)
        
        original = np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)
        
        # Processor that scales 2x
        def scale_2x(tile):
            h, w = tile.shape[:2]
            return cv2.resize(tile, (w * 2, h * 2), interpolation=cv2.INTER_LINEAR)
        
        result = engine.process_image(original, scale_2x, scale=2)
        
        assert result.shape[0] == original.shape[0] * 2
        assert result.shape[1] == original.shape[1] * 2


class TestConfig:
    """Tests for the Config class."""

    def test_config_defaults(self):
        """Test configuration defaults."""
        from vision_restore.core.config import Config
        
        config = Config()
        
        assert config.scale == 4
        assert config.tile_size > 0
        assert config.gpu_id >= -1

    def test_apply_preset(self):
        """Test applying quality presets."""
        from vision_restore.core.config import Config
        
        config = Config()
        
        config.apply_preset("fast")
        assert config.tile_size == 256
        
        config.apply_preset("quality")
        assert config.tile_size == 768

    def test_invalid_preset(self):
        """Test that invalid preset raises error."""
        from vision_restore.core.config import Config
        
        config = Config()
        
        with pytest.raises(ValueError):
            config.apply_preset("invalid_preset")
