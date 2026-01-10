"""
Vision-Restore AI - Tiling Tests

Comprehensive tests for the tiling engine.
"""

import pytest
import numpy as np

pytest.importorskip("cv2")


class TestTileCreation:
    """Tests for tile creation."""

    def test_single_tile_for_small_image(self):
        """Small images should result in a single tile."""
        from vision_restore.engine.tiling import TilingEngine
        
        engine = TilingEngine(tile_size=512)
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        
        tiles = engine.split_into_tiles(image)
        
        assert len(tiles) == 1
        assert tiles[0].width == 100
        assert tiles[0].height == 100

    def test_multiple_tiles_for_large_image(self):
        """Large images should be split into multiple tiles."""
        from vision_restore.engine.tiling import TilingEngine
        
        engine = TilingEngine(tile_size=256, tile_pad=32)
        image = np.zeros((800, 1200, 3), dtype=np.uint8)
        
        tiles = engine.split_into_tiles(image)
        
        assert len(tiles) > 4  # Should have several tiles

    def test_tiles_cover_entire_image(self):
        """All tiles together should cover the entire image."""
        from vision_restore.engine.tiling import TilingEngine
        
        engine = TilingEngine(tile_size=200, tile_pad=20)
        image = np.zeros((500, 600, 3), dtype=np.uint8)
        
        tiles = engine.split_into_tiles(image)
        
        # Create a coverage map
        coverage = np.zeros((500, 600), dtype=int)
        
        for tile in tiles:
            y_end = min(tile.y + tile.height, 500)
            x_end = min(tile.x + tile.width, 600)
            coverage[tile.y:y_end, tile.x:x_end] += 1
        
        # Every pixel should be covered at least once
        assert (coverage >= 1).all()


class TestBlendMask:
    """Tests for blend mask creation."""

    def test_blend_mask_shape(self):
        """Blend mask should have correct shape."""
        from vision_restore.engine.tiling import TilingEngine
        
        engine = TilingEngine()
        
        mask = engine.create_blend_mask(
            height=100, width=150,
            pad_top=10, pad_bottom=10,
            pad_left=10, pad_right=10,
        )
        
        assert mask.shape == (100, 150)

    def test_blend_mask_center_is_one(self):
        """Center of blend mask should be 1.0."""
        from vision_restore.engine.tiling import TilingEngine
        
        engine = TilingEngine()
        
        mask = engine.create_blend_mask(
            height=100, width=100,
            pad_top=10, pad_bottom=10,
            pad_left=10, pad_right=10,
        )
        
        # Center should be 1.0 (or very close)
        center_value = mask[50, 50]
        assert abs(center_value - 1.0) < 0.01

    def test_blend_mask_edges_are_zero(self):
        """Edges of blend mask should approach 0."""
        from vision_restore.engine.tiling import TilingEngine
        
        engine = TilingEngine()
        
        mask = engine.create_blend_mask(
            height=100, width=100,
            pad_top=10, pad_bottom=10,
            pad_left=10, pad_right=10,
        )
        
        # Corners should be very low (near 0)
        assert mask[0, 0] < 0.1
        assert mask[0, 99] < 0.1
        assert mask[99, 0] < 0.1
        assert mask[99, 99] < 0.1


class TestTileMerging:
    """Tests for tile merging."""

    def test_merge_preserves_content(self):
        """Merging tiles should preserve image content."""
        from vision_restore.engine.tiling import TilingEngine
        
        engine = TilingEngine(tile_size=200, tile_pad=20)
        
        # Create a gradient image
        original = np.zeros((400, 600, 3), dtype=np.uint8)
        for i in range(400):
            for j in range(600):
                original[i, j] = [i % 256, j % 256, (i + j) % 256]
        
        tiles = engine.split_into_tiles(original)
        result = engine.merge_tiles(tiles, 600, 400)
        
        # Should be very similar to original
        diff = np.abs(original.astype(float) - result.astype(float)).mean()
        assert diff < 10  # Small average difference

    def test_merge_scaled_tiles(self):
        """Test merging tiles that have been scaled."""
        from vision_restore.engine.tiling import TilingEngine, Tile
        
        engine = TilingEngine(tile_size=100)
        
        # Create simple scaled tiles manually
        tile1 = Tile(
            data=np.full((200, 200, 3), 100, dtype=np.uint8),
            x=0, y=0,
            width=200, height=200,
        )
        tile2 = Tile(
            data=np.full((200, 200, 3), 150, dtype=np.uint8),
            x=200, y=0,
            width=200, height=200,
        )
        
        result = engine.merge_tiles([tile1, tile2], 400, 200)
        
        assert result.shape == (200, 400, 3)


class TestOptimalTileSize:
    """Tests for optimal tile size calculation."""

    def test_optimal_tile_size_high_vram(self):
        """High VRAM should allow larger tiles."""
        from vision_restore.engine.tiling import calculate_optimal_tile_size
        
        tile_size = calculate_optimal_tile_size(
            image_width=1000,
            image_height=1000,
            available_vram_mb=8000,  # 8GB
            scale=4,
        )
        
        assert tile_size >= 512

    def test_optimal_tile_size_low_vram(self):
        """Low VRAM should use smaller tiles."""
        from vision_restore.engine.tiling import calculate_optimal_tile_size
        
        tile_size = calculate_optimal_tile_size(
            image_width=1000,
            image_height=1000,
            available_vram_mb=200,  # 200MB (very constrained)
            scale=4,
        )
        
        # With very low VRAM, should get smaller tiles than high VRAM
        assert tile_size <= 384  # Should be forced to use smaller tiles
