"""
Vision-Restore AI - Intelligent Tiling Engine

Memory-efficient image processing through intelligent tiling with seamless blending.
Supports images of any size by splitting them into manageable tiles.
"""

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple
import numpy as np

from vision_restore.core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Tile:
    """Represents a single image tile with position information."""
    
    data: np.ndarray
    x: int  # X position in original image
    y: int  # Y position in original image
    width: int
    height: int
    pad_left: int = 0
    pad_right: int = 0
    pad_top: int = 0
    pad_bottom: int = 0
    
    @property
    def padded_width(self) -> int:
        """Width including padding."""
        return self.width + self.pad_left + self.pad_right
    
    @property
    def padded_height(self) -> int:
        """Height including padding."""
        return self.height + self.pad_top + self.pad_bottom


class TilingEngine:
    """Memory-efficient image tiling for large image processing.
    
    Features:
    - Automatic tile calculation based on image size
    - Configurable overlap (padding) to prevent visible seams
    - Seamless stitching using weighted blending
    - Progress callback support for UI updates
    """

    def __init__(
        self,
        tile_size: int = 512,
        tile_pad: int = 32,
        min_tile_size: int = 64,
    ):
        """Initialize the tiling engine.
        
        Args:
            tile_size: Target size for each tile
            tile_pad: Overlap/padding between tiles for seamless blending
            min_tile_size: Minimum tile size (smaller images won't be tiled)
        """
        self.tile_size = tile_size
        self.tile_pad = tile_pad
        self.min_tile_size = min_tile_size

    def should_tile(self, image: np.ndarray) -> bool:
        """Determine if an image should be tiled.
        
        Args:
            image: Input image
            
        Returns:
            True if image should be tiled
        """
        h, w = image.shape[:2]
        return w > self.tile_size or h > self.tile_size

    def calculate_tile_grid(
        self,
        width: int,
        height: int,
    ) -> List[Tuple[int, int, int, int]]:
        """Calculate tile positions for an image.
        
        Args:
            width: Image width
            height: Image height
            
        Returns:
            List of (x, y, tile_width, tile_height) tuples
        """
        tiles = []
        
        # Calculate step size (tile size minus overlap)
        step = self.tile_size - 2 * self.tile_pad
        if step <= 0:
            step = self.tile_size // 2
        
        y = 0
        while y < height:
            x = 0
            tile_h = min(self.tile_size, height - y)
            
            while x < width:
                tile_w = min(self.tile_size, width - x)
                tiles.append((x, y, tile_w, tile_h))
                
                x += step
                if x + self.tile_pad >= width:
                    break
            
            y += step
            if y + self.tile_pad >= height:
                break
        
        return tiles

    def split_into_tiles(self, image: np.ndarray) -> List[Tile]:
        """Split an image into overlapping tiles.
        
        Args:
            image: Input image (H, W, C)
            
        Returns:
            List of Tile objects
        """
        h, w = image.shape[:2]
        
        # For small images, return single tile
        if w <= self.tile_size and h <= self.tile_size:
            return [Tile(
                data=image.copy(),
                x=0, y=0,
                width=w, height=h,
            )]
        
        tiles = []
        tile_positions = self.calculate_tile_grid(w, h)
        
        for x, y, tile_w, tile_h in tile_positions:
            # Calculate padding needed for this tile
            pad_left = min(self.tile_pad, x)
            pad_top = min(self.tile_pad, y)
            pad_right = min(self.tile_pad, w - x - tile_w)
            pad_bottom = min(self.tile_pad, h - y - tile_h)
            
            # Extract tile with padding
            x_start = x - pad_left
            y_start = y - pad_top
            x_end = x + tile_w + pad_right
            y_end = y + tile_h + pad_bottom
            
            tile_data = image[y_start:y_end, x_start:x_end].copy()
            
            tiles.append(Tile(
                data=tile_data,
                x=x, y=y,
                width=tile_w, height=tile_h,
                pad_left=pad_left,
                pad_right=pad_right,
                pad_top=pad_top,
                pad_bottom=pad_bottom,
            ))
        
        logger.debug(f"Split image ({w}x{h}) into {len(tiles)} tiles")
        return tiles

    def process_tiles(
        self,
        tiles: List[Tile],
        processor: Callable[[np.ndarray], np.ndarray],
        scale: int = 1,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> List[Tile]:
        """Process each tile through a processing function.
        
        Args:
            tiles: List of tiles to process
            processor: Function that processes a single tile
            scale: Scale factor applied by processor
            progress_callback: Optional progress callback
            
        Returns:
            List of processed tiles
        """
        processed_tiles = []
        total = len(tiles)
        
        for i, tile in enumerate(tiles):
            if progress_callback:
                progress = i / total
                progress_callback(progress, f"Processing tile {i+1}/{total}")
            
            try:
                # Process tile
                processed_data = processor(tile.data)
                
                # Create new tile with scaled dimensions
                processed_tile = Tile(
                    data=processed_data,
                    x=tile.x * scale,
                    y=tile.y * scale,
                    width=tile.width * scale,
                    height=tile.height * scale,
                    pad_left=tile.pad_left * scale,
                    pad_right=tile.pad_right * scale,
                    pad_top=tile.pad_top * scale,
                    pad_bottom=tile.pad_bottom * scale,
                )
                processed_tiles.append(processed_tile)
                
            except Exception as e:
                logger.error(f"Failed to process tile {i+1}: {e}")
                raise
        
        if progress_callback:
            progress_callback(1.0, f"Processed {total} tiles")
        
        return processed_tiles

    def create_blend_mask(
        self,
        height: int,
        width: int,
        pad_top: int,
        pad_bottom: int,
        pad_left: int,
        pad_right: int,
    ) -> np.ndarray:
        """Create a gradient blending mask for seamless tile merging.
        
        Args:
            height: Mask height
            width: Mask width
            pad_top, pad_bottom, pad_left, pad_right: Padding amounts
            
        Returns:
            Blending mask (0 to 1)
        """
        mask = np.ones((height, width), dtype=np.float32)
        
        # Create feathered edges
        if pad_top > 0:
            for i in range(pad_top):
                mask[i, :] *= i / pad_top
        
        if pad_bottom > 0:
            for i in range(pad_bottom):
                mask[height - 1 - i, :] *= i / pad_bottom
        
        if pad_left > 0:
            for i in range(pad_left):
                mask[:, i] *= i / pad_left
        
        if pad_right > 0:
            for i in range(pad_right):
                mask[:, width - 1 - i] *= i / pad_right
        
        return mask

    def merge_tiles(
        self,
        tiles: List[Tile],
        output_width: int,
        output_height: int,
    ) -> np.ndarray:
        """Merge processed tiles back into a single image.
        
        Uses weighted blending in overlap regions for seamless results.
        
        Args:
            tiles: List of processed tiles
            output_width: Final output width
            output_height: Final output height
            
        Returns:
            Merged image
        """
        if not tiles:
            raise ValueError("No tiles to merge")
        
        # Get number of channels from first tile
        channels = tiles[0].data.shape[2] if len(tiles[0].data.shape) == 3 else 1
        
        # Create output arrays
        output = np.zeros((output_height, output_width, channels), dtype=np.float32)
        weight_sum = np.zeros((output_height, output_width), dtype=np.float32)
        
        for tile in tiles:
            # Remove padding from tile data to get actual content
            data = tile.data
            if len(data.shape) == 2:
                data = data[:, :, np.newaxis]
            
            # Calculate content region (excluding padding)
            content_y_start = tile.pad_top
            content_y_end = tile.padded_height - tile.pad_bottom
            content_x_start = tile.pad_left
            content_x_end = tile.padded_width - tile.pad_right
            
            # Check if we need to use padded data for blending
            y_start = tile.y
            y_end = tile.y + tile.height
            x_start = tile.x
            x_end = tile.x + tile.width
            
            # Get tile content
            tile_content = data[content_y_start:content_y_end, content_x_start:content_x_end]
            
            # Create blend mask
            mask = self.create_blend_mask(
                tile.height, tile.width,
                tile.pad_top, tile.pad_bottom,
                tile.pad_left, tile.pad_right,
            )
            
            # Ensure dimensions match
            if tile_content.shape[:2] != (tile.height, tile.width):
                # Resize if there's a mismatch
                import cv2
                tile_content = cv2.resize(
                    tile_content, 
                    (tile.width, tile.height),
                    interpolation=cv2.INTER_LANCZOS4,
                )
                if len(tile_content.shape) == 2:
                    tile_content = tile_content[:, :, np.newaxis]
            
            # Blend tile into output
            for c in range(channels):
                output[y_start:y_end, x_start:x_end, c] += (
                    tile_content[:, :, c].astype(np.float32) * mask
                )
            weight_sum[y_start:y_end, x_start:x_end] += mask
        
        # Normalize by weight sum
        weight_sum = np.maximum(weight_sum, 1e-10)
        for c in range(channels):
            output[:, :, c] /= weight_sum
        
        # Convert back to uint8
        output = np.clip(output, 0, 255).astype(np.uint8)
        
        if channels == 1:
            output = output[:, :, 0]
        
        logger.debug(f"Merged {len(tiles)} tiles into {output_width}x{output_height} image")
        return output

    def process_image(
        self,
        image: np.ndarray,
        processor: Callable[[np.ndarray], np.ndarray],
        scale: int = 1,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> np.ndarray:
        """Process an entire image with automatic tiling.
        
        This is a convenience method that combines split, process, and merge.
        
        Args:
            image: Input image
            processor: Processing function for each tile
            scale: Scale factor applied by processor
            progress_callback: Optional progress callback
            
        Returns:
            Processed image
        """
        h, w = image.shape[:2]
        output_w, output_h = w * scale, h * scale
        
        # For small images, process directly
        if not self.should_tile(image):
            if progress_callback:
                progress_callback(0.5, "Processing image...")
            result = processor(image)
            if progress_callback:
                progress_callback(1.0, "Complete")
            return result
        
        # Split into tiles
        if progress_callback:
            progress_callback(0.0, "Splitting image into tiles...")
        tiles = self.split_into_tiles(image)
        
        # Process tiles
        def tile_progress(p: float, msg: str):
            if progress_callback:
                # Map tile progress to 10-90% of total
                progress_callback(0.1 + p * 0.8, msg)
        
        processed_tiles = self.process_tiles(tiles, processor, scale, tile_progress)
        
        # Merge tiles
        if progress_callback:
            progress_callback(0.95, "Merging tiles...")
        
        result = self.merge_tiles(processed_tiles, output_w, output_h)
        
        if progress_callback:
            progress_callback(1.0, "Complete")
        
        return result


def calculate_optimal_tile_size(
    image_width: int,
    image_height: int,
    available_vram_mb: int = 4000,
    scale: int = 4,
    safety_factor: float = 0.7,
) -> int:
    """Calculate optimal tile size based on available memory.
    
    Args:
        image_width: Input image width
        image_height: Input image height
        available_vram_mb: Available VRAM in megabytes
        scale: Upscaling factor
        safety_factor: Fraction of VRAM to use
        
    Returns:
        Optimal tile size
    """
    # Estimate memory per pixel (float32 RGB, with intermediate buffers)
    bytes_per_pixel = 4 * 3  # float32, 3 channels
    buffer_multiplier = 4  # Account for intermediate processing buffers
    
    # Calculate safe VRAM usage
    safe_vram = available_vram_mb * 1024 * 1024 * safety_factor
    
    # Try tile sizes from large to small
    for tile_size in [768, 640, 512, 384, 256, 192, 128]:
        # Memory for input tile
        input_mem = tile_size * tile_size * bytes_per_pixel
        
        # Memory for output tile
        output_mem = (tile_size * scale) ** 2 * bytes_per_pixel
        
        # Total with buffers
        total_mem = (input_mem + output_mem) * buffer_multiplier
        
        if total_mem < safe_vram:
            logger.debug(f"Selected tile size {tile_size} for {available_vram_mb}MB VRAM")
            return tile_size
    
    logger.warning("Using minimum tile size due to memory constraints")
    return 128
