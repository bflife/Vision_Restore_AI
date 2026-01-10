"""
Vision-Restore AI - GUI Tests

Tests for GUI components (non-visual functionality).
"""

import pytest


class TestConfig:
    """Tests for configuration saving and loading."""

    def test_config_to_dict(self):
        """Test converting config to dictionary."""
        from vision_restore.core.config import Config
        
        config = Config()
        config.scale = 8
        config.enable_face_enhance = False
        
        data = config.to_dict()
        
        assert data["scale"] == 8
        assert data["enable_face_enhance"] == False

    def test_preset_application(self):
        """Test that presets correctly modify settings."""
        from vision_restore.core.config import Config
        
        config = Config()
        
        # Apply fast preset
        config.apply_preset("fast")
        assert config.tile_size == 256
        assert config.enable_denoise == False
        
        # Apply quality preset
        config.apply_preset("quality")
        assert config.tile_size == 768
        assert config.enable_denoise == True


class TestValidators:
    """Tests for input validators."""

    def test_validate_scale_valid(self):
        """Test validation of valid scale values."""
        from vision_restore.utils.validators import validate_scale
        
        assert validate_scale(2)[0] == True
        assert validate_scale(4)[0] == True
        assert validate_scale(8)[0] == True

    def test_validate_scale_invalid(self):
        """Test validation of invalid scale values."""
        from vision_restore.utils.validators import validate_scale
        
        assert validate_scale(1)[0] == False
        assert validate_scale(3)[0] == False
        assert validate_scale(16)[0] == False

    def test_validate_quality_valid(self):
        """Test validation of valid quality values."""
        from vision_restore.utils.validators import validate_quality
        
        assert validate_quality(1)[0] == True
        assert validate_quality(50)[0] == True
        assert validate_quality(100)[0] == True

    def test_validate_quality_invalid(self):
        """Test validation of invalid quality values."""
        from vision_restore.utils.validators import validate_quality
        
        assert validate_quality(0)[0] == False
        assert validate_quality(101)[0] == False
        assert validate_quality(-5)[0] == False


class TestImageIO:
    """Tests for image I/O utilities."""

    def test_numpy_to_pil_conversion(self):
        """Test numpy to PIL conversion."""
        import numpy as np
        from vision_restore.utils.image_io import numpy_to_pil
        
        # Create BGR numpy image
        image = np.zeros((100, 150, 3), dtype=np.uint8)
        image[:, :, 0] = 255  # Blue channel
        
        pil_image = numpy_to_pil(image)
        
        assert pil_image.size == (150, 100)  # PIL uses (width, height)
        assert pil_image.mode == "RGB"
        
        # Check that BGR to RGB conversion happened
        r, g, b = pil_image.split()
        assert np.array(r).max() == 0  # Red should be 0
        assert np.array(b).max() == 255  # Blue should be 255

    def test_pil_to_numpy_conversion(self):
        """Test PIL to numpy conversion."""
        import numpy as np
        from PIL import Image
        from vision_restore.utils.image_io import pil_to_numpy
        
        # Create RGB PIL image
        pil_image = Image.new("RGB", (150, 100), color=(255, 0, 0))  # Red
        
        numpy_image = pil_to_numpy(pil_image)
        
        assert numpy_image.shape == (100, 150, 3)
        # Check BGR conversion (blue should now have the red value)
        assert numpy_image[0, 0, 2] == 255  # Red -> Blue channel in BGR

    def test_get_image_info(self):
        """Test getting image information."""
        import numpy as np
        from vision_restore.utils.image_io import get_image_info
        
        image = np.zeros((1080, 1920, 3), dtype=np.uint8)
        
        info = get_image_info(image)
        
        assert info["width"] == 1920
        assert info["height"] == 1080
        assert info["channels"] == 3
        assert info["megapixels"] == pytest.approx(2.0736, rel=0.01)

    def test_resize_image(self):
        """Test image resizing."""
        import numpy as np
        from vision_restore.utils.image_io import resize_image
        
        # Create a large image
        image = np.zeros((2000, 3000, 3), dtype=np.uint8)
        
        # Resize to fit within 1000px
        resized = resize_image(image, max_size=1000)
        
        assert max(resized.shape[:2]) <= 1000


class TestModelManager:
    """Tests for model manager."""

    def test_get_required_models_basic(self):
        """Test getting required models list."""
        from vision_restore.core.model_manager import ModelManager
        from vision_restore.core.config import Config
        
        config = Config()
        config.scale = 4
        manager = ModelManager(config)
        
        models = manager.get_required_models(face_enhance=False)
        
        assert "realesrgan-x4plus" in models

    def test_get_required_models_with_face(self):
        """Test getting required models with face enhancement."""
        from vision_restore.core.model_manager import ModelManager
        from vision_restore.core.config import Config
        
        config = Config()
        manager = ModelManager(config)
        
        models = manager.get_required_models(face_enhance=True)
        
        assert "realesrgan-x4plus" in models
        assert "GFPGANv1.4" in models
