"""
Vision-Restore AI - Python Client Example

Demonstrates how to use the Vision-Restore AI API from Python code.
"""

import requests
import base64
from pathlib import Path
from typing import Optional, Dict, Any
import json


class VisionRestoreClient:
    """Client for Vision-Restore AI API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize the client.
        
        Args:
            base_url: Base URL of the API service
        """
        self.base_url = base_url.rstrip('/')
        self.api_url = f"{self.base_url}/api/v1"
    
    def check_status(self) -> Dict[str, Any]:
        """
        Check service status.
        
        Returns:
            Service status information
        """
        response = requests.get(f"{self.api_url}/status")
        response.raise_for_status()
        return response.json()
    
    def get_models(self) -> Dict[str, Any]:
        """
        Get available models.
        
        Returns:
            Available models information
        """
        response = requests.get(f"{self.api_url}/models")
        response.raise_for_status()
        return response.json()
    
    def enhance_image(
        self,
        image_path: str,
        output_path: Optional[str] = None,
        scale: int = 4,
        upscale_model: str = "auto",
        face_model: str = "auto",
        face_enhance: bool = True,
        codeformer_fidelity: float = 0.7,
        enable_denoise: bool = True,
        enable_sharpen: bool = True,
        denoise_strength: float = 0.3,
        sharpen_amount: float = 0.2,
        output_format: str = "png",
        quality: int = 95,
        return_base64: bool = False,
    ) -> Optional[bytes]:
        """
        Enhance an image.
        
        Args:
            image_path: Path to input image
            output_path: Path to save output image (if None and not return_base64, returns bytes)
            scale: Upscale factor (2, 4, or 8)
            upscale_model: Upscaling model (auto, realesrgan, swinir, hat)
            face_model: Face restoration model (auto, gfpgan, codeformer, none)
            face_enhance: Enable face enhancement
            codeformer_fidelity: CodeFormer fidelity (0=quality, 1=fidelity)
            enable_denoise: Enable denoising
            enable_sharpen: Enable sharpening
            denoise_strength: Denoise strength (0.0-1.0)
            sharpen_amount: Sharpen amount (0.0-1.0)
            output_format: Output format (png, jpg, webp)
            quality: Output quality (1-100)
            return_base64: Return base64 encoded image instead of binary
        
        Returns:
            Enhanced image bytes (if not saving to file) or None
        """
        # Prepare files and data
        files = {
            'file': open(image_path, 'rb')
        }
        
        data = {
            'scale': scale,
            'upscale_model': upscale_model,
            'face_model': face_model,
            'face_enhance': face_enhance,
            'codeformer_fidelity': codeformer_fidelity,
            'enable_denoise': enable_denoise,
            'enable_sharpen': enable_sharpen,
            'denoise_strength': denoise_strength,
            'sharpen_amount': sharpen_amount,
            'output_format': output_format,
            'quality': quality,
            'return_base64': return_base64,
        }
        
        try:
            # Send request
            response = requests.post(
                f"{self.api_url}/enhance",
                files=files,
                data=data,
            )
            response.raise_for_status()
            
            if return_base64:
                # JSON response with base64 data
                result = response.json()
                image_data = base64.b64decode(result['data'])
                
                if output_path:
                    with open(output_path, 'wb') as f:
                        f.write(image_data)
                    print(f"✓ Image saved to: {output_path}")
                    print(f"  Processing time: {result.get('processing_time', 'N/A'):.2f}s")
                    print(f"  Output size: {result['output_size']['width']}x{result['output_size']['height']}")
                    return None
                else:
                    return image_data
            else:
                # Binary image response
                image_bytes = response.content
                
                if output_path:
                    with open(output_path, 'wb') as f:
                        f.write(image_bytes)
                    
                    # Get metadata from headers
                    processing_time = response.headers.get('X-Processing-Time', 'N/A')
                    width = response.headers.get('X-Output-Width', 'N/A')
                    height = response.headers.get('X-Output-Height', 'N/A')
                    
                    print(f"✓ Image saved to: {output_path}")
                    print(f"  Processing time: {processing_time}s")
                    print(f"  Output size: {width}x{height}")
                    return None
                else:
                    return image_bytes
        
        finally:
            files['file'].close()
    
    def batch_enhance(
        self,
        image_paths: list,
        output_dir: Optional[str] = None,
        scale: int = 4,
        upscale_model: str = "auto",
        face_model: str = "auto",
        face_enhance: bool = True,
        output_format: str = "png",
        quality: int = 95,
    ) -> Dict[str, Any]:
        """
        Batch enhance multiple images.
        
        Args:
            image_paths: List of input image paths
            output_dir: Directory to save output images (optional)
            scale: Upscale factor (2, 4, or 8)
            upscale_model: Upscaling model
            face_model: Face restoration model
            face_enhance: Enable face enhancement
            output_format: Output format (png, jpg, webp)
            quality: Output quality (1-100)
        
        Returns:
            Batch processing results
        """
        # Prepare files
        files = [
            ('files', (Path(path).name, open(path, 'rb'), 'image/*'))
            for path in image_paths
        ]
        
        data = {
            'scale': scale,
            'upscale_model': upscale_model,
            'face_model': face_model,
            'face_enhance': face_enhance,
            'output_format': output_format,
            'quality': quality,
        }
        
        try:
            # Send request
            response = requests.post(
                f"{self.api_url}/batch",
                files=files,
                data=data,
            )
            response.raise_for_status()
            
            result = response.json()
            
            # Save images if output directory specified
            if output_dir:
                Path(output_dir).mkdir(parents=True, exist_ok=True)
                
                for item in result.get('results', []):
                    if item.get('success'):
                        filename = item['filename']
                        image_data = base64.b64decode(item['data'])
                        
                        # Generate output filename
                        stem = Path(filename).stem
                        output_path = Path(output_dir) / f"{stem}_enhanced.{output_format}"
                        
                        with open(output_path, 'wb') as f:
                            f.write(image_data)
                        
                        print(f"✓ {filename} -> {output_path}")
            
            print(f"\n✓ Batch processing complete: {result['message']}")
            print(f"  Total time: {result.get('processing_time', 'N/A'):.2f}s")
            
            return result
        
        finally:
            # Close all file handles
            for _, file_tuple in files:
                file_tuple[1].close()


def example_basic():
    """Basic usage example."""
    print("=" * 60)
    print("Example 1: Basic Image Enhancement")
    print("=" * 60)
    
    client = VisionRestoreClient("http://localhost:8000")
    
    # Check service status
    print("\n1. Checking service status...")
    status = client.check_status()
    print(f"   Status: {status['status']}")
    print(f"   GPU Available: {status['gpu_available']}")
    if status['gpu_device']:
        print(f"   GPU Device: {status['gpu_device']}")
    
    # Get available models
    print("\n2. Getting available models...")
    models = client.get_models()
    print(f"   Upscale models: {', '.join(models['upscale_models'])}")
    print(f"   Face models: {', '.join(models['face_models'])}")
    
    # Enhance an image
    print("\n3. Enhancing image...")
    client.enhance_image(
        image_path="input.jpg",
        output_path="output_enhanced.png",
        scale=4,
        upscale_model="auto",
        face_enhance=True,
    )


def example_advanced():
    """Advanced usage with custom parameters."""
    print("=" * 60)
    print("Example 2: Advanced Enhancement with Custom Parameters")
    print("=" * 60)
    
    client = VisionRestoreClient("http://localhost:8000")
    
    # Portrait enhancement with CodeFormer
    print("\nEnhancing portrait with CodeFormer...")
    client.enhance_image(
        image_path="portrait.jpg",
        output_path="portrait_restored.png",
        scale=4,
        upscale_model="realesrgan",
        face_model="codeformer",
        face_enhance=True,
        codeformer_fidelity=0.7,  # Balance between quality and fidelity
        enable_denoise=True,
        denoise_strength=0.5,
        enable_sharpen=True,
        sharpen_amount=0.3,
        output_format="png",
        quality=95,
    )


def example_batch():
    """Batch processing example."""
    print("=" * 60)
    print("Example 3: Batch Image Processing")
    print("=" * 60)
    
    client = VisionRestoreClient("http://localhost:8000")
    
    # Process multiple images
    image_files = [
        "image1.jpg",
        "image2.jpg",
        "image3.jpg",
    ]
    
    print(f"\nProcessing {len(image_files)} images...")
    result = client.batch_enhance(
        image_paths=image_files,
        output_dir="batch_output",
        scale=4,
        upscale_model="auto",
        face_enhance=True,
    )


def example_quality_modes():
    """Different quality modes example."""
    print("=" * 60)
    print("Example 4: Different Quality Modes")
    print("=" * 60)
    
    client = VisionRestoreClient("http://localhost:8000")
    
    # Fast mode (Real-ESRGAN only)
    print("\n1. Fast mode (Real-ESRGAN)...")
    client.enhance_image(
        image_path="input.jpg",
        output_path="output_fast.png",
        scale=4,
        upscale_model="realesrgan",
        face_enhance=False,
        enable_denoise=False,
        enable_sharpen=False,
    )
    
    # Quality mode (SwinIR with processing)
    print("\n2. Quality mode (SwinIR)...")
    client.enhance_image(
        image_path="input.jpg",
        output_path="output_quality.png",
        scale=4,
        upscale_model="swinir",
        face_enhance=True,
        enable_denoise=True,
        enable_sharpen=True,
    )
    
    # Maximum quality (HAT with all features)
    print("\n3. Maximum quality mode (HAT)...")
    client.enhance_image(
        image_path="input.jpg",
        output_path="output_max.png",
        scale=4,
        upscale_model="hat",
        face_enhance=True,
        face_model="codeformer",
        enable_denoise=True,
        denoise_strength=0.6,
        enable_sharpen=True,
        sharpen_amount=0.5,
    )


if __name__ == "__main__":
    print("\nVision-Restore AI - Python Client Examples\n")
    
    # Run examples (uncomment the ones you want to try)
    
    # example_basic()
    # example_advanced()
    # example_batch()
    # example_quality_modes()
    
    print("\nNote: Make sure the API service is running before running these examples!")
    print("Start the service with: docker-compose --profile cpu up")
