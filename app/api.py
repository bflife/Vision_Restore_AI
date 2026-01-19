"""
Vision-Restore AI - FastAPI REST API v1.0

Provides RESTful endpoints for image restoration and upscaling services.

Endpoints:
- POST /api/v1/enhance - Process single image
- POST /api/v1/batch - Process multiple images
- GET /api/v1/status - Check service status
- GET /api/v1/models - Get available models
"""

import io
import base64
import logging
from typing import Optional, List, Dict, Any
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np
from PIL import Image
import cv2

from vision_restore.engine.orchestrator import EnhancementOrchestrator, get_available_models
from vision_restore.core.config import Config, get_config
from vision_restore.core.logger import get_logger

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = get_logger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Vision-Restore AI API",
    description="Professional Image Restoration & Upscaling Service",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this appropriately in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global orchestrator instance (singleton pattern)
_orchestrator: Optional[EnhancementOrchestrator] = None


def get_orchestrator() -> EnhancementOrchestrator:
    """Get or create the enhancement orchestrator singleton."""
    global _orchestrator
    if _orchestrator is None:
        config = get_config()
        _orchestrator = EnhancementOrchestrator(config)
        logger.info("Enhancement orchestrator initialized")
    return _orchestrator


# ============================================================================
# Pydantic Models (Request/Response Schemas)
# ============================================================================

class EnhanceRequest(BaseModel):
    """Image enhancement request parameters."""
    scale: int = Field(default=4, ge=2, le=8, description="Upscale factor (2, 4, or 8)")
    upscale_model: str = Field(default="auto", description="Upscaling model: auto, realesrgan, swinir, hat")
    face_model: str = Field(default="auto", description="Face restoration model: auto, gfpgan, codeformer, none")
    face_enhance: bool = Field(default=True, description="Enable face enhancement")
    codeformer_fidelity: float = Field(default=0.7, ge=0.0, le=1.0, description="CodeFormer fidelity (0=quality, 1=fidelity)")
    enable_denoise: bool = Field(default=True, description="Enable denoising")
    enable_sharpen: bool = Field(default=True, description="Enable sharpening")
    denoise_strength: float = Field(default=0.3, ge=0.0, le=1.0, description="Denoise strength")
    sharpen_amount: float = Field(default=0.2, ge=0.0, le=1.0, description="Sharpen amount")
    output_format: str = Field(default="png", description="Output format: png, jpg, webp")
    quality: int = Field(default=95, ge=1, le=100, description="Output quality (for jpg/webp)")


class EnhanceResponse(BaseModel):
    """Image enhancement response."""
    success: bool
    message: str
    output_size: Optional[Dict[str, int]] = None
    processing_time: Optional[float] = None
    model_used: Optional[str] = None
    face_count: Optional[int] = None


class BatchEnhanceRequest(BaseModel):
    """Batch image enhancement request."""
    scale: int = Field(default=4, ge=2, le=8)
    upscale_model: str = Field(default="auto")
    face_model: str = Field(default="auto")
    face_enhance: bool = Field(default=True)
    output_format: str = Field(default="png")


class StatusResponse(BaseModel):
    """Service status response."""
    status: str
    version: str
    available_models: Dict[str, List[str]]
    gpu_available: bool
    gpu_device: Optional[str] = None


class ModelInfo(BaseModel):
    """Model information."""
    upscale_models: List[str]
    face_models: List[str]


# ============================================================================
# Utility Functions
# ============================================================================

def decode_image(file_bytes: bytes) -> np.ndarray:
    """Decode uploaded image file to numpy array.
    
    Args:
        file_bytes: Raw image file bytes
        
    Returns:
        Image as numpy array (BGR format for OpenCV)
    """
    try:
        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(file_bytes))
        
        # Convert to RGB if necessary
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Convert to numpy array and BGR format (OpenCV convention)
        image_np = np.array(image)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        return image_bgr
    except Exception as e:
        logger.error(f"Error decoding image: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")


def encode_image(image: np.ndarray, format: str = "png", quality: int = 95) -> bytes:
    """Encode numpy array to image bytes.
    
    Args:
        image: Image as numpy array (BGR format)
        format: Output format (png, jpg, webp)
        quality: Output quality for jpg/webp
        
    Returns:
        Encoded image bytes
    """
    try:
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        # Encode to bytes
        buffer = io.BytesIO()
        
        if format.lower() == "png":
            pil_image.save(buffer, format="PNG", optimize=True)
        elif format.lower() in ["jpg", "jpeg"]:
            pil_image.save(buffer, format="JPEG", quality=quality, optimize=True)
        elif format.lower() == "webp":
            pil_image.save(buffer, format="WEBP", quality=quality)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        buffer.seek(0)
        return buffer.getvalue()
    except Exception as e:
        logger.error(f"Error encoding image: {e}")
        raise HTTPException(status_code=500, detail=f"Image encoding failed: {str(e)}")


def image_to_base64(image: np.ndarray, format: str = "png", quality: int = 95) -> str:
    """Convert numpy array to base64 encoded string.
    
    Args:
        image: Image as numpy array
        format: Output format
        quality: Output quality
        
    Returns:
        Base64 encoded image string
    """
    image_bytes = encode_image(image, format, quality)
    return base64.b64encode(image_bytes).decode('utf-8')


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "service": "Vision-Restore AI API",
        "version": "1.0.0",
        "status": "running",
        "documentation": "/docs",
        "endpoints": {
            "enhance": "/api/v1/enhance",
            "batch": "/api/v1/batch",
            "status": "/api/v1/status",
            "models": "/api/v1/models",
        }
    }


@app.get("/api/v1/status", response_model=StatusResponse, tags=["Service"])
async def get_status():
    """Get service status and available models.
    
    Returns:
        StatusResponse: Service status information
    """
    try:
        orchestrator = get_orchestrator()
        config = orchestrator.config
        
        # Get available models
        available = get_available_models()
        
        # Check GPU availability
        gpu_available = False
        gpu_device = None
        try:
            import torch
            gpu_available = torch.cuda.is_available()
            if gpu_available:
                gpu_device = torch.cuda.get_device_name(0)
        except:
            pass
        
        return StatusResponse(
            status="healthy",
            version="1.0.0",
            available_models=available,
            gpu_available=gpu_available,
            gpu_device=gpu_device,
        )
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/models", response_model=ModelInfo, tags=["Service"])
async def get_models():
    """Get available AI models.
    
    Returns:
        ModelInfo: Available models information
    """
    try:
        available = get_available_models()
        return ModelInfo(
            upscale_models=available.get("upscale_models", []),
            face_models=available.get("face_models", []),
        )
    except Exception as e:
        logger.error(f"Error getting models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/enhance", tags=["Processing"])
async def enhance_image(
    file: UploadFile = File(..., description="Image file to enhance"),
    scale: int = Form(4, description="Upscale factor (2, 4, or 8)"),
    upscale_model: str = Form("auto", description="Upscaling model"),
    face_model: str = Form("auto", description="Face restoration model"),
    face_enhance: bool = Form(True, description="Enable face enhancement"),
    codeformer_fidelity: float = Form(0.7, description="CodeFormer fidelity"),
    enable_denoise: bool = Form(True, description="Enable denoising"),
    enable_sharpen: bool = Form(True, description="Enable sharpening"),
    denoise_strength: float = Form(0.3, description="Denoise strength"),
    sharpen_amount: float = Form(0.2, description="Sharpen amount"),
    output_format: str = Form("png", description="Output format"),
    quality: int = Form(95, description="Output quality"),
    return_base64: bool = Form(False, description="Return base64 encoded image"),
):
    """
    Enhance a single image with AI upscaling and restoration.
    
    **Parameters:**
    - **file**: Image file (png, jpg, webp, etc.)
    - **scale**: Upscale factor (2, 4, or 8)
    - **upscale_model**: Model for upscaling (auto, realesrgan, swinir, hat)
    - **face_model**: Model for face restoration (auto, gfpgan, codeformer, none)
    - **face_enhance**: Enable face enhancement
    - **codeformer_fidelity**: CodeFormer fidelity (0=quality, 1=fidelity)
    - **enable_denoise**: Enable denoising preprocessing
    - **enable_sharpen**: Enable sharpening postprocessing
    - **output_format**: Output format (png, jpg, webp)
    - **return_base64**: Return base64 encoded image instead of binary
    
    **Returns:**
    - Enhanced image file (binary) or JSON with base64 encoded image
    """
    import time
    start_time = time.time()
    
    try:
        logger.info(f"Processing image: {file.filename}, scale={scale}, model={upscale_model}")
        
        # Read and decode input image
        file_bytes = await file.read()
        input_image = decode_image(file_bytes)
        
        logger.info(f"Input image size: {input_image.shape[:2]}")
        
        # Get orchestrator and configure
        orchestrator = get_orchestrator()
        config = orchestrator.config
        
        # Update configuration
        config.scale = scale
        config.enable_denoise = enable_denoise
        config.enable_sharpen = enable_sharpen
        config.denoise_strength = denoise_strength
        config.sharpen_amount = sharpen_amount
        config.enable_face_enhance = face_enhance
        config.codeformer_fidelity = codeformer_fidelity
        
        # Progress tracking
        progress_data = {"progress": 0.0, "message": "Starting..."}
        
        def progress_callback(progress: float, message: str):
            progress_data["progress"] = progress
            progress_data["message"] = message
            logger.debug(f"Progress: {progress:.1%} - {message}")
        
        # Run enhancement
        output_image = orchestrator.enhance(
            image=input_image,
            scale=scale,
            face_enhance=face_enhance,
            preprocess=enable_denoise,
            postprocess=enable_sharpen,
            progress_callback=progress_callback,
            upscale_model=upscale_model,
            face_model=face_model,
        )
        
        processing_time = time.time() - start_time
        output_h, output_w = output_image.shape[:2]
        
        logger.info(f"Enhancement completed in {processing_time:.2f}s, output size: {output_w}x{output_h}")
        
        # Return base64 encoded or binary image
        if return_base64:
            image_base64 = image_to_base64(output_image, output_format, quality)
            return JSONResponse({
                "success": True,
                "message": "Image enhanced successfully",
                "data": image_base64,
                "output_size": {"width": output_w, "height": output_h},
                "processing_time": processing_time,
                "model_used": upscale_model,
            })
        else:
            # Return binary image
            image_bytes = encode_image(output_image, output_format, quality)
            
            # Determine content type
            content_types = {
                "png": "image/png",
                "jpg": "image/jpeg",
                "jpeg": "image/jpeg",
                "webp": "image/webp",
            }
            content_type = content_types.get(output_format.lower(), "image/png")
            
            return StreamingResponse(
                io.BytesIO(image_bytes),
                media_type=content_type,
                headers={
                    "Content-Disposition": f"attachment; filename=enhanced.{output_format}",
                    "X-Processing-Time": str(processing_time),
                    "X-Output-Width": str(output_w),
                    "X-Output-Height": str(output_h),
                }
            )
    
    except Exception as e:
        logger.error(f"Enhancement failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Enhancement failed: {str(e)}")


@app.post("/api/v1/batch", tags=["Processing"])
async def batch_enhance(
    files: List[UploadFile] = File(..., description="Image files to enhance"),
    scale: int = Form(4),
    upscale_model: str = Form("auto"),
    face_model: str = Form("auto"),
    face_enhance: bool = Form(True),
    output_format: str = Form("png"),
    quality: int = Form(95),
):
    """
    Batch enhance multiple images.
    
    **Parameters:**
    - **files**: Multiple image files
    - **scale**: Upscale factor (2, 4, or 8)
    - **upscale_model**: Model for upscaling
    - **face_model**: Model for face restoration
    - **face_enhance**: Enable face enhancement
    - **output_format**: Output format (png, jpg, webp)
    
    **Returns:**
    - JSON with base64 encoded enhanced images
    """
    import time
    start_time = time.time()
    
    try:
        logger.info(f"Batch processing {len(files)} images")
        
        results = []
        orchestrator = get_orchestrator()
        
        for idx, file in enumerate(files):
            try:
                logger.info(f"Processing file {idx+1}/{len(files)}: {file.filename}")
                
                # Read and decode
                file_bytes = await file.read()
                input_image = decode_image(file_bytes)
                
                # Enhance
                output_image = orchestrator.enhance(
                    image=input_image,
                    scale=scale,
                    face_enhance=face_enhance,
                    upscale_model=upscale_model,
                    face_model=face_model,
                )
                
                # Encode to base64
                image_base64 = image_to_base64(output_image, output_format, quality)
                
                results.append({
                    "filename": file.filename,
                    "success": True,
                    "data": image_base64,
                    "output_size": {
                        "width": output_image.shape[1],
                        "height": output_image.shape[0]
                    }
                })
                
            except Exception as e:
                logger.error(f"Failed to process {file.filename}: {e}")
                results.append({
                    "filename": file.filename,
                    "success": False,
                    "error": str(e)
                })
        
        processing_time = time.time() - start_time
        success_count = sum(1 for r in results if r.get("success", False))
        
        return JSONResponse({
            "success": True,
            "message": f"Processed {success_count}/{len(files)} images successfully",
            "processing_time": processing_time,
            "results": results,
        })
    
    except Exception as e:
        logger.error(f"Batch processing failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Batch processing failed: {str(e)}")


@app.get("/health", tags=["Service"])
async def health_check():
    """Health check endpoint for container orchestration."""
    return {"status": "healthy", "service": "vision-restore-ai"}


# ============================================================================
# Application Lifecycle
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logger.info("Starting Vision-Restore AI API service...")
    
    # Pre-initialize orchestrator
    try:
        orchestrator = get_orchestrator()
        logger.info("Orchestrator initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize orchestrator: {e}")
    
    logger.info("API service is ready to accept requests")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down Vision-Restore AI API service...")
    
    global _orchestrator
    if _orchestrator is not None:
        _orchestrator.cleanup()
        _orchestrator = None
    
    logger.info("Cleanup completed")


if __name__ == "__main__":
    import uvicorn
    
    # Run with uvicorn
    uvicorn.run(
        "app.api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
