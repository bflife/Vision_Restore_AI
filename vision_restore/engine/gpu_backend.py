
"""
Vision-Restore AI - GPU Backend Manager

Handles detection and configuration of GPU backends (CUDA, DirectML, CPU).
"""

import os
import sys
from typing import Optional, List, Dict, Any, Union
import logging

# Configure logger locally to avoid circular imports if logger depends on this
logger = logging.getLogger(__name__)

class GPUBackendManager:
    """Manages GPU backend detection and device selection."""
    
    def __init__(self):
        self._cuda_available = False
        self._directml_available = False
        self._directml_device_name = "privateuseone:0" # Default for torch-directml
        self._checked = False
        self._detect_backends()
        
    def _detect_backends(self):
        """Detect available GPU backends."""
        if self._checked:
            return
            
        # Check CUDA
        try:
            import torch
            if torch.cuda.is_available():
                self._cuda_available = True
                logger.info("CUDA backend detected")
        except ImportError:
            pass
            
        # Check DirectML (for AMD/Intel on Windows)
        try:
            # Check ONNX Runtime DirectML (for ONNX models)
            import onnxruntime as ort
            providers = ort.get_available_providers()
            if 'DmlExecutionProvider' in providers:
                # Also check torch-directml for PyTorch models
                try:
                    import torch_directml
                    self._directml_available = True
                    self._directml_device_name = torch_directml.device().type + ":" + str(torch_directml.device().index)
                    logger.info(f"DirectML backend detected (torch-directml device: {self._directml_device_name})")
                except ImportError:
                    logger.info("DirectML available in ONNX but torch-directml not installed")
        except ImportError:
            pass
            
        self._checked = True
        
    def get_torch_device(self, gpu_id: int = 0) -> Any:
        """Get PyTorch device based on availability and config.
        
        Args:
            gpu_id: GPU ID to use (-1 for CPU)
            
        Returns:
            torch.device object
        """
        import torch
        
        if gpu_id < 0:
            return torch.device("cpu")
            
        if self._cuda_available:
            try:
                device = torch.device(f"cuda:{gpu_id}")
                # Verify it works
                torch.cuda.get_device_name(device)
                return device
            except Exception as e:
                logger.warning(f"Failed to create CUDA device {gpu_id}: {e}")
                
        if self._directml_available:
            try:
                import torch_directml
                return torch_directml.device(gpu_id)
            except Exception as e:
                logger.warning(f"Failed to create DirectML device: {e}")
                
        # Fallback
        logger.warning("No GPU backend available, falling back to CPU")
        return torch.device("cpu")
        
    def get_torch_device_for_model(self, model_type: str, gpu_id: int = 0) -> Any:
        """Get appropriate device for a specific model type.
        
        Some models might have specific requirements or known issues with certain backends.
        
        Args:
            model_type: Type of model ('swinir', 'hat', 'realesrgan', etc.)
            gpu_id: GPU ID (-1 for CPU)
            
        Returns:
            torch.device object
        """
        # Feature: logic to force CPU for specific problematic models if needed
        # For now, just return standard device
        return self.get_torch_device(gpu_id)

    @property
    def is_cuda_available(self) -> bool:
        return self._cuda_available
        
    @property
    def is_directml_available(self) -> bool:
        return self._directml_available

# Singleton instance
gpu_backend = GPUBackendManager()
