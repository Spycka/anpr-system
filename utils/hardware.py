#!/usr/bin/env python3
"""
Hardware detection module for Orange Pi 5 Ultra.
Detects and selects the optimal hardware acceleration (NPU, GPU, CPU).
"""
import os
import sys
import logging
import subprocess
import importlib.util
from enum import Enum, auto
from typing import Dict, Any, Tuple

# Configure logger
logger = logging.getLogger("hardware_detection")

class AcceleratorType(Enum):
    """Types of available hardware accelerators."""
    NPU = auto()  # RK3588 NPU via RKNN
    GPU = auto()  # Mali-G610 MP4 via OpenCL/OpenGL
    CPU = auto()  # Fallback to CPU

class HardwareDetector:
    """Detects and selects optimal hardware for ANPR tasks."""
    
    def __init__(self):
        self.capabilities = {
            AcceleratorType.NPU: False,
            AcceleratorType.GPU: False,
            AcceleratorType.CPU: True  # CPU is always available
        }
        self.device_info = {}
        self._detect_capabilities()
    
    def _detect_capabilities(self) -> None:
        """Detect available hardware acceleration capabilities."""
        # Check for NPU (RKNN) availability
        self._check_rknn_availability()
        
        # Check for GPU (OpenCL) availability
        self._check_gpu_availability()
        
        # Log detected capabilities
        logger.info(f"Hardware capabilities: NPU={self.capabilities[AcceleratorType.NPU]}, "
                   f"GPU={self.capabilities[AcceleratorType.GPU]}, "
                   f"CPU={self.capabilities[AcceleratorType.CPU]}")
        
        # Log detailed device info if available
        for k, v in self.device_info.items():
            logger.debug(f"Device info - {k}: {v}")
    
    def _check_rknn_availability(self) -> None:
        """Check if RKNN (NPU) is available."""
        try:
            # Check if rknn_toolkit_lite is installed
            if importlib.util.find_spec("rknnlite") is not None:
                from rknnlite.api import RKNNLite
                
                # Create RKNN-Lite runtime environment
                rknn_lite = RKNNLite()
                
                # Try to get NPU information
                try:
                    # Different methods to check for NPU based on the version
                    # These might change based on the specific SDK version
                    if hasattr(rknn_lite, 'get_sdk_version'):
                        version_info = rknn_lite.get_sdk_version()
                        self.device_info['rknn_version'] = version_info
                        self.capabilities[AcceleratorType.NPU] = True
                        logger.info(f"RKNN SDK version: {version_info}")
                    else:
                        # Alternative method for older versions
                        self.capabilities[AcceleratorType.NPU] = True
                        logger.info("RKNN detected (version unknown)")
                except Exception as e:
                    logger.warning(f"RKNN initialization error: {e}")
                    self.capabilities[AcceleratorType.NPU] = False
            else:
                logger.info("RKNN toolkit not installed")
                self.capabilities[AcceleratorType.NPU] = False
        except Exception as e:
            logger.warning(f"Error checking RKNN availability: {e}")
            self.capabilities[AcceleratorType.NPU] = False
    
    def _check_gpu_availability(self) -> None:
        """Check GPU (Mali) availability via OpenCL."""
        try:
            # Check if pyopencl is installed
            if importlib.util.find_spec("pyopencl") is not None:
                import pyopencl as cl
                try:
                    # Get OpenCL platforms
                    platforms = cl.get_platforms()
                    
                    if platforms:
                        # Store platform information
                        self.device_info['opencl_platforms'] = []
                        gpu_available = False
                        
                        for platform in platforms:
                            platform_info = {
                                'name': platform.name,
                                'vendor': platform.vendor,
                                'version': platform.version,
                                'devices': []
                            }
                            
                            try:
                                # Get devices for this platform
                                devices = platform.get_devices()
                                
                                for device in devices:
                                    device_info = {
                                        'name': device.name,
                                        'type': cl.device_type.to_string(device.type),
                                        'vendor': device.vendor,
                                        'max_compute_units': device.max_compute_units
                                    }
                                    platform_info['devices'].append(device_info)
                                    
                                    # Check for Mali GPU
                                    if (device.type == cl.device_type.GPU and 
                                        ('Mali' in device.name or 'ARM' in device.vendor)):
                                        gpu_available = True
                                        logger.info(f"Mali GPU detected: {device.name}")
                            except cl.LogicError as e:
                                logger.warning(f"OpenCL error getting devices: {e}")
                                
                            self.device_info['opencl_platforms'].append(platform_info)
                        
                        self.capabilities[AcceleratorType.GPU] = gpu_available
                    else:
                        logger.info("No OpenCL platforms found")
                        self.capabilities[AcceleratorType.GPU] = False
                except cl.LogicError as e:
                    logger.warning(f"OpenCL error: {e}")
                    self.capabilities[AcceleratorType.GPU] = False
            else:
                # Check via command line if we can't use pyopencl
                self._check_gpu_via_command()
        except Exception as e:
            logger.warning(f"Error checking GPU availability: {e}")
            self.capabilities[AcceleratorType.GPU] = False
    
    def _check_gpu_via_command(self) -> None:
        """Fallback method to check GPU via command line."""
        try:
            # Try to detect Mali GPU via device info
            result = subprocess.run(
                "cat /proc/device-tree/compatible", 
                shell=True, 
                capture_output=True, 
                text=True
            )
            output = result.stdout.lower()
            
            # Check for Mali GPU in device tree
            if 'mali' in output or 'g610' in output:
                logger.info("Mali GPU detected via device tree")
                self.capabilities[AcceleratorType.GPU] = True
                return
            
            # Check via lspci or other commands if available
            result = subprocess.run(
                "command -v lspci && lspci | grep -i 'mali\|arm\|gpu'", 
                shell=True, 
                capture_output=True, 
                text=True
            )
            
            if 'mali' in result.stdout.lower() or 'gpu' in result.stdout.lower():
                logger.info("GPU detected via lspci")
                self.capabilities[AcceleratorType.GPU] = True
                return
                
            logger.info("No GPU detected via command line")
            self.capabilities[AcceleratorType.GPU] = False
        except Exception as e:
            logger.warning(f"Error in command line GPU detection: {e}")
            self.capabilities[AcceleratorType.GPU] = False
    
    def get_best_available_accelerator(self) -> Tuple[AcceleratorType, Dict[str, Any]]:
        """
        Returns the best available accelerator in preference order: NPU > GPU > CPU.
        
        Returns:
            Tuple[AcceleratorType, Dict[str, Any]]: The best accelerator type and device info
        """
        if self.capabilities[AcceleratorType.NPU]:
            return AcceleratorType.NPU, {'type': 'NPU', 'details': self.device_info.get('rknn_version', 'Unknown')}
        elif self.capabilities[AcceleratorType.GPU]:
            return AcceleratorType.GPU, {'type': 'GPU', 'details': self.device_info.get('opencl_platforms', 'Unknown')}
        else:
            return AcceleratorType.CPU, {'type': 'CPU', 'details': 'Fallback mode'}
    
    def get_capabilities(self) -> Dict[AcceleratorType, bool]:
        """
        Returns the capability status of each accelerator type.
        
        Returns:
            Dict[AcceleratorType, bool]: Dictionary of accelerator types and their availability
        """
        return self.capabilities.copy()

# For direct testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize detector
    detector = HardwareDetector()
    
    # Get best accelerator
    accelerator, info = detector.get_best_available_accelerator()
    
    print(f"Best available accelerator: {accelerator.name}")
    print(f"Details: {info}")
    print("\nAll capabilities:")
    for acc_type, available in detector.get_capabilities().items():
        print(f"  {acc_type.name}: {'Available' if available else 'Not available'}")
