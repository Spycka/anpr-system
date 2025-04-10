"""
ANPR System for Orange Pi 5 Ultra

A modular, real-time Automatic Number Plate Recognition (ANPR) system 
designed specifically for the Orange Pi 5 Ultra hardware.
"""

# Define version info
__version__ = '1.0.0'
__author__ = 'Your Name'
__license__ = 'MIT'

# Import primary modules for easier access
from detectors import YOLO11sGPU, YOLO11sRKNN
from vision import PlateOCR, SkewCorrector
from input import CameraInput
from utils import (
    HardwareDetector, AcceleratorType,
    setup_logging, get_logger,
    PlateChecker,
    get_gpio, GPIOMode, GPIOState,
    ColorDetector, MakeDetector
)

# Define public API
__all__ = [
    # Main modules
    'YOLO11sGPU', 'YOLO11sRKNN',
    'PlateOCR', 'SkewCorrector',
    'CameraInput',
    
    # Utils
    'HardwareDetector', 'AcceleratorType',
    'setup_logging', 'get_logger',
    'PlateChecker',
    'get_gpio', 'GPIOMode', 'GPIOState',
    'ColorDetector', 'MakeDetector',
    
    # Version info
    '__version__', '__author__', '__license__'
]
