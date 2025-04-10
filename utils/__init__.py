"""
Utility modules for ANPR system.
Contains common functionality used across the system.
"""
from utils.hardware import HardwareDetector, AcceleratorType
from utils.logger import setup_logging, get_logger
from utils.plate_checker import PlateChecker
from utils.gpio import get_gpio, GPIOMode, GPIOState
from utils.vehicle_color import ColorDetector
from utils.vehicle_make import MakeDetector

__all__ = [
    # Hardware detection
    'HardwareDetector',
    'AcceleratorType',
    
    # Logging
    'setup_logging',
    'get_logger',
    
    # Plate handling
    'PlateChecker',
    
    # GPIO control
    'get_gpio',
    'GPIOMode',
    'GPIOState',
    
    # Vehicle feature detection
    'ColorDetector',
    'MakeDetector'
]
