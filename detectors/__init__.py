"""
License plate detector module for ANPR system.
Contains different detector implementations for license plate detection.
"""
from detectors.yolo11_gpu import YOLO11sGPU
from detectors.yolo11_rknn import YOLO11sRKNN

__all__ = ['YOLO11sGPU', 'YOLO11sRKNN']
