#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
License Plate Deskewing Module
Performs skew correction for license plate images

This module handles detection and correction of skewed license plates
using projection profiles and Radon transform methods.
"""

import cv2
import numpy as np
import logging
from math import degrees, radians

# Initialize logger
logger = logging.getLogger('ocr')

def find_skew_radon(img):
    """
    Find skew angle using Radon transform
    
    Args:
        img: Input image (grayscale)
        
    Returns:
        Skew angle in degrees
    """
    try:
        # Convert to binary if not already
        if len(img.shape) > 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Ensure image is binary
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find all non-zero points
        pts = np.column_stack(np.where(binary > 0))
        
        # Safety check for empty image
        if len(pts) < 10:
            return 0
        
        # Fit line to points
        [vx, vy, x, y] = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
        
        # Calculate angle
        slope = vy / vx if vx != 0 else float('inf')
        angle = degrees(np.arctan(slope))
        
        # Limit angle to reasonable range for license plates
        if angle < -45:
            angle += 90
        elif angle > 45:
            angle -= 90
        
        return angle
        
    except Exception as e:
        logger.error(f"Error in skew detection: {str(e)}")
        return 0


def find_skew_projection(img):
    """
    Find skew angle using horizontal projection profiles
    
    Args:
        img: Input image (grayscale)
        
    Returns:
        Skew angle in degrees
    """
    try:
        # Convert to binary if not already
        if len(img.shape) > 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Ensure image is binary
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Define range of angles to try
        angles = np.arange(-15, 15, 0.5)
        scores = []
        
        h, w = binary.shape
        center = (w // 2, h // 2)
        
        for angle in angles:
            # Rotate image
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(binary, M, (w, h), flags=cv2.INTER_CUBIC, 
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            
            # Calculate horizontal projection profile
            projection = np.sum(rotated, axis=1)
            
            # Calculate score as the standard deviation of the projection
            # Higher standard deviation means more distinct rows of text
            score = np.std(projection)
            scores.append(score)
        
        # Find angle with maximum score
        best_angle = angles[np.argmax(scores)]
        return best_angle
        
    except Exception as e:
        logger.error(f"Error in projection-based skew detection: {str(e)}")
        return 0


def deskew_plate(img):
    """
    Deskew license plate image
    
    Args:
        img: Input image (grayscale)
        
    Returns:
        Deskewed image
    """
    try:
        # Make a copy of input
        img_copy = img.copy()
        
        # Find skew angle using both methods
        angle_radon = find_skew_radon(img_copy)
        angle_projection = find_skew_projection(img_copy)
        
        # Use Radon method for larger angles, projection for fine-tuning
        angle = angle_radon if abs(angle_radon) > 5 else angle_projection
        
        # Skip rotation if angle is very small
        if abs(angle) < 0.5:
            return img_copy
        
        logger.debug(f"Deskewing plate with angle: {angle:.2f} degrees")
        
        # Rotate image to correct skew
        h, w = img_copy.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img_copy, M, (w, h), flags=cv2.INTER_CUBIC, 
                                 borderMode=cv2.BORDER_REPLICATE)
        
        return rotated
        
    except Exception as e:
        logger.error(f"Error in plate deskewing: {str(e)}")
        return img  # Return original if deskewing fails
