#!/usr/bin/env python3
"""
Skew correction module for license plate images.
Uses projection profile methods to detect and correct skew angles.
"""
import math
import numpy as np
import cv2
from typing import Tuple, Optional, List, Dict, Any
import logging

# Import logger
from utils.logger import get_logger
logger = get_logger("skew")

class SkewCorrector:
    """
    Skew correction for license plate images.
    
    Methods:
    - Projection profile analysis
    - Hough line transform
    - Contour-based orientation
    """
    
    def __init__(self, 
                 max_angle: float = 30.0,
                 angle_step: float = 0.5,
                 min_confidence: float = 0.6):
        """
        Initialize skew corrector.
        
        Args:
            max_angle: Maximum angle to check (degrees)
            angle_step: Angle step size for projection profile method
            min_confidence: Minimum confidence score for correction
        """
        self.max_angle = max_angle
        self.angle_step = angle_step
        self.min_confidence = min_confidence
        
        logger.debug(f"Skew corrector initialized (max_angle={max_angle}, "
                    f"angle_step={angle_step}, min_confidence={min_confidence})")
    
    def correct(self, image: np.ndarray, method: str = "auto") -> Tuple[np.ndarray, float, float]:
        """
        Correct skew in license plate image.
        
        Args:
            image: Input image (grayscale or BGR)
            method: Correction method ("profile", "hough", "contour", or "auto")
            
        Returns:
            Tuple[np.ndarray, float, float]: 
                - Corrected image
                - Detected skew angle (degrees)
                - Confidence score (0-1)
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Normalize image size for consistent processing
        height, width = gray.shape
        if width > 600:
            scale_factor = 600 / width
            new_width = 600
            new_height = int(height * scale_factor)
            gray = cv2.resize(gray, (new_width, new_height))
        
        # Apply threshold to binarize
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Determine method to use
        if method == "auto":
            # Start with projection profile method
            corrected, angle, confidence = self._correct_projection_profile(binary)
            
            # If confidence is low, try Hough lines
            if confidence < self.min_confidence:
                logger.debug(f"Low confidence with profile method ({confidence:.2f}), trying Hough lines")
                corrected_hough, angle_hough, confidence_hough = self._correct_hough_lines(binary)
                
                # Use result with higher confidence
                if confidence_hough > confidence:
                    logger.debug(f"Using Hough method (confidence: {confidence_hough:.2f})")
                    corrected, angle, confidence = corrected_hough, angle_hough, confidence_hough
        elif method == "profile":
            corrected, angle, confidence = self._correct_projection_profile(binary)
        elif method == "hough":
            corrected, angle, confidence = self._correct_hough_lines(binary)
        elif method == "contour":
            corrected, angle, confidence = self._correct_contour(binary)
        else:
            logger.warning(f"Unknown method '{method}', falling back to auto")
            return self.correct(image, "auto")
        
        # Resize corrected image to match input dimensions if needed
        if corrected.shape[:2] != image.shape[:2]:
            if len(image.shape) == 3:  # Color image
                if len(corrected.shape) == 2:  # If output is grayscale but input was color
                    corrected = cv2.cvtColor(corrected, cv2.COLOR_GRAY2BGR)
                corrected = cv2.resize(corrected, (image.shape[1], image.shape[0]))
            else:  # Grayscale
                if len(corrected.shape) == 3:  # If output is color but input was grayscale
                    corrected = cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY)
                corrected = cv2.resize(corrected, (image.shape[1], image.shape[0]))
        
        return corrected, angle, confidence
    
    def _correct_projection_profile(self, binary: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """
        Correct skew using projection profile method.
        
        Args:
            binary: Binarized input image
            
        Returns:
            Tuple[np.ndarray, float, float]:
                - Corrected image
                - Detected angle
                - Confidence score
        """
        height, width = binary.shape
        
        # Calculate inverted image (text is white)
        binary_inv = cv2.bitwise_not(binary)
        
        best_angle = 0.0
        best_variance = 0.0
        best_profile = None
        
        # Test different angles
        angle_range = np.arange(-self.max_angle, self.max_angle + self.angle_step, self.angle_step)
        
        for angle in angle_range:
            # Create rotation matrix
            M = cv2.getRotationMatrix2D((width/2, height/2), angle, 1)
            
            # Rotate image
            rotated = cv2.warpAffine(binary_inv, M, (width, height), 
                                    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            
            # Calculate horizontal projection profile
            profile = np.sum(rotated, axis=1) / 255  # Convert to pixel count
            
            # Calculate variance of profile
            variance = np.var(profile)
            
            # Update best angle if variance is higher
            if variance > best_variance:
                best_variance = variance
                best_angle = angle
                best_profile = profile
        
        logger.debug(f"Best angle from projection profile: {best_angle:.2f}°")
        
        # Calculate confidence based on improvement in variance
        baseline_variance = np.var(np.sum(binary_inv, axis=1) / 255)
        if baseline_variance > 0:
            confidence = min(1.0, best_variance / baseline_variance)
        else:
            confidence = 0.5  # Default if baseline variance is too low
        
        # Apply the best rotation to the original image
        M = cv2.getRotationMatrix2D((width/2, height/2), best_angle, 1)
        corrected = cv2.warpAffine(binary, M, (width, height), 
                                  flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        
        return corrected, best_angle, confidence
    
    def _correct_hough_lines(self, binary: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """
        Correct skew using Hough line transform.
        
        Args:
            binary: Binarized input image
            
        Returns:
            Tuple[np.ndarray, float, float]:
                - Corrected image
                - Detected angle
                - Confidence score
        """
        height, width = binary.shape
        
        # Edge detection
        edges = cv2.Canny(binary, 50, 150, apertureSize=3)
        
        # Detect lines using Hough transform
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=int(width/3))
        
        if lines is None or len(lines) == 0:
            logger.debug("No lines detected with Hough transform")
            return binary, 0.0, 0.0
        
        # Extract angles and calculate median
        angles = []
        for line in lines:
            rho, theta = line[0]
            # Convert from polar to degrees, considering only near-horizontal lines
            angle_deg = (theta * 180 / np.pi) - 90
            
            # Filter angles to keep only those that make sense for text lines
            if abs(angle_deg) < self.max_angle:
                angles.append(angle_deg)
        
        if not angles:
            logger.debug("No valid angles found in Hough lines")
            return binary, 0.0, 0.0
        
        # Get median angle for robustness
        median_angle = np.median(angles)
        
        # Calculate confidence based on angle consistency
        angle_std = np.std(angles)
        if angle_std < 1e-6:  # Avoid division by zero
            confidence = 1.0
        else:
            confidence = min(1.0, 1.0 / (1.0 + angle_std))
        
        logger.debug(f"Hough lines angle: {median_angle:.2f}° (std: {angle_std:.2f}, confidence: {confidence:.2f})")
        
        # Apply rotation to correct skew
        M = cv2.getRotationMatrix2D((width/2, height/2), median_angle, 1)
        corrected = cv2.warpAffine(binary, M, (width, height), 
                                  flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        
        return corrected, median_angle, confidence
    
    def _correct_contour(self, binary: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """
        Correct skew using contour orientation.
        
        Args:
            binary: Binarized input image
            
        Returns:
            Tuple[np.ndarray, float, float]:
                - Corrected image
                - Detected angle
                - Confidence score
        """
        height, width = binary.shape
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            logger.debug("No contours found for orientation detection")
            return binary, 0.0, 0.0
        
        # Find largest contour by area
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get minimum area rectangle
        rect = cv2.minAreaRect(largest_contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        
        # Calculate angle from rectangle
        angle = rect[2]
        
        # Adjust angle to be between -45 and 45 degrees
        if angle < -45:
            angle = 90 + angle
        
        # Calculate confidence based on rectangle aspect ratio
        width_rect = rect[1][0]
        height_rect = rect[1][1]
        
        if width_rect > height_rect:
            aspect_ratio = width_rect / max(height_rect, 1)
        else:
            aspect_ratio = height_rect / max(width_rect, 1)
            angle = 90 - angle  # Adjust angle for vertical rectangle
        
        # Higher aspect ratio means more confidence in the orientation
        confidence = min(1.0, (aspect_ratio - 1) / 4)
        
        logger.debug(f"Contour angle: {angle:.2f}° (aspect ratio: {aspect_ratio:.2f}, confidence: {confidence:.2f})")
        
        # Apply rotation to correct skew
        M = cv2.getRotationMatrix2D((width/2, height/2), angle, 1)
        corrected = cv2.warpAffine(binary, M, (width, height), 
                                  flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        
        return corrected, angle, confidence

# Helper functions for preprocessing before OCR
def preprocess_for_ocr(image: np.ndarray, preprocess_method: str = "adaptive") -> np.ndarray:
    """
    Preprocess image for OCR.
    
    Args:
        image: Input image (BGR)
        preprocess_method: Preprocessing method
        
    Returns:
        np.ndarray: Preprocessed image
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply different preprocessing methods
    if preprocess_method == "adaptive":
        # Adaptive thresholding for varying lighting conditions
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
    elif preprocess_method == "otsu":
        # Otsu's thresholding for bimodal images
        _, binary = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
    elif preprocess_method == "canny":
        # Canny edge detection
        binary = cv2.Canny(blurred, 100, 200)
    elif preprocess_method == "laplacian":
        # Laplacian edge enhancement
        laplacian = cv2.Laplacian(blurred, cv2.CV_8U, ksize=3)
        _, binary = cv2.threshold(laplacian, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        # Default to adaptive threshold
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
    
    # Apply morphological operations to enhance text
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    return binary

def try_all_preprocessing(image: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Apply all preprocessing methods to an image.
    
    Args:
        image: Input image
        
    Returns:
        Dict[str, np.ndarray]: Dictionary of preprocessed images
    """
    methods = ["adaptive", "otsu", "canny", "laplacian"]
    results = {}
    
    for method in methods:
        results[method] = preprocess_for_ocr(image, method)
    
    return results

# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load a test image (replace with your own)
    img_path = "test_plate.jpg"
    img = cv2.imread(img_path)
    
    if img is None:
        print(f"Error: Could not load image from {img_path}")
        exit(1)
    
    # Create skew corrector
    corrector = SkewCorrector()
    
    # Correct skew (try all methods)
    corrected_profile, angle_profile, conf_profile = corrector.correct(img, "profile")
    corrected_hough, angle_hough, conf_hough = corrector.correct(img, "hough")
    corrected_contour, angle_contour, conf_contour = corrector.correct(img, "contour")
    corrected_auto, angle_auto, conf_auto = corrector.correct(img, "auto")
    
    # Print results
    print(f"Profile method: angle={angle_profile:.2f}°, confidence={conf_profile:.2f}")
    print(f"Hough method: angle={angle_hough:.2f}°, confidence={conf_hough:.2f}")
    print(f"Contour method: angle={angle_contour:.2f}°, confidence={conf_contour:.2f}")
    print(f"Auto method: angle={angle_auto:.2f}°, confidence={conf_auto:.2f}")
    
    # Display results
    cv2.imshow("Original", img)
    cv2.imshow("Profile", corrected_profile)
    cv2.imshow("Hough", corrected_hough)
    cv2.imshow("Contour", corrected_contour)
    cv2.imshow("Auto", corrected_auto)
    
    # Try different preprocessing methods
    preprocessed = try_all_preprocessing(corrected_auto)
    
    for method, processed in preprocessed.items():
        cv2.imshow(f"Preprocess: {method}", processed)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
