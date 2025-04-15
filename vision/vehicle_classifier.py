#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Vehicle Classification Module
Performs make detection and color classification for vehicles

This module handles the detection of vehicle make using a ResNet18 model
and color classification using K-means clustering.
"""

import os
import time
import logging
import numpy as np
import cv2
from collections import Counter

# Try importing RKNN modules with version compatibility handling
try:
    # First, try RKNNLite which is optimized for runtime inference
    from rknnlite.api import RKNNLite
    RKNN_MODE = "lite"
except ImportError:
    try:
        # Fall back to RKNN API if RKNNLite is not available
        from rknn.api import RKNN
        RKNN_MODE = "full"
    except ImportError:
        # If neither is available, set flag for mock mode
        RKNN_MODE = None

class VehicleClassifier:
    """
    Vehicle classifier using ResNet18 for make detection
    and K-means clustering for color classification
    """
    
    # Define standard colors and their RGB values (for color classification)
    STANDARD_COLORS = {
        'BLACK': [0, 0, 0],
        'WHITE': [255, 255, 255],
        'RED': [255, 0, 0],
        'GREEN': [0, 255, 0],
        'BLUE': [0, 0, 255],
        'YELLOW': [255, 255, 0],
        'SILVER': [192, 192, 192],
        'GRAY': [128, 128, 128],
        'BROWN': [165, 42, 42],
        'ORANGE': [255, 165, 0],
        'PURPLE': [128, 0, 128],
    }
    
    # Common vehicle makes
    VEHICLE_MAKES = [
        'TOYOTA', 'HONDA', 'FORD', 'CHEVROLET', 'MERCEDES', 'BMW', 'AUDI',
        'VOLKSWAGEN', 'NISSAN', 'HYUNDAI', 'KIA', 'SUBARU', 'MAZDA',
        'LEXUS', 'JEEP', 'TESLA', 'VOLVO', 'ACURA', 'DODGE', 'RAM',
        'CADILLAC', 'CHRYSLER', 'GMC', 'LINCOLN', 'BUICK', 'INFINITI',
        'MITSUBISHI', 'MINI', 'PORSCHE', 'JAGUAR', 'LAND ROVER',
        'UNKNOWN'  # Last class for unknown makes
    ]
    
    def __init__(self, model_path=None, enable_make=True, enable_color=True, mock=False):
        """
        Initialize the vehicle classifier
        
        Args:
            model_path (str): Path to the RKNN ResNet18 model for make detection
            enable_make (bool): Whether to enable make detection
            enable_color (bool): Whether to enable color detection
            mock (bool): Run in mock mode without actual detection
        """
        self.logger = logging.getLogger('vehicle_classifier')
        self.model_path = model_path
        self.enable_make = enable_make
        self.enable_color = enable_color
        self.mock_mode = mock or RKNN_MODE is None
        
        # Check if features are enabled
        if not self.enable_make and not self.enable_color:
            self.logger.warning("Both make and color detection are disabled, classifier will do nothing")
        
        # Initialize make detector if enabled
        if self.enable_make and not self.mock_mode:
            self._init_make_detector()
        else:
            self.logger.info("Make detection disabled or running in mock mode")
        
        # Initialize color classifier (no need for model initialization)
        if self.enable_color:
            self.logger.info("Color classification enabled")
    
    def _init_make_detector(self):
        """Initialize ResNet18 model for make detection"""
        if not self.model_path or not os.path.exists(self.model_path):
            self.logger.error(f"Make detection model not found: {self.model_path}")
            self.logger.warning("Falling back to mock mode for make detection")
            self.enable_make = False
            return
        
        try:
            self.logger.info(f"Initializing ResNet18 make detection model: {self.model_path}")
            
            # Use appropriate RKNN API based on what's available
            if RKNN_MODE == "lite":
                self.logger.info("Using RKNNLite API for make detection")
                self.rknn = RKNNLite()
            else:
                self.logger.info("Using RKNN API for make detection")
                self.rknn = RKNN()
            
            # Load RKNN model
            ret = self.rknn.load_rknn(self.model_path)
            if ret != 0:
                # If fails, try with force=True option
                self.logger.warning("Failed to load RKNN model, retrying with force=True")
                ret = self.rknn.load_rknn(self.model_path, force=True)
                if ret != 0:
                    raise RuntimeError(f"Failed to load RKNN model: {ret}")
            
            # Initialize runtime
            ret = self.rknn.init_runtime(target='rk3588')
            if ret != 0:
                # Try alternative initialization
                self.logger.warning("Failed to init runtime with target, trying without target")
                ret = self.rknn.init_runtime()
                if ret != 0:
                    raise RuntimeError(f"Failed to init RKNN runtime: {ret}")
            
            self.logger.info("Make detection model initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing make detection model: {str(e)}")
            self.logger.warning("Falling back to mock mode for make detection")
            self.enable_make = False
    
    def preprocess_for_make(self, img):
        """
        Preprocess image for ResNet18 inference
        
        Args:
            img: Input image (BGR format)
            
        Returns:
            Preprocessed image ready for RKNN inference
        """
        # Resize to network input size (ResNet18 typically uses 224x224)
        resized = cv2.resize(img, (224, 224))
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Keep in uint8 range for NPU acceleration
        # DO NOT normalize to 0-1 range for quantized NPU models
        input_data = np.expand_dims(rgb, axis=0)
        
        return input_data
    
    def detect_make(self, img):
        """
        Detect vehicle make using ResNet18 model
        
        Args:
            img: Input image (BGR format)
            
        Returns:
            tuple: (make, confidence)
        """
        # Check if make detection is enabled
        if not self.enable_make:
            return "UNKNOWN", 0.0
        
        # Mock mode for testing without hardware
        if self.mock_mode:
            # Simulate processing delay
            time.sleep(0.05)
            
            # Return random make with reasonable confidence
            make_idx = np.random.randint(0, len(self.VEHICLE_MAKES) - 1)  # Exclude UNKNOWN
            make = self.VEHICLE_MAKES[make_idx]
            confidence = np.random.uniform(0.7, 0.95)
            
            return make, confidence
        
        try:
            # Preprocess image
            input_data = self.preprocess_for_make(img)
            
            # Run inference
            outputs = self.rknn.inference(inputs=[input_data])
            
            # Process output
            # Assuming output is a probability distribution over make classes
            predictions = outputs[0][0]
            
            # Get most likely class
            make_idx = np.argmax(predictions)
            confidence = float(predictions[make_idx])
            
            # Convert to make name
            if make_idx < len(self.VEHICLE_MAKES):
                make = self.VEHICLE_MAKES[make_idx]
            else:
                make = "UNKNOWN"
            
            return make, confidence
            
        except Exception as e:
            self.logger.error(f"Error during make detection: {str(e)}")
            return "UNKNOWN", 0.0
    
    def detect_color(self, img):
        """
        Detect vehicle color using K-means clustering
        
        Args:
            img: Input image (BGR format)
            
        Returns:
            tuple: (color, confidence)
        """
        # Check if color detection is enabled
        if not self.enable_color:
            return "UNKNOWN", 0.0
        
        # Mock mode for testing without hardware
        if self.mock_mode:
            # Simulate processing delay
            time.sleep(0.02)
            
            # Return random color with reasonable confidence
            colors = list(self.STANDARD_COLORS.keys())
            color_idx = np.random.randint(0, len(colors))
            color = colors[color_idx]
            confidence = np.random.uniform(0.7, 0.95)
            
            return color, confidence
        
        try:
            # Resize image to reduce processing time
            img_small = cv2.resize(img, (128, 128))
            
            # Convert to RGB format (easier for color comparison)
            rgb = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)
            
            # Reshape to list of pixels
            pixels = rgb.reshape(-1, 3)
            
            # Apply K-means clustering
            k = 5  # Number of clusters
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
            _, labels, centers = cv2.kmeans(np.float32(pixels), k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            
            # Count pixels in each cluster
            counts = Counter(labels.flatten())
            
            # Calculate cluster sizes as percentages
            total_pixels = len(labels)
            percentages = {i: count / total_pixels for i, count in counts.items()}
            
            # Get the most dominant clusters (excluding small ones)
            dominant_clusters = [i for i, pct in percentages.items() if pct > 0.15]
            
            # If no dominant clusters, use the largest
            if not dominant_clusters:
                dominant_clusters = [counts.most_common(1)[0][0]]
            
            # Find closest standard color for each dominant cluster
            color_matches = []
            for cluster_idx in dominant_clusters:
                center = centers[cluster_idx]
                
                # Find closest standard color
                min_dist = float('inf')
                best_color = "UNKNOWN"
                
                for color_name, color_rgb in self.STANDARD_COLORS.items():
                    dist = np.sqrt(np.sum((center - color_rgb) ** 2))
                    if dist < min_dist:
                        min_dist = dist
                        best_color = color_name
                
                # Add color match with confidence based on cluster size
                confidence = percentages[cluster_idx]
                color_matches.append((best_color, confidence))
            
            # Get the most confident color match
            color_matches.sort(key=lambda x: x[1], reverse=True)
            color, confidence = color_matches[0]
            
            return color, confidence
            
        except Exception as e:
            self.logger.error(f"Error during color detection: {str(e)}")
            return "UNKNOWN", 0.0
    
    def classify_vehicle(self, img):
        """
        Classify vehicle make and color from image
        
        Args:
            img: Input image (BGR format)
            
        Returns:
            tuple: (make, make_confidence, color, color_confidence)
        """
        # Make detection
        make, make_confidence = self.detect_make(img) if self.enable_make else ("UNKNOWN", 0.0)
        
        # Color detection
        color, color_confidence = self.detect_color(img) if self.enable_color else ("UNKNOWN", 0.0)
        
        # Log results
        self.logger.info(f"Vehicle classification: Make={make} ({make_confidence:.2f}), Color={color} ({color_confidence:.2f})")
        
        return make, make_confidence, color, color_confidence
    
    def release(self):
        """Release resources"""
        if self.enable_make and not self.mock_mode and hasattr(self, 'rknn'):
            try:
                self.rknn.release()
                self.logger.info("Make detection model resources released")
            except Exception as e:
                self.logger.error(f"Error releasing make detection model resources: {str(e)}")
