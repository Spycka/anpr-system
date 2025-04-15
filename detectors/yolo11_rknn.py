#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
YOLO11s RKNN Detector for Rockchip RK3588
License Plate Detection using YOLO11s model on RK3588 NPU

This module handles the detection of license plates in video frames
using the YOLO11s model optimized for the RK3588 NPU, aligned with
the airockchip/ultralytics_yolo11 implementation.
"""

import os
import time
import logging
import numpy as np
import cv2

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


class LicensePlateDetector:
    """
    License Plate Detector using YOLO11s optimized for RK3588 NPU
    Aligned with airockchip/ultralytics_yolo11 implementation
    """
    
    def __init__(self, model_path, conf_threshold=0.3, nms_threshold=0.45, 
                 input_size=(640, 640), mock=False):
        """
        Initialize the detector with model and parameters
        
        Args:
            model_path (str): Path to the RKNN model file
            conf_threshold (float): Confidence threshold for detections
            nms_threshold (float): Non-maximum suppression threshold
            input_size (tuple): Input size for the network (width, height)
            mock (bool): Run in mock mode without NPU
        """
        self.logger = logging.getLogger('detection')
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.input_size = input_size
        self.mock_mode = mock or RKNN_MODE is None
        
        # Initialize RKNN model
        if not self.mock_mode:
            self._init_rknn()
        else:
            self.logger.warning("Running in MOCK MODE - detection will be simulated")
    
    def _init_rknn(self):
        """Initialize RKNN model for inference"""
        self.logger.info(f"Initializing RKNN model: {self.model_path}")
        
        try:
            # Use appropriate RKNN API based on what's available
            if RKNN_MODE == "lite":
                self.logger.info("Using RKNNLite API for inference")
                self.rknn = RKNNLite()
                
                # RK3588 specific parameters for RKNNLite
                rknn_lite_core_mask = 14 # Use the NPU cores (bits 1, 2, 3)
                
                # Load RKNN model
                ret = self.rknn.load_rknn(self.model_path)
                if ret != 0:
                    # If fails, try with force=True option
                    self.logger.warning("Failed to load RKNN model, retrying with force=True")
                    ret = self.rknn.load_rknn(self.model_path, force=True)
                    if ret != 0:
                        raise RuntimeError(f"Failed to load RKNN model: {ret}")
                
                # Initialize runtime with RK3588 specific parameters
                ret = self.rknn.init_runtime(core_mask=rknn_lite_core_mask)
                if ret != 0:
                    # Try alternative initialization without core mask
                    self.logger.warning("Failed to init runtime with core_mask, trying without")
                    ret = self.rknn.init_runtime()
                    if ret != 0:
                        raise RuntimeError(f"Failed to init RKNN runtime: {ret}")
            else:
                self.logger.info("Using RKNN API for inference")
                self.rknn = RKNN()
                
                # Load RKNN model
                ret = self.rknn.load_rknn(self.model_path)
                if ret != 0:
                    # If fails, try with force=True option
                    self.logger.warning("Failed to load RKNN model, retrying with force=True")
                    ret = self.rknn.load_rknn(self.model_path, force=True)
                    if ret != 0:
                        raise RuntimeError(f"Failed to load RKNN model: {ret}")
                
                # Initialize runtime for RK3588
                ret = self.rknn.init_runtime(target='rk3588')
                if ret != 0:
                    # Try alternative initialization
                    self.logger.warning("Failed to init runtime with target, trying without target")
                    ret = self.rknn.init_runtime()
                    if ret != 0:
                        raise RuntimeError(f"Failed to init RKNN runtime: {ret}")
            
            self.logger.info("RKNN model initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing RKNN model: {str(e)}")
            self.logger.warning("Falling back to MOCK MODE")
            self.mock_mode = True
    
    def preprocess(self, img):
        """
        Preprocess image for RKNN inference following airockchip/ultralytics_yolo11 approach
        
        Args:
            img: Input image (BGR format)
            
        Returns:
            Preprocessed image ready for RKNN inference
        """
        # Get original dimensions and calculate resize ratio
        orig_h, orig_w = img.shape[:2]
        input_h, input_w = self.input_size
        
        # Calculate resize ratio
        ratio_h = input_h / orig_h
        ratio_w = input_w / orig_w
        ratio = min(ratio_h, ratio_w)
        
        # Calculate new dimensions
        new_h = int(orig_h * ratio)
        new_w = int(orig_w * ratio)
        
        # Resize image
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Create padded image (center padding)
        padded = np.full((input_h, input_w, 3), 114, dtype=np.uint8)  # Use 114 padding as in YOLO
        offset_h = (input_h - new_h) // 2
        offset_w = (input_w - new_w) // 2
        padded[offset_h:offset_h+new_h, offset_w:offset_w+new_w, :] = resized
        
        # Convert BGR to RGB (YOLO uses RGB)
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        
        # Important: Keep in uint8 range for NPU acceleration
        # DO NOT normalize to 0-1 range for quantized NPU models
        input_data = np.expand_dims(rgb, axis=0)
        
        # Store padding info for later use in postprocessing
        self.padding_info = {
            'ratio': ratio,
            'offset_h': offset_h,
            'offset_w': offset_w,
            'orig_h': orig_h,
            'orig_w': orig_w
        }
        
        return input_data
    
    def postprocess(self, outputs, img_shape):
        """
        Process YOLO outputs to get bounding boxes, following airockchip implementation
        
        Args:
            outputs: YOLO model outputs
            img_shape: Original image shape (height, width)
            
        Returns:
            List of detections [x1, y1, x2, y2, confidence, class_id]
        """
        # Process the output according to YOLO11s format
        # For YOLO11s, the output includes detection boxes with format:
        # [x, y, w, h, confidence, class_scores...]
        
        # Get predictions
        predictions = outputs[0][0]  # First output, first batch item
        
        # Get original image dimensions and padding info
        orig_h, orig_w = img_shape
        ratio = self.padding_info['ratio']
        offset_h = self.padding_info['offset_h']
        offset_w = self.padding_info['offset_w']
        
        # Process each prediction
        boxes = []
        
        for pred in predictions:
            # Extract confidence score
            obj_conf = pred[4]
            
            # Skip low confidence detections
            if obj_conf < self.conf_threshold:
                continue
            
            # Get class with highest score
            class_scores = pred[5:]
            class_id = np.argmax(class_scores)
            class_conf = class_scores[class_id]
            
            # Calculate final confidence
            confidence = obj_conf * class_conf
            
            # Skip low confidence after class weighting
            if confidence < self.conf_threshold:
                continue
            
            # Extract box coordinates (x, y, w, h format, normalized)
            # These are relative to the padded image, need to adjust
            cx, cy, w, h = pred[0:4]
            
            # Adjust for padding
            cx = (cx - offset_w) / ratio
            cy = (cy - offset_h) / ratio
            w = w / ratio
            h = h / ratio
            
            # Convert to corner coordinates
            x1 = max(0, cx - w/2)
            y1 = max(0, cy - h/2)
            x2 = min(orig_w, cx + w/2)
            y2 = min(orig_h, cy + h/2)
            
            # Add to boxes
            boxes.append([x1, y1, x2, y2, confidence, class_id])
        
        # Apply non-maximum suppression
        if boxes:
            boxes = np.array(boxes)
            indices = self._nms(boxes[:, :4], boxes[:, 4], self.nms_threshold)
            boxes = boxes[indices]
        
        return boxes
    
    def _nms(self, boxes, scores, thresh):
        """
        Apply non-maximum suppression
        
        Args:
            boxes: Array of boxes in [x1, y1, x2, y2] format
            scores: Array of confidence scores
            thresh: NMS threshold
            
        Returns:
            Indices of boxes to keep
        """
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]
            
        return keep
    
    def detect(self, img):
        """
        Detect license plates in an image
        
        Args:
            img: Input image (BGR format)
            
        Returns:
            List of detections [x1, y1, x2, y2, confidence, class_id]
        """
        # Get original image shape for scaling
        img_shape = img.shape[:2]  # (height, width)
        
        # Mock mode for testing without hardware
        if self.mock_mode:
            # Simulate processing delay
            time.sleep(0.05)
            
            # Generate a random detection in each call with 70% probability
            if np.random.random() < 0.7:
                h, w = img_shape
                # Random box dimensions in the middle area of the image
                x1 = np.random.uniform(w * 0.3, w * 0.6)
                y1 = np.random.uniform(h * 0.3, h * 0.6)
                width = np.random.uniform(w * 0.15, w * 0.25)
                height = np.random.uniform(h * 0.05, h * 0.1)
                x2 = x1 + width
                y2 = y1 + height
                confidence = np.random.uniform(0.7, 0.95)
                
                # Return mock detection
                return [[x1, y1, x2, y2, confidence, 0]]
            return []
        
        try:
            # Preprocess image
            input_data = self.preprocess(img)
            
            # Run inference
            outputs = self.rknn.inference(inputs=[input_data])
            
            # Post-process to get detections
            detections = self.postprocess(outputs, img_shape)
            
            return detections
            
        except Exception as e:
            self.logger.error(f"Error during detection: {str(e)}")
            return []
    
    def release(self):
        """Release RKNN resources"""
        if not self.mock_mode and hasattr(self, 'rknn'):
            try:
                self.rknn.release()
                self.logger.info("RKNN resources released")
            except Exception as e:
                self.logger.error(f"Error releasing RKNN resources: {str(e)}")
