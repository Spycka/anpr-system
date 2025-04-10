#!/usr/bin/env python3
"""
YOLO11s license plate detector with RKNN (NPU) acceleration.
Designed specifically for the RK3588 NPU in Orange Pi 5 Ultra.
"""
import os
import time
import logging
import json
import numpy as np
import cv2
from typing import List, Tuple, Dict, Any, Optional, Union
import threading

# Import logger
from utils.logger import get_logger
from utils.hardware import AcceleratorType, HardwareDetector

# Configure logger
logger = get_logger("detection")

class YOLO11sRKNN:
    """
    YOLO11s license plate detector with RKNN (NPU) acceleration.
    
    Optimized for RK3588 NPU on Orange Pi 5 Ultra.
    Uses rknn_toolkit_lite for efficient inference.
    """
    
    # Model settings
    DEFAULT_MODEL_PATH = os.path.join("models", "YOLO11s.rknn")
    DEFAULT_CONF_THRESHOLD = 0.5
    DEFAULT_IOU_THRESHOLD = 0.45
    DEFAULT_INPUT_SIZE = 640
    
    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        conf_threshold: float = DEFAULT_CONF_THRESHOLD,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
        input_size: int = DEFAULT_INPUT_SIZE,
        class_names: Optional[List[str]] = None
    ):
        """
        Initialize YOLO11s RKNN detector.
        
        Args:
            model_path: Path to YOLO11s RKNN model file
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            input_size: Input image size (square)
            class_names: List of class names
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.input_size = input_size
        
        # Default class names if not provided
        if class_names is None:
            self.class_names = ['license_plate']
        else:
            self.class_names = class_names
        
        # Hardware detection
        self.hardware_detector = HardwareDetector()
        self.best_accelerator, _ = self.hardware_detector.get_best_available_accelerator()
        
        # Check if RKNN is available
        self.rknn_available = False
        try:
            from rknnlite.api import RKNNLite
            self.rknn_available = True
        except ImportError:
            logger.error("RKNN toolkit not installed. Please install rknn_toolkit_lite for NPU acceleration.")
        
        # Initialize model (will be loaded on first use)
        self.rknn = None
        self.model_lock = threading.Lock()
        
        logger.info(f"YOLO11s RKNN initialized (conf_threshold={conf_threshold}, "
                  f"iou_threshold={iou_threshold}, input_size={input_size})")
    
    def _ensure_model_loaded(self) -> bool:
        """
        Ensure model is loaded.
        
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        if self.rknn is not None:
            return True
        
        if not self.rknn_available:
            logger.error("Cannot load model: RKNN toolkit not available")
            return False
        
        with self.model_lock:
            if self.rknn is not None:
                return True
            
            try:
                # Check if model file exists
                if not os.path.isfile(self.model_path):
                    logger.error(f"Model file not found: {self.model_path}")
                    return False
                
                logger.info(f"Loading YOLO11s RKNN model from {self.model_path}")
                
                # Import RKNN
                from rknnlite.api import RKNNLite
                
                start_time = time.time()
                
                # Create RKNN object
                self.rknn = RKNNLite()
                
                # Load RKNN model
                ret = self.rknn.load_rknn(self.model_path)
                if ret != 0:
                    logger.error(f"Load RKNN model failed with error {ret}")
                    self.rknn = None
                    return False
                
                # Initialize runtime
                ret = self.rknn.init_runtime(target='rk3588')
                if ret != 0:
                    logger.error(f"Init RKNN runtime failed with error {ret}")
                    self.rknn = None
                    return False
                
                logger.info(f"Model loaded in {time.time() - start_time:.2f} seconds")
                
                return True
            
            except Exception as e:
                logger.error(f"Failed to load YOLO11s RKNN model: {str(e)}")
                self.rknn = None
                return False
    
    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for RKNN inference.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            np.ndarray: Preprocessed image
        """
        # Resize to input size
        img = cv2.resize(image, (self.input_size, self.input_size))
        
        # Convert to RGB (RKNN models usually expect RGB)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Convert to float32 and normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        
        # Transpose to NCHW format (batch, channel, height, width)
        img = np.transpose(img, (2, 0, 1))
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        return img
    
    def _non_max_suppression(self, boxes: np.ndarray, scores: np.ndarray, 
                             class_ids: np.ndarray) -> List[Dict[str, Any]]:
        """
        Apply non-maximum suppression to detections.
        
        Args:
            boxes: Detection boxes [x1, y1, x2, y2]
            scores: Detection confidence scores
            class_ids: Detection class IDs
            
        Returns:
            List[Dict[str, Any]]: List of filtered detections
        """
        # Filter by confidence threshold
        mask = scores >= self.conf_threshold
        boxes = boxes[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]
        
        # If no detections, return empty list
        if len(boxes) == 0:
            return []
        
        # Apply NMS
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(), 
            scores.tolist(), 
            self.conf_threshold, 
            self.iou_threshold
        )
        
        # Create result list
        detections = []
        
        # Format may differ in OpenCV versions
        if isinstance(indices, tuple):
            indices = indices[0]
        
        for i in indices:
            # Get the detection info
            if isinstance(i, (tuple, list)):
                i = i[0]  # Handle different OpenCV output formats
            
            box = boxes[i].astype(int).tolist()
            score = float(scores[i])
            class_id = int(class_ids[i])
            
            # Get class name
            class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
            
            # Format detection
            detection = {
                'box': box,
                'confidence': score,
                'class_id': class_id,
                'class_name': class_name
            }
            
            detections.append(detection)
        
        return detections
    
    def _postprocess(self, outputs: List[np.ndarray], 
                     image_shape: Tuple[int, int]) -> List[Dict[str, Any]]:
        """
        Process RKNN model outputs to get detections.
        
        Args:
            outputs: RKNN model output tensors
            image_shape: Original image shape (height, width)
            
        Returns:
            List[Dict[str, Any]]: List of detections
        """
        # YOLO outputs are typically formatted as follows:
        # 1. boxes [batch, num_boxes, 4]
        # 2. scores [batch, num_boxes, num_classes]
        # This may vary depending on how the model was exported to RKNN
        
        try:
            # Extract boxes and scores
            boxes = outputs[0]  # Assuming first output is boxes
            scores = outputs[1]  # Assuming second output is scores
            
            # Get highest confidence class and its index
            class_scores = np.max(scores, axis=-1)
            class_ids = np.argmax(scores, axis=-1)
            
            # Scale boxes to original image size
            original_h, original_w = image_shape
            scale_h = original_h / self.input_size
            scale_w = original_w / self.input_size
            
            # Scale boxes: [x1, y1, x2, y2]
            boxes[:, 0] *= scale_w  # x1
            boxes[:, 1] *= scale_h  # y1
            boxes[:, 2] *= scale_w  # x2
            boxes[:, 3] *= scale_h  # y2
            
            # Apply NMS
            detections = self._non_max_suppression(boxes, class_scores, class_ids)
            
            return detections
        
        except Exception as e:
            logger.error(f"Error in postprocessing: {str(e)}")
            return []
    
    def detect(self, image: np.ndarray, return_details: bool = False) -> Union[List[Dict[str, Any]], np.ndarray]:
        """
        Detect license plates in an image.
        
        Args:
            image: Input image (BGR format)
            return_details: If True, return detailed detection information
            
        Returns:
            Union[List[Dict[str, Any]], np.ndarray]:
                - If return_details=True: List of detection dictionaries
                - If return_details=False: Image with detection visualizations
        """
        # Ensure model is loaded
        if not self._ensure_model_loaded():
            if return_details:
                return []
            else:
                return image.copy()
        
        start_time = time.time()
        
        try:
            # Get image shape
            original_shape = image.shape[:2]
            
            # Preprocess image
            preprocessed = self._preprocess(image)
            
            # Inference
            outputs = self.rknn.inference(inputs=[preprocessed])
            
            # Postprocess
            detections = self._postprocess(outputs, original_shape)
            
            # Log processing time and detection count
            processing_time = time.time() - start_time
            logger.debug(f"RKNN: Detected {len(detections)} license plates in {processing_time:.3f}s")
            
            if return_details:
                return detections
            else:
                # Draw detections on image
                return self._draw_detections(image, detections)
        
        except Exception as e:
            logger.error(f"RKNN detection error: {str(e)}")
            
            if return_details:
                return []
            else:
                return image.copy()
    
    def extract_plates(self, image: np.ndarray, margin: float = 0.0) -> List[Dict[str, Any]]:
        """
        Detect and extract license plate regions from an image.
        
        Args:
            image: Input image
            margin: Margin to add around detections (as fraction of box size)
            
        Returns:
            List[Dict[str, Any]]: List of dictionaries with plate info and cropped images
        """
        # Detect plates
        detections = self.detect(image, return_details=True)
        
        if not detections:
            return []
        
        # Extract plate regions
        plates = []
        
        for i, det in enumerate(detections):
            # Get box coordinates
            x1, y1, x2, y2 = det['box']
            
            # Add margin
            width, height = x2 - x1, y2 - y1
            x1 = max(0, int(x1 - margin * width))
            y1 = max(0, int(y1 - margin * height))
            x2 = min(image.shape[1], int(x2 + margin * width))
            y2 = min(image.shape[0], int(y2 + margin * height))
            
            # Crop plate region
            plate_img = image[y1:y2, x1:x2]
            
            # Create plate info
            plate_info = {
                'image': plate_img,
                'box': [x1, y1, x2, y2],
                'confidence': det['confidence'],
                'class_id': det['class_id'],
                'class_name': det['class_name'],
                'index': i
            }
            
            plates.append(plate_info)
        
        return plates
    
    def _draw_detections(self, image: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """
        Draw detection boxes and labels on an image.
        
        Args:
            image: Input image
            detections: List of detection dictionaries
            
        Returns:
            np.ndarray: Image with drawn detections
        """
        # Make a copy of the image
        img_draw = image.copy()
        
        for det in detections:
            # Get detection info
            x1, y1, x2, y2 = det['box']
            confidence = det['confidence']
            class_name = det['class_name']
            
            # Draw bounding box
            cv2.rectangle(img_draw, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label background
            label = f"{class_name}: {confidence:.2f}"
            (label_width, label_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                img_draw, 
                (x1, y1 - label_height - 5), 
                (x1 + label_width + 5, y1), 
                (0, 255, 0), 
                -1
            )
            
            # Draw label text
            cv2.putText(
                img_draw, 
                label, 
                (x1 + 3, y1 - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (0, 0, 0), 
                1
            )
        
        return img_draw
    
    def save_detections(self, image: np.ndarray, 
                        detections: List[Dict[str, Any]], 
                        save_dir: str,
                        save_full: bool = False,
                        prefix: str = "plate") -> List[str]:
        """
        Save detection images and metadata.
        
        Args:
            image: Input image
            detections: List of detection dictionaries
            save_dir: Directory to save images
            save_full: Whether to save full image with annotations
            prefix: Filename prefix for saved images
            
        Returns:
            List[str]: List of saved file paths
        """
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        saved_files = []
        
        # Generate timestamp for filenames
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Save full image with annotations if requested
        if save_full and detections:
            # Draw detections on image
            full_img = self._draw_detections(image, detections)
            
            # Save full image
            full_path = os.path.join(save_dir, f"{prefix}_full_{timestamp}.jpg")
            cv2.imwrite(full_path, full_img)
            saved_files.append(full_path)
        
        # Save individual plate crops
        for i, det in enumerate(detections):
            # Get box coordinates
            x1, y1, x2, y2 = det['box']
            
            # Crop plate region
            plate_img = image[y1:y2, x1:x2]
            
            # Generate filename
            confidence = det['confidence']
            class_name = det['class_name']
            plate_path = os.path.join(
                save_dir, 
                f"{prefix}_{timestamp}_{i:02d}_{class_name}_{confidence:.2f}.jpg"
            )
            
            # Save cropped image
            cv2.imwrite(plate_path, plate_img)
            saved_files.append(plate_path)
            
            # Save metadata
            metadata = det.copy()
            metadata['timestamp'] = timestamp
            metadata['image_path'] = plate_path
            
            # Remove 'box' entry for JSON serialization
            if 'box' in metadata:
                x1, y1, x2, y2 = metadata['box']
                metadata['box'] = {
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                    'width': x2-x1, 'height': y2-y1
                }
            
            # Save metadata as JSON
            meta_path = os.path.join(
                save_dir,
                f"{prefix}_{timestamp}_{i:02d}_{class_name}_{confidence:.2f}.json"
            )
            
            with open(meta_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            saved_files.append(meta_path)
        
        return saved_files
    
    def __del__(self):
        """Clean up resources when object is destroyed."""
        if self.rknn is not None:
            self.rknn.release()
            self.rknn = None

# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create detector
    detector = YOLO11sRKNN(
        model_path=YOLO11sRKNN.DEFAULT_MODEL_PATH,
        conf_threshold=0.5,
        iou_threshold=0.45
    )
    
    # Load test image (replace with your own)
    img_path = "test_image.jpg"
    img = cv2.imread(img_path)
    
    if img is not None:
        # Detect license plates
        detections = detector.detect(img, return_details=True)
        
        # Print detection results
        print(f"Detected {len(detections)} license plates:")
        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det['box']
            print(f"  Plate {i+1}: {det['class_name']} (conf: {det['confidence']:.2f}), "
                 f"box: [{x1}, {y1}, {x2}, {y2}]")
        
        # Extract plate regions
        plates = detector.extract_plates(img, margin=0.1)
        
        # Display results
        result_img = detector._draw_detections(img, detections)
        cv2.imshow("Detections", result_img)
        
        # Display cropped plates
        for i, plate in enumerate(plates):
            cv2.imshow(f"Plate {i+1}", plate['image'])
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print(f"Error: Could not load image from {img_path}")
