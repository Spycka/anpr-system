#!/usr/bin/env python3
"""
YOLO11s license plate detector with GPU acceleration (OpenCL/OpenGL).
Designed specifically for Orange Pi 5 Ultra.
"""
import os
import time
import logging
import json
import numpy as np
import cv2
from typing import List, Tuple, Dict, Any, Optional, Union
import threading
import torch

# Import project modules
from utils.logger import get_logger
from utils.hardware import AcceleratorType, HardwareDetector

# Configure logger
logger = get_logger("detection")

class YOLO11sGPU:
    """
    YOLO11s license plate detector with GPU acceleration.
    
    Optimized for Mali-G610 MP4 GPU on Orange Pi 5 Ultra.
    Uses PyTorch with OpenCL/OpenGL acceleration.
    """
    
    # Model settings
    DEFAULT_MODEL_PATH = os.path.join("models", "YOLO11s.pt")
    DEFAULT_CONF_THRESHOLD = 0.5
    DEFAULT_IOU_THRESHOLD = 0.45
    DEFAULT_INPUT_SIZE = 640
    
    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        conf_threshold: float = DEFAULT_CONF_THRESHOLD,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
        input_size: int = DEFAULT_INPUT_SIZE,
        device: Optional[str] = None,
        class_names: Optional[List[str]] = None
    ):
        """
        Initialize YOLO11s detector.
        
        Args:
            model_path: Path to YOLO11s model file
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            input_size: Input image size (square)
            device: Device to run model on ('cpu', 'gpu', or None for auto)
            class_names: List of class names (if None, will try to load from model)
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.input_size = input_size
        self.class_names = class_names
        
        # Hardware detection
        self.hardware_detector = HardwareDetector()
        self.best_accelerator, _ = self.hardware_detector.get_best_available_accelerator()
        
        # Determine device to use
        if device is None:
            # Auto-detect best device
            if self.best_accelerator == AcceleratorType.GPU:
                self.device = 'gpu'
                logger.info("Using GPU acceleration for YOLO11s")
            else:
                self.device = 'cpu'
                logger.info("Using CPU for YOLO11s (no GPU acceleration available)")
        else:
            self.device = device.lower()
            
        # Check if PyTorch is available
        self.torch_available = False
        try:
            import torch
            self.torch_available = True
        except ImportError:
            logger.error("PyTorch not installed. Please install PyTorch for YOLO11s.")
            return
            
        # Initialize model (will be loaded on first use)
        self.model = None
        self.model_lock = threading.Lock()
        
        # Set device string for PyTorch
        if self.device == 'gpu' and self.best_accelerator == AcceleratorType.GPU:
            # For GPU, use 'mps' on Mac, else default to CPU
            # Note: we can't use CUDA since it's not supported on Orange Pi
            if torch.backends.mps.is_available():
                self.torch_device = torch.device('mps')
                logger.info("Using MPS acceleration for PyTorch")
            else:
                # Fallback to CPU (we can't use CUDA on Orange Pi)
                self.torch_device = torch.device('cpu')
                logger.info("Falling back to CPU for PyTorch (MPS not available)")
        else:
            self.torch_device = torch.device('cpu')
            logger.info("Using CPU for PyTorch")
                
        logger.info(f"YOLO11s initialized (conf_threshold={conf_threshold}, "
                   f"iou_threshold={iou_threshold}, input_size={input_size})")
    
    def _ensure_model_loaded(self) -> bool:
        """
        Ensure model is loaded.
        
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        if self.model is not None:
            return True
        
        if not self.torch_available:
            logger.error("Cannot load model: PyTorch not available")
            return False
        
        with self.model_lock:
            if self.model is not None:
                return True
            
            try:
                # Check if model file exists
                if not os.path.isfile(self.model_path):
                    logger.error(f"Model file not found: {self.model_path}")
                    return False
                
                logger.info(f"Loading YOLO11s model from {self.model_path}")
                
                start_time = time.time()
                
                # Load model
                self.model = torch.hub.load('ultralytics/yolov5', 'custom', 
                                          path=self.model_path, 
                                          device=self.torch_device)
                
                # Set NMS thresholds
                self.model.conf = self.conf_threshold  # Confidence threshold
                self.model.iou = self.iou_threshold    # IoU threshold
                self.model.max_det = 100               # Maximum detections per image
                
                # Get class names if not provided
                if self.class_names is None:
                    self.class_names = self.model.names
                
                logger.info(f"Model loaded in {time.time() - start_time:.2f} seconds")
                logger.info(f"Class names: {self.class_names}")
                
                return True
            
            except Exception as e:
                logger.error(f"Failed to load YOLO11s model: {str(e)}")
                self.model = None
                return False
    
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
            # Convert image to RGB (PyTorch models expect RGB)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Run inference
            results = self.model(rgb_image, size=self.input_size)
            
            # Process results
            detections = []
            
            # Extract detections from PyTorch results
            for i, det in enumerate(results.xyxy[0]):
                x1, y1, x2, y2, confidence, class_id = det.cpu().numpy()
                
                # Convert to integers
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                class_id = int(class_id)
                
                # Get class name
                class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
                
                # Filter by confidence
                if confidence >= self.conf_threshold:
                    detection = {
                        'box': [x1, y1, x2, y2],
                        'confidence': float(confidence),
                        'class_id': class_id,
                        'class_name': class_name
                    }
                    detections.append(detection)
            
            # Log processing time and detection count
            processing_time = time.time() - start_time
            logger.debug(f"Detected {len(detections)} license plates in {processing_time:.3f}s")
            
            if return_details:
                return detections
            else:
                # Draw detections on image
                return self._draw_detections(image, detections)
        
        except Exception as e:
            logger.error(f"Detection error: {str(e)}")
            
            if return_details:
                return []
            else:
                return image.copy()
    
    def detect_batch(self, images: List[np.ndarray]) -> List[List[Dict[str, Any]]]:
        """
        Detect license plates in a batch of images.
        
        Args:
            images: List of input images
            
        Returns:
            List[List[Dict[str, Any]]]: List of detection lists for each image
        """
        # Ensure model is loaded
        if not self._ensure_model_loaded():
            return [[] for _ in images]
        
        start_time = time.time()
        
        try:
            # Convert images to RGB
            rgb_images = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in images]
            
            # Run inference on batch
            results = self.model(rgb_images, size=self.input_size)
            
            # Process results
            all_detections = []
            
            for i, result in enumerate(results.xyxy):
                detections = []
                
                for det in result:
                    x1, y1, x2, y2, confidence, class_id = det.cpu().numpy()
                    
                    # Convert to integers
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    class_id = int(class_id)
                    
                    # Get class name
                    class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
                    
                    # Filter by confidence
                    if confidence >= self.conf_threshold:
                        detection = {
                            'box': [x1, y1, x2, y2],
                            'confidence': float(confidence),
                            'class_id': class_id,
                            'class_name': class_name
                        }
                        detections.append(detection)
                
                all_detections.append(detections)
            
            # Log processing time and detection count
            processing_time = time.time() - start_time
            total_detections = sum(len(dets) for dets in all_detections)
            logger.debug(f"Batch detection: {total_detections} license plates in "
                       f"{len(images)} images ({processing_time:.3f}s)")
            
            return all_detections
        
        except Exception as e:
            logger.error(f"Batch detection error: {str(e)}")
            return [[] for _ in images]
    
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

# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create detector
    detector = YOLO11sGPU(
        model_path=YOLO11sGPU.DEFAULT_MODEL_PATH,
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
        
        # Save detections
        saved_files = detector.save_detections(
            img, detections, "captures", save_full=True
        )
        
        print(f"Saved {len(saved_files)} files:")
        for file in saved_files:
            print(f"  {file}")
        
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
