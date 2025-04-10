#!/usr/bin/env python3
"""
Vehicle make detection using ResNet18.
Classifies vehicle images to identify the manufacturer (make).
"""
import os
import time
import logging
import json
import numpy as np
import cv2
from typing import List, Tuple, Dict, Any, Optional, Union
import threading

# Try importing PyTorch
try:
    import torch
    import torch.nn as nn
    import torchvision.models as models
    import torchvision.transforms as transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Import project modules
from utils.logger import get_logger
from utils.hardware import AcceleratorType, HardwareDetector

# Configure logger
logger = get_logger("vehicle_make")

class MakeDetector:
    """
    Vehicle make detector using ResNet18.
    
    Identifies the manufacturer (make) of a vehicle from images.
    Uses transfer learning with a pre-trained ResNet18 model.
    """
    
    # Default settings
    DEFAULT_MODEL_PATH = os.path.join("models", "resnet18_vehicle_make.pth")
    DEFAULT_LABELS_PATH = os.path.join("models", "vehicle_make_labels.json")
    DEFAULT_CONFIDENCE_THRESHOLD = 0.6
    
    # Common vehicle makes
    DEFAULT_MAKES = [
        "acura", "alfa-romeo", "aston-martin", "audi", "bentley", "bmw", 
        "bugatti", "buick", "cadillac", "chevrolet", "chrysler", "citroen", 
        "dodge", "ferrari", "fiat", "ford", "gmc", "honda", "hyundai", 
        "infiniti", "jaguar", "jeep", "kia", "lamborghini", "land-rover", 
        "lexus", "lincoln", "maserati", "mazda", "mclaren", "mercedes-benz", 
        "mini", "mitsubishi", "nissan", "pagani", "peugeot", "porsche", 
        "ram", "renault", "rolls-royce", "subaru", "suzuki", "tesla", 
        "toyota", "volkswagen", "volvo"
    ]
    
    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        labels_path: str = DEFAULT_LABELS_PATH,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        device: Optional[str] = None,
        top_k: int = 3
    ):
        """
        Initialize vehicle make detector.
        
        Args:
            model_path: Path to model weights
            labels_path: Path to labels file
            confidence_threshold: Confidence threshold for predictions
            device: Device to run on ('cpu', 'gpu', or None for auto-detect)
            top_k: Number of top predictions to return
        """
        self.model_path = model_path
        self.labels_path = labels_path
        self.confidence_threshold = confidence_threshold
        self.top_k = top_k
        
        # Hardware detection
        self.hardware_detector = HardwareDetector()
        self.best_accelerator, _ = self.hardware_detector.get_best_available_accelerator()
        
        # Determine device to use
        if device is None:
            # Auto-detect best device
            if self.best_accelerator == AcceleratorType.GPU:
                self.device = 'gpu'
                logger.info("Using GPU acceleration for make detection")
            else:
                self.device = 'cpu'
                logger.info("Using CPU for make detection (no GPU acceleration available)")
        else:
            self.device = device.lower()
        
        # Check PyTorch availability
        self.torch_available = TORCH_AVAILABLE
        if not self.torch_available:
            logger.error("PyTorch not installed. Please install PyTorch for vehicle make detection.")
            return
        
        # Set device for PyTorch
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
        
        # Load labels
        self.labels = self._load_labels()
        
        # Initialize model (will be loaded on first use)
        self.model = None
        self.model_lock = threading.Lock()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        logger.info(f"Vehicle make detector initialized (threshold={confidence_threshold}, "
                  f"device={self.device})")
    
    def _load_labels(self) -> List[str]:
        """
        Load class labels.
        
        Returns:
            List[str]: List of class labels
        """
        # Try to load from labels file
        if os.path.isfile(self.labels_path):
            try:
                with open(self.labels_path, 'r') as f:
                    data = json.load(f)
                
                if isinstance(data, list):
                    logger.info(f"Loaded {len(data)} make labels from {self.labels_path}")
                    return data
                elif isinstance(data, dict) and 'labels' in data:
                    labels = data['labels']
                    logger.info(f"Loaded {len(labels)} make labels from {self.labels_path}")
                    return labels
                else:
                    logger.warning(f"Invalid labels format in {self.labels_path}")
            
            except Exception as e:
                logger.error(f"Failed to load labels from {self.labels_path}: {str(e)}")
        
        # Fall back to default labels
        logger.info(f"Using {len(self.DEFAULT_MAKES)} default make labels")
        return self.DEFAULT_MAKES
    
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
                if os.path.isfile(self.model_path):
                    # Load pretrained model
                    logger.info(f"Loading vehicle make model from {self.model_path}")
                    
                    start_time = time.time()
                    
                    # Create model
                    num_classes = len(self.labels)
                    model = models.resnet18(pretrained=False)
                    model.fc = nn.Linear(model.fc.in_features, num_classes)
                    
                    # Load weights
                    state_dict = torch.load(self.model_path, map_location=self.torch_device)
                    
                    # Handle different state dict formats
                    if 'model' in state_dict:
                        model.load_state_dict(state_dict['model'])
                    elif 'state_dict' in state_dict:
                        model.load_state_dict(state_dict['state_dict'])
                    else:
                        model.load_state_dict(state_dict)
                    
                    model.to(self.torch_device)
                    model.eval()
                    
                    self.model = model
                    logger.info(f"Model loaded in {time.time() - start_time:.2f} seconds")
                    
                    return True
                else:
                    # Create untrained model for testing
                    logger.warning(f"Model file not found: {self.model_path}")
                    logger.info("Creating untrained model for testing")
                    
                    num_classes = len(self.labels)
                    model = models.resnet18(pretrained=True)
                    model.fc = nn.Linear(model.fc.in_features, num_classes)
                    
                    model.to(self.torch_device)
                    model.eval()
                    
                    self.model = model
                    
                    logger.warning("Using untrained model - predictions will be inaccurate")
                    return True
            
            except Exception as e:
                logger.error(f"Failed to load vehicle make model: {str(e)}")
                self.model = None
                return False
    
    def detect_make(self, image: np.ndarray, 
                    return_details: bool = False) -> Union[str, Dict[str, Any]]:
        """
        Detect vehicle make from image.
        
        Args:
            image: Input image (BGR format)
            return_details: Whether to return detailed prediction info
            
        Returns:
            Union[str, Dict[str, Any]]:
                - If return_details=False: Predicted vehicle make
                - If return_details=True: Dict with predictions and confidence scores
        """
        # Ensure model is loaded
        if not self._ensure_model_loaded():
            if return_details:
                return {'make': 'unknown', 'confidence': 0.0, 'predictions': []}
            else:
                return 'unknown'
        
        start_time = time.time()
        
        try:
            with torch.no_grad():
                # Convert image to RGB (PyTorch models expect RGB)
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Preprocess image
                input_tensor = self.transform(rgb_image)
                input_batch = input_tensor.unsqueeze(0).to(self.torch_device)
                
                # Model inference
                output = self.model(input_batch)
                
                # Get probabilities
                probabilities = torch.nn.functional.softmax(output[0], dim=0)
                
                # Get top k predictions
                top_k_probs, top_k_indices = torch.topk(probabilities, min(self.top_k, len(self.labels)))
                
                # Convert to list
                top_k_probs = top_k_probs.cpu().numpy()
                top_k_indices = top_k_indices.cpu().numpy()
                
                # Get labels and create predictions list
                predictions = []
                for i, (idx, prob) in enumerate(zip(top_k_indices, top_k_probs)):
                    make = self.labels[idx] if idx < len(self.labels) else f"class_{idx}"
                    predictions.append({
                        'make': make,
                        'confidence': float(prob)
                    })
                
                # Get top prediction
                if predictions:
                    top_make = predictions[0]['make']
                    top_confidence = predictions[0]['confidence']
                else:
                    top_make = 'unknown'
                    top_confidence = 0.0
                
                # Apply confidence threshold
                if top_confidence < self.confidence_threshold:
                    top_make = 'unknown'
                
                # Log processing time and top prediction
                processing_time = time.time() - start_time
                logger.debug(f"Vehicle make detection in {processing_time:.3f}s: "
                           f"{top_make} ({top_confidence:.2f})")
                
                if return_details:
                    return {
                        'make': top_make,
                        'confidence': top_confidence,
                        'predictions': predictions,
                        'processing_time': processing_time
                    }
                else:
                    return top_make
        
        except Exception as e:
            logger.error(f"Error detecting vehicle make: {str(e)}")
            
            if return_details:
                return {
                    'make': 'unknown',
                    'confidence': 0.0,
                    'predictions': [],
                    'error': str(e),
                    'processing_time': time.time() - start_time
                }
            else:
                return 'unknown'
    
    def visualize_prediction(self, image: np.ndarray) -> np.ndarray:
        """
        Create a visualization of the make prediction.
        
        Args:
            image: Input image
            
        Returns:
            np.ndarray: Visualization image
        """
        # Get prediction with details
        result = self.detect_make(image, return_details=True)
        
        # Copy image for visualization
        vis_img = image.copy()
        
        # Draw top prediction
        top_make = result['make']
        top_confidence = result.get('confidence', 0.0)
        
        # Draw result at the top
        cv2.putText(
            vis_img,
            f"Make: {top_make.upper()} ({top_confidence:.2f})",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2,
            cv2.LINE_AA
        )
        
        # Draw top-k predictions at the bottom
        predictions = result.get('predictions', [])
        
        for i, pred in enumerate(predictions):
            y_pos = vis_img.shape[0] - 20 - (i * 30)
            make = pred['make']
            confidence = pred['confidence']
            
            cv2.putText(
                vis_img,
                f"{i+1}. {make.upper()}: {confidence:.2f}",
                (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )
        
        return vis_img

# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create make detector
    detector = MakeDetector(
        model_path=MakeDetector.DEFAULT_MODEL_PATH,
        labels_path=MakeDetector.DEFAULT_LABELS_PATH,
        confidence_threshold=0.5,
        top_k=5
    )
    
    # Load test image (replace with your own)
    img_path = "test_vehicle.jpg"
    img = cv2.imread(img_path)
    
    if img is not None:
        # Detect vehicle make
        result = detector.detect_make(img, return_details=True)
        
        # Print results
        print(f"Top vehicle make: {result['make']} (confidence: {result['confidence']:.2f})")
        print("\nAll predictions:")
        for i, pred in enumerate(result['predictions']):
            print(f"  {i+1}. {pred['make']} ({pred['confidence']:.2f})")
        
        # Create visualization
        vis_img = detector.visualize_prediction(img)
        
        # Display result
        cv2.imshow("Vehicle Make", vis_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print(f"Error: Could not load image from {img_path}")
