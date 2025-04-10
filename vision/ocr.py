#!/usr/bin/env python3
"""
OCR module for license plate recognition.
Includes preprocessing, skew correction, and text normalization.
"""
import os
import re
import time
import numpy as np
import cv2
from typing import List, Tuple, Dict, Any, Optional, Union
import logging
import threading

# Try importing EasyOCR
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

# Import project modules
from utils.logger import get_logger
from vision.skew import SkewCorrector, preprocess_for_ocr
from utils.hardware import AcceleratorType, HardwareDetector

# Configure logger
logger = get_logger("ocr")

class PlateOCR:
    """
    Enhanced OCR for license plates with preprocessing and text normalization.
    """
    
    # Default values
    DEFAULT_LANGUAGES = ['en']
    DEFAULT_MIN_CONFIDENCE = 0.6
    
    # Normalization mapping for confusing characters
    CHAR_NORMALIZATION = {
        'O': '0',  # O -> 0
        'I': '1',  # I -> 1
        'Z': '2',  # Z -> 2 (optional)
        'A': '4',  # A -> 4 (optional)
        'S': '5',  # S -> 5 (optional)
        'G': '6',  # G -> 6 (optional)
        'T': '7',  # T -> 7 (optional)
        'B': '8',  # B -> 8 (optional)
        # Add more if needed
    }
    
    # Character sets for different plate formats
    ALPHANUMERIC_CHARS = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
    NUMERIC_CHARS = set('0123456789')
    
    def __init__(
        self,
        languages: List[str] = None,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
        skew_correction: bool = True,
        enable_gpu: bool = True,
        preprocessing_method: str = "adaptive",
        text_cleanup: bool = True,
        verbose: bool = False
    ):
        """
        Initialize PlateOCR.
        
        Args:
            languages: List of language codes for OCR
            min_confidence: Minimum confidence threshold for OCR results
            skew_correction: Whether to apply skew correction
            enable_gpu: Whether to use GPU acceleration if available
            preprocessing_method: Image preprocessing method
            text_cleanup: Whether to apply text normalization
            verbose: Enable verbose logging
        """
        self.languages = languages or self.DEFAULT_LANGUAGES
        self.min_confidence = min_confidence
        self.use_skew_correction = skew_correction
        self.enable_gpu = enable_gpu
        self.preprocessing_method = preprocessing_method
        self.text_cleanup = text_cleanup
        self.verbose = verbose
        
        # Initialize hardware detection
        self.hardware_detector = HardwareDetector()
        self.best_accelerator, _ = self.hardware_detector.get_best_available_accelerator()
        
        # Enable GPU if available and requested
        self.gpu_available = (self.best_accelerator == AcceleratorType.GPU) and enable_gpu
        
        # Skew corrector
        self.skew_corrector = SkewCorrector() if skew_correction else None
        
        # OCR reader (initialize lazily)
        self.reader = None
        self.reader_lock = threading.Lock()
        
        # Check if EasyOCR is available
        if not EASYOCR_AVAILABLE:
            logger.warning("EasyOCR not found. Please install: pip install easyocr")
        
        logger.info(f"PlateOCR initialized (languages={self.languages}, "
                   f"GPU={'enabled' if self.gpu_available else 'disabled'}, "
                   f"skew_correction={'enabled' if skew_correction else 'disabled'})")
    
    def _ensure_reader_initialized(self) -> bool:
        """
        Ensure OCR reader is initialized.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        if self.reader is not None:
            return True
        
        with self.reader_lock:
            if self.reader is not None:
                return True
            
            if not EASYOCR_AVAILABLE:
                logger.error("Cannot initialize OCR reader: EasyOCR not installed")
                return False
            
            try:
                logger.info(f"Initializing EasyOCR (languages={self.languages}, "
                          f"GPU={self.gpu_available})")
                
                start_time = time.time()
                self.reader = easyocr.Reader(
                    self.languages,
                    gpu=self.gpu_available,
                    model_storage_directory=os.path.join(os.getcwd(), "models", "easyocr"),
                    download_enabled=True
                )
                
                logger.info(f"EasyOCR initialized in {time.time() - start_time:.2f} seconds")
                return True
            
            except Exception as e:
                logger.error(f"Failed to initialize EasyOCR: {str(e)}")
                return False
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for OCR.
        
        Args:
            image: Input image
            
        Returns:
            np.ndarray: Preprocessed image
        """
        # Apply skew correction if enabled
        if self.use_skew_correction and self.skew_corrector is not None:
            corrected, angle, confidence = self.skew_corrector.correct(image)
            
            if self.verbose:
                logger.debug(f"Skew correction: angle={angle:.2f}°, confidence={confidence:.2f}")
            
            # Use corrected image if confidence is good
            if confidence > 0.6:
                image = corrected
        
        # Apply preprocessing
        processed = preprocess_for_ocr(image, self.preprocessing_method)
        
        return processed
    
    def _clean_text(self, text: str, is_plate: bool = True) -> str:
        """
        Clean and normalize OCR text.
        
        Args:
            text: Raw OCR text
            is_plate: Whether the text is a license plate (applies different normalization)
            
        Returns:
            str: Cleaned text
        """
        if not text:
            return ""
        
        # Remove whitespace
        cleaned = text.strip().upper()
        
        # Skip further processing if not enabled
        if not self.text_cleanup:
            return cleaned
        
        if is_plate:
            # Remove spaces and special characters, keep only alphanumeric
            cleaned = re.sub(r'[^A-Z0-9]', '', cleaned)
            
            # Apply character normalization for plates
            for char, replacement in self.CHAR_NORMALIZATION.items():
                cleaned = cleaned.replace(char, replacement)
        
        return cleaned
    
    def recognize(self, image: np.ndarray, 
                  return_details: bool = False,
                  try_all_preprocessing: bool = False) -> Union[str, Dict[str, Any]]:
        """
        Recognize text in license plate image.
        
        Args:
            image: Input image
            return_details: If True, return detailed information
            try_all_preprocessing: If True, try multiple preprocessing methods
            
        Returns:
            Union[str, Dict[str, Any]]: 
                - If return_details=False: Recognized plate text
                - If return_details=True: Dict with text, confidence, and processing details
        """
        # Ensure reader is initialized
        if not self._ensure_reader_initialized():
            if return_details:
                return {'text': '', 'confidence': 0.0, 'error': 'OCR not initialized'}
            else:
                return ''
        
        start_time = time.time()
        preprocessing_results = {}
        best_text = ''
        best_confidence = 0.0
        best_method = ''
        
        try:
            if try_all_preprocessing:
                # Try multiple preprocessing methods
                preprocessing_methods = ['adaptive', 'otsu', 'canny', 'laplacian']
                
                for method in preprocessing_methods:
                    # Save current method
                    current_method = self.preprocessing_method
                    
                    # Set current method
                    self.preprocessing_method = method
                    
                    # Preprocess image
                    processed = self._preprocess_image(image)
                    
                    # Perform OCR
                    results = self.reader.readtext(processed)
                    
                    # Restore original method
                    self.preprocessing_method = current_method
                    
                    # Process results
                    if results:
                        # Sort by confidence
                        results.sort(key=lambda x: x[2], reverse=True)
                        
                        # Get highest confidence result
                        _, text, confidence = results[0]
                        
                        # Clean text
                        cleaned_text = self._clean_text(text)
                        
                        preprocessing_results[method] = {
                            'text': cleaned_text,
                            'confidence': confidence
                        }
                        
                        # Update best result
                        if confidence > best_confidence:
                            best_text = cleaned_text
                            best_confidence = confidence
                            best_method = method
            
            else:
                # Use single preprocessing method
                processed = self._preprocess_image(image)
                
                # Perform OCR
                results = self.reader.readtext(processed)
                
                # Process results
                if results:
                    # Sort by confidence
                    results.sort(key=lambda x: x[2], reverse=True)
                    
                    # Get highest confidence result
                    _, text, confidence = results[0]
                    
                    # Clean text
                    best_text = self._clean_text(text)
                    best_confidence = confidence
                    best_method = self.preprocessing_method
            
            # Log results
            processing_time = time.time() - start_time
            
            if self.verbose or best_confidence > self.min_confidence:
                logger.info(f"OCR result: '{best_text}' (confidence: {best_confidence:.2f}, "
                          f"method: {best_method}, time: {processing_time:.3f}s)")
            
            # Return results
            if return_details:
                return {
                    'text': best_text,
                    'confidence': best_confidence,
                    'method': best_method,
                    'processing_time': processing_time,
                    'all_results': preprocessing_results if try_all_preprocessing else None
                }
            else:
                # Only return text if confidence meets threshold
                if best_confidence >= self.min_confidence:
                    return best_text
                else:
                    return ''
        
        except Exception as e:
            logger.error(f"OCR error: {str(e)}")
            
            if return_details:
                return {
                    'text': '',
                    'confidence': 0.0,
                    'error': str(e),
                    'processing_time': time.time() - start_time
                }
            else:
                return ''
    
    def recognize_batch(self, images: List[np.ndarray], 
                         min_confidence: float = None) -> List[Dict[str, Any]]:
        """
        Recognize text in a batch of license plate images.
        
        Args:
            images: List of input images
            min_confidence: Override default confidence threshold
            
        Returns:
            List[Dict[str, Any]]: List of recognition results
        """
        if min_confidence is None:
            min_confidence = self.min_confidence
        
        results = []
        
        for i, image in enumerate(images):
            try:
                # Recognize text
                result = self.recognize(image, return_details=True)
                
                # Add image index
                result['image_index'] = i
                
                # Add to results if confidence is high enough
                if result['confidence'] >= min_confidence:
                    results.append(result)
            
            except Exception as e:
                logger.error(f"Error in batch recognition for image {i}: {str(e)}")
        
        return results

# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create PlateOCR instance
    ocr = PlateOCR(
        languages=['en'],
        enable_gpu=True,
        verbose=True
    )
    
    # Load test image (replace with your own)
    img_path = "test_plate.jpg"
    img = cv2.imread(img_path)
    
    if img is not None:
        # Recognize text
        result = ocr.recognize(img, return_details=True, try_all_preprocessing=True)
        
        # Print results
        print(f"OCR Result:")
        print(f"  Text: {result['text']}")
        print(f"  Confidence: {result['confidence']:.2f}")
        print(f"  Method: {result['method']}")
        print(f"  Processing time: {result['processing_time']:.3f}s")
        
        if result['all_results']:
            print("\nAll preprocessing methods:")
            for method, res in result['all_results'].items():
                print(f"  {method}: '{res['text']}' (confidence: {res['confidence']:.2f})")
    else:
        print(f"Error: Could not load image from {img_path}")
