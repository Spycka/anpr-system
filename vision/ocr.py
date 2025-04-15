#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
License Plate OCR Module
Performs OCR on license plate images using EasyOCR

This module handles the OCR processing of license plate images,
including preprocessing and character normalization.
"""

import os
import re
import time
import logging
import numpy as np
import cv2
from skew import deskew_plate  # Import from local module


class PlateOCR:
    """
    License Plate OCR class using EasyOCR with preprocessing
    and character normalization optimized for license plates
    """
    
    def __init__(self, confidence_threshold=0.65, lang_list=None, mock=False):
        """
        Initialize the OCR system
        
        Args:
            confidence_threshold (float): Minimum confidence for OCR results
            lang_list (list): List of languages to use (default: ['en'])
            mock (bool): Run in mock mode without actual OCR
        """
        self.logger = logging.getLogger('ocr')
        self.confidence_threshold = confidence_threshold
        self.lang_list = lang_list or ['en']
        self.mock_mode = mock
        
        # Define character normalization map for common OCR errors on plates
        self.char_map = {
            'O': '0',
            'I': '1',
            'Z': '2',
            'B': '8',
            'S': '5',
            'G': '6',
            'T': '7',
            'D': '0',
            'Q': '0'
        }
        
        # Load EasyOCR only if not in mock mode
        if not self.mock_mode:
            self._init_ocr()
        else:
            self.logger.warning("Running in MOCK MODE - OCR will be simulated")
    
    def _init_ocr(self):
        """Initialize EasyOCR reader"""
        self.logger.info(f"Initializing EasyOCR with languages: {self.lang_list}")
        
        try:
            import easyocr
            # Initialize EasyOCR reader (this may take some time)
            self.reader = easyocr.Reader(
                lang_list=self.lang_list,
                gpu=False,  # GPU not supported on Mali
                model_storage_directory=os.path.join(os.path.dirname(__file__), '..', 'models'),
                download_enabled=True
            )
            self.logger.info("EasyOCR initialized successfully")
            
        except ImportError:
            self.logger.error("EasyOCR not installed, falling back to mock mode")
            self.mock_mode = True
        except Exception as e:
            self.logger.error(f"Error initializing EasyOCR: {str(e)}")
            self.logger.warning("Falling back to MOCK MODE")
            self.mock_mode = True
    
    def preprocess_plate(self, plate_img):
        """
        Preprocess license plate image for better OCR results
        
        Args:
            plate_img: License plate image
            
        Returns:
            Preprocessed image
        """
        try:
            # Convert to grayscale if not already
            if len(plate_img.shape) == 3:
                gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
            else:
                gray = plate_img.copy()
            
            # Apply deskewing if the plate is skewed
            gray = deskew_plate(gray)
            
            # Apply adaptive thresholding to handle varying lighting conditions
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, 11, 2
            )
            
            # Remove noise with morphological operations
            kernel = np.ones((3, 3), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            
            # Invert back to black text on white background
            binary = cv2.bitwise_not(binary)
            
            # Resize if too small
            h, w = binary.shape
            if h < 50 or w < 100:
                scale_factor = max(50 / h, 100 / w)
                binary = cv2.resize(binary, None, fx=scale_factor, fy=scale_factor, 
                                   interpolation=cv2.INTER_CUBIC)
            
            return binary
            
        except Exception as e:
            self.logger.error(f"Error in plate preprocessing: {str(e)}")
            return plate_img  # Return original if preprocessing fails
    
    def normalize_plate_text(self, text):
        """
        Normalize the plate text by correcting common OCR errors
        
        Args:
            text: Raw OCR text
            
        Returns:
            Normalized plate text
        """
        # Remove whitespace and convert to uppercase
        text = text.strip().upper()
        
        # Apply character map replacements
        result = ''
        for char in text:
            if char in self.char_map:
                result += self.char_map[char]
            else:
                result += char
        
        # Remove non-alphanumeric characters except hyphen and space
        result = re.sub(r'[^A-Z0-9\- ]', '', result)
        
        # Remove extra spaces
        result = re.sub(r'\s+', ' ', result).strip()
        
        return result
    
    def read_plate(self, plate_img):
        """
        Perform OCR on a license plate image
        
        Args:
            plate_img: License plate image
            
        Returns:
            tuple: (plate_text, confidence)
        """
        # Check if image is valid
        if plate_img is None or plate_img.size == 0:
            return '', 0.0
        
        # Mock mode for testing without OCR
        if self.mock_mode:
            # Simulate processing delay
            time.sleep(0.1)
            
            # Generate random plate in format: XXX-1234
            letters = 'ABCDEFGHJKLMNPRSTUVWXYZ'  # Skip confusing letters
            numbers = '0123456789'
            
            plate = ''
            for _ in range(3):
                plate += letters[np.random.randint(0, len(letters))]
            
            plate += '-'
            
            for _ in range(4):
                plate += numbers[np.random.randint(0, len(numbers))]
            
            confidence = np.random.uniform(0.7, 0.95)
            return plate, confidence
        
        try:
            # Preprocess plate image
            preprocessed_img = self.preprocess_plate(plate_img)
            
            # Perform OCR
            results = self.reader.readtext(preprocessed_img)
            
            if not results:
                return '', 0.0
            
            # Process results
            text_parts = []
            confidence_sum = 0
            
            for (_, text, confidence) in results:
                if confidence > self.confidence_threshold:
                    text_parts.append(text)
                    confidence_sum += confidence
            
            # Combine all detected text parts
            combined_text = ' '.join(text_parts)
            
            # Calculate average confidence
            avg_confidence = confidence_sum / len(results) if results else 0
            
            # Normalize the plate text
            normalized_text = self.normalize_plate_text(combined_text)
            
            return normalized_text, avg_confidence
            
        except Exception as e:
            self.logger.error(f"Error in OCR processing: {str(e)}")
            return '', 0.0
