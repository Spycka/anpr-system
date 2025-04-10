#!/usr/bin/env python3
"""
ANPR System for Orange Pi 5 Ultra

Main application module. Implements real-time license plate detection,
OCR, and gate control based on allowlist.
"""
import os
import sys
import time
import argparse
import json
import signal
import threading
import queue
from typing import Dict, Any, List, Optional, Tuple, Union
import logging
import cv2
import numpy as np

# Import project modules
from utils.logger import setup_logging, get_logger
from utils.hardware import HardwareDetector, AcceleratorType
from utils.plate_checker import PlateChecker
from utils.gpio import get_gpio, GPIOMode, GPIOState
from utils.vehicle_color import ColorDetector
from utils.vehicle_make import MakeDetector
from input.camera import CameraInput
from vision.ocr import PlateOCR
from detectors.yolo11_gpu import YOLO11sGPU
from detectors.yolo11_rknn import YOLO11sRKNN

# Set up logging
logger = get_logger("main")

class ANPRSystem:
    """
    Main ANPR System for Orange Pi 5 Ultra.
    
    Coordinates all components of the Automatic Number Plate Recognition system:
    - Camera input
    - License plate detection
    - OCR
    - Vehicle make detection (optional)
    - Vehicle color detection (optional)
    - Plate validation and gate control
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize ANPR system.
        
        Args:
            config: Configuration dictionary with the following keys:
                - rtsp_url: RTSP stream URL
                - resolution: Stream resolution
                - save_dir: Directory to save captured plates
                - allowlist: Path to allowlist file
                - enable_make: Whether to enable vehicle make detection
                - enable_color: Whether to enable vehicle color detection
                - mock_gpio: Whether to use mock GPIO
                - show_video: Whether to show video window
                - show_debug: Whether to show debug information
                - log_dir: Directory for log files
                - simulate_dir: Directory for simulation images (None for live camera)
        """
        self.config = config
        self.running = False
        self.processing_lock = threading.Lock()
        
        # Create output directory
        if not os.path.exists(config['save_dir']):
            os.makedirs(config['save_dir'])
        
        # Detect available hardware
        self.hardware_detector = HardwareDetector()
        self.best_accelerator, accelerator_info = self.hardware_detector.get_best_available_accelerator()
        
        logger.info(f"Best available accelerator: {self.best_accelerator.name}")
        logger.info(f"Hardware info: {accelerator_info}")
        
        # Initialize camera
        self._init_camera()
        
        # Initialize detector (RKNN if available, otherwise GPU/CPU)
        self._init_detector()
        
        # Initialize OCR
        self._init_ocr()
        
        # Initialize optional components
        self._init_optional_components()
        
        # Initialize plate checker
        self.plate_checker = PlateChecker(
            allowlist_path=config['allowlist'],
            mock_gpio=config['mock_gpio'],
            enable_make_verification=config['enable_make'],
            enable_color_verification=config['enable_color']
        )
        
        # Performance metrics
        self.metrics = {
            'processed_frames': 0,
            'detected_plates': 0,
            'matched_plates': 0,
            'avg_processing_time': 0,
            'last_recognized_plate': '',
            'start_time': time.time()
        }
        
        # Visualization settings
        self.show_video = config['show_video']
        self.show_debug = config['show_debug']
        
        logger.info("ANPR system initialized successfully")
    
    def _init_camera(self) -> None:
        """Initialize camera input."""
        # Check if we're in simulation mode
        simulate_mode = (self.config.get('simulate_dir') is not None)
        
        # Create camera input
        self.camera = CameraInput(
            rtsp_url=self.config['rtsp_url'],
            resolution=self.config['resolution'],
            buffer_size=5,
            simulate_mode=simulate_mode,
            simulate_dir=self.config.get('simulate_dir')
        )
        
        logger.info(f"Camera initialized (simulate_mode={simulate_mode})")
    
    def _init_detector(self) -> None:
        """Initialize license plate detector."""
        # Use RKNN if NPU is available
        if self.best_accelerator == AcceleratorType.NPU:
            logger.info("Initializing RKNN detector for NPU acceleration")
            
            self.detector = YOLO11sRKNN(
                model_path=os.path.join("models", "YOLO11s.rknn"),
                conf_threshold=0.5,
                iou_threshold=0.45
            )
        else:
            logger.info("Initializing GPU/CPU detector")
            
            self.detector = YOLO11sGPU(
                model_path=os.path.join("models", "YOLO11s.pt"),
                conf_threshold=0.5,
                iou_threshold=0.45,
                device=None  # Auto-detect
            )
    
    def _init_ocr(self) -> None:
        """Initialize OCR module."""
        self.ocr = PlateOCR(
            languages=['en'],
            min_confidence=0.6,
            skew_correction=True,
            enable_gpu=(self.best_accelerator == AcceleratorType.GPU),
            preprocessing_method="adaptive",
            text_cleanup=True,
            verbose=self.config['show_debug']
        )
    
    def _init_optional_components(self) -> None:
        """Initialize optional components."""
        # Vehicle make detection
        if self.config['enable_make']:
            self.make_detector = MakeDetector(
                model_path=os.path.join("models", "resnet18_vehicle_make.pth"),
                confidence_threshold=0.6,
                device=None  # Auto-detect
            )
            logger.info("Vehicle make detection enabled")
        else:
            self.make_detector = None
            logger.info("Vehicle make detection disabled")
        
        # Vehicle color detection
        if self.config['enable_color']:
            self.color_detector = ColorDetector(
                n_clusters=5,
                verbose=self.config['show_debug']
            )
            logger.info("Vehicle color detection enabled")
        else:
            self.color_detector = None
            logger.info("Vehicle color detection disabled")
    
    def start(self) -> None:
        """Start the ANPR system."""
        if self.running:
            logger.warning("ANPR system already running")
            return
        
        # Start camera
        self.camera.start()
        
        # Set running flag
        self.running = True
        
        logger.info("ANPR system started")
        
        # Main processing loop
        try:
            while self.running:
                self._process_frame()
                
                # Handle display
                if self.show_video:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        logger.info("User requested exit")
                        break
        
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        
        finally:
            self.stop()
    
    def stop(self) -> None:
        """Stop the ANPR system."""
        if not self.running:
            return
        
        self.running = False
        
        # Stop camera
        self.camera.stop()
        
        # Close windows
        if self.show_video:
            cv2.destroyAllWindows()
        
        # Log metrics
        runtime = time.time() - self.metrics['start_time']
        fps = self.metrics['processed_frames'] / runtime if runtime > 0 else 0
        
        logger.info(f"ANPR system stopped")
        logger.info(f"Runtime: {runtime:.1f}s, Processed frames: {self.metrics['processed_frames']}")
        logger.info(f"Average FPS: {fps:.2f}")
        logger.info(f"Detected plates: {self.metrics['detected_plates']}")
        logger.info(f"Matched plates: {self.metrics['matched_plates']}")
    
    def _process_frame(self) -> None:
        """Process a single frame."""
        # Skip if already processing (avoid overloading)
        if self.processing_lock.locked():
            return
        
        # Get frame from camera
        frame = self.camera.get_frame(timeout=0.1)
        
        if frame is None:
            return
        
        # Acquire lock for processing
        with self.processing_lock:
            try:
                start_time = time.time()
                
                # Update metrics
                self.metrics['processed_frames'] += 1
                
                # Detect license plates
                plate_detections = self.detector.detect(frame, return_details=True)
                
                # Skip if no plates detected
                if not plate_detections:
                    # Display frame if requested
                    if self.show_video:
                        self._display_frame(frame, [], None)
                    return
                
                # Extract plate regions
                plates = self.detector.extract_plates(frame, margin=0.1)
                
                if not plates:
                    return
                
                # Update metrics
                self.metrics['detected_plates'] += len(plates)
                
                # Process each plate
                results = []
                
                for plate_info in plates:
                    result = self._process_plate(frame, plate_info)
                    if result:
                        results.append(result)
                
                # Check if any plates matched
                matched_plates = [r for r in results if r.get('match', False)]
                if matched_plates:
                    self.metrics['matched_plates'] += 1
                
                # Update average processing time
                processing_time = time.time() - start_time
                self.metrics['avg_processing_time'] = (
                    (self.metrics['avg_processing_time'] * (self.metrics['processed_frames'] - 1) + 
                     processing_time) / self.metrics['processed_frames']
                )
                
                # Display results
                if self.show_video:
                    self._display_frame(frame, results, processing_time)
            
            except Exception as e:
                logger.error(f"Error processing frame: {str(e)}")
    
    def _process_plate(self, frame: np.ndarray, plate_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process a detected license plate.
        
        Args:
            frame: Original frame
            plate_info: Plate detection info
            
        Returns:
            Optional[Dict[str, Any]]: Processing result or None on failure
        """
        try:
            # Extract info
            plate_img = plate_info['image']
            box = plate_info['box']
            confidence = plate_info['confidence']
            
            # Skip if confidence is too low
            if confidence < 0.5:
                return None
            
            # Run OCR
            ocr_result = self.ocr.recognize(plate_img, return_details=True)
            plate_text = ocr_result['text']
            ocr_confidence = ocr_result['confidence']
            
            # Skip if OCR confidence is too low or text is empty
            if ocr_confidence < 0.6 or not plate_text:
                return None
            
            # Prepare result
            result = {
                'plate_text': plate_text,
                'box': box,
                'det_confidence': confidence,
                'ocr_confidence': ocr_confidence,
                'match': False
            }
            
            # Perform vehicle make detection (if enabled)
            if self.make_detector is not None:
                # Get vehicle region (use whole frame, or enlarge plate area)
                x1, y1, x2, y2 = box
                
                # Extract a larger region for the vehicle
                height, width = frame.shape[:2]
                
                # Expand box upward and to sides (vehicles are usually above the plate)
                plate_width = x2 - x1
                plate_height = y2 - y1
                
                # Create vehicle box (above the plate)
                veh_x1 = max(0, x1 - plate_width)
                veh_y1 = max(0, y1 - plate_height * 4)  # Expand upward
                veh_x2 = min(width, x2 + plate_width)
                veh_y2 = min(height, y2)
                
                # Extract vehicle image
                vehicle_img = frame[veh_y1:veh_y2, veh_x1:veh_x2]
                
                # Skip if vehicle image is too small
                if vehicle_img.size > 0 and vehicle_img.shape[0] > 50 and vehicle_img.shape[1] > 50:
                    # Detect make
                    make_result = self.make_detector.detect_make(vehicle_img, return_details=True)
                    
                    # Add to result
                    result['vehicle_make'] = make_result['make']
                    result['make_confidence'] = make_result['confidence']
            
            # Perform vehicle color detection (if enabled)
            if self.color_detector is not None:
                # Use the same vehicle region as for make detection
                if 'vehicle_img' in locals() and vehicle_img.size > 0:
                    # Detect color
                    color_result = self.color_detector.detect_color(
                        vehicle_img, 
                        return_details=True
                    )
                    
                    # Add to result
                    result['vehicle_color'] = color_result['main_color']
            
            # Check against allowlist
            make = result.get('vehicle_make', None)
            color = result.get('vehicle_color', None)
            
            match, details = self.plate_checker.check_plate(
                plate_text, 
                make=make, 
                color=color
            )
            
            # Add match result
            result['match'] = match
            result['match_details'] = details
            
            # Save plate image if it's a match or for debugging
            if match or self.config['show_debug']:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"{plate_text}_{timestamp}.jpg"
                save_path = os.path.join(self.config['save_dir'], filename)
                
                # Save plate image
                cv2.imwrite(save_path, plate_img)
                
                # Save metadata
                meta_path = os.path.join(
                    self.config['save_dir'], 
                    f"{plate_text}_{timestamp}.json"
                )
                
                with open(meta_path, 'w') as f:
                    # Create serializable copy
                    meta = result.copy()
                    # Remove non-serializable items
                    if 'vehicle_img' in meta:
                        del meta['vehicle_img']
                    
                    json.dump(meta, f, indent=2)
                
                logger.info(f"Saved plate image and metadata: {save_path}")
            
            # Update last recognized plate
            if plate_text:
                self.metrics['last_recognized_plate'] = plate_text
            
            return result
        
        except Exception as e:
            logger.error(f"Error processing plate: {str(e)}")
            return None
    
    def _display_frame(self, frame: np.ndarray, 
                       results: List[Dict[str, Any]],
                       processing_time: Optional[float]) -> None:
        """
        Display frame with results.
        
        Args:
            frame: Original frame
            results: Processing results
            processing_time: Processing time in seconds
        """
        # Create copy for display
        display = frame.copy()
        
        # Draw detection boxes and results
        for result in results:
            # Get info
            box = result['box']
            plate_text = result['plate_text']
            match = result['match']
            ocr_confidence = result.get('ocr_confidence', 0.0)
            
            # Draw box with color based on match
            color = (0, 255, 0) if match else (0, 0, 255)
            x1, y1, x2, y2 = box
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
            
            # Draw plate text
            text = f"{plate_text} ({ocr_confidence:.2f})"
            cv2.putText(
                display, text, (x1, y1 - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )
            
            # Draw additional info if available
            if 'vehicle_make' in result:
                make_text = f"Make: {result['vehicle_make']}"
                cv2.putText(
                    display, make_text, (x1, y2 + 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2
                )
            
            if 'vehicle_color' in result:
                color_text = f"Color: {result['vehicle_color']}"
                cv2.putText(
                    display, color_text, (x1, y2 + 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2
                )
            
            # Draw match status
            match_text = "MATCH" if match else "NO MATCH"
            match_color = (0, 255, 0) if match else (0, 0, 255)
            cv2.putText(
                display, match_text, (x1, y2 + 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, match_color, 2
            )
        
        # Draw performance info
        if self.show_debug and processing_time is not None:
            # Calculate FPS
            fps = 1.0 / processing_time if processing_time > 0 else 0
            
            # Draw FPS
            cv2.putText(
                display, f"FPS: {fps:.1f}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
            )
            
            # Draw processing time
            cv2.putText(
                display, f"Processing: {processing_time * 1000:.1f} ms", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
            )
            
            # Draw frame count
            cv2.putText(
                display, f"Frames: {self.metrics['processed_frames']}", (10, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
            )
            
            # Draw detections count
            cv2.putText(
                display, f"Detections: {self.metrics['detected_plates']}", (10, 120), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
            )
            
            # Draw matches count
            cv2.putText(
                display, f"Matches: {self.metrics['matched_plates']}", (10, 150), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
            )
        
        # Show last recognized plate
        if self.metrics['last_recognized_plate']:
            cv2.putText(
                display, f"Last: {self.metrics['last_recognized_plate']}", 
                (display.shape[1] - 300, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2
            )
        
        # Display the frame
        cv2.imshow("ANPR System", display)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="ANPR System for Orange Pi 5 Ultra")
    
    # Camera settings
    parser.add_argument("--rtsp-url", type=str, default="rtsp://username:password@camera-ip:554/stream",
                      help="RTSP stream URL")
    parser.add_argument("--resolution", type=str, default="720p",
                      help="Stream resolution (480p, 720p, 1080p, or WIDTHxHEIGHT)")
    
    # Feature flags
    parser.add_argument("--enable-color", action="store_true",
                      help="Enable vehicle color detection")
    parser.add_argument("--enable-make", action="store_true",
                      help="Enable vehicle make detection")
    
    # File paths
    parser.add_argument("--allowlist", type=str, default="allowlist.txt",
                      help="Path to allowlist file")
    parser.add_argument("--save-dir", type=str, default="captures",
                      help="Directory to save captured plates")
    parser.add_argument("--log-dir", type=str, default="logs",
                      help="Directory for log files")
    
    # Display options
    parser.add_argument("--show-video", action="store_true",
                      help="Show video window")
    parser.add_argument("--show-debug", action="store_true",
                      help="Show debug information")
    
    # Testing options
    parser.add_argument("--simulate", type=str, default=None,
                      help="Use simulation images from directory instead of camera")
    parser.add_argument("--mock-gpio", action="store_true",
                      help="Use mock GPIO (for testing)")
    
    return parser.parse_args()

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Set up logging
    setup_logging(log_dir=args.log_dir, log_to_console=True)
    
    # Create configuration
    config = {
        'rtsp_url': args.rtsp_url,
        'resolution': args.resolution,
        'save_dir': args.save_dir,
        'allowlist': args.allowlist,
        'enable_make': args.enable_make,
        'enable_color': args.enable_color,
        'mock_gpio': args.mock_gpio,
        'show_video': args.show_video,
        'show_debug': args.show_debug,
        'log_dir': args.log_dir,
        'simulate_dir': args.simulate
    }
    
    # Log configuration
    logger.info("Starting ANPR system with configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    
    # Create and start system
    try:
        system = ANPRSystem(config)
        
        # Handle graceful shutdown
        def signal_handler(sig, frame):
            logger.info("Shutdown signal received")
            system.stop()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Start system
        system.start()
    
    except Exception as e:
        logger.error(f"Error in ANPR system: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
