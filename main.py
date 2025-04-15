#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ANPR System - Main Application
License Plate Recognition System for Access Control
For Orange Pi 5 Ultra with RK3588 NPU

This is the main CLI entry point that handles the core functionality:
- Camera input processing
- License plate detection using YOLO11s on RK3588 NPU
- OCR for license plate text recognition
- Access control based on allowlist verification
- GPIO control for gate activation
"""

import os
import sys
import time
import argparse
import threading
import queue
import signal
import logging
from datetime import datetime

# Import local modules
from detectors.yolo11_rknn import LicensePlateDetector
from vision.ocr import PlateOCR
from vision.vehicle_classifier import VehicleClassifier
from input.camera import CameraInput
from utils.plate_checker import PlateChecker
from utils.gpio import GateController
from utils.logger import setup_logger

# Global flag for clean shutdown
running = True

def signal_handler(sig, frame):
    """Handle interrupt signals gracefully"""
    global running
    logging.info("Shutdown signal received, stopping ANPR system...")
    running = False

class ANPRSystem:
    """Main ANPR System class that orchestrates all components"""
    
    def __init__(self, config):
        """Initialize the ANPR system with configuration"""
        self.config = config
        self.setup_logging()
        
        # Initialize components
        self.logger.info("Initializing ANPR system components...")
        
        # Initialize camera
        self.camera = CameraInput(
            source=config['camera_source'],
            buffer_size=config['buffer_size'],
            width=config['frame_width'],
            height=config['frame_height'],
            mock=config['mock_mode']
        )
        
        # Initialize license plate detector (YOLO11s on RKNN)
        self.detector = LicensePlateDetector(
            model_path=config['model_path'],
            conf_threshold=config['detection_confidence'],
            nms_threshold=config['nms_threshold'],
            mock=config['mock_mode']
        )
        
        # Initialize OCR
        self.ocr = PlateOCR(
            confidence_threshold=config['ocr_confidence']
        )
        
        # Initialize vehicle classifier if enabled
        if config['enable_vehicle_classification']:
            self.vehicle_classifier = VehicleClassifier(
                model_path=config['vehicle_make_model_path'],
                enable_make=config['enable_make_detection'],
                enable_color=config['enable_color_detection'],
                mock=config['mock_mode']
            )
        else:
            self.vehicle_classifier = None
        
        # Initialize plate checker (allowlist verification)
        self.plate_checker = PlateChecker(
            allowlist_path=config['allowlist_path'],
            auto_reload=True
        )
        
        # Initialize gate controller
        self.gate_controller = GateController(
            pin=config['gpio_pin'],
            pulse_time=config['gate_pulse_time'],
            cooldown_time=config['gate_cooldown_time'],
            mock=config['mock_mode']
        )
        
        # Initialize queues for thread communication
        self.detection_queue = queue.Queue(maxsize=5)
        self.ocr_queue = queue.Queue(maxsize=10)
        self.last_processed_plates = {}
        
        self.logger.info("ANPR system initialization complete")
    
    def setup_logging(self):
        """Setup logging for all components"""
        log_dir = self.config['log_dir']
        os.makedirs(log_dir, exist_ok=True)
        
        # Setup system logger
        self.logger = setup_logger(
            'system', 
            os.path.join(log_dir, 'system.log'),
            log_level=logging.INFO if not self.config['debug'] else logging.DEBUG
        )
        
        # Setup detection logger
        self.detection_logger = setup_logger(
            'detection', 
            os.path.join(log_dir, 'detection.log')
        )
        
        # Setup OCR logger
        self.ocr_logger = setup_logger(
            'ocr', 
            os.path.join(log_dir, 'ocr.log')
        )
        
        # Setup access logger
        self.access_logger = setup_logger(
            'plate_checker', 
            os.path.join(log_dir, 'plate_checker.log')
        )

    def camera_thread_func(self):
        """Thread function for camera capture and frame processing"""
        self.logger.info("Starting camera thread")
        fail_count = 0
        
        while running:
            try:
                # Get frame from camera
                success, frame = self.camera.read()
                
                if not success:
                    fail_count += 1
                    if fail_count > 5:
                        self.logger.error("Failed to read from camera after multiple attempts")
                        time.sleep(2)  # Wait before retrying
                    continue
                
                fail_count = 0  # Reset fail counter on success
                
                # Check if detection queue is full
                if self.detection_queue.full():
                    # Skip frame if queue is backed up
                    continue
                
                # Add frame to detection queue
                self.detection_queue.put(frame, block=False)
                time.sleep(1.0 / self.config['processing_fps'])  # Throttle processing
                
            except queue.Full:
                # Queue is full, skip this frame
                pass
            except Exception as e:
                self.logger.error(f"Error in camera thread: {str(e)}")
                time.sleep(0.5)  # Avoid tight loop on error
    
    def detection_thread_func(self):
        """Thread function for license plate detection using RKNN"""
        self.logger.info("Starting detection thread")
        
        while running:
            try:
                # Get frame from queue
                frame = self.detection_queue.get(timeout=1.0)
                
                # Run detection
                start_time = time.time()
                detections = self.detector.detect(frame)
                detection_time = time.time() - start_time
                
                # Log detection performance
                self.detection_logger.info(f"Detection time: {detection_time:.4f}s, Detections: {len(detections)}")
                
                # Process each detection
                for i, detection in enumerate(detections):
                    x1, y1, x2, y2, confidence, _ = detection
                    
                    # Extract plate region (with some margin)
                    margin = 5
                    h, w = frame.shape[:2]
                    x1, y1 = max(0, int(x1) - margin), max(0, int(y1) - margin)
                    x2, y2 = min(w, int(x2) + margin), min(h, int(y2) + margin)
                    
                    plate_img = frame[y1:y2, x1:x2]
                    
                    # Skip if plate is too small
                    if plate_img.size == 0 or plate_img.shape[0] < 20 or plate_img.shape[1] < 40:
                        continue
                    
                    # Put in OCR queue
                    if not self.ocr_queue.full():
                        self.ocr_queue.put((plate_img, confidence, time.time()))
                
                # Mark task as done
                self.detection_queue.task_done()
                
            except queue.Empty:
                # No frames available, just continue
                pass
            except Exception as e:
                self.logger.error(f"Error in detection thread: {str(e)}")
                time.sleep(0.5)  # Avoid tight loop on error
    
    def ocr_thread_func(self):
        """Thread function for OCR processing and access control"""
        self.logger.info("Starting OCR thread")
        
        captures_dir = self.config['captures_dir']
        os.makedirs(captures_dir, exist_ok=True)
        
        while running:
            try:
                # Get plate image from queue
                plate_img, confidence, timestamp = self.ocr_queue.get(timeout=1.0)
                
                # Run OCR
                start_time = time.time()
                plate_text, ocr_confidence = self.ocr.read_plate(plate_img)
                ocr_time = time.time() - start_time
                
                # Log OCR results
                self.ocr_logger.info(
                    f"OCR time: {ocr_time:.4f}s, Plate: {plate_text}, Confidence: {ocr_confidence:.2f}"
                )
                
                # Skip if confidence is too low or plate text is empty
                if not plate_text or ocr_confidence < self.config['ocr_confidence']:
                    self.ocr_queue.task_done()
                    continue
                
                # Check if we processed this plate recently (debounce)
                current_time = time.time()
                if plate_text in self.last_processed_plates:
                    last_time = self.last_processed_plates[plate_text]
                    if current_time - last_time < self.config['plate_debounce_time']:
                        self.ocr_queue.task_done()
                        continue
                
                # Update last processed time
                self.last_processed_plates[plate_text] = current_time
                
                # Run vehicle classification if enabled
                make = None
                color = None
                if self.vehicle_classifier is not None:
                    # Get the full frame (vehicle) for classification
                    # For this we'd typically need to store the original frame with the plate detection
                    # but for simplicity, we'll use the plate image itself
                    make, make_confidence, color, color_confidence = self.vehicle_classifier.classify_vehicle(plate_img)
                    
                    # Only use results if confidence is high enough
                    if make_confidence < self.config['make_confidence_threshold']:
                        make = None
                    if color_confidence < self.config['color_confidence_threshold']:
                        color = None
                    
                    self.logger.info(f"Vehicle classification for {plate_text}: Make={make or 'unknown'}, Color={color or 'unknown'}")
                
                # Check if plate is in allowlist with vehicle attributes
                is_authorized = self.plate_checker.check_plate(plate_text, make, color)
                
                # Save plate image with timestamp and recognition result
                timestamp_str = datetime.fromtimestamp(timestamp).strftime("%Y%m%d_%H%M%S")
                status = "authorized" if is_authorized else "unauthorized"
                filename = f"{timestamp_str}_{plate_text}_{status}.jpg"
                file_path = os.path.join(captures_dir, filename)
                
                try:
                    import cv2
                    cv2.imwrite(file_path, plate_img)
                except Exception as e:
                    self.logger.error(f"Failed to save plate image: {str(e)}")
                
                # Log access control decision with additional attributes
                attributes = []
                if make:
                    attributes.append(f"make={make}")
                if color:
                    attributes.append(f"color={color}")
                
                attr_str = f" with {', '.join(attributes)}" if attributes else ""
                self.access_logger.info(
                    f"Plate: {plate_text}{attr_str}, Authorized: {is_authorized}, Confidence: {ocr_confidence:.2f}"
                )
                
                # Open gate if plate is authorized
                if is_authorized:
                    self.logger.info(f"Opening gate for authorized plate: {plate_text}{attr_str}")
                    self.gate_controller.open_gate()
                
                # Mark task as done
                self.ocr_queue.task_done()
                
            except queue.Empty:
                # No plates available, just continue
                pass
            except Exception as e:
                self.logger.error(f"Error in OCR thread: {str(e)}")
                time.sleep(0.5)  # Avoid tight loop on error
    
    def housekeeping_thread_func(self):
        """Thread function for system maintenance tasks"""
        self.logger.info("Starting housekeeping thread")
        
        while running:
            try:
                # Clean up old entries in last_processed_plates
                current_time = time.time()
                expired_plates = []
                
                for plate, timestamp in self.last_processed_plates.items():
                    if current_time - timestamp > self.config['plate_debounce_time'] * 2:
                        expired_plates.append(plate)
                
                for plate in expired_plates:
                    del self.last_processed_plates[plate]
                
                # Sleep for a while
                time.sleep(60)  # Run every minute
                
            except Exception as e:
                self.logger.error(f"Error in housekeeping thread: {str(e)}")
                time.sleep(5)  # Avoid tight loop on error
    
    def run(self):
        """Main method to run the ANPR system"""
        self.logger.info("Starting ANPR system")
        
        try:
            # Create and start all threads
            threads = []
            
            # Camera thread
            camera_thread = threading.Thread(
                target=self.camera_thread_func,
                daemon=True,
                name="CameraThread"
            )
            threads.append(camera_thread)
            
            # Detection thread
            detection_thread = threading.Thread(
                target=self.detection_thread_func,
                daemon=True,
                name="DetectionThread"
            )
            threads.append(detection_thread)
            
            # OCR thread
            ocr_thread = threading.Thread(
                target=self.ocr_thread_func,
                daemon=True,
                name="OCRThread"
            )
            threads.append(ocr_thread)
            
            # Housekeeping thread
            housekeeping_thread = threading.Thread(
                target=self.housekeeping_thread_func,
                daemon=True,
                name="HousekeepingThread"
            )
            threads.append(housekeeping_thread)
            
            # Start all threads
            for thread in threads:
                thread.start()
            
            # Main loop - keep running until interrupted
            self.logger.info("ANPR system running. Press Ctrl+C to stop.")
            
            while running:
                time.sleep(1)
                
        except KeyboardInterrupt:
            self.logger.info("Keyboard interrupt received, stopping ANPR system...")
        except Exception as e:
            self.logger.error(f"Error in main thread: {str(e)}")
        finally:
            # Cleanup
            global running
            running = False
            
            # Wait for threads to finish
            self.logger.info("Shutting down ANPR system...")
            
            # Close camera connection
            self.camera.release()
            
            # Release detector resources
            self.detector.release()
            
            self.logger.info("ANPR system shutdown complete")


def get_default_config():
    """Get default configuration for the ANPR system"""
    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    return {
        # Paths
        'model_path': os.path.join(script_dir, 'models', 'yolo11s.rknn'),
        'vehicle_make_model_path': os.path.join(script_dir, 'models', 'resnet18_makes.rknn'),
        'allowlist_path': os.path.join(script_dir, 'allowlist.txt'),
        'log_dir': os.path.join(script_dir, 'logs'),
        'captures_dir': os.path.join(script_dir, 'captures'),
        
        # Camera settings
        'camera_source': 0,  # 0 for USB camera, RTSP URL for IP camera
        'frame_width': 1280,
        'frame_height': 720,
        'buffer_size': 4,
        
        # Processing settings
        'processing_fps': 15,  # Target processing frame rate
        'detection_confidence': 0.3,
        'nms_threshold': 0.45,
        'ocr_confidence': 0.65,
        'plate_debounce_time': 5.0,  # Time in seconds to ignore repeated plate reads
        
        # Vehicle classification settings
        'enable_vehicle_classification': False,  # Disabled by default
        'enable_make_detection': False,
        'enable_color_detection': False,
        'make_confidence_threshold': 0.7,
        'color_confidence_threshold': 0.6,
        
        # Gate control settings
        'gpio_pin': 17,  # GPIO pin for gate control
        'gate_pulse_time': 1.0,  # Seconds to keep output HIGH
        'gate_cooldown_time': 5.0,  # Seconds to wait before allowing another trigger
        
        # Debug and mock settings
        'debug': False,
        'mock_mode': False,
    }


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='ANPR System for Orange Pi 5 Ultra')
    
    parser.add_argument('--camera', type=str, default=None,
                        help='Camera source (0 for USB camera, RTSP URL for IP camera)')
    
    parser.add_argument('--width', type=int, default=None,
                        help='Camera frame width')
    
    parser.add_argument('--height', type=int, default=None,
                        help='Camera frame height')
    
    parser.add_argument('--model', type=str, default=None,
                        help='Path to RKNN model file')
    
    parser.add_argument('--allowlist', type=str, default=None,
                        help='Path to allowlist file')
    
    parser.add_argument('--gpio-pin', type=int, default=None,
                        help='GPIO pin for gate control')
    
    # Vehicle classification options
    parser.add_argument('--enable-classification', action='store_true',
                        help='Enable vehicle classification')
    
    parser.add_argument('--enable-make', action='store_true',
                        help='Enable vehicle make detection')
    
    parser.add_argument('--enable-color', action='store_true',
                        help='Enable vehicle color detection')
    
    parser.add_argument('--make-model', type=str, default=None,
                        help='Path to ResNet18 RKNN model for make detection')
    
    # Debug options
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging')
    
    parser.add_argument('--mock', action='store_true',
                        help='Enable mock mode (no hardware required)')
    
    return parser.parse_args()


def main():
    """Main entry point"""
    # Register signal handler
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Get default configuration
    config = get_default_config()
    
    # Update config with command line arguments
    if args.camera is not None:
        try:
            # Try to convert to integer for USB camera
            config['camera_source'] = int(args.camera)
        except ValueError:
            # If not an integer, treat as RTSP URL
            config['camera_source'] = args.camera
    
    if args.width is not None:
        config['frame_width'] = args.width
    
    if args.height is not None:
        config['frame_height'] = args.height
    
    if args.model is not None:
        config['model_path'] = args.model
    
    if args.allowlist is not None:
        config['allowlist_path'] = args.allowlist
    
    if args.gpio_pin is not None:
        config['gpio_pin'] = args.gpio_pin
    
    # Vehicle classification options
    if args.enable_classification:
        config['enable_vehicle_classification'] = True
        
    if args.enable_make:
        config['enable_vehicle_classification'] = True
        config['enable_make_detection'] = True
        
    if args.enable_color:
        config['enable_vehicle_classification'] = True
        config['enable_color_detection'] = True
        
    if args.make_model is not None:
        config['vehicle_make_model_path'] = args.make_model
    
    # Debug and mock options
    if args.debug:
        config['debug'] = True
    
    if args.mock:
        config['mock_mode'] = True
    
    # Create and run the ANPR system
    anpr = ANPRSystem(config)
    anpr.run()


if __name__ == "__main__":
    main()
