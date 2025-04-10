#!/usr/bin/env python3
"""
Test module for ANPR System.
Verifies that all components are functioning correctly.
"""
import os
import sys
import time
import argparse
import logging
import cv2
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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
setup_logging(log_to_console=True)
logger = get_logger("test")

class ANPRSystemTester:
    """
    Test harness for ANPR System components.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize test harness.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.results = {}
        
        # Create test directory if it doesn't exist
        os.makedirs(config['test_dir'], exist_ok=True)
        
        logger.info("ANPR System tester initialized")
    
    def run_all_tests(self) -> Dict[str, Any]:
        """
        Run all component tests.
        
        Returns:
            Dict[str, Any]: Test results
        """
        logger.info("Starting all tests...")
        
        # Test hardware detection
        self.test_hardware_detection()
        
        # Test GPIO
        self.test_gpio()
        
        # Test camera
        self.test_camera()
        
        # Test object detection
        self.test_object_detection()
        
        # Test OCR
        self.test_ocr()
        
        # Test plate checker
        self.test_plate_checker()
        
        # Test optional components
        if self.config['enable_color']:
            self.test_color_detection()
        
        if self.config['enable_make']:
            self.test_make_detection()
        
        # Log overall results
        success_count = sum(1 for r in self.results.values() if r.get('success', False))
        total_count = len(self.results)
        
        logger.info(f"All tests completed: {success_count}/{total_count} passed")
        
        return self.results
    
    def test_hardware_detection(self) -> Dict[str, Any]:
        """
        Test hardware detection module.
        
        Returns:
            Dict[str, Any]: Test results
        """
        logger.info("Testing hardware detection...")
        
        try:
            # Initialize hardware detector
            detector = HardwareDetector()
            
            # Get best accelerator
            best_accelerator, info = detector.get_best_available_accelerator()
            
            # Get all capabilities
            capabilities = detector.get_capabilities()
            
            # Check if any accelerator is available
            if best_accelerator in [AcceleratorType.NPU, AcceleratorType.GPU]:
                logger.info(f"Hardware detection successful: {best_accelerator.name} available")
                result = {
                    'success': True,
                    'accelerator': best_accelerator.name,
                    'info': info,
                    'capabilities': {k.name: v for k, v in capabilities.items()}
                }
            else:
                logger.warning("No hardware acceleration available, using CPU fallback")
                result = {
                    'success': True,  # CPU is still a valid fallback
                    'accelerator': 'CPU',
                    'info': info,
                    'capabilities': {k.name: v for k, v in capabilities.items()}
                }
            
            self.results['hardware_detection'] = result
            return result
        
        except Exception as e:
            logger.error(f"Hardware detection test failed: {str(e)}")
            result = {
                'success': False,
                'error': str(e)
            }
            self.results['hardware_detection'] = result
            return result
    
    def test_gpio(self) -> Dict[str, Any]:
        """
        Test GPIO module.
        
        Returns:
            Dict[str, Any]: Test results
        """
        logger.info("Testing GPIO module...")
        
        try:
            # Initialize GPIO (in mock mode for testing)
            gpio = get_gpio(mock_mode=True)
            
            # Set up a test pin
            test_pin = 17
            gpio.setup(test_pin, GPIOMode.OUT)
            
            # Test setting output
            gpio.output(test_pin, GPIOState.HIGH)
            high_state = gpio.input(test_pin)
            
            gpio.output(test_pin, GPIOState.LOW)
            low_state = gpio.input(test_pin)
            
            # Test pulse
            gpio.pulse(test_pin, 0.1)  # Short pulse for testing
            time.sleep(0.2)  # Wait for pulse to complete
            
            # Clean up
            gpio.cleanup(test_pin)
            
            # Check if test passed
            if high_state == GPIOState.HIGH and low_state == GPIOState.LOW:
                logger.info("GPIO test successful")
                result = {
                    'success': True,
                    'high_state': high_state.name,
                    'low_state': low_state.name
                }
            else:
                logger.warning("GPIO state inconsistent")
                result = {
                    'success': False,
                    'high_state': high_state.name,
                    'low_state': low_state.name,
                    'error': 'GPIO state inconsistent'
                }
            
            self.results['gpio'] = result
            return result
        
        except Exception as e:
            logger.error(f"GPIO test failed: {str(e)}")
            result = {
                'success': False,
                'error': str(e)
            }
            self.results['gpio'] = result
            return result
    
    def test_camera(self) -> Dict[str, Any]:
        """
        Test camera input module.
        
        Returns:
            Dict[str, Any]: Test results
        """
        logger.info("Testing camera input...")
        
        try:
            # Initialize camera in simulation mode
            sim_dir = self.config.get('simulate_dir', 'simulation_images')
            
            # Check if simulation directory exists
            if not os.path.isdir(sim_dir):
                logger.warning(f"Simulation directory not found: {sim_dir}")
                # Create a test image for simulation
                os.makedirs(sim_dir, exist_ok=True)
                test_img = np.zeros((720, 1280, 3), dtype=np.uint8)
                cv2.putText(test_img, "Test Image", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imwrite(os.path.join(sim_dir, "test_image.jpg"), test_img)
            
            camera = CameraInput(
                rtsp_url="dummy_url",  # Not used in simulation mode
                resolution="720p",
                buffer_size=2,
                simulate_mode=True,
                simulate_dir=sim_dir
            )
            
            # Start camera
            camera.start()
            
            # Get a frame
            frame = camera.get_frame(timeout=1.0)
            
            # Stop camera
            camera.stop()
            
            # Check if frame was received
            if frame is not None and frame.size > 0:
                logger.info(f"Camera test successful: received frame of shape {frame.shape}")
                
                # Save frame for inspection
                test_frame_path = os.path.join(self.config['test_dir'], "test_camera_frame.jpg")
                cv2.imwrite(test_frame_path, frame)
                
                result = {
                    'success': True,
                    'frame_shape': frame.shape,
                    'frame_path': test_frame_path
                }
            else:
                logger.warning("Camera test: No frame received")
                result = {
                    'success': False,
                    'error': 'No frame received'
                }
            
            self.results['camera'] = result
            return result
        
        except Exception as e:
            logger.error(f"Camera test failed: {str(e)}")
            result = {
                'success': False,
                'error': str(e)
            }
            self.results['camera'] = result
            return result
    
    def test_object_detection(self) -> Dict[str, Any]:
        """
        Test object detection module.
        
        Returns:
            Dict[str, Any]: Test results
        """
        logger.info("Testing object detection...")
        
        try:
            # Get hardware info
            hardware_result = self.results.get('hardware_detection', {})
            best_accelerator = hardware_result.get('accelerator', 'CPU')
            
            # Create a test image with a license plate
            test_img = np.zeros((720, 1280, 3), dtype=np.uint8)
            
            # Draw a car shape
            cv2.rectangle(test_img, (400, 300), (900, 500), (0, 0, 255), -1)
            cv2.rectangle(test_img, (500, 200), (800, 300), (0, 0, 255), -1)
            
            # Draw a license plate
            plate_x1, plate_y1 = 550, 400
            plate_x2, plate_y2 = 750, 450
            cv2.rectangle(test_img, (plate_x1, plate_y1), (plate_x2, plate_y2), (200, 200, 200), -1)
            cv2.rectangle(test_img, (plate_x1, plate_y1), (plate_x2, plate_y2), (0, 0, 0), 2)
            
            # Add plate text
            cv2.putText(test_img, "ABC123", (plate_x1 + 10, plate_y1 + 35),
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            
            # Save test image
            test_img_path = os.path.join(self.config['test_dir'], "test_detection_input.jpg")
            cv2.imwrite(test_img_path, test_img)
            
            # Initialize appropriate detector based on hardware
            if best_accelerator == 'NPU' and os.path.isfile(os.path.join("models", "YOLO11s.rknn")):
                # Use RKNN detector
                detector = YOLO11sRKNN(
                    model_path=os.path.join("models", "YOLO11s.rknn"),
                    conf_threshold=0.3,  # Lower threshold for testing
                    iou_threshold=0.45
                )
                detector_type = "RKNN"
            else:
                # Use GPU/CPU detector
                detector = YOLO11sGPU(
                    model_path=os.path.join("models", "YOLO11s.pt"),
                    conf_threshold=0.3,  # Lower threshold for testing
                    iou_threshold=0.45
                )
                detector_type = "PyTorch"
            
            # Run detection
            detections = detector.detect(test_img, return_details=True)
            
            # Draw detections
            result_img = detector._draw_detections(test_img, detections)
            
            # Save result image
            result_img_path = os.path.join(self.config['test_dir'], "test_detection_result.jpg")
            cv2.imwrite(result_img_path, result_img)
            
            # Check if any detections were made
            if len(detections) > 0:
                logger.info(f"Detection test successful: {len(detections)} objects detected")
                result = {
                    'success': True,
                    'detector_type': detector_type,
                    'detections': len(detections),
                    'input_path': test_img_path,
                    'result_path': result_img_path
                }
            else:
                logger.warning("Detection test: No objects detected")
                result = {
                    'success': False,
                    'detector_type': detector_type,
                    'error': 'No objects detected',
                    'input_path': test_img_path,
                    'result_path': result_img_path
                }
            
            self.results['object_detection'] = result
            return result
        
        except Exception as e:
            logger.error(f"Object detection test failed: {str(e)}")
            result = {
                'success': False,
                'error': str(e)
            }
            self.results['object_detection'] = result
            return result
    
    def test_ocr(self) -> Dict[str, Any]:
        """
        Test OCR module.
        
        Returns:
            Dict[str, Any]: Test results
        """
        logger.info("Testing OCR module...")
        
        try:
            # Create a test image with a license plate text
            test_img = np.ones((200, 400, 3), dtype=np.uint8) * 255
            
            # Draw a license plate
            cv2.rectangle(test_img, (50, 50), (350, 150), (200, 200, 200), -1)
            cv2.rectangle(test_img, (50, 50), (350, 150), (0, 0, 0), 2)
            
            # Add plate text
            cv2.putText(test_img, "ABC123", (100, 120),
                      cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
            
            # Save test image
            test_img_path = os.path.join(self.config['test_dir'], "test_ocr_input.jpg")
            cv2.imwrite(test_img_path, test_img)
            
            # Initialize OCR
            ocr = PlateOCR(
                languages=['en'],
                min_confidence=0.3,  # Lower threshold for testing
                skew_correction=True,
                preprocessing_method="adaptive"
            )
            
            # Run OCR
            ocr_result = ocr.recognize(test_img, return_details=True, try_all_preprocessing=True)
            
            # Check OCR result
            if ocr_result['text']:
                logger.info(f"OCR test successful: Text detected: '{ocr_result['text']}' "
                          f"(conf: {ocr_result['confidence']:.2f})")
                result = {
                    'success': True,
                    'text': ocr_result['text'],
                    'confidence': ocr_result['confidence'],
                    'method': ocr_result['method'],
                    'input_path': test_img_path
                }
            else:
                logger.warning("OCR test: No text detected")
                result = {
                    'success': False,
                    'error': 'No text detected',
                    'input_path': test_img_path
                }
            
            self.results['ocr'] = result
            return result
        
        except Exception as e:
            logger.error(f"OCR test failed: {str(e)}")
            result = {
                'success': False,
                'error': str(e)
            }
            self.results['ocr'] = result
            return result
    
    def test_plate_checker(self) -> Dict[str, Any]:
        """
        Test plate checker module.
        
        Returns:
            Dict[str, Any]: Test results
        """
        logger.info("Testing plate checker...")
        
        try:
            # Create temporary allowlist for testing
            test_allowlist_path = os.path.join(self.config['test_dir'], "test_allowlist.txt")
            with open(test_allowlist_path, 'w') as f:
                f.write("# Test allowlist\n")
                f.write("ABC123\n")
                f.write("XYZ789\n")
            
            # Create temporary JSON allowlist
            test_json_path = os.path.join(self.config['test_dir'], "test_allowlist.json")
            with open(test_json_path, 'w') as f:
                f.write('{\n')
                f.write('  "ABC123": {\n')
                f.write('    "make": "Toyota",\n')
                f.write('    "color": "Blue",\n')
                f.write('    "owner": "Test User"\n')
                f.write('  },\n')
                f.write('  "XYZ789": {\n')
                f.write('    "make": "Honda",\n')
                f.write('    "color": "Red",\n')
                f.write('    "owner": "Test User 2"\n')
                f.write('  }\n')
                f.write('}\n')
            
            # Initialize plate checker
            checker = PlateChecker(
                allowlist_path=test_allowlist_path,
                mock_gpio=True,
                enable_make_verification=True,
                enable_color_verification=True
            )
            
            # Test positive match
            match1, details1 = checker.check_plate("ABC123", make="Toyota", color="Blue")
            
            # Test negative match
            match2, details2 = checker.check_plate("DEF456")
            
            # Test formatting variant
            match3, details3 = checker.check_plate("ABC-123")
            
            # Test add and remove
            checker.add_to_allowlist("DEF456", {"make": "BMW", "color": "Black"})
            match4, details4 = checker.check_plate("DEF456")
            
            checker.remove_from_allowlist("DEF456")
            match5, details5 = checker.check_plate("DEF456")
            
            # Calculate pass rate
            checks_passed = (match1 and match3 and match4 and not match2 and not match5)
            
            if checks_passed:
                logger.info("Plate checker test successful")
                result = {
                    'success': True,
                    'tests': {
                        'positive_match': match1,
                        'negative_match': not match2,
                        'format_variant': match3,
                        'add_plate': match4,
                        'remove_plate': not match5
                    }
                }
            else:
                logger.warning("Plate checker test: Some checks failed")
                result = {
                    'success': False,
                    'error': 'Some checks failed',
                    'tests': {
                        'positive_match': match1,
                        'negative_match': not match2,
                        'format_variant': match3,
                        'add_plate': match4,
                        'remove_plate': not match5
                    }
                }
            
            self.results['plate_checker'] = result
            return result
        
        except Exception as e:
            logger.error(f"Plate checker test failed: {str(e)}")
            result = {
                'success': False,
                'error': str(e)
            }
            self.results['plate_checker'] = result
            return result
    
    def test_color_detection(self) -> Dict[str, Any]:
        """
        Test color detection module.
        
        Returns:
            Dict[str, Any]: Test results
        """
        logger.info("Testing color detection...")
        
        try:
            # Create test images with different colors
            colors = {
                'red': (0, 0, 255),
                'green': (0, 255, 0),
                'blue': (255, 0, 0),
                'black': (0, 0, 0),
                'white': (255, 255, 255)
            }
            
            # Initialize color detector
            detector = ColorDetector(n_clusters=3)
            
            results = {}
            
            for color_name, color_value in colors.items():
                # Create a test image
                test_img = np.ones((300, 400, 3), dtype=np.uint8) * color_value
                
                # Add some noise for realism
                noise = np.random.randint(0, 30, test_img.shape, dtype=np.int16)
                test_img = np.clip(test_img + noise - 15, 0, 255).astype(np.uint8)
                
                # Save test image
                test_img_path = os.path.join(self.config['test_dir'], f"test_color_{color_name}.jpg")
                cv2.imwrite(test_img_path, test_img)
                
                # Detect color
                detected = detector.detect_color(test_img, return_details=True)
                
                # Create visualization
                vis_img = detector.visualize_colors(test_img)
                vis_path = os.path.join(self.config['test_dir'], f"test_color_{color_name}_result.jpg")
                cv2.imwrite(vis_path, vis_img)
                
                # Check result
                results[color_name] = {
                    'expected': color_name,
                    'detected': detected['main_color'],
                    'confidence': detected.get('colors', [{}])[0].get('weight', 0) if detected.get('colors') else 0,
                    'input_path': test_img_path,
                    'result_path': vis_path
                }
            
            # Count correct detections
            correct_count = sum(1 for k, v in results.items() 
                              if v['detected'] == k or v['detected'] in k or k in v['detected'])
            
            if correct_count >= 3:  # At least 3 out of 5 should be correct
                logger.info(f"Color detection test successful: {correct_count}/5 colors correct")
                result = {
                    'success': True,
                    'correct_count': correct_count,
                    'color_results': results
                }
            else:
                logger.warning(f"Color detection test: Only {correct_count}/5 colors correct")
                result = {
                    'success': False,
                    'error': 'Too few correct detections',
                    'correct_count': correct_count,
                    'color_results': results
                }
            
            self.results['color_detection'] = result
            return result
        
        except Exception as e:
            logger.error(f"Color detection test failed: {str(e)}")
            result = {
                'success': False,
                'error': str(e)
            }
            self.results['color_detection'] = result
            return result
    
    def test_make_detection(self) -> Dict[str, Any]:
        """
        Test make detection module.
        
        Returns:
            Dict[str, Any]: Test results
        """
        logger.info("Testing make detection...")
        
        try:
            # Initialize make detector
            detector = MakeDetector(
                confidence_threshold=0.3,  # Lower threshold for testing
                top_k=3
            )
            
            # Check if model exists
            model_exists = os.path.isfile(detector.model_path)
            
            # Create a test image
            test_img = np.zeros((400, 600, 3), dtype=np.uint8)
            
            # Draw a car shape
            cv2.rectangle(test_img, (150, 100), (450, 250), (0, 0, 255), -1)  # Main body
            cv2.rectangle(test_img, (200, 50), (400, 100), (0, 0, 255), -1)   # Roof
            cv2.circle(test_img, (200, 300), 30, (0, 0, 0), -1)  # Left wheel
            cv2.circle(test_img, (400, 300), 30, (0, 0, 0), -1)  # Right wheel
            
            # Save test image
            test_img_path = os.path.join(self.config['test_dir'], "test_make.jpg")
            cv2.imwrite(test_img_path, test_img)
            
            # Run make detection
            make_result = detector.detect_make(test_img, return_details=True)
            
            # Create visualization
            vis_img = detector.visualize_prediction(test_img)
            vis_path = os.path.join(self.config['test_dir'], "test_make_result.jpg")
            cv2.imwrite(vis_path, vis_img)
            
            # Check if detection was made
            if make_result['make'] != 'unknown' or len(make_result['predictions']) > 0:
                logger.info(f"Make detection test successful: Detected '{make_result['make']}'")
                result = {
                    'success': True,
                    'model_exists': model_exists,
                    'make': make_result['make'],
                    'confidence': make_result['confidence'],
                    'predictions': make_result['predictions'],
                    'input_path': test_img_path,
                    'result_path': vis_path
                }
            else:
                # Without a trained model, this might not be an error
                if model_exists:
                    logger.warning("Make detection test: No make detected with existing model")
                    success = False
                    error = 'No make detected'
                else:
                    logger.info("Make detection test: No model found, detection as expected")
                    success = True
                    error = 'No model found'
                
                result = {
                    'success': success,
                    'model_exists': model_exists,
                    'error': error,
                    'make': make_result['make'],
                    'input_path': test_img_path,
                    'result_path': vis_path
                }
            
            self.results['make_detection'] = result
            return result
        
        except Exception as e:
            logger.error(f"Make detection test failed: {str(e)}")
            result = {
                'success': False,
                'error': str(e)
            }
            self.results['make_detection'] = result
            return result
    
    def print_summary(self) -> None:
        """Print a summary of test results."""
        print("\n" + "=" * 60)
        print("ANPR SYSTEM TEST SUMMARY")
        print("=" * 60)
        
        success_count = 0
        total_count = 0
        
        for component, result in self.results.items():
            success = result.get('success', False)
            if success:
                success_count += 1
            total_count += 1
            
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"{status} - {component}")
            
            # Print error if any
            if not success and 'error' in result:
                print(f"      Error: {result['error']}")
        
        # Print overall result
        print("-" * 60)
        print(f"Overall: {success_count}/{total_count} tests passed")
        print("=" * 60)
        
        # Print additional information
        if 'hardware_detection' in self.results:
            hw = self.results['hardware_detection']
            if hw.get('success', False):
                print(f"\nHardware: Using {hw.get('accelerator', 'unknown')} acceleration")
                print(f"Capabilities: {hw.get('capabilities', {})}")
        
        print("\nTest artifacts saved to: " + os.path.abspath(self.config['test_dir']))
        print("=" * 60)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="ANPR System Tester")
    
    parser.add_argument("--enable-color", action="store_true",
                      help="Test color detection")
    parser.add_argument("--enable-make", action="store_true",
                      help="Test make detection")
    parser.add_argument("--test-dir", type=str, default="test_results",
                      help="Directory for test results")
    parser.add_argument("--simulate-dir", type=str, default="simulation_images",
                      help="Directory for simulation images")
    
    return parser.parse_args()

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Create test configuration
    config = {
        'enable_color': args.enable_color,
        'enable_make': args.enable_make,
        'test_dir': args.test_dir,
        'simulate_dir': args.simulate_dir
    }
    
    # Initialize tester
    tester = ANPRSystemTester(config)
    
    # Run all tests
    tester.run_all_tests()
    
    # Print summary
    tester.print_summary()

if __name__ == "__main__":
    main()
