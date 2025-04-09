import os
import sys
import cv2
import time
import torch
import random
import logging
import numpy as np
import shutil
import re
import threading
import concurrent.futures
import queue
from PIL import Image
from pathlib import Path
from logging.handlers import RotatingFileHandler
from scipy.ndimage import rotate
from collections import deque

# Import optional GPU/NPU acceleration libraries
try:
    import rknn_toolkit_lite as rknnlite
    RKNN_AVAILABLE = True
    print("RKNN NPU acceleration available")
except ImportError:
    RKNN_AVAILABLE = False
    print("RKNN NPU acceleration not available")

# Try to import TensorRT for GPU acceleration
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TENSORRT_AVAILABLE = True
    print("TensorRT GPU acceleration available")
except ImportError:
    TENSORRT_AVAILABLE = False
    print("TensorRT GPU acceleration not available")

# Try to import GStreamer for advanced camera handling
try:
    import gi
    gi.require_version('Gst', '1.0')
    from gi.repository import Gst
    Gst.init(None)
    GSTREAMER_AVAILABLE = True
    print("GStreamer support available")
except (ImportError, ValueError):
    GSTREAMER_AVAILABLE = False
    print("GStreamer support not available")

# Import YOLOv8 (using Ultralytics library)
try:
    import ultralytics
    from ultralytics import YOLO
except ImportError:
    print("Ultralytics not found. Installing...")
    os.system("pip install ultralytics")
    import ultralytics
    from ultralytics import YOLO

# Try to import easyocr
try:
    import easyocr
except ImportError:
    print("EasyOCR not found. Installing...")
    os.system("pip install easyocr")
    import easyocr

# Try to import scikit-learn for color detection
try:
    from sklearn.cluster import KMeans
except ImportError:
    print("Scikit-learn not found. Installing...")
    os.system("pip install scikit-learn")
    from sklearn.cluster import KMeans

# Hardware detection and capability analysis
def detect_hardware_capabilities():
    """Detect available hardware acceleration options and return the best one to use"""
    capabilities = {
        'cpu': True,
        'cuda': False,
        'tensorrt': False,
        'rknn_npu': False,
        'best_option': 'cpu'
    }
    
    # Check for CUDA GPU
    try:
        if torch.cuda.is_available():
            capabilities['cuda'] = True
            capabilities['best_option'] = 'cuda'
            print(f"CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"CUDA compute capability: {torch.cuda.get_device_capability()}")
            
            # Check memory availability - important for performance
            total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # in GB
            print(f"CUDA total memory: {total_mem:.2f} GB")
            
            if total_mem < 2.0:
                print("Warning: Low GPU memory may limit performance with large models")
    except Exception as e:
        print(f"Error checking CUDA availability: {e}")
    
    # Check for RK3588 NPU
    if RKNN_AVAILABLE:
        try:
            # Check if RKNN model exists before trying to load it
            rknn_model_path = './models/anpr_rknn.rknn'
            if os.path.exists(rknn_model_path):
                rknn = rknnlite.RKNNLite()
                ret = rknn.load_rknn(rknn_model_path)
                if ret == 0:
                    capabilities['rknn_npu'] = True
                    capabilities['best_option'] = 'rknn_npu'
                    print("RK3588 NPU available and working")
                rknn.release()
            else:
                print(f"RKNN model not found at {rknn_model_path}. NPU detection skipped.")
                print("Run with --enable-npu flag to convert models for NPU.")
        except Exception as e:
            print(f"RKNN test failed: {e}")
    
    # Check for TensorRT
    if TENSORRT_AVAILABLE:
        try:
            capabilities['tensorrt'] = True
            if capabilities['best_option'] == 'cuda':
                capabilities['best_option'] = 'tensorrt'
                print("TensorRT acceleration available")
        except Exception as e:
            print(f"Error checking TensorRT availability: {e}")
    
    print(f"Best acceleration option: {capabilities['best_option']}")
    return capabilities

# Function to parse resolution string or predefined resolution
def parse_resolution(resolution_str):
    """Parse resolution string or predefined resolution setting."""
    # Predefined resolution options
    resolution_presets = {
        "480p": (640, 480),
        "720p": (1280, 720),
        "1080p": (1920, 1080),
        "4k": (3840, 2160)
    }
    
    # If it's a predefined resolution name
    if resolution_str.lower() in resolution_presets:
        width, height = resolution_presets[resolution_str.lower()]
        print(f"Using predefined resolution: {resolution_str} ({width}x{height})")
        return width, height
    
    # If it's a custom resolution string (e.g., "640x480")
    try:
        width, height = map(int, resolution_str.split("x"))
        print(f"Using custom resolution: {width}x{height}")
        return width, height
    except:
        print(f"Invalid resolution format: {resolution_str}, falling back to 640x480")
        return 640, 480

# Performance warning function for high resolutions
def warn_if_high_performance_needed(width, height, hw_capabilities):
    """Warn if the resolution might cause performance issues"""
    high_res_threshold = 1280 * 720  # 720p
    very_high_res_threshold = 1920 * 1080  # 1080p
    
    resolution_size = width * height
    
    # Check if we have hardware acceleration
    has_acceleration = hw_capabilities['cuda'] or hw_capabilities['rknn_npu'] or hw_capabilities['tensorrt']
    
    if resolution_size >= very_high_res_threshold and not has_acceleration:
        print("\nWARNING: Using very high resolution (1080p or higher) without hardware acceleration")
        print("This may cause performance issues. Consider:")
        print("1. Using a lower resolution")
        print("2. Reducing frame processing frequency")
        print("3. Enabling hardware acceleration if available\n")
    elif resolution_size >= high_res_threshold and not has_acceleration:
        print("\nNote: Using high resolution (720p or higher) without hardware acceleration")
        print("This may affect performance. Consider enabling hardware acceleration if available.\n")
    elif resolution_size >= very_high_res_threshold and has_acceleration:
        print(f"\nUsing high resolution with {hw_capabilities['best_option']} acceleration.\n")

# GStreamer Camera class
class GStreamerCamera:
    def __init__(self, pipeline_str, img_size=(640, 480)):
        self.img_size = img_size
        self.pipeline_str = pipeline_str
        self.running = False
        self.frame_queue = queue.Queue(maxsize=1)  # Only keep latest frame
        
        # Initialize GStreamer pipeline
        self.pipeline = Gst.parse_launch(pipeline_str)
        
        # Get sink element to receive frames
        self.sink = self.pipeline.get_by_name('appsink')
        self.sink.set_property('emit-signals', True)
        self.sink.connect('new-sample', self._on_new_sample)
        
        # Set pipeline to PLAYING state
        self.pipeline.set_state(Gst.State.PLAYING)
        self.running = True
        
        # Start buffer processing thread
        self.thread = threading.Thread(target=self._buffer_thread, daemon=True)
        self.thread.start()
        
        print(f"GStreamer camera initialized with pipeline: {pipeline_str}")
        
    def _on_new_sample(self, sink):
        """Callback for new GStreamer frame"""
        sample = sink.emit('pull-sample')
        if not sample:
            return Gst.FlowReturn.ERROR
            
        buffer = sample.get_buffer()
        caps = sample.get_caps()
        
        # Get buffer data
        success, map_info = buffer.map(Gst.MapFlags.READ)
        if not success:
            buffer.unmap(map_info)
            return Gst.FlowReturn.ERROR
            
        # Get frame dimensions from caps
        structure = caps.get_structure(0)
        width = structure.get_value('width')
        height = structure.get_value('height')
        
        # Create numpy array from buffer data
        frame = np.ndarray(
            shape=(height, width, 3),
            dtype=np.uint8,
            buffer=map_info.data
        )
        
        # Resize if needed
        if (width, height) != self.img_size:
            frame = cv2.resize(frame, self.img_size)
            
        # Put frame in queue, replacing old frame if queue is full
        try:
            if self.frame_queue.full():
                self.frame_queue.get_nowait()  # Discard old frame
            self.frame_queue.put_nowait(frame.copy())  # Copy to avoid reference issues
        except queue.Full:
            pass  # Queue is full, skip this frame
            
        buffer.unmap(map_info)
        return Gst.FlowReturn.OK
        
    def _buffer_thread(self):
        """Thread to handle frame buffering"""
        blank_frame = np.zeros((self.img_size[1], self.img_size[0], 3), dtype=np.uint8)
        
        while self.running:
            try:
                # Handle GStreamer event loop
                bus = self.pipeline.get_bus()
                msg = bus.timed_pop_filtered(
                    100 * Gst.MSECOND,
                    Gst.MessageType.ERROR | Gst.MessageType.EOS
                )
                
                if msg:
                    if msg.type == Gst.MessageType.ERROR:
                        err, debug = msg.parse_error()
                        print(f"GStreamer error: {err}, {debug}")
                        # Try to restart pipeline
                        self.pipeline.set_state(Gst.State.NULL)
                        time.sleep(1)
                        self.pipeline.set_state(Gst.State.PLAYING)
                    elif msg.type == Gst.MessageType.EOS:
                        print("End of stream received")
                        # Try to restart pipeline
                        self.pipeline.set_state(Gst.State.NULL)
                        time.sleep(1)
                        self.pipeline.set_state(Gst.State.PLAYING)
                        
                # Add blank frame if queue is empty (no frames received yet)
                if self.frame_queue.empty():
                    self.frame_queue.put(blank_frame.copy())
                    
            except Exception as e:
                print(f"GStreamer buffer thread error: {e}")
                time.sleep(0.1)  # Prevent busy loop on error
                
    def run(self):
        """Get the latest frame"""
        try:
            # Get frame from queue with timeout
            frame = self.frame_queue.get(timeout=0.5)
            return frame
        except queue.Empty:
            # Return blank frame if no frame available
            print("GStreamer camera: No frame available")
            return np.zeros((self.img_size[1], self.img_size[0], 3), dtype=np.uint8)
        except Exception as e:
            print(f"Error getting frame from GStreamer camera: {e}")
            return np.zeros((self.img_size[1], self.img_size[0], 3), dtype=np.uint8)
            
    def release(self):
        """Release resources"""
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self.pipeline.set_state(Gst.State.NULL)
        print("GStreamer camera released")

# Enhanced IP Camera class
class IPCamera:
    def __init__(self, rtsp_url, img_size=(640, 480), fps=30, use_tcp=True, reconnect_interval=5, max_failures=10):
        self.rtsp_url = rtsp_url
        self.img_size = img_size
        self.fps = fps
        self.use_tcp = use_tcp
        self.reconnect_interval = reconnect_interval
        self.last_reconnect_attempt = 0
        self.connection_healthy = True
        self.frame_buffer = None  # Store last good frame
        self.consecutive_failures = 0
        self.max_failures = max_failures  # Maximum number of failures before stopping reconnection attempts
        self.total_reconnect_attempts = 0
        self.max_reconnect_attempts = 20  # Limit total reconnection attempts to avoid infinite loops
        
        # Use GStreamer pipeline if available for better performance
        if GSTREAMER_AVAILABLE:
            transport_protocol = "tcp" if use_tcp else "udp"
            self.pipeline_str = (
                f'rtspsrc location="{rtsp_url}" protocols={transport_protocol} latency=100 ! '
                f'rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! '
                f'videoscale ! video/x-raw,width={img_size[0]},height={img_size[1]} ! '
                f'appsink name=appsink'
            )
            try:
                self.camera = GStreamerCamera(self.pipeline_str, img_size)
                self.using_gstreamer = True
                print("Using GStreamer for RTSP streaming")
                return
            except Exception as e:
                print(f"GStreamer initialization failed: {e}")
                print("Falling back to OpenCV")
                self.using_gstreamer = False
        else:
            self.using_gstreamer = False
        
        # Fall back to OpenCV if GStreamer fails or is not available
        # Set up RTSP options for better reliability
        self.cap = None
        self._initialize_opencv_camera()
        
    def _initialize_opencv_camera(self):
        """Initialize or reinitialize the OpenCV camera connection"""
        # Close existing connection if any
        if self.cap is not None:
            self.cap.release()
            
        # Modify URL if needed to use TCP
        connection_url = self.rtsp_url
        if self.use_tcp and 'rtsp://' in self.rtsp_url:
            # Use OpenCV's built-in RTSP over TCP functionality
            os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp'
            
        # Create VideoCapture
        self.cap = cv2.VideoCapture(connection_url)
        
        # Set buffer size to minimize latency
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Try to set the requested resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.img_size[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.img_size[1])
        
        # Try to set FPS
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        
        # Check if connection was successful
        if not self.cap.isOpened():
            print("Failed to open IP camera connection")
            self.connection_healthy = False
        else:
            self.connection_healthy = True
            self.consecutive_failures = 0
            print("Successfully connected to IP camera")
            
            # Check what resolution we actually got
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            if actual_width > 0 and actual_height > 0:
                if actual_width != self.img_size[0] or actual_height != self.img_size[1]:
                    print(f"Warning: Requested camera resolution {self.img_size[0]}x{self.img_size[1]} "
                          f"but got {actual_width}x{actual_height}")
        
    def run(self):
        """Capture and return a frame"""
        if self.using_gstreamer:
            return self.camera.run()
            
        # OpenCV implementation
        try:
            # Check if reconnection is needed
            current_time = time.time()
            if (not self.connection_healthy and 
                (current_time - self.last_reconnect_attempt) > self.reconnect_interval and
                self.total_reconnect_attempts < self.max_reconnect_attempts):
                
                print(f"Attempting to reconnect to IP camera (attempt {self.total_reconnect_attempts + 1}/{self.max_reconnect_attempts})...")
                self._initialize_opencv_camera()
                self.last_reconnect_attempt = current_time
                self.total_reconnect_attempts += 1
                
                # Reset consecutive failures counter on reconnection attempt
                self.consecutive_failures = 0
            
            # Try to capture frame
            if self.connection_healthy:
                ret, frame = self.cap.read()
                
                if ret and frame is not None and frame.size > 0:
                    # Successfully got a frame
                    self.consecutive_failures = 0
                    self.total_reconnect_attempts = 0  # Reset reconnection counter on success
                    
                    # Resize if needed
                    h, w = frame.shape[:2]
                    if w != self.img_size[0] or h != self.img_size[1]:
                        frame = cv2.resize(frame, self.img_size)
                        
                    # Update frame buffer with the good frame
                    self.frame_buffer = frame.copy()
                    return frame
                else:
                    # Failed to get frame
                    self.consecutive_failures += 1
                    print(f"Failed to read from IP camera (attempt {self.consecutive_failures}/{self.max_failures})")
                    
                    # Mark connection as unhealthy after several consecutive failures
                    if self.consecutive_failures >= self.max_failures:
                        self.connection_healthy = False
            
            # If we reach here, use the frame buffer if available
            if self.frame_buffer is not None:
                return self.frame_buffer.copy()
                
            # Fall back to blank frame
            return np.zeros((self.img_size[1], self.img_size[0], 3), dtype=np.uint8)
                
        except Exception as e:
            print(f"Error capturing from IP camera: {e}")
            self.connection_healthy = False
            
            # Return the most recent good frame if available
            if self.frame_buffer is not None:
                return self.frame_buffer.copy()
                
            # Fall back to blank frame
            return np.zeros((self.img_size[1], self.img_size[0], 3), dtype=np.uint8)
            
    def release(self):
        """Release camera resources"""
        if self.using_gstreamer:
            if hasattr(self, 'camera'):
                self.camera.release()
        else:
            if self.cap is not None:
                self.cap.release()

# RKNN NPU-based Detector for license plates
class RKNNDetector:
    """
    Detector that uses the Rockchip Neural Network (RKNN) NPU for acceleration.
    
    This detector requires a model converted to RKNN format. You can convert YOLO models
    to RKNN format using the rknn-toolkit (not the lite version) with the following steps:
    
    1. Install rknn-toolkit: pip install rknn-toolkit
    2. Use the converter script to convert your YOLO model to RKNN format
    3. Place the converted model in the models directory
    
    The model should output detection boxes in YOLO format.
    """
    def __init__(self, model_path, img_size=640, log_level='INFO', log_dir='./logs/'):
        self.model_path = model_path
        self.img_size = img_size
        self.log_level = log_level
        self.log_dir = log_dir
        self.input_shape = None
        self.output_shapes = None
        
        # Set up logging
        if self.log_level:
            self.num_log_level = getattr(logging, self.log_level.upper(), 20)
            os.makedirs(os.path.dirname(self.log_dir), exist_ok=True)
            log_formatter = logging.Formatter("%(asctime)s %(message)s")
            logFile = self.log_dir + 'detection.log'
            my_handler = RotatingFileHandler(logFile, mode='a', maxBytes=25 * 1024 * 1024,
                                             backupCount=10, encoding='utf-8', delay=False)
            my_handler.setFormatter(log_formatter)
            my_handler.setLevel(self.num_log_level)
            self.logger = logging.getLogger('rknn_detector')  
            self.logger.setLevel(self.num_log_level)
            self.logger.addHandler(my_handler)
        
        try:
            # Initialize RKNN
            self.rknn = rknnlite.RKNNLite()
            
            # Load RKNN model
            print(f"Loading RKNN model from {model_path}...")
            ret = self.rknn.load_rknn(model_path)
            if ret != 0:
                raise RuntimeError(f"Load RKNN model failed with error {ret}")
                
            # Initialize runtime environment
            print("Initializing RKNN runtime...")
            ret = self.rknn.init_runtime(target='rk3588')
            if ret != 0:
                raise RuntimeError(f"Init RKNN runtime failed with error {ret}")
            
            # Get input and output shapes for reference
            self.input_shape = self.rknn.get_input_shape()[0]
            self.output_shapes = self.rknn.get_output_shape()
            
            print(f"RKNN input shape: {self.input_shape}")
            print(f"RKNN output shapes: {self.output_shapes}")
            print("RKNN detector initialized successfully")
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize RKNN detector: {e}")
        
    def run(self, inp_image, conf_thres=0.25):
        # Run Inference
        t0 = time.time()
        
        # Clone the input image to avoid modifying the original
        self.im0 = inp_image.copy() if inp_image is not None else np.zeros((640, 480, 3), dtype=np.uint8)
        
        # Get file name (for logging purposes)
        self.file_name = f"frame_{int(time.time())}.jpg"
        
        # Preprocess image to match RKNN input requirements
        # Resize to the model's input size
        t1 = time.time()
        try:
            # Resize input image
            model_input_size = (self.img_size, self.img_size)  # Usually square for YOLO models
            input_img = cv2.resize(self.im0, model_input_size)
            
            # Convert to RGB (RKNN models often expect RGB input)
            input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
            
            # Run inference
            outputs = self.rknn.inference(inputs=[input_img])
            t2 = time.time()
            
            # Process outputs (assuming YOLO format)
            # Outputs may need specific post-processing depending on the model
            # This is a simplified example
            
            # Process detections
            bbox = None  # bounding box of detected object with max conf
            cropped_img = None  # cropped detected object with max conf
            det_conf = None  # confidence level for detected object with max conf
            
            # Process RKNN YOLO outputs
            # This will depend on how your RKNN model was converted
            # The following is a placeholder for post-processing logic
            detections = self._process_yolo_output(outputs, conf_thres)
            
            if detections and len(detections) > 0:
                # Get the detection with the highest confidence
                best_detection = max(detections, key=lambda x: x['confidence'])
                
                # Extract box coordinates
                x1, y1, x2, y2 = best_detection['bbox']
                
                # Convert coordinates to original image scale
                orig_h, orig_w = self.im0.shape[:2]
                scale_x = orig_w / model_input_size[0]
                scale_y = orig_h / model_input_size[1]
                
                x1 = int(x1 * scale_x)
                y1 = int(y1 * scale_y)
                x2 = int(x2 * scale_x)
                y2 = int(y2 * scale_y)
                
                # Store bbox and confidence
                bbox = [x1, y1, x2, y2]
                det_conf = best_detection['confidence']
                
                # Crop the license plate from the image with slightly tighter margins
                if (y2 > y1) and (x2 > x1):  # Ensure valid dimensions
                    margin_x = int((x2 - x1) * 0.05)  # 5% margin reduction
                    margin_y = int((y2 - y1) * 0.1)   # 10% margin reduction
                    # Ensure margins don't go outside the image
                    x1_crop = max(0, x1 + margin_x)
                    y1_crop = max(0, y1 + margin_y)
                    x2_crop = min(self.im0.shape[1], x2 - margin_x)
                    y2_crop = min(self.im0.shape[0], y2 - margin_y)
                    
                    cropped_img = self.im0[y1_crop:y2_crop, x1_crop:x2_crop].copy()
                    if cropped_img.size > 0:  # Check if the cropped image is not empty
                        # Convert BGR to RGB for EasyOCR
                        cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
                
                # Print results
                print(f'1 numberplate detected. Confidence: {det_conf:.2f}')
                print(f'Detection time: Load data ({(1E3 * (t1 - t0)):.1f}ms), Inference ({(1E3 * (t2 - t1)):.1f}ms)')
                
                # Log results if needed
                if self.log_level:
                    self.logger.debug(
                        f'{self.file_name} 1 numberplate detected. Conf: {det_conf:.2f}, '
                        f'Box: {x1},{y1},{x2},{y2}, '
                        f'Time: Load ({(1E3 * (t1 - t0)):.1f}ms), Inference ({(1E3 * (t2 - t1)):.1f}ms)'
                    )
        except Exception as e:
            print(f"Error during RKNN detection: {e}")
            t2 = time.time()
            
        print(f'Detection total time: {time.time() - t0:.3f}s')
        return {
            'file_name': self.file_name, 
            'orig_img': self.im0, 
            'cropped_img': cropped_img, 
            'bbox': bbox,
            'det_conf': det_conf
        }
        
    def _process_yolo_output(self, outputs, conf_thres):
        """Process YOLO outputs from RKNN inference
        
        NOTE: This is a placeholder implementation that should be adjusted to match the
        specific RKNN model's output format. The actual output format depends on:
        1. The original model architecture (YOLOv5, YOLOv8, etc.)
        2. How the model was converted to RKNN
        3. The specific version of RKNN toolkit used
        
        You will need to modify this method based on your specific model's output format.
        """
        # Example implementation (adjust according to your RKNN model's output)
        detections = []
        
        try:
            # Print output shapes for debugging
            print(f"RKNN output shapes for debugging:")
            for i, output in enumerate(outputs):
                print(f"  Output {i} shape: {output.shape}")
            
            # Different RKNN-converted YOLO models have different output formats
            # Below are several possible formats - uncomment and adapt the one that matches your model
            
            # Option 1: YOLOv5/v8-style output with [batch, num_boxes, 5+num_classes]
            # predictions = outputs[0]  # First output tensor
            # for pred in predictions[0]:  # Process first batch
            #     conf = pred[4]  # Confidence score is at index 4
            #     if conf > conf_thres:
            #         # Extract box coordinates (adjusted to x1y1x2y2 format)
            #         x, y, w, h = pred[0:4]
            #         x1 = x - w/2
            #         y1 = y - h/2
            #         x2 = x + w/2
            #         y2 = y + h/2
            #         detections.append({
            #             'bbox': [x1, y1, x2, y2],
            #             'confidence': float(conf)
            #         })
            
            # Option 2: Split outputs (common in some RKNN conversions)
            if len(outputs) >= 3:  # Model has separate outputs for boxes, scores, classes
                boxes = outputs[0]  # assuming first output contains boxes
                confidences = outputs[1]  # assuming second output contains confidences
                
                # Filter by confidence
                for i in range(len(confidences)):
                    conf = confidences[i][0]  # Adjust indexing as needed
                    if conf > conf_thres:
                        box = boxes[i]  # Adjust indexing as needed
                        detections.append({
                            'bbox': box,  # [x1, y1, x2, y2]
                            'confidence': float(conf)
                        })
            else:
                # Option 3: Single output tensor with all information
                predictions = outputs[0]  # First output tensor
                
                # Process each prediction (adjust indexing as needed)
                for pred in predictions:
                    # Assuming format [x1, y1, x2, y2, conf, class1, class2, ...]
                    conf = pred[4]
                    if conf > conf_thres:
                        x1, y1, x2, y2 = pred[0:4]
                        detections.append({
                            'bbox': [x1, y1, x2, y2],
                            'confidence': float(conf)
                        })
            
        except Exception as e:
            print(f"Error processing YOLO output: {e}")
            print("You may need to adjust the _process_yolo_output method to match your model's output format")
            
        return detections
        
    def release(self):
        """Release RKNN resources"""
        try:
            if hasattr(self, 'rknn'):
                self.rknn.release()
                print("RKNN resources released")
        except Exception as e:
            print(f"Error releasing RKNN resources: {e}")

# TensorRT accelerated Detector
class TensorRTDetector:
    def __init__(self, model_path, img_size=640, log_level='INFO', log_dir='./logs/'):
        self.model_path = model_path
        self.img_size = img_size
        self.log_level = log_level
        self.log_dir = log_dir
        
        # Set up logging
        if self.log_level:
            self.num_log_level = getattr(logging, self.log_level.upper(), 20)
            os.makedirs(os.path.dirname(self.log_dir), exist_ok=True)
            log_formatter = logging.Formatter("%(asctime)s %(message)s")
            logFile = self.log_dir + 'detection.log'
            my_handler = RotatingFileHandler(logFile, mode='a', maxBytes=25 * 1024 * 1024,
                                             backupCount=10, encoding='utf-8', delay=False)
            my_handler.setFormatter(log_formatter)
            my_handler.setLevel(self.num_log_level)
            self.logger = logging.getLogger('tensorrt_detector')  
            self.logger.setLevel(self.num_log_level)
            self.logger.addHandler(my_handler)
        
        # Initialize TensorRT
        self._initialize_tensorrt()
        
    def _initialize_tensorrt(self):
        """Initialize TensorRT engine"""
        try:
            # Create TensorRT logger
            logger = trt.Logger(trt.Logger.WARNING)
            
            # Create runtime
            self.runtime = trt.Runtime(logger)
            
            # Load engine
            with open(self.model_path, 'rb') as f:
                engine_bytes = f.read()
                self.engine = self.runtime.deserialize_cuda_engine(engine_bytes)
                
            # Create execution context
            self.context = self.engine.create_execution_context()
            
            # Allocate buffers
            self._allocate_buffers()
            
            print("TensorRT detector initialized successfully")
        except Exception as e:
            print(f"Error initializing TensorRT detector: {e}")
            raise
            
    def _allocate_buffers(self):
        """Allocate input and output buffers for TensorRT inference"""
        # Get input and output binding information
        self.input_binding_idx = 0
        self.output_binding_idx = 1
        
        # Get input and output shapes
        self.input_shape = self.engine.get_binding_shape(self.input_binding_idx)
        self.output_shape = self.engine.get_binding_shape(self.output_binding_idx)
        
        # Allocate device and host memory
        self.d_input = cuda.mem_alloc(self.input_shape.volume() * np.dtype(np.float32).itemsize)
        self.d_output = cuda.mem_alloc(self.output_shape.volume() * np.dtype(np.float32).itemsize)
        self.h_input = cuda.pagelocked_empty(self.input_shape.volume(), dtype=np.float32)
        self.h_output = cuda.pagelocked_empty(self.output_shape.volume(), dtype=np.float32)
        
        # Create CUDA stream
        self.stream = cuda.Stream()
        
    def run(self, inp_image, conf_thres=0.25):
        # Run inference
        t0 = time.time()
        
        # Clone the input image to avoid modifying the original
        self.im0 = inp_image.copy() if inp_image is not None else np.zeros((640, 480, 3), dtype=np.uint8)
        
        # Get file name (for logging purposes)
        self.file_name = f"frame_{int(time.time())}.jpg"
        
        t1 = time.time()
        try:
            # Preprocess image to match TensorRT input requirements
            preprocessed_img = self._preprocess_image(self.im0)
            
            # Copy input data to host buffer
            np.copyto(self.h_input, preprocessed_img.ravel())
            
            # Transfer input data to device
            cuda.memcpy_htod_async(self.d_input, self.h_input, self.stream)
            
            # Run inference
            self.context.execute_async_v2(
                bindings=[int(self.d_input), int(self.d_output)],
                stream_handle=self.stream.handle
            )
            
            # Transfer output data from device to host
            cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
            
            # Synchronize stream
            self.stream.synchronize()
            
            t2 = time.time()
            
            # Process output
            detections = self._process_output(self.h_output, conf_thres)
            
            # Initialize return values
            bbox = None
            cropped_img = None
            det_conf = None
            
            if detections and len(detections) > 0:
                # Get the detection with the highest confidence
                best_detection = max(detections, key=lambda x: x['confidence'])
                
                # Extract box coordinates and confidence
                x1, y1, x2, y2 = best_detection['bbox']
                det_conf = best_detection['confidence']
                
                # Convert coordinates back to original image scale
                orig_h, orig_w = self.im0.shape[:2]
                model_input_size = (self.img_size, self.img_size)  # Usually square for YOLO models
                scale_x = orig_w / model_input_size[0]
                scale_y = orig_h / model_input_size[1]
                
                x1 = int(x1 * scale_x)
                y1 = int(y1 * scale_y)
                x2 = int(x2 * scale_x)
                y2 = int(y2 * scale_y)
                
                # Store bbox
                bbox = [x1, y1, x2, y2]
                
                # Crop the license plate with margins
                if (y2 > y1) and (x2 > x1):  # Ensure valid dimensions
                    margin_x = int((x2 - x1) * 0.05)  # 5% margin reduction
                    margin_y = int((y2 - y1) * 0.1)   # 10% margin reduction
                    # Ensure margins don't go outside the image
                    x1_crop = max(0, x1 + margin_x)
                    y1_crop = max(0, y1 + margin_y)
                    x2_crop = min(self.im0.shape[1], x2 - margin_x)
                    y2_crop = min(self.im0.shape[0], y2 - margin_y)
                    
                    cropped_img = self.im0[y1_crop:y2_crop, x1_crop:x2_crop].copy()
                    if cropped_img.size > 0:  # Check if the cropped image is not empty
                        # Convert BGR to RGB for EasyOCR
                        cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
                
                # Print results
                print(f'1 numberplate detected. Confidence: {det_conf:.2f}')
                print(f'Detection time: Load data ({(1E3 * (t1 - t0)):.1f}ms), Inference ({(1E3 * (t2 - t1)):.1f}ms)')
                
                # Log results if needed
                if self.log_level:
                    self.logger.debug(
                        f'{self.file_name} 1 numberplate detected. Conf: {det_conf:.2f}, '
                        f'Box: {x1},{y1},{x2},{y2}, '
                        f'Time: Load ({(1E3 * (t1 - t0)):.1f}ms), Inference ({(1E3 * (t2 - t1)):.1f}ms)'
                    )
        except Exception as e:
            print(f"Error during TensorRT detection: {e}")
            t2 = time.time()
        
        print(f'Detection total time: {time.time() - t0:.3f}s')
        return {
            'file_name': self.file_name, 
            'orig_img': self.im0, 
            'cropped_img': cropped_img, 
            'bbox': bbox,
            'det_conf': det_conf
        }
        
    def _preprocess_image(self, image):
        """Preprocess image for TensorRT inference"""
        # Resize to model input size
        input_img = cv2.resize(image, (self.img_size, self.img_size))
        
        # Convert BGR to RGB
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1] and convert to float32
        input_img = input_img.astype(np.float32) / 255.0
        
        # Transpose to CHW format (NCHW expected by TensorRT)
        input_img = input_img.transpose(2, 0, 1)
        
        # Add batch dimension
        input_img = np.expand_dims(input_img, axis=0)
        
        return input_img
        
    def _process_output(self, output, conf_thres):
        """Process TensorRT output to get detections
        
        NOTE: This is a placeholder implementation that should be adjusted
        to match your specific TensorRT model's output format. The actual 
        output format depends on how the model was converted to TensorRT.
        
        You will need to modify this method based on your specific model's output format.
        """
        detections = []
        
        # Example output processing logic
        try:
            # Reshape output to expected format
            # This will depend on your TensorRT model
            output = output.reshape(self.output_shape)
            
            # Debug output shape
            print(f"TensorRT output shape: {output.shape}")
            
            # Different TensorRT models may have different output formats
            # Here are a few common formats:
            
            # Option 1: YOLOv5/v8 format with [batch, num_boxes, 5+num_classes]
            if len(output.shape) == 3 and output.shape[2] > 5:  # YOLO format
                # Process first batch
                for i in range(output.shape[1]):  # Iterate over detections
                    confidence = output[0, i, 4]  # Confidence is at index 4
                    if confidence > conf_thres:
                        # Extract box coordinates (center format)
                        x_center = output[0, i, 0]
                        y_center = output[0, i, 1]
                        width = output[0, i, 2]
                        height = output[0, i, 3]
                        
                        # Convert to x1y1x2y2 format
                        x1 = x_center - width / 2
                        y1 = y_center - height / 2
                        x2 = x_center + width / 2
                        y2 = y_center + height / 2
                        
                        detections.append({
                            'bbox': [x1, y1, x2, y2],
                            'confidence': float(confidence)
                        })
            
            # Option 2: Single dimension output with detections serialized
            elif len(output.shape) == 1:
                # Determine how many values per detection (typically 5-7)
                # This depends on your model configuration
                values_per_detection = 7  # Adjust based on your model
                num_detections = len(output) // values_per_detection
                
                for i in range(num_detections):
                    start_idx = i * values_per_detection
                    # Format: [x1, y1, x2, y2, confidence, class_id, ...]
                    x1 = output[start_idx]
                    y1 = output[start_idx + 1]
                    x2 = output[start_idx + 2]
                    y2 = output[start_idx + 3]
                    confidence = output[start_idx + 4]
                    
                    if confidence > conf_thres:
                        detections.append({
                            'bbox': [x1, y1, x2, y2],
                            'confidence': float(confidence)
                        })
            
            # Option 3: Default fallback (simple 2D format)
            else:
                for i in range(output.shape[0]):  # Iterate over detections
                    confidence = output[i, 4]  # Confidence is at index 4 in this example
                    if confidence > conf_thres:
                        # Extract box coordinates
                        x_center = output[i, 0]
                        y_center = output[i, 1]
                        width = output[i, 2]
                        height = output[i, 3]
                        
                        # Convert to x1y1x2y2 format
                        x1 = x_center - width / 2
                        y1 = y_center - height / 2
                        x2 = x_center + width / 2
                        y2 = y_center + height / 2
                        
                        detections.append({
                            'bbox': [x1, y1, x2, y2],
                            'confidence': float(confidence)
                        })
                        
        except Exception as e:
            print(f"Error processing TensorRT output: {e}")
            print("You may need to adjust the _process_output method to match your model's output format")
            
        return detections
        
    def release(self):
        """Release TensorRT resources"""
        # Nothing specific to clean up for TensorRT
        pass

# YOLO Detector class with CUDA acceleration
class YOLODetector:
    def __init__(self, model_weights, img_size=640, device='cuda:0' if torch.cuda.is_available() else 'cpu', 
                 log_level='INFO', log_dir='./logs/'):
        # Initialize
        self.model_weights = model_weights
        self.img_size = img_size
        self.device = device
        self.log_level = log_level
        self.log_dir = log_dir
        
        if self.log_level:
            self.num_log_level = getattr(logging, self.log_level.upper(), 20)
            os.makedirs(os.path.dirname(self.log_dir), exist_ok=True)
            log_formatter = logging.Formatter("%(asctime)s %(message)s")
            logFile = self.log_dir + 'detection.log'
            my_handler = RotatingFileHandler(logFile, mode='a', maxBytes=25 * 1024 * 1024,
                                             backupCount=10, encoding='utf-8', delay=False)
            my_handler.setFormatter(log_formatter)
            my_handler.setLevel(self.num_log_level)
            self.logger = logging.getLogger('yolo_detector')  
            self.logger.setLevel(self.num_log_level)
            self.logger.addHandler(my_handler)
        
        try:
            # Load YOLO model
            self.model = YOLO(self.model_weights)
            print(f"Successfully loaded model from {self.model_weights}")
            print(f"Using device: {self.device}")
            
            # Set model device
            if 'cuda' in self.device and torch.cuda.is_available():
                print(f"Using CUDA acceleration with {torch.cuda.get_device_name(0)}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Using default YOLO11s model")
            try:
                # Try to download YOLO11s model
                self.model = YOLO("yolo11s.pt")
            except:
                print("Failed to load YOLO11s model, falling back to YOLO11n")
                try:
                    self.model = YOLO("yolo11n.pt")
                except:
                    print("Failed to load YOLO11n, falling back to YOLOv8")
                    self.model = YOLO("yolov8n.pt")
        
        # Set up colors for visualization
        self.colors = [[0, 255, 127]]  # Default color for license plate
        
    def run(self, inp_image, conf_thres=0.25):
        # Run Inference
        t0 = time.time()
        
        # Clone the input image to avoid modifying the original
        self.im0 = inp_image.copy() if inp_image is not None else np.zeros((640, 480, 3), dtype=np.uint8)
        
        # Get file name (for logging purposes)
        self.file_name = f"frame_{int(time.time())}.jpg"
        
        # Run YOLO inference
        t1 = time.time()
        try:
            results = self.model.predict(self.im0, conf=conf_thres, device=self.device)
            t2 = time.time()
            
            # Process detections
            bbox = None  # bounding box of detected object with max conf
            cropped_img = None  # cropped detected object with max conf
            det_conf = None  # confidence level for detected object with max conf
            
            # Get the first result (only one image is processed at a time)
            if len(results) > 0:
                result = results[0]
                
                if len(result.boxes) > 0:
                    # Get boxes (convert to numpy for easier handling)
                    boxes = result.boxes.xyxy.cpu().numpy()  # xyxy format
                    confidences = result.boxes.conf.cpu().numpy()
                    class_ids = result.boxes.cls.cpu().numpy().astype(int)
                    
                    # Find the detection with the highest confidence
                    max_conf_idx = np.argmax(confidences)
                    max_conf = confidences[max_conf_idx]
                    max_box = boxes[max_conf_idx]
                    max_class = class_ids[max_conf_idx]
                    
                    # Extract information for the max confidence detection
                    bbox = torch.tensor(max_box)  # Convert back to tensor for compatibility
                    x1, y1, x2, y2 = map(int, max_box)
                    
                    # Crop the license plate from the image with slightly tighter margins
                    if (y2 > y1) and (x2 > x1):  # Ensure valid dimensions
                        margin_x = int((x2 - x1) * 0.05)  # 5% margin reduction
                        margin_y = int((y2 - y1) * 0.1)   # 10% margin reduction
                        # Ensure margins don't go outside the image
                        x1_crop = max(0, x1 + margin_x)
                        y1_crop = max(0, y1 + margin_y)
                        x2_crop = min(self.im0.shape[1], x2 - margin_x)
                        y2_crop = min(self.im0.shape[0], y2 - margin_y)
                        
                        cropped_img = self.im0[y1_crop:y2_crop, x1_crop:x2_crop].copy()
                        if cropped_img.size > 0:  # Check if the cropped image is not empty
                            # Convert BGR to RGB for EasyOCR
                            cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
                    
                    # Store confidence
                    det_conf = max_conf
                    
                    # Print results
                    print(f'1 numberplate detected. Confidence: {max_conf:.2f}')
                    print(f'Detection time: Load data ({(1E3 * (t1 - t0)):.1f}ms), Inference ({(1E3 * (t2 - t1)):.1f}ms)')
                    
                    # Log results if needed
                    if self.log_level:
                        self.logger.debug(
                            f'{self.file_name} 1 numberplate detected. Conf: {max_conf:.2f}, '
                            f'Box: {x1},{y1},{x2},{y2}, '
                            f'Time: Load ({(1E3 * (t1 - t0)):.1f}ms), Inference ({(1E3 * (t2 - t1)):.1f}ms)'
                        )
        except Exception as e:
            print(f"Error during detection: {e}")
            t2 = time.time()
        
        print(f'Detection total time: {time.time() - t0:.3f}s')
        return {
            'file_name': self.file_name, 
            'orig_img': self.im0, 
            'cropped_img': cropped_img, 
            'bbox': bbox,
            'det_conf': det_conf
        }

# Skew Correction (projection profile) - Enhanced version
def find_score(arr, angle):
    data = rotate(arr, angle, reshape=False, order=0)
    hist = np.sum(data, axis=1)
    score = np.sum((hist[1:] - hist[:-1]) ** 2)
    return hist, score

def find_angle(img, delta=0.5, limit=10):
    angles = np.arange(-limit, limit+delta, delta)
    scores = []
    for angle in angles:
        hist, score = find_score(img, angle)
        scores.append(score)
    best_score = max(scores)
    best_angle = angles[scores.index(best_score)]
    print(f'Best angle: {best_angle}')
    return best_angle

def correct_skew(img):
    # correct skew
    try:
        if img is None or img.size == 0:
            return img
            
        # Convert to grayscale if needed
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
            
        best_angle = find_angle(gray)
        data = rotate(img, best_angle, reshape=False, order=0)
        return data
    except Exception as e:
        print(f"Error in skew correction: {e}")
        return img  # Return original image on error

# Enhanced OCR image preprocessing function with multiple techniques
def ocr_img_preprocess(img, method='auto'):
    """
    Preprocess an image for OCR using various techniques.
    
    Args:
        img: Input image (RGB or grayscale)
        method: Preprocessing method to use. Options:
            - 'auto': Try all methods and return all results
            - 'basic': Simple binary thresholding
            - 'adaptive': Adaptive thresholding with morphological operations
            - 'otsu': Otsu's thresholding with Gaussian blur
            - 'canny': Canny edge detection
            - 'edge_enhance': Edge enhancement using Laplacian
    
    Returns:
        When method='auto': List of preprocessed images
        Otherwise: Single preprocessed image
        Returns None if input image is invalid
    """
    if img is None or img.size == 0:
        print("Cannot preprocess empty image")
        return None
    
    methods = ['basic', 'adaptive', 'otsu', 'canny', 'edge_enhance'] if method == 'auto' else [method]
    results = []
    
    try:
        # Convert to grayscale - use BGR2GRAY since OpenCV uses BGR by default
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # Image is already converted to RGB
        else:
            gray = img
        
        # Correct skew if needed
        gray = correct_skew(gray)
        
        for method in methods:
            if method == 'basic':
                # Simple binary thresholding
                _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                results.append(thresh)
                
            elif method == 'adaptive':
                # Apply bilateral filter for noise removal while preserving edges
                filtered = cv2.bilateralFilter(gray, 11, 17, 17)
                
                # Apply adaptive thresholding to handle different lighting conditions
                thresh = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                               cv2.THRESH_BINARY_INV, 11, 2)
                
                # Apply morphological operations to clean up the image
                kernel = np.ones((3, 3), np.uint8)
                thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
                thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
                
                results.append(thresh)
                
            elif method == 'otsu':
                # Apply GaussianBlur to reduce noise
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                
                # Apply Otsu's thresholding
                _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                
                # Noise removal with morphological operations
                kernel = np.ones((3, 3), np.uint8)
                thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
                
                results.append(thresh)
                
            elif method == 'canny':
                # Use Canny edge detection
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                edges = cv2.Canny(blurred, 50, 150)
                
                # Dilate edges to connect components
                kernel = np.ones((3, 3), np.uint8)
                edges = cv2.dilate(edges, kernel, iterations=1)
                
                results.append(edges)
                
            elif method == 'edge_enhance':
                # Edge enhancement using Laplacian
                laplacian = cv2.Laplacian(gray, cv2.CV_8U)
                
                # Normalize and threshold
                _, thresh = cv2.threshold(laplacian, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
                results.append(thresh)
                
        # If auto method, return all results for multi-hypothesis processing
        if method == 'auto':
            return results
        else:
            return results[0]
            
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        if img is not None:
            try:
                # Fallback to simple grayscale conversion
                return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img
            except:
                return img
        return None

# Enhanced license plate text cleaning function
def clean_plate_text(text, country='generic'):
    """
    Clean up detected license plate text with improved accuracy.
    
    Args:
        text: Raw text from OCR
        country: Country format to use for validation ('generic' or country code)
               Currently supported: 'generic', 'lithuania' 
               (Add more country-specific formats as needed)
    
    Returns:
        Cleaned and validated license plate text, or None if cleaning failed
    """
    if not text:
        return None
        
    # Normalize plate text (remove spaces, convert to uppercase)
    normalized = text.upper().replace(' ', '')
    
    # Basic error correction mappings for OCR mistakes
    char_map = {
        'O': '0', 'Q': '0', 'D': '0',  # Similar to 0
        'I': '1', 'J': '1', 'L': '1',  # Similar to 1
        'Z': '2',  # Similar to 2
        'A': '4',  # Sometimes confused
        'S': '5', 'Ş': '5',  # Similar to 5
        'G': '6',  # Similar to 6
        'T': '7',  # Similar to 7
        'B': '8',  # Similar to 8
        'g': '9',  # Similar to 9
    }
    
    # Apply error correction
    corrected = ''.join(char_map.get(c, c) for c in normalized)
    
    # Country-specific format handling
    if country == 'lithuania':
        # Try to extract standard Lithuanian plate format (3 letters + 3 digits)
        match = re.search(r'[A-Z]{3}\s*\d{3}', corrected)
        if match:
            return match.group(0).replace(" ", "")
    # Add more country-specific formats as needed
    # elif country == 'uk':
    #     # UK format: two letters, two digits, space, three letters
    #     match = re.search(r'[A-Z]{2}\d{2}\s*[A-Z]{3}', corrected)
    #     if match:
    #         return match.group(0).replace(" ", "")
    # elif country == 'usa':
    #     # Various US formats - this is simplified
    #     match = re.search(r'[A-Z0-9]{5,8}', corrected)
    #     if match:
    #         return match.group(0)
    
    # Generic cleaning for non-country specific formats
    # Remove common OCR noise and special characters
    cleaned = ''.join(c for c in corrected if c.isalnum())
    
    # Reject if too short to be a valid plate
    if len(cleaned) < 4:
        return None
        
    # If too long, try to extract reasonable segments (3-8 chars)
    if len(cleaned) > 8:
        # Look for common patterns that might be license plates
        # Pattern: 2-3 letters followed by 2-5 digits (common in many countries)
        patterns = re.findall(r'[A-Z]{2,3}\d{2,5}', cleaned)
        if patterns:
            return patterns[0]
        
        # Pattern: 1-3 digits followed by 2-4 letters (also common)
        patterns = re.findall(r'\d{1,3}[A-Z]{2,4}', cleaned)
        if patterns:
            return patterns[0]
        
        # If no patterns found, just take the middle characters
        # (License plates are usually 6-8 characters)
        mid_point = len(cleaned) // 2
        plate_len = 7  # typical length
        start = max(0, mid_point - plate_len // 2)
        end = min(len(cleaned), start + plate_len)
        cleaned = cleaned[start:end]
    
    return cleaned

# Enhanced EasyOCR class with multi-hypothesis processing
class EnhancedEasyOcr:
    def __init__(self, lang=['en'], allow_list='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ', 
                 min_size=50, log_level='INFO', log_dir='./logs/'):
        try:
            # Use CUDA if available
            gpu = torch.cuda.is_available()
            self.reader = easyocr.Reader(lang, gpu=gpu)
            self.allow_list = allow_list
            self.min_size = min_size
            self.log_level = log_level
            
            # Track recent successful detections for confidence-based filtering
            self.recent_detections = deque(maxlen=10)
            
            if self.log_level:
                self.num_log_level = getattr(logging, log_level.upper(), 20)
                self.log_dir = log_dir
                os.makedirs(os.path.dirname(self.log_dir), exist_ok=True)
                
                # Set logger
                log_formatter = logging.Formatter("%(asctime)s %(message)s")
                logFile = self.log_dir + 'ocr.log'
                my_handler = RotatingFileHandler(logFile, mode='a', maxBytes=25 * 1024 * 1024,
                                                backupCount=10, encoding='utf-8', delay=False)
                my_handler.setFormatter(log_formatter)
                my_handler.setLevel(self.num_log_level)
                self.logger = logging.getLogger('enhanced_ocr')  
                self.logger.setLevel(self.num_log_level)
                self.logger.addHandler(my_handler)
        except Exception as e:
            print(f"Error initializing EasyOCR: {e}")
            self.reader = None
            
    def run(self, detect_result_dict):
        try:
            if detect_result_dict.get('cropped_img') is not None and self.reader is not None:
                t0 = time.time()
                img = detect_result_dict['cropped_img']
                
                # Image should already be in RGB format from detector
                file_name = detect_result_dict.get('file_name', 'unknown')
                
                # Process with multiple preprocessing methods for better accuracy
                processed_imgs = ocr_img_preprocess(img, method='auto')
                if processed_imgs is None or len(processed_imgs) == 0:
                    return {'text': None, 'confid': None}
                
                # Run OCR on all processed images in parallel
                ocr_results = []
                
                # For each preprocessing method
                for processed_img in processed_imgs:
                    # Adjust OCR parameters for license plates
                    try:
                        result = self.reader.readtext(
                            processed_img, 
                            allowlist=self.allow_list,
                            min_size=self.min_size,
                            paragraph=False,  # Important: treat each text separately
                            contrast_ths=0.2,
                            adjust_contrast=0.5,
                            text_threshold=0.7  # Higher threshold for more precise text
                        )
                        
                        # Extract text and confidence
                        if result:
                            # Combine detected text segments
                            text = "".join([x[1] for x in result])
                            # Use average confidence of segments
                            confid = float(np.mean([x[2] for x in result]))
                            
                            # Clean up the detected text
                            cleaned_text = clean_plate_text(text)
                            
                            if cleaned_text and len(cleaned_text) >= 4:  # Minimum reasonable plate length
                                ocr_results.append({
                                    'text': cleaned_text,
                                    'confid': confid
                                })
                    except Exception as e:
                        print(f"Error in OCR processing for a preprocessing method: {e}")
                
                # Select the best result based on confidence and historical data
                if ocr_results:
                    # Sort by confidence
                    ocr_results.sort(key=lambda x: x['confid'], reverse=True)
                    
                    # Get highest confidence result
                    best_result = ocr_results[0]
                    text = best_result['text']
                    confid = best_result['confid']
                    
                    # Add to recent detections for tracking
                    self.recent_detections.append(text)
                    
                    # Log all results
                    print(f"All OCR results: {[r['text'] for r in ocr_results]}")
                    print(f"Selected result: {text} with confidence {confid:.2f}")
                    
                    # Round confidence to 2 decimal places
                    confid = round(confid, 2)
                    
                    t1 = time.time()
                    print(f'Recognized number: {text}, conf.:{confid}.\nOCR total time: {(t1 - t0):.3f}s')
                    
                    if self.log_level:
                        # Write results to file if debug mode
                        self.logger.debug(f'{file_name} Recognized number: {text}, conf.:{confid}, OCR total time: {(t1 - t0):.3f}s.')
                    
                    return {'text': text, 'confid': confid}
                else:
                    # No valid OCR results
                    return {'text': None, 'confid': None}
            else:
                return {'text': None, 'confid': None}
        except Exception as e:
            print(f"Error in OCR processing: {e}")
            return {'text': None, 'confid': None}

# Vehicle Make Classifier
class VehicleMakeClassifier:
    def __init__(self, model_path='./models/vehicle_make_resnet18.pth', 
                 labels_path='./models/vehicle_make_labels.txt',
                 device='cuda:0' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model_path = model_path
        self.labels_path = labels_path
        self.enabled = False  # Disabled by default
        
        # Only initialize if the model file exists
        if os.path.exists(model_path) and os.path.exists(labels_path):
            try:
                # Load the model
                self.model = torch.load(model_path, map_location=self.device)
                self.model.eval()
                
                # Load class labels
                with open(labels_path, 'r') as f:
                    self.labels = [line.strip() for line in f.readlines()]
                
                self.enabled = True
                print(f"Vehicle make classifier initialized with {len(self.labels)} classes")
            except Exception as e:
                print(f"Error initializing vehicle make classifier: {e}")
                self.enabled = False
        else:
            print(f"Vehicle make classifier disabled (model files not found)")
            
    def extract_vehicle_image(self, frame, plate_bbox, expansion_factor=2.0):
        """Extract the vehicle image from the frame using the license plate location"""
        if frame is None or plate_bbox is None:
            return None
            
        try:
            h, w = frame.shape[:2]
            
            # Convert to list if tensor
            if isinstance(plate_bbox, torch.Tensor):
                plate_bbox = plate_bbox.tolist()
                
            x1, y1, x2, y2 = map(int, plate_bbox)
            
            # Calculate expanded region (upward and to the sides from the plate)
            plate_width = x2 - x1
            plate_height = y2 - y1
            
            # Expand upward more than to the sides
            ex1 = max(0, int(x1 - plate_width * (expansion_factor - 1) / 2))
            ey1 = max(0, int(y1 - plate_height * expansion_factor))
            ex2 = min(w, int(x2 + plate_width * (expansion_factor - 1) / 2))
            ey2 = min(h, y2)  # Don't expand downward
            
            # Extract vehicle image
            vehicle_img = frame[ey1:ey2, ex1:ex2].copy()
            
            if vehicle_img.size == 0:
                return None
                
            return vehicle_img
        except Exception as e:
            print(f"Error extracting vehicle image: {e}")
            return None
            
    def preprocess_image(self, image):
        """Preprocess the image for model input"""
        try:
            # Resize
            img = cv2.resize(image, (224, 224))
            
            # Convert to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Convert to float and normalize
            img = img.astype(np.float32) / 255.0
            
            # Apply ImageNet normalization
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            img = (img - mean) / std
            
            # Convert to tensor and add batch dimension
            img = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)
            
            return img
        except Exception as e:
            print(f"Error preprocessing image: {e}")
            return None
            
    def predict(self, frame, plate_bbox):
        """Predict the make of the vehicle"""
        if not self.enabled:
            return None
            
        try:
            # Extract vehicle image
            vehicle_img = self.extract_vehicle_image(frame, plate_bbox)
            if vehicle_img is None:
                return None
                
            # Preprocess image
            img_tensor = self.preprocess_image(vehicle_img)
            if img_tensor is None:
                return None
                
            # Send to device
            img_tensor = img_tensor.to(self.device)
            
            # Get prediction
            with torch.no_grad():
                outputs = self.model(img_tensor)
                _, predicted = torch.max(outputs, 1)
                
                # Get confidence
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence = probabilities[0][predicted.item()].item()
                
                # Get class label
                make = self.labels[predicted.item()]
                
                return {
                    'make': make,
                    'confidence': round(confidence, 3)
                }
        except Exception as e:
            print(f"Error predicting vehicle make: {e}")
            return None
            
    def set_enabled(self, enabled):
        """Enable or disable the classifier"""
        if enabled and not self.enabled and os.path.exists(self.model_path):
            try:
                # Initialize the model if it's not already initialized
                if not hasattr(self, 'model'):
                    self.model = torch.load(self.model_path, map_location=self.device)
                    self.model.eval()
                    
                    # Load class labels if not already loaded
                    if not hasattr(self, 'labels'):
                        with open(self.labels_path, 'r') as f:
                            self.labels = [line.strip() for line in f.readlines()]
                    
                self.enabled = True
                print("Vehicle make classifier enabled")
            except Exception as e:
                print(f"Error enabling vehicle make classifier: {e}")
        elif not enabled:
            self.enabled = False
            print("Vehicle make classifier disabled")

# Vehicle Color Detector
class VehicleColorDetector:
    def __init__(self, color_map_path='./models/color_map.json'):
        self.enabled = False  # Disabled by default
        self.color_map_path = color_map_path
        
        # Load color mapping if available
        if os.path.exists(color_map_path):
            try:
                import json
                with open(color_map_path, 'r') as f:
                    self.color_map = json.load(f)
                self.enabled = True
                print("Vehicle color detector initialized")
            except Exception as e:
                print(f"Error loading color map: {e}")
                self.color_map = self._default_color_map()
        else:
            # Use default color mapping
            self.color_map = self._default_color_map()
            print("Using default color mapping for vehicle color detection")
            
    def _default_color_map(self):
        """Return default color map"""
        return {
            "black": [0, 0, 0],
            "white": [255, 255, 255],
            "red": [255, 0, 0],
            "green": [0, 255, 0],
            "blue": [0, 0, 255],
            "yellow": [255, 255, 0],
            "gray": [128, 128, 128],
            "silver": [192, 192, 192],
            "brown": [165, 42, 42],
            "orange": [255, 165, 0]
        }
        
    def extract_vehicle_image(self, frame, plate_bbox, expansion_factor=2.5):
        """Extract the vehicle image from the frame using the license plate location"""
        if frame is None or plate_bbox is None:
            return None
            
        try:
            h, w = frame.shape[:2]
            
            # Convert to list if tensor
            if isinstance(plate_bbox, torch.Tensor):
                plate_bbox = plate_bbox.tolist()
                
            x1, y1, x2, y2 = map(int, plate_bbox)
            
            # Calculate expanded region (upward and to the sides from the plate)
            plate_width = x2 - x1
            plate_height = y2 - y1
            
            # Expand more upward to capture the vehicle body
            ex1 = max(0, int(x1 - plate_width * expansion_factor / 2))
            ey1 = max(0, int(y1 - plate_height * expansion_factor))
            ex2 = min(w, int(x2 + plate_width * expansion_factor / 2))
            ey2 = min(h, y1)  # Do not include the license plate itself
            
            # Extract vehicle image
            vehicle_img = frame[ey1:ey2, ex1:ex2].copy()
            
            if vehicle_img.size == 0:
                # Try a different approach if the vehicle body is not visible above the plate
                # In this case, we'll include more of the image around the plate
                ex1 = max(0, int(x1 - plate_width))
                ey1 = max(0, int(y1 - plate_height * 0.5))
                ex2 = min(w, int(x2 + plate_width))
                ey2 = min(h, int(y2 + plate_height * 0.5))
                
                vehicle_img = frame[ey1:ey2, ex1:ex2].copy()
                
                if vehicle_img.size == 0:
                    return None
                    
            return vehicle_img
        except Exception as e:
            print(f"Error extracting vehicle image for color detection: {e}")
            return None
            
    def detect_color(self, frame, plate_bbox):
        """Detect the color of the vehicle"""
        if not self.enabled:
            return None
            
        try:
            # Extract vehicle image
            vehicle_img = self.extract_vehicle_image(frame, plate_bbox)
            if vehicle_img is None:
                return None
                
            # Convert to RGB if needed
            if len(vehicle_img.shape) == 3 and vehicle_img.shape[2] == 3:
                img_rgb = cv2.cvtColor(vehicle_img, cv2.COLOR_BGR2RGB)
            else:
                return None
                
            # Reshape image for clustering
            pixels = img_rgb.reshape(-1, 3)
            
            # Use k-means clustering to find dominant colors
            # Usually cars have 1-2 dominant colors
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            kmeans.fit(pixels)
            
            # Get dominant colors
            dominant_colors = kmeans.cluster_centers_.astype(int)
            
            # Get cluster sizes (number of pixels for each color)
            counts = np.bincount(kmeans.labels_)
            
            # Sort colors by count (most dominant first)
            dominant_colors = dominant_colors[np.argsort(counts)[::-1]]
            
            # Map to named colors
            result = self._map_to_named_color(dominant_colors[0])
            
            return result
        except Exception as e:
            print(f"Error detecting vehicle color: {e}")
            return None
            
    def _map_to_named_color(self, rgb_color):
        """Map RGB color to a named color"""
        min_distance = float('inf')
        closest_color = None
        
        # Compare RGB distance to each named color
        for color_name, color_rgb in self.color_map.items():
            # Calculate Euclidean distance
            distance = np.sqrt(np.sum((np.array(rgb_color) - np.array(color_rgb)) ** 2))
            
            if distance < min_distance:
                min_distance = distance
                closest_color = color_name
                
        return {
            'color': closest_color,
            'rgb': rgb_color.tolist(),
            'confidence': max(0, min(1, 1.0 - min_distance / 441.7))  # Scale confidence
        }
        
    def set_enabled(self, enabled):
        """Enable or disable the color detector"""
        self.enabled = enabled
        print(f"Vehicle color detector {'enabled' if enabled else 'disabled'}")

# Allowed plates checker class with make/color verification support
class PlateChecker:
    def __init__(self, allow_list_file=None, json_file=None, verify_make_color=False, 
                 log_level='INFO', log_dir='./logs/'):
        self.allow_list = []
        self.extended_info = {}  # Dictionary to store make/color information
        self.log_level = log_level
        self.log_dir = log_dir
        self.allow_list_file = allow_list_file or './allowed_plates.txt'
        self.json_file = json_file or './allowed_plates.json'
        self.verify_make_color = verify_make_color
        
        # Set up logging
        if self.log_level:
            self.num_log_level = getattr(logging, self.log_level.upper(), 20)
            os.makedirs(os.path.dirname(self.log_dir), exist_ok=True)
            log_formatter = logging.Formatter("%(asctime)s %(message)s")
            logFile = self.log_dir + 'plate_checker.log'
            my_handler = RotatingFileHandler(logFile, mode='a', maxBytes=25 * 1024 * 1024,
                                          backupCount=10, encoding='utf-8', delay=False)
            my_handler.setFormatter(log_formatter)
            my_handler.setLevel(self.num_log_level)
            self.logger = logging.getLogger('plate_checker')  
            self.logger.setLevel(self.num_log_level)
            self.logger.addHandler(my_handler)
        
        # Load allowed plate list
        self.load_allowed_plates()
        
        # Load extended information if available
        if self.json_file:
            self.load_extended_info()
            
    def load_allowed_plates(self):
        try:
            if os.path.exists(self.allow_list_file):
                with open(self.allow_list_file, 'r') as f:
                    self.allow_list = [line.strip() for line in f.readlines() if line.strip()]
                
                print(f"Loaded {len(self.allow_list)} allowed plates from {self.allow_list_file}")
                if self.log_level:
                    self.logger.info(f"Loaded {len(self.allow_list)} allowed plates from {self.allow_list_file}")
            else:
                print(f"Warning: Allowed plate list file {self.allow_list_file} not found")
                if self.log_level:
                    self.logger.warning(f"Allowed plate list file {self.allow_list_file} not found")
                # Create empty file
                try:
                    with open(self.allow_list_file, 'w') as f:
                        pass
                    print(f"Created empty allowed plates file: {self.allow_list_file}")
                except Exception as e:
                    print(f"Error creating allowed plates file: {e}")
        except Exception as e:
            print(f"Error loading allowed plates: {e}")
            if self.log_level:
                self.logger.error(f"Error loading allowed plates: {e}")
            
            # Create empty file as fallback
            try:
                with open(self.allow_list_file, 'w') as f:
                    pass
            except:
                pass
                
    def load_extended_info(self):
        """Load extended plate information from JSON file with make/color details"""
        try:
            if os.path.exists(self.json_file):
                import json
                with open(self.json_file, 'r') as f:
                    try:
                        # Check if file is empty
                        file_content = f.read().strip()
                        if not file_content:
                            print(f"Warning: Extended plate info file {self.json_file} is empty")
                            if self.log_level:
                                self.logger.warning(f"Extended plate info file {self.json_file} is empty")
                            return
                            
                        # Parse JSON (seek back to beginning of file)
                        f.seek(0)
                        data = json.load(f)
                        
                        if not isinstance(data, list):
                            print(f"Error: {self.json_file} should contain a JSON array/list of objects")
                            if self.log_level:
                                self.logger.error(f"{self.json_file} should contain a JSON array/list of objects")
                            return
                        
                        # Process each entry
                        added_plates = []
                        for entry in data:
                            # Skip entries without plate field
                            if 'plate' not in entry:
                                print(f"Warning: Skipping entry without 'plate' field in {self.json_file}")
                                if self.log_level:
                                    self.logger.warning(f"Skipping entry without 'plate' field in {self.json_file}")
                                continue
                                
                            try:
                                plate = str(entry['plate']).upper().replace(' ', '')
                                
                                # Store in extended_info dictionary
                                self.extended_info[plate] = {
                                    'make': entry.get('make'),
                                    'color': entry.get('color'),
                                    'require_match': bool(entry.get('require_match', False))
                                }
                                
                                # Also add to allowed list if not already there
                                if plate not in self.allow_list:
                                    self.allow_list.append(plate)
                                    added_plates.append(plate)
                            except Exception as e:
                                print(f"Error processing entry in {self.json_file}: {e}")
                                if self.log_level:
                                    self.logger.error(f"Error processing entry in {self.json_file}: {e}")
                                continue
                        
                        print(f"Loaded {len(self.extended_info)} plates with extended information from {self.json_file}")
                        if added_plates:
                            print(f"Added {len(added_plates)} new plates to allowed list")
                            # Update the text file to include these plates
                            try:
                                with open(self.allow_list_file, 'a') as f:
                                    for plate in added_plates:
                                        f.write(f"{plate}\n")
                            except Exception as e:
                                print(f"Error updating allowed_plates.txt with new plates: {e}")
                                if self.log_level:
                                    self.logger.error(f"Error updating allowed_plates.txt with new plates: {e}")
                                    
                        if self.log_level:
                            self.logger.info(f"Loaded {len(self.extended_info)} plates with extended information from {self.json_file}")
                        
                    except json.JSONDecodeError as e:
                        print(f"Error: {self.json_file} is not a valid JSON file - {e}")
                        if self.log_level:
                            self.logger.error(f"{self.json_file} is not a valid JSON file - {e}")
                    except Exception as e:
                        print(f"Error processing {self.json_file}: {e}")
                        if self.log_level:
                            self.logger.error(f"Error processing {self.json_file}: {e}")
            else:
                if self.verify_make_color:
                    print(f"Warning: Extended plate info file {self.json_file} not found but make/color verification is enabled")
                    print(f"         Make/color verification will have no effect without the JSON file")
                    if self.log_level:
                        self.logger.warning(f"Extended plate info file {self.json_file} not found but make/color verification is enabled")
                    
                    # Create empty JSON file as template
                    try:
                        import json
                        with open(self.json_file, 'w') as f:
                            json.dump([
                                {"plate": "ABC123", "make": "BMW", "color": "blue", "require_match": true},
                                {"plate": "DEF456", "make": "Toyota", "color": "red", "require_match": true}
                            ], f, indent=2)
                        print(f"Created template JSON file at {self.json_file}")
                    except Exception as e:
                        print(f"Error creating template JSON file: {e}")
        except Exception as e:
            print(f"Error loading extended plate information: {e}")
            if self.log_level:
                self.logger.error(f"Error loading extended plate information: {e}")
            
    def check(self, plate_number, vehicle_make=None, vehicle_color=None):
        """
        Check if a plate is allowed, optionally verifying make and color
        
        Args:
            plate_number: The license plate number to check
            vehicle_make: Optional dict with detected make info {'make': 'BMW', 'confidence': 0.85}
            vehicle_color: Optional dict with detected color info {'color': 'blue', 'confidence': 0.92}
            
        Returns:
            "Allowed": If the plate is allowed (and make/color match if required)
            "Prohibited!": If the plate is not in the allowed list
            "Prohibited! Make Mismatch": If plate is allowed but make doesn't match
            "Prohibited! Color Mismatch": If plate is allowed but color doesn't match
            "Unknown": If there was an error or plate_number is None
        """
        if not plate_number:
            return "Unknown"
            
        try:
            # Normalize plate number (remove spaces, convert to uppercase)
            normalized_plate = str(plate_number).upper().replace(' ', '')
            
            # First check if plate is in allowed list
            if normalized_plate in self.allow_list:
                # If we're not verifying make/color, or no extended info exists, just allow
                if not self.verify_make_color or normalized_plate not in self.extended_info:
                    if self.log_level:
                        self.logger.info(f"Plate {normalized_plate} is in the allowed list")
                    return "Allowed"
                
                # Get extended info for this plate
                ext_info = self.extended_info[normalized_plate]
                
                # If we don't require matching, just allow
                if not ext_info.get('require_match', False):
                    if self.log_level:
                        self.logger.info(f"Plate {normalized_plate} is in the allowed list (no matching required)")
                    return "Allowed"
                
                # Verify make if needed
                if ext_info.get('make') and vehicle_make and 'make' in vehicle_make:
                    try:
                        expected_make = str(ext_info['make']).lower()
                        detected_make = str(vehicle_make['make']).lower()
                        
                        if expected_make != detected_make:
                            if self.log_level:
                                self.logger.info(f"Plate {normalized_plate} make mismatch: expected {expected_make}, got {detected_make}")
                            return f"Prohibited! Make Mismatch"
                    except Exception as e:
                        print(f"Error comparing make values: {e}")
                        if self.log_level:
                            self.logger.error(f"Error comparing make values: {e}")
                
                # Verify color if needed
                if ext_info.get('color') and vehicle_color and 'color' in vehicle_color:
                    try:
                        expected_color = str(ext_info['color']).lower()
                        detected_color = str(vehicle_color['color']).lower()
                        
                        if expected_color != detected_color:
                            if self.log_level:
                                self.logger.info(f"Plate {normalized_plate} color mismatch: expected {expected_color}, got {detected_color}")
                            return f"Prohibited! Color Mismatch"
                    except Exception as e:
                        print(f"Error comparing color values: {e}")
                        if self.log_level:
                            self.logger.error(f"Error comparing color values: {e}")
                
                # If we get here, everything matched
                if self.log_level:
                    self.logger.info(f"Plate {normalized_plate} is in the allowed list and make/color match")
                return "Allowed"
            
            # Plate not in allowed list
            if self.log_level:
                self.logger.info(f"Plate {normalized_plate} is NOT in the allowed list")
            return "Prohibited!"
        except Exception as e:
            print(f"Error checking plate: {e}")
            if self.log_level:
                self.logger.error(f"Error checking plate {plate_number}: {e}")
            return "Unknown"
        
    def add_plate(self, plate_number, make=None, color=None, require_match=False):
        """
        Add a plate to the allowed list and save to file.
        
        Args:
            plate_number: The license plate number to add
            make: Optional vehicle make
            color: Optional vehicle color
            require_match: Whether to require make/color matching
        
        Returns:
            True if added successfully, False otherwise
        """
        if not plate_number:
            return False
            
        try:
            # Normalize plate number
            normalized_plate = str(plate_number).upper().replace(' ', '')
            
            # Add to allowed list if not already there
            if normalized_plate not in self.allow_list:
                self.allow_list.append(normalized_plate)
                
                # Save updated list to file
                try:
                    with open(self.allow_list_file, 'w') as f:
                        for plate in self.allow_list:
                            f.write(f"{plate}\n")
                except Exception as e:
                    print(f"Error saving allowed plates file: {e}")
                    if self.log_level:
                        self.logger.error(f"Error saving allowed plates file: {e}")
                    return False
                        
                if self.log_level:
                    self.logger.info(f"Added plate {normalized_plate} to allowed list")
                    
                # If make or color is provided, add to extended info
                if make or color:
                    self.extended_info[normalized_plate] = {
                        'make': make,
                        'color': color,
                        'require_match': bool(require_match)
                    }
                    
                    # Save extended info to JSON file
                    self.save_extended_info()
                    
                return True
            # Plate already in allowed list, but we might want to update extended info
            elif make or color:
                self.extended_info[normalized_plate] = {
                    'make': make,
                    'color': color,
                    'require_match': bool(require_match)
                }
                
                # Save extended info to JSON file
                self.save_extended_info()
                
                if self.log_level:
                    self.logger.info(f"Updated extended info for plate {normalized_plate}")
                    
                return True
                
            return False
        except Exception as e:
            print(f"Error adding plate: {e}")
            if self.log_level:
                self.logger.error(f"Error adding plate {plate_number}: {e}")
            return False
        
    def remove_plate(self, plate_number):
        """Remove a plate from the allowed list and save to file"""
        if not plate_number:
            return False
            
        try:
            # Normalize plate number
            normalized_plate = str(plate_number).upper().replace(' ', '')
            
            if normalized_plate in self.allow_list:
                self.allow_list.remove(normalized_plate)
                
                # Save updated list to file
                try:
                    with open(self.allow_list_file, 'w') as f:
                        for plate in self.allow_list:
                            f.write(f"{plate}\n")
                except Exception as e:
                    print(f"Error saving allowed plates file: {e}")
                    if self.log_level:
                        self.logger.error(f"Error saving allowed plates file: {e}")
                    return False
                
                # Remove from extended info if it exists
                if normalized_plate in self.extended_info:
                    del self.extended_info[normalized_plate]
                    self.save_extended_info()
                        
                if self.log_level:
                    self.logger.info(f"Removed plate {normalized_plate} from allowed list")
                return True
            return False
        except Exception as e:
            print(f"Error removing plate: {e}")
            if self.log_level:
                self.logger.error(f"Error removing plate {plate_number}: {e}")
            return False
            
    def save_extended_info(self):
        """Save extended plate information to JSON file"""
        try:
            if hasattr(self, 'extended_info') and self.extended_info:
                import json
                
                # Convert to list format
                data = []
                for plate, info in self.extended_info.items():
                    entry = {'plate': plate}
                    entry.update(info)
                    data.append(entry)
                
                try:
                    with open(self.json_file, 'w') as f:
                        json.dump(data, f, indent=2)
                        
                    print(f"Saved {len(data)} plates with extended information to {self.json_file}")
                    if self.log_level:
                        self.logger.info(f"Saved {len(data)} plates with extended information to {self.json_file}")
                except Exception as e:
                    print(f"Error writing to {self.json_file}: {e}")
                    if self.log_level:
                        self.logger.error(f"Error writing to {self.json_file}: {e}")
        except Exception as e:
            print(f"Error saving extended plate information: {e}")
            if self.log_level:
                self.logger.error(f"Error saving extended plate information: {e}")
            
    def set_verify_make_color(self, verify):
        """Enable or disable make/color verification"""
        try:
            self.verify_make_color = bool(verify)
            print(f"Make/color verification {'enabled' if verify else 'disabled'}")
            if self.log_level:
                self.logger.info(f"Make/color verification {'enabled' if verify else 'disabled'}")
                
            # If enabling but no extended info loaded, warn
            if verify and (not hasattr(self, 'extended_info') or not self.extended_info):
                print("Warning: No extended plate information available. Verification will have no effect.")
                if self.log_level:
                    self.logger.warning("No extended plate information available. Verification will have no effect.")
        except Exception as e:
            print(f"Error setting verify_make_color: {e}")
            if self.log_level:
                self.logger.error(f"Error setting verify_make_color: {e}")
                
    def get_extended_info(self, plate_number):
        """Get extended info for a specific plate"""
        if not plate_number:
            return None
            
        try:
            normalized_plate = str(plate_number).upper().replace(' ', '')
            
            if normalized_plate in self.extended_info:
                return self.extended_info[normalized_plate]
                
            return None
        except Exception as e:
            print(f"Error getting extended info: {e}")
            if self.log_level:
                self.logger.error(f"Error getting extended info: {e}")
            return None

# Relay Controller Class
class RelayController:
    def __init__(self, gpio_pin=17, active_time=1.0, log_level='INFO', log_dir='./logs/'):
        self.gpio_pin = gpio_pin
        self.active_time = active_time  # Time in seconds to keep relay active
        self.log_level = log_level
        self.log_dir = log_dir
        self.is_active = False
        
        # Set up logging
        if self.log_level:
            self.num_log_level = getattr(logging, self.log_level.upper(), 20)
            os.makedirs(os.path.dirname(self.log_dir), exist_ok=True)
            log_formatter = logging.Formatter("%(asctime)s %(message)s")
            logFile = self.log_dir + 'relay.log'
            my_handler = RotatingFileHandler(logFile, mode='a', maxBytes=25 * 1024 * 1024,
                                          backupCount=10, encoding='utf-8', delay=False)
            my_handler.setFormatter(log_formatter)
            my_handler.setLevel(self.num_log_level)
            self.logger = logging.getLogger("relay_logger")  
            self.logger.setLevel(self.num_log_level)
            self.logger.addHandler(my_handler)
        
        # Try to use GPIO on the Orange Pi
        try:
            import OPi.GPIO as GPIO
            self.GPIO = GPIO
            self.GPIO.setmode(GPIO.BOARD)
            self.GPIO.setup(self.gpio_pin, GPIO.OUT)
            self.GPIO.output(self.gpio_pin, GPIO.LOW)  # Initialize to LOW (relay off)
            self.gpio_available = True
            print(f"Relay initialized on GPIO pin {self.gpio_pin}")
            if self.log_level:
                self.logger.info(f"Relay initialized on GPIO pin {self.gpio_pin}")
        except ImportError:
            print("OPi.GPIO not available, trying standard RPi.GPIO...")
            try:
                import RPi.GPIO as GPIO
                self.GPIO = GPIO
                self.GPIO.setmode(GPIO.BOARD)
                self.GPIO.setup(self.gpio_pin, GPIO.OUT)
                self.GPIO.output(self.gpio_pin, GPIO.LOW)  # Initialize to LOW (relay off)
                self.gpio_available = True
                print(f"Relay initialized on GPIO pin {self.gpio_pin} (using RPi.GPIO)")
                if self.log_level:
                    self.logger.info(f"Relay initialized on GPIO pin {self.gpio_pin} (using RPi.GPIO)")
            except Exception as e:
                print(f"GPIO not available. Relay will be simulated: {e}")
                self.gpio_available = False
                if self.log_level:
                    self.logger.warning(f"GPIO not available. Relay will be simulated: {e}")
    
    def activate(self):
        """Activate the relay and schedule deactivation after active_time seconds"""
        if self.is_active:
            return  # Already active, don't restart the timer
            
        self.is_active = True
        
        # Activate relay
        if self.gpio_available:
            try:
                self.GPIO.output(self.gpio_pin, self.GPIO.HIGH)
                print(f"Relay activated on pin {self.gpio_pin}")
                if self.log_level:
                    self.logger.info(f"Relay activated on pin {self.gpio_pin}")
            except Exception as e:
                print(f"Error activating relay: {e}")
                if self.log_level:
                    self.logger.error(f"Error activating relay: {e}")
        else:
            print(f"[SIMULATION] Relay activated on pin {self.gpio_pin}")
        
        # Schedule deactivation
        threading.Timer(self.active_time, self.deactivate).start()
    
    def deactivate(self):
        """Deactivate the relay"""
        self.is_active = False
        
        if self.gpio_available:
            try:
                self.GPIO.output(self.gpio_pin, self.GPIO.LOW)
                print(f"Relay deactivated on pin {self.gpio_pin}")
                if self.log_level:
                    self.logger.info(f"Relay deactivated on pin {self.gpio_pin}")
            except Exception as e:
                print(f"Error deactivating relay: {e}")
                if self.log_level:
                    self.logger.error(f"Error deactivating relay: {e}")
        else:
            print(f"[SIMULATION] Relay deactivated on pin {self.gpio_pin}")
    
    def cleanup(self):
        """Clean up GPIO resources"""
        if self.gpio_available:
            try:
                # Ensure relay is off
                self.GPIO.output(self.gpio_pin, self.GPIO.LOW)
                # Clean up GPIO settings
                self.GPIO.cleanup(self.gpio_pin)
                print("GPIO cleanup completed")
                if self.log_level:
                    self.logger.info("GPIO cleanup completed")
            except Exception as e:
                print(f"Error during GPIO cleanup: {e}")
                if self.log_level:
                    self.logger.error(f"Error during GPIO cleanup: {e}")

# Enhanced visualization class
class EnhancedVisualize:
    def __init__(self, im0, file_name, cropped_img=None, bbox=None, det_conf=None, ocr_num=None, ocr_conf=None, 
                 num_check_response=None, vehicle_make=None, vehicle_color=None,
                 out_img_size=(720, 1280), outp_orig_img_size=640, log_dir='./logs/', 
                 save_jpg_qual=65, log_img_qnt_limit=10800):
        self.im0 = im0.copy() if im0 is not None else np.zeros((480, 640, 3), dtype=np.uint8)
        self.input_img = im0.copy() if im0 is not None else None
        self.file_name = file_name
        self.cropped_img = cropped_img
        self.bbox = bbox
        self.det_conf = det_conf
        self.ocr_num = ocr_num
        self.ocr_conf = ocr_conf
        self.num_check_response = num_check_response
        self.vehicle_make = vehicle_make
        self.vehicle_color = vehicle_color
        self.out_img_size = out_img_size
        self.save_jpg_qual = save_jpg_qual
        self.log_dir = log_dir
        
        # Create log directories
        self.imgs_log_dir = self.log_dir + 'imgs/'
        os.makedirs(os.path.dirname(self.imgs_log_dir), exist_ok=True)
        self.crop_imgs_log_dir = self.log_dir + 'imgs/crop/'
        os.makedirs(os.path.dirname(self.crop_imgs_log_dir), exist_ok=True)
        self.orig_imgs_log_dir = self.log_dir + 'imgs/inp/'
        os.makedirs(os.path.dirname(self.orig_imgs_log_dir), exist_ok=True)
        self.log_img_qnt_limit = log_img_qnt_limit
        
        # Create a copy of the original image for visualization
        self.display_img = self.im0.copy()
        
        # Draw bounding box on the image if present
        if (self.bbox is not None) and (self.det_conf is not None):
            # Convert bbox tensor to coordinates
            try:
                if isinstance(self.bbox, torch.Tensor):
                    bbox_coords = self.bbox.tolist()
                else:
                    bbox_coords = self.bbox
                    
                x1, y1, x2, y2 = map(int, bbox_coords)
                
                # Format confidence score appropriately
                if isinstance(self.det_conf, torch.Tensor) and hasattr(self.det_conf, 'item'):
                    conf_value = self.det_conf.item()
                else:
                    conf_value = float(self.det_conf)
                    
                label = f'Plate {conf_value:.2f}'
                color = (0, 255, 0)  # Green
                thickness = 2
                
                # Draw rectangle
                cv2.rectangle(self.display_img, (x1, y1), (x2, y2), color, thickness)
                
                # Add label on top of the bounding box
                label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(self.display_img, (x1, y1 - label_size[1] - 5), (x1 + label_size[0], y1), color, -1)
                cv2.putText(self.display_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            except Exception as e:
                print(f"Error drawing bounding box: {e}")
        
        # Add OCR results at the bottom of the image if available
        if self.ocr_num is not None:
            try:
                # OCR result text
                if self.ocr_conf is not None:
                    if isinstance(self.ocr_conf, float):
                        ocr_text = f"Plate: {self.ocr_num} (Conf: {self.ocr_conf:.2f})"
                    else:
                        ocr_text = f"Plate: {self.ocr_num} (Conf: {self.ocr_conf})"
                else:
                    ocr_text = f"Plate: {self.ocr_num}"
                
                # Status text and color
                if self.num_check_response == "Allowed":
                    status_text = "Status: ALLOWED"
                    status_color = (0, 255, 0)  # Green
                else:
                    status_text = "Status: PROHIBITED"
                    status_color = (0, 0, 255)  # Red
                
                # Add background for text visibility
                h, w = self.display_img.shape[:2]
                cv2.rectangle(self.display_img, (0, h - 60), (w, h), (0, 0, 0), -1)
                
                # Add OCR result
                cv2.putText(self.display_img, ocr_text, (10, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Add status
                cv2.putText(self.display_img, status_text, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
                
                # Add vehicle make and color if available
                if self.vehicle_make or self.vehicle_color:
                    extra_info_text = "Vehicle: "
                    
                    if self.vehicle_make:
                        extra_info_text += f"{self.vehicle_make['make']} "
                        
                    if self.vehicle_color:
                        extra_info_text += f"({self.vehicle_color['color']})"
                        
                    # Add text above the OCR result
                    cv2.putText(self.display_img, extra_info_text, (10, h - 60), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            except Exception as e:
                print(f"Error adding OCR results to display: {e}")
            
    def show(self):
        # Show the image in a window
        cv2.imshow('ANPR Detection', self.display_img)
        
    def show_details(self):
        try:
            # Create a dedicated window for detailed results
            window_name = "ANPR Details"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            
            # Create an image with plate details
            h, w = (600, 800)  # Larger size for a details window with more information
            details_img = np.zeros((h, w, 3), np.uint8)
            details_img[:, :] = (255, 255, 255)  # White background
            
            # Add title
            title = "License Plate Details"
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(details_img, title, (20, 30), font, 1.2, (0, 0, 0), 2, cv2.LINE_AA)
            
            # Horizontal separator
            cv2.line(details_img, (20, 40), (w-20, 40), (0, 0, 0), 1)
            
            y_pos = 80  # Starting y position for content
            
            # Add cropped plate image if available
            if self.cropped_img is not None and self.cropped_img.size > 0:
                # Calculate size to maintain aspect ratio
                height, width = self.cropped_img.shape[:2]
                max_width = w - 40
                ratio = min(max_width / width, 150 / height)
                new_width = int(width * ratio)
                new_height = int(height * ratio)
                
                # Resize and embed the cropped image
                plate_img = cv2.resize(self.cropped_img, (new_width, new_height), interpolation=cv2.INTER_AREA)
                
                # Convert RGB to BGR for display
                if len(plate_img.shape) == 3 and plate_img.shape[2] == 3:
                    plate_img = cv2.cvtColor(plate_img, cv2.COLOR_RGB2BGR)
                
                x_offset = (w - new_width) // 2  # Center horizontally
                
                # Safely copy the plate image to the details image
                details_img[y_pos:y_pos+new_height, x_offset:x_offset+new_width] = plate_img
                
                # Draw border around the plate image
                cv2.rectangle(details_img, (x_offset-1, y_pos-1), 
                             (x_offset+new_width+1, y_pos+new_height+1), (0, 0, 0), 1)
                
                y_pos += new_height + 30
            
            # Add OCR results
            if self.ocr_num is not None:
                plate_text = f"Plate Number: {self.ocr_num}"
                
                if self.ocr_conf is not None:
                    if isinstance(self.ocr_conf, float):
                        conf_text = f"OCR Confidence: {self.ocr_conf:.2f}"
                    else:
                        conf_text = f"OCR Confidence: {self.ocr_conf}"
                else:
                    conf_text = "OCR Confidence: N/A"
                
                cv2.putText(details_img, plate_text, (20, y_pos), font, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
                y_pos += 40
                
                # Add vehicle information if available
                if self.vehicle_make:
                    make_text = f"Vehicle Make: {self.vehicle_make['make']}"
                    conf_text = f"Make Confidence: {self.vehicle_make['confidence']:.2f}"
                    cv2.putText(details_img, make_text, (20, y_pos), font, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
                    y_pos += 40
            else:
                cv2.putText(details_img, "No plate text detected", (20, y_pos), font, 0.7, (0, 0, 150), 2, cv2.LINE_AA)
                y_pos += 40
            
            # Add status information
            if self.num_check_response is not None:
                status_text = f"Status: {self.num_check_response}"
                status_color = (0, 150, 0) if self.num_check_response == "Allowed" else (0, 0, 150)
                cv2.putText(details_img, status_text, (20, y_pos), font, 0.8, status_color, 2, cv2.LINE_AA)
                y_pos += 40
            
            # Add timestamp
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(details_img, f"Time: {timestamp}", (20, y_pos), font, 0.6, (100, 100, 100), 1, cv2.LINE_AA)
            
            # Show the details window
            cv2.imshow(window_name, details_img)
            
            return details_img  # Return the image in case it needs to be saved
        except Exception as e:
            print(f"Error showing details: {e}")
            return None
        
    def save(self):
        try:
            # Get all files in the directory sorted by modification time (oldest first)
            if os.path.exists(self.imgs_log_dir):
                files = [os.path.join(self.imgs_log_dir, f) for f in os.listdir(self.imgs_log_dir) 
                         if os.path.isfile(os.path.join(self.imgs_log_dir, f))]
                files.sort(key=lambda x: os.path.getmtime(x))
                
                # Remove oldest files if over the limit
                while len(files) >= self.log_img_qnt_limit and files:
                    oldest_file = files.pop(0)  # Get the oldest file
                    try:
                        os.remove(oldest_file)
                        print(f"Removed old file: {oldest_file}")
                    except Exception as e:
                        print(f"Error removing file {oldest_file}: {e}")
            
            # Write compressed jpeg with results
            cv2.imwrite(f"{self.imgs_log_dir}{self.file_name}", self.display_img, 
                       [int(cv2.IMWRITE_JPEG_QUALITY), self.save_jpg_qual])
            print(f"Saved detection image to {self.imgs_log_dir}{self.file_name}")
        except Exception as e:
            print(f"Error saving detection image: {e}")
        
    def save_input(self):
        try:
            if self.input_img is not None:
                # Get all files in the directory sorted by modification time (oldest first)
                if os.path.exists(self.orig_imgs_log_dir):
                    files = [os.path.join(self.orig_imgs_log_dir, f) for f in os.listdir(self.orig_imgs_log_dir) 
                             if os.path.isfile(os.path.join(self.orig_imgs_log_dir, f))]
                    files.sort(key=lambda x: os.path.getmtime(x))
                    
                    # Remove oldest files if over the limit
                    while len(files) >= self.log_img_qnt_limit and files:
                        oldest_file = files.pop(0)  # Get the oldest file
                        try:
                            os.remove(oldest_file)
                        except Exception as e:
                            print(f"Error removing file {oldest_file}: {e}")
                
                # Write compressed jpeg with results
                cv2.imwrite(f"{self.orig_imgs_log_dir}orig_inp_{self.file_name}", self.input_img)
        except Exception as e:
            print(f"Error saving input image: {e}")
            
    def save_crop(self):
        try:
            if self.cropped_img is not None and self.cropped_img.size > 0:
                # Get all files in the directory sorted by modification time (oldest first)
                if os.path.exists(self.crop_imgs_log_dir):
                    files = [os.path.join(self.crop_imgs_log_dir, f) for f in os.listdir(self.crop_imgs_log_dir) 
                             if os.path.isfile(os.path.join(self.crop_imgs_log_dir, f))]
                    files.sort(key=lambda x: os.path.getmtime(x))
                    
                    # Remove oldest files if over the limit
                    while len(files) >= self.log_img_qnt_limit and files:
                        oldest_file = files.pop(0)  # Get the oldest file
                        try:
                            os.remove(oldest_file)
                        except Exception as e:
                            print(f"Error removing file {oldest_file}: {e}")
                
                # Convert RGB to BGR for saving
                save_img = cv2.cvtColor(self.cropped_img, cv2.COLOR_RGB2BGR)
                        
                # Write compressed jpeg with results
                cv2.imwrite(f"{self.crop_imgs_log_dir}crop_{self.file_name}", save_img)
        except Exception as e:
            print(f"Error saving cropped image: {e}")


# Function to check and install required packages
def check_and_install_requirements():
    print("Checking and installing required packages...")
    
    required_packages = {
        'opencv-python': 'cv2',
        'torch': 'torch',
        'numpy': 'numpy',
        'pillow': 'PIL',
        'easyocr': 'easyocr',
        'scipy': 'scipy',
        'ultralytics': 'ultralytics',
        'scikit-learn': 'sklearn'
    }
    
    # Orange Pi specific packages
    try:
        import OPi.GPIO
        print("OPi.GPIO is already installed")
    except ImportError:
        print("Installing OPi.GPIO...")
        os.system("pip install OPi.GPIO")
    
    # Try to install TensorRT if not already available
    if not TENSORRT_AVAILABLE:
        try:
            print("Trying to install TensorRT...")
            os.system("pip install tensorrt")
            os.system("pip install pycuda")
        except:
            print("TensorRT installation failed. This may require manual installation.")
    
    # Try to install RKNN toolkit if not already available
    if not RKNN_AVAILABLE:
        try:
            print("Trying to install RKNN Toolkit Lite...")
            os.system("pip install rknn-toolkit-lite")
        except:
            print("RKNN Toolkit installation failed. This may require manual installation.")
    
    # Check and install remaining packages
    missing_packages = []
    for package, module in required_packages.items():
        try:
            __import__(module)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Installing missing packages: {', '.join(missing_packages)}")
        for package in missing_packages:
            os.system(f"pip install {package}")
        print("Required packages installed.")
    else:
        print("All required packages are already installed.")

# Function to prepare models
def prepare_models(models_dir='./models', use_tensorrt=False, use_rknn=False):
    """Prepare and return paths to the necessary models"""
    os.makedirs(models_dir, exist_ok=True)
    
    model_paths = {
        'detector': None,
        'vehicle_make': None,
        'color_map': None
    }
    
    # Prepare detector model
    if use_rknn and RKNN_AVAILABLE:
        rknn_model_path = os.path.join(models_dir, 'anpr_rknn.rknn')
        if os.path.exists(rknn_model_path):
            model_paths['detector'] = rknn_model_path
            print(f"Using RKNN model: {rknn_model_path}")
        else:
            print("RKNN model not found. You'll need to convert a YOLO model to RKNN format.")
            print("Falling back to standard YOLO model.")
            
    if use_tensorrt and TENSORRT_AVAILABLE and model_paths['detector'] is None:
        tensorrt_model_path = os.path.join(models_dir, 'anpr_tensorrt.engine')
        if os.path.exists(tensorrt_model_path):
            model_paths['detector'] = tensorrt_model_path
            print(f"Using TensorRT model: {tensorrt_model_path}")
        else:
            print("TensorRT engine not found. You'll need to convert a YOLO model to TensorRT format.")
            print("Falling back to standard YOLO model.")
    
    # Default to YOLO model if no hardware-specific model is available
    if model_paths['detector'] is None:
        # Look for a custom license plate detection model
        custom_model_path = os.path.join(models_dir, 'yolo11_license_plates.pt')
        if os.path.exists(custom_model_path):
            model_paths['detector'] = custom_model_path
            print(f"Using custom license plate model: {custom_model_path}")
        else:
            # Otherwise use YOLO11s
            yolo11s_path = os.path.join(models_dir, 'yolo11s.pt')
            if not os.path.exists(yolo11s_path):
                print("\nDownloading YOLO11s model...")
                try:
                    from ultralytics import YOLO
                    model = YOLO("yolo11s.pt")
                    model.model.save(yolo11s_path)
                    print(f"YOLO11s model saved to {yolo11s_path}")
                except Exception as e:
                    print(f"Error downloading YOLO11s model: {e}")
                    print("Falling back to YOLOv8n...")
                    yolo11s_path = os.path.join(models_dir, 'yolov8n.pt')
                    if not os.path.exists(yolo11s_path):
                        model = YOLO("yolov8n.pt")
                        model.model.save(yolo11s_path)
            
            model_paths['detector'] = yolo11s_path
            print(f"Using YOLO model: {yolo11s_path}")
    
    # Check for vehicle make model
    make_model_path = os.path.join(models_dir, 'vehicle_make_resnet18.pth')
    make_labels_path = os.path.join(models_dir, 'vehicle_make_labels.txt')
    
    if os.path.exists(make_model_path) and os.path.exists(make_labels_path):
        model_paths['vehicle_make'] = make_model_path
        print(f"Vehicle make classification model found: {make_model_path}")
    else:
        print("Vehicle make classification model not found. This feature will be disabled.")
        print(f"To enable, place a ResNet18 model at {make_model_path} and labels at {make_labels_path}")
    
    # Check for color map
    color_map_path = os.path.join(models_dir, 'color_map.json')
    if os.path.exists(color_map_path):
        model_paths['color_map'] = color_map_path
        print(f"Vehicle color map found: {color_map_path}")
    else:
        print("Vehicle color map not found. Will use default color mapping.")
    
    return model_paths

# Function to create the appropriate detector based on hardware capabilities
def create_detector(model_path, hw_capabilities, img_size=640, log_level='INFO', log_dir='./logs/'):
    """Create and return an appropriate detector based on hardware capabilities"""
    detector_type = hw_capabilities['best_option']
    
    if detector_type == 'rknn_npu' and RKNN_AVAILABLE and model_path.endswith('.rknn'):
        print("Creating RKNN NPU-based detector")
        return RKNNDetector(
            model_path=model_path,
            img_size=img_size,
            log_level=log_level,
            log_dir=log_dir
        )
    elif detector_type == 'tensorrt' and TENSORRT_AVAILABLE and model_path.endswith('.engine'):
        print("Creating TensorRT-accelerated detector")
        return TensorRTDetector(
            model_path=model_path,
            img_size=img_size,
            log_level=log_level,
            log_dir=log_dir
        )
    else:
        # Use CUDA if available, otherwise CPU
        device = 'cuda:0' if hw_capabilities['cuda'] else 'cpu'
        print(f"Creating YOLO detector with device: {device}")
        return YOLODetector(
            model_weights=model_path,
            img_size=img_size,
            device=device,
            log_level=log_level,
            log_dir=log_dir
        )

# Enhanced continuous monitoring mode with parallel processing
def enhanced_continuous_mode(camera, detector, ocr, checker, relay_controller, 
                             vehicle_make_classifier, vehicle_color_detector,
                             log_dir, hw_capabilities):
    """
    Enhanced continuous monitoring mode with parallel processing and UI controls.
    
    This mode:
    1. Captures frames continuously from the camera
    2. Processes frames in parallel to detect and recognize license plates
    3. Provides UI controls for adjusting settings
    4. Displays results and allows user interaction
    
    Args:
        camera: Camera object for capturing frames
        detector: Object detection model for license plates
        ocr: OCR model for text recognition
        checker: Plate checker for validating against allowed list
        relay_controller: Controller for relay activation
        vehicle_make_classifier: Classifier for vehicle make detection (optional)
        vehicle_color_detector: Detector for vehicle color (optional)
        log_dir: Directory for logging
        hw_capabilities: Dict with hardware acceleration capabilities
    """
    print("\nEnhanced continuous monitoring mode started. Press 'q' to exit.")
    print("Additional commands: 'a' to add a plate to allowed list, 'r' to remove a plate")
    
    # Create the main window with a trackbar for confidence threshold
    window_name = 'ANPR Continuous Monitor'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 600)
    
    # Create control panel window
    control_name = 'ANPR Controls'
    cv2.namedWindow(control_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(control_name, 400, 300)
    
    # Create trackbars for settings
    conf_threshold = 25  # default 0.25 (scaled to 0-100 for trackbar)
    cv2.createTrackbar('Confidence %', control_name, conf_threshold, 100, lambda x: None)
    
    # Processing frequency (frames to skip)
    skip_frames = 5  # Process every 5th frame by default
    cv2.createTrackbar('Process every N frames', control_name, skip_frames, 30, lambda x: None)
    
    # Show details option
    show_details = 1  # 1 = On, 0 = Off
    cv2.createTrackbar('Show Details', control_name, show_details, 1, lambda x: None)
    
    # Save detected plates option
    save_detections = 1  # 1 = On, 0 = Off
    cv2.createTrackbar('Save Detections', control_name, save_detections, 1, lambda x: None)
    
    # Enable vehicle make detection
    vehicle_make_enabled = 1 if vehicle_make_classifier and vehicle_make_classifier.enabled else 0
    cv2.createTrackbar('Vehicle Make', control_name, vehicle_make_enabled, 1, 
                      lambda x: vehicle_make_classifier.set_enabled(x == 1) if vehicle_make_classifier else None)
    
    # Enable vehicle color detection
    vehicle_color_enabled = 1 if vehicle_color_detector and vehicle_color_detector.enabled else 0
    cv2.createTrackbar('Vehicle Color', control_name, vehicle_color_enabled, 1,
                      lambda x: vehicle_color_detector.set_enabled(x == 1) if vehicle_color_detector else None)
                      
    # Enable make/color verification
    verify_make_color = 0  # Off by default
    cv2.createTrackbar('Verify Make/Color', control_name, verify_make_color, 1,
                      lambda x: checker.set_verify_make_color(x == 1))
    
    # Create a frame counter and processing queue
    frame_count = 0
    last_plate = None
    last_vis = None
    processing_times = deque(maxlen=10)  # Store recent processing times
    
    # Create a thread pool for parallel processing
    executor = None
    if hw_capabilities['cuda']:
        # When using GPU, we need fewer threads as the GPU is the bottleneck
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
    else:
        # More threads when using CPU to parallelize different stages
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
    
    # Use a queue for pending results, with a limited size to prevent memory issues
    results_queue = queue.Queue(maxsize=2)
    
    def process_frame(frame, conf_thres, enable_make, enable_color):
        """Process a single frame and return results (runs in a separate thread)"""
        start_time = time.time()
        
        try:
            # Skip processing if the frame is empty or invalid
            if frame is None or frame.size == 0:
                print("Warning: Empty or invalid frame received, skipping processing")
                return None
                
            # Run detection
            det_result = detector.run(frame, conf_thres=conf_thres)
            
            # Only continue with OCR if a plate was detected
            if det_result['bbox'] is not None:
                # Run OCR on detected plate
                ocr_result = ocr.run(det_result)
                
                # Get plate number and check if allowed
                plate_number = ocr_result.get('text')
                
                # Only process if a plate number was detected
                if plate_number:
                    check_result = checker.check(plate_number)
                    
                    # Get vehicle make if enabled
                    make_result = None
                    if enable_make and vehicle_make_classifier and vehicle_make_classifier.enabled:
                        try:
                            make_result = vehicle_make_classifier.predict(frame, det_result['bbox'])
                        except Exception as e:
                            print(f"Error in vehicle make classification: {e}")
                    
                    # Get vehicle color if enabled
                    color_result = None
                    if enable_color and vehicle_color_detector and vehicle_color_detector.enabled:
                        try:
                            color_result = vehicle_color_detector.detect_color(frame, det_result['bbox'])
                        except Exception as e:
                            print(f"Error in vehicle color detection: {e}")
                    
                    # Create filename with timestamp and plate number
                    # Sanitize filename to remove invalid characters
                    safe_plate = ''.join(c for c in plate_number if c.isalnum())
                    filename = f"plate_{safe_plate}_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
                    
                    # Visualize results
                    vis = EnhancedVisualize(
                        im0=det_result['orig_img'],
                        file_name=filename,
                        cropped_img=det_result['cropped_img'],
                        bbox=det_result['bbox'],
                        det_conf=det_result['det_conf'],
                        ocr_num=plate_number,
                        ocr_conf=ocr_result.get('confid'),
                        num_check_response=check_result,
                        vehicle_make=make_result,
                        vehicle_color=color_result,
                        log_dir=log_dir
                    )
                    
                    processing_time = time.time() - start_time
                    
                    return {
                        'vis': vis,
                        'plate_number': plate_number,
                        'check_result': check_result,
                        'make_result': make_result,
                        'color_result': color_result,
                        'processing_time': processing_time
                    }
            
            # No valid results
            return None
        except Exception as e:
            print(f"Error in frame processing: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    try:
        # Handle keyboard interactions 
        key = cv2.waitKey(1) & 0xFF
        
        # Exit on 'q' key
        if key == ord('q'):
            break
        elif key == ord('a') and last_plate:  # Add plate to allowed list
            # Get current settings
            make_info = None
            if last_result and 'make_result' in last_result and last_result['make_result']:
                make_info = last_result['make_result']['make']
                
            color_info = None
            if last_result and 'color_result' in last_result and last_result['color_result']:
                color_info = last_result['color_result']['color']
                
            # Get current verification setting
            verify_needed = cv2.getTrackbarPos('Verify Make/Color', control_name) == 1
                
            # Add plate with extended info if available
            if checker.add_plate(last_plate, make=make_info, color=color_info, require_match=verify_needed):
                print(f"Added plate '{last_plate}' to allowed list", end="")
                if make_info or color_info:
                    print(f" with {make_info or 'unknown make'}/{color_info or 'unknown color'} ({verify_needed and 'verification required' or 'verification optional'})")
                else:
                    print()
            else:
                print(f"Plate '{last_plate}' already in allowed list")
                # Update extended info if available
                if make_info or color_info:
                    checker.add_plate(last_plate, make=make_info, color=color_info, require_match=verify_needed)
                    print(f"Updated plate '{last_plate}' with {make_info or 'unknown make'}/{color_info or 'unknown color'} ({verify_needed and 'verification required' or 'verification optional'})")
                    
        elif key == ord('r') and last_plate:  # Remove plate from allowed list
            if checker.remove_plate(last_plate):
                print(f"Removed plate '{last_plate}' from allowed list")
            else:
                print(f"Plate '{last_plate}' not in allowed list")
    
    except KeyboardInterrupt:
        print("Continuous mode stopped by user")
        
        while True:
            try:
                start_time = time.time()
                
                # Capture frame
                frame = camera.run()
                
                # Get current settings from trackbars
                conf_threshold = cv2.getTrackbarPos('Confidence %', control_name)
                conf_threshold = max(1, conf_threshold) / 100  # Convert to 0.01-1.00
                
                skip_frames = max(1, cv2.getTrackbarPos('Process every N frames', control_name))
                show_details = cv2.getTrackbarPos('Show Details', control_name)
                save_detections = cv2.getTrackbarPos('Save Detections', control_name)
                vehicle_make_enabled = cv2.getTrackbarPos('Vehicle Make', control_name) == 1
                vehicle_color_enabled = cv2.getTrackbarPos('Vehicle Color', control_name) == 1
                
                # Check for results from background processing
                try:
                    if not results_queue.empty():
                        processing_active = False
                        result = results_queue.get_nowait()
                        
                        if result:
                            last_plate = result['plate_number']
                            last_vis = result['vis']
                            last_result = result  # Store the complete result for later use
                            
                            # Add processing time to our tracking
                            if 'processing_time' in result:
                                processing_times.append(result['processing_time'])
                            
                            # Show result in main window
                            cv2.imshow(window_name, last_vis.display_img)
                            
                            # Show detailed results if enabled
                            if show_details:
                                last_vis.show_details()
                            
                            # Save images if enabled
                            if save_detections:
                                last_vis.save()
                                last_vis.save_input()
                                last_vis.save_crop()
                            
                            # Format result information for display
                            make_info = ""
                            if result['make_result']:
                                make_info = f", Make: {result['make_result']['make']} ({result['make_result']['confidence']:.2f})"
                                
                            color_info = ""
                            if result['color_result']:
                                color_info = f", Color: {result['color_result']['color']} ({result['color_result']['confidence']:.2f})"
                            
                            # Calculate average processing time
                            avg_time = sum(processing_times) / max(1, len(processing_times))
                            
                            print(f"Detected plate: {last_plate}, Status: {result['check_result']}{make_info}{color_info}")
                            print(f"Processing time: {result['processing_time']:.3f}s (avg: {avg_time:.3f}s)")
                            
                            # Activate relay if plate is allowed
                            if "Allowed" in result['check_result']:
                                print(f"Allowed plate detected: {last_plate} - Activating relay")
                                relay_controller.activate()
                            else:
                                print(f"Access denied: {result['check_result']}")
                except queue.Empty:
                    pass  # Queue is empty, continue with the loop
                except Exception as e:
                    print(f"Error processing results: {e}")
                    import traceback
                    traceback.print_exc()
                    
                # Process frame if it's time and we're not already processing
                if frame_count % skip_frames == 0 and not processing_active and frame is not None:
                    # Submit frame for processing in background
                    future = executor.submit(
                        process_frame, 
                        frame.copy(), 
                        conf_threshold,
                        vehicle_make_enabled,
                        vehicle_color_enabled
                    )
                    
                    # Add callback to put result in queue when done
                    def on_complete(f):
                        try:
                            result = f.result()
                            if not results_queue.full():
                                results_queue.put(result)
                            else:
                                print("Warning: Results queue is full, dropping result")
                        except Exception as e:
                            print(f"Error in background processing: {e}")
                            import traceback
                            traceback.print_exc()
                    
                    future.add_done_callback(on_complete)
                    processing_active = True
                
                # If no active processing or results, show the original frame
                if (not processing_active or results_queue.empty()) and frame is not None:
                    # Calculate FPS
                    frame_time = time.time() - start_time
                    frame_times.append(frame_time)
                    if time.time() - last_fps_update > 0.5:  # Update FPS every 0.5 seconds
                        fps = len(frame_times) / sum(frame_times)
                        last_fps_update = time.time()
                    
                    # Display the original frame with FPS overlay
                    display_frame = frame.copy()
                    
                    # Add FPS and settings info
                    cv2.putText(display_frame, f"FPS: {fps:.1f}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(display_frame, f"Conf: {conf_threshold:.2f}, Skip: {skip_frames}", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Add last plate info if available
                    if last_plate:
                        h, w = display_frame.shape[:2]
                        cv2.putText(display_frame, f"Last plate: {last_plate}", 
                                   (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    cv2.imshow(window_name, display_frame)
                
                # Increment frame counter
                frame_count += 1
                
                # Check for key presses
                key = cv2.waitKey(1) & 0xFF
                
                # Exit on 'q' key
                if key == ord('q'):
                    break
                elif key == ord('a') and last_plate:  # Add plate to allowed list
                    if checker.add_plate(last_plate):
                        print(f"Added plate '{last_plate}' to allowed list")
                    else:
                        print(f"Plate '{last_plate}' already in allowed list")
                        
                elif key == ord('r') and last_plate:  # Remove plate from allowed list
                    if checker.remove_plate(last_plate):
                        print(f"Removed plate '{last_plate}' from allowed list")
                    else:
                        print(f"Plate '{last_plate}' not in allowed list")
                        
                # Limit loop speed to avoid excessive CPU usage when processing is fast
                time_spent = time.time() - start_time
                if time_spent < 0.01:  # Aim for max 100 fps for UI updates
                    time.sleep(0.01 - time_spent)
                
            except Exception as e:
                print(f"Error in continuous mode: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.1)  # Small delay to prevent rapid error loops
                
    except KeyboardInterrupt:
        print("Continuous mode stopped by user")
    finally:
        # Shutdown the thread pool
        if executor:
            executor.shutdown(wait=False)
        cv2.destroyAllWindows()

# Main function to run the enhanced ANPR system
def main():
    """
    Main function to run the enhanced ANPR system.
    
    This function:
    1. Parses command line arguments
    2. Initializes system components
    3. Sets up camera, detector, OCR, and other modules
    4. Runs the continuous monitoring mode
    """
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Enhanced Automatic Number Plate Recognition System for Orange Pi')
    
    # Camera options
    camera_group = parser.add_argument_group('Camera Options')
    camera_group.add_argument('--ip-camera', type=str, default=None,
                        help='RTSP URL for IP camera (e.g., rtsp://user:pass@192.168.1.108:554/stream)')
    camera_group.add_argument('--gstreamer-pipeline', type=str, default=None,
                        help='Custom GStreamer pipeline string for camera input')
    camera_group.add_argument('--resolution', type=str, default='720p',
                        help='Camera resolution (e.g., 640x480, 480p, 720p, 1080p, 4k)')
    camera_group.add_argument('--fps', type=int, default=30,
                        help='Target frames per second (default: 30)')
    camera_group.add_argument('--use-tcp', action='store_true',
                        help='Use TCP for RTSP streaming (more reliable but higher latency)')
    
    # Detection and recognition options
    detection_group = parser.add_argument_group('Detection and Recognition Options')
    detection_group.add_argument('--conf', type=float, default=0.25,
                        help='Detection confidence threshold (default: 0.25)')
    detection_group.add_argument('--country', type=str, default='generic',
                        help='Country format for license plates (default: generic)')
    
    # Hardware acceleration options
    hw_group = parser.add_argument_group('Hardware Acceleration Options')
    hw_group.add_argument('--enable-tensorrt', action='store_true',
                        help='Enable TensorRT acceleration if available')
    hw_group.add_argument('--enable-npu', action='store_true',
                        help='Enable NPU acceleration if available')
    
    # Additional features
    features_group = parser.add_argument_group('Additional Features')
    features_group.add_argument('--disable-vehicle-make', action='store_true',
                        help='Disable vehicle make detection')
    features_group.add_argument('--disable-vehicle-color', action='store_true',
                        help='Disable vehicle color detection')
    features_group.add_argument('--verify-make-color', action='store_true',
                        help='Enable make/color verification for access control')
    features_group.add_argument('--allowed-plates-json', type=str, default='./allowed_plates.json',
                        help='Path to JSON file with make/color information (default: ./allowed_plates.json)')
    
    # System options
    system_group = parser.add_argument_group('System Options')
    system_group.add_argument('--log-level', type=str, default='INFO',
                        help='Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)')
    system_group.add_argument('--allowed-plates', type=str, default='./allowed_plates.txt',
                        help='Path to allowed plates list file')
    system_group.add_argument('--gpio-pin', type=int, default=17,
                        help='GPIO pin number for relay control (default: 17)')
    system_group.add_argument('--relay-time', type=float, default=1.0,
                        help='Time in seconds to keep relay active (default: 1.0)')
    system_group.add_argument('--models-dir', type=str, default='./models',
                        help='Directory containing model files (default: ./models)')
    system_group.add_argument('--log-dir', type=str, default='./logs/',
                        help='Directory for log files (default: ./logs/)')
    
    args = parser.parse_args()
    
    try:
        # Check and install requirements
        check_and_install_requirements()
        
        # Detect hardware capabilities
        hw_capabilities = detect_hardware_capabilities()
        
        # Create log directory
        log_dir = args.log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Parse resolution
        width, height = parse_resolution(args.resolution)
        
        # Warn about high resolution performance
        warn_if_high_performance_needed(width, height, hw_capabilities)
        
        # Prepare models
        model_paths = prepare_models(
            models_dir=args.models_dir,
            use_tensorrt=args.enable_tensorrt and hw_capabilities['tensorrt'],
            use_rknn=args.enable_npu and hw_capabilities['rknn_npu']
        )
        
        # Initialize camera
        try:
            if args.gstreamer_pipeline and GSTREAMER_AVAILABLE:
                # Use custom GStreamer pipeline
                camera = GStreamerCamera(args.gstreamer_pipeline, img_size=(width, height))
                print(f"Using custom GStreamer pipeline")
            elif args.ip_camera:
                # Use IP camera
                camera = IPCamera(args.ip_camera, img_size=(width, height), fps=args.fps, use_tcp=args.use_tcp)
                print(f"Using IP camera: {args.ip_camera}")
            else:
                print("No camera source specified. Please provide one of the following options:")
                print("  --ip-camera RTSP_URL    : Use an IP camera with RTSP stream")
                print("  --gstreamer-pipeline STR: Use a custom GStreamer pipeline")
                print("\nExample:")
                print("  python enhanced-anpr-system.py --ip-camera rtsp://user:pass@192.168.1.100:554/stream")
                return
            
            # Test camera with multiple attempts
            print("Testing camera connection...")
            max_attempts = 3
            for attempt in range(max_attempts):
                test_frame = camera.run()
                if test_frame is not None and test_frame.size > 0:
                    print(f"Successfully connected to camera ({test_frame.shape[1]}x{test_frame.shape[0]})")
                    break
                    
                print(f"Failed to get a valid frame (attempt {attempt+1}/{max_attempts})")
                time.sleep(1)
            else:  # This runs if the for loop completes without a break
                raise RuntimeError(f"Failed to get a valid frame after {max_attempts} attempts")
                
        except Exception as e:
            print(f"Error initializing camera: {e}")
            print("Please check your camera connection and try again.")
            return
        
        # Initialize detector with appropriate hardware acceleration
        detector = create_detector(
            model_path=model_paths['detector'],
            hw_capabilities=hw_capabilities,
            log_level=args.log_level,
            log_dir=log_dir
        )
        
        # Initialize enhanced OCR
        ocr = EnhancedEasyOcr(lang=['en'], log_level=args.log_level, log_dir=log_dir)
        
        # Initialize plate checker with make/color verification support
        checker = PlateChecker(
            allow_list_file=args.allowed_plates, 
            json_file=args.allowed_plates_json,
            verify_make_color=args.verify_make_color,
            log_level=args.log_level, 
            log_dir=log_dir
        )
        
        # Initialize relay controller
        relay_controller = RelayController(
            gpio_pin=args.gpio_pin, 
            active_time=args.relay_time,
            log_level=args.log_level, 
            log_dir=log_dir
        )
        
        # Initialize vehicle make classifier (if not disabled)
        vehicle_make_classifier = VehicleMakeClassifier(
            model_path=model_paths['vehicle_make'],
            device='cuda:0' if hw_capabilities['cuda'] else 'cpu'
        )
        if args.disable_vehicle_make:
            vehicle_make_classifier.set_enabled(False)
        
        # Initialize vehicle color detector (if not disabled)
        vehicle_color_detector = VehicleColorDetector(
            color_map_path=model_paths['color_map']
        )
        if args.disable_vehicle_color:
            vehicle_color_detector.set_enabled(False)
        
        print(f"\nEnhanced ANPR System started.")
        print(f"Using hardware acceleration: {hw_capabilities['best_option']}")
        print(f"Using detector model: {model_paths['detector']}")
        print(f"Camera resolution: {width}x{height} at {args.fps} FPS")
        print(f"Relay control: GPIO pin {args.gpio_pin}, active time {args.relay_time}s")
        print(f"Vehicle make detection: {'Enabled' if vehicle_make_classifier.enabled else 'Disabled'}")
        print(f"Vehicle color detection: {'Enabled' if vehicle_color_detector.enabled else 'Disabled'}")
        
        # Run enhanced continuous monitoring mode
        enhanced_continuous_mode(
            camera, detector, ocr, checker, relay_controller,
            vehicle_make_classifier, vehicle_color_detector,
            log_dir, hw_capabilities
        )
                
    except Exception as e:
        print(f"Error in main: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\nShutting down Enhanced ANPR System...")
        
        # Clean up OpenCV windows
        try:
            cv2.destroyAllWindows()
            print("Closed OpenCV windows")
        except Exception as e:
            print(f"Error closing OpenCV windows: {e}")
            
        # Release camera resources
        try:
            if 'camera' in locals() and hasattr(camera, 'release'):
                camera.release()
                print("Released camera resources")
        except Exception as e:
            print(f"Error releasing camera resources: {e}")
            
        # Release detector resources
        try:
            if 'detector' in locals() and hasattr(detector, 'release'):
                detector.release()
                print("Released detector resources")
        except Exception as e:
            print(f"Error releasing detector resources: {e}")
        
        # Clean up GPIO
        try:
            if 'relay_controller' in locals():
                relay_controller.cleanup()
                print("Cleaned up GPIO resources")
        except Exception as e:
            print(f"Error cleaning up GPIO resources: {e}")
            
        print("Enhanced ANPR System stopped successfully")

if __name__ == "__main__":
    main()
                    cv2.putText(details_img, conf_text, (20, y_pos), font, 0.7, (0, 80, 0), 2, cv2.LINE_AA)
                    y_pos += 40
                    
                if self.vehicle_color:
                    color_text = f"Vehicle Color: {self.vehicle_color['color']}"
                    conf_text = f"Color Confidence: {self.vehicle_color['confidence']:.2f}"
                    
                    # Add color sample box
                    color_rgb = self.vehicle_color['rgb']
                    color_box_size = 20
                    color_box_x = 160
                    cv2.rectangle(details_img, (color_box_x, y_pos - 15), 
                                 (color_box_x + color_box_size, y_pos + 5), 
                                 (color_rgb[2], color_rgb[1], color_rgb[0]), -1)
                    cv2.rectangle(details_img, (color_box_x, y_pos - 15), 
                                 (color_box_x + color_box_size, y_pos + 5), 
                                 (0, 0, 0), 1)
                                 
                    cv2.putText(details_img, color_text, (20, y_pos), font, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
                    y_pos += 40
                    cv2.putText(details_img, conf_text, (20, y_pos), font, 0.7, (0, 80, 0), 2, cv2.LINE_AA)
                    y_pos += 40
                cv2.putText(details_img, conf_text, (20, y_pos), font, 0.7, (0, 0,