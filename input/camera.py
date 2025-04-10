#!/usr/bin/env python3
"""
Camera input module for ANPR system.
Handles RTSP stream connection with auto-reconnect capability.
"""
import os
import time
import threading
import queue
import logging
import cv2
import numpy as np
from typing import Optional, Tuple, Dict, Any, List, Union

# Import logger
from utils.logger import get_logger
logger = get_logger("camera")

class CameraInput:
    """
    Camera input handler for RTSP streams with auto-reconnection.
    
    Features:
    - RTSP stream handling with GStreamer or OpenCV
    - Auto-reconnection on failure
    - Resolution control (480p, 720p, 1080p, or custom)
    - Frame buffering with configurable size
    - Thread-safe frame access
    """
    
    # Standard resolution presets
    RESOLUTIONS = {
        "480p": (640, 480),
        "720p": (1280, 720),
        "1080p": (1920, 1080)
    }
    
    # Default reconnection parameters
    DEFAULT_RECONNECT_DELAY = 5.0  # seconds
    DEFAULT_MAX_RECONNECT_ATTEMPTS = 10
    
    # Default buffer size
    DEFAULT_BUFFER_SIZE = 10
    
    def __init__(
        self,
        rtsp_url: str,
        resolution: str = "720p",
        buffer_size: int = DEFAULT_BUFFER_SIZE,
        use_gstreamer: bool = True,
        reconnect_delay: float = DEFAULT_RECONNECT_DELAY,
        max_reconnect_attempts: int = DEFAULT_MAX_RECONNECT_ATTEMPTS,
        simulate_mode: bool = False,
        simulate_dir: Optional[str] = None
    ):
        """
        Initialize the camera input handler.
        
        Args:
            rtsp_url: RTSP stream URL
            resolution: Resolution preset ("480p", "720p", "1080p") or custom "WIDTHxHEIGHT"
            buffer_size: Number of frames to buffer
            use_gstreamer: Whether to try GStreamer backend first
            reconnect_delay: Seconds to wait before reconnection attempt
            max_reconnect_attempts: Maximum reconnection attempts (0 for infinite)
            simulate_mode: If True, use simulation mode (for testing without camera)
            simulate_dir: Directory containing test images for simulation
        """
        self.rtsp_url = rtsp_url
        self.buffer_size = buffer_size
        self.use_gstreamer = use_gstreamer
        self.reconnect_delay = reconnect_delay
        self.max_reconnect_attempts = max_reconnect_attempts
        self.simulate_mode = simulate_mode
        self.simulate_dir = simulate_dir
        
        # Parse resolution
        self.width, self.height = self._parse_resolution(resolution)
        
        # Frame buffer (thread-safe queue)
        self.frame_buffer = queue.Queue(maxsize=buffer_size)
        
        # Connection status
        self.connected = False
        self.running = False
        self.reconnect_count = 0
        
        # Video capture object
        self.cap = None
        
        # Thread for capture
        self.capture_thread = None
        
        # Lock for thread-safe access
        self.lock = threading.Lock()
        
        # Performance metrics
        self.fps = 0
        self.last_frame_time = 0
        self.metrics = {
            "frames_received": 0,
            "frames_dropped": 0,
            "reconnect_attempts": 0,
            "connection_uptime": 0,
            "current_fps": 0
        }
        
        # Simulation variables
        self.simulate_images = []
        self.simulate_index = 0
        
        logger.info(f"Camera input initialized with URL: {rtsp_url}, "
                   f"Resolution: {self.width}x{self.height}, "
                   f"{'Simulation' if simulate_mode else 'Live'} mode")
    
    def _parse_resolution(self, resolution: str) -> Tuple[int, int]:
        """
        Parse resolution string into width and height.
        
        Args:
            resolution: Resolution string ("480p", "720p", "1080p" or "WIDTHxHEIGHT")
            
        Returns:
            Tuple[int, int]: Width and height
            
        Raises:
            ValueError: If resolution format is invalid
        """
        # Check if it's a preset
        if resolution in self.RESOLUTIONS:
            return self.RESOLUTIONS[resolution]
        
        # Parse custom resolution (format: "WIDTHxHEIGHT")
        try:
            width_str, height_str = resolution.lower().split("x")
            width = int(width_str)
            height = int(height_str)
            return width, height
        except (ValueError, TypeError):
            logger.error(f"Invalid resolution format: {resolution}")
            logger.info("Using default 720p resolution")
            return self.RESOLUTIONS["720p"]
    
    def _load_simulation_images(self) -> None:
        """Load images for simulation mode."""
        if not self.simulate_dir or not os.path.isdir(self.simulate_dir):
            logger.error(f"Simulation directory not found: {self.simulate_dir}")
            return
        
        # Get image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        
        for file in os.listdir(self.simulate_dir):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(self.simulate_dir, file))
        
        if not image_files:
            logger.error(f"No image files found in simulation directory: {self.simulate_dir}")
            return
        
        # Load images
        for file in sorted(image_files):
            try:
                img = cv2.imread(file)
                if img is not None:
                    # Resize to match specified resolution
                    img = cv2.resize(img, (self.width, self.height))
                    self.simulate_images.append(img)
                    logger.debug(f"Loaded simulation image: {file}")
            except Exception as e:
                logger.error(f"Error loading simulation image {file}: {str(e)}")
        
        logger.info(f"Loaded {len(self.simulate_images)} simulation images")
    
    def _get_next_simulation_frame(self) -> Optional[np.ndarray]:
        """
        Get the next frame in simulation mode.
        
        Returns:
            Optional[np.ndarray]: Next simulation frame or None if no images
        """
        if not self.simulate_images:
            return None
        
        # Get next image
        img = self.simulate_images[self.simulate_index]
        
        # Update index (cycle through images)
        self.simulate_index = (self.simulate_index + 1) % len(self.simulate_images)
        
        return img.copy()
    
    def _build_capture_pipeline(self) -> None:
        """Build the video capture pipeline based on settings."""
        if self.simulate_mode:
            # Simulation mode - load test images
            logger.info("Using simulation mode for camera input")
            self._load_simulation_images()
            return
        
        try:
            # Close existing capture if any
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            
            # Try GStreamer pipeline if requested
            if self.use_gstreamer and cv2.getBuildInformation().find("GStreamer") != -1:
                logger.info("Attempting to use GStreamer backend")
                
                # Build GStreamer pipeline
                gst_pipeline = (
                    f"rtspsrc location={self.rtsp_url} latency=0 buffer-mode=auto ! "
                    f"rtph264depay ! h264parse ! avdec_h264 ! "
                    f"videoscale ! video/x-raw,width={self.width},height={self.height} ! "
                    f"videoconvert ! appsink max-buffers=1 drop=true"
                )
                
                self.cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
                
                # Check if GStreamer pipeline opened successfully
                if not self.cap.isOpened():
                    logger.warning("GStreamer pipeline failed, falling back to OpenCV")
                    self.cap.release()
                    self.cap = None
                else:
                    logger.info("Successfully connected using GStreamer pipeline")
                    return
            
            # Fallback to standard OpenCV
            logger.info("Using standard OpenCV for RTSP connection")
            
            # Configure OpenCV parameters for RTSP
            self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
            
            # Set buffer size
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Set resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            
            # Check if OpenCV connection opened successfully
            if not self.cap.isOpened():
                logger.error(f"Failed to open RTSP stream: {self.rtsp_url}")
                self.cap = None
                return
            
            logger.info("Successfully connected using OpenCV")
        
        except Exception as e:
            logger.error(f"Error creating capture pipeline: {str(e)}")
            if self.cap is not None:
                self.cap.release()
                self.cap = None
    
    def _capture_loop(self) -> None:
        """Main capture loop to grab frames and manage the connection."""
        logger.info("Capture thread started")
        
        connection_start_time = time.time()
        frame_count = 0
        fps_update_interval = 5.0  # Update FPS every 5 seconds
        last_fps_update = time.time()
        
        while self.running:
            # Check if we're in simulation mode
            if self.simulate_mode:
                # Sleep to simulate frame rate
                time.sleep(1.0 / 30.0)  # 30 FPS simulation
                
                # Get simulation frame
                frame = self._get_next_simulation_frame()
                
                if frame is not None:
                    # Update metrics
                    current_time = time.time()
                    frame_count += 1
                    self.metrics["frames_received"] += 1
                    
                    # Add frame to buffer (drop if full)
                    try:
                        if self.frame_buffer.full():
                            self.frame_buffer.get_nowait()  # Remove oldest frame
                            self.metrics["frames_dropped"] += 1
                        
                        self.frame_buffer.put_nowait(frame)
                    except queue.Full:
                        self.metrics["frames_dropped"] += 1
                    
                    # Calculate FPS
                    if current_time - last_fps_update >= fps_update_interval:
                        self.fps = frame_count / (current_time - last_fps_update)
                        self.metrics["current_fps"] = self.fps
                        frame_count = 0
                        last_fps_update = current_time
                        logger.debug(f"Simulation FPS: {self.fps:.2f}")
                
                # No need to handle reconnection in simulation mode
                continue
            
            # Real camera mode
            if self.cap is None or not self.cap.isOpened():
                # Not connected, attempt to reconnect
                self._handle_reconnection()
                continue
            
            # Capture frame
            ret, frame = self.cap.read()
            
            if not ret:
                logger.warning("Failed to grab frame, connection may be lost")
                with self.lock:
                    self.connected = False
                
                # Handle reconnection
                self._handle_reconnection()
                continue
            
            # Successfully got a frame
            with self.lock:
                self.connected = True
                self.reconnect_count = 0
            
            # Update metrics
            current_time = time.time()
            frame_count += 1
            self.metrics["frames_received"] += 1
            self.metrics["connection_uptime"] = current_time - connection_start_time
            
            # Add frame to buffer (drop if full)
            try:
                if self.frame_buffer.full():
                    self.frame_buffer.get_nowait()  # Remove oldest frame
                    self.metrics["frames_dropped"] += 1
                
                self.frame_buffer.put_nowait(frame)
            except queue.Full:
                self.metrics["frames_dropped"] += 1
            
            # Calculate FPS
            if current_time - last_fps_update >= fps_update_interval:
                self.fps = frame_count / (current_time - last_fps_update)
                self.metrics["current_fps"] = self.fps
                frame_count = 0
                last_fps_update = current_time
                logger.debug(f"Camera FPS: {self.fps:.2f}")
        
        # Clean up when thread exits
        logger.info("Capture thread stopping")
        if self.cap is not None:
            self.cap.release()
            self.cap = None
    
    def _handle_reconnection(self) -> None:
        """Handle reconnection attempts to the RTSP stream."""
        # Check if we should stop trying
        if (self.max_reconnect_attempts > 0 and 
                self.reconnect_count >= self.max_reconnect_attempts):
            logger.error(f"Maximum reconnection attempts ({self.max_reconnect_attempts}) reached")
            self.running = False
            return
        
        # Increment reconnect counter
        with self.lock:
            self.connected = False
            self.reconnect_count += 1
            self.metrics["reconnect_attempts"] += 1
        
        logger.info(f"Reconnection attempt {self.reconnect_count} "
                   f"(max: {self.max_reconnect_attempts if self.max_reconnect_attempts > 0 else 'unlimited'})")
        
        # Wait before reconnecting
        time.sleep(self.reconnect_delay)
        
        # Rebuild capture pipeline
        self._build_capture_pipeline()
    
    def start(self) -> bool:
        """
        Start the camera capture thread.
        
        Returns:
            bool: True if started successfully, False otherwise
        """
        with self.lock:
            if self.running:
                logger.warning("Camera capture already running")
                return True
            
            # Set running flag
            self.running = True
        
        # Build capture pipeline
        self._build_capture_pipeline()
        
        # Start capture thread
        self.capture_thread = threading.Thread(target=self._capture_loop)
        self.capture_thread.daemon = True
        self.capture_thread.start()
        
        logger.info("Camera capture started")
        return True
    
    def stop(self) -> None:
        """Stop the camera capture thread."""
        with self.lock:
            self.running = False
        
        # Wait for thread to finish
        if self.capture_thread is not None and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=3.0)
        
        # Release capture
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        
        logger.info("Camera capture stopped")
    
    def get_frame(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """
        Get the next frame from the buffer.
        
        Args:
            timeout: Maximum time to wait for a frame (seconds)
            
        Returns:
            Optional[np.ndarray]: Frame image or None if timeout
        """
        try:
            return self.frame_buffer.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def is_connected(self) -> bool:
        """
        Check if camera is connected and receiving frames.
        
        Returns:
            bool: True if connected, False otherwise
        """
        with self.lock:
            return self.connected
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get camera performance metrics.
        
        Returns:
            Dict[str, Any]: Dictionary of metrics
        """
        with self.lock:
            return self.metrics.copy()
    
    def __enter__(self):
        """Context manager enter method."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit method."""
        self.stop()

# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Example RTSP URL (replace with your camera's URL)
    rtsp_url = "rtsp://username:password@camera-ip:554/stream"
    
    # For testing without a camera, use simulation mode
    simulate = True
    
    if simulate:
        # Create simulation directory if it doesn't exist
        sim_dir = "simulation_images"
        os.makedirs(sim_dir, exist_ok=True)
        
        # Create some test images for simulation (if needed)
        if not os.listdir(sim_dir):
            for i in range(5):
                # Create a simple test image
                img = np.zeros((720, 1280, 3), dtype=np.uint8)
                cv2.putText(img, f"Test Image {i+1}", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.rectangle(img, (500, 300), (800, 500), (0, 0, 255), 2)
                cv2.imwrite(os.path.join(sim_dir, f"test_image_{i+1}.jpg"), img)
        
        # Create camera with simulation
        camera = CameraInput(
            rtsp_url=rtsp_url,
            resolution="720p",
            simulate_mode=True,
            simulate_dir=sim_dir
        )
    else:
        # Create real camera
        camera = CameraInput(
            rtsp_url=rtsp_url,
            resolution="720p"
        )
    
    # Start camera
    camera.start()
    
    try:
        # Process frames for 10 seconds
        start_time = time.time()
        while time.time() - start_time < 10:
            # Get frame
            frame = camera.get_frame(timeout=0.5)
            
            if frame is not None:
                # Display frame
                cv2.imshow("Camera Feed", frame)
                
                # Exit if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                print("No frame received")
        
        # Print metrics
        print("\nCamera Metrics:")
        for key, value in camera.get_metrics().items():
            print(f"  {key}: {value}")
    
    finally:
        # Clean up
        camera.stop()
        cv2.destroyAllWindows()
        print("Camera test completed")
