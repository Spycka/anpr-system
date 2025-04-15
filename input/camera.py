#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Camera Input Module
Handles video input from various sources (USB camera, RTSP stream, etc.)

This module provides a unified interface for capturing video frames
from different camera sources with automatic reconnection.
"""

import os
import time
import logging
import threading
import numpy as np
import cv2

class CameraInput:
    """
    Camera Input class for handling video from USB or RTSP sources
    
    Features:
    - Support for USB cameras and RTSP streams
    - Automatic reconnection on stream failure
    - Background frame capture for better performance
    - Frame buffer management
    """
    
    def __init__(self, source=0, width=1280, height=720, buffer_size=4, mock=False):
        """
        Initialize camera input
        
        Args:
            source: Camera source (0 for default USB camera, URL for RTSP stream)
            width: Desired frame width
            height: Desired frame height
            buffer_size: Maximum number of frames to buffer
            mock: Run in mock mode without camera
        """
        self.logger = logging.getLogger('system')
        self.source = source
        self.width = width
        self.height = height
        self.buffer_size = buffer_size
        self.mock_mode = mock
        
        # Frame capture and storage
        self.cap = None
        self.frame_buffer = []
        self.current_frame = None
        self.last_frame_time = 0
        
        # Threading control
        self.running = False
        self.lock = threading.Lock()
        self.capture_thread = None
        
        # Reconnection settings
        self.reconnect_delay = 1.0  # Initial delay in seconds
        self.max_reconnect_delay = 30.0  # Maximum delay
        self.reconnect_attempts = 0
        self.last_reconnect_time = 0
        
        # Mock parameters
        self.mock_frame = None
        
        # Initialize camera
        if not self.mock_mode:
            self._init_camera()
            self._start_capture_thread()
        else:
            self.logger.warning("Running in MOCK MODE - using synthetic video")
            self._init_mock_source()
    
    def _init_camera(self):
        """Initialize the camera source"""
        try:
            self.logger.info(f"Initializing camera from source: {self.source}")
            
            # Determine if source is RTSP URL
            is_rtsp = isinstance(self.source, str) and (
                self.source.lower().startswith('rtsp://') or 
                self.source.lower().startswith('http://')
            )
            
            if is_rtsp:
                # For RTSP streams, use FFmpeg backend for better performance
                self.cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)
            else:
                # For USB cameras, use default backend
                self.cap = cv2.VideoCapture(self.source)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Lower buffer for USB
            
            # Set resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            
            # Check if camera opened successfully
            if not self.cap.isOpened():
                raise RuntimeError(f"Failed to open camera source: {self.source}")
            
            # Read first frame to verify connection
            ret, frame = self.cap.read()
            if not ret or frame is None:
                raise RuntimeError("Failed to read first frame from camera")
            
            # Store first frame
            self.current_frame = frame
            self.last_frame_time = time.time()
            
            self.logger.info(f"Camera initialized successfully: {frame.shape[1]}x{frame.shape[0]}")
            self.reconnect_attempts = 0
            self.reconnect_delay = 1.0
            
        except Exception as e:
            self.logger.error(f"Error initializing camera: {str(e)}")
            self.logger.warning("Camera initialization failed, will retry...")
            
            # Schedule reconnection attempt
            self.last_reconnect_time = time.time()
            
            # Release capture if it was created
            if self.cap is not None:
                self.cap.release()
                self.cap = None
    
    def _init_mock_source(self):
        """Initialize a synthetic video source for mock mode"""
        # Create a blank frame with text
        self.mock_frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Add some text
        cv2.putText(
            self.mock_frame, "MOCK VIDEO", (self.width // 4, self.height // 2),
            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2
        )
        
        # Draw a placeholder for license plate
        plate_width = int(self.width * 0.25)
        plate_height = int(self.height * 0.1)
        x1 = (self.width - plate_width) // 2
        y1 = int(self.height * 0.7)
        cv2.rectangle(
            self.mock_frame, (x1, y1), (x1 + plate_width, y1 + plate_height),
            (0, 255, 0), 2
        )
        
        # Add plate text
        cv2.putText(
            self.mock_frame, "ABC-1234", (x1 + 10, y1 + plate_height - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )
        
        self.current_frame = self.mock_frame.copy()
        self.last_frame_time = time.time()
    
    def _capture_thread_func(self):
        """Background thread for continuous frame capture"""
        self.logger.info("Starting background capture thread")
        
        while self.running:
            # Check if camera needs reconnection
            if self.cap is None or not self.cap.isOpened():
                self._attempt_reconnect()
                continue
            
            try:
                # Capture frame
                ret, frame = self.cap.read()
                
                if not ret or frame is None:
                    # Frame capture failed
                    self.logger.warning("Failed to read frame, will attempt reconnect")
                    self._attempt_reconnect()
                    continue
                
                # Update current frame with lock
                with self.lock:
                    self.current_frame = frame
                    self.last_frame_time = time.time()
                    
                    # Store in buffer (limited size)
                    self.frame_buffer.append(frame.copy())
                    if len(self.frame_buffer) > self.buffer_size:
                        self.frame_buffer.pop(0)
                
                # Small sleep to avoid high CPU usage
                time.sleep(0.01)
                
            except Exception as e:
                self.logger.error(f"Error in capture thread: {str(e)}")
                self._attempt_reconnect()
    
    def _start_capture_thread(self):
        """Start the background capture thread"""
        if self.capture_thread is not None and self.capture_thread.is_alive():
            return  # Thread already running
        
        self.running = True
        self.capture_thread = threading.Thread(
            target=self._capture_thread_func,
            daemon=True,
            name="CameraCapture"
        )
        self.capture_thread.start()
    
    def _attempt_reconnect(self):
        """Attempt to reconnect to the camera source with exponential backoff"""
        current_time = time.time()
        elapsed = current_time - self.last_reconnect_time
        
        # Check if enough time has passed since last attempt
        if elapsed < self.reconnect_delay:
            time.sleep(0.1)  # Small sleep to avoid tight loop
            return
        
        # Update reconnection attempt counters
        self.reconnect_attempts += 1
        self.last_reconnect_time = current_time
        
        # Log reconnection attempt
        self.logger.info(f"Attempting camera reconnect (attempt {self.reconnect_attempts}, " +
                         f"delay {self.reconnect_delay:.1f}s)")
        
        # Release existing capture if any
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        
        # Try to initialize camera
        self._init_camera()
        
        # Update reconnection delay with exponential backoff
        if self.cap is None or not self.cap.isOpened():
            self.reconnect_delay = min(self.reconnect_delay * 1.5, self.max_reconnect_delay)
        else:
            # Reset delay on successful reconnection
            self.reconnect_delay = 1.0
    
    def read(self):
        """
        Read the latest frame from the camera
        
        Returns:
            (success, frame) tuple similar to cv2.VideoCapture.read()
        """
        # Handle mock mode
        if self.mock_mode:
            # Add some visual changes to the mock frame to simulate movement
            mock_frame = self.mock_frame.copy()
            
            # Add timestamp
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(
                mock_frame, timestamp, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
            )
            
            # Add some movement (wobbling plate)
            offset_x = int(np.sin(time.time() * 2) * 5)
            offset_y = int(np.cos(time.time() * 3) * 3)
            
            plate_width = int(self.width * 0.25)
            plate_height = int(self.height * 0.1)
            x1 = (self.width - plate_width) // 2 + offset_x
            y1 = int(self.height * 0.7) + offset_y
            
            # Clear previous plate area
            cv2.rectangle(
                mock_frame, 
                ((self.width - plate_width) // 2 - 10, int(self.height * 0.7) - 10), 
                ((self.width + plate_width) // 2 + 10, int(self.height * 0.7) + plate_height + 10),
                (0, 0, 0), -1
            )
            
            # Draw new plate
            cv2.rectangle(
                mock_frame, (x1, y1), (x1 + plate_width, y1 + plate_height),
                (0, 255, 0), 2
            )
            
            # Add plate text
            cv2.putText(
                mock_frame, "ABC-1234", (x1 + 10, y1 + plate_height - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )
            
            return True, mock_frame
        
        # Get frame with lock
        with self.lock:
            frame = self.current_frame
            elapsed = time.time() - self.last_frame_time
        
        # Check if frame exists and is not too old
        if frame is None:
            return False, None
        
        # Check for stale frames (no update for too long)
        if elapsed > 5.0:  # Consider frame stale after 5 seconds
            self.logger.warning(f"Frame is stale ({elapsed:.1f}s old)")
            if not self.mock_mode:
                self._attempt_reconnect()
            return False, None
        
        return True, frame.copy()
    
    def release(self):
        """Release camera resources"""
        # Stop background thread
        self.running = False
        
        if self.capture_thread is not None and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=1.0)
        
        # Release OpenCV capture
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        
        self.logger.info("Camera resources released")
