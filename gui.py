    def update_plate_display(self, plate_img, vehicle_data):
        """Update the plate display with the latest plate"""
        if plate_img is None or not vehicle_data:
            return
        
        # Extract data
        plate_text = vehicle_data['plate']
        confidence = vehicle_data['confidence']
        is_authorized = vehicle_data['authorized']
        make = vehicle_data.get('make')
        color = vehicle_data.get('color')
        
        # Store for saving
        self.latest_plate_img = plate_img.copy()
        
        # Resize to fit canvas
        canvas_width = self.plate_canvas.winfo_width()
        canvas_height = self.plate_canvas.winfo_height()
        
        if canvas_width > 1 and canvas_height > 1:
            # Only resize if canvas has valid dimensions
            aspect_ratio = plate_img.shape[1] / plate_img.shape[0]
            
            if canvas_width / canvas_height > aspect_ratio:
                # Canvas is wider than image
                new_height = canvas_height
                new_width = int(new_height * aspect_ratio)
            else:
                # Canvas is taller than image
                new_width = canvas_width
                new_height = int(new_width / aspect_ratio)
            
            # Resize image to fit canvas
            plate_img_resized = cv2.resize(plate_img, (new_width, new_height))
            
            # Convert to RGB for tkinter
            plate_img_rgb = cv2.cvtColor(plate_img_resized, cv2.COLOR_BGR2RGB)
            
            # Convert to ImageTk format
            img = Image.fromarray(plate_img_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            
            # Update canvas
            self.plate_canvas.config(width=new_width, height=new_height)
            self.plate_canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
            self.plate_canvas.image = imgtk  # Keep reference
        
        # Update plate info
        self.plate_text_var.set(plate_text)
        self.plate_conf_var.set(f"{confidence:.2f}")
        
        # Update status
        status = "AUTHORIZED" if is_authorized else "DENIED"
        color_code = "green" if is_authorized else "red"
        self.plate_status_var.set(status)
        
        # Update latest plate label
        self.latest_plate_label.config(text=plate_text)
        
        # Update make/color info if available
        vehicle_info = []
        if make:
            vehicle_info.append(f"Make: {make}")
        if color:
            vehicle_info.append(f"Color: {color}")
        
        vehicle_info_str = ", ".join(vehicle_info) if vehicle_info else "No classification"
        
        # Update GUI status with all information
        status_text = f"Detected plate: {plate_text} - {status}"
        if vehicle_info:
            status_text += f" - {vehicle_info_str}"
        self.status_bar.config(text=status_text)#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ANPR System - GUI Application
Graphical Interface for License Plate Recognition System

This module provides a GUI for monitoring and controlling the
ANPR system, displaying camera feed, detection results, and logs.
"""

import os
import sys
import time
import threading
import queue
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk
import logging
from datetime import datetime

# Import local modules
from detectors.yolo11_rknn import LicensePlateDetector
from vision.vehicle_classifier import VehicleClassifier
from input.camera import CameraInput
from utils.plate_checker import PlateChecker
from utils.gpio import GateController
from utils.logger import setup_logger, configure_global_logging

# Global flag for clean shutdown
running = True

class ANPRGuiApp:
    """GUI Application for ANPR System"""
    
    def __init__(self, root, config):
        """Initialize the GUI application"""
        self.root = root
        self.config = config
        
        # Setup window
        self.root.title("ANPR System - Orange Pi 5 Ultra")
        self.root.geometry("1280x800")
        self.root.minsize(1024, 768)
        
        # Setup logging
        self.setup_logging()
        
        # Initialize components
        self.initialize_components()
        
        # Create GUI layout
        self.create_layout()
        
        # Queue for thread communication
        self.frame_queue = queue.Queue(maxsize=5)
        self.detection_queue = queue.Queue(maxsize=10)
        self.log_queue = queue.Queue(maxsize=100)
        
        # Start threads
        self.start_threads()
        
        # Setup periodic updates
        self.update_gui()
        
        # Setup cleanup on window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        self.logger.info("GUI application initialized")
    
    def setup_logging(self):
        """Setup logging for the GUI application"""
        configure_global_logging(
            self.config['log_dir'],
            debug=self.config['debug']
        )
        
        self.logger = setup_logger(
            'gui',
            os.path.join(self.config['log_dir'], 'gui.log'),
            level=logging.INFO if not self.config['debug'] else logging.DEBUG
        )
    
    def initialize_components(self):
        """Initialize system components"""
        self.logger.info("Initializing system components...")
        
        # Initialize camera
        self.camera = CameraInput(
            source=self.config['camera_source'],
            buffer_size=self.config['buffer_size'],
            width=self.config['frame_width'],
            height=self.config['frame_height'],
            mock=self.config['mock_mode']
        )
        
        # Initialize license plate detector
        self.detector = LicensePlateDetector(
            model_path=self.config['model_path'],
            conf_threshold=self.config['detection_confidence'],
            nms_threshold=self.config['nms_threshold'],
            mock=self.config['mock_mode']
        )
        
        # Initialize OCR
        self.ocr = PlateOCR(
            confidence_threshold=self.config['ocr_confidence'],
            mock=self.config['mock_mode']
        )
        
        # Initialize vehicle classifier if enabled
        if self.config['enable_vehicle_classification']:
            self.vehicle_classifier = VehicleClassifier(
                model_path=self.config['vehicle_make_model_path'],
                enable_make=self.config['enable_make_detection'],
                enable_color=self.config['enable_color_detection'],
                mock=self.config['mock_mode']
            )
        else:
            self.vehicle_classifier = None
        
        # Initialize plate checker
        self.plate_checker = PlateChecker(
            allowlist_path=self.config['allowlist_path'],
            auto_reload=True
        )
        
        # Initialize gate controller
        self.gate_controller = GateController(
            pin=self.config['gpio_pin'],
            pulse_time=self.config['gate_pulse_time'],
            cooldown_time=self.config['gate_cooldown_time'],
            mock=self.config['mock_mode']
        )
        
        # Detection state
        self.last_plates = {}  # plate_text -> (timestamp, confidence)
        self.latest_plate_img = None
        self.processing_fps = 0
        self.detections_count = 0
        self.last_time = time.time()
        self.fps_counter = 0
    
    def create_layout(self):
        """Create the GUI layout"""
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Top control panel
        control_frame = ttk.LabelFrame(main_frame, text="Control Panel")
        control_frame.pack(fill=tk.X, pady=(0, 5))
        
        # Camera source selection
        ttk.Label(control_frame, text="Camera Source:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.camera_source_var = tk.StringVar(value=str(self.config['camera_source']))
        camera_entry = ttk.Entry(control_frame, textvariable=self.camera_source_var, width=30)
        camera_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        
        ttk.Button(control_frame, text="Connect", command=self.reconnect_camera).grid(
            row=0, column=2, padx=5, pady=5, sticky=tk.W)
        
        # Detection settings
        ttk.Label(control_frame, text="Detection Confidence:").grid(
            row=0, column=3, padx=(20,5), pady=5, sticky=tk.W)
        
        self.conf_var = tk.DoubleVar(value=self.config['detection_confidence'])
        conf_scale = ttk.Scale(control_frame, from_=0.1, to=0.9, length=150,
                             orient=tk.HORIZONTAL, variable=self.conf_var)
        conf_scale.grid(row=0, column=4, padx=5, pady=5, sticky=tk.W)
        ttk.Label(control_frame, textvariable=self.conf_var).grid(
            row=0, column=5, padx=5, pady=5, sticky=tk.W)
        
        # Gate control
        ttk.Button(control_frame, text="Open Gate", command=self.manual_open_gate).grid(
            row=0, column=6, padx=(20,5), pady=5, sticky=tk.W)
        
        # Logs clear
        ttk.Button(control_frame, text="Clear Logs", command=self.clear_logs).grid(
            row=0, column=7, padx=5, pady=5, sticky=tk.W)
        
        # Main content - split into left and right panes
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left side - camera feed and detection
        left_frame = ttk.Frame(content_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # Camera feed
        camera_frame = ttk.LabelFrame(left_frame, text="Camera Feed")
        camera_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        
        self.camera_canvas = tk.Canvas(camera_frame, bg="black")
        self.camera_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Detection stats
        stats_frame = ttk.LabelFrame(left_frame, text="Detection Statistics")
        stats_frame.pack(fill=tk.X, pady=(0, 5))
        
        stats_grid = ttk.Frame(stats_frame)
        stats_grid.pack(fill=tk.X, expand=True, padx=5, pady=5)
        
        ttk.Label(stats_grid, text="Processing FPS:").grid(row=0, column=0, padx=5, pady=2, sticky=tk.W)
        self.fps_label = ttk.Label(stats_grid, text="0.0")
        self.fps_label.grid(row=0, column=1, padx=5, pady=2, sticky=tk.W)
        
        ttk.Label(stats_grid, text="Detections:").grid(row=1, column=0, padx=5, pady=2, sticky=tk.W)
        self.detections_label = ttk.Label(stats_grid, text="0")
        self.detections_label.grid(row=1, column=1, padx=5, pady=2, sticky=tk.W)
        
        ttk.Label(stats_grid, text="Latest Plate:").grid(row=0, column=2, padx=(20,5), pady=2, sticky=tk.W)
        self.latest_plate_label = ttk.Label(stats_grid, text="None")
        self.latest_plate_label.grid(row=0, column=3, padx=5, pady=2, sticky=tk.W)
        
        ttk.Label(stats_grid, text="Status:").grid(row=1, column=2, padx=(20,5), pady=2, sticky=tk.W)
        self.status_label = ttk.Label(stats_grid, text="Ready")
        self.status_label.grid(row=1, column=3, padx=5, pady=2, sticky=tk.W)
        
        # Right side - plate image, allowlist, and logs
        right_frame = ttk.Frame(content_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Latest plate image
        plate_frame = ttk.LabelFrame(right_frame, text="Latest Plate")
        plate_frame.pack(fill=tk.X, pady=(0, 5))
        
        self.plate_canvas = tk.Canvas(plate_frame, bg="black", height=100)
        self.plate_canvas.pack(fill=tk.X, padx=5, pady=5)
        
        # Plate info
        plate_info_frame = ttk.Frame(plate_frame)
        plate_info_frame.pack(fill=tk.X, padx=5, pady=(0, 5))
        
        ttk.Label(plate_info_frame, text="Text:").grid(row=0, column=0, padx=5, pady=2, sticky=tk.W)
        self.plate_text_var = tk.StringVar(value="None")
        ttk.Label(plate_info_frame, textvariable=self.plate_text_var).grid(
            row=0, column=1, padx=5, pady=2, sticky=tk.W)
        
        ttk.Label(plate_info_frame, text="Confidence:").grid(row=0, column=2, padx=(20,5), pady=2, sticky=tk.W)
        self.plate_conf_var = tk.StringVar(value="0.0")
        ttk.Label(plate_info_frame, textvariable=self.plate_conf_var).grid(
            row=0, column=3, padx=5, pady=2, sticky=tk.W)
        
        ttk.Label(plate_info_frame, text="Status:").grid(row=0, column=4, padx=(20,5), pady=2, sticky=tk.W)
        self.plate_status_var = tk.StringVar(value="Unknown")
        plate_status_label = ttk.Label(plate_info_frame, textvariable=self.plate_status_var)
        plate_status_label.grid(row=0, column=5, padx=5, pady=2, sticky=tk.W)
        
        # Create a notebook for tabs
        notebook = ttk.Notebook(right_frame)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # Allowlist tab
        allowlist_frame = ttk.Frame(notebook)
        notebook.add(allowlist_frame, text="Allowlist")
        
        # Allowlist controls
        allowlist_control_frame = ttk.Frame(allowlist_frame)
        allowlist_control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(allowlist_control_frame, text="License Plate:").pack(side=tk.LEFT, padx=5)
        self.new_plate_var = tk.StringVar()
        ttk.Entry(allowlist_control_frame, textvariable=self.new_plate_var, width=15).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(allowlist_control_frame, text="Add", command=self.add_plate).pack(side=tk.LEFT, padx=5)
        ttk.Button(allowlist_control_frame, text="Remove", command=self.remove_plate).pack(side=tk.LEFT, padx=5)
        ttk.Button(allowlist_control_frame, text="Reload", command=self.reload_allowlist).pack(side=tk.LEFT, padx=5)
        
        # Allowlist display
        allowlist_display_frame = ttk.Frame(allowlist_frame)
        allowlist_display_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.allowlist_text = scrolledtext.ScrolledText(allowlist_display_frame, height=10)
        self.allowlist_text.pack(fill=tk.BOTH, expand=True)
        
        # Logs tab
        logs_frame = ttk.Frame(notebook)
        notebook.add(logs_frame, text="Logs")
        
        self.log_text = scrolledtext.ScrolledText(logs_frame, height=10)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Status bar
        self.status_bar = ttk.Label(main_frame, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(fill=tk.X, side=tk.BOTTOM, pady=(5, 0))
        
        # Initial allowlist load
        self.update_allowlist_display()
    
    def start_threads(self):
        """Start background threads for processing"""
        # Camera thread
        self.camera_thread = threading.Thread(
            target=self.camera_thread_func,
            daemon=True,
            name="CameraThread"
        )
        self.camera_thread.start()
        
        # Detection thread
        self.detection_thread = threading.Thread(
            target=self.detection_thread_func,
            daemon=True,
            name="DetectionThread"
        )
        self.detection_thread.start()
        
        # Log capture thread
        self.log_handler = LogQueueHandler(self.log_queue)
        root_logger = logging.getLogger()
        root_logger.addHandler(self.log_handler)
    
    def update_gui(self):
        """Update GUI elements - called periodically"""
        # Update camera feed if available
        try:
            if not self.frame_queue.empty():
                frame, detections = self.frame_queue.get_nowait()
                self.update_camera_display(frame, detections)
                self.frame_queue.task_done()
        except queue.Empty:
            pass
        
        # Update plate detection display if available
        try:
            if not self.detection_queue.empty():
                plate_img, vehicle_data = self.detection_queue.get_nowait()
                self.update_plate_display(plate_img, vehicle_data)
                self.detection_queue.task_done()
        except queue.Empty:
            pass
        
        # Update logs if available
        self.update_logs()
        
        # Update statistics
        self.update_statistics()
        
        # Schedule next update (every 33ms = ~30 FPS for GUI updates)
        self.root.after(33, self.update_gui)
    
    def camera_thread_func(self):
        """Background thread for camera capture and display"""
        self.logger.info("Starting camera thread")
        
        while running:
            try:
                # Read frame from camera
                success, frame = self.camera.read()
                
                if not success:
                    time.sleep(0.1)  # Small sleep to avoid tight loop
                    continue
                
                # Run detection
                detections = self.detector.detect(frame)
                
                # Put frame and detections in queue for display
                if not self.frame_queue.full():
                    self.frame_queue.put((frame.copy(), detections))
                
                # Update frame count for FPS calculation
                self.fps_counter += 1
                
                # Process detected plates
                for detection in detections:
                    x1, y1, x2, y2, confidence, _ = detection
                    
                    # Extract plate region
                    margin = 5
                    h, w = frame.shape[:2]
                    x1, y1 = max(0, int(x1) - margin), max(0, int(y1) - margin)
                    x2, y2 = min(w, int(x2) + margin), min(h, int(y2) + margin)
                    
                    plate_img = frame[y1:y2, x1:x2]
                    
                    # Skip if plate is too small
                    if plate_img.size == 0 or plate_img.shape[0] < 20 or plate_img.shape[1] < 40:
                        continue
                    
                    # Process with OCR in the detection thread
                    self.detection_thread_queue_plate(plate_img, confidence)
                
                # Throttle processing based on target FPS
                target_frame_time = 1.0 / self.config['processing_fps']
                elapsed = time.time() - self.last_time
                if elapsed < target_frame_time:
                    time.sleep(target_frame_time - elapsed)
                
            except Exception as e:
                self.logger.error(f"Error in camera thread: {str(e)}")
                time.sleep(0.5)  # Avoid tight loop on error
    
    def detection_thread_queue_plate(self, plate_img, confidence):
        """Queue a plate image for OCR processing"""
        # Run OCR to get plate text
        plate_text, ocr_confidence = self.ocr.read_plate(plate_img)
        
        # Skip if confidence is too low or plate text is empty
        if not plate_text or ocr_confidence < self.config['ocr_confidence']:
            return
        
        # Check if already processed recently
        current_time = time.time()
        if plate_text in self.last_plates:
            last_time, _ = self.last_plates[plate_text]
            if current_time - last_time < self.config['plate_debounce_time']:
                return
        
        # Update last seen time
        self.last_plates[plate_text] = (current_time, ocr_confidence)
        
        # Run vehicle classification if enabled
        make = None
        color = None
        if self.vehicle_classifier is not None:
            # Classify vehicle
            make, make_confidence, color, color_confidence = self.vehicle_classifier.classify_vehicle(plate_img)
            
            # Only use results if confidence is high enough
            if make_confidence < self.config['make_confidence_threshold']:
                make = None
            if color_confidence < self.config['color_confidence_threshold']:
                color = None
            
            self.logger.info(f"Vehicle classification for {plate_text}: Make={make or 'unknown'}, Color={color or 'unknown'}")
        
        # Check against allowlist with vehicle attributes
        is_authorized = self.plate_checker.check_plate(plate_text, make, color)
        
        # Increment detection count
        self.detections_count += 1
        
        # Queue for display update
        if not self.detection_queue.full():
            vehicle_data = {
                'plate': plate_text,
                'confidence': ocr_confidence,
                'authorized': is_authorized,
                'make': make,
                'color': color
            }
            self.detection_queue.put((plate_img.copy(), vehicle_data))
        
        # Save plate image
        self.save_plate_image(plate_img, plate_text, is_authorized, ocr_confidence, make, color)
        
        # Open gate if authorized
        if is_authorized:
            self.gate_controller.open_gate()
    
    def detection_thread_func(self):
        """Background thread for OCR processing"""
        self.logger.info("Starting detection thread")
        
        while running:
            try:
                # Clean up old entries in last_plates
                current_time = time.time()
                expired_plates = []
                
                for plate, (timestamp, _) in self.last_plates.items():
                    if current_time - timestamp > self.config['plate_debounce_time'] * 2:
                        expired_plates.append(plate)
                
                for plate in expired_plates:
                    del self.last_plates[plate]
                
                # Sleep to avoid tight loop
                time.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"Error in detection thread: {str(e)}")
                time.sleep(0.5)  # Avoid tight loop on error
    
    def update_camera_display(self, frame, detections):
        """Update the camera display with the latest frame and detections"""
        if frame is None:
            return
        
        # Create a copy for drawing
        display_frame = frame.copy()
        
        # Draw detection boxes
        for detection in detections:
            x1, y1, x2, y2, confidence, class_id = detection
            
            # Convert to integers
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Draw bounding box
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw confidence and class
            label = f"Plate: {confidence:.2f}"
            cv2.putText(display_frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(display_frame, timestamp, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Draw FPS
        fps_text = f"FPS: {self.processing_fps:.1f}"
        cv2.putText(display_frame, fps_text, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Convert to RGB for tkinter
        display_frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        
        # Resize to fit canvas
        canvas_width = self.camera_canvas.winfo_width()
        canvas_height = self.camera_canvas.winfo_height()
        
        if canvas_width > 1 and canvas_height > 1:
            # Only resize if canvas has valid dimensions
            aspect_ratio = display_frame.shape[1] / display_frame.shape[0]
            
            if canvas_width / canvas_height > aspect_ratio:
                # Canvas is wider than frame
                new_height = canvas_height
                new_width = int(new_height * aspect_ratio)
            else:
                # Canvas is taller than frame
                new_width = canvas_width
                new_height = int(new_width / aspect_ratio)
            
            # Resize frame to fit canvas
            display_frame_resized = cv2.resize(display_frame_rgb, (new_width, new_height))
            
            # Convert to ImageTk format
            img = Image.fromarray(display_frame_resized)
            imgtk = ImageTk.PhotoImage(image=img)
            
            # Update canvas
            self.camera_canvas.config(width=new_width, height=new_height)
            self.camera_canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
            self.camera_canvas.image = imgtk  # Keep reference
    
    def update_plate_display(self, plate_img, plate_text, confidence, is_authorized):
        """Update the plate display with the latest plate"""
        if plate_img is None:
            return
        
        # Store for saving
        self.latest_plate_img = plate_img.copy()
        
        # Resize to fit canvas
        canvas_width = self.plate_canvas.winfo_width()
        canvas_height = self.plate_canvas.winfo_height()
        
        if canvas_width > 1 and canvas_height > 1:
            # Only resize if canvas has valid dimensions
            aspect_ratio = plate_img.shape[1] / plate_img.shape[0]
            
            if canvas_width / canvas_height > aspect_ratio:
                # Canvas is wider than image
                new_height = canvas_height
                new_width = int(new_height * aspect_ratio)
            else:
                # Canvas is taller than image
                new_width = canvas_width
                new_height = int(new_width / aspect_ratio)
            
            # Resize image to fit canvas
            plate_img_resized = cv2.resize(plate_img, (new_width, new_height))
            
            # Convert to RGB for tkinter
            plate_img_rgb = cv2.cvtColor(plate_img_resized, cv2.COLOR_BGR2RGB)
            
            # Convert to ImageTk format
            img = Image.fromarray(plate_img_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            
            # Update canvas
            self.plate_canvas.config(width=new_width, height=new_height)
            self.plate_canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
            self.plate_canvas.image = imgtk  # Keep reference
        
        # Update plate info
        self.plate_text_var.set(plate_text)
        self.plate_conf_var.set(f"{confidence:.2f}")
        
        # Update status
        status = "AUTHORIZED" if is_authorized else "DENIED"
        color = "green" if is_authorized else "red"
        self.plate_status_var.set(status)
        
        # Update latest plate label
        self.latest_plate_label.config(text=plate_text)
        
        # Update GUI status
        self.status_bar.config(text=f"Detected plate: {plate_text} - {status}")
    
    def update_logs(self):
        """Update log display with new log entries"""
        while not self.log_queue.empty():
            try:
                record = self.log_queue.get_nowait()
                formatted_message = self.log_handler.format(record)
                self.log_text.insert(tk.END, formatted_message + "\n")
                self.log_text.see(tk.END)  # Scroll to bottom
                self.log_queue.task_done()
            except queue.Empty:
                break
    
    def update_statistics(self):
        """Update statistics display"""
        # Calculate FPS
        current_time = time.time()
        elapsed = current_time - self.last_time
        
        if elapsed >= 1.0:  # Update once per second
            self.processing_fps = self.fps_counter / elapsed
            self.fps_counter = 0
            self.last_time = current_time
            
            # Update labels
            self.fps_label.config(text=f"{self.processing_fps:.1f}")
            self.detections_label.config(text=str(self.detections_count))
    
    def update_allowlist_display(self):
        """Update allowlist display"""
        allowlist = self.plate_checker.get_allowlist()
        
        # Clear existing text
        self.allowlist_text.delete(1.0, tk.END)
        
        # Add header
        self.allowlist_text.insert(tk.END, "License Plate Allowlist\n")
        self.allowlist_text.insert(tk.END, "-------------------\n")
        
        # Add each plate with attributes
        i = 1
        for plate, attrs in allowlist.items():
            plate_info = f"{i}. {plate}"
            
            # Add attributes if present
            if 'make' in attrs:
                plate_info += f" - Make: {attrs['make']}"
            
            if 'color' in attrs:
                plate_info += f" - Color: {attrs['color']}"
            
            self.allowlist_text.insert(tk.END, f"{plate_info}\n")
            i += 1
        
        if not allowlist:
            self.allowlist_text.insert(tk.END, "No plates in allowlist\n")
    
    def reconnect_camera(self):
        """Reconnect to camera with new source"""
        try:
            # Get new source from entry
            new_source = self.camera_source_var.get()
            
            # Try to convert to integer for USB camera
            try:
                new_source = int(new_source)
            except ValueError:
                # If not an integer, treat as RTSP URL
                pass
            
            # Release old camera
            self.camera.release()
            
            # Update config
            self.config['camera_source'] = new_source
            
            # Create new camera
            self.camera = CameraInput(
                source=new_source,
                buffer_size=self.config['buffer_size'],
                width=self.config['frame_width'],
                height=self.config['frame_height'],
                mock=self.config['mock_mode']
            )
            
            # Update status
            self.status_bar.config(text=f"Connected to camera source: {new_source}")
            self.logger.info(f"Reconnected to camera source: {new_source}")
            
        except Exception as e:
            error_msg = f"Error connecting to camera: {str(e)}"
            self.status_bar.config(text=error_msg)
            self.logger.error(error_msg)
            messagebox.showerror("Camera Error", error_msg)
    
    def manual_open_gate(self):
        """Manually open gate"""
        success = self.gate_controller.open_gate()
        
        if success:
            self.logger.info("Gate opened manually")
            self.status_bar.config(text="Gate opened manually")
        else:
            self.logger.info("Gate is on cooldown, cannot open")
            self.status_bar.config(text="Gate is on cooldown, cannot open")
    
    def add_plate(self):
        """Add plate to allowlist"""
        plate = self.new_plate_var.get().strip()
        
        if not plate:
            messagebox.showwarning("Input Error", "Please enter a license plate")
            return
        
        # Ask for vehicle attributes
        make = None
        color = None
        
        # Only ask for attributes if vehicle classification is enabled
        if self.config['enable_vehicle_classification']:
            # Simple dialog to get make and color
            dialog = tk.Toplevel(self.root)
            dialog.title("Vehicle Attributes")
            dialog.geometry("300x200")
            dialog.resizable(False, False)
            dialog.transient(self.root)
            dialog.grab_set()
            
            # Make input
            make_frame = ttk.Frame(dialog)
            make_frame.pack(fill=tk.X, padx=10, pady=10)
            ttk.Label(make_frame, text="Make:").pack(side=tk.LEFT)
            make_var = tk.StringVar()
            make_entry = ttk.Entry(make_frame, textvariable=make_var)
            make_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
            
            # Color input
            color_frame = ttk.Frame(dialog)
            color_frame.pack(fill=tk.X, padx=10, pady=10)
            ttk.Label(color_frame, text="Color:").pack(side=tk.LEFT)
            color_var = tk.StringVar()
            color_entry = ttk.Entry(color_frame, textvariable=color_var)
            color_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
            
            # Note that * means any
            note_frame = ttk.Frame(dialog)
            note_frame.pack(fill=tk.X, padx=10, pady=10)
            ttk.Label(note_frame, text="Note: Use '*' for any make/color").pack(anchor=tk.W)
            
            # Buttons
            button_frame = ttk.Frame(dialog)
            button_frame.pack(fill=tk.X, padx=10, pady=10)
            ttk.Button(button_frame, text="OK", command=dialog.destroy).pack(side=tk.RIGHT)
            
            # Wait for dialog to close
            make_entry.focus_set()
            self.root.wait_window(dialog)
            
            # Get values
            make = make_var.get().strip() or None
            color = color_var.get().strip() or None
        
        # Add plate to allowlist
        success = self.plate_checker.add_plate(plate, make, color)
        
        if success:
            attrs_str = ""
            if make:
                attrs_str += f" with make={make}"
            if color:
                attrs_str += f" with color={color}"
            
            self.logger.info(f"Added plate to allowlist: {plate}{attrs_str}")
            self.status_bar.config(text=f"Added plate to allowlist: {plate}{attrs_str}")
            self.new_plate_var.set("")  # Clear entry
            self.update_allowlist_display()
        else:
            error_msg = f"Failed to add plate to allowlist: {plate}"
            self.logger.error(error_msg)
            self.status_bar.config(text=error_msg)
            messagebox.showerror("Allowlist Error", error_msg)
    
    def remove_plate(self):
        """Remove plate from allowlist"""
        plate = self.new_plate_var.get().strip()
        
        if not plate:
            messagebox.showwarning("Input Error", "Please enter a license plate")
            return
        
        success = self.plate_checker.remove_plate(plate)
        
        if success:
            self.logger.info(f"Removed plate from allowlist: {plate}")
            self.status_bar.config(text=f"Removed plate from allowlist: {plate}")
            self.new_plate_var.set("")  # Clear entry
            self.update_allowlist_display()
        else:
            error_msg = f"Failed to remove plate from allowlist (not found): {plate}"
            self.logger.warning(error_msg)
            self.status_bar.config(text=error_msg)
            messagebox.showwarning("Allowlist Warning", error_msg)
    
    def reload_allowlist(self):
        """Reload allowlist from file"""
        self.plate_checker._load_allowlist()
        self.update_allowlist_display()
        self.status_bar.config(text="Allowlist reloaded")
    
    def clear_logs(self):
        """Clear log display"""
        self.log_text.delete(1.0, tk.END)
        self.status_bar.config(text="Logs cleared")
    
    def save_plate_image(self, plate_img, plate_text, is_authorized, confidence, make=None, color=None):
        """Save plate image to captures directory"""
        try:
            # Create captures directory if it doesn't exist
            captures_dir = self.config['captures_dir']
            os.makedirs(captures_dir, exist_ok=True)
            
            # Format filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            status = "authorized" if is_authorized else "unauthorized"
            
            # Add make and color to filename if available
            make_str = f"_{make}" if make else ""
            color_str = f"_{color}" if color else ""
            
            filename = f"{timestamp}_{plate_text}{make_str}{color_str}_{status}_{confidence:.2f}.jpg"
            filepath = os.path.join(captures_dir, filename)
            
            # Save image
            cv2.imwrite(filepath, plate_img)
            
        except Exception as e:
            self.logger.error(f"Error saving plate image: {str(e)}")
    
    def on_closing(self):
        """Handle window close event"""
        global running
        
        # Ask user to confirm
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            # Set flag to stop threads
            running = False
            
            # Cleanup resources
            self.logger.info("Shutting down GUI application...")
            
            try:
                # Release camera
                if hasattr(self, 'camera'):
                    self.camera.release()
                
                # Release detector
                if hasattr(self, 'detector'):
                    self.detector.release()
                
                # Release vehicle classifier
                if hasattr(self, 'vehicle_classifier') and self.vehicle_classifier is not None:
                    self.vehicle_classifier.release()
                
                # Release gate controller
                if hasattr(self, 'gate_controller'):
                    self.gate_controller.cleanup()
                
                # Stop plate checker
                if hasattr(self, 'plate_checker'):
                    self.plate_checker.stop()
                
            except Exception as e:
                self.logger.error(f"Error during cleanup: {str(e)}")
            
            # Close window
            self.root.destroy()
            sys.exit(0)


class LogQueueHandler(logging.Handler):
    """Handler for redirecting logs to a queue for GUI display"""
    
    def __init__(self, log_queue):
        super().__init__()
        self.log_queue = log_queue
        self.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
    
    def emit(self, record):
        """Put log record in queue"""
        try:
            self.log_queue.put(record)
        except queue.Full:
            pass


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
    import argparse
    
    parser = argparse.ArgumentParser(description='ANPR System GUI for Orange Pi 5 Ultra')
    
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
    """Main entry point for GUI application"""
    global running
    running = True
    
    try:
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
        
        if args.debug:
            config['debug'] = True
        
        if args.mock:
            config['mock_mode'] = True
        
        # Create root window
        root = tk.Tk()
        
        # Create application
        app = ANPRGuiApp(root, config)
        
        # Start main loop
        root.mainloop()
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()