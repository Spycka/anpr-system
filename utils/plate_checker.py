#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
License Plate Checker Module
Verifies license plates against an allowlist

This module handles the verification of detected license plates
against an allowlist of authorized vehicles.
"""

import os
import threading
import time
import logging

class PlateChecker:
    """
    License Plate Checker class for allowlist verification
    with automatic reloading support
    """
    
    def __init__(self, allowlist_path, auto_reload=True, reload_interval=60.0):
        """
        Initialize the plate checker
        
        Args:
            allowlist_path (str): Path to allowlist file
            auto_reload (bool): Automatically reload allowlist on changes
            reload_interval (float): Seconds between checking for changes
        """
        self.logger = logging.getLogger('plate_checker')
        self.allowlist_path = allowlist_path
        self.auto_reload = auto_reload
        self.reload_interval = reload_interval
        
        # Allowlist storage
        self.allowlist = set()
        self.last_modified_time = 0
        self.last_reload_time = 0
        
        # Threading
        self.lock = threading.Lock()
        self.running = False
        self.reload_thread = None
        
        # Initial load
        self._load_allowlist()
        
        # Start auto-reload thread if enabled
        if self.auto_reload:
            self._start_auto_reload()
    
    def _load_allowlist(self):
        """Load the allowlist from file"""
        try:
            # Check if file exists
            if not os.path.exists(self.allowlist_path):
                self.logger.warning(f"Allowlist file not found: {self.allowlist_path}")
                self.logger.info("Creating empty allowlist file")
                with open(self.allowlist_path, 'w') as f:
                    f.write("# License Plate Allowlist\n")
                    f.write("# Each plate should be on a separate line\n")
                    f.write("# Lines starting with # are ignored\n\n")
            
            # Read allowlist file
            with open(self.allowlist_path, 'r') as f:
                lines = f.readlines()
            
            # Process lines
            new_allowlist = set()
            for line in lines:
                # Skip comments and empty lines
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                # Normalize plate and add to set
                plate = self._normalize_plate(line)
                if plate:
                    new_allowlist.add(plate)
            
            # Update allowlist with lock
            with self.lock:
                old_count = len(self.allowlist)
                self.allowlist = new_allowlist
                new_count = len(self.allowlist)
            
            # Update modification time
            self.last_modified_time = os.path.getmtime(self.allowlist_path)
            self.last_reload_time = time.time()
            
            self.logger.info(f"Loaded allowlist: {new_count} plates (changed from {old_count})")
            
        except Exception as e:
            self.logger.error(f"Error loading allowlist: {str(e)}")
    
    def _normalize_plate(self, plate):
        """
        Normalize plate format for consistent matching
        
        Args:
            plate (str): License plate text
            
        Returns:
            Normalized plate text
        """
        if not plate:
            return ""
        
        # Remove whitespace and convert to uppercase
        return plate.strip().upper()
    
    def _start_auto_reload(self):
        """Start background thread for auto-reloading allowlist"""
        if self.reload_thread is not None and self.reload_thread.is_alive():
            return  # Already running
        
        self.running = True
        self.reload_thread = threading.Thread(
            target=self._auto_reload_thread,
            daemon=True,
            name="AllowlistReload"
        )
        self.reload_thread.start()
        self.logger.info("Allowlist auto-reload thread started")
    
    def _auto_reload_thread(self):
        """Background thread for checking allowlist file changes"""
        while self.running:
            try:
                # Check if file exists
                if os.path.exists(self.allowlist_path):
                    # Get modification time
                    mtime = os.path.getmtime(self.allowlist_path)
                    
                    # Reload if file was modified
                    if mtime > self.last_modified_time:
                        self.logger.info("Allowlist file modified, reloading...")
                        self._load_allowlist()
                
                # Sleep for interval
                time.sleep(self.reload_interval)
                
            except Exception as e:
                self.logger.error(f"Error in allowlist reload thread: {str(e)}")
                time.sleep(self.reload_interval)
    
    def check_plate(self, plate_text):
        """
        Check if a plate is in the allowlist
        
        Args:
            plate_text (str): License plate text to check
            
        Returns:
            bool: True if plate is authorized, False otherwise
        """
        # Normalize input plate
        plate = self._normalize_plate(plate_text)
        
        if not plate:
            return False
        
        # Check against allowlist with lock
        with self.lock:
            is_authorized = plate in self.allowlist
        
        # Log the check
        status = "AUTHORIZED" if is_authorized else "DENIED"
        self.logger.info(f"Plate check: {plate} - {status}")
        
        return is_authorized
    
    def add_plate(self, plate_text):
        """
        Add a plate to the allowlist
        
        Args:
            plate_text (str): License plate to add
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Normalize input plate
        plate = self._normalize_plate(plate_text)
        
        if not plate:
            return False
        
        try:
            # Add to allowlist with lock
            with self.lock:
                self.allowlist.add(plate)
            
            # Append to file
            with open(self.allowlist_path, 'a') as f:
                f.write(f"\n{plate}")
            
            self.logger.info(f"Plate added to allowlist: {plate}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding plate to allowlist: {str(e)}")
            return False
    
    def remove_plate(self, plate_text):
        """
        Remove a plate from the allowlist
        
        Args:
            plate_text (str): License plate to remove
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Normalize input plate
        plate = self._normalize_plate(plate_text)
        
        if not plate:
            return False
        
        try:
            # Remove from allowlist with lock
            with self.lock:
                if plate in self.allowlist:
                    self.allowlist.remove(plate)
                else:
                    # Plate not in allowlist
                    return False
            
            # Rewrite file with updated allowlist
            with open(self.allowlist_path, 'r') as f:
                lines = f.readlines()
            
            with open(self.allowlist_path, 'w') as f:
                for line in lines:
                    # Skip the plate we're removing
                    line_plate = self._normalize_plate(line)
                    if line_plate and line_plate == plate:
                        continue
                    
                    # Write all other lines
                    f.write(line)
            
            self.logger.info(f"Plate removed from allowlist: {plate}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error removing plate from allowlist: {str(e)}")
            return False
    
    def get_allowlist(self):
        """
        Get the current allowlist
        
        Returns:
            list: Current allowlist of plates
        """
        with self.lock:
            return sorted(list(self.allowlist))
    
    def stop(self):
        """Stop background thread and release resources"""
        self.running = False
        
        if self.reload_thread is not None and self.reload_thread.is_alive():
            self.reload_thread.join(timeout=1.0)
        
        self.logger.info("Plate checker stopped")
