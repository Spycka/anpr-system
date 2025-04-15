#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GPIO Control Module for Gate Operation
Controls the gate relay through GPIO pins

This module handles the activation of the gate relay through
GPIO pins on the Orange Pi 5 Ultra.
"""

import time
import threading
import logging

class GateController:
    """
    Gate Controller class for managing gate operation
    through GPIO pins with safety features
    """
    
    def __init__(self, pin=17, pulse_time=1.0, cooldown_time=5.0, mock=False):
        """
        Initialize the gate controller
        
        Args:
            pin (int): GPIO pin number (SUNXI numbering)
            pulse_time (float): Duration in seconds to keep output HIGH
            cooldown_time (float): Minimum time between successive activations
            mock (bool): Run in mock mode without hardware
        """
        self.logger = logging.getLogger('system')
        self.pin = pin
        self.pulse_time = pulse_time
        self.cooldown_time = cooldown_time
        self.mock_mode = mock
        
        # State tracking
        self.last_activation = 0
        self.is_active = False
        
        # Threading
        self.lock = threading.Lock()
        
        # Initialize GPIO
        if not self.mock_mode:
            self._init_gpio()
        else:
            self.logger.warning("Running in MOCK MODE - gate control will be simulated")
    
    def _init_gpio(self):
        """Initialize GPIO pin for gate control"""
        try:
            import OPi.GPIO as GPIO
            
            # Store GPIO module for later use
            self.GPIO = GPIO
            
            # Set GPIO mode for Orange Pi
            GPIO.setmode(GPIO.SUNXI)
            
            # Configure pin as output, initially LOW
            GPIO.setup(self.pin, GPIO.OUT, initial=GPIO.LOW)
            
            self.logger.info(f"GPIO initialized successfully on pin {self.pin}")
            
        except ImportError:
            self.logger.error("OPi.GPIO module not installed, falling back to mock mode")
            self.mock_mode = True
        except Exception as e:
            self.logger.error(f"Error initializing GPIO: {str(e)}")
            self.logger.warning("Falling back to MOCK MODE")
            self.mock_mode = True
    
    def _gate_pulse_thread(self):
        """Background thread for controlling gate pulse timing"""
        try:
            self.logger.info(f"Activating gate relay on pin {self.pin}")
            
            # Set the output to HIGH
            if not self.mock_mode:
                self.GPIO.output(self.pin, self.GPIO.HIGH)
            
            # Wait for pulse duration
            time.sleep(self.pulse_time)
            
            # Set the output back to LOW
            if not self.mock_mode:
                self.GPIO.output(self.pin, self.GPIO.LOW)
            
            self.logger.info("Gate relay deactivated")
            
            # Update state with lock
            with self.lock:
                self.is_active = False
            
        except Exception as e:
            self.logger.error(f"Error in gate pulse thread: {str(e)}")
            
            # Make sure to set output LOW in case of error
            if not self.mock_mode:
                try:
                    self.GPIO.output(self.pin, self.GPIO.LOW)
                except:
                    pass
            
            # Update state with lock
            with self.lock:
                self.is_active = False
    
    def open_gate(self):
        """
        Activate the gate relay to open the gate
        
        Returns:
            bool: True if gate was activated, False if on cooldown or already active
        """
        current_time = time.time()
        
        # Check if on cooldown with lock
        with self.lock:
            if self.is_active:
                self.logger.debug("Gate already active, ignoring request")
                return False
            
            elapsed = current_time - self.last_activation
            if elapsed < self.cooldown_time:
                self.logger.debug(f"Gate on cooldown, ignoring request ({elapsed:.1f}s < {self.cooldown_time:.1f}s)")
                return False
            
            # Update state
            self.is_active = True
            self.last_activation = current_time
        
        if self.mock_mode:
            self.logger.info("MOCK: Gate would be opened now")
            
            # Start a thread to simulate gate pulse
            thread = threading.Thread(
                target=self._gate_pulse_thread,
                daemon=True,
                name="GatePulse"
            )
            thread.start()
            
        else:
            # Start a thread to handle the pulse timing
            thread = threading.Thread(
                target=self._gate_pulse_thread,
                daemon=True,
                name="GatePulse"
            )
            thread.start()
        
        return True
    
    def cleanup(self):
        """Clean up GPIO resources"""
        if not self.mock_mode and hasattr(self, 'GPIO'):
            try:
                # Set output low
                self.GPIO.output(self.pin, self.GPIO.LOW)
                
                # Clean up GPIO
                self.GPIO.cleanup(self.pin)
                self.logger.info("GPIO resources cleaned up")
            except Exception as e:
                self.logger.error(f"Error cleaning up GPIO: {str(e)}")
