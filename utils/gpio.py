#!/usr/bin/env python3
"""
GPIO control module for Orange Pi 5 Ultra.
Handles triggering GPIO pins for gate/barrier control.
"""
import os
import time
import threading
import logging
from typing import Optional, Dict, Any
from enum import Enum

# Import logger
from utils.logger import get_logger
logger = get_logger("gpio")

class GPIOMode(Enum):
    """GPIO pin modes."""
    IN = "in"
    OUT = "out"

class GPIOState(Enum):
    """GPIO pin states."""
    HIGH = 1
    LOW = 0

class GPIO:
    """
    GPIO control for Orange Pi 5 Ultra.
    
    This class provides an abstraction for controlling GPIO pins on the Orange Pi 5 Ultra.
    It supports both physical GPIO access and a mock mode for testing.
    """
    
    # Orange Pi 5 Ultra GPIO path
    GPIO_PATH = "/sys/class/gpio"
    
    # Pin mapping for Orange Pi 5 Ultra
    # Maps logical pin numbers to system GPIO numbers
    PIN_MAPPING = {
        # This mapping may need to be adjusted for Orange Pi 5 Ultra
        # Based on the board's specific GPIO numbering
        17: 17,  # Example: logical pin 17 maps to system GPIO 17
        # Add more mappings as needed for the specific board
    }
    
    def __init__(self, mock_mode: bool = False):
        """
        Initialize GPIO controller.
        
        Args:
            mock_mode: If True, operate in mock mode (for testing without hardware)
        """
        self.mock_mode = mock_mode
        self.exported_pins = set()
        self.pin_states = {}
        
        logger.info(f"GPIO initialized in {'mock' if mock_mode else 'hardware'} mode")
        
        # Map to track active timers for each pin
        self._pin_timers: Dict[int, threading.Timer] = {}
        
        # Check if we're running as root or have proper permissions
        if not self.mock_mode and os.geteuid() != 0:
            logger.warning("GPIO may require root privileges. Some operations might fail.")
    
    def _get_system_gpio_number(self, pin: int) -> int:
        """
        Convert logical pin number to system GPIO number.
        
        Args:
            pin: Logical pin number
            
        Returns:
            int: System GPIO number
            
        Raises:
            ValueError: If pin is not in the mapping
        """
        if pin in self.PIN_MAPPING:
            return self.PIN_MAPPING[pin]
        else:
            raise ValueError(f"Pin {pin} is not in the PIN_MAPPING for Orange Pi 5 Ultra")
    
    def setup(self, pin: int, mode: GPIOMode) -> None:
        """
        Set up a GPIO pin for use.
        
        Args:
            pin: Logical pin number
            mode: Pin mode (IN or OUT)
            
        Raises:
            ValueError: If pin setup fails
        """
        if self.mock_mode:
            # Mock mode - just track the pin state
            self.exported_pins.add(pin)
            self.pin_states[pin] = GPIOState.LOW
            logger.debug(f"Mock GPIO: Pin {pin} set up as {mode.value}")
            return
        
        try:
            # Convert to system GPIO number
            gpio_num = self._get_system_gpio_number(pin)
            
            # Export GPIO if not already exported
            export_path = os.path.join(self.GPIO_PATH, "export")
            if not os.path.exists(os.path.join(self.GPIO_PATH, f"gpio{gpio_num}")):
                with open(export_path, "w") as f:
                    f.write(str(gpio_num))
                time.sleep(0.1)  # Small delay to allow system to set up
            
            # Set direction
            direction_path = os.path.join(self.GPIO_PATH, f"gpio{gpio_num}", "direction")
            with open(direction_path, "w") as f:
                f.write(mode.value)
            
            # Track exported pin
            self.exported_pins.add(pin)
            
            logger.debug(f"GPIO: Pin {pin} (system GPIO {gpio_num}) set up as {mode.value}")
        except Exception as e:
            logger.error(f"Failed to set up GPIO pin {pin}: {str(e)}")
            raise ValueError(f"Failed to set up GPIO pin {pin}: {str(e)}")
    
    def output(self, pin: int, state: GPIOState) -> None:
        """
        Set output state of a GPIO pin.
        
        Args:
            pin: Logical pin number
            state: Pin state (HIGH or LOW)
            
        Raises:
            ValueError: If pin is not set up or operation fails
        """
        if pin not in self.exported_pins:
            raise ValueError(f"Pin {pin} is not set up. Call setup() first.")
        
        if self.mock_mode:
            # Mock mode - just update the state
            self.pin_states[pin] = state
            logger.debug(f"Mock GPIO: Pin {pin} set to {state.name}")
            return
        
        try:
            # Convert to system GPIO number
            gpio_num = self._get_system_gpio_number(pin)
            
            # Set value
            value_path = os.path.join(self.GPIO_PATH, f"gpio{gpio_num}", "value")
            with open(value_path, "w") as f:
                f.write(str(state.value))
            
            logger.debug(f"GPIO: Pin {pin} (system GPIO {gpio_num}) set to {state.name}")
        except Exception as e:
            logger.error(f"Failed to set GPIO pin {pin} to {state.name}: {str(e)}")
            raise ValueError(f"Failed to set GPIO pin {pin} to {state.name}: {str(e)}")
    
    def input(self, pin: int) -> GPIOState:
        """
        Read input state of a GPIO pin.
        
        Args:
            pin: Logical pin number
            
        Returns:
            GPIOState: Current pin state (HIGH or LOW)
            
        Raises:
            ValueError: If pin is not set up or operation fails
        """
        if pin not in self.exported_pins:
            raise ValueError(f"Pin {pin} is not set up. Call setup() first.")
        
        if self.mock_mode:
            # Mock mode - return the tracked state
            return self.pin_states.get(pin, GPIOState.LOW)
        
        try:
            # Convert to system GPIO number
            gpio_num = self._get_system_gpio_number(pin)
            
            # Read value
            value_path = os.path.join(self.GPIO_PATH, f"gpio{gpio_num}", "value")
            with open(value_path, "r") as f:
                value = int(f.read().strip())
            
            return GPIOState.HIGH if value == 1 else GPIOState.LOW
        except Exception as e:
            logger.error(f"Failed to read GPIO pin {pin}: {str(e)}")
            raise ValueError(f"Failed to read GPIO pin {pin}: {str(e)}")
    
    def cleanup(self, pin: Optional[int] = None) -> None:
        """
        Clean up GPIO pins (unexport).
        
        Args:
            pin: Specific pin to clean up, or None for all pins
        """
        if self.mock_mode:
            # Mock mode - just clear the tracking
            if pin is None:
                self.exported_pins.clear()
                self.pin_states.clear()
                logger.debug("Mock GPIO: All pins cleaned up")
            elif pin in self.exported_pins:
                self.exported_pins.remove(pin)
                if pin in self.pin_states:
                    del self.pin_states[pin]
                logger.debug(f"Mock GPIO: Pin {pin} cleaned up")
            return
        
        try:
            # Clean up specific pin or all pins
            if pin is not None:
                pins_to_cleanup = [pin]
            else:
                pins_to_cleanup = list(self.exported_pins)
            
            for p in pins_to_cleanup:
                # Cancel any active timers for this pin
                if p in self._pin_timers and self._pin_timers[p].is_alive():
                    self._pin_timers[p].cancel()
                    del self._pin_timers[p]
                
                # Skip if pin is not in our tracked set
                if p not in self.exported_pins:
                    continue
                
                # Convert to system GPIO number
                gpio_num = self._get_system_gpio_number(p)
                
                # Unexport
                unexport_path = os.path.join(self.GPIO_PATH, "unexport")
                with open(unexport_path, "w") as f:
                    f.write(str(gpio_num))
                
                # Remove from tracking
                self.exported_pins.remove(p)
                logger.debug(f"GPIO: Pin {p} (system GPIO {gpio_num}) cleaned up")
        except Exception as e:
            logger.error(f"Error in GPIO cleanup: {str(e)}")
    
    def pulse(self, pin: int, duration: float = 1.0) -> None:
        """
        Pulse a pin HIGH for a specified duration (in seconds), then back to LOW.
        
        Args:
            pin: Logical pin number
            duration: Pulse duration in seconds
            
        Raises:
            ValueError: If pin is not set up
        """
        if pin not in self.exported_pins:
            raise ValueError(f"Pin {pin} is not set up. Call setup() first.")
        
        # Cancel any existing timer for this pin
        if pin in self._pin_timers and self._pin_timers[pin].is_alive():
            self._pin_timers[pin].cancel()
        
        # Set pin HIGH
        self.output(pin, GPIOState.HIGH)
        logger.info(f"GPIO: Pin {pin} pulsed HIGH for {duration} seconds")
        
        # Create a timer to set pin back to LOW after duration
        timer = threading.Timer(duration, self._pulse_complete, args=[pin])
        timer.daemon = True
        timer.start()
        
        # Store the timer reference
        self._pin_timers[pin] = timer
    
    def _pulse_complete(self, pin: int) -> None:
        """
        Callback for pulse timer completion.
        
        Args:
            pin: Logical pin number
        """
        try:
            self.output(pin, GPIOState.LOW)
            logger.debug(f"GPIO: Pin {pin} pulse completed, set back to LOW")
            
            # Remove timer reference
            if pin in self._pin_timers:
                del self._pin_timers[pin]
        except Exception as e:
            logger.error(f"Error in pulse completion for pin {pin}: {str(e)}")
    
    def __del__(self) -> None:
        """Clean up on object destruction."""
        self.cleanup()

# Singleton instance
_gpio_instance = None

def get_gpio(mock_mode: bool = False) -> GPIO:
    """
    Get the GPIO singleton instance.
    
    Args:
        mock_mode: If True, use mock mode
        
    Returns:
        GPIO: GPIO controller instance
    """
    global _gpio_instance
    if _gpio_instance is None:
        _gpio_instance = GPIO(mock_mode=mock_mode)
    return _gpio_instance

# Example usage
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Test in mock mode
    gpio = get_gpio(mock_mode=True)
    
    # Set up pin 17 as output
    gpio.setup(17, GPIOMode.OUT)
    
    # Pulse pin 17 HIGH for 1 second
    gpio.pulse(17, 1.0)
    
    # Wait for the pulse to complete
    time.sleep(2)
    
    # Clean up
    gpio.cleanup()
    
    print("GPIO test completed")
