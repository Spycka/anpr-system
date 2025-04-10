#!/usr/bin/env python3
"""
Plate checker module for ANPR system.
Validates detected plates against allowlist and triggers GPIO when matched.
"""
import os
import re
import json
import time
import threading
from typing import List, Dict, Any, Optional, Set, Tuple, Union
import logging

# Import project modules
from utils.logger import get_logger
from utils.gpio import get_gpio, GPIOMode, GPIOState

# Configure logger
logger = get_logger("plate_checker")

class PlateChecker:
    """
    License plate validation and GPIO control.
    
    Features:
    - Validate plates against allowlist.txt
    - Optional make/color verification from allowlist.json
    - Trigger GPIO pin 17 for 1 second on match
    - Rate limiting to prevent repeated triggers
    """
    
    # Default settings
    DEFAULT_ALLOWLIST_PATH = "allowlist.txt"
    DEFAULT_JSON_ALLOWLIST_PATH = "allowlist.json"
    DEFAULT_GPIO_PIN = 17
    DEFAULT_PULSE_DURATION = 1.0  # seconds
    DEFAULT_COOLDOWN = 3.0  # seconds
    
    def __init__(
        self,
        allowlist_path: str = DEFAULT_ALLOWLIST_PATH,
        gpio_pin: int = DEFAULT_GPIO_PIN,
        pulse_duration: float = DEFAULT_PULSE_DURATION,
        cooldown: float = DEFAULT_COOLDOWN,
        use_gpio: bool = True,
        mock_gpio: bool = False,
        enable_make_verification: bool = False,
        enable_color_verification: bool = False
    ):
        """
        Initialize plate checker.
        
        Args:
            allowlist_path: Path to allowlist file (txt or json)
            gpio_pin: GPIO pin to trigger on match
            pulse_duration: Duration of GPIO pulse (seconds)
            cooldown: Minimum time between triggers (seconds)
            use_gpio: Whether to use GPIO triggering
            mock_gpio: Whether to use mock GPIO (for testing)
            enable_make_verification: Whether to verify vehicle make
            enable_color_verification: Whether to verify vehicle color
        """
        self.allowlist_path = allowlist_path
        self.gpio_pin = gpio_pin
        self.pulse_duration = pulse_duration
        self.cooldown = cooldown
        self.use_gpio = use_gpio
        self.mock_gpio = mock_gpio
        self.enable_make_verification = enable_make_verification
        self.enable_color_verification = enable_color_verification
        
        # Allowlist storage
        self.simple_allowlist: Set[str] = set()
        self.json_allowlist: Dict[str, Dict[str, Any]] = {}
        
        # Keep track of last trigger time for rate limiting
        self.last_trigger_time = 0
        self.trigger_lock = threading.Lock()
        
        # Initialize GPIO
        self.gpio = None
        if use_gpio:
            self.gpio = get_gpio(mock_mode=mock_gpio)
            try:
                self.gpio.setup(gpio_pin, GPIOMode.OUT)
                self.gpio.output(gpio_pin, GPIOState.LOW)
                logger.info(f"GPIO pin {gpio_pin} initialized for plate triggers")
            except Exception as e:
                logger.error(f"Failed to initialize GPIO pin {gpio_pin}: {str(e)}")
                self.use_gpio = False
        
        # Load allowlist
        self._load_allowlist()
        
        logger.info(f"Plate checker initialized with {len(self.simple_allowlist)} plates "
                   f"(use_gpio={use_gpio}, mock_gpio={mock_gpio})")
    
    def _load_allowlist(self) -> None:
        """Load allowlist from file (txt or json)."""
        # Reset allowlists
        self.simple_allowlist = set()
        self.json_allowlist = {}
        
        # Check if allowlist file exists
        if not os.path.isfile(self.allowlist_path):
            logger.warning(f"Allowlist file not found: {self.allowlist_path}")
            return
        
        # Load based on file extension
        try:
            if self.allowlist_path.lower().endswith('.json'):
                # Load JSON allowlist
                with open(self.allowlist_path, 'r') as f:
                    self.json_allowlist = json.load(f)
                
                # Extract plate numbers for simple lookups
                for plate, details in self.json_allowlist.items():
                    self.simple_allowlist.add(self._normalize_plate(plate))
                
                logger.info(f"Loaded JSON allowlist with {len(self.json_allowlist)} entries "
                          f"from {self.allowlist_path}")
            else:
                # Load simple text allowlist
                with open(self.allowlist_path, 'r') as f:
                    for line in f:
                        # Skip empty lines and comments
                        line = line.strip()
                        if line and not line.startswith('#'):
                            self.simple_allowlist.add(self._normalize_plate(line))
                
                logger.info(f"Loaded text allowlist with {len(self.simple_allowlist)} entries "
                          f"from {self.allowlist_path}")
                
                # Check if JSON allowlist exists as well
                json_path = self.DEFAULT_JSON_ALLOWLIST_PATH
                if os.path.isfile(json_path):
                    try:
                        with open(json_path, 'r') as f:
                            self.json_allowlist = json.load(f)
                        logger.info(f"Additionally loaded JSON allowlist with "
                                   f"{len(self.json_allowlist)} entries from {json_path}")
                    except Exception as e:
                        logger.error(f"Failed to load JSON allowlist from {json_path}: {str(e)}")
        
        except Exception as e:
            logger.error(f"Failed to load allowlist from {self.allowlist_path}: {str(e)}")
    
    def reload_allowlist(self) -> bool:
        """
        Reload allowlist from file.
        
        Returns:
            bool: True if reload successful, False otherwise
        """
        try:
            self._load_allowlist()
            return True
        except Exception as e:
            logger.error(f"Failed to reload allowlist: {str(e)}")
            return False
    
    def add_to_allowlist(self, plate: str, 
                         details: Optional[Dict[str, Any]] = None,
                         save_to_file: bool = True) -> bool:
        """
        Add a plate to the allowlist.
        
        Args:
            plate: License plate number
            details: Optional details dict (for JSON allowlist)
            save_to_file: Whether to save changes to file
            
        Returns:
            bool: True if addition successful, False otherwise
        """
        try:
            # Normalize plate
            normalized_plate = self._normalize_plate(plate)
            
            # Add to simple allowlist
            self.simple_allowlist.add(normalized_plate)
            
            # Add to JSON allowlist if details provided
            if details is not None:
                self.json_allowlist[normalized_plate] = details
            
            # Save to file if requested
            if save_to_file:
                if details is not None:
                    # Save to JSON allowlist
                    json_path = (self.allowlist_path 
                               if self.allowlist_path.lower().endswith('.json') 
                               else self.DEFAULT_JSON_ALLOWLIST_PATH)
                    
                    with open(json_path, 'w') as f:
                        json.dump(self.json_allowlist, f, indent=2)
                    
                    logger.info(f"Added plate {normalized_plate} to JSON allowlist and saved to {json_path}")
                else:
                    # Save to text allowlist
                    txt_path = (self.allowlist_path 
                              if not self.allowlist_path.lower().endswith('.json') 
                              else self.DEFAULT_ALLOWLIST_PATH)
                    
                    # Read existing content
                    existing_lines = []
                    if os.path.isfile(txt_path):
                        with open(txt_path, 'r') as f:
                            existing_lines = [line.strip() for line in f.readlines()]
                    
                    # Add new plate if not already in file
                    if normalized_plate not in existing_lines:
                        with open(txt_path, 'a') as f:
                            f.write(f"{normalized_plate}\n")
                    
                    logger.info(f"Added plate {normalized_plate} to text allowlist")
            
            return True
        
        except Exception as e:
            logger.error(f"Failed to add plate {plate} to allowlist: {str(e)}")
            return False
    
    def remove_from_allowlist(self, plate: str, save_to_file: bool = True) -> bool:
        """
        Remove a plate from the allowlist.
        
        Args:
            plate: License plate number
            save_to_file: Whether to save changes to file
            
        Returns:
            bool: True if removal successful, False otherwise
        """
        try:
            # Normalize plate
            normalized_plate = self._normalize_plate(plate)
            
            # Remove from simple allowlist
            if normalized_plate in self.simple_allowlist:
                self.simple_allowlist.remove(normalized_plate)
            
            # Remove from JSON allowlist
            if normalized_plate in self.json_allowlist:
                del self.json_allowlist[normalized_plate]
            
            # Save to file if requested
            if save_to_file:
                # Update JSON allowlist
                if self.json_allowlist:
                    json_path = (self.allowlist_path 
                               if self.allowlist_path.lower().endswith('.json') 
                               else self.DEFAULT_JSON_ALLOWLIST_PATH)
                    
                    with open(json_path, 'w') as f:
                        json.dump(self.json_allowlist, f, indent=2)
                
                # Update text allowlist
                txt_path = (self.allowlist_path 
                          if not self.allowlist_path.lower().endswith('.json') 
                          else self.DEFAULT_ALLOWLIST_PATH)
                
                if os.path.isfile(txt_path):
                    # Read existing content
                    with open(txt_path, 'r') as f:
                        lines = [line.strip() for line in f.readlines()]
                    
                    # Remove plate from lines
                    lines = [line for line in lines if self._normalize_plate(line) != normalized_plate]
                    
                    # Write back to file
                    with open(txt_path, 'w') as f:
                        for line in lines:
                            f.write(f"{line}\n")
            
            logger.info(f"Removed plate {normalized_plate} from allowlist")
            return True
        
        except Exception as e:
            logger.error(f"Failed to remove plate {plate} from allowlist: {str(e)}")
            return False
    
    def _normalize_plate(self, plate: str) -> str:
        """
        Normalize license plate string for consistent matching.
        
        Args:
            plate: License plate string
            
        Returns:
            str: Normalized plate string
        """
        if not plate:
            return ""
        
        # Remove spaces and special characters
        normalized = re.sub(r'[^A-Za-z0-9]', '', plate)
        
        # Convert to uppercase
        normalized = normalized.upper()
        
        return normalized
    
    def check_plate(self, plate: str, 
                    make: Optional[str] = None,
                    color: Optional[str] = None,
                    trigger_gpio: bool = True) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if a plate is in the allowlist.
        
        Args:
            plate: License plate to check
            make: Optional vehicle make for verification
            color: Optional vehicle color for verification
            trigger_gpio: Whether to trigger GPIO on match
            
        Returns:
            Tuple[bool, Dict[str, Any]]:
                - Match status (True/False)
                - Additional details (from JSON allowlist if available)
        """
        # Normalize plate
        normalized_plate = self._normalize_plate(plate)
        
        # Check if empty plate
        if not normalized_plate:
            logger.debug("Empty plate, skipping check")
            return False, {}
        
        # Check if plate is in allowlist
        in_allowlist = normalized_plate in self.simple_allowlist
        
        # Get details from JSON allowlist if available
        details = self.json_allowlist.get(normalized_plate, {})
        
        # Verify make if enabled and available
        make_match = True
        if (self.enable_make_verification and make and 
            'make' in details and details['make']):
            # Simple case-insensitive substring match
            allowlist_make = details['make'].lower()
            detected_make = make.lower()
            make_match = allowlist_make in detected_make or detected_make in allowlist_make
            
            if not make_match:
                logger.debug(f"Make mismatch for plate {plate}: expected '{details['make']}', got '{make}'")
        
        # Verify color if enabled and available
        color_match = True
        if (self.enable_color_verification and color and 
            'color' in details and details['color']):
            # Simple case-insensitive substring match
            allowlist_color = details['color'].lower()
            detected_color = color.lower()
            color_match = allowlist_color in detected_color or detected_color in allowlist_color
            
            if not color_match:
                logger.debug(f"Color mismatch for plate {plate}: expected '{details['color']}', got '{color}'")
        
        # Determine overall match
        match = in_allowlist and make_match and color_match
        
        # Log result
        if match:
            logger.info(f"MATCH: Plate {plate} is in allowlist" + 
                       (f" (make={make})" if make else "") + 
                       (f" (color={color})" if color else ""))
        else:
            if in_allowlist:
                logger.info(f"PARTIAL MATCH: Plate {plate} is in allowlist, but " + 
                          (f"make {make} doesn't match, " if not make_match else "") + 
                          (f"color {color} doesn't match, " if not color_match else ""))
            else:
                logger.debug(f"NO MATCH: Plate {plate} not in allowlist")
        
        # Trigger GPIO if matched
        if match and trigger_gpio:
            self._trigger_gpio()
        
        return match, details
    
    def check_plates(self, plates: List[str],
                     makes: Optional[List[str]] = None,
                     colors: Optional[List[str]] = None) -> List[Tuple[bool, Dict[str, Any]]]:
        """
        Check multiple plates against allowlist.
        
        Args:
            plates: List of plates to check
            makes: Optional list of vehicle makes
            colors: Optional list of vehicle colors
            
        Returns:
            List[Tuple[bool, Dict[str, Any]]]: List of match results
        """
        results = []
        
        for i, plate in enumerate(plates):
            make = makes[i] if makes and i < len(makes) else None
            color = colors[i] if colors and i < len(colors) else None
            
            # Only trigger GPIO for the first match
            trigger = (i == 0)
            
            match, details = self.check_plate(plate, make, color, trigger)
            results.append((match, details))
        
        return results
    
    def _trigger_gpio(self) -> bool:
        """
        Trigger GPIO pin (with rate limiting).
        
        Returns:
            bool: True if triggered, False otherwise
        """
        if not self.use_gpio or self.gpio is None:
            logger.debug("GPIO triggering disabled")
            return False
        
        with self.trigger_lock:
            # Check cooldown period
            current_time = time.time()
            time_since_last = current_time - self.last_trigger_time
            
            if time_since_last < self.cooldown:
                logger.debug(f"GPIO trigger cooldown active, {self.cooldown - time_since_last:.1f}s remaining")
                return False
            
            # Update last trigger time
            self.last_trigger_time = current_time
        
        try:
            # Pulse GPIO pin
            logger.info(f"Triggering GPIO pin {self.gpio_pin} for {self.pulse_duration}s")
            self.gpio.pulse(self.gpio_pin, self.pulse_duration)
            return True
        
        except Exception as e:
            logger.error(f"Failed to trigger GPIO: {str(e)}")
            return False
    
    def get_allowlist(self) -> Dict[str, Any]:
        """
        Get current allowlist.
        
        Returns:
            Dict[str, Any]: Dictionary with allowlist information
        """
        return {
            'simple_allowlist': list(self.simple_allowlist),
            'json_allowlist': self.json_allowlist,
            'count': len(self.simple_allowlist)
        }
    
    def __del__(self):
        """Clean up resources when object is destroyed."""
        if self.gpio is not None and self.use_gpio:
            try:
                self.gpio.cleanup(self.gpio_pin)
            except:
                pass

# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create sample allowlist.txt if it doesn't exist
    if not os.path.isfile("allowlist.txt"):
        with open("allowlist.txt", "w") as f:
            f.write("# Sample allowlist\n")
            f.write("ABC123\n")
            f.write("XYZ789\n")
    
    # Create sample allowlist.json if it doesn't exist
    if not os.path.isfile("allowlist.json"):
        sample_allowlist = {
            "ABC123": {
                "make": "Toyota",
                "color": "Blue",
                "owner": "John Doe",
                "notes": "Employee"
            },
            "XYZ789": {
                "make": "Honda",
                "color": "Red",
                "owner": "Jane Smith",
                "notes": "Visitor"
            }
        }
        
        with open("allowlist.json", "w") as f:
            json.dump(sample_allowlist, f, indent=2)
    
    # Create plate checker (use mock GPIO for testing)
    checker = PlateChecker(
        mock_gpio=True,
        enable_make_verification=True,
        enable_color_verification=True
    )
    
    # Test various plates
    test_plates = [
        ("ABC123", "Toyota", "Blue"),  # Perfect match
        ("ABC-123", None, None),      # Match with different format
        ("XYZ789", "Ford", "Red"),    # Make mismatch
        ("XYZ789", "Honda", "Green"), # Color mismatch
        ("DEF456", "BMW", "Black")    # Not in allowlist
    ]
    
    print("\nTesting plate checker:")
    for plate, make, color in test_plates:
        match, details = checker.check_plate(plate, make, color)
        
        if match:
            print(f"✅ {plate}: MATCH" + 
                 (f" (make={make})" if make else "") + 
                 (f" (color={color})" if color else ""))
            if details:
                print(f"   Details: {details}")
        else:
            print(f"❌ {plate}: NO MATCH" + 
                 (f" (make={make})" if make else "") + 
                 (f" (color={color})" if color else ""))
    
    # Test adding and removing
    print("\nAdding new plate:")
    checker.add_to_allowlist("DEF456", {"make": "BMW", "color": "Black"})
    
    match, details = checker.check_plate("DEF456", "BMW", "Black")
    print(f"DEF456 after adding: {'MATCH' if match else 'NO MATCH'}")
    
    print("\nRemoving plate:")
    checker.remove_from_allowlist("DEF456")
    
    match, details = checker.check_plate("DEF456", "BMW", "Black")
    print(f"DEF456 after removing: {'MATCH' if match else 'NO MATCH'}")
