#!/usr/bin/env python3
"""
Logging module for the ANPR system.
Provides rotating log files for different components.
"""
import os
import sys
import logging
import logging.handlers
from typing import Dict, Optional
from pathlib import Path

class LoggerManager:
    """
    Manager for multiple rotating log files.
    Creates standardized loggers for different system components.
    """
    
    # Default log directory (relative to project root)
    DEFAULT_LOG_DIR = "logs"
    
    # Default log levels for different components
    DEFAULT_LOG_LEVELS = {
        "detection": logging.INFO,
        "ocr": logging.INFO,
        "plate_checker": logging.INFO,
        "camera": logging.INFO,
        "hardware": logging.INFO,
        "main": logging.INFO,
        "vehicle_make": logging.INFO,
        "vehicle_color": logging.INFO,
        "gpio": logging.INFO
    }
    
    # Default log rotation settings
    MAX_LOG_SIZE_BYTES = 5 * 1024 * 1024  # 5 MB
    BACKUP_COUNT = 5  # Keep 5 backup log files
    
    def __init__(self, log_dir: Optional[str] = None, log_to_console: bool = True):
        """
        Initialize the logger manager.
        
        Args:
            log_dir: Directory to store log files. If None, uses DEFAULT_LOG_DIR
            log_to_console: Whether to log to console in addition to files
        """
        # Set log directory
        self.log_dir = log_dir or os.path.join(os.getcwd(), self.DEFAULT_LOG_DIR)
        self.log_to_console = log_to_console
        
        # Create log directory if it doesn't exist
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Dictionary to store created loggers
        self.loggers = {}
        
        # Set up console handler if requested
        self.console_handler = None
        if log_to_console:
            self.console_handler = logging.StreamHandler(sys.stdout)
            self.console_handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
    
    def get_logger(self, component: str, level: Optional[int] = None) -> logging.Logger:
        """
        Get or create a logger for a specific component.
        
        Args:
            component: Name of the component (detection, ocr, etc.)
            level: Log level (if None, uses DEFAULT_LOG_LEVELS for the component)
            
        Returns:
            logging.Logger: Configured logger instance
        """
        # Return existing logger if already created
        if component in self.loggers:
            return self.loggers[component]
        
        # Create a new logger
        logger = logging.getLogger(component)
        
        # Set log level
        log_level = level or self.DEFAULT_LOG_LEVELS.get(component, logging.INFO)
        logger.setLevel(log_level)
        
        # Clear any existing handlers
        logger.handlers = []
        
        # Create log file path
        log_file = os.path.join(self.log_dir, f"{component}.log")
        
        # Create rotating file handler
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=self.MAX_LOG_SIZE_BYTES,
            backupCount=self.BACKUP_COUNT
        )
        
        # Set formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # Add file handler to logger
        logger.addHandler(file_handler)
        
        # Add console handler if enabled
        if self.log_to_console and self.console_handler:
            logger.addHandler(self.console_handler)
        
        # Store in dictionary
        self.loggers[component] = logger
        
        return logger
    
    def set_log_level(self, component: str, level: int) -> None:
        """
        Set the log level for a specific component.
        
        Args:
            component: Name of the component
            level: Log level (logging.DEBUG, logging.INFO, etc.)
        """
        if component in self.loggers:
            self.loggers[component].setLevel(level)
        else:
            # Create logger with specified level
            self.get_logger(component, level)
    
    def set_all_log_levels(self, level: int) -> None:
        """
        Set the log level for all components.
        
        Args:
            level: Log level (logging.DEBUG, logging.INFO, etc.)
        """
        for component in self.DEFAULT_LOG_LEVELS.keys():
            self.set_log_level(component, level)

# Module-level logger manager instance
logger_manager = None

def setup_logging(log_dir: Optional[str] = None, log_to_console: bool = True) -> LoggerManager:
    """
    Set up logging system with the specified settings.
    
    Args:
        log_dir: Directory to store log files
        log_to_console: Whether to log to console
        
    Returns:
        LoggerManager: Logger manager instance
    """
    global logger_manager
    logger_manager = LoggerManager(log_dir, log_to_console)
    return logger_manager

def get_logger(component: str) -> logging.Logger:
    """
    Get a logger for the specified component.
    
    Args:
        component: Name of the component
        
    Returns:
        logging.Logger: Logger instance
    """
    global logger_manager
    if logger_manager is None:
        # Auto-create logger manager with defaults if not explicitly set up
        logger_manager = LoggerManager()
    
    return logger_manager.get_logger(component)

# Testing code
if __name__ == "__main__":
    # Set up logging
    log_manager = setup_logging(log_to_console=True)
    
    # Get loggers for different components
    detection_logger = get_logger("detection")
    ocr_logger = get_logger("ocr")
    plate_checker_logger = get_logger("plate_checker")
    
    # Test logging
    detection_logger.debug("Debug message from detection")
    detection_logger.info("Info message from detection")
    detection_logger.warning("Warning message from detection")
    
    ocr_logger.info("Info message from OCR")
    plate_checker_logger.info("Info message from plate checker")
    
    print("Logging test complete. Check log files in the 'logs' directory.")
