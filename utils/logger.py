#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Logging Module for ANPR System
Handles log configuration and rotation

This module provides logging functionality with rotation
to prevent logs from filling up storage.
"""

import os
import logging
from logging.handlers import RotatingFileHandler

def setup_logger(name, log_file, level=logging.INFO, max_size=10*1024*1024, backup_count=5):
    """
    Setup a logger with file rotation
    
    Args:
        name (str): Logger name
        log_file (str): Path to log file
        level: Logging level (default: INFO)
        max_size (int): Maximum log file size in bytes (default: 10MB)
        backup_count (int): Number of backup files to keep (default: 5)
        
    Returns:
        Logger instance
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Create file handler with rotation
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=max_size,
        backupCount=backup_count
    )
    file_handler.setFormatter(formatter)
    
    # Create console handler for debugging
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Get or create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def configure_global_logging(log_dir, debug=False):
    """
    Configure global logging for the entire application
    
    Args:
        log_dir (str): Directory for log files
        debug (bool): Enable debug logging
    """
    # Create log directory
    os.makedirs(log_dir, exist_ok=True)
    
    # Set global logging level
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Suppress verbose logging from libraries
    logging.getLogger('opencv').setLevel(logging.WARNING)
    logging.getLogger('easyocr').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('torch').setLevel(logging.WARNING)
    logging.getLogger('rknn').setLevel(logging.WARNING)
    
    # Create root logger
    setup_logger(
        'root',
        os.path.join(log_dir, 'system.log'),
        level=level
    )
    
    # Log startup message
    logging.info("Logging system initialized")
