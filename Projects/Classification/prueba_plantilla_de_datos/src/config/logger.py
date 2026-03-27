"""
Configuration module for application logging.

This module sets up a centralized logging configuration that can be imported
and used across the application. It configures both console and file handlers
with appropriate formatting and log levels.
"""
import os
from datetime import datetime
from typing import Optional

import logging
from logging.handlers import RotatingFileHandler

# Constants
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s - %(name)s - %(funcName)s'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
MAX_LOG_SIZE = 5 * 1024 * 1024  # 5MB
BACKUP_COUNT = 5

# Ensure logs directory exists
LOG_DIR = 'logs'

os.makedirs(LOG_DIR, exist_ok=True)

# Create a custom logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# Create formatter
formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

try:
    # Console handler (shows INFO and above)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # File handler (shows DEBUG and above, rotates when reaches MAX_LOG_SIZE)
    log_file = os.path.join(LOG_DIR, f'app_{datetime.now().strftime("%Y%m%d")}.log')
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=MAX_LOG_SIZE,
        backupCount=BACKUP_COUNT,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

except Exception as e:
    logging.error("Failed to configure logging handlers: %s", e)
    raise

# Add handlers to the logger
if not logger.handlers:  # Avoid adding handlers multiple times
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get a logger with the specified name.
    
    Args:
        name: Name of the logger. If None, returns the root logger.
              Defaults to None.
    
    Returns:
        Configured logger instance.
    """
    if name is not None and not isinstance(name, str):
        raise TypeError("Logger name must be a string or None")

    return logging.getLogger(name)
