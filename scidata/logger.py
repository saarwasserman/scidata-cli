"""Structured JSON logging for scidata"""

import logging
import json
import warnings
from pathlib import Path
from typing import Any

from scidata.config import settings


class JSONFormatter(logging.Formatter):
    """Format log records as JSON"""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add extra fields if they exist
        if hasattr(record, "extra_fields"):
            log_data.update(record.extra_fields)
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_data)


def get_logger(name: str, log_file: str = "logs/scidata.log", log_level: int = logging.INFO) -> logging.Logger:
    """
    Get a configured logger with JSON output to stdout and file.
    
    Args:
        name: Logger name (typically __name__)
        log_file: Path to log file
        log_level: Logging level
    
    Returns:
        Configured logger instance
    """
    logging.captureWarnings(True)
    logger = logging.getLogger(name)
    
    # Only configure once
    if logger.handlers:
        return logger
    
    logger.setLevel(log_level)
    
    # JSON formatter
    json_formatter = JSONFormatter()
    
    # Stdout handler
    stdout_handler = logging.StreamHandler()
    stdout_handler.setLevel(log_level)
    stdout_handler.setFormatter(json_formatter)
    logger.addHandler(stdout_handler)

    return logger


def setup_warning_logging():
    """Configure Python warnings to be logged through the logging system in JSON format"""
    # Capture warnings through logging
    logging.captureWarnings(True)
    warnings_logger = logging.getLogger("py.warnings")
    
    # Only configure once
    if warnings_logger.handlers:
        return
    
    warnings_logger.setLevel(logging.WARNING)
    
    # JSON formatter
    json_formatter = JSONFormatter()
    
    # Stdout handler for warnings
    stdout_handler = logging.StreamHandler()
    stdout_handler.setLevel(logging.WARNING)
    stdout_handler.setFormatter(json_formatter)
    warnings_logger.addHandler(stdout_handler)

# Root logger for scidata
logger = get_logger("scidata", log_level=settings.log_level)

# Set up warning logging to JSON format
setup_warning_logging()
