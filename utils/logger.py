"""
Fixed logging configuration to handle Unicode characters properly
"""
import logging
import sys
from datetime import datetime
from pathlib import Path
from app.config import config

# Create logs directory if it doesn't exist
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)


def setup_logger(name: str = "ai_cost_optimizer") -> logging.Logger:
    """Set up logger with console and file handlers - FIXED for Unicode support"""

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, config.log_level.upper()))

    # Remove existing handlers
    logger.handlers.clear()

    # FIXED: Console handler with UTF-8 encoding support
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setStream(sys.stdout)

    # Simple format without emojis to prevent encoding issues
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # FIXED: File handler with UTF-8 encoding
    try:
        file_handler = logging.FileHandler(
            LOG_DIR / f"app_{datetime.now().strftime('%Y%m%d')}.log",
            encoding='utf-8'  # FIXED: Explicit UTF-8 encoding
        )
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    except Exception as e:
        # If file logging fails, continue with console only
        print(f"Warning: Could not set up file logging: {e}")

    return logger


# Custom filter to remove problematic Unicode characters
class UnicodeFilter(logging.Filter):
    """Filter to handle Unicode characters in log messages"""

    def filter(self, record):
        if hasattr(record, 'msg') and isinstance(record.msg, str):
            # Replace problematic Unicode characters
            record.msg = record.msg.encode('ascii', 'replace').decode('ascii')
        return True


# Create default logger with Unicode handling
logger = setup_logger()
logger.addFilter(UnicodeFilter())