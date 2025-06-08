"""
FIXED logging configuration to prevent Unicode encoding errors
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
    """Set up logger with console and file handlers - FIXED encoding issues"""

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, config.log_level.upper()))

    # Remove existing handlers
    logger.handlers.clear()

    # FIXED: Console handler with UTF-8 encoding to prevent Unicode errors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setEncoding('utf-8')  # FIXED: Explicit UTF-8 encoding
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # FIXED: File handler with UTF-8 encoding
    file_handler = logging.FileHandler(
        LOG_DIR / f"app_{datetime.now().strftime('%Y%m%d')}.log",
        encoding='utf-8'  # FIXED: Explicit UTF-8 encoding
    )
    file_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
    )
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)

    return logger


# Create default logger
logger = setup_logger()