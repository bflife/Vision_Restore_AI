"""
Vision-Restore AI - Logging Infrastructure

Provides consistent logging across the application with file and console output.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


# Custom formatter with colors for console output
class ColoredFormatter(logging.Formatter):
    """Colored log formatter for console output."""

    # ANSI color codes
    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[35m",  # Magenta
        "RESET": "\033[0m",      # Reset
    }

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors."""
        # Get color for this level
        color = self.COLORS.get(record.levelname, self.COLORS["RESET"])
        reset = self.COLORS["RESET"]

        # Format the message
        record.levelname = f"{color}{record.levelname}{reset}"
        return super().format(record)


class LoggerManager:
    """Manages logging configuration for the application."""

    _instance: Optional["LoggerManager"] = None
    _initialized: bool = False

    def __new__(cls) -> "LoggerManager":
        """Singleton pattern for logger manager."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize the logger manager."""
        if LoggerManager._initialized:
            return
        
        LoggerManager._initialized = True
        
        # Create logs directory
        self.logs_dir = Path.home() / ".vision-restore" / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Log file path with timestamp
        timestamp = datetime.now().strftime("%Y%m%d")
        self.log_file = self.logs_dir / f"vision_restore_{timestamp}.log"
        
        # Configure root logger
        self._configure_root_logger()

    def _configure_root_logger(self) -> None:
        """Configure the root logger with console and file handlers."""
        # Get root logger for our application
        root_logger = logging.getLogger("vision_restore")
        root_logger.setLevel(logging.DEBUG)
        
        # Remove existing handlers
        root_logger.handlers.clear()
        
        # Console handler with colors
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_format = "%(asctime)s │ %(levelname)s │ %(message)s"
        console_handler.setFormatter(ColoredFormatter(console_format, datefmt="%H:%M:%S"))
        root_logger.addHandler(console_handler)
        
        # File handler for detailed logging
        file_handler = logging.FileHandler(self.log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_format = "%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s"
        file_handler.setFormatter(logging.Formatter(file_format, datefmt="%Y-%m-%d %H:%M:%S"))
        root_logger.addHandler(file_handler)
        
        # Log startup message
        root_logger.info("=" * 60)
        root_logger.info("Vision-Restore AI - Session Started")
        root_logger.info("=" * 60)

    def get_logger(self, name: str) -> logging.Logger:
        """Get a logger with the given name."""
        if name.startswith("vision_restore"):
            return logging.getLogger(name)
        return logging.getLogger(f"vision_restore.{name}")


# Module-level function for easy access
def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name.
    
    Args:
        name: Module name or logger identifier
        
    Returns:
        Configured logger instance
    """
    manager = LoggerManager()
    return manager.get_logger(name)


def log_exception(logger: logging.Logger, message: str, exc: Exception) -> None:
    """Log an exception with full traceback.
    
    Args:
        logger: Logger instance
        message: Error message
        exc: Exception to log
    """
    logger.error(f"{message}: {exc}", exc_info=True)


def log_progress(
    logger: logging.Logger,
    current: int,
    total: int,
    message: str = "",
    bar_width: int = 40,
) -> None:
    """Log a progress bar.
    
    Args:
        logger: Logger instance
        current: Current progress value
        total: Total value
        message: Optional message to display
        bar_width: Width of the progress bar
    """
    if total == 0:
        return
    
    progress = current / total
    filled = int(bar_width * progress)
    bar = "█" * filled + "░" * (bar_width - filled)
    
    logger.info(f"[{bar}] {progress*100:.1f}% {message}")
