import logging, sys, json
from typing import Any

class JsonFormatter(logging.Formatter):
    def format(self, record):
        base = {
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
        }
        if record.exc_info:
            base["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(base)

def setup_logging():
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    root.addHandler(h)

def log_timing(logger: logging.Logger, operation: str, duration: float, **kwargs: Any) -> None:
    """Log timing information for operations."""
    extra_info = ", ".join([f"{k}={v}" for k, v in kwargs.items()])
    logger.info(f"Operation completed: {operation}, duration_seconds={duration}, {extra_info}")

def log_error(logger: logging.Logger, error: Exception, context: str = "", **kwargs: Any) -> None:
    """Log error with context information."""
    extra_info = ", ".join([f"{k}={v}" for k, v in kwargs.items()])
    logger.error(f"Error in {context}: {str(error)}, error_type={type(error).__name__}, {extra_info}", exc_info=True)

setup_logging()
