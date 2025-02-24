"""Error logging and tracking system for Document Q&A."""
import logging
from pathlib import Path
from datetime import datetime
import json
from typing import Any, Dict, TypedDict


class ErrorStats(TypedDict):
    total_errors: int
    error_types: Dict[str, int]
    last_updated: str


class ErrorLogger:
    def __init__(self, log_dir: str = "logs") -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Set up file handler
        self.logger = logging.getLogger("document_qa")
        self.logger.setLevel(logging.ERROR)
        
        # Create handlers
        self._setup_file_handler()
        self._setup_error_tracking()

    def _setup_file_handler(self) -> None:
        """Set up file handler for logging."""
        log_file = self.log_dir / f"errors_{datetime.now():%Y%m%d}.log"
        handler = logging.FileHandler(log_file)
        handler.setLevel(logging.ERROR)
        
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def _setup_error_tracking(self) -> None:
        """Set up error tracking file."""
        self.error_file = self.log_dir / "error_tracking.json"
        if not self.error_file.exists():
            initial_stats: ErrorStats = {
                "total_errors": 0,
                "error_types": {},
                "last_updated": str(datetime.now())
            }
            self._save_error_tracking(initial_stats)

    def log_error(
        self,
        error: Exception,
        context: Dict[str, Any],
        source: str
    ) -> None:
        """Log an error with context."""
        error_type = type(error).__name__
        error_msg = str(error)
        
        # Log to file
        self.logger.error(
            "Error in %s: %s - %s",
            source, error_type, error_msg,
            extra={"context": context}
        )
        
        # Update error tracking
        self._update_error_tracking(error_type)

    def _update_error_tracking(self, error_type: str) -> None:
        """Update error tracking statistics."""
        tracking = self._load_error_tracking()
        
        tracking["total_errors"] += 1
        tracking["error_types"][error_type] = (
            tracking["error_types"].get(error_type, 0) + 1
        )
        tracking["last_updated"] = str(datetime.now())
        
        self._save_error_tracking(tracking)

    def _load_error_tracking(self) -> ErrorStats:
        """Load error tracking data."""
        with open(self.error_file, encoding="utf-8") as f:
            data: ErrorStats = json.load(f)
        return data

    def _save_error_tracking(self, data: ErrorStats) -> None:
        """Save error tracking data."""
        with open(self.error_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def get_error_summary(self) -> ErrorStats:
        """Get summary of errors."""
        return self._load_error_tracking()


# Global error logger instance
error_logger = ErrorLogger() 