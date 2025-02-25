"""Development server runner script."""

import os
import logging
from typing import NoReturn

import uvicorn  # type: ignore

from app.core.config import settings


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> NoReturn:
    """Run the development server."""
    logger.info("Starting development server...")
    logger.info("Environment: %s", os.getenv("NODE_ENV", "development"))
    logger.info("API URL: http://localhost:8001")
    logger.info("API Documentation: http://localhost:8001/docs")
    logger.info("Sentry DSN: %s", settings.SENTRY_DSN)
    
    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=8001,
        reload=True,
        log_level="debug"
    )


if __name__ == "__main__":
    main()