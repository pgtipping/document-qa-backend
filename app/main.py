"""Main application module."""

import logging
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration
from sentry_sdk.integrations.logging import LoggingIntegration
from sentry_sdk.integrations.asyncio import AsyncioIntegration
from app.api.routes import router as api_router
from app.core.config import settings

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

logger.debug("Starting application initialization")

# Initialize Sentry
logger.debug("Initializing Sentry")
sentry_sdk.init(
    dsn=settings.SENTRY_DSN,
    enable_tracing=True,
    traces_sample_rate=1.0,
    profiles_sample_rate=1.0,
    environment="development",  # Change this based on your environment
    integrations=[
        FastApiIntegration(
            transaction_style="endpoint"
        ),
        LoggingIntegration(
            level=logging.INFO,        # Capture info and above as breadcrumbs
            event_level=logging.ERROR  # Send errors as events
        ),
        AsyncioIntegration(),
    ],
)

logger.debug("Creating FastAPI application")
app = FastAPI(
    title="Document Q&A API",
    description="API for document upload and Q&A using LLM",
    version="1.0.0",
)

# Configure CORS
logger.debug("Configuring CORS")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    logger.debug("Health check endpoint called")
    return {"status": "healthy"}

@app.get("/sentry-debug")
async def trigger_error():
    """Endpoint to test Sentry error reporting."""
    logger.debug("Sentry debug endpoint called")
    raise Exception("Test error from /sentry-debug endpoint")

@app.get("/")
async def root() -> dict:
    """Root endpoint providing API information."""
    logger.debug("Root endpoint called")
    return {
        "message": "Welcome to Document Q&A API",
        "version": "1.0.0",
        "documentation": {
            "swagger": "/docs",
            "redoc": "/redoc"
        },
        "endpoints": {
            "upload_document": "/api/upload",
            "ask_question": "/api/ask",
            "list_documents": "/api/documents",
            "list_models": "/api/models"
        }
    }

logger.debug("Including API routes")
# Include API routes
app.include_router(api_router, prefix="/api")

logger.debug("Application initialization complete")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001) 