"""Configuration settings for the Document Q&A application."""

import os
from pathlib import Path
from typing import List, Any, ClassVar, Dict
from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# Get the absolute path to the backend directory
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

# Load environment variables from .env file
env_file = os.path.join(BACKEND_DIR, ".env")
load_dotenv(env_file)


class Settings(BaseSettings):
    """Application settings."""
    
    # File upload settings
    UPLOAD_DIR: str = os.path.join(BACKEND_DIR, "uploads")
    MAX_UPLOAD_SIZE_STR: str = "10485760#bytes"
    ALLOWED_EXTENSIONS_STR: str = "txt,pdf,doc,docx"
    
    # API settings
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    TOGETHER_API_KEY: str = os.getenv("TOGETHER_API_KEY", "")
    DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    
    # S3 settings
    AWS_ACCESS_KEY_ID: str = os.getenv("AWS_ACCESS_KEY_ID", "")
    AWS_SECRET_ACCESS_KEY: str = os.getenv("AWS_SECRET_ACCESS_KEY", "")
    AWS_REGION: str = os.getenv("AWS_REGION", "us-east-1")
    S3_BUCKET: str = os.getenv("S3_BUCKET", "")
    S3_PERFORMANCE_LOGS_PREFIX: str = "performance_logs/"
    
    # Sentry settings
    SENTRY_DSN: str = os.getenv("SENTRY_DSN", "")
    
    # CORS settings
    ALLOWED_ORIGINS: List[str] = [
        # Development URLs
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        # Add your production URLs here
        os.getenv("FRONTEND_URL", ""),  # Main frontend URL
        os.getenv("ADDITIONAL_ALLOWED_ORIGIN", ""),  # Additional origin
    ]
    
    # Available models
    AVAILABLE_MODELS: ClassVar[Dict[str, Dict[str, str]]] = {
        "groq": {
            "llama-3.2-3b-preview": "Meta Llama 3.2-3B",
        },
        "together": {
            "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo": "Meta Llama 3.1-8B",
        },
        "deepseek": {
            "deepseek-chat": "Deepseek V3",
        },
        "gemini": {
            "gemini-1.5-flash-8b": "Gemini 1.5 Flash-8B",
        },
        "openai": {
            "gpt-4o-mini": "GPT-4o mini",
        }
    }
    
    # Default model settings - will be set to first available provider
    DEFAULT_PROVIDER: str = ""
    DEFAULT_MODEL: str = ""
    
    model_config = SettingsConfigDict(
        env_file=env_file,
        env_file_encoding="utf-8",
        env_prefix="",
        extra="allow"
    )

    def __init__(self, **kwargs: Any) -> None:
        """Initialize settings with validation."""
        print(f"Looking for .env file at: {env_file}")
        print(f"File exists: {os.path.exists(env_file)}")
        
        super().__init__(**kwargs)
        # Create upload directory if it doesn't exist
        Path(self.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
        
        # Find first available provider
        for provider, api_key in {
            "groq": self.GROQ_API_KEY,
            "together": self.TOGETHER_API_KEY,
            "deepseek": self.DEEPSEEK_API_KEY,
            "gemini": self.GEMINI_API_KEY,
            "openai": self.OPENAI_API_KEY
        }.items():
            if api_key:
                print(f"{provider.upper()}_API_KEY is set")
                self.DEFAULT_PROVIDER = provider
                self.DEFAULT_MODEL = next(iter(self.AVAILABLE_MODELS[provider].keys()))
                break
                
        if not self.DEFAULT_PROVIDER:
            raise ValueError(
                "No API keys found. At least one provider API key must be set "
                "in environment variables or .env file"
            )

    @property
    def ALLOWED_EXTENSIONS(self) -> List[str]:
        """Get allowed extensions as a list."""
        return [ext.strip() for ext in self.ALLOWED_EXTENSIONS_STR.split(",")]
    
    @property
    def MAX_UPLOAD_SIZE(self) -> int:
        """Get max upload size as an integer."""
        return int(self.MAX_UPLOAD_SIZE_STR.split("#", maxsplit=1)[0])


settings = Settings()