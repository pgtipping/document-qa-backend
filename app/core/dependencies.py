"""Dependencies for FastAPI endpoints."""

from app.services.llm import LLMService

# Singleton instance
_llm_service = None

def get_llm_service() -> LLMService:
    """Get or create LLM service instance."""
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service 