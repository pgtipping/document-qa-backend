from fastapi import APIRouter, Depends
from app.services.llm import LLMService
from app.core.dependencies import get_llm_service
from typing import Dict

router = APIRouter()

@router.get("/test-providers")
async def test_providers(
    llm_service: LLMService = Depends(get_llm_service)
) -> Dict[str, bool]:
    """Test all available LLM providers."""
    return await llm_service._test_all_providers()

@router.get("/test-provider/{provider}")
async def test_provider(
    provider: str,
    llm_service: LLMService = Depends(get_llm_service)
) -> Dict[str, bool]:
    """Test a specific LLM provider."""
    result = await llm_service.test_provider(provider)
    return {provider: result} 