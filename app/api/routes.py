from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.document import DocumentService
from app.services.llm import LLMService
from app.models.schemas import QuestionRequest, QuestionResponse
from typing import Dict, Any


router = APIRouter()
document_service = DocumentService()
llm_service = LLMService()


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)) -> Dict[str, str]:
    """Upload a document for Q&A."""
    try:
        document_id = await document_service.save_document(file)
        return {
            "document_id": document_id,
            "message": "Document uploaded successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/ask", response_model=QuestionResponse)
async def ask_question(question: QuestionRequest) -> QuestionResponse:
    """Ask a question about the uploaded document."""
    try:
        # Find provider for the model if provided
        if question.model:
            from app.core.config import settings
            provider = None
            for p, models in settings.AVAILABLE_MODELS.items():
                if question.model in models:
                    provider = p
                    break
            
            if provider:
                llm_service.set_model(provider, question.model)
            else:
                error_msg = f"Model {question.model} not found in any provider"
                raise ValueError(error_msg)
            
        answer = await llm_service.get_answer(
            question.document_id,
            question.question
        )
        return QuestionResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/documents")
async def list_documents() -> Dict[str, Any]:
    """List all uploaded documents."""
    try:
        documents = await document_service.list_documents()
        return {"documents": documents}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/models")
async def list_models() -> Dict[str, Any]:
    """List available models and providers."""
    try:
        from app.core.config import settings
        return {
            "models": settings.AVAILABLE_MODELS,
            "default_provider": settings.DEFAULT_PROVIDER,
            "default_model": settings.DEFAULT_MODEL
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/test/test-providers")
async def test_providers() -> Dict[str, bool]:
    """Test all available LLM providers."""
    return await llm_service._test_all_providers()

@router.get("/test/test-provider/{provider}")
async def test_provider(provider: str) -> Dict[str, bool]:
    """Test a specific LLM provider."""
    result = await llm_service.test_provider(provider)
    return {provider: result} 