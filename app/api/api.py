from fastapi import APIRouter
from app.api.endpoints import documents, chat, metrics, test

api_router = APIRouter()

api_router.include_router(
    documents.router, prefix="/documents", tags=["documents"]
)
api_router.include_router(chat.router, prefix="/chat", tags=["chat"])
api_router.include_router(metrics.router, prefix="/metrics", tags=["metrics"])
api_router.include_router(test.router, prefix="/test", tags=["test"]) 