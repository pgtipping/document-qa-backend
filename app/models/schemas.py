"""Pydantic models for the Document Q&A application."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class QuestionRequest(BaseModel):
    """Request model for document questions."""
    
    document_id: str = Field(..., description="ID of the document to query")
    question: str = Field(..., description="Question about the document")
    model: Optional[str] = Field(
        None,
        description="Model to use for answering the question"
    )


class QuestionResponse(BaseModel):
    """Response model for document questions."""
    
    answer: str = Field(..., description="Answer to the question")


class Document(BaseModel):
    """Document model for file metadata."""
    
    id: str = Field(..., description="Unique identifier for the document")
    filename: str = Field(..., description="Original filename")
    size: int = Field(..., description="File size in bytes")
    content_type: str = Field(..., description="MIME type of the file")
    upload_date: datetime = Field(
        default_factory=datetime.now,
        description="Date and time of upload"
    )