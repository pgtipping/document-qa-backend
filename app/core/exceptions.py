"""Custom exceptions for the Document Q&A application."""

from fastapi import HTTPException
from typing import Any, Optional, Dict


class DocumentQAException(HTTPException):
    """Base exception for Document Q&A application."""
    def __init__(
        self,
        status_code: int = 500,
        detail: Any = None,
        headers: Optional[Dict[str, str]] = None
    ) -> None:
        """Initialize the exception.
        
        Args:
            status_code: HTTP status code
            detail: Error detail message
            headers: Optional HTTP headers
        """
        super().__init__(
            status_code=status_code,
            detail=detail,
            headers=headers
        )


class DocumentNotFoundError(DocumentQAException):
    """Raised when a document is not found."""
    def __init__(self, document_id: str) -> None:
        """Initialize the exception.
        
        Args:
            document_id: ID of the document that was not found
        """
        super().__init__(
            status_code=404,
            detail=f"Document not found: {document_id}"
        )


class InvalidFileTypeError(DocumentQAException):
    """Raised when an invalid file type is uploaded."""
    def __init__(self, allowed_types: set[str]) -> None:
        """Initialize the exception.
        
        Args:
            allowed_types: Set of allowed file extensions
        """
        super().__init__(
            status_code=400,
            detail=f"File type not allowed. Allowed types: {allowed_types}"
        )


class FileSizeLimitError(DocumentQAException):
    """Raised when file size exceeds limit."""
    def __init__(self, max_size: int) -> None:
        """Initialize the exception.
        
        Args:
            max_size: Maximum allowed file size in bytes
        """
        super().__init__(
            status_code=400,
            detail=f"File too large. Max size: {max_size} bytes"
        )


class LLMConfigError(DocumentQAException):
    """Raised when LLM configuration is invalid."""
    def __init__(self, message: str = "LLM configuration error") -> None:
        """Initialize the exception.
        
        Args:
            message: Error message describing the configuration issue
        """
        super().__init__(
            status_code=500,
            detail=message
        ) 