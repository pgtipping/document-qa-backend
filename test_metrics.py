import asyncio
from app.services.llm import LLMService
from app.services.document import DocumentService
from fastapi import UploadFile
import io

async def test_metrics():
    try:
        # Create a more substantial test document
        content = b"""# Test Document

This is a comprehensive test document for metrics logging.

## Introduction
The purpose of this document is to test the performance metrics logging system.
It contains multiple paragraphs and sections to ensure proper content extraction.

## Features
1. Multiple sections
2. Various paragraphs
3. Different content types
4. Structured layout

## Conclusion
This document serves as a test case for our document Q&A system.
It helps verify that our metrics logging system works correctly."""

        test_file = UploadFile(
            filename="test.txt",
            file=io.BytesIO(content)
        )
        
        # Initialize services
        doc_service = DocumentService()
        llm_service = LLMService()
        
        # Save document
        doc_id = await doc_service.save_document(test_file)
        print("Document saved with ID:", doc_id)
        
        # Ask a question
        question = "What is the purpose of this document?"
        answer = await llm_service.get_answer(doc_id, question)
        print("\nQuestion:", question)
        print("Answer:", answer)
        
        print("\nTest completed successfully!")
        
    except Exception as e:
        print("Error during test:", str(e))

if __name__ == "__main__":
    asyncio.run(test_metrics()) 