import pytest
from fastapi import UploadFile, HTTPException
from app.services.document import DocumentService
import os
import tempfile
import shutil
from app.core.config import settings
from typing import AsyncGenerator, BinaryIO, cast


@pytest.fixture  # type: ignore[misc]
async def document_service() -> AsyncGenerator[DocumentService, None]:
    """Create a document service with temporary upload directory."""
    # Create a temporary directory for testing
    test_upload_dir = tempfile.mkdtemp()
    settings.UPLOAD_DIR = test_upload_dir
    
    service = DocumentService()
    yield service
    
    # Cleanup after tests
    shutil.rmtree(test_upload_dir)


@pytest.fixture  # type: ignore[misc]
async def sample_file() -> AsyncGenerator[UploadFile, None]:
    """Create a sample file for testing."""
    content = b"Test document content"
    temp_file = tempfile.SpooledTemporaryFile()
    temp_file.write(content)
    temp_file.seek(0)
    
    # Cast to BinaryIO to satisfy type checker
    binary_file = cast(BinaryIO, temp_file)
    
    file = UploadFile(
        filename="test.txt",
        file=binary_file
    )
    yield file
    
    # Cleanup
    await file.close()
    temp_file.close()


async def test_save_document(
    document_service: DocumentService,
    sample_file: UploadFile
) -> None:
    """Test saving a valid document."""
    # Test saving a valid document
    document_id = await document_service.save_document(sample_file)
    assert document_id is not None
    assert len(document_id) > 0
    
    # Verify file exists
    files = os.listdir(settings.UPLOAD_DIR)
    assert len(files) == 1
    assert files[0].startswith(document_id)


async def test_invalid_extension(document_service: DocumentService) -> None:
    """Test saving a file with invalid extension."""
    # Test saving a file with invalid extension
    temp_file = tempfile.SpooledTemporaryFile()
    binary_file = cast(BinaryIO, temp_file)
    
    invalid_file = UploadFile(
        filename="test.invalid",
        file=binary_file
    )
    
    with pytest.raises(HTTPException) as exc_info:
        await document_service.save_document(invalid_file)
    assert exc_info.value.status_code == 400
    
    # Cleanup
    await invalid_file.close()
    temp_file.close()


async def test_list_documents(
    document_service: DocumentService,
    sample_file: UploadFile
) -> None:
    """Test listing documents."""
    # Save a test document
    document_id = await document_service.save_document(sample_file)
    
    # Test listing documents
    documents = await document_service.list_documents()
    assert len(documents) == 1
    assert documents[0].id == document_id
    assert documents[0].filename.endswith(".txt")
    assert documents[0].content_type == "text/plain" 