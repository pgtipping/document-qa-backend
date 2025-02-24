import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.services.document import DocumentService
import tempfile
import os
import hashlib
from typing import List, Generator


pytestmark = pytest.mark.asyncio


@pytest.fixture  # type: ignore[misc]
def test_client() -> TestClient:
    """Create a test client."""
    return TestClient(app)


@pytest.fixture  # type: ignore[misc]
def test_files() -> Generator[List[str], None, None]:
    """Create test files with known content."""
    files = []
    content = "Test document content for regression testing."
    
    # Create files with different extensions
    for ext in ["txt", "pdf", "doc", "docx"]:
        temp = tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False)
        temp.write(content.encode())
        temp.close()
        files.append(temp.name)
    
    yield files
    
    # Cleanup
    for file in files:
        os.unlink(file)


class TestRegression:
    """Regression tests for core functionality."""

    def test_document_upload_regression(
        self,
        test_client: TestClient,
        test_files: List[str]
    ) -> None:
        """Test document upload functionality hasn't regressed."""
        for file_path in test_files:
            with open(file_path, "rb") as f:
                response = test_client.post(
                    "/api/upload",
                    files={"file": f}
                )
                assert response.status_code == 200
                assert "document_id" in response.json()

    def test_document_listing_regression(
        self,
        test_client: TestClient,
        test_files: List[str]
    ) -> None:
        """Test document listing functionality hasn't regressed."""
        # Upload test files
        uploaded_ids = []
        for file_path in test_files:
            with open(file_path, "rb") as f:
                response = test_client.post(
                    "/api/upload",
                    files={"file": f}
                )
                uploaded_ids.append(response.json()["document_id"])

        # Test listing
        response = test_client.get("/api/documents")
        assert response.status_code == 200
        documents = response.json()["documents"]
        
        # Verify all uploaded documents are listed
        listed_ids = [doc["id"] for doc in documents]
        for doc_id in uploaded_ids:
            assert doc_id in listed_ids

    def test_qa_regression(
        self,
        test_client: TestClient,
        test_files: List[str]
    ) -> None:
        """Test Q&A functionality hasn't regressed."""
        # Upload a test document
        with open(test_files[0], "rb") as f:
            response = test_client.post(
                "/api/upload",
                files={"file": f}
            )
            document_id = response.json()["document_id"]

        # Test question answering
        test_questions = [
            "What is this document about?",
            "What is the main content?",
            "Summarize the document."
        ]

        for question in test_questions:
            response = test_client.post(
                "/api/ask",
                json={
                    "document_id": document_id,
                    "question": question
                }
            )
            assert response.status_code == 200
            assert "answer" in response.json()

    def test_security_regression(self, test_client: TestClient) -> None:
        """Test security measures haven't regressed."""
        # Test file size limit
        large_content = "x" * (10 * 1024 * 1024 + 1)  # Exceeds 10MB
        with tempfile.NamedTemporaryFile(suffix=".txt") as temp:
            temp.write(large_content.encode())
            temp.seek(0)
            response = test_client.post(
                "/api/upload",
                files={"file": temp}
            )
            assert response.status_code == 400
            assert "too large" in response.json()["detail"].lower()

        # Test invalid file type
        with tempfile.NamedTemporaryFile(suffix=".exe") as temp:
            temp.write(b"malicious content")
            temp.seek(0)
            response = test_client.post(
                "/api/upload",
                files={"file": temp}
            )
            assert response.status_code == 400
            assert "not allowed" in response.json()["detail"].lower()

    async def test_file_integrity_regression(
        self,
        test_client: TestClient,
        test_files: List[str]
    ) -> None:
        """Test file integrity checks haven't regressed."""
        # Upload a file and verify its hash
        with open(test_files[0], "rb") as f:
            content = f.read()
            original_hash = hashlib.sha256(content).hexdigest()
            
            f.seek(0)
            response = test_client.post(
                "/api/upload",
                files={"file": f}
            )
            assert response.status_code == 200
            document_id = response.json()["document_id"]

        # Verify the stored file matches the original
        doc_service = DocumentService()
        file_path = await doc_service.get_document_path(document_id)
        with open(str(file_path), "rb") as f:
            stored_content = f.read()
            stored_hash = hashlib.sha256(stored_content).hexdigest()
            assert stored_hash == original_hash 