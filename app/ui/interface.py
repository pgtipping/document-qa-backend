"""Gradio interface for the Document Q&A application."""

import os
from pathlib import Path
from typing import Any, Tuple, Optional, Dict

import gradio as gr
from fastapi import UploadFile
from app.services.document import DocumentService
from app.services.llm import LLMService
from app.core.logger import error_logger
from app.core.config import settings


# Ensure upload directory exists
UPLOAD_DIR = Path(settings.UPLOAD_DIR)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

document_service = DocumentService()
llm_service = LLMService()

# Supported MIME types and their extensions
SUPPORTED_MIME_TYPES = {
    'application/pdf': '.pdf',
    'text/plain': '.txt',
    'application/msword': '.doc',
    'application/vnd.openxmlformats-officedocument'
    '.wordprocessingml.document': '.docx'
}

FILE_TYPES = ["pdf", "txt", "doc", "docx"]  # Explicit list for Gradio

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB


def validate_file(file_path: str) -> Tuple[bool, str]:
    """Validate the uploaded file.
    
    Args:
        file_path: Path to the uploaded file
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        if not os.path.exists(file_path):
            return False, "File not found"
            
        # Check file size
        file_size = os.path.getsize(file_path)
        if file_size > MAX_FILE_SIZE:
            return False, (
                f"File size exceeds maximum limit of "
                f"{MAX_FILE_SIZE/1024/1024}MB"
            )
            
        # Get file extension
        file_ext = os.path.splitext(file_path)[1].lower().lstrip('.')
        if file_ext not in FILE_TYPES:
            return False, f"Unsupported file extension: {file_ext}"
            
        # Try to read file content
        with open(file_path, 'rb') as f:
            content = f.read()
            if not content:
                return False, "File is empty"
                
        return True, ""
        
    except Exception as e:
        return False, f"File validation error: {str(e)}"


async def check_llm_connection() -> Tuple[bool, str]:
    """Check if LLM service is accessible.
    
    Returns:
        Tuple of (is_connected, error_message)
    """
    try:
        # Use a simple static test instead of document-based test
        test_response = await llm_service.test_connection()
        if test_response:
            return True, ""
        return False, "LLM service not responding"
    except Exception as e:
        return False, f"LLM connection error: {str(e)}"


def handle_error(
    exception: Exception, 
    context: Optional[Dict[str, Any]] = None
) -> str:
    """Handle and log errors, returning a user-friendly message.
    
    Args:
        exception: The exception that occurred
        context: Additional context for logging
        
    Returns:
        A user-friendly error message
    """
    if context is None:
        context = {}
        
    error_logger.log_error(
        exception,
        {"function": "process_file_and_question", **context},
        "interface"
    )
    return (
        f"An error occurred while processing your request: "
        f"{str(exception)}"
    )


async def process_file_and_question(
    file_obj: Any,
    question: str,
    progress: Optional[gr.Progress] = None
) -> str:
    """Process uploaded file and answer question.
    
    Args:
        file_obj: The uploaded file object from Gradio
        question: The question to answer about the document
        progress: Progress indicator (optional)
        
    Returns:
        The answer to the question or an error message
    """
    if not file_obj:
        return "Please upload a document first."
    if not question:
        return "Please ask a question."
    
    try:
        if progress:
            progress(0, desc="Checking LLM connection...")
        # Check LLM connection first
        llm_ok, llm_error = await check_llm_connection()
        if not llm_ok:
            return f"LLM service is not available: {llm_error}"
        
        if progress:
            progress(0.2, desc="Processing file...")
        # Handle both filepath and binary types
        if isinstance(file_obj, str):
            temp_path = file_obj
        elif hasattr(file_obj, "name"):
            temp_path = file_obj.name
        else:
            return "Invalid file upload format"
            
        # Validate file
        if progress:
            progress(0.3, desc="Validating file...")
        is_valid, error_msg = validate_file(temp_path)
        if not is_valid:
            return f"File validation failed: {error_msg}"
            
        if progress:
            progress(0.4, desc="Saving document...")
        with open(temp_path, "rb") as f:
            upload_file = UploadFile(
                filename=os.path.basename(temp_path),
                file=f
            )
            
            try:
                # Save document
                document_id = await document_service.save_document(upload_file)
                if not document_id:
                    return "Failed to save document"
                
                # Verify the file was actually saved
                doc_path = await document_service.get_document_path(document_id)
                if not doc_path.exists():
                    return "Document was not saved properly"
                    
                if progress:
                    progress(0.6, desc="Verifying document content...")
                # Verify document content is accessible
                doc_content = await document_service.get_document_content(
                    document_id
                )
                if not doc_content:
                    return "Failed to access document content"
                    
                if progress:
                    progress(0.8, desc="Getting answer from LLM...")
                # Get answer
                answer = await llm_service.get_answer(document_id, question)
                if not answer:
                    return "Failed to get answer from LLM"
                    
                if progress:
                    progress(1.0, desc="Complete!")
                return answer
                
            except Exception as e:
                return handle_error(e, {
                    "file_name": os.path.basename(temp_path),
                    "file_size": os.path.getsize(temp_path),
                    "question": question
                })
                
    except Exception as e:
        return handle_error(e)


def launch_interface() -> None:
    """Launch the Gradio interface."""
    with gr.Blocks(title="Document Q&A System") as demo:
        gr.Markdown(
            """
            # Document Q&A System
            Upload a document and ask questions about its contents.
            
            Supported file types: PDF, TXT, DOC, DOCX
            Maximum file size: 10MB
            
            Upload directory: {upload_dir}
            """.format(upload_dir=UPLOAD_DIR)
        )
        
        # Status indicators
        with gr.Row():
            upload_status = gr.Textbox(
                label="Upload Status",
                interactive=False
            )
            llm_status = gr.Textbox(
                label="LLM Status",
                interactive=False
            )
            
        with gr.Row():
            file_input = gr.File(
                label="Upload Document",
                file_types=FILE_TYPES,
                type="binary",
                elem_id="file_input"
            )
            text_input = gr.Textbox(
                label="Question",
                placeholder="Ask a question about the document...",
                lines=2
            )
        text_output = gr.Textbox(
            label="Answer",
            lines=4,
            show_copy_button=True
        )
        
        submit_btn = gr.Button("Ask Question", variant="primary")
        
        # Update upload status when file is uploaded
        def update_upload_status(file: Any) -> str:
            if not file:
                return "No file uploaded"
            try:
                is_valid, error = validate_file(file.name)
                if not is_valid:
                    return f"Upload failed: {error}"
                return (
                    f"File uploaded successfully: "
                    f"{os.path.basename(file.name)}"
                )
            except Exception as e:
                return f"Upload failed: {str(e)}"
                
        file_input.upload(
            fn=update_upload_status,
            inputs=[file_input],
            outputs=[upload_status]
        )
        
        # Update LLM status on page load
        async def update_llm_status() -> str:
            is_connected, error = await check_llm_connection()
            if is_connected:
                return "LLM service is ready"
            return f"LLM error: {error}"
            
        demo.load(
            fn=update_llm_status,
            outputs=[llm_status]
        )
        
        submit_btn.click(
            fn=process_file_and_question,
            inputs=[file_input, text_input],
            outputs=text_output,
            show_progress='full'
        )
    
    # Try different ports if 7860 is taken
    for port in range(7860, 7870):
        try:
            demo.launch(
                share=False,
                server_name="0.0.0.0",
                server_port=port,
                show_error=True
            )
            break
        except OSError:
            if port == 7869:  # Last attempt
                raise OSError(
                    "Could not find an available port in range 7860-7869"
                )
            continue 