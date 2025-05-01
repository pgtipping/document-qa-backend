from groq import Groq
from together import Together
from openai import OpenAI
from google.generativeai import GenerativeModel
from google.generativeai import configure as configure_genai
from openai import AsyncOpenAI
from app.core.config import settings
from app.services.document import DocumentService
import hashlib
from typing import Dict, Optional, List, Any
import logging
import asyncio
import json
import boto3
import aiofiles
from botocore.exceptions import ClientError
from datetime import datetime
from pathlib import Path
import time
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

# Set debug loggers
debug_loggers = [
    "app.services.llm.completion",
    "app.services.llm.metrics"
]

for logger_name in debug_loggers:
    logging.getLogger(logger_name).setLevel(logging.DEBUG)

class LLMService:
    """Service for managing LLM operations."""

    def __init__(self) -> None:
        """Initialize the LLM service."""
        # Initialize clients for each provider
        self.clients = {}
        self.available_providers = []
        
        # Initialize metadata keywords for document analysis
        self.metadata_keywords = {
            'title', 'author', 'date', 'summary', 'abstract', 'metadata',
            'version', 'publisher', 'copyright', 'document', 'type'
        }
        
        # Initialize each provider independently
        self._init_groq()
        self._init_together()
        self._init_deepseek()
        self._init_gemini()
        self._init_openai()
        self._init_openrouter()
        
        # The fallback order is defined by the user
        self.fallback_providers = ["openrouter", "google", "groq"]
        
        # Check if at least one of the fallback providers is available
        if not any(provider in self.available_providers for provider in self.fallback_providers):
             logger.error("None of the specified fallback LLM providers are available")
             self.current_provider = ""
             self.current_model = ""
        else:
             logger.info("Fallback LLM providers initialized")
             # Set current provider/model to the first available in the fallback list for initial state
             for provider in self.fallback_providers:
                 if provider in self.available_providers:
                     self.current_provider = provider
                     models = settings.AVAILABLE_MODELS[self.current_provider]
                     self.current_model = next(iter(models.keys()))
                     msg = (
                         f"Initial provider set to {self.current_provider} "
                         f"with model {self.current_model} based on fallback order"
                     )
                     logger.info(msg)
                     break

        # Initialize other service components
        self.document_service = DocumentService()
        self.cache = {}
        self.cache_ttl = 3600
        self.max_chunk_size = 500
        self.max_chunks = 8
        self.max_context_length = 4000
        self.timing_metrics = {}
        
        # Initialize S3 client if credentials are available
        if settings.AWS_ACCESS_KEY_ID and settings.AWS_SECRET_ACCESS_KEY:
            try:
                self.s3_client = boto3.client(
                    's3',
                    aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                    aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
                    region_name=settings.AWS_REGION
                )
                logger.info("S3 client initialized")
            except Exception as e:
                logger.error(f"Failed to initialize S3 client: {str(e)}")
                self.s3_client = None
        else:
            logger.warning("AWS credentials not found, S3 logging disabled")
            self.s3_client = None

    def _init_groq(self) -> None:
        """Initialize Groq client."""
        if not settings.GROQ_API_KEY:
            return
            
        try:
            self.clients["groq"] = Groq(api_key=settings.GROQ_API_KEY)
            self.available_providers.append("groq")
            logger.info("Groq initialized")
        except Exception as e:
            logger.error(f"Groq init failed: {str(e)}")

    def _init_together(self) -> None:
        """Initialize Together client."""
        if not settings.TOGETHER_API_KEY:
            return
            
        try:
            self.clients["together"] = Together(
                api_key=settings.TOGETHER_API_KEY,
                timeout=60
            )
            self.available_providers.append("together")
            logger.info("Together initialized")
        except Exception as e:
            logger.error(f"Together init failed: {str(e)}")

    def _init_deepseek(self) -> None:
        """Initialize Deepseek client."""
        if not settings.DEEPSEEK_API_KEY:
            return
            
        try:
            self.clients["deepseek"] = OpenAI(
                api_key=settings.DEEPSEEK_API_KEY,
                base_url="https://api.deepseek.com"
            )
            self.available_providers.append("deepseek")
            logger.info("Deepseek initialized")
        except Exception as e:
            logger.error(f"Deepseek init failed: {str(e)}")

    def _init_gemini(self) -> None:
        """Initialize Gemini client."""
        if not settings.GEMINI_API_KEY:
            return
            
        try:
            configure_genai(api_key=settings.GEMINI_API_KEY)
            self.clients["gemini"] = GenerativeModel("gemini-1.5-flash-8b")
            self.available_providers.append("gemini")
            logger.info("Gemini initialized")
        except Exception as e:
            logger.error(f"Gemini init failed: {str(e)}")

    def _init_openai(self) -> None:
        """Initialize OpenAI client."""
        if not settings.OPENAI_API_KEY:
            return
            
        try:
            self.clients["openai"] = AsyncOpenAI(
                api_key=settings.OPENAI_API_KEY
            )
            self.available_providers.append("openai")
            logger.info("OpenAI initialized")
        except Exception as e:
            logger.error(f"OpenAI init failed: {str(e)}")

    def _init_openrouter(self) -> None:
        """Initialize OpenRouter client."""
        if not settings.OPENROUTER_API_KEY:
            return

        try:
            self.clients["openrouter"] = AsyncOpenAI(
                api_key=settings.OPENROUTER_API_KEY,
                base_url="https://openrouter.ai/api/v1"
            )
            self.available_providers.append("openrouter")
            logger.info("OpenRouter initialized")
        except Exception as e:
            logger.error(f"OpenRouter init failed: {str(e)}")

    # Removed set_model as model selection is removed from UI
    # def set_model(self, provider: str, model: str) -> None:
    #     """Set the current model and provider to use."""
    #     if provider not in settings.AVAILABLE_MODELS:
    #         avail = ", ".join(self.available_providers)
    #         raise ValueError(
    #             f"Provider {provider} not supported. "
    #             f"Available providers: {avail}"
    #         )
            
    #     if model not in settings.AVAILABLE_MODELS[provider]:
    #         models = ", ".join(settings.AVAILABLE_MODELS[provider].keys())
    #         raise ValueError(
    #             f"Model {model} not supported for provider {provider}. "
    #             f"Available models: {models}"
    #         )
            
    #     if provider not in self.clients:
    #         raise ValueError(
    #             f"Provider {provider} not initialized. "
    #             "Please check API key and provider status."
    #         )
            
    #     self.current_provider = provider
    #     self.current_model = model
    #     msg = f"Set model to {model} from provider {provider}"
    #     logger.debug(msg)

    def _split_into_chunks(self, text: str) -> List[str]:
        """Split text into smaller, more focused chunks."""
        # Clean and normalize text
        text = text.replace('\n', ' ').replace('\r', ' ')
        text = ' '.join(text.split())  # Normalize whitespace
        
        # Split by sentences for finer control
        sentence_pattern = r'(?<=[.!?])\s+'
        sentences = re.split(sentence_pattern, text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            current_len = len(current_chunk) + len(sentence)
            if current_len < self.max_chunk_size:
                current_chunk += sentence + " "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + " "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        logger.debug(f"Split text into {len(chunks)} chunks")
        return chunks

    def _get_relevant_chunks(
        self, 
        chunks: List[str], 
        question: str
    ) -> List[str]:
        """Get the most relevant chunks for the question.
        
        Uses semantic similarity and keyword matching to find best content.
        """
        # Extract keywords from the question
        stop_words = {
            'what', 'when', 'where', 'who', 'why', 'how', 'is', 'are', 'the',
            'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'
        }
        question_words = re.findall(r'\w+', question.lower())
        keywords = set(question_words) - stop_words
        logger.debug(f"Extracted keywords: {keywords}")
        
        # Check if this is a metadata question
        is_metadata_question = any(
            word in self.metadata_keywords 
            for word in keywords
        )
        
        # Score chunks based on multiple factors
        chunk_scores = []
        for i, chunk in enumerate(chunks):
            # Calculate keyword density
            chunk_words = len(chunk.split())
            if chunk_words == 0:
                continue
                
            # Count exact keyword matches
            keyword_matches = sum(
                1 for keyword in keywords 
                if keyword in chunk.lower()
            )
            
            # Count partial matches (substrings)
            def is_partial_match(kw: str, w: str) -> bool:
                return (kw in w) or (w in kw)
                
            partial_matches = sum(
                1 for keyword in keywords
                for word in chunk.lower().split()
                if is_partial_match(keyword, word)
            )
            
            # Calculate base scores
            density = keyword_matches / chunk_words
            partial_score = partial_matches / chunk_words
            
            # Calculate context score based on surrounding chunks
            context_score = 0.0
            if i > 0:  # Check previous chunk
                prev_chunk = chunks[i-1].lower()
                context_score += sum(
                    1 for keyword in keywords 
                    if keyword in prev_chunk
                ) / len(prev_chunk.split())
            if i < len(chunks) - 1:  # Check next chunk
                next_chunk = chunks[i+1].lower()
                context_score += sum(
                    1 for keyword in keywords 
                    if keyword in next_chunk
                ) / len(next_chunk.split())
            context_score = context_score / 2  # Normalize to 0-1 range
            
            # Additional metadata score if relevant
            metadata_score = 0.0
            if is_metadata_question:
                metadata_matches = sum(
                    1 for word in self.metadata_keywords
                    if word in chunk.lower()
                )
                metadata_score = metadata_matches / chunk_words
            
            # Combine scores with weights
            final_score = (
                density * 0.4 +  # Exact keyword matches
                partial_score * 0.2 +  # Partial matches
                context_score * 0.2 +  # Surrounding context relevance
                metadata_score * 0.2  # Metadata terms if relevant
            )
            
            chunk_scores.append((final_score, i, chunk))
            logger.debug(f"Chunk {i} score: {final_score}")
        
        # Sort by score
        chunk_scores.sort(reverse=True)
        
        # Select chunks with context
        selected_indices = set()
        selected_chunks = []
        
        # Add highest scoring chunks and their context
        for score, idx, chunk in chunk_scores:
            if len(selected_chunks) >= self.max_chunks:
                break
                
            # If this chunk or its neighbors aren't already selected
            if idx not in selected_indices:
                # Add the chunk
                selected_indices.add(idx)
                selected_chunks.append(chunk)
                
                # Consider adding surrounding context
                if score > 0.1:  # Only add context for relevant chunks
                    # Add previous chunk if it exists and not already selected
                    if idx > 0 and (idx-1) not in selected_indices:
                        selected_indices.add(idx-1)
                        selected_chunks.append(chunks[idx-1])
                    
                    # Add next chunk if it exists and not already selected
                    if idx < len(chunks)-1 and (idx+1) not in selected_indices:
                        selected_indices.add(idx+1)
                        selected_chunks.append(chunks[idx+1])
        
        # Sort chunks by their original order to maintain document flow
        selected_chunks.sort(key=lambda x: chunks.index(x))
        
        # Truncate if total length exceeds max_context_length
        total_length = sum(len(chunk) for chunk in selected_chunks)
        if total_length > self.max_context_length:
            truncated = []
            current_length = 0
            for chunk in selected_chunks:
                if current_length + len(chunk) <= self.max_context_length:
                    truncated.append(chunk)
                    current_length += len(chunk)
                else:
                    remaining = self.max_context_length - current_length
                    if remaining > 100:  # Only add partial if substantial
                        truncated.append(chunk[:remaining])
                    break
            selected_chunks = truncated
        
        logger.debug(f"Selected {len(selected_chunks)} chunks")
        for i, chunk in enumerate(selected_chunks):
            logger.debug(f"Chunk {i} preview: {chunk[:100]}...")
        return selected_chunks

    def _create_prompt(self, content: str, question: str) -> str:
        """Create a detailed prompt for the LLM."""
        prompt = (
            "You are a helpful assistant that answers questions based on the "
            "provided document content. Your task is to:\n"
            "1. Read the following content carefully\n"
            "2. Answer the question accurately using ONLY the provided content\n"
            "3. If you cannot find the answer in the content, say so\n"
            "4. Do not make up or infer information not present in the "
            "content\n\n"
            "Important: For questions about title, author, or other metadata, "
            "look for explicit mentions in the text. Do not guess or infer.\n\n"
            f"Content:\n{content}\n\n"
            f"Question: {question}\n\n"
            "Answer: "
        )
        logger.debug(f"Created prompt with content length: {len(content)}")
        return prompt

    def _record_timing(self, step: str, start_time: float) -> float:
        """Record timing for a step and return new start time."""
        elapsed = time.time() - start_time
        self.timing_metrics[step] = elapsed
        return time.time()

    def _get_timing_summary(self) -> str:
        """Generate timing summary."""
        total_time = sum(self.timing_metrics.values())
        summary = ["Processing Time Breakdown:"]
        
        for step, duration in self.timing_metrics.items():
            percentage = (duration / total_time) * 100
            summary.append(
                f"- {step}: {duration:.2f}s ({percentage:.1f}%)"
            )
        
        summary.append(f"Total Time: {total_time:.2f}s")
        return "\n".join(summary)

    async def _log_performance_metrics(
        self,
        document_id: str,
        question: str,
        content_metrics: Dict[str, Any],
        timing_metrics: Dict[str, float],
        doc_timing_metrics: Dict[str, float]
    ) -> None:
        """Log detailed performance metrics to S3."""
        if not self.s3_client or not settings.S3_BUCKET:
            logger.warning("S3 not configured, skipping performance logging")
            return

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_key = f"{settings.S3_PERFORMANCE_LOGS_PREFIX}{timestamp}_{document_id}.json"
        
        # Calculate totals with safety checks
        llm_total = sum(timing_metrics.values()) or 1  # Use 1 if sum is 0
        
        # Format metrics for JSON
        metric_entry = {
            "timestamp": timestamp,
            "model": self.current_model,
            "provider": self.current_provider,
            "question": question,
            "document_metrics": {
                "size_kb": content_metrics["size_kb"],
                "total_chunks": content_metrics["total_chunks"],
                "selected_chunks": content_metrics["selected_chunks"],
                "chunk_size": self.max_chunk_size,
                "context_length": content_metrics["context_length"]
            },
            "llm_timing": [
                {
                    "name": step,
                    "value": duration,
                    "percentage": (duration / llm_total) * 100
                }
                for step, duration in timing_metrics.items()
            ],
            "doc_timing": [
                {
                    "name": step,
                    "value": duration,
                    "percentage": (duration / doc_total) * 100
                }
                for step, duration in doc_timing_metrics.items()
            ],
            "total_llm_time": sum(timing_metrics.values()),
            "total_doc_time": sum(doc_timing_metrics.values())
        }

        try:
            # Upload metrics to S3
            self.s3_client.put_object(
                Bucket=settings.S3_BUCKET,
                Key=log_key,
                Body=json.dumps(metric_entry, indent=2),
                ContentType='application/json'
            )
            logger.debug(f"Performance metrics uploaded to S3: {log_key}")
                
        except ClientError as e:
            logger.error(f"Failed to upload metrics to S3: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error uploading metrics: {str(e)}")

    async def get_answer(self, document_id: str, question: str) -> str:
        """Get answer to a question based on document content."""
        start_time = time.time()
        self.timing_metrics = {}  # Reset timing for each request
        doc_timing_metrics = {} # Timing for document processing

        # 1. Retrieve and process document content
        doc_start_time = time.time()
        document_content = await self.document_service.get_document_content(document_id)
        doc_timing_metrics["get_document_content"] = time.time() - doc_start_time

        if not document_content:
            logger.warning(f"Document with ID {document_id} not found.")
            return "Could not retrieve document content."

        # 2. Split content into chunks
        doc_start_time = time.time()
        chunks = self._split_into_chunks(document_content)
        doc_timing_metrics["split_into_chunks"] = time.time() - doc_start_time

        if not chunks:
            logger.warning("Document is empty or could not be processed into chunks.")
            return "Could not process document content."

        # 3. Get relevant chunks based on the question
        doc_start_time = time.time()
        relevant_chunks = self._get_relevant_chunks(chunks, question)
        doc_timing_metrics["get_relevant_chunks"] = time.time() - doc_start_time

        if not relevant_chunks:
            logger.warning("No relevant content found for the question.")
            # Log metrics even if no relevant chunks are found
            content_metrics = {
                "size_kb": len(document_content.encode('utf-8')) / 1024,
                "total_chunks": len(chunks),
                "selected_chunks": 0,
                "context_length": 0
            }
            await self._log_performance_metrics(
                document_id, question, content_metrics, self.timing_metrics, doc_timing_metrics
            )
            return "Could not find relevant information in the document to answer the question."

        # Combine relevant chunks for the prompt
        context_content = "\n\n".join(relevant_chunks)

        # Log content metrics
        content_metrics = {
            "size_kb": len(document_content.encode('utf-8')) / 1024,
            "total_chunks": len(chunks),
            "selected_chunks": len(relevant_chunks),
            "context_length": len(context_content)
        }
        logger.debug(f"Content metrics: {content_metrics}")

        # 4. Create prompt for the LLM
        prompt = self._create_prompt(context_content, question)

        # 5. Get completion from LLM with fallback
        llm_start_time = time.time()
        answer = await self.get_completion(prompt)
        self.timing_metrics["get_completion"] = time.time() - llm_start_time

        # 6. Log performance metrics
        await self._log_performance_metrics(
            document_id, question, content_metrics, self.timing_metrics, doc_timing_metrics
        )

        return answer

    async def test_connection(self) -> bool:
        """Test connection to the current LLM provider."""
        if not self.current_provider or not self.current_model:
            logger.warning("No LLM provider or model is currently set.")
            return False

        try:
            # Attempt a simple completion or ping based on provider capabilities
            # This is a simplified test; a more robust test would be provider-specific
            prompt = "Hello, world!"
            await self.get_completion(prompt)
            logger.info(f"Connection test successful for {self.current_provider}")
            return True
        except Exception as e:
            logger.error(f"Connection test failed for {self.current_provider}: {str(e)}")
            return False

    async def _test_all_providers(self) -> Dict[str, bool]:
        """Test connection for all available providers."""
        results = {}
        for provider in self.available_providers:
            try:
                # Use a small, fast model for testing if available
                test_model = next(iter(settings.AVAILABLE_MODELS[provider].keys()))
                
                # Temporarily set provider and model for testing
                original_provider = self.current_provider
                original_model = self.current_model
                self.current_provider = provider
                self.current_model = test_model

                # Attempt a simple completion
                prompt = "Test connection."
                await self.get_completion(prompt)
                results[provider] = True
                logger.info(f"Provider {provider} test successful.")
            except Exception as e:
                results[provider] = False
                logger.error(f"Provider {provider} test failed: {str(e)}")
            finally:
                # Restore original provider and model
                self.current_provider = original_provider
                self.current_model = original_model
        return results

    async def test_provider(self, provider: str) -> bool:
        """Test a specific LLM provider."""
        if provider not in self.available_providers:
            logger.warning(f"Provider {provider} is not available.")
            return False

        try:
            # Use a small, fast model for testing if available
            test_model = next(iter(settings.AVAILABLE_MODELS[provider].keys()))
            
            # Temporarily set provider and model for testing
            original_provider = self.current_provider
            original_model = self.current_model
            self.current_provider = provider
            self.current_model = test_model

            # Attempt a simple completion
            prompt = "Test connection."
            await self.get_completion(prompt)
            logger.info(f"Provider {provider} test successful.")
            return True
        except Exception as e:
            logger.error(f"Provider {provider} test failed: {str(e)}")
            return False
        finally:
            # Restore original provider and model
            self.current_provider = original_provider
            self.current_model = original_model

    async def get_completion(self, prompt: str) -> str:
        """Get completion from the current LLM provider with fallback."""
        providers_to_try = [self.current_provider] + [p for p in self.fallback_providers if p != self.current_provider]
        
        for provider in providers_to_try:
            if provider not in self.clients:
                logger.warning(f"Provider {provider} not initialized, skipping.")
                continue

            client = self.clients[provider]
            model = settings.AVAILABLE_MODELS.get(provider, {}).get(self.current_model) # Use current_model for consistency

            if not model:
                 logger.warning(f"Model {self.current_model} not available for provider {provider}, skipping.")
                 continue

            logger.info(f"Attempting completion with provider: {provider}, model: {model}")

            try:
                if provider == "groq":
                    response = client.chat.completions.create(
                        messages=[{"role": "user", "content": prompt}],
                        model=model,
                        temperature=0.7,
                        max_tokens=1500,
                    )
                    return response.choices[0].message.content.strip()
                elif provider == "together":
                    response = client.chat.completions.create(
                        messages=[{"role": "user", "content": prompt}],
                        model=model,
                        temperature=0.7,
                        max_tokens=1500,
                    )
                    return response.choices[0].message.content.strip()
                elif provider == "deepseek":
                    response = client.chat.completions.create(
                        messages=[{"role": "user", "content": prompt}],
                        model=model,
                        temperature=0.7,
                        max_tokens=1500,
                    )
                    return response.choices[0].message.content.strip()
                elif provider == "google":
                    # Google's API has a different structure
                    response = client.generate_content(
                        contents=[{"role": "user", "parts": [{"text": prompt}]}],
                        generation_config={"temperature": 0.7, "max_output_tokens": 1500}
                    )
                    return response.text.strip()
                elif provider == "openai" or self.current_provider == "openrouter":
                    response = await client.chat.completions.create(
                        messages=[{"role": "user", "content": prompt}],
                        model=model,
                        temperature=0.7,
                        max_tokens=1500,
                    )
                    return response.choices[0].message.content.strip()

            except Exception as e:
                logger.error(f"Provider {provider} failed: {str(e)}")
                # Continue to the next provider in the fallback list

        logger.error("All available LLM providers failed.")
        return "Error: Unable to get a response from any LLM provider."

    async def _extract_with_llm(self, path: Path) -> str:
        """Extract text content from a document using an LLM."""
        try:
            async with aiofiles.open(path, mode='r', encoding='utf-8') as f:
                content = await f.read()

            # Truncate content if it's too long for the LLM
            max_llm_extract_length = 3000  # Example limit
            if len(content) > max_llm_extract_length:
                content = content[:max_llm_extract_length] + "..."
                logger.warning(f"Truncated document content for LLM extraction: {path}")

            prompt = (
                "Extract the main text content from the following document. "
                "Focus on the narrative or informational text and ignore "
                "headers, footers, page numbers, and other non-content elements.\n\n"
                f"Document content:\n{content}"
            )

            extracted_text = await self.get_completion(prompt)
            logger.debug(f"Extracted text from {path} using LLM.")
            return extracted_text

        except Exception as e:
            logger.error(f"Failed to extract text from {path} using LLM: {str(e)}")
            return ""