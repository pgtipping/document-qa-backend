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
from botocore.exceptions import ClientError
from datetime import datetime
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
        
        # Set default provider and model if any available
        if self.available_providers:
            self.current_provider = self.available_providers[0]
            models = settings.AVAILABLE_MODELS[self.current_provider]
            self.current_model = next(iter(models.keys()))
            msg = (
                f"Using {self.current_provider} "
                f"with model {self.current_model}"
            )
            logger.info(msg)
        else:
            logger.error("No LLM providers available")
            self.current_provider = ""
            self.current_model = ""

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

    def set_model(self, provider: str, model: str) -> None:
        """Set the current model and provider to use."""
        if provider not in settings.AVAILABLE_MODELS:
            avail = ", ".join(self.available_providers)
            raise ValueError(
                f"Provider {provider} not supported. "
                f"Available providers: {avail}"
            )
            
        if model not in settings.AVAILABLE_MODELS[provider]:
            models = ", ".join(settings.AVAILABLE_MODELS[provider].keys())
            raise ValueError(
                f"Model {model} not supported for provider {provider}. "
                f"Available models: {models}"
            )
            
        if provider not in self.clients:
            raise ValueError(
                f"Provider {provider} not initialized. "
                "Please check API key and provider status."
            )
            
        self.current_provider = provider
        self.current_model = model
        msg = f"Set model to {model} from provider {provider}"
        logger.debug(msg)

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
        doc_total = sum(doc_timing_metrics.values()) or 1  # Use 1 if sum is 0
        
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
        """Get an answer from the LLM based on the document content."""
        start_time = time.time()
        self.timing_metrics.clear()  # Reset timing metrics

        if not self.current_provider or not self.current_model:
            available = ", ".join(self.available_providers) if self.available_providers else "none"
            raise ValueError(f"No LLM provider selected. Available providers: {available}")
            
        if not self.clients.get(self.current_provider):
            raise ValueError(f"Selected provider {self.current_provider} is not properly initialized")

        try:
            # Check cache
            cache_key = self._generate_cache_key(document_id, question)
            cached_response = self._get_from_cache(cache_key)
            if cached_response:
                self._record_timing("Cache Retrieval", start_time)
                return cached_response

            # Track content metrics
            content_metrics = {
                "size_kb": 0,
                "total_chunks": 0,
                "selected_chunks": 0,
                "context_length": 0
            }

            # Get document content
            content_start = time.time()
            content = await self.document_service.get_document_content(document_id)
            content_str = content.decode('utf-8')
            content_metrics["size_kb"] = len(content_str) / 1024
            start_time = self._record_timing("Document Retrieval", content_start)

            # Split content and get relevant chunks
            chunk_start = time.time()
            chunks = self._split_into_chunks(content_str)
            content_metrics["total_chunks"] = len(chunks)
            start_time = self._record_timing("Content Chunking", chunk_start)

            # Get relevant chunks
            relevance_start = time.time()
            relevant_chunks = self._get_relevant_chunks(chunks, question)
            content_metrics["selected_chunks"] = len(relevant_chunks)
            relevant_content = " ".join(relevant_chunks)
            content_metrics["context_length"] = len(relevant_content)
            start_time = self._record_timing("Relevance Analysis", relevance_start)

            # Create prompt
            prompt_start = time.time()
            prompt = self._create_prompt(relevant_content, question)
            start_time = self._record_timing("Prompt Creation", prompt_start)

            # Get LLM response
            llm_start = time.time()
            response = await self.get_completion(prompt)
            start_time = self._record_timing("LLM Processing", llm_start)

            # Cache the response
            cache_start = time.time()
            self._add_to_cache(cache_key, response)
            self._record_timing("Cache Update", cache_start)

            # Log metrics asynchronously - don't wait for it
            try:
                asyncio.create_task(self._log_performance_metrics(
                    document_id,
                    question,
                    content_metrics,
                    self.timing_metrics.copy(),  # Copy to avoid race conditions
                    self.document_service.timing_metrics.copy()
                ))
            except Exception as e:
                # Just log the error but don't fail the request
                logger.error(f"Failed to log metrics: {str(e)}")

            return response

        except UnicodeDecodeError as e:
            logger.error(f"Failed to decode document content: {str(e)}")
            raise ValueError("Failed to read document content. The document might be corrupted or in an unsupported format.")
        except Exception as e:
            logger.error(f"Error in get_answer: {str(e)}")
            raise ValueError(f"Failed to get answer: {str(e)}")

    def _generate_cache_key(self, document_id: str, question: str) -> str:
        """Generate a unique cache key for a document-question pair."""
        # Include provider and model in the cache key to separate responses by model
        if not self.current_provider or not self.current_model:
            raise ValueError("No model selected for caching")
            
        combined = f"{document_id}:{question.lower().strip()}:{self.current_provider}:{self.current_model}"
        return hashlib.sha256(combined.encode()).hexdigest()

    def _get_from_cache(self, cache_key: str) -> Optional[str]:
        """Get a response from cache if it exists and is not expired."""
        if cache_key in self.cache:
            answer, timestamp = self.cache[cache_key]
            if time.time() - timestamp < self.cache_ttl:
                return answer
            else:
                del self.cache[cache_key]
        return None

    def _add_to_cache(self, cache_key: str, answer: str) -> None:
        """Add a response to the cache with current timestamp."""
        self.cache[cache_key] = (answer, time.time())

    async def test_connection(self) -> bool:
        """Test if the LLM service is accessible."""
        try:
            # Test each available provider
            results = await self._test_all_providers()
            return any(results.values())
        except Exception as e:
            logger.error(f"Connection test failed: {str(e)}")
            return False

    async def _test_all_providers(self) -> Dict[str, bool]:
        """Test all available providers and return their status."""
        results = {}
        original_provider = self.current_provider
        original_model = self.current_model
        
        test_prompt = "Respond with 'OK' if you can read this message."
        
        for provider in self.available_providers:
            try:
                # Set the provider and its first available model
                models = settings.AVAILABLE_MODELS[provider]
                model = next(iter(models.keys()))
                self.set_model(provider, model)
                
                # Test the provider
                logger.info(f"Testing {provider} with model {model}...")
                response = await self.get_completion(test_prompt)
                
                # Check if response is valid
                is_working = bool(response and response.strip())
                results[provider] = is_working
                
                status = "working" if is_working else "failed"
                logger.info(f"{provider} status: {status}")
                
            except Exception as e:
                logger.error(f"{provider} test failed: {str(e)}")
                results[provider] = False
        
        # Restore original provider and model
        if original_provider and original_model:
            self.set_model(original_provider, original_model)
        
        return results

    async def test_provider(self, provider: str) -> bool:
        """Test a specific provider's endpoint."""
        if provider not in self.available_providers:
            logger.error(f"Provider {provider} not available")
            return False
            
        try:
            # Save current provider/model
            original_provider = self.current_provider
            original_model = self.current_model
            
            # Set the test provider and model
            models = settings.AVAILABLE_MODELS[provider]
            model = next(iter(models.keys()))
            self.set_model(provider, model)
            
            # Test with a simple prompt
            logger.info(f"Testing {provider} with model {model}...")
            test_prompt = "Respond with 'OK' if you can read this message."
            response = await self.get_completion(test_prompt)
            
            # Verify response
            is_working = bool(response and response.strip())
            status = "working" if is_working else "failed"
            logger.info(f"{provider} status: {status}")
            
            # Restore original provider/model
            if original_provider and original_model:
                self.set_model(original_provider, original_model)
                
            return is_working
            
        except Exception as e:
            logger.error(f"{provider} test failed: {str(e)}")
            return False

    async def get_completion(self, prompt: str) -> str:
        """Get completion from the current model."""
        if not self.current_provider or not self.current_model:
            raise ValueError("No model selected")

        client = self.clients.get(self.current_provider)
        if not client:
            raise ValueError(
                f"Provider {self.current_provider} not initialized"
            )

        try:
            if self.current_provider == "groq":
                response = client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model=self.current_model,
                )
                return str(response.choices[0].message.content)

            if self.current_provider == "together":
                max_retries = 3
                retry_delay = 1
                last_error = None
                
                for attempt in range(max_retries):
                    try:
                        response = client.chat.completions.create(
                            messages=[{"role": "user", "content": prompt}],
                            model=self.current_model,
                            timeout=60
                        )
                        return str(response.choices[0].message.content)
                    except Exception as e:
                        last_error = e
                        msg = (
                            f"Together retry {attempt + 1}/{max_retries}: {e}"
                        )
                        logger.error(msg)
                        if attempt < max_retries - 1:
                            await asyncio.sleep(retry_delay)
                            retry_delay *= 2
                            continue
                raise last_error

            if self.current_provider == "deepseek":
                response = client.chat.completions.create(
                    model=self.current_model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a helpful assistant"
                        },
                        {"role": "user", "content": prompt}
                    ],
                    stream=False
                )
                return str(response.choices[0].message.content)

            if self.current_provider == "gemini":
                response = client.generate_content(prompt)
                return str(response.text)

            if self.current_provider == "openai":
                response = await client.chat.completions.create(
                    model=self.current_model,
                    messages=[{"role": "user", "content": prompt}]
                )
                return str(response.choices[0].message.content)

            raise ValueError(f"Provider {self.current_provider} not supported")

        except Exception as e:
            error = f"Error from {self.current_provider}: {str(e)}"
            logger.error(error)
            raise ValueError(error)

    async def _extract_with_llm(self, path: Path) -> str:
        """Extract text content using LLM for challenging documents."""
        try:
            # For text-based models, we'll try to read the file as text first
            try:
                async with aiofiles.open(path, 'r', encoding='utf-8') as f:
                    raw_content = await f.read()
            except UnicodeDecodeError:
                # If text reading fails, try reading as binary and decode with latin-1
                async with aiofiles.open(path, 'rb') as f:
                    content = await f.read()
                    raw_content = content.decode('latin-1', errors='ignore')

            # Create a specialized prompt for document extraction
            prompt = (
                "You are a document content extraction specialist. Your task is to:\n"
                "1. Extract and organize all text content from the provided document\n"
                "2. Preserve the logical structure and flow\n"
                "3. Properly format tables, lists, and other elements\n"
                "4. Include metadata like title and headers\n"
                "5. Remove any non-text elements or formatting artifacts\n\n"
                f"Document type: {path.suffix}\n"
                "Document content:\n\n"
                f"{raw_content[:10000]}"  # First 10K chars to stay within context limits
            )

            # Use a specialized model for document extraction
            extracted_content = await self.get_completion(prompt)
            
            if not extracted_content:
                raise ValueError("LLM extraction returned empty content")
            
            return str(extracted_content)

        except Exception as e:
            logger.error(f"LLM extraction failed: {str(e)}")
            raise ValueError(f"LLM extraction failed: {str(e)}") 