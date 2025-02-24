"""Module for error recovery and resilience patterns."""
from typing import TypeVar, Callable, Any, Optional, AsyncGenerator, Type
import asyncio
from datetime import datetime, timedelta
from functools import wraps
import logging
from contextlib import asynccontextmanager

T = TypeVar('T')

logger = logging.getLogger(__name__)


class CircuitBreaker:
    def __init__(
        self,
        failure_threshold: int = 5,
        reset_timeout: int = 60,
        half_open_timeout: int = 30
    ) -> None:
        """Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            reset_timeout: Time in seconds before attempting reset
            half_open_timeout: Time in seconds to stay in half-open state
        """
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.half_open_timeout = half_open_timeout
        
        self.failures = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = "closed"
        self._lock = asyncio.Lock()

    async def __aenter__(self) -> None:
        """Enter the circuit breaker context."""
        await self.before_call()

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any]
    ) -> None:
        """Exit the circuit breaker context.
        
        Args:
            exc_type: The type of the exception that was raised, if any
            exc_val: The instance of the exception that was raised, if any
            exc_tb: The traceback of the exception that was raised, if any
        """
        if exc_type is not None:
            await self.on_failure()
        else:
            await self.on_success()

    async def before_call(self) -> None:
        """Check circuit state before making a call."""
        async with self._lock:
            now = datetime.now()
            
            if self.state == "open":
                if (
                    self.last_failure_time
                    and now - self.last_failure_time > timedelta(
                        seconds=self.reset_timeout
                    )
                ):
                    self.state = "half-open"
                    logger.info("Circuit breaker entering half-open state")
                else:
                    raise CircuitBreakerError("Circuit is open")
            
            elif self.state == "half-open":
                if (
                    self.last_failure_time
                    and now - self.last_failure_time > timedelta(
                        seconds=self.half_open_timeout
                    )
                ):
                    self.state = "closed"
                    self.failures = 0
                    logger.info("Circuit breaker reset to closed state")

    async def on_success(self) -> None:
        """Handle successful call."""
        async with self._lock:
            if self.state == "half-open":
                self.state = "closed"
                self.failures = 0
                logger.info("Circuit breaker reset to closed state")

    async def on_failure(self) -> None:
        """Handle failed call."""
        async with self._lock:
            self.failures += 1
            self.last_failure_time = datetime.now()
            
            if self.failures >= self.failure_threshold:
                self.state = "open"
                logger.warning(
                    f"Circuit breaker opened after {self.failures} failures"
                )


class CircuitBreakerError(Exception):
    """Raised when circuit breaker prevents operation."""


def with_retry(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,)
) -> Callable:
    """Decorator for retrying async functions with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff: Multiplier for delay after each retry
        exceptions: Tuple of exceptions to catch and retry
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            current_delay = delay
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_retries:
                        break
                    
                    logger.warning(
                        f"Retry {attempt + 1}/{max_retries} "
                        f"after error: {str(e)}"
                    )
                    await asyncio.sleep(current_delay)
                    current_delay *= backoff
            
            raise last_exception or Exception("All retry attempts failed")
        return wrapper
    return decorator


@asynccontextmanager
async def resilient_operation(
    circuit_breaker: Optional[CircuitBreaker] = None,
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,)
) -> AsyncGenerator[None, None]:
    """Context manager combining circuit breaker and retries.
    
    Args:
        circuit_breaker: Optional circuit breaker instance
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff: Multiplier for delay after each retry
        exceptions: Tuple of exceptions to catch and retry
    """
    if circuit_breaker:
        await circuit_breaker.before_call()
    
    current_delay = delay
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            yield
            if circuit_breaker:
                await circuit_breaker.on_success()
            return
        except exceptions as e:
            last_exception = e
            if attempt == max_retries:
                break
            
            logger.warning(
                f"Retry {attempt + 1}/{max_retries} after error: {str(e)}"
            )
            if circuit_breaker:
                await circuit_breaker.on_failure()
            await asyncio.sleep(current_delay)
            current_delay *= backoff
    
    if circuit_breaker:
        await circuit_breaker.on_failure()
    raise last_exception or Exception("All retry attempts failed") 