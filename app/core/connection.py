"""Module for managing connections in the Document Q&A application."""
from typing import AsyncGenerator
import asyncio
import collections
import time
from contextlib import asynccontextmanager
import httpx


class RateLimiter:
    def __init__(self, max_requests: int, time_window: int) -> None:
        """Initialize rate limiter.
   
        Args:
            max_requests: Maximum number of requests allowed in time window
            time_window: Time window in seconds
        """
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests: collections.deque[float] = collections.deque()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Acquire a rate limit slot."""
        async with self._lock:
            now = time.time()

            # Remove expired timestamps
            while self.requests and now - self.requests[0] > self.time_window:
                self.requests.popleft()

            # If at limit, wait until oldest request expires
            if len(self.requests) >= self.max_requests:
                wait_time = self.requests[0] + self.time_window - now
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
                self.requests.popleft()

            # Add current request
            self.requests.append(now)


class ConnectionPool:
    def __init__(
        self,
        pool_size: int = 10,
        max_requests: int = 60,
        time_window: int = 60
    ) -> None:
        """Initialize connection pool with rate limiting.

        Args:
            pool_size: Maximum number of concurrent connections
            max_requests: Maximum requests per time window
            time_window: Time window in seconds
        """
        self.pool_size = pool_size
        self.semaphore = asyncio.Semaphore(pool_size)
        self.rate_limiter = RateLimiter(max_requests, time_window)
        self._clients: list[httpx.AsyncClient] = []

    async def initialize(self) -> None:
        """Initialize the connection pool."""
        self._clients = [
            httpx.AsyncClient(timeout=30.0)
            for _ in range(self.pool_size)
        ]

    async def cleanup(self) -> None:
        """Clean up all connections."""
        await asyncio.gather(
            *(client.aclose() for client in self._clients)
        )
        self._clients.clear()

    @asynccontextmanager
    async def get_client(self) -> AsyncGenerator[httpx.AsyncClient, None]:
        """Get a client from the pool with rate limiting."""
        if not self._clients:
            await self.initialize()

        async with self.semaphore:
            await self.rate_limiter.acquire()
            client = (
                self._clients.pop() 
                if self._clients 
                else httpx.AsyncClient()
            )
            try:
                yield client
            finally:
                if len(self._clients) < self.pool_size:
                    self._clients.append(client)
                else:
                    await client.aclose()


class ConnectionManager:
    """Manages connections to the document service."""

    def __init__(self) -> None:
        """Initialize the connection manager."""
        pass

    def connect(self) -> None:
        """Establish a connection."""
        # connection logic here

    def disconnect(self) -> None:
        """Terminate the connection."""
        # disconnection logic here


# Global connection pool instance
pool = ConnectionPool() 