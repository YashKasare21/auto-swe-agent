"""Exponential backoff retry decorator for LLM API calls."""

from __future__ import annotations

import functools
import time
from typing import Callable, Tuple, TypeVar

T = TypeVar("T")


def with_retry(
    max_retries: int = 3,
    base_delay: float = 2.0,
    max_delay: float = 30.0,
    exponential_base: float = 2.0,
    retryable_exceptions: Tuple[type, ...] = (Exception,),
) -> Callable:
    """Retry a function with exponential backoff.

    Delay sequence: base_delay * (exponential_base ** attempt), capped at max_delay.
    Only retries on exceptions listed in retryable_exceptions.
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exc: Exception | None = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exc = e
                    if attempt >= max_retries:
                        raise
                    delay = min(base_delay * (exponential_base**attempt), max_delay)
                    print(
                        f"[RETRY] attempt {attempt + 1}/{max_retries} failed ({type(e).__name__}). "
                        f"Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
            raise last_exc  # unreachable, but satisfies type checkers

        return wrapper

    return decorator
