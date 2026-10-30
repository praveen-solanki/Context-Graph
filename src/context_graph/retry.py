"""Retry policy for transient failures.

Only :class:`~context_graph.errors.TransientError` is retried, with bounded
exponential backoff (A.13). Every attempt is logged, and the final failure is
re-raised so no work is ever silently dropped.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable

from context_graph.config import RetrySettings
from context_graph.errors import TransientError
from context_graph.observability import get_logger, log_event

_logger = get_logger("retry")


async def retry_async[T](
    operation: Callable[[], Awaitable[T]],
    *,
    settings: RetrySettings,
    description: str,
) -> T:
    """Run ``operation``, retrying only transient failures."""
    last_error: TransientError | None = None
    for attempt in range(1, settings.max_attempts + 1):
        try:
            return await operation()
        except TransientError as error:
            last_error = error
            if attempt == settings.max_attempts:
                break
            delay = min(
                settings.base_backoff_seconds * 2 ** (attempt - 1),
                settings.max_backoff_seconds,
            )
            log_event(
                _logger,
                "retry.attempt_failed",
                operation=description,
                attempt=attempt,
                max_attempts=settings.max_attempts,
                delay_seconds=round(delay, 3),
                error=str(error),
            )
            await asyncio.sleep(delay * (0.5 + random.random()))
    assert last_error is not None
    log_event(
        _logger,
        "retry.exhausted",
        operation=description,
        attempts=settings.max_attempts,
        error=str(last_error),
    )
    raise last_error
