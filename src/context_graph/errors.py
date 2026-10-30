"""Error hierarchy for the ingestion pipeline.

Errors carry enough information to decide whether work may be retried; nothing
in the pipeline is allowed to swallow a failure silently.
"""

from __future__ import annotations


class ContextGraphError(Exception):
    """Base class for all pipeline errors."""


class ConfigurationError(ContextGraphError):
    """Invalid or missing configuration."""


class TransientError(ContextGraphError):
    """A failure that may succeed on retry (network, rate limit, timeout)."""


class ProviderError(ContextGraphError):
    """Non-retryable failure returned by an external provider."""


class ExtractionValidationError(ContextGraphError):
    """LLM output failed schema validation, was truncated, or exceeded limits."""


class GraphWriteError(ContextGraphError):
    """A durable graph write failed."""


class UnresolvedSignalError(ContextGraphError):
    """A signal could not be resolved safely and must not be written."""
