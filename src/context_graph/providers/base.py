"""Provider interfaces.

Core business logic depends only on these protocols, so a provider can be
swapped without touching the pipeline.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from pydantic import BaseModel


@runtime_checkable
class LLMProvider(Protocol):
    """Structured-output language model used for extraction and adjudication."""

    @property
    def model(self) -> str:
        """Model identifier recorded in context lineage."""
        ...

    def estimate_tokens(self, text: str) -> int:
        """Cheap local token estimate used for batching decisions."""
        ...

    async def count_tokens(self, *, system: str, user: str) -> int:
        """Accurate, model-specific token count for an assembled request."""
        ...

    async def structured[M: BaseModel](
        self,
        *,
        system: str,
        user: str,
        schema: type[M],
        max_output_tokens: int,
    ) -> M:
        """Return a validated instance of ``schema``.

        Implementations must raise :class:`~context_graph.errors.TransientError`
        for retryable failures and
        :class:`~context_graph.errors.ExtractionValidationError` when the model
        output is truncated or does not satisfy ``schema``.
        """
        ...


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Embedding model used for semantic candidate discovery."""

    @property
    def model(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed ``texts`` in one call, preserving input order."""
        ...
