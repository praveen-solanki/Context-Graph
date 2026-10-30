"""Public tool surface.

``ingest_conversation`` is the single entry point for the complete ingestion
pipeline. New, incremental, bulk and historical input all use this one tool;
the caller never selects an internal processing mode (A.13).
"""

from __future__ import annotations

from collections.abc import Sequence
from types import TracebackType
from typing import Self

from pydantic import BaseModel, ConfigDict

from context_graph.config import Settings, get_settings
from context_graph.domain.conversation import SessionInput
from context_graph.domain.ids import new_id
from context_graph.graph.neo4j_store import Neo4jGraphStore
from context_graph.ledger.mongo import MongoLedger
from context_graph.observability import configure_logging
from context_graph.pipeline.orchestrator import IngestionPipeline, IngestionReport
from context_graph.providers.chat_backend import (
    EmbeddingProviderImpl,
    ChatLLMProvider,
    ChatTransport,
)
from context_graph.providers.base import EmbeddingProvider, LLMProvider


class IngestionRequest(BaseModel):
    """Input accepted by :func:`ingest_conversation`.

    Conversation is the only input. Scope membership is authorization metadata
    with no authoritative source in a conversation, so it is never taken from
    the caller and never inferred from what a user says about themselves.
    """

    model_config = ConfigDict(frozen=True)

    session: SessionInput


class ContextGraphIngestion:
    """Owns the pipeline's external resources for the lifetime of a run."""

    def __init__(
        self,
        *,
        settings: Settings,
        store: Neo4jGraphStore,
        ledger: MongoLedger,
        llm: LLMProvider,
        embeddings: EmbeddingProvider,
        run_id: str | None = None,
        transport: ChatTransport | None = None,
    ) -> None:
        self._store = store
        self._ledger = ledger
        self._transport = transport
        self._pipeline = IngestionPipeline(
            settings=settings,
            store=store,
            ledger=ledger,
            llm=llm,
            embeddings=embeddings,
            run_id=run_id or settings.run_id or new_id("run"),
        )
        self._prepared = False

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> Self:
        """Build the default Neo4j + MongoDB + chat-backend stack."""
        resolved = settings or get_settings()
        configure_logging(resolved.log_level, resolved.log_file)
        # One transport serves extraction and embeddings, so concurrency and
        # retries are limited once for the whole chat backend.
        transport = ChatTransport(resolved.chat, retry=resolved.retry)
        return cls(
            settings=resolved,
            store=Neo4jGraphStore(resolved.neo4j),
            ledger=MongoLedger(resolved.mongo),
            llm=ChatLLMProvider(
                backend=resolved.chat, llm=resolved.llm, transport=transport
            ),
            embeddings=EmbeddingProviderImpl(
                embedding=resolved.embedding, retry=resolved.retry
            ),
            transport=transport,
        )

    async def __aenter__(self) -> Self:
        await self.prepare()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def prepare(self) -> None:
        """Ensure schema, indexes and the run record exist (idempotent)."""
        if not self._prepared:
            await self._pipeline.prepare()
            self._prepared = True

    async def ingest_conversation(self, request: IngestionRequest) -> IngestionReport:
        """Run the complete ingestion pipeline for one session."""
        await self.prepare()
        return await self._pipeline.ingest(request.session)

    async def ingest_conversations(
        self, requests: Sequence[IngestionRequest]
    ) -> list[IngestionReport]:
        """Ingest independent sessions concurrently, bounded by configuration."""
        await self.prepare()
        return await self._pipeline.ingest_many([request.session for request in requests])

    async def aclose(self) -> None:
        """Release graph, ledger and provider connections."""
        await self._store.close()
        await self._ledger.close()
        if self._transport is not None:
            await self._transport.aclose()


async def ingest_conversation(
    request: IngestionRequest, settings: Settings | None = None
) -> IngestionReport:
    """Ingest one conversation with a self-contained resource lifetime."""
    async with ContextGraphIngestion.from_settings(settings) as ingestion:
        return await ingestion.ingest_conversation(request)
