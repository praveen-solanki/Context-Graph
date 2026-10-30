"""MongoDB implementation of the processing ledger."""

from __future__ import annotations

from collections.abc import Awaitable, Sequence
from types import TracebackType
from typing import Any, Self

from pymongo import AsyncMongoClient, ReturnDocument
from pymongo.errors import (
    AutoReconnect,
    ConnectionFailure,
    NetworkTimeout,
    PyMongoError,
    ServerSelectionTimeoutError,
)

from context_graph.config import MongoSettings
from context_graph.domain.enums import BatchStatus
from context_graph.domain.extraction import ExtractionBatch, ExtractionEnvelope
from context_graph.errors import ContextGraphError, TransientError
from context_graph.ledger.models import (
    ExtractionBatchRecord,
    RunRecord,
    SegmentationRecord,
    SessionCheckpoint,
    utcnow,
)
from context_graph.observability import get_logger, log_event

_logger = get_logger("ledger.mongo")

_RUNS = "runs"
_SEGMENTATIONS = "session_segmentations"
_CHECKPOINTS = "session_checkpoints"
_BATCHES = "extraction_batches"


class MongoLedger:
    """Checkpoint and extraction-result store for resumable ingestion."""

    def __init__(self, settings: MongoSettings, client: AsyncMongoClient[Any] | None = None) -> None:
        self._client = client or AsyncMongoClient(settings.uri)
        self._db = self._client[settings.database]

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.close()

    async def close(self) -> None:
        """Close the MongoDB connection pool."""
        await self._client.close()

    async def ensure_indexes(self) -> None:
        """Create the uniqueness indexes the ledger relies on."""
        try:
            await self._db[_RUNS].create_index("run_id", unique=True)
            await self._db[_CHECKPOINTS].create_index(["run_id", "session_id"], unique=True)
            await self._db[_BATCHES].create_index(["run_id", "batch_key"], unique=True)
            await self._db[_SEGMENTATIONS].create_index(["run_id", "session_id"], unique=True)
        except PyMongoError as error:
            raise self._translate(error, "ensure_indexes") from error

    async def upsert_run(self, run: RunRecord) -> None:
        """Record (or refresh) this run's configuration."""
        document = run.model_dump(mode="json")
        document["updated_at"] = utcnow().isoformat()
        await self._execute(
            self._db[_RUNS].update_one({"run_id": run.run_id}, {"$set": document}, upsert=True),
            "upsert_run",
        )

    async def checkpoint(self, run_id: str, session_id: str) -> SessionCheckpoint:
        """Current checkpoint for a session; a missing one starts before sequence 0."""
        document = await self._execute(
            self._db[_CHECKPOINTS].find_one({"run_id": run_id, "session_id": session_id}),
            "checkpoint",
        )
        if document is None:
            return SessionCheckpoint(run_id=run_id, session_id=session_id)
        return SessionCheckpoint.model_validate(document)

    async def segmentation(self, run_id: str, session_id: str) -> SegmentationRecord | None:
        """Episodes this run already segmented a session into, if any."""
        document = await self._execute(
            self._db[_SEGMENTATIONS].find_one({"run_id": run_id, "session_id": session_id}),
            "segmentation",
        )
        return None if document is None else SegmentationRecord.model_validate(document)

    async def save_segmentation(self, record: SegmentationRecord) -> None:
        """Store a session's grouping so a resumed run reproduces it exactly."""
        document = record.model_dump(mode="json")
        await self._execute(
            self._db[_SEGMENTATIONS].update_one(
                {"run_id": record.run_id, "session_id": record.session_id},
                {"$set": document},
                upsert=True,
            ),
            "save_segmentation",
        )

    async def batch_record(self, run_id: str, batch_key: str) -> ExtractionBatchRecord | None:
        """Stored result for a batch, so successful LLM work is never repeated."""
        document = await self._execute(
            self._db[_BATCHES].find_one({"run_id": run_id, "batch_key": batch_key}),
            "batch_record",
        )
        return None if document is None else ExtractionBatchRecord.model_validate(document)

    async def start_batch(self, batch: ExtractionBatch) -> ExtractionBatchRecord:
        """Register an attempt for a batch and return its ledger record."""
        document = await self._execute(
            self._db[_BATCHES].find_one_and_update(
                {"run_id": batch.run_id, "batch_key": batch.batch_key},
                {
                    "$set": {
                        "session_id": batch.session_id,
                        "episode_ids": list(batch.episode_ids),
                        "sequences": list(batch.sequences),
                        "status": BatchStatus.PENDING.value,
                        "updated_at": utcnow().isoformat(),
                    },
                    "$inc": {"attempts": 1},
                },
                upsert=True,
                return_document=ReturnDocument.AFTER,
            ),
            "start_batch",
        )
        return ExtractionBatchRecord.model_validate(document)

    async def save_extraction(
        self, run_id: str, batch_key: str, envelope: ExtractionEnvelope
    ) -> None:
        """Persist a validated extraction result before any graph write."""
        await self._set_batch(
            run_id,
            batch_key,
            {
                "status": BatchStatus.VALIDATED.value,
                "result": envelope.model_dump(mode="json"),
                "error": None,
            },
        )

    async def mark_committed(self, run_id: str, batch_key: str) -> None:
        """Mark the batch's graph write as complete."""
        await self._set_batch(run_id, batch_key, {"status": BatchStatus.COMMITTED.value})

    async def mark_failed(self, run_id: str, batch_key: str, error: str) -> None:
        """Record a terminal failure with its diagnostic message."""
        await self._set_batch(
            run_id, batch_key, {"status": BatchStatus.FAILED.value, "error": error}
        )
        log_event(_logger, "ledger.batch_failed", run_id=run_id, batch_key=batch_key, error=error)

    async def record_completed(
        self, run_id: str, session_id: str, sequences: Sequence[int]
    ) -> None:
        """Record sequences whose graph write succeeded."""
        if not sequences:
            return
        await self._update_checkpoint(
            run_id,
            session_id,
            {
                "$addToSet": {"completed_above": {"$each": sorted(sequences)}},
                "$pull": {"failed_sequences": {"$in": list(sequences)}},
            },
        )

    async def record_failed(self, run_id: str, session_id: str, sequences: Sequence[int]) -> None:
        """Record sequences that remain eligible for retry."""
        if not sequences:
            return
        await self._update_checkpoint(
            run_id,
            session_id,
            {"$addToSet": {"failed_sequences": {"$each": sorted(sequences)}}},
        )

    async def advance_checkpoint(
        self, run_id: str, session_id: str, ordered_candidates: Sequence[int]
    ) -> SessionCheckpoint:
        """Advance the checkpoint through completed sequences only.

        The contiguity rule itself lives on
        :class:`~context_graph.ledger.models.SessionCheckpoint` (A.4).
        """
        current = await self.checkpoint(run_id, session_id)
        advanced = current.advanced_through(ordered_candidates)
        if advanced is current:
            return current

        await self._update_checkpoint(
            run_id,
            session_id,
            {
                "$set": {
                    "checkpoint_sequence": advanced.checkpoint_sequence,
                    "completed_above": advanced.completed_above,
                    "updated_at": advanced.updated_at.isoformat(),
                }
            },
        )
        log_event(
            _logger,
            "ledger.checkpoint_advanced",
            run_id=run_id,
            session_id=session_id,
            checkpoint=advanced.checkpoint_sequence,
            pending_above=len(advanced.completed_above),
        )
        return advanced

    async def _set_batch(self, run_id: str, batch_key: str, changes: dict[str, Any]) -> None:
        await self._execute(
            self._db[_BATCHES].update_one(
                {"run_id": run_id, "batch_key": batch_key},
                {"$set": {**changes, "updated_at": utcnow().isoformat()}},
            ),
            "update_batch",
        )

    async def _update_checkpoint(
        self, run_id: str, session_id: str, update: dict[str, Any]
    ) -> None:
        payload = dict(update)
        payload.setdefault("$set", {})["updated_at"] = utcnow().isoformat()
        await self._execute(
            self._db[_CHECKPOINTS].update_one(
                {"run_id": run_id, "session_id": session_id}, payload, upsert=True
            ),
            "update_checkpoint",
        )

    @staticmethod
    async def _execute[T](awaitable: Awaitable[T], operation: str) -> T:
        """Await a driver call, translating failures into pipeline errors."""
        try:
            return await awaitable
        except PyMongoError as error:
            raise MongoLedger._translate(error, operation) from error

    @staticmethod
    def _translate(error: PyMongoError, operation: str) -> ContextGraphError:
        transient = (AutoReconnect, ConnectionFailure, NetworkTimeout, ServerSelectionTimeoutError)
        if isinstance(error, transient):
            return TransientError(f"mongo {operation} failed transiently: {error}")
        return ContextGraphError(f"mongo {operation} failed: {error}")
