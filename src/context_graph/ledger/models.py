"""Documents stored in the processing ledger.

MongoDB holds only what is needed to resume extraction safely. Raw
conversation content stays in Neo4j and is never duplicated here (A.3).
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from pydantic import AwareDatetime, BaseModel, Field

from context_graph.domain.context import Lineage
from context_graph.domain.enums import BatchStatus
from context_graph.domain.extraction import ExtractionEnvelope


def utcnow() -> datetime:
    """Single source of the pipeline's wall clock."""
    return datetime.now(UTC)


class RunRecord(BaseModel):
    """Identifies a pipeline run, its graph target and its model versions."""

    run_id: str
    graph_database: str
    lineage: Lineage
    started_at: AwareDatetime = Field(default_factory=utcnow)
    updated_at: AwareDatetime = Field(default_factory=utcnow)


class SessionCheckpoint(BaseModel):
    """Highest contiguous Episode sequence committed for a run and session."""

    run_id: str
    session_id: str
    #: ``-1`` means nothing has been committed yet for this session.
    checkpoint_sequence: int = -1
    #: Completed sequences beyond the checkpoint, kept until the gap closes.
    completed_above: list[int] = Field(default_factory=list)
    failed_sequences: list[int] = Field(default_factory=list)
    updated_at: AwareDatetime = Field(default_factory=utcnow)

    def advanced_through(self, ordered_candidates: Sequence[int]) -> SessionCheckpoint:
        """Return the checkpoint advanced over completed sequences only.

        Walks the session's candidate sequences in order and stops at the first
        gap, so a failure at 104 keeps the checkpoint at 103 even when 105 has
        already succeeded (A.4).
        """
        completed = set(self.completed_above)
        sequence = self.checkpoint_sequence
        for candidate in ordered_candidates:
            if candidate <= sequence:
                continue
            if candidate not in completed:
                break
            sequence = candidate
            completed.discard(candidate)
        if sequence == self.checkpoint_sequence:
            return self
        return self.model_copy(
            update={
                "checkpoint_sequence": sequence,
                "completed_above": sorted(completed),
                "updated_at": utcnow(),
            }
        )


class SegmentationRecord(BaseModel):
    """Episodes a session was segmented into, so a rerun reuses the grouping.

    Grouping is model-assisted and therefore not bit-stable across runs. Storing
    the result keeps episode identities - and everything derived from them -
    stable when a run is resumed.
    """

    run_id: str
    session_id: str
    message_count: int
    episodes: list[dict[str, Any]] = Field(default_factory=list)
    updated_at: AwareDatetime = Field(default_factory=utcnow)


class ExtractionBatchRecord(BaseModel):
    """Validated extraction result and retry state for one batch."""

    run_id: str
    session_id: str
    batch_key: str
    episode_ids: list[str]
    sequences: list[int]
    status: BatchStatus = BatchStatus.PENDING
    attempts: int = 0
    result: ExtractionEnvelope | None = None
    error: str | None = None
    updated_at: AwareDatetime = Field(default_factory=utcnow)
