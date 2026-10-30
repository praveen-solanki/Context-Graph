"""Deterministic in-memory providers used by unit and end-to-end tests."""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Sequence
from datetime import datetime
from hashlib import blake2b

from pydantic import BaseModel

from context_graph.domain.context import (
    SHARED_SUBJECT,
    ContextInstance,
    ContextSupport,
    EvidenceLink,
    MemoryConcept,
)
from context_graph.domain.conversation import Episode, SessionInput
from context_graph.domain.enums import (
    Authority,
    BatchStatus,
    ContextStatus,
    ScopeType,
    SourceTrust,
)
from context_graph.domain.extraction import (
    ExtractionBatch,
    ExtractionEnvelope,
    ProposedEpisode,
    SegmentationProposal,
)
from context_graph.errors import ExtractionValidationError
from context_graph.graph.neo4j_store import Neo4jGraphStore
from context_graph.ledger.models import (
    ExtractionBatchRecord,
    RunRecord,
    SegmentationRecord,
    SessionCheckpoint,
)
from context_graph.ledger.mongo import MongoLedger


class FakeLLMProvider:
    """Scripted :class:`~context_graph.providers.base.LLMProvider`."""

    def __init__(
        self,
        responses: Sequence[BaseModel] | None = None,
        *,
        responder: Callable[[str, type[BaseModel]], BaseModel] | None = None,
        chars_per_token: float = 4.0,
    ) -> None:
        self._responses = list(responses or [])
        self._responder = responder
        self._chars_per_token = chars_per_token
        self.calls: list[tuple[str, str]] = []

    @property
    def model(self) -> str:
        return "fake-llm"

    def estimate_tokens(self, text: str) -> int:
        return math.ceil(len(text) / self._chars_per_token)

    async def count_tokens(self, *, system: str, user: str) -> int:
        return self.estimate_tokens(system) + self.estimate_tokens(user)

    async def structured[M: BaseModel](
        self, *, system: str, user: str, schema: type[M], max_output_tokens: int
    ) -> M:
        if schema is SegmentationProposal:
            # Default grouping for tests: one episode per source span, which
            # reproduces the pre-segmentation episode shape.
            proposal = SegmentationProposal(
                episodes=tuple(
                    ProposedEpisode(span_refs=(ref,)) for ref in _source_refs(user)
                )
            )
            return proposal  # pyright: ignore[reportReturnType]
        self.calls.append((system, user))
        if self._responder is not None:
            result = self._responder(user, schema)
        elif self._responses:
            result = self._responses.pop(0)
        else:
            raise ExtractionValidationError("fake provider has no scripted response")
        if isinstance(result, Exception):
            raise result
        if not isinstance(result, schema):
            raise ExtractionValidationError(f"fake response is not a {schema.__name__}")
        return result


class FakeEmbeddingProvider:
    """Deterministic hash embeddings: equal text gives equal vectors."""

    def __init__(self, dimension: int = 16) -> None:
        self._dimension = dimension
        self.calls = 0

    @property
    def model(self) -> str:
        return "fake-embedding"

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls += 1
        return [self._vector(text) for text in texts]

    def _vector(self, text: str) -> list[float]:
        digest = blake2b(text.casefold().encode("utf-8"), digest_size=self._dimension).digest()
        raw = [byte / 255.0 for byte in digest]
        norm = math.sqrt(sum(value * value for value in raw)) or 1.0
        return [value / norm for value in raw]


class InMemoryGraphStore(Neo4jGraphStore):
    """In-memory stand-in for the Neo4j store with the same write semantics."""

    def __init__(self) -> None:  # noqa: D107 - deliberately skips driver setup
        self.sessions: dict[str, SessionInput] = {}
        self.episodes: dict[tuple[str, str], Episode] = {}
        self.concepts: dict[str, MemoryConcept] = {}
        self.concept_embeddings: dict[str, list[float]] = {}
        self.contexts: dict[str, ContextInstance] = {}
        self.evidence: dict[str, dict[str, EvidenceLink]] = {}
        self.context_embeddings: dict[str, list[float]] = {}
        self.schema_ready = False

    async def close(self) -> None:
        return None

    async def ensure_schema(self, embedding_dimension: int) -> None:
        self.schema_ready = True

    async def persist_session(self, session: SessionInput) -> None:
        self.sessions[session.session_id] = session

    async def persist_episodes(self, episodes: Sequence[Episode]) -> int:
        for episode in episodes:
            self.episodes.setdefault((episode.session_id, episode.episode_id), episode)
        return len(episodes)

    async def count_episodes(self, session_id: str) -> int:
        return sum(1 for key in self.episodes if key[0] == session_id)

    async def episodes_after(self, session_id: str, after_sequence: int, limit: int) -> list[Episode]:
        selected = [
            episode
            for (stored_session, _), episode in self.episodes.items()
            if stored_session == session_id and episode.sequence > after_sequence
        ]
        return sorted(selected, key=lambda episode: episode.sequence)[:limit]

    async def merge_concept(
        self, concept: MemoryConcept, embedding: list[float] | None
    ) -> MemoryConcept:
        stored = self.concepts.setdefault(concept.concept_key, concept)
        if embedding is not None:
            self.concept_embeddings.setdefault(stored.concept_key, embedding)
        return stored

    async def concept_by_key(self, concept_key: str) -> MemoryConcept | None:
        return self.concepts.get(concept_key)

    async def search_concepts(
        self, embedding: Sequence[float], top_k: int
    ) -> list[tuple[MemoryConcept, float]]:
        scored = [
            (self.concepts[key], _cosine(embedding, vector))
            for key, vector in self.concept_embeddings.items()
        ]
        return sorted(scored, key=lambda item: item[1], reverse=True)[:top_k]

    async def search_contexts(
        self, *, embedding: Sequence[float], top_k: int, subject_user_id: str
    ) -> list[tuple[ContextInstance, float]]:
        visible = (subject_user_id, SHARED_SUBJECT)
        live = (ContextStatus.ACTIVE, ContextStatus.CANDIDATE)
        scored = [
            (context, _cosine(embedding, self.context_embeddings[context.context_id]))
            for context in self.contexts.values()
            if context.subject_user_id in visible
            and context.status in live
            and context.context_id in self.context_embeddings
        ]
        return sorted(scored, key=lambda item: item[1], reverse=True)[:top_k]

    async def active_contexts(
        self, *, subject_user_id: str, scope_type: ScopeType, scope_id: str, concept_id: str
    ) -> list[ContextInstance]:
        live = (ContextStatus.ACTIVE, ContextStatus.CANDIDATE)
        matches = [
            context
            for context in self.contexts.values()
            if context.subject_user_id == subject_user_id
            and context.scope_type is scope_type
            and context.scope_id == scope_id
            and context.concept_id == concept_id
            and context.status in live
        ]
        return sorted(matches, key=lambda context: context.observed_at, reverse=True)

    async def write_context(
        self,
        *,
        context: ContextInstance,
        embedding: list[float] | None,
        evidence: EvidenceLink,
        supersedes_context_ids: Sequence[str] = (),
    ) -> ContextInstance:
        existing = next(
            (c for c in self.contexts.values() if c.fingerprint == context.fingerprint), None
        )
        stored = (
            context
            if existing is None
            else existing.model_copy(
                update={
                    "status": context.status,
                    "superseded_at": None,
                    "observed_at": max(existing.observed_at, context.observed_at),
                }
            )
        )
        self.contexts[stored.context_id] = stored
        if embedding is not None:
            self.context_embeddings.setdefault(stored.context_id, embedding)
        await self.add_evidence(evidence.model_copy(update={"context_id": stored.context_id}))
        for old_id in supersedes_context_ids:
            if old_id == stored.context_id:
                continue
            old = self.contexts[old_id]
            self.contexts[old_id] = old.model_copy(
                update={
                    "status": ContextStatus.SUPERSEDED,
                    "superseded_at": context.recorded_at,
                }
            )
        return stored

    async def add_evidence(self, evidence: EvidenceLink) -> int:
        links = self.evidence.setdefault(evidence.context_id, {})
        links.setdefault(evidence.episode_id, evidence)
        return len(links)

    async def revoke_context(
        self, *, context_id: str, session_id: str, episode_id: str, revoked_at: datetime
    ) -> ContextInstance | None:
        existing = self.contexts.get(context_id)
        if existing is None:
            return None
        revoked = existing.model_copy(update={"status": ContextStatus.REVOKED})
        self.contexts[context_id] = revoked
        return revoked

    async def support_for(self, context_ids: Sequence[str]) -> dict[str, ContextSupport]:
        supports: dict[str, ContextSupport] = {}
        for context_id in context_ids:
            links = list(self.evidence.get(context_id, {}).values())
            context = self.contexts[context_id]
            supports[context_id] = ContextSupport(
                context_id=context_id,
                evidence_count=len(links),
                distinct_sessions=len({link.session_id for link in links}),
                distinct_users=len(
                    {
                        link.actor_id
                        for link in links
                        if link.source_trust is SourceTrust.FIRST_PARTY
                    }
                ),
                first_observed_at=min((link.observed_at for link in links), default=None),
                last_observed_at=max((link.observed_at for link in links), default=None),
                has_authoritative_source=context.authority is Authority.AUTHORITATIVE,
            )
        return supports

    async def candidates(self, limit: int) -> list[ContextInstance]:
        pending = [
            context
            for context in self.contexts.values()
            if context.status is ContextStatus.CANDIDATE
        ]
        return sorted(pending, key=lambda context: context.recorded_at)[:limit]

    async def set_status(
        self, context_id: str, status: ContextStatus, now: datetime
    ) -> ContextInstance | None:
        existing = self.contexts.get(context_id)
        if existing is None:
            return None
        updated = existing.model_copy(update={"status": status})
        self.contexts[context_id] = updated
        return updated

class InMemoryLedger(MongoLedger):
    """In-memory stand-in for the MongoDB processing ledger."""

    def __init__(self) -> None:  # noqa: D107 - deliberately skips client setup
        self.runs: dict[str, RunRecord] = {}
        self.segmentations: dict[tuple[str, str], SegmentationRecord] = {}
        self.checkpoints: dict[tuple[str, str], SessionCheckpoint] = {}
        self.batches: dict[tuple[str, str], ExtractionBatchRecord] = {}
        self.indexes_ready = False

    async def close(self) -> None:
        return None

    async def ensure_indexes(self) -> None:
        self.indexes_ready = True

    async def upsert_run(self, run: RunRecord) -> None:
        self.runs[run.run_id] = run

    async def checkpoint(self, run_id: str, session_id: str) -> SessionCheckpoint:
        return self.checkpoints.setdefault(
            (run_id, session_id), SessionCheckpoint(run_id=run_id, session_id=session_id)
        )

    async def segmentation(self, run_id: str, session_id: str) -> SegmentationRecord | None:
        return self.segmentations.get((run_id, session_id))

    async def save_segmentation(self, record: SegmentationRecord) -> None:
        self.segmentations[(record.run_id, record.session_id)] = record

    async def batch_record(self, run_id: str, batch_key: str) -> ExtractionBatchRecord | None:
        return self.batches.get((run_id, batch_key))

    async def start_batch(self, batch: ExtractionBatch) -> ExtractionBatchRecord:
        key = (batch.run_id, batch.batch_key)
        existing = self.batches.get(key)
        record = ExtractionBatchRecord(
            run_id=batch.run_id,
            session_id=batch.session_id,
            batch_key=batch.batch_key,
            episode_ids=list(batch.episode_ids),
            sequences=list(batch.sequences),
            attempts=(existing.attempts if existing else 0) + 1,
            result=existing.result if existing else None,
        )
        self.batches[key] = record
        return record

    async def save_extraction(
        self, run_id: str, batch_key: str, envelope: ExtractionEnvelope
    ) -> None:
        record = self.batches[(run_id, batch_key)]
        self.batches[(run_id, batch_key)] = record.model_copy(
            update={"result": envelope, "status": BatchStatus.VALIDATED, "error": None}
        )

    async def mark_committed(self, run_id: str, batch_key: str) -> None:
        record = self.batches[(run_id, batch_key)]
        self.batches[(run_id, batch_key)] = record.model_copy(
            update={"status": BatchStatus.COMMITTED}
        )

    async def mark_failed(self, run_id: str, batch_key: str, error: str) -> None:
        record = self.batches[(run_id, batch_key)]
        self.batches[(run_id, batch_key)] = record.model_copy(
            update={"status": BatchStatus.FAILED, "error": error}
        )

    async def record_completed(
        self, run_id: str, session_id: str, sequences: Sequence[int]
    ) -> None:
        current = await self.checkpoint(run_id, session_id)
        self.checkpoints[(run_id, session_id)] = current.model_copy(
            update={
                "completed_above": sorted(set(current.completed_above) | set(sequences)),
                "failed_sequences": [
                    sequence
                    for sequence in current.failed_sequences
                    if sequence not in set(sequences)
                ],
            }
        )

    async def record_failed(self, run_id: str, session_id: str, sequences: Sequence[int]) -> None:
        current = await self.checkpoint(run_id, session_id)
        self.checkpoints[(run_id, session_id)] = current.model_copy(
            update={"failed_sequences": sorted(set(current.failed_sequences) | set(sequences))}
        )

    async def advance_checkpoint(
        self, run_id: str, session_id: str, ordered_candidates: Sequence[int]
    ) -> SessionCheckpoint:
        current = await self.checkpoint(run_id, session_id)
        advanced = current.advanced_through(ordered_candidates)
        self.checkpoints[(run_id, session_id)] = advanced
        return advanced


_SOURCE_REF = re.compile(r"^\[(s\d+-\d+) ", re.MULTILINE)


def _source_refs(user: str) -> list[str]:
    """Source references from a rendered segmentation request, in order."""
    body = user.split("SOURCE (group all of these):", 1)[-1]
    return _SOURCE_REF.findall(body)


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    return max(-1.0, min(1.0, dot))
