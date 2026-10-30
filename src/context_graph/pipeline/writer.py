"""Durable context writes: values, supersession, evidence and revocation.

Values are append-only. The same value refreshes evidence on the existing
instance; a changed value creates a new instance that supersedes the old one at
the same scope. Explicit withdrawal is recorded as a revocation, never as an
ordinary contradiction (spec section 4 and 10).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime

from pydantic import BaseModel, ConfigDict

from context_graph.domain.context import (
    ContextInstance,
    EvidenceLink,
    Lineage,
    Provenance,
)
from context_graph.domain.conversation import Episode, SessionInput
from context_graph.domain.enums import Cardinality, ContextStatus, SignalOperation
from context_graph.domain.extraction import ExtractedSignal
from context_graph.domain.ids import canonical_value, new_id
from context_graph.errors import UnresolvedSignalError
from context_graph.graph.neo4j_store import Neo4jGraphStore
from context_graph.ledger.models import utcnow
from context_graph.observability import get_logger, log_event
from context_graph.pipeline.policy import ScopeAssignment
from context_graph.pipeline.resolution import ConceptResolution

_logger = get_logger("pipeline.writer")


class ContextWriteResult(BaseModel):
    """What a single signal changed in the graph."""

    model_config = ConfigDict(frozen=True)

    context_id: str
    concept_key: str
    status: ContextStatus
    superseded_context_ids: tuple[str, ...] = ()
    revoked_context_ids: tuple[str, ...] = ()
    evidence_episode_ids: tuple[str, ...] = ()


class ContextWriter:
    """Create or reuse canonical context and attach its evidence."""

    def __init__(self, store: Neo4jGraphStore) -> None:
        self._store = store

    async def write(
        self,
        *,
        signal: ExtractedSignal,
        resolution: ConceptResolution,
        decision: ScopeAssignment,
        session: SessionInput,
        episodes: Mapping[str, Episode],
        lineage: Lineage,
        embedding: list[float] | None,
    ) -> ContextWriteResult:
        """Apply one extracted signal to the graph."""
        supporting = self._supporting_episodes(signal, episodes)
        primary = supporting[0]
        concept = resolution.concept

        if signal.operation is SignalOperation.REVOKE:
            return await self._revoke(signal, resolution, decision, primary)

        recorded_at = utcnow()
        cardinality = signal.cardinality or concept.cardinality
        context = ContextInstance(
            context_id=new_id("ctx"),
            fingerprint=ContextInstance.fingerprint_for(
                subject_user_id=decision.subject_user_id,
                scope_type=decision.scope_type,
                scope_id=decision.scope_id,
                concept_id=concept.concept_id,
                value=signal.value,
            ),
            context_type=signal.context_type,
            concept_id=concept.concept_id,
            concept_key=concept.concept_key,
            subject_user_id=decision.subject_user_id,
            scope_type=decision.scope_type,
            scope_id=decision.scope_id,
            value=signal.value,
            cardinality=cardinality,
            condition=signal.condition,
            visibility=decision.visibility,
            authority=decision.authority,
            status=decision.status,
            observed_at=primary.timestamp,
            recorded_at=recorded_at,
            valid_from=signal.valid_from or primary.timestamp,
            valid_until=signal.valid_until,
            provenance=Provenance(
                source_episode_id=primary.episode_id,
                source_session_id=session.session_id,
                source_actor=primary.sender_id,
                source_type=primary.source_type,
                extraction_confidence=signal.confidence,
            ),
            lineage=lineage.model_copy(
                update={"adjudicator_model": resolution.adjudicator_model}
            ),
        )

        superseded = await self._superseded_targets(context, cardinality)
        stored = await self._store.write_context(
            context=context,
            embedding=embedding,
            evidence=self._evidence(context.context_id, signal, session, primary, recorded_at),
            supersedes_context_ids=superseded,
        )
        for episode in supporting[1:]:
            await self._store.add_evidence(
                self._evidence(stored.context_id, signal, session, episode, recorded_at)
            )

        return ContextWriteResult(
            context_id=stored.context_id,
            concept_key=stored.concept_key,
            status=stored.status,
            superseded_context_ids=tuple(superseded),
            evidence_episode_ids=tuple(episode.episode_id for episode in supporting),
        )

    async def _superseded_targets(
        self, context: ContextInstance, cardinality: Cardinality
    ) -> list[str]:
        """Active instances this value replaces at the same scope."""
        if cardinality is not Cardinality.SINGLE:
            return []
        active = await self._store.active_contexts(
            subject_user_id=context.subject_user_id,
            scope_type=context.scope_type,
            scope_id=context.scope_id,
            concept_id=context.concept_id,
        )
        value_key = context.value_key
        return [
            instance.context_id
            for instance in active
            if instance.value_key != value_key and instance.fingerprint != context.fingerprint
        ]

    async def _revoke(
        self,
        signal: ExtractedSignal,
        resolution: ConceptResolution,
        decision: ScopeAssignment,
        episode: Episode,
    ) -> ContextWriteResult:
        """Withdraw matching context; withdrawal outranks recency and evidence."""
        active = await self._store.active_contexts(
            subject_user_id=decision.subject_user_id,
            scope_type=decision.scope_type,
            scope_id=decision.scope_id,
            concept_id=resolution.concept.concept_id,
        )
        value_key = canonical_value(signal.value)
        named = [instance for instance in active if instance.value_key == value_key]
        # A revocation may name the value it withdraws, or only the concept.
        targets = named or active
        revoked: list[str] = []
        for instance in targets:
            result = await self._store.revoke_context(
                context_id=instance.context_id,
                session_id=episode.session_id,
                episode_id=episode.episode_id,
                revoked_at=utcnow(),
            )
            if result is not None:
                revoked.append(result.context_id)

        if not revoked:
            log_event(
                _logger,
                "writer.revocation_had_no_target",
                concept_key=resolution.concept.concept_key,
                scope_type=decision.scope_type.value,
                scope_id=decision.scope_id,
            )
        return ContextWriteResult(
            context_id=revoked[0] if revoked else "",
            concept_key=resolution.concept.concept_key,
            status=ContextStatus.REVOKED,
            revoked_context_ids=tuple(revoked),
            evidence_episode_ids=(episode.episode_id,),
        )

    @staticmethod
    def _evidence(
        context_id: str,
        signal: ExtractedSignal,
        session: SessionInput,
        episode: Episode,
        recorded_at: datetime,
    ) -> EvidenceLink:
        return EvidenceLink(
            context_id=context_id,
            episode_id=episode.episode_id,
            session_id=session.session_id,
            actor_id=episode.sender_id,
            source_type=episode.source_type,
            extraction_confidence=signal.confidence,
            observed_at=episode.timestamp,
            recorded_at=recorded_at,
        )

    @staticmethod
    def _supporting_episodes(
        signal: ExtractedSignal, episodes: Mapping[str, Episode]
    ) -> Sequence[Episode]:
        """Episodes backing a signal, ordered; the earliest is the primary source."""
        supporting = [
            episodes[episode_id]
            for episode_id in signal.source_episode_ids
            if episode_id in episodes
        ]
        if not supporting:
            raise UnresolvedSignalError(
                f"signal '{signal.concept}' cites no episode from its own batch"
            )
        return sorted(supporting, key=lambda episode: episode.sequence)
