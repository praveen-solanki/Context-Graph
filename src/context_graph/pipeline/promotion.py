"""Promotion of shared-scope candidates (asynchronous maintenance).

Promotion never happens on the write path: candidates accumulate independent
evidence and a separate pass decides, per type and scope, whether they become
shared context (spec section 4, A.13).
"""

from __future__ import annotations

from context_graph.domain.enums import ContextStatus
from context_graph.graph.neo4j_store import Neo4jGraphStore
from context_graph.ledger.models import utcnow
from context_graph.observability import get_logger, log_event
from context_graph.pipeline.policy import WriterPolicy

_logger = get_logger("pipeline.promotion")


class PromotionService:
    """Evaluate shared-scope candidates against the promotion policy."""

    def __init__(self, *, store: Neo4jGraphStore, policy: WriterPolicy, batch_size: int) -> None:
        self._store = store
        self._policy = policy
        self._batch_size = batch_size

    async def run(self) -> list[str]:
        """Promote every candidate whose evidence satisfies its scope's rule."""
        candidates = await self._store.candidates(self._batch_size)
        if not candidates:
            return []

        support = await self._store.support_for([c.context_id for c in candidates])
        promoted: list[str] = []
        for candidate in candidates:
            evidence = support.get(candidate.context_id)
            if evidence is None or not self._policy.may_promote(candidate, evidence):
                continue
            await self._store.set_status(candidate.context_id, ContextStatus.ACTIVE, utcnow())
            promoted.append(candidate.context_id)
            log_event(
                _logger,
                "promotion.candidate_promoted",
                context_id=candidate.context_id,
                scope_type=candidate.scope_type.value,
                scope_id=candidate.scope_id,
                evidence_count=evidence.evidence_count,
                distinct_users=evidence.distinct_users,
            )
        log_event(
            _logger,
            "promotion.pass_completed",
            examined=len(candidates),
            promoted=len(promoted),
        )
        return promoted
