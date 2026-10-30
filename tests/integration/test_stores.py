"""Integration tests for the real Neo4j and MongoDB stores.

Opt in with ``CG_INTEGRATION=1`` and point ``CG_NEO4J__*`` / ``CG_MONGO__*`` at a
disposable database: these tests write real nodes and documents.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from context_graph.config import get_settings
from context_graph.domain.context import (
    ContextInstance,
    EvidenceLink,
    Lineage,
    MemoryConcept,
    Provenance,
)
from context_graph.domain.conversation import SessionInput
from context_graph.domain.enums import (
    Cardinality,
    ContextStatus,
    ContextType,
    ScopeType,
    SourceType,
)
from context_graph.domain.ids import new_id
from context_graph.graph.neo4j_store import Neo4jGraphStore
from context_graph.ledger.models import RunRecord, utcnow
from context_graph.ledger.mongo import MongoLedger
from tests.factories import make_message, single_span_episodes

pytestmark = pytest.mark.integration

DIMENSION = 8


def lineage(run_id: str) -> Lineage:
    return Lineage(
        run_id=run_id,
        extractor_model="integration",
        prompt_version="p",
        policy_version="v",
        embedding_model="integration",
        embedding_dimension=DIMENSION,
    )


@pytest.fixture
async def store() -> AsyncIterator[Neo4jGraphStore]:
    graph = Neo4jGraphStore(get_settings().neo4j)
    await graph.ensure_schema(DIMENSION)
    yield graph
    await graph.close()


@pytest.fixture
async def ledger() -> AsyncIterator[MongoLedger]:
    processing = MongoLedger(get_settings().mongo)
    await processing.ensure_indexes()
    yield processing
    await processing.close()


def session(session_id: str) -> SessionInput:
    return SessionInput(
        session_id=session_id,
        user_id="U-int",
        agent_ids=("agent-int",),
        messages=(
            make_message(1, "Always keep my answers concise."),
            make_message(2, "Understood."),
        ),
    )


async def test_episodes_are_idempotent_and_readable_by_sequence(store: Neo4jGraphStore) -> None:
    conversation = session(new_id("S"))
    await store.persist_session(conversation)
    await store.persist_episodes(single_span_episodes(conversation))
    await store.persist_episodes(single_span_episodes(conversation))

    assert await store.count_episodes(conversation.session_id) == 2
    after_first = await store.episodes_after(conversation.session_id, 1, 10)
    assert [episode.sequence for episode in after_first] == [2]


async def test_context_write_is_idempotent_and_supersedes_the_previous_value(
    store: Neo4jGraphStore,
) -> None:
    conversation = session(new_id("S"))
    await store.persist_session(conversation)
    episodes = single_span_episodes(conversation)
    await store.persist_episodes(episodes)

    concept = await store.merge_concept(
        MemoryConcept(
            concept_id=new_id("concept"),
            concept="response_detail",
            concept_key=MemoryConcept.key_for(f"response_detail_{conversation.session_id}"),
            cardinality=Cardinality.SINGLE,
        ),
        [0.1] * DIMENSION,
    )

    def instance(value: str) -> ContextInstance:
        now = utcnow()
        return ContextInstance(
            context_id=new_id("ctx"),
            fingerprint=ContextInstance.fingerprint_for(
                subject_user_id=conversation.user_id,
                scope_type=ScopeType.USER,
                scope_id=conversation.user_id,
                concept_id=concept.concept_id,
                value=value,
            ),
            context_type=ContextType.PREFERENCE,
            concept_id=concept.concept_id,
            concept_key=concept.concept_key,
            subject_user_id=conversation.user_id,
            scope_type=ScopeType.USER,
            scope_id=conversation.user_id,
            value=value,
            observed_at=episodes[0].timestamp,
            recorded_at=now,
            provenance=Provenance(
                source_episode_id=episodes[0].episode_id,
                source_session_id=conversation.session_id,
                source_actor=conversation.user_id,
                source_type=SourceType.USER_MESSAGE,
                extraction_confidence=0.9,
            ),
            lineage=lineage("run-int"),
        )

    def evidence(context_id: str) -> EvidenceLink:
        return EvidenceLink(
            context_id=context_id,
            episode_id=episodes[0].episode_id,
            session_id=conversation.session_id,
            actor_id=conversation.user_id,
            source_type=SourceType.USER_MESSAGE,
            extraction_confidence=0.9,
            observed_at=episodes[0].timestamp,
            recorded_at=utcnow(),
        )

    concise = instance("concise")
    first = await store.write_context(
        context=concise, embedding=[0.1] * DIMENSION, evidence=evidence(concise.context_id)
    )
    repeat = await store.write_context(
        context=instance("concise"),
        embedding=[0.1] * DIMENSION,
        evidence=evidence(first.context_id),
    )
    assert repeat.context_id == first.context_id

    support = await store.support_for([first.context_id])
    assert support[first.context_id].evidence_count == 1  # retries never inflate support

    detailed = instance("detailed")
    second = await store.write_context(
        context=detailed,
        embedding=[0.2] * DIMENSION,
        evidence=evidence(detailed.context_id),
        supersedes_context_ids=[first.context_id],
    )
    active = await store.active_contexts(
        subject_user_id=conversation.user_id,
        scope_type=ScopeType.USER,
        scope_id=conversation.user_id,
        concept_id=concept.concept_id,
    )
    assert [context.context_id for context in active] == [second.context_id]

    revoked = await store.revoke_context(
        context_id=second.context_id,
        session_id=conversation.session_id,
        episode_id=episodes[0].episode_id,
        revoked_at=utcnow(),
    )
    assert revoked is not None and revoked.status is ContextStatus.REVOKED


async def test_concept_vector_search_returns_the_nearest_concept(store: Neo4jGraphStore) -> None:
    marker = new_id("c")
    target = await store.merge_concept(
        MemoryConcept(
            concept_id=new_id("concept"),
            concept=f"meeting_schedule_{marker}",
            concept_key=MemoryConcept.key_for(f"meeting_schedule_{marker}"),
        ),
        [1.0] + [0.0] * (DIMENSION - 1),
    )

    matches = await store.search_concepts([1.0] + [0.0] * (DIMENSION - 1), top_k=1)

    assert matches and matches[0][0].concept_id == target.concept_id
    assert matches[0][1] > 0.99


async def test_concept_vector_search_scores_are_raw_cosine(store: Neo4jGraphStore) -> None:
    marker = new_id("c")
    stored = [0.0] * DIMENSION
    stored[-1] = 1.0
    query = [0.0] * DIMENSION
    query[0] = query[-1] = 1.0
    target = await store.merge_concept(
        MemoryConcept(
            concept_id=new_id("concept"),
            concept=f"half_aligned_{marker}",
            concept_key=MemoryConcept.key_for(f"half_aligned_{marker}"),
        ),
        stored,
    )

    matches = await store.search_concepts(query, top_k=1)

    assert matches and matches[0][0].concept_id == target.concept_id
    # cos 45 degrees; the index's own (1 + cos) / 2 would be 0.854.
    assert abs(matches[0][1] - 0.7071) < 0.01


async def test_ledger_round_trip_persists_checkpoints_and_results(ledger: MongoLedger) -> None:
    run_id = new_id("run")
    await ledger.upsert_run(
        RunRecord(run_id=run_id, graph_database="neo4j", lineage=lineage(run_id))
    )

    await ledger.record_completed(run_id, "S-int", [1, 2])
    advanced = await ledger.advance_checkpoint(run_id, "S-int", [1, 2, 3])
    assert advanced.checkpoint_sequence == 2

    await ledger.record_failed(run_id, "S-int", [3])
    stalled = await ledger.advance_checkpoint(run_id, "S-int", [1, 2, 3, 4])
    assert stalled.checkpoint_sequence == 2
