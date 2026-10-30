"""Concept resolution order and deterministic writer policy."""

from __future__ import annotations

from context_graph.config import PromotionSettings, ResolutionSettings, RetrySettings
from context_graph.domain.context import (
    SHARED_SUBJECT,
    ContextSupport,
    Lineage,
    MemoryConcept,
    Provenance,
)
from context_graph.domain.conversation import Episode, SessionInput
from context_graph.domain.enums import (
    Authority,
    ContextStatus,
    ContextType,
    Role,
    ScopeType,
    SourceType,
    Visibility,
)
from context_graph.domain.extraction import ConceptAdjudication, ExtractedSignal, ScopeCue
from context_graph.domain.ids import new_id
from context_graph.pipeline.policy import ScopeAssignment, UnresolvedSignal, WriterPolicy
from context_graph.pipeline.resolution import ConceptResolver, embedding_text
from tests.factories import make_message, single_span_episode
from tests.fakes import FakeEmbeddingProvider, FakeLLMProvider, InMemoryGraphStore


def signal(concept: str, **overrides: object) -> ExtractedSignal:
    payload: dict[str, object] = {
        "concept": concept,
        "value": "concise",
        "context_type": ContextType.PREFERENCE,
        "source_episode_ids": ("e1",),
    }
    payload.update(overrides)
    return ExtractedSignal.model_validate(payload)


def resolver(store: InMemoryGraphStore, llm: FakeLLMProvider) -> ConceptResolver:
    return ConceptResolver(
        store=store,
        llm=llm,
        settings=ResolutionSettings(),
        retry_settings=RetrySettings(max_attempts=1, base_backoff_seconds=0.0),
    )


async def embed(text: str) -> list[float]:
    return (await FakeEmbeddingProvider().embed([text]))[0]


async def test_exact_concept_key_resolves_without_a_vector_search() -> None:
    store = InMemoryGraphStore()
    known = MemoryConcept(
        concept_id="concept_1",
        concept="response_detail",
        concept_key=MemoryConcept.key_for("response_detail"),
    )
    await store.merge_concept(known, await embed("response detail"))

    resolution = await resolver(store, FakeLLMProvider()).resolve(
        signal("response_detail"), await embed("response detail")
    )

    assert resolution.method == "deterministic"
    assert resolution.concept.concept_id == "concept_1"


async def test_semantically_identical_text_reuses_the_existing_concept() -> None:
    store = InMemoryGraphStore()
    known = MemoryConcept(
        concept_id="concept_1",
        concept="response_detail",
        concept_key=MemoryConcept.key_for("response_detail"),
    )
    stored_text = embedding_text(signal("response_detail"))
    await store.merge_concept(known, await embed(stored_text))

    # A differently worded concept whose embedding matches the stored concept.
    resolution = await resolver(store, FakeLLMProvider()).resolve(
        signal("answer_length"), await embed(stored_text)
    )

    assert resolution.method == "embedding"
    assert resolution.concept.concept_id == "concept_1"


async def test_unknown_concept_is_created_rather_than_guessed() -> None:
    store = InMemoryGraphStore()
    llm = FakeLLMProvider()

    resolution = await resolver(store, llm).resolve(
        signal("deployment_strategy"), await embed("deployment strategy")
    )

    assert resolution.method == "created"
    assert resolution.concept.concept_key == MemoryConcept.key_for("deployment_strategy")
    assert llm.calls == []


async def test_ambiguous_candidates_are_adjudicated_by_the_llm() -> None:
    store = InMemoryGraphStore()
    known = MemoryConcept(
        concept_id="concept_1",
        concept="response_detail",
        concept_key=MemoryConcept.key_for("response_detail"),
    )
    await store.merge_concept(known, await embed("response detail"))
    llm = FakeLLMProvider(
        [ConceptAdjudication(is_match=True, concept_key=known.concept_key, reason="same meaning")]
    )

    # Deliberately below the accept threshold but inside the ambiguous band.
    candidate_embedding = await embed("answer length")
    resolution = await ConceptResolver(
        store=store,
        llm=llm,
        settings=ResolutionSettings(accept_similarity=0.99, ambiguous_similarity=0.0),
        retry_settings=RetrySettings(max_attempts=1, base_backoff_seconds=0.0),
    ).resolve(signal("answer_length"), candidate_embedding)

    assert resolution.method == "adjudicated"
    assert resolution.concept.concept_id == "concept_1"
    assert resolution.adjudicator_model == llm.model
    assert len(llm.calls) == 1


async def test_rejected_adjudication_creates_a_distinct_concept() -> None:
    store = InMemoryGraphStore()
    await store.merge_concept(
        MemoryConcept(
            concept_id="concept_1",
            concept="response_detail",
            concept_key=MemoryConcept.key_for("response_detail"),
        ),
        await embed("response detail"),
    )
    llm = FakeLLMProvider([ConceptAdjudication(is_match=False)])

    resolution = await ConceptResolver(
        store=store,
        llm=llm,
        settings=ResolutionSettings(accept_similarity=0.99, ambiguous_similarity=0.0),
        retry_settings=RetrySettings(max_attempts=1, base_backoff_seconds=0.0),
    ).resolve(signal("meeting_schedule"), await embed("meeting schedule"))

    assert resolution.method == "created"
    assert resolution.concept.concept_key == MemoryConcept.key_for("meeting_schedule")


def session() -> SessionInput:
    return SessionInput(
        session_id="S1", user_id="U123", messages=(make_message(1, "Keep answers concise"),)
    )


def episode(role: Role = Role.USER) -> Episode:
    return single_span_episode("S1", make_message(1, "Keep answers concise", role), "U123")


def policy() -> WriterPolicy:
    return WriterPolicy(PromotionSettings())


def test_durability_language_lifts_an_uncued_signal_to_user_scope() -> None:
    decision = policy().decide(signal("response_detail", durability_cue=True), session(), episode())
    assert isinstance(decision, ScopeAssignment)
    assert decision.scope_type is ScopeType.USER
    assert decision.scope_id == "U123"
    assert decision.visibility is Visibility.PRIVATE
    assert decision.status is ContextStatus.ACTIVE


def test_an_uncued_one_time_instruction_stays_with_its_session() -> None:
    decision = policy().decide(signal("response_detail"), session(), episode())
    assert isinstance(decision, ScopeAssignment)
    assert decision.scope_type is ScopeType.SESSION
    assert decision.scope_id == "S1"


def test_shared_scope_cue_becomes_a_candidate_owned_by_the_scope() -> None:
    decision = policy().decide(
        signal("deploy_day", scope_cue=ScopeCue(scope_type=ScopeType.TEAM, scope_ref="Search")),
        session(),
        episode(),
    )
    assert isinstance(decision, ScopeAssignment)
    assert decision.status is ContextStatus.CANDIDATE
    assert decision.visibility is Visibility.SHARED
    assert decision.subject_user_id == SHARED_SUBJECT
    assert decision.scope_id == "search"


def test_scope_cue_without_an_identifier_stays_unresolved() -> None:
    decision = policy().decide(
        signal("deploy_day", scope_cue=ScopeCue(scope_type=ScopeType.PROJECT)), session(), episode()
    )
    assert isinstance(decision, UnresolvedSignal)
    assert decision.reason == "unresolved_scope"


def test_a_signal_about_another_person_stays_unresolved() -> None:
    decision = policy().decide(signal("response_detail", subject="Rahul"), session(), episode())
    assert isinstance(decision, UnresolvedSignal)
    assert decision.reason == "unknown_subject"


def test_agent_output_is_never_user_asserted() -> None:
    decision = policy().decide(signal("response_detail"), session(), episode(Role.AGENT))
    assert isinstance(decision, ScopeAssignment)
    assert decision.authority is Authority.AGENT_DERIVED


def context_for(scope_type: ScopeType, authority: Authority = Authority.USER_ASSERTED):
    from context_graph.domain.context import ContextInstance
    from context_graph.ledger.models import utcnow

    return ContextInstance(
        context_id=new_id("ctx"),
        fingerprint="fp_1",
        context_type=ContextType.PREFERENCE,
        concept_id="concept_1",
        concept_key="personalization:x",
        subject_user_id=SHARED_SUBJECT,
        scope_type=scope_type,
        scope_id="search",
        value="thursday",
        status=ContextStatus.CANDIDATE,
        authority=authority,
        observed_at=utcnow(),
        recorded_at=utcnow(),
        provenance=Provenance(
            source_episode_id="e1",
            source_session_id="S1",
            source_actor="U1",
            source_type=SourceType.USER_MESSAGE,
            extraction_confidence=0.9,
        ),
        lineage=Lineage(
            run_id="run-1",
            extractor_model="m",
            prompt_version="p",
            policy_version="v",
            embedding_model="e",
            embedding_dimension=16,
        ),
    )


def support(**overrides: object) -> ContextSupport:
    payload: dict[str, object] = {
        "context_id": "c1",
        "evidence_count": 3,
        "distinct_sessions": 3,
        "distinct_users": 3,
    }
    payload.update(overrides)
    return ContextSupport.model_validate(payload)


def test_promotion_thresholds_are_scope_dependent() -> None:
    rules = policy()
    assert rules.may_promote(context_for(ScopeType.TEAM), support())
    assert not rules.may_promote(context_for(ScopeType.TEAM), support(distinct_users=2))
    assert not rules.may_promote(context_for(ScopeType.DEPARTMENT), support())


def test_organization_scope_requires_an_authoritative_source() -> None:
    rules = policy()
    evidence = support(evidence_count=9, distinct_sessions=9, distinct_users=9)
    assert not rules.may_promote(context_for(ScopeType.ORGANIZATION), evidence)
    assert rules.may_promote(
        context_for(ScopeType.ORGANIZATION, Authority.AUTHORITATIVE),
        evidence.model_copy(update={"has_authoritative_source": True}),
    )
