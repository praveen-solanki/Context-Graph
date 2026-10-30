"""End-to-end ingestion behaviour against in-memory persistence."""

from __future__ import annotations

from collections.abc import Sequence

import pytest
from pydantic import BaseModel

from context_graph.config import BatchingSettings, Settings
from context_graph.domain.conversation import MessageInput, SessionInput
from context_graph.domain.enums import (
    Authority,
    ContextStatus,
    ContextType,
    Role,
    ScopeType,
    SignalOperation,
    Visibility,
)
from context_graph.domain.extraction import ExtractedSignal, ExtractionEnvelope, ScopeCue
from context_graph.pipeline.orchestrator import IngestionPipeline, IngestionReport
from context_graph.tool import ContextGraphIngestion, IngestionRequest
from tests.factories import make_message, single_span_episodes
from tests.fakes import FakeEmbeddingProvider, FakeLLMProvider, InMemoryGraphStore, InMemoryLedger


def envelope(*signals: ExtractedSignal) -> ExtractionEnvelope:
    return ExtractionEnvelope(signals=signals)


def signal(
    concept: str,
    value: str,
    *,
    episode_ids: tuple[str, ...],
    scope: ScopeCue | None = None,
    subject: str = "user",
    durability_cue: bool = False,
    operation: SignalOperation = SignalOperation.ASSERT,
    context_type: ContextType = ContextType.PREFERENCE,
) -> ExtractedSignal:
    return ExtractedSignal(
        concept=concept,
        value=value,
        context_type=context_type,
        scope_cue=scope,
        source_episode_ids=episode_ids,
        subject=subject,
        durability_cue=durability_cue,
        operation=operation,
    )


def session_with(*texts: str, session_id: str = "S1", user_id: str = "U123") -> SessionInput:
    messages: list[MessageInput] = [
        make_message(index, text) for index, text in enumerate(texts, start=1)
    ]
    return SessionInput(session_id=session_id, user_id=user_id, messages=tuple(messages))


def episode_ids(session: SessionInput) -> list[str]:
    return [episode.episode_id for episode in single_span_episodes(session)]


class Harness:
    """Pipeline wired to in-memory persistence and scripted providers."""

    def __init__(
        self,
        responses: Sequence[BaseModel],
        settings: Settings | None = None,
        embeddings: FakeEmbeddingProvider | None = None,
    ) -> None:
        self.store = InMemoryGraphStore()
        self.ledger = InMemoryLedger()
        self.llm = FakeLLMProvider(list(responses))
        self.embeddings = embeddings or FakeEmbeddingProvider()
        self.pipeline = IngestionPipeline(
            settings=settings or Settings(),
            store=self.store,
            ledger=self.ledger,
            llm=self.llm,
            embeddings=self.embeddings,
            run_id="run-1",
        )

    async def ingest(self, session: SessionInput) -> IngestionReport:
        await self.pipeline.prepare()
        return await self.pipeline.ingest(session)


async def test_session_instruction_and_durable_preference_coexist() -> None:
    # A.9: a session instruction is not replaced by a later user-level preference.
    first = session_with("For this chat, keep answers short.")
    second = session_with("I always prefer concise answers.", session_id="S2")
    harness = Harness(
        [
            envelope(
                signal(
                    "response_detail",
                    "concise",
                    episode_ids=tuple(episode_ids(first)),
                    scope=ScopeCue(scope_type=ScopeType.SESSION),
                )
            ),
            envelope(
                signal(
                    "response_detail",
                    "concise",
                    episode_ids=tuple(episode_ids(second)),
                    scope=ScopeCue(scope_type=ScopeType.USER),
                    durability_cue=True,
                )
            ),
        ]
    )

    await harness.ingest(first)
    await harness.ingest(second)

    scopes = {
        (context.scope_type, context.scope_id): context
        for context in harness.store.contexts.values()
    }
    assert set(scopes) == {(ScopeType.SESSION, "S1"), (ScopeType.USER, "U123")}
    assert all(context.status is ContextStatus.ACTIVE for context in scopes.values())
    # One concept, two scoped instances.
    assert len({context.concept_id for context in scopes.values()}) == 1


async def test_same_value_reuses_the_instance_and_does_not_double_count_evidence() -> None:
    session = session_with("Keep my replies concise.", "Keep my replies concise please.")
    ids = episode_ids(session)
    harness = Harness(
        [
            envelope(
                signal("response_detail", "concise", episode_ids=(ids[0],)),
                signal("response_detail", "Concise", episode_ids=(ids[1],)),
            )
        ]
    )

    report = await harness.ingest(session)

    assert len(harness.store.contexts) == 1
    context_id = next(iter(harness.store.contexts))
    assert set(harness.store.evidence[context_id]) == set(ids)
    assert report.contexts_superseded == 0


async def test_changed_value_creates_a_new_instance_that_supersedes_the_old_one() -> None:
    first = session_with("Keep answers concise, always.")
    second = session_with("Actually, always give me detailed answers.", session_id="S2")
    harness = Harness(
        [
            envelope(
                signal(
                    "response_detail",
                    "concise",
                    episode_ids=tuple(episode_ids(first)),
                    durability_cue=True,
                )
            ),
            envelope(
                signal(
                    "response_detail",
                    "detailed",
                    episode_ids=tuple(episode_ids(second)),
                    durability_cue=True,
                )
            ),
        ]
    )

    await harness.ingest(first)
    report = await harness.ingest(second)

    statuses = {context.value: context.status for context in harness.store.contexts.values()}
    assert statuses == {"concise": ContextStatus.SUPERSEDED, "detailed": ContextStatus.ACTIVE}
    assert report.contexts_superseded == 1


async def test_shared_scope_is_stored_as_a_candidate_not_published() -> None:
    session = session_with("Our Search team always deploys on Thursdays.")
    harness = Harness(
        [
            envelope(
                signal(
                    "deployment_day",
                    "thursday",
                    episode_ids=tuple(episode_ids(session)),
                    scope=ScopeCue(scope_type=ScopeType.TEAM, scope_ref="Search"),
                )
            )
        ]
    )

    await harness.ingest(session)

    context = next(iter(harness.store.contexts.values()))
    assert context.status is ContextStatus.CANDIDATE
    assert context.visibility is Visibility.SHARED
    assert context.scope_type is ScopeType.TEAM and context.scope_id == "search"
    assert context.authority is Authority.USER_ASSERTED


async def test_team_candidate_is_promoted_once_enough_users_support_it() -> None:
    sessions = [
        session_with(
            "The Search team runs retros on Friday.", session_id=f"S{index}", user_id=f"U{index}"
        )
        for index in range(1, 4)
    ]
    harness = Harness(
        [
            envelope(
                signal(
                    "retro_day",
                    "friday",
                    episode_ids=tuple(episode_ids(session)),
                    scope=ScopeCue(scope_type=ScopeType.TEAM, scope_ref="Search"),
                )
            )
            for session in sessions
        ]
    )

    reports = [await harness.ingest(session) for session in sessions]

    context = next(iter(harness.store.contexts.values()))
    assert context.status is ContextStatus.ACTIVE  # promoted by the maintenance pass
    assert reports[-1].promoted_context_ids == (context.context_id,)
    assert reports[0].promoted_context_ids == ()


async def test_explicit_revocation_withdraws_the_context() -> None:
    first = session_with("Always keep my answers concise.")
    second = session_with("Forget my preference about answer length.", session_id="S2")
    harness = Harness(
        [
            envelope(
                signal(
                    "response_detail",
                    "concise",
                    episode_ids=tuple(episode_ids(first)),
                    durability_cue=True,
                )
            ),
            envelope(
                signal(
                    "response_detail",
                    "concise",
                    episode_ids=tuple(episode_ids(second)),
                    durability_cue=True,
                    operation=SignalOperation.REVOKE,
                )
            ),
        ]
    )

    await harness.ingest(first)
    report = await harness.ingest(second)

    context = next(iter(harness.store.contexts.values()))
    assert context.status is ContextStatus.REVOKED
    assert report.contexts_revoked == 1


async def test_signals_about_other_people_stay_unresolved() -> None:
    session = session_with("Rahul prefers long design documents.")
    harness = Harness(
        [
            envelope(
                signal(
                    "document_detail",
                    "long",
                    episode_ids=tuple(episode_ids(session)),
                    subject="Rahul",
                )
            )
        ]
    )

    report = await harness.ingest(session)

    assert report.signals_unresolved == 1
    assert harness.store.contexts == {}


async def test_gate_skips_acknowledgements_without_blocking_the_checkpoint() -> None:
    session = session_with("thanks!", "ok", "Always use metric units.")
    ids = episode_ids(session)
    harness = Harness(
        [envelope(signal("units", "metric", episode_ids=(ids[2],), durability_cue=True))]
    )

    report = await harness.ingest(session)

    assert report.episodes_gated_out == 2
    # Checkpoints track episode ordinals, which start at 0.
    assert report.checkpoint_sequence == 2
    assert len(harness.llm.calls) == 1


async def test_reingesting_a_session_is_idempotent() -> None:
    session = session_with("Always keep my answers concise.")
    signals = envelope(
        signal(
            "response_detail",
            "concise",
            episode_ids=tuple(episode_ids(session)),
            durability_cue=True,
        )
    )
    harness = Harness([signals, signals])

    first = await harness.ingest(session)
    second = await harness.ingest(session)

    assert len(harness.store.episodes) == 1
    assert len(harness.store.contexts) == 1
    context_id = next(iter(harness.store.contexts))
    assert len(harness.store.evidence[context_id]) == 1
    assert first.checkpoint_sequence == second.checkpoint_sequence == 0
    # The second run had no new episodes: the model was not called again.
    assert len(harness.llm.calls) == 1


async def test_self_stated_membership_is_private_context_not_an_authorization_edge() -> None:
    # "I'm on the Search team" is a fact about the user. It must not become a
    # MEMBER_OF edge: a user cannot grant themselves access by saying so.
    session = session_with("I'm on the Search team.")
    harness = Harness(
        [
            envelope(
                signal(
                    "team_membership",
                    "search",
                    episode_ids=tuple(episode_ids(session)),
                    scope=ScopeCue(scope_type=ScopeType.USER),
                    context_type=ContextType.FACT,
                )
            )
        ]
    )
    ingestion = ContextGraphIngestion(
        settings=Settings(),
        store=harness.store,
        ledger=harness.ledger,
        llm=harness.llm,
        embeddings=harness.embeddings,
        run_id="run-1",
    )

    await ingestion.ingest_conversation(IngestionRequest(session=session))

    context = next(iter(harness.store.contexts.values()))
    assert context.context_type is ContextType.FACT
    assert context.scope_type is ScopeType.USER and context.scope_id == "U123"
    assert context.visibility is Visibility.PRIVATE
    assert not hasattr(harness.store, "memberships")
    assert harness.store.schema_ready and harness.ledger.indexes_ready


@pytest.mark.parametrize("role", [Role.AGENT])
async def test_agent_authored_context_is_agent_derived(role: Role) -> None:
    session = SessionInput(
        session_id="S1",
        user_id="U123",
        messages=(make_message(1, "I will always answer concisely for you.", role),),
    )
    harness = Harness(
        [
            envelope(
                signal(
                    "response_detail",
                    "concise",
                    episode_ids=tuple(episode_ids(session)),
                    durability_cue=True,
                )
            )
        ]
    )

    await harness.ingest(session)

    context = next(iter(harness.store.contexts.values()))
    assert context.authority is Authority.AGENT_DERIVED


async def test_stale_reference_is_resolved_by_a_targeted_lookup() -> None:
    first = session_with("Always deploy Apollo with blue-green releases.")
    second = session_with(
        "Use the same strategy we discussed.", "And document it please.", session_id="S2"
    )
    # One episode per batch, so the recalled reference reaches the next batch.
    settings = Settings(batching=BatchingSettings(max_episodes_per_batch=1))
    harness = Harness(
        [
            envelope(
                signal(
                    "deployment_strategy",
                    "blue-green",
                    episode_ids=tuple(episode_ids(first)),
                    durability_cue=True,
                )
            ),
            ExtractionEnvelope(unresolved_references=("the strategy we discussed",)),
            ExtractionEnvelope(),
        ],
        settings=settings,
        embeddings=_AliasEmbedder({"the strategy we discussed": "deployment_strategy: blue-green"}),
    )

    await harness.ingest(first)
    await harness.ingest(second)

    last_prompt = harness.llm.calls[-1][1]
    assert "the strategy we discussed refers to: personalization:deployment_strategy=blue-green" in last_prompt


class _AliasEmbedder(FakeEmbeddingProvider):
    """Embeds selected texts as their alias, simulating semantic similarity."""

    def __init__(self, aliases: dict[str, str]) -> None:
        super().__init__()
        self._aliases = aliases

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return await super().embed([self._aliases.get(text, text) for text in texts])


async def test_independent_sessions_are_ingested_concurrently() -> None:
    sessions = [
        session_with("Always keep answers concise.", session_id=f"S{index}", user_id=f"U{index}")
        for index in range(1, 4)
    ]
    harness = Harness(
        [
            envelope(
                signal(
                    "response_detail",
                    "concise",
                    episode_ids=tuple(episode_ids(session)),
                    durability_cue=True,
                )
            )
            for session in sessions
        ]
    )
    await harness.pipeline.prepare()

    reports = await harness.pipeline.ingest_many(sessions)

    assert [report.session_id for report in reports] == [s.session_id for s in sessions]
    assert all(report.errors == () for report in reports)
    assert len(harness.store.sessions) == 3
