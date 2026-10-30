"""Relevance gate, token batching and rolling session state."""

from __future__ import annotations

from context_graph.config import BatchingSettings, GateSettings, LLMSettings
from context_graph.domain.conversation import Episode
from context_graph.domain.enums import ContextType, GateDecision, Role, ScopeType
from context_graph.domain.extraction import ExtractedSignal, ScopeCue
from context_graph.pipeline.batching import TokenBatcher
from context_graph.pipeline.gate import RelevanceGate
from context_graph.pipeline.rolling_state import RollingSessionState
from tests.factories import make_message, single_span_episode
from tests.fakes import FakeLLMProvider


def episode(text: str, sequence: int = 1, role: Role = Role.USER) -> Episode:
    return single_span_episode("S1", make_message(sequence, text, role))


def test_gate_skips_acknowledgements_but_keeps_real_content() -> None:
    gate = RelevanceGate(GateSettings())
    assert gate.decide(episode("thanks, got it!")) is GateDecision.SKIP
    assert gate.decide(episode("ok")) is GateDecision.SKIP
    assert gate.decide(episode("")) is GateDecision.SKIP
    assert gate.decide(episode("Keep my replies concise")) is GateDecision.EXTRACT


def test_gate_sends_uncertain_short_text_to_the_llm() -> None:
    gate = RelevanceGate(GateSettings())
    # Short, but not a pure acknowledgement: extract rather than risk losing it.
    assert gate.decide(episode("use dark mode")) is GateDecision.EXTRACT


def batcher(budget_tokens: int, **batching: int) -> TokenBatcher:
    llm_settings = LLMSettings(
        context_limit_tokens=budget_tokens + 1_000,
        max_output_tokens=500,
        prompt_overhead_tokens=200,
        safety_margin_tokens=200,
        rolling_state_token_budget=100,
    )
    return TokenBatcher(FakeLLMProvider(), llm_settings, BatchingSettings(**batching))


def test_batches_are_sized_by_tokens_not_message_count() -> None:
    # Budget is 25 tokens = 100 characters with the fake tokenizer.
    episodes = [episode("x" * 60, sequence) for sequence in range(1, 5)]
    batches = batcher(25).batches("r1", "S1", episodes)

    assert [len(batch.chunks) for batch in batches] == [1, 1, 1, 1]
    assert all(batch.token_estimate <= 25 for batch in batches)


def test_small_episodes_share_one_batch_with_overlap_for_continuity() -> None:
    episodes = [episode("short message", sequence) for sequence in range(1, 7)]
    batches = batcher(25, max_episodes_per_batch=3, overlap_episodes=1).batches(
        "r1", "S1", episodes
    )

    assert [len(batch.chunks) for batch in batches] == [3, 3]
    assert batches[0].overlap_chunks == ()
    assert batches[1].overlap_chunks[0].episode.sequence == 3


def test_oversized_episode_is_chunked_but_keeps_its_episode_id() -> None:
    long_episode = episode("y" * 500)
    chunks = batcher(25).chunk(long_episode)

    assert len(chunks) > 1
    assert {chunk.episode_id for chunk in chunks} == {long_episode.episode_id}
    assert "".join(chunk.text for chunk in chunks) == long_episode.text
    assert all(chunk.is_partial for chunk in chunks)


def signal(concept: str, value: str) -> ExtractedSignal:
    return ExtractedSignal(
        concept=concept,
        value=value,
        context_type=ContextType.PREFERENCE,
        scope_cue=ScopeCue(scope_type=ScopeType.USER),
        source_episode_ids=("e1",),
    )


def test_rolling_state_replaces_restated_concepts_and_stays_bounded() -> None:
    state = RollingSessionState(
        max_items=2, token_budget=1_000, estimate_tokens=FakeLLMProvider().estimate_tokens
    )
    state.update([signal("response_detail", "verbose"), signal("tone", "formal")])
    state.update([signal("response_detail", "concise"), signal("editor", "vim")])

    rendered = state.render()
    assert "response_detail=concise" in rendered
    assert "verbose" not in rendered
    assert "tone" not in rendered  # evicted by the item bound
    assert len(rendered.splitlines()) == 2


def test_rolling_state_trims_oldest_items_to_fit_the_token_budget() -> None:
    state = RollingSessionState(
        max_items=10, token_budget=5, estimate_tokens=FakeLLMProvider().estimate_tokens
    )
    state.update([signal("a", "1"), signal("b", "2"), signal("c", "3")])
    assert len(state.render().splitlines()) <= 1
