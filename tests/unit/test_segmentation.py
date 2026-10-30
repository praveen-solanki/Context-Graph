"""Grouping messages into bounded episodes."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from context_graph.config import LLMSettings, RetrySettings, SegmentationSettings
from context_graph.domain.conversation import Episode, MessageInput, SessionInput
from context_graph.domain.enums import Role
from context_graph.domain.extraction import ProposedEpisode, SegmentationProposal
from context_graph.errors import TransientError
from context_graph.pipeline.segmentation import EpisodeBuilder
from tests.factories import make_message
from tests.fakes import FakeLLMProvider


def session_of(*messages: tuple[str, Role] | str, session_id: str = "S1") -> SessionInput:
    built: list[MessageInput] = []
    for index, item in enumerate(messages, start=1):
        text, role = item if isinstance(item, tuple) else (item, Role.USER)
        built.append(make_message(index, text, role))
    return SessionInput(session_id=session_id, user_id="U1", messages=tuple(built))


class ScriptedLLM(FakeLLMProvider):
    """Returns prepared groupings, one per window, recording each request."""

    def __init__(self, groupings: list[list[list[str]]] | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._groupings = groupings or []
        self.requests: list[str] = []
        self.failures = 0

    async def structured[M: BaseModel](
        self, *, system: str, user: str, schema: type[M], max_output_tokens: int
    ) -> M:
        if schema is not SegmentationProposal:
            return await super().structured(
                system=system, user=user, schema=schema, max_output_tokens=max_output_tokens
            )
        self.requests.append(user)
        if not self._groupings:
            return await super().structured(
                system=system, user=user, schema=schema, max_output_tokens=max_output_tokens
            )
        grouping = self._groupings.pop(0)
        if grouping == "fail":  # pyright: ignore[reportUnnecessaryComparison]
            self.failures += 1
            raise TransientError("model unavailable")
        proposal = SegmentationProposal(
            episodes=tuple(
                ProposedEpisode(
                    # A trailing "!" in the script marks an open-ended episode.
                    span_refs=tuple(ref.rstrip("!") for ref in refs),
                    open_ended=refs[-1].endswith("!"),
                )
                for refs in grouping
            )
        )
        return proposal  # pyright: ignore[reportReturnType]


def builder(llm: FakeLLMProvider, **overrides: Any) -> EpisodeBuilder:
    return EpisodeBuilder(
        llm=llm,
        llm_settings=LLMSettings(),
        settings=SegmentationSettings(**overrides),
        retry_settings=RetrySettings(max_attempts=1, base_backoff_seconds=0.0),
    )


def texts(episodes: list[Episode]) -> list[list[str]]:
    return [[span.text for span in episode.spans] for episode in episodes]


async def test_related_messages_are_grouped_into_one_episode() -> None:
    session = session_of(
        ("How should we roll out the migration?", Role.USER),
        ("Start with a shadow read, then backfill.", Role.AGENT),
        ("ok", Role.USER),
        ("What is for lunch?", Role.USER),
    )
    llm = ScriptedLLM([[["s1-0", "s2-0", "s3-0"], ["s4-0"]]])

    episodes = await builder(llm).build(session)

    assert len(episodes) == 2
    # The short acknowledgement stayed with the discussion it answers.
    assert texts(episodes)[0] == [
        "How should we roll out the migration?",
        "Start with a shadow read, then backfill.",
        "ok",
    ]
    assert texts(episodes)[1] == ["What is for lunch?"]
    assert [episode.sequence for episode in episodes] == [0, 1]


async def test_an_episode_keeps_both_speakers_with_their_own_identity() -> None:
    session = session_of(("question?", Role.USER), ("answer.", Role.AGENT))
    llm = ScriptedLLM([[["s1-0", "s2-0"]]])

    episode = (await builder(llm).build(session))[0]

    assert [span.role for span in episode.spans] == [Role.USER, Role.AGENT]
    assert episode.roles == ("user", "agent")
    assert episode.user_senders == ("U1",)
    assert episode.agent_senders == ("agent",)
    # Episode-level provenance comes from the first span.
    assert episode.role is Role.USER and episode.sender_id == "U1"


async def test_oversized_message_is_split_into_bounded_spans_with_exact_offsets() -> None:
    body = ". ".join(f"sentence number {index} about the incident" for index in range(200))
    session = session_of(body)
    llm = ScriptedLLM()  # default fake: one episode per span

    episodes = await builder(llm, max_span_tokens=60, max_episode_tokens=10_000).build(session)
    spans = [span for episode in episodes for span in episode.spans]

    assert len(spans) > 1
    # Offsets are exact and reconstruct the original message without loss.
    assert "".join(span.text for span in spans) == body
    assert spans[0].char_start == 0
    assert spans[-1].char_end == len(body)
    for earlier, later in zip(spans, spans[1:], strict=False):
        assert earlier.char_end == later.char_start
    assert all(span.message_id == "m1" for span in spans)
    assert all(span.is_partial for span in spans)
    assert [span.part for span in spans] == list(range(len(spans)))


async def test_text_without_sentence_or_paragraph_breaks_is_still_bounded() -> None:
    body = "x" * 6_000
    session = session_of(body)

    episodes = await builder(ScriptedLLM(), max_span_tokens=50).build(session)
    spans = [span for episode in episodes for span in episode.spans]

    assert len(spans) > 1
    assert "".join(span.text for span in spans) == body


async def test_a_boundary_may_fall_inside_one_message_with_two_topics() -> None:
    body = "First, the deploy window moved to Thursday.\n\nSeparately, I prefer concise answers."
    session = session_of(body)
    llm = ScriptedLLM([[["s1-0"], ["s1-1"]]])

    episodes = await builder(llm, max_span_tokens=12).build(session)

    assert len(episodes) == 2
    assert episodes[0].spans[0].message_id == episodes[1].spans[0].message_id
    assert episodes[0].spans[0].char_end == episodes[1].spans[0].char_start


async def test_metadata_speaker_and_timestamp_survive_grouping() -> None:
    message = MessageInput(
        message_id="m1",
        timestamp=make_message(1, "x").timestamp,
        role=Role.USER,
        sequence=1,
        text="Always keep answers concise.",
        sender_id="U-praveen",
        metadata={"channel": "slack", "thread": "abc"},
    )
    session = SessionInput(session_id="S1", user_id="U1", messages=(message,))

    episode = (await builder(ScriptedLLM()).build(session))[0]
    span = episode.spans[0]

    assert span.metadata == {"channel": "slack", "thread": "abc"}
    assert span.sender_id == "U-praveen"
    assert span.timestamp == message.timestamp
    assert span.text == message.text


async def test_unfinished_tail_carries_into_the_next_window() -> None:
    session = session_of(*[f"message {index}" for index in range(1, 5)])
    # Two spans per window. Window one leaves s2 open, so it must reappear in
    # window two and be decided there.
    llm = ScriptedLLM(
        [
            [["s1-0"], ["s2-0!"]],
            [["s2-0", "s3-0"]],
            [["s4-0"]],
        ]
    )

    episodes = await builder(llm, max_window_tokens=6, max_span_tokens=3).build(session)

    assert len(llm.requests) == 3
    assert "[s2-0 user]" in llm.requests[1]  # carried, and re-decided
    assert texts(episodes) == [["message 1"], ["message 2", "message 3"], ["message 4"]]
    # Every span belongs to exactly one finalized episode.
    refs = [span.ref for episode in episodes for span in episode.spans]
    assert len(refs) == len(set(refs)) == 4
    assert [episode.sequence for episode in episodes] == [0, 1, 2]


async def test_previous_content_is_replayed_as_context_only() -> None:
    session = session_of(*[f"message {index}" for index in range(1, 5)])
    llm = ScriptedLLM([[["s1-0", "s2-0"]], [["s3-0", "s4-0"]]])

    episodes = await builder(
        llm, max_window_tokens=6, max_span_tokens=3, context_spans=2
    ).build(session)

    assert "CONTEXT" in llm.requests[1]
    assert "[CONTEXT s2-0 user]" in llm.requests[1]
    # Context is shown for continuity only, never assigned again.
    refs = [span.ref for episode in episodes for span in episode.spans]
    assert refs == ["s1-0", "s2-0", "s3-0", "s4-0"]


async def test_a_long_discussion_becomes_several_bounded_episodes() -> None:
    long_text = " ".join(["detailed discussion of the migration plan"] * 200)
    session = session_of(long_text, "and one more thing")
    settings = {"max_span_tokens": 60, "max_episode_tokens": 60, "max_window_tokens": 100_000}

    # The model groups the entire discussion as one topic; the size limit then
    # continues it across several bounded episodes.
    refs = [span.ref for span in builder(ScriptedLLM(), **settings).spans_for(session)]
    llm = ScriptedLLM([[refs]])

    episodes = await builder(llm, **settings).build(session)

    assert len(episodes) > 2
    # Fake tokenizer: 4 characters per token, so the episode limit is 240 chars.
    assert all(len(episode.content) <= 60 * 4 + 40 for episode in episodes)
    # Continuations are linked to the episode they continue from.
    assert [episode.continues_episode_id is not None for episode in episodes][1:] == [True] * (
        len(episodes) - 1
    )
    # Nothing was lost while splitting.
    joined = "".join(span.text for episode in episodes for span in episode.spans)
    assert joined == long_text + "and one more thing"


async def test_session_ending_mid_exchange_is_still_finalized() -> None:
    session = session_of("so about the rollout", "we should probably")
    # The model insists the discussion is unfinished; the session has ended.
    llm = ScriptedLLM([[["s1-0", "s2-0!"]]])

    episodes = await builder(llm).build(session)

    assert len(episodes) == 1
    assert texts(episodes)[0] == ["so about the rollout", "we should probably"]


@pytest.mark.parametrize(
    "grouping",
    [
        [["s1-0", "s9-0"]],  # reference that does not exist
        [["s1-0"], ["s1-0", "s2-0"]],  # one span claimed twice
        [["s2-0", "s1-0"]],  # conversation reordered
        [["s1-0"]],  # a span left unassigned
    ],
)
async def test_invalid_proposals_fall_back_without_losing_content(
    grouping: list[list[str]],
) -> None:
    session = session_of("first message", "second message")
    llm = ScriptedLLM([grouping])

    episodes = await builder(llm).build(session)

    covered = [span.text for episode in episodes for span in episode.spans]
    assert covered == ["first message", "second message"]


async def test_a_rejected_proposal_is_retried_with_the_reason() -> None:
    session = session_of("first message", "second message")
    llm = ScriptedLLM([[["s1-0"]], [["s1-0", "s2-0"]]])

    episodes = await builder(llm).build(session)

    assert len(llm.requests) == 2
    assert "rejected" not in llm.requests[0]
    assert "unassigned" in llm.requests[1] and "s2-0" in llm.requests[1]
    assert texts(episodes) == [["first message", "second message"]]


async def test_model_failure_falls_back_to_size_based_grouping() -> None:
    session = session_of("first message", "second message")
    llm = ScriptedLLM(["fail"])  # pyright: ignore[reportArgumentType]

    episodes = await builder(llm).build(session)

    covered = [span.text for episode in episodes for span in episode.spans]
    assert covered == ["first message", "second message"]


async def test_grouping_is_reproducible_so_reruns_do_not_duplicate() -> None:
    session = session_of("first message", "second message", "third message")
    grouping = [["s1-0", "s2-0"], ["s3-0"]]

    first = await builder(ScriptedLLM([grouping])).build(session)
    second = await builder(ScriptedLLM([list(grouping)])).build(session)

    assert [episode.episode_id for episode in first] == [
        episode.episode_id for episode in second
    ]


async def test_every_message_is_covered_exactly_once() -> None:
    session = session_of(*[f"message {index}" for index in range(1, 10)])

    episodes = await builder(ScriptedLLM()).build(session)

    covered = [span.message_id for episode in episodes for span in episode.spans]
    assert covered == [f"m{index}" for index in range(1, 10)]


async def test_repeatedly_alternating_topics_stay_in_one_episode() -> None:
    session = session_of(
        "about the deploy window",
        "and also the reviewer count",
        "back to the deploy window",
        "and the reviewers again",
    )
    llm = ScriptedLLM([[["s1-0", "s2-0", "s3-0", "s4-0"]]])

    episodes = await builder(llm).build(session)

    assert len(episodes) == 1
    assert len(episodes[0].spans) == 4


async def test_episode_identity_is_stable_and_session_scoped() -> None:
    session = session_of("first message", "second message")
    other = session_of("first message", "second message", session_id="S2")
    grouping = [["s1-0", "s2-0"]]

    here = await builder(ScriptedLLM([list(grouping)])).build(session)
    there = await builder(ScriptedLLM([list(grouping)])).build(other)

    assert here[0].episode_id != there[0].episode_id


async def test_regrouping_changes_identity_which_is_why_runs_reuse_their_grouping() -> None:
    """Documents why segmentation is stored in the ledger and reused on resume."""
    session = session_of("first message", "second message")

    together = await builder(ScriptedLLM([[["s1-0", "s2-0"]]])).build(session)
    apart = await builder(ScriptedLLM([[["s1-0"], ["s2-0"]]])).build(session)

    assert len(together) == 1 and len(apart) == 2
    assert together[0].episode_id not in {episode.episode_id for episode in apart}


def test_span_and_episode_limits_must_be_consistent() -> None:
    from pydantic import ValidationError

    from context_graph.config import SegmentationSettings

    with pytest.raises(ValidationError, match="max_span_tokens must not exceed"):
        SegmentationSettings(max_span_tokens=500, max_episode_tokens=100)
