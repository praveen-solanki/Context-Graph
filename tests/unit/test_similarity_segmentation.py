"""Similarity-based segmentation: the alternative to the LLM-boundary strategy."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from context_graph.config import SimilaritySegmentationSettings
from context_graph.domain.conversation import Episode, SessionInput
from context_graph.domain.enums import Role
from context_graph.pipeline.similarity_segmentation import SimilarityEpisodeBuilder
from tests.factories import make_message
from tests.fakes import FakeEmbeddingProvider


def session_of(*messages: tuple[str, Role] | str, session_id: str = "S1") -> SessionInput:
    """A session from plain text, defaulting to alternating user/agent turns."""
    roles = [item[1] if isinstance(item, tuple) else (Role.USER if index % 2 == 0 else Role.AGENT)
             for index, item in enumerate(messages)]
    texts = [item[0] if isinstance(item, tuple) else item for item in messages]
    return SessionInput(
        session_id=session_id,
        user_id="U1",
        agent_ids=("agent-1",),
        messages=tuple(
            make_message(sequence, text, role)
            for sequence, (text, role) in enumerate(zip(texts, roles, strict=True), start=1)
        ),
    )


def builder(**settings: int | float) -> SimilarityEpisodeBuilder:
    return SimilarityEpisodeBuilder(
        embeddings=FakeEmbeddingProvider(),
        settings=SimilaritySegmentationSettings.model_validate(settings),
    )


def reconstructed(session: SessionInput, episodes: list[Episode]) -> str:
    """Every span's text, in (sequence, part) order - should equal the original."""
    spans = sorted(
        (span for episode in episodes for span in episode.spans),
        key=lambda span: (span.sequence, span.part),
    )
    by_message: dict[str, list[str]] = {}
    for span in spans:
        by_message.setdefault(span.message_id, []).append(span.text)
    return "".join("".join(parts) for parts in by_message.values())


class KeywordEmbeddingProvider:
    """Two orthogonal topics, so a real boundary is findable by cosine distance."""

    model = "keyword-fake"
    dimension = 2

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [[1.0, 0.0] if "alpha" in text else [0.0, 1.0] for text in texts]


async def test_an_empty_session_has_no_episodes() -> None:
    assert await builder().build(session_of()) == []


async def test_every_character_of_every_message_is_covered_exactly_once() -> None:
    session = session_of(*[f"message number {index} with some real content here" for index in range(30)])

    episodes = await builder(min_size=50, target_size=120, max_size=200).build(session)

    assert reconstructed(session, episodes) == "".join(message.text for message in session.messages)


async def test_no_episode_exceeds_the_configured_max_size() -> None:
    session = session_of(*[f"turn {index} says something of moderate length about the topic" for index in range(40)])

    episodes = await builder(min_size=80, target_size=150, max_size=250).build(session)

    assert all(sum(len(span.text) for span in episode.spans) <= 250 for episode in episodes)
    assert len(episodes) > 1


async def test_episodes_are_sequential_and_identity_is_deterministic() -> None:
    session = session_of(*[f"content for turn {index} that is reasonably sized" for index in range(12)])
    settings = {"min_size": 40, "target_size": 80, "max_size": 150}

    first = await builder(**settings).build(session)
    second = await builder(**settings).build(session)

    assert [episode.sequence for episode in first] == list(range(len(first)))
    assert [episode.episode_id for episode in first] == [episode.episode_id for episode in second]


async def test_a_message_split_across_episodes_gets_a_consistent_part_count() -> None:
    long_message = "x" * 900
    session = session_of(long_message, "ok", session_id="S-split")

    episodes = await builder(min_size=50, target_size=200, max_size=400).build(session)

    all_spans = [span for episode in episodes for span in episode.spans if span.message_id.endswith("m1")]
    assert len(all_spans) > 1
    assert all(span.part_count == len(all_spans) for span in all_spans)
    assert sorted(span.part for span in all_spans) == list(range(len(all_spans)))
    assert "".join(span.text for span in sorted(all_spans, key=lambda span: span.part)) == long_message


async def test_a_message_that_is_never_split_keeps_the_single_span_default() -> None:
    session = session_of("short", "also short")

    episodes = await builder(min_size=1, target_size=5, max_size=1000).build(session)

    spans = [span for episode in episodes for span in episode.spans]
    assert all(span.part == 0 and span.part_count == 1 for span in spans)


async def test_a_tiny_acknowledgement_is_absorbed_into_the_episode_it_follows() -> None:
    session = session_of(
        ("Here is a fairly detailed explanation of the whole plan for the quarter.", Role.AGENT),
        ("thanks", Role.USER),
    )

    episodes = await builder(min_size=10, target_size=40, max_size=500).build(session)

    assert len(episodes) == 1
    assert [span.text for span in episodes[0].spans] == [session.messages[0].text, "thanks"]


async def test_a_protected_region_is_not_cut_internally_below_max_size() -> None:
    url = "https://example.test/path/to/a/resource?query=value&more=stuff-here"
    words = " ".join(f"word{index}" for index in range(40))
    session = session_of(f"{words} check this out {url} let me know what you think {words}")

    episodes = await SimilarityEpisodeBuilder(
        embeddings=FakeEmbeddingProvider(),
        settings=SimilaritySegmentationSettings(min_size=10, target_size=60, max_size=len(url) + 10),
    ).build(session)

    # Forced below max_size, the message is still only ever cut outside the URL.
    spans = [span for episode in episodes for span in episode.spans]
    assert len(spans) > 1
    assert any(url in span.text for span in spans)
    assert "".join(span.text for span in spans) == session.messages[0].text


async def test_the_split_point_favours_the_real_topic_change() -> None:
    # Four turns, two topics. A real boundary exists only between turns 2 and 3;
    # the keyword embedder makes that the one low-similarity seam in the window.
    alpha_a = "alpha topic details discussion content words here today"
    alpha_b = "alpha topic further details and discussion continuing on"
    beta_a = "beta subject matter conversation information words there"
    beta_b = "beta subject additional conversation information shared"
    session = session_of(alpha_a, alpha_b, beta_a, beta_b)
    size = max(len(text) for text in (alpha_a, alpha_b, beta_a, beta_b))

    episodes = await SimilarityEpisodeBuilder(
        embeddings=KeywordEmbeddingProvider(),
        settings=SimilaritySegmentationSettings(min_size=1, target_size=2 * size, max_size=3 * size),
    ).build(session)

    assert len(episodes) == 2
    assert all("beta" not in span.text for span in episodes[0].spans)
    assert all("alpha" not in span.text for span in episodes[1].spans)


def test_settings_require_ordered_bounds() -> None:
    with pytest.raises(ValueError, match="min_size <= target_size <= max_size"):
        SimilaritySegmentationSettings(min_size=100, target_size=50, max_size=200)
