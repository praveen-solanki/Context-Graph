"""Domain identity, validation and batching-envelope behaviour."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from context_graph.domain.context import ContextInstance
from context_graph.domain.conversation import MessageInput, SessionInput
from context_graph.domain.enums import Role, ScopeType, SourceTrust, SourceType
from context_graph.domain.extraction import EpisodeChunk, ExtractedSignal, ExtractionBatch, ExtractionEnvelope
from tests.factories import BASE_TIME, make_message, single_span_episode


def test_episode_identity_is_stable_for_same_session_and_message() -> None:
    message = make_message(1, "hello")
    first = single_span_episode("S1", message)
    second = single_span_episode("S1", message)
    other_session = single_span_episode("S2", message)

    assert first.episode_id == second.episode_id
    assert first.episode_id != other_session.episode_id


def test_session_rejects_duplicate_sequences() -> None:
    with pytest.raises(ValidationError):
        SessionInput(
            session_id="S1",
            user_id="U1",
            messages=(make_message(1, "a"), make_message(1, "b")),
        )


def test_naive_timestamps_are_rejected_at_the_boundary() -> None:
    with pytest.raises(ValidationError):
        MessageInput(
            message_id="m1",
            timestamp=datetime(2026, 9, 12, 10, 0),  # noqa: DTZ001 - intentionally naive
            role=Role.USER,
            sequence=1,
            text="hi",
        )


def test_episodes_are_ordered_by_sequence() -> None:
    session = SessionInput(
        session_id="S1",
        user_id="U1",
        messages=(make_message(3, "c"), make_message(1, "a"), make_message(2, "b")),
    )
    assert [message.sequence for message in session.ordered_messages()] == [1, 2, 3]


def test_source_trust_separates_first_party_text_from_derived_output() -> None:
    assert SourceType.from_role(Role.USER).trust is SourceTrust.FIRST_PARTY
    assert SourceType.from_role(Role.AGENT).trust is SourceTrust.DERIVED
    assert SourceType.from_role(Role.TOOL).trust is SourceTrust.UNVERIFIED


def test_fingerprint_is_value_sensitive_and_scope_sensitive() -> None:
    def fingerprint(value: str, scope: ScopeType = ScopeType.USER, scope_id: str = "U1") -> str:
        return ContextInstance.fingerprint_for(
            subject_user_id="U1",
            scope_type=scope,
            scope_id=scope_id,
            concept_id="c1",
            value=value,
        )

    assert fingerprint("Concise ") == fingerprint("concise")
    assert fingerprint("concise") != fingerprint("detailed")
    assert fingerprint("concise") != fingerprint("concise", ScopeType.SESSION, "S1")


def _chunk(sequence: int) -> EpisodeChunk:
    episode = single_span_episode("S1", make_message(sequence, f"message {sequence}"))
    return EpisodeChunk(episode=episode, char_end=len(episode.text), token_estimate=10)


def test_batch_key_is_deterministic_and_split_halves_the_batch() -> None:
    chunks = tuple(_chunk(index) for index in range(1, 5))
    batch = ExtractionBatch(run_id="r1", session_id="S1", chunks=chunks, token_estimate=40)

    assert batch.batch_key == batch.model_copy().batch_key
    first, second = batch.split()
    assert len(first.chunks) == 2 and len(second.chunks) == 2
    assert first.token_estimate == 20
    assert ExtractionBatch(
        run_id="r1", session_id="S1", chunks=chunks[:1], token_estimate=10
    ).split() == ()


def test_envelope_drops_signals_attributed_outside_the_batch() -> None:
    envelope = ExtractionEnvelope(
        signals=(
            ExtractedSignal(concept="response_detail", value="concise", source_episode_ids=("e1", "e9")),
            ExtractedSignal(concept="stale", value="x", source_episode_ids=("e9",)),
        )
    )
    filtered = envelope.for_episodes(frozenset({"e1"}))

    assert len(filtered.signals) == 1
    assert filtered.signals[0].source_episode_ids == ("e1",)


def test_signal_rejects_inverted_validity_window() -> None:
    with pytest.raises(ValidationError):
        ExtractedSignal(
            concept="c",
            value="v",
            source_episode_ids=("e1",),
            valid_from=BASE_TIME,
            valid_until=datetime(2025, 1, 1, tzinfo=UTC),
        )


@pytest.mark.parametrize("literal", ["null", "None", "", " NULL "])
def test_signal_reads_a_stringified_null_as_absent(literal: str) -> None:
    signal = ExtractedSignal.model_validate(
        {
            "concept": "c",
            "value": "v",
            "source_episode_ids": ["e1"],
            "condition": literal,
            "valid_from": literal,
            "valid_until": literal,
            "temporal_cue": literal,
            "evidence_quote": literal,
            "cardinality": literal,
            "scope_cue": {"scope_type": "USER", "scope_ref": literal},
        }
    )

    assert signal.condition is None
    assert signal.valid_from is None and signal.valid_until is None
    assert signal.temporal_cue is None and signal.evidence_quote is None
    assert signal.cardinality is None
    assert signal.scope_cue is not None and signal.scope_cue.scope_ref is None


def test_signal_keeps_a_real_condition_and_value() -> None:
    signal = ExtractedSignal(concept="c", value="null", source_episode_ids=("e1",), condition="on Fridays")

    assert signal.condition == "on Fridays"
    assert signal.value == "null"


def test_session_state_defaults_to_open_and_follows_the_end_time() -> None:
    from context_graph.domain.enums import SessionState

    ongoing = SessionInput(session_id="S1", user_id="U1", messages=(make_message(1, "hi"),))
    assert ongoing.state is SessionState.OPEN

    ended = SessionInput(
        session_id="S1",
        user_id="U1",
        messages=(make_message(1, "hi"),),
        session_end_time=BASE_TIME,
    )
    assert ended.state is SessionState.CLOSED


def test_session_explicitly_marked_open_cannot_carry_an_end_time() -> None:
    from context_graph.domain.enums import SessionState

    with pytest.raises(ValidationError):
        SessionInput(
            session_id="S1",
            user_id="U1",
            messages=(make_message(1, "hi"),),
            session_end_time=BASE_TIME,
            state=SessionState.OPEN,
        )


def test_episode_exposes_first_span_fields_for_existing_consumers() -> None:
    from context_graph.domain.conversation import Episode, SourceSpan

    user = make_message(1, "Keep answers concise", Role.USER)
    agent = make_message(2, "Understood.", Role.AGENT)
    episode = Episode.from_spans(
        session_id="S1",
        sequence=0,
        spans=[
            SourceSpan.whole_message(user, "U1"),
            SourceSpan.whole_message(agent, "agent-1"),
        ],
    )

    # The single-message surface downstream code relies on still works.
    assert episode.role is Role.USER
    assert episode.sender_id == "U1"
    assert episode.message_id == "m1"
    assert episode.timestamp == user.timestamp
    assert episode.ended_at == agent.timestamp
    assert episode.source_type is SourceType.USER_MESSAGE
    # And the full membership is available.
    assert episode.message_ids == ("m1", "m2")
    assert episode.senders == ("U1", "agent-1")
    assert episode.text == "user: Keep answers concise\nagent: Understood."
    assert episode.content == "Keep answers concise\nUnderstood."


def test_episode_rejects_out_of_order_or_repeated_spans() -> None:
    from context_graph.domain.conversation import Episode, SourceSpan

    first = SourceSpan.whole_message(make_message(1, "a"), "U1")
    second = SourceSpan.whole_message(make_message(2, "b"), "U1")

    with pytest.raises(ValidationError):
        Episode(episode_id="e", session_id="S1", sequence=0, spans=(second, first))
    with pytest.raises(ValidationError):
        Episode(episode_id="e", session_id="S1", sequence=0, spans=(first, first))


def test_source_span_text_must_match_its_offsets() -> None:
    from context_graph.domain.conversation import SourceSpan

    with pytest.raises(ValidationError):
        SourceSpan(
            message_id="m1",
            sequence=1,
            role=Role.USER,
            sender_id="U1",
            timestamp=BASE_TIME,
            source_type=SourceType.USER_MESSAGE,
            char_start=0,
            char_end=99,
            text="short",
        )


def test_episode_survives_a_storage_round_trip_with_all_span_detail() -> None:
    from context_graph.domain.conversation import Episode, SourceSpan
    from context_graph.graph import mapping

    user = make_message(1, "Keep answers concise", Role.USER)
    agent = make_message(2, "Understood.", Role.AGENT)
    episode = Episode.from_spans(
        session_id="S1",
        sequence=3,
        spans=[
            SourceSpan.whole_message(user, "U1"),
            SourceSpan.whole_message(agent, "agent-1"),
        ],
    )

    restored = mapping.episode_from_node(mapping.episode_properties(episode))

    assert restored == episode
    assert [span.text for span in restored.spans] == ["Keep answers concise", "Understood."]
    assert [span.sender_id for span in restored.spans] == ["U1", "agent-1"]


def test_episodes_stored_before_spans_existed_are_still_readable() -> None:
    """Backward compatibility with per-message episodes already in the graph."""
    from context_graph.graph import mapping

    legacy = {
        "episode_id": "ep_legacy",
        "session_id": "S1",
        "message_id": "m1",
        "role": "user",
        "sender_id": "U1",
        "timestamp": BASE_TIME,
        "sequence": 4,
        "text": "Always keep my answers concise.",
        "source_type": "user_message",
        "metadata": "{}",
    }

    episode = mapping.episode_from_node(legacy)

    assert len(episode.spans) == 1
    assert episode.spans[0].text == "Always keep my answers concise."
    assert episode.spans[0].char_start == 0
    assert episode.spans[0].char_end == len(legacy["text"])  # pyright: ignore[reportArgumentType]
    assert episode.content == "Always keep my answers concise."
    assert episode.sequence == 4
