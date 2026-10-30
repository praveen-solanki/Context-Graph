"""Incremental detection and checkpoint contiguity (spec A.4)."""

from __future__ import annotations

from context_graph.ledger.models import SessionCheckpoint


def checkpoint(sequence: int, completed: list[int]) -> SessionCheckpoint:
    return SessionCheckpoint(
        run_id="r1", session_id="S1", checkpoint_sequence=sequence, completed_above=completed
    )


def test_checkpoint_stops_at_the_first_gap() -> None:
    # 104 failed, 105 succeeded: the checkpoint must stay at 103.
    advanced = checkpoint(103, [105]).advanced_through([104, 105, 106])
    assert advanced.checkpoint_sequence == 103
    assert advanced.completed_above == [105]


def test_checkpoint_advances_through_contiguous_completions() -> None:
    advanced = checkpoint(103, [104, 105]).advanced_through([104, 105, 106])
    assert advanced.checkpoint_sequence == 105
    assert advanced.completed_above == []


def test_checkpoint_ignores_sequences_at_or_below_itself() -> None:
    advanced = checkpoint(103, [104]).advanced_through([101, 102, 103, 104])
    assert advanced.checkpoint_sequence == 104


def test_checkpoint_is_unchanged_when_nothing_is_completed() -> None:
    current = checkpoint(103, [])
    assert current.advanced_through([104, 105]) is current


def test_fresh_session_starts_before_the_first_sequence() -> None:
    fresh = SessionCheckpoint(run_id="r1", session_id="S1")
    assert fresh.checkpoint_sequence == -1
    assert fresh.advanced_through([0, 1]).checkpoint_sequence == -1
