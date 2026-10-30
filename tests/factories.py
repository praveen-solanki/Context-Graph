"""Plain builders shared across test suites.

Fixtures live in ``conftest.py``; anything a test module imports by name lives
here, because importing ``conftest`` directly is not supported by pytest.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from context_graph.domain.conversation import (
    Episode,
    MessageInput,
    SessionInput,
    SourceSpan,
)
from context_graph.domain.enums import Role

REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_TIME = datetime(2026, 9, 12, 10, 0, tzinfo=UTC)
NOW = datetime(2026, 9, 13, 9, 0, tzinfo=UTC)


def make_message(sequence: int, text: str, role: Role = Role.USER) -> MessageInput:
    """Build a message with predictable identity and timestamp."""
    return MessageInput(
        message_id=f"m{sequence}",
        timestamp=BASE_TIME.replace(minute=sequence % 60),
        role=role,
        sequence=sequence,
        text=text,
    )


def single_span_episode(
    session_id: str, message: MessageInput, sender_id: str | None = None, sequence: int | None = None
) -> Episode:
    """One message as one episode - the smallest episode a session can produce."""
    speaker = sender_id or message.sender_id or message.agent_id or message.role.value
    return Episode.from_spans(
        session_id=session_id,
        sequence=message.sequence if sequence is None else sequence,
        spans=[SourceSpan.whole_message(message, speaker)],
    )


def single_span_episodes(session: SessionInput) -> list[Episode]:
    """Every message of a session as its own episode."""
    return [
        single_span_episode(session.session_id, message, session.speaker_of(message))
        for message in session.ordered_messages()
    ]


def context_node(
    concept: str = "response_detail",
    value: str = "concise",
    scope_type: str = "USER",
    scope_id: str = "U1",
    evidence: int = 2,
) -> dict[str, Any]:
    """A stored Context node as Neo4j returns it."""
    return {
        "context_id": f"ctx_{concept}",
        "fingerprint": f"fp_{concept}",
        "context_type": "Preference",
        "concept_id": f"concept_{concept}",
        "concept_key": f"personalization:{concept}",
        "subject_user_id": "U1",
        "scope_type": scope_type,
        "scope_id": scope_id,
        "value": value,
        "value_key": value,
        "cardinality": "single",
        "condition": None,
        "visibility": "private",
        "authority": "user_asserted",
        "status": "active",
        "observed_at": NOW,
        "recorded_at": NOW,
        "valid_from": None,
        "valid_until": None,
        "superseded_at": None,
        "evidence_count": evidence,
        "source_episode_id": "e1",
        "source_session_id": "S1",
        "source_actor": "U1",
        "source_type": "user_message",
        "source_trust": "first_party",
        "extraction_confidence": 0.9,
        "run_id": "run-1",
        "extractor_model": "gpt-test",
        "prompt_version": "p",
        "policy_version": "v",
        "embedding_model": "embed-test",
        "embedding_dimension": 8,
        "adjudicator_model": None,
    }
