"""Conversation input and immutable episodic evidence (spec section 2)."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Any, Self

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from context_graph.domain.enums import Role, SessionState, SourceType
from context_graph.domain.ids import deterministic_id


class MessageInput(BaseModel):
    """One upstream conversation message as received by the ingestion tool."""

    model_config = ConfigDict(frozen=True)

    message_id: str
    timestamp: AwareDatetime
    role: Role
    sequence: int = Field(ge=0)
    text: str
    sender_id: str | None = None
    agent_id: str | None = None
    episode_id: str | None = None
    source_type: SourceType | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SessionInput(BaseModel):
    """A conversation session submitted for ingestion."""

    model_config = ConfigDict(frozen=True)

    session_id: str
    user_id: str
    messages: tuple[MessageInput, ...]
    agent_ids: tuple[str, ...] = ()
    session_start_time: AwareDatetime | None = None
    session_end_time: AwareDatetime | None = None
    #: Lifecycle (A.9). Omit it: a session with an end time is CLOSED, otherwise
    #: OPEN. Pass it explicitly only for an end event that carries no timestamp.
    state: SessionState = SessionState.OPEN
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _derive_state(cls, data: Any) -> Any:
        """An end time means the session is closed, so the caller need not say so."""
        if isinstance(data, dict) and data.get("session_end_time") and not data.get("state"):
            return {**data, "state": SessionState.CLOSED}
        return data

    @model_validator(mode="after")
    def _validate(self) -> Self:
        sequences = [message.sequence for message in self.messages]
        if len(set(sequences)) != len(sequences):
            raise ValueError(f"duplicate message sequence in session {self.session_id}")
        if self.session_end_time is not None and self.state is SessionState.OPEN:
            raise ValueError("session marked OPEN cannot carry an end time")
        return self

    def ordered_messages(self) -> list[MessageInput]:
        """Messages in their supplied sequence order."""
        return sorted(self.messages, key=lambda message: message.sequence)

    def speaker_of(self, message: MessageInput) -> str:
        """Resolve a message's speaker, never inventing one."""
        return (
            message.sender_id
            or message.agent_id
            or (self.user_id if message.role is Role.USER else None)
            or message.role.value
        )


class SourceSpan(BaseModel):
    """An exact, contiguous slice of one original message.

    A span is the unit of provenance: it keeps the message it came from, that
    message's own speaker and timestamp, and the character offsets that locate
    the text inside it. Splitting a message never rewrites its text.
    """

    model_config = ConfigDict(frozen=True)

    message_id: str
    sequence: int = Field(ge=0)
    role: Role
    sender_id: str
    timestamp: AwareDatetime
    source_type: SourceType
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)
    text: str
    #: Index of this span within its message when the message was split.
    part: int = Field(default=0, ge=0)
    part_count: int = Field(default=1, ge=1)
    #: The original message metadata, carried through unchanged.
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_offsets(self) -> Self:
        if self.char_end < self.char_start:
            raise ValueError("span end must not precede its start")
        if len(self.text) != self.char_end - self.char_start:
            raise ValueError("span text does not match its offsets")
        return self

    @property
    def ref(self) -> str:
        """Stable reference used when asking a model to group spans."""
        return f"s{self.sequence}-{self.part}"

    @property
    def is_partial(self) -> bool:
        return self.part_count > 1

    @classmethod
    def whole_message(cls, message: MessageInput, sender_id: str) -> SourceSpan:
        """A span covering an entire message."""
        return cls(
            message_id=message.message_id,
            sequence=message.sequence,
            role=message.role,
            sender_id=sender_id,
            timestamp=message.timestamp,
            source_type=message.source_type or SourceType.from_role(message.role),
            char_start=0,
            char_end=len(message.text),
            text=message.text,
            metadata=message.metadata,
        )

    def slice(self, start: int, end: int, part: int, part_count: int) -> SourceSpan:
        """A sub-span of this span, offsets relative to the original message."""
        return self.model_copy(
            update={
                "char_start": start,
                "char_end": end,
                "text": self.text[start - self.char_start : end - self.char_start],
                "part": part,
                "part_count": part_count,
            }
        )


class Episode(BaseModel):
    """Immutable conversational evidence: a bounded group of source spans.

    An Episode holds related conversation content rather than a single message,
    so a short reply stays with the discussion it belongs to. Identity is
    ``(session_id, episode_id)``: re-submitting the same pair is the same
    Episode and must not be duplicated.

    The single-message fields (``role``, ``sender_id``, ``timestamp``,
    ``message_id``, ``source_type``) are derived from the first span, so every
    downstream consumer keeps working unchanged; per-span speakers, times and
    offsets remain available in :attr:`spans`.
    """

    model_config = ConfigDict(frozen=True)

    episode_id: str
    session_id: str
    sequence: int = Field(ge=0)
    spans: tuple[SourceSpan, ...] = Field(min_length=1)
    #: Set when a discussion outgrew the episode size limit and continued.
    continues_episode_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_spans(self) -> Self:
        ordered = [(span.sequence, span.part) for span in self.spans]
        if ordered != sorted(ordered):
            raise ValueError(f"episode {self.episode_id} spans are out of order")
        if len(set(ordered)) != len(ordered):
            raise ValueError(f"episode {self.episode_id} repeats a source span")
        return self

    @classmethod
    def from_spans(
        cls,
        *,
        session_id: str,
        sequence: int,
        spans: Sequence[SourceSpan],
        continues_episode_id: str | None = None,
    ) -> Episode:
        """Build an Episode with a deterministic identity from its own spans."""
        first, last = spans[0], spans[-1]
        return cls(
            episode_id=deterministic_id(
                "ep",
                session_id,
                first.message_id,
                str(first.char_start),
                last.message_id,
                str(last.char_end),
            ),
            session_id=session_id,
            sequence=sequence,
            spans=tuple(spans),
            continues_episode_id=continues_episode_id,
        )

    @property
    def text(self) -> str:
        """The conversation this episode holds, speaker-labelled and in order."""
        return "\n".join(f"{span.role.value}: {span.text}" for span in self.spans)

    @property
    def content(self) -> str:
        """The spoken content alone, without speaker labels."""
        return "\n".join(span.text for span in self.spans)

    @property
    def message_id(self) -> str:
        """First contributing message; full list in :attr:`message_ids`."""
        return self.spans[0].message_id

    @property
    def role(self) -> Role:
        """Role of the first span; every span keeps its own."""
        return self.spans[0].role

    @property
    def sender_id(self) -> str:
        """Speaker of the first span; every span keeps its own."""
        return self.spans[0].sender_id

    @property
    def source_type(self) -> SourceType:
        """Source type of the first span; every span keeps its own."""
        return self.spans[0].source_type

    @property
    def timestamp(self) -> datetime:
        """When the episode's first span was said."""
        return self.spans[0].timestamp

    @property
    def ended_at(self) -> datetime:
        """When the episode's last span was said."""
        return self.spans[-1].timestamp

    @property
    def observed_at(self) -> datetime:
        """Time the underlying conversation happened."""
        return self.timestamp

    @property
    def message_ids(self) -> tuple[str, ...]:
        """Distinct messages contributing to this episode, in order."""
        return tuple(dict.fromkeys(span.message_id for span in self.spans))

    @property
    def senders(self) -> tuple[str, ...]:
        """Distinct speakers, in order of first appearance."""
        return tuple(dict.fromkeys(span.sender_id for span in self.spans))

    @property
    def user_senders(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(span.sender_id for span in self.spans if span.role is Role.USER)
        )

    @property
    def agent_senders(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(span.sender_id for span in self.spans if span.role is Role.AGENT)
        )

    @property
    def roles(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(span.role.value for span in self.spans))
