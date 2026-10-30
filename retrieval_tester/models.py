"""Models for retrieval results and retrieval-tester sessions."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from context_graph.domain.conversation import MessageInput, SessionInput
from context_graph.domain.enums import ContextType, Role, ScopeType, SessionState

#: Agent identity recorded on the agent's own turns.
AGENT_ID = "dummy-agent"


def utcnow() -> datetime:
    """Wall clock for this application."""
    return datetime.now(UTC)


#: Context that is a standing instruction: always given to the agent.
STANDING_TYPES = frozenset(
    {ContextType.PREFERENCE, ContextType.CONSTRAINT, ContextType.WORKING_STYLE}
)


class RetrievedContext(BaseModel):
    """One Context instance selected for the agent, as stored in the graph."""

    model_config = ConfigDict(frozen=True)

    context_id: str
    concept_key: str
    context_type: ContextType
    value: str
    scope_type: ScopeType
    scope_id: str
    condition: str | None = None
    evidence_count: int = 0
    observed_at: AwareDatetime | None = None
    valid_until: AwareDatetime | None = None
    #: First supporting episode text, so the agent can say where this came from.
    evidence_quote: str | None = None
    #: Similarity to the current message, when relevance ranking ran.
    similarity: float | None = None

    @property
    def concept(self) -> str:
        """Concept name without its namespace."""
        return self.concept_key.split(":")[-1]

    @property
    def is_standing_instruction(self) -> bool:
        """Whether this must always be given to the agent, regardless of topic."""
        return self.context_type in STANDING_TYPES

    def as_prompt_line(self) -> str:
        """Render for the agent's system prompt."""
        condition = f" (only when {self.condition})" if self.condition else ""
        quote = f' — said: "{self.evidence_quote}"' if self.evidence_quote else ""
        return f"- [{self.scope_type.value}] {self.concept}: {self.value}{condition}{quote}"


class Colleague(BaseModel):
    """A teammate, from the organizational directory."""

    model_config = ConfigDict(frozen=True)

    user_id: str
    name: str
    title: str | None = None
    projects: tuple[str, ...] = ()

    def as_prompt_line(self) -> str:
        role = f", {self.title}" if self.title else ""
        working = f" — works on {', '.join(self.projects)}" if self.projects else ""
        return f"- {self.name}{role}{working}"


class Directory(BaseModel):
    """Where this user sits in the organization, and who is around them.

    Directory facts are shared: everyone in a team may see who else is in it and
    what the team owns. Personal context stays private to its owner.
    """

    model_config = ConfigDict(frozen=True)

    team_id: str | None = None
    team_name: str | None = None
    department_name: str | None = None
    organization_name: str | None = None
    colleagues: tuple[Colleague, ...] = ()
    team_projects: tuple[str, ...] = ()
    my_projects: tuple[str, ...] = ()

    @property
    def is_known(self) -> bool:
        return self.team_id is not None

    def as_prompt_block(self) -> str:
        if not self.is_known:
            return ""
        place = f"This user is on the {self.team_name} team"
        if self.department_name:
            place += f", part of {self.department_name}"
        if self.organization_name:
            place += f" at {self.organization_name}"
        lines = [place + "."]
        if self.my_projects:
            lines.append(f"They work on: {', '.join(self.my_projects)}.")
        if self.team_projects:
            lines.append(f"The team owns: {', '.join(self.team_projects)}.")
        if self.colleagues:
            lines.append("Their teammates:")
            lines.extend(colleague.as_prompt_line() for colleague in self.colleagues)
        return "\n".join(lines)


class SessionRecall(BaseModel):
    """A past conversation this user had, for episodic recall."""

    model_config = ConfigDict(frozen=True)

    session_id: str
    state: str
    started_at: AwareDatetime | None = None
    ended_at: AwareDatetime | None = None
    episode_count: int = 0
    opening_message: str | None = None
    #: Concepts that were derived from this conversation.
    concepts: tuple[str, ...] = ()

    def as_prompt_line(self) -> str:
        when = self.started_at.strftime("%Y-%m-%d %H:%M") if self.started_at else "unknown date"
        topics = ", ".join(self.concepts[:6]) or "nothing durable was recorded"
        opening = f' It opened with: "{self.opening_message}".' if self.opening_message else ""
        return (
            f"- {when} ({self.episode_count} messages, {self.state}): {topics}.{opening}"
        )


class ValueChange(BaseModel):
    """A preference that was replaced, so the agent knows what changed."""

    model_config = ConfigDict(frozen=True)

    concept_key: str
    previous_value: str
    current_value: str | None = None
    changed_at: AwareDatetime | None = None

    def as_prompt_line(self) -> str:
        concept = self.concept_key.split(":")[-1]
        now = f", now {self.current_value}" if self.current_value else ""
        when = f" ({self.changed_at:%Y-%m-%d})" if self.changed_at else ""
        return f"- {concept}: previously {self.previous_value}{now}{when}"


class RetrievalDebug(BaseModel):
    """What retrieval did, so the UI can show why the agent saw what it saw."""

    model_config = ConfigDict(frozen=True)

    user_id: str
    duration_ms: float
    candidates_considered: int
    returned: int
    truncated: bool = False
    by_scope: dict[str, int] = Field(default_factory=dict)
    #: The message relevance was ranked against, when there was one.
    query: str | None = None
    relevance_used: bool = False
    sessions_recalled: int = 0
    changes_recalled: int = 0
    directory_found: bool = False
    notes: tuple[str, ...] = ()


class RetrievalResult(BaseModel):
    """Everything retrieved for one request, and the record of how it was chosen."""

    model_config = ConfigDict(frozen=True)

    items: tuple[RetrievedContext, ...] = ()
    #: Where the user sits in the organization; shared with their team.
    directory: Directory = Field(default_factory=lambda: Directory())
    #: Past conversations, so the agent can answer "what did we discuss".
    sessions: tuple[SessionRecall, ...] = ()
    #: Preferences that were superseded, so the agent knows what changed.
    history: tuple[ValueChange, ...] = ()
    debug: RetrievalDebug

    @property
    def is_empty(self) -> bool:
        return not (self.items or self.sessions or self.history or self.directory.is_known)

    def as_prompt_block(self) -> str:
        """Render every section the agent should see."""
        sections: list[str] = []
        if self.directory.is_known:
            sections.append(self.directory.as_prompt_block())
        if self.items:
            sections.append(
                "What is true about this user:\n"
                + "\n".join(item.as_prompt_line() for item in self.items)
            )
        if self.sessions:
            sections.append(
                "Earlier conversations with this user, most recent first:\n"
                + "\n".join(session.as_prompt_line() for session in self.sessions)
            )
        if self.history:
            sections.append(
                "Preferences this user changed:\n"
                + "\n".join(change.as_prompt_line() for change in self.history)
            )
        return "\n\n".join(sections)


class AgentSession(BaseModel):
    """A retrieval-tester session, saved to local JSON only.

    The conversation is stored as a :class:`SessionInput`, the exact shape the
    ingestion tool accepts, so the saved file can be ingested later **if you
    choose to**. This application still never writes to the Context Graph and
    never calls ingestion: feeding a saved file to the pipeline is a separate,
    deliberate step you take yourself.
    """

    model_config = ConfigDict(frozen=True)

    session: SessionInput
    model: str
    retrieved_context: tuple[RetrievedContext, ...] = ()
    retrieval_debug: RetrievalDebug | None = None
    #: The organizational placement the agent was given.
    directory: Directory | None = None
    #: Structural lookups the agent ran against the graph, in order.
    graph_lookups: tuple[dict[str, Any], ...] = ()

    @classmethod
    def start(
        cls,
        *,
        session_id: str,
        user_id: str,
        model: str,
        retrieval: RetrievalResult,
    ) -> AgentSession:
        """Open a session for ``user_id`` with the Context retrieved for it."""
        return cls(
            session=SessionInput(
                session_id=session_id,
                user_id=user_id,
                agent_ids=(AGENT_ID,),
                messages=(),
                session_start_time=utcnow(),
                metadata={"source": "retrieval_tester", "model": model},
            ),
            model=model,
            retrieved_context=retrieval.items,
            retrieval_debug=retrieval.debug,
            directory=retrieval.directory,
        )

    @property
    def session_id(self) -> str:
        return self.session.session_id

    @property
    def user_id(self) -> str:
        return self.session.user_id

    @property
    def turns(self) -> tuple[MessageInput, ...]:
        """The conversation so far, in order."""
        return self.session.messages

    @property
    def ended_at(self) -> datetime | None:
        return self.session.session_end_time

    def with_message(self, role: Role, text: str) -> AgentSession:
        """Return the session with one more message appended."""
        sequence = len(self.session.messages) + 1
        message = MessageInput(
            message_id=f"{self.session_id}-m{sequence:03d}",
            timestamp=utcnow(),
            role=role,
            sequence=sequence,
            text=text,
            sender_id=self.user_id if role is Role.USER else None,
            agent_id=None if role is Role.USER else AGENT_ID,
        )
        return self.model_copy(
            update={"session": self.session.model_copy(update={"messages": (*self.session.messages, message)})}
        )

    def closed(self) -> AgentSession:
        """Return the session marked finished, so its Context is no longer live."""
        if self.session.session_end_time is not None:
            return self
        return self.model_copy(
            update={
                "session": self.session.model_copy(
                    update={"session_end_time": utcnow(), "state": SessionState.CLOSED}
                )
            }
        )

    def to_document(self) -> dict[str, Any]:
        """Serialize as an ingestion request plus retrieval-tester metadata.

        ``session`` is what the ingestion tool reads; ``retrieval_tester`` records
        what the agent saw and is ignored by the pipeline.
        """
        return {
            "session": self.session.model_dump(mode="json"),
            "retrieval_tester": {
                "model": self.model,
                "retrieved_context": [item.model_dump(mode="json") for item in self.retrieved_context],
                "retrieval_debug": self.retrieval_debug.model_dump(mode="json")
                if self.retrieval_debug
                else None,
                "directory": self.directory.model_dump(mode="json") if self.directory else None,
                "graph_lookups": list(self.graph_lookups),
            },
        }

    @classmethod
    def from_document(cls, document: dict[str, Any]) -> AgentSession:
        """Rebuild a session saved by :meth:`to_document`."""
        extra: dict[str, Any] = document.get("retrieval_tester") or {}
        return cls(
            session=SessionInput.model_validate(document["session"]),
            model=extra.get("model", "unknown"),
            retrieved_context=tuple(
                RetrievedContext.model_validate(item) for item in extra.get("retrieved_context", [])
            ),
            retrieval_debug=RetrievalDebug.model_validate(extra["retrieval_debug"])
            if extra.get("retrieval_debug")
            else None,
            directory=Directory.model_validate(extra["directory"])
            if extra.get("directory")
            else None,
            graph_lookups=tuple(extra.get("graph_lookups", [])),
        )
