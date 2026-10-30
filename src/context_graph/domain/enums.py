"""Controlled vocabulary for the Context Graph ontology.

Every string literal that carries meaning in the graph lives here so schema,
policy, persistence and prompts share one definition.
"""

from __future__ import annotations

from enum import StrEnum


class Role(StrEnum):
    """Who produced a conversation message."""

    USER = "user"
    AGENT = "agent"
    SYSTEM = "system"
    TOOL = "tool"


class SourceTrust(StrEnum):
    """How much the evidence source is trusted (spec section 6)."""

    FIRST_PARTY = "first_party"
    DERIVED = "derived"
    UNVERIFIED = "unverified"


class SourceType(StrEnum):
    """Provenance of the evidence behind a context instance."""

    USER_MESSAGE = "user_message"
    AGENT_MESSAGE = "agent_message"
    TOOL_OUTPUT = "tool_output"
    PASTED_CONTENT = "pasted_content"
    SYSTEM_MESSAGE = "system_message"

    @property
    def trust(self) -> SourceTrust:
        """First-party user text outranks agent, tool and pasted content."""
        if self is SourceType.USER_MESSAGE:
            return SourceTrust.FIRST_PARTY
        if self in (SourceType.AGENT_MESSAGE, SourceType.SYSTEM_MESSAGE):
            return SourceTrust.DERIVED
        return SourceTrust.UNVERIFIED

    @classmethod
    def from_role(cls, role: Role) -> SourceType:
        """Default source type for a message role."""
        return {
            Role.USER: cls.USER_MESSAGE,
            Role.AGENT: cls.AGENT_MESSAGE,
            Role.SYSTEM: cls.SYSTEM_MESSAGE,
            Role.TOOL: cls.TOOL_OUTPUT,
        }[role]


class ContextType(StrEnum):
    """Logical context types carried by a Context node (spec section 8)."""

    PREFERENCE = "Preference"
    GOAL = "Goal"
    TASK = "Task"
    DECISION = "Decision"
    CONSTRAINT = "Constraint"
    WORKING_STYLE = "WorkingStyle"
    STATE = "State"
    ACTIVITY = "Activity"
    EVENT = "Event"
    INTEREST = "Interest"
    FACT = "Fact"
    #: Genuinely new signal kept unresolved instead of forcing a known type.
    OTHER = "Other"


class ScopeType(StrEnum):
    """Scope a context instance applies to."""

    SESSION = "SESSION"
    USER = "USER"
    TASK = "TASK"
    PROJECT = "PROJECT"
    TEAM = "TEAM"
    DEPARTMENT = "DEPARTMENT"
    ORGANIZATION = "ORGANIZATION"

    @property
    def is_shared(self) -> bool:
        """Shared scopes require a governed writer; conversation only adds evidence."""
        return self in (ScopeType.TEAM, ScopeType.DEPARTMENT, ScopeType.ORGANIZATION)

    @property
    def node_label(self) -> str:
        """Ontology node label this scope resolves to for ``APPLIES_TO``."""
        return {
            ScopeType.SESSION: "Session",
            ScopeType.USER: "User",
            ScopeType.TASK: "Task",
            ScopeType.PROJECT: "Project",
            ScopeType.TEAM: "Team",
            ScopeType.DEPARTMENT: "Department",
            ScopeType.ORGANIZATION: "Organization",
        }[self]

    @property
    def id_property(self) -> str:
        """Identity property of the scope node."""
        return {
            ScopeType.SESSION: "session_id",
            ScopeType.USER: "user_id",
            ScopeType.TASK: "task_id",
            ScopeType.PROJECT: "project_id",
            ScopeType.TEAM: "team_id",
            ScopeType.DEPARTMENT: "department_id",
            ScopeType.ORGANIZATION: "organization_id",
        }[self]


class Visibility(StrEnum):
    """Whether a context instance is user-private or governed shared context."""

    PRIVATE = "private"
    SHARED = "shared"


class Authority(StrEnum):
    """Who asserted the context and with what write authority."""

    USER_ASSERTED = "user_asserted"
    AGENT_DERIVED = "agent_derived"
    AUTHORITATIVE = "authoritative"


class ContextStatus(StrEnum):
    """Lifecycle state of a context instance."""

    ACTIVE = "active"
    #: Shared-scope proposal awaiting governed promotion; never directly published.
    CANDIDATE = "candidate"
    SUPERSEDED = "superseded"
    REVOKED = "revoked"


class Cardinality(StrEnum):
    """Whether a concept holds one value, many values, or a set."""

    SINGLE = "single"
    MULTI = "multi"
    SET = "set"


class SessionState(StrEnum):
    """Session lifecycle (A.9)."""

    OPEN = "open"
    CLOSED = "closed"


class SignalOperation(StrEnum):
    """Whether an extracted signal asserts or withdraws context."""

    ASSERT = "assert"
    REVOKE = "revoke"


class GateDecision(StrEnum):
    """Cheap relevance gate outcome (A.7)."""

    EXTRACT = "extract"
    SKIP = "skip"


class SegmentationStrategy(StrEnum):
    """Which strategy groups a session's messages into episodes."""

    #: The model proposes topic boundaries over token-bounded windows (default).
    LLM_BOUNDARY = "llm_boundary"
    #: Character-size windows, split where embedding similarity is lowest.
    SIMILARITY = "similarity"


class EpisodeState(StrEnum):
    """Explicit processing states so each stage is distinguishable (section 12)."""

    DISCOVERED = "discovered"
    GATED_SKIPPED = "gated_skipped"
    EXTRACTED = "extracted"
    VALIDATED = "validated"
    GRAPH_COMMITTED = "graph_committed"
    CHECKPOINTED = "checkpointed"
    FAILED = "failed"


class BatchStatus(StrEnum):
    """Extraction batch lifecycle recorded in the MongoDB ledger (A.3)."""

    PENDING = "pending"
    EXTRACTED = "extracted"
    VALIDATED = "validated"
    COMMITTED = "committed"
    FAILED = "failed"
