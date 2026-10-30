"""Deterministic write-time policy: subject, scope, visibility and authority.

The LLM may detect that a statement concerns a Team, Project or Organization.
It cannot grant itself write permission: this module decides whether the source
may create or publish context at the target scope (spec section 5 and 10).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from context_graph.config import PromotionSettings
from context_graph.domain.context import SHARED_SUBJECT, ContextInstance, ContextSupport
from context_graph.domain.conversation import Episode, SessionInput
from context_graph.domain.enums import (
    Authority,
    ContextStatus,
    ScopeType,
    SourceTrust,
    Visibility,
)
from context_graph.domain.extraction import ExtractedSignal
from context_graph.domain.ids import slugify

#: Bump when these rules change; stored in every Context's lineage.
POLICY_VERSION = "writer-policy-v1"

#: Ways a conversation refers to the speaking user.
_SELF_REFERENCES = frozenset({"user", "me", "i", "myself", "my", "self", "the user"})


class ScopeAssignment(BaseModel):
    """An allowed write target with its governance metadata."""

    model_config = ConfigDict(frozen=True)

    subject_user_id: str
    scope_type: ScopeType
    scope_id: str
    visibility: Visibility
    authority: Authority
    status: ContextStatus

    @property
    def is_shared_candidate(self) -> bool:
        """Shared-scope proposals are stored as candidates, never published."""
        return self.status is ContextStatus.CANDIDATE


class UnresolvedSignal(BaseModel):
    """A signal that cannot be written safely and is kept unresolved."""

    model_config = ConfigDict(frozen=True)

    reason: Literal["unknown_subject", "unresolved_scope"]
    detail: str


WriteDecision = ScopeAssignment | UnresolvedSignal


class WriterPolicy:
    """Decides the scope, visibility, authority and status of every signal."""

    def __init__(self, promotion: PromotionSettings) -> None:
        self._promotion = promotion

    def decide(
        self, signal: ExtractedSignal, session: SessionInput, episode: Episode
    ) -> WriteDecision:
        """Resolve where, and with what authority, a signal may be written."""
        if signal.subject.strip().casefold() not in _SELF_REFERENCES:
            return UnresolvedSignal(
                reason="unknown_subject",
                detail=f"subject '{signal.subject}' cannot be resolved to a known user",
            )

        scope_type, scope_id = self._scope(signal, session)
        if scope_id is None:
            return UnresolvedSignal(
                reason="unresolved_scope",
                detail=f"{scope_type} scope was cued without a resolvable identifier",
            )

        authority = (
            Authority.USER_ASSERTED
            if episode.source_type.trust is SourceTrust.FIRST_PARTY
            else Authority.AGENT_DERIVED
        )
        return ScopeAssignment(
            # Shared context belongs to its scope, so independent users
            # contribute evidence to one candidate rather than to one each.
            subject_user_id=SHARED_SUBJECT if scope_type.is_shared else session.user_id,
            scope_type=scope_type,
            scope_id=scope_id,
            visibility=Visibility.SHARED if scope_type.is_shared else Visibility.PRIVATE,
            authority=authority,
            # A normal conversation may contribute evidence to a shared scope,
            # but it can never publish there directly.
            status=ContextStatus.CANDIDATE if scope_type.is_shared else ContextStatus.ACTIVE,
        )

    @staticmethod
    def _scope(signal: ExtractedSignal, session: SessionInput) -> tuple[ScopeType, str | None]:
        """Map the LLM's scope cue onto a concrete, resolvable scope."""
        cue = signal.scope_cue
        if cue is None:
            # No cue: explicit durability language makes it a user preference,
            # anything else stays with the session it was said in (A.9).
            scope_type = ScopeType.USER if signal.durability_cue else ScopeType.SESSION
        else:
            scope_type = cue.scope_type

        if scope_type is ScopeType.USER:
            return scope_type, session.user_id
        if scope_type is ScopeType.SESSION:
            return scope_type, session.session_id
        reference = cue.scope_ref if cue else None
        return scope_type, slugify(reference) if reference else None

    def may_promote(self, context: ContextInstance, support: ContextSupport) -> bool:
        """Whether accumulated evidence satisfies this scope's promotion rule."""
        if context.status is not ContextStatus.CANDIDATE:
            return False
        rule = self._promotion.rule_for(context.scope_type)
        if rule.requires_authoritative_source and not support.has_authoritative_source:
            return False
        return (
            support.evidence_count >= rule.min_evidence
            and support.distinct_sessions >= rule.min_sessions
            and support.distinct_users >= rule.min_users
        )
