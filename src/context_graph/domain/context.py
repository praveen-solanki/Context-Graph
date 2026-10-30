"""Canonical context model: reusable concept, scoped instance, evidence."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from context_graph.domain.enums import (
    Authority,
    Cardinality,
    ContextStatus,
    ContextType,
    ScopeType,
    SourceTrust,
    SourceType,
    Visibility,
)
from context_graph.domain.ids import canonical_value, concept_key, context_fingerprint

DEFAULT_NAMESPACE = "personalization"

#: Subject of a shared-scope context: the scope itself, not an individual user.
#: Independent users reporting the same convention therefore accumulate evidence
#: on one candidate instead of creating one instance each.
SHARED_SUBJECT = ""


class MemoryConcept(BaseModel):
    """Reusable semantic anchor that similar extractions resolve onto."""

    model_config = ConfigDict(frozen=True)

    concept_id: str
    concept: str
    concept_key: str
    namespace: str = DEFAULT_NAMESPACE
    description: str | None = None
    cardinality: Cardinality = Cardinality.SINGLE
    value_type: str | None = None

    @staticmethod
    def key_for(concept: str, namespace: str = DEFAULT_NAMESPACE) -> str:
        """Deterministic concept key (exact-duplicate protection only)."""
        return concept_key(namespace, concept)


class Provenance(BaseModel):
    """Who produced the evidence behind a context instance and how trusted it is."""

    model_config = ConfigDict(frozen=True)

    source_episode_id: str
    source_session_id: str
    source_actor: str
    source_type: SourceType
    extraction_confidence: float = Field(ge=0.0, le=1.0)

    @property
    def source_trust(self) -> SourceTrust:
        return self.source_type.trust


class Lineage(BaseModel):
    """Model and policy versions that produced a context instance."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    extractor_model: str
    prompt_version: str
    policy_version: str
    embedding_model: str
    embedding_dimension: int
    adjudicator_model: str | None = None


class ContextInstance(BaseModel):
    """A concrete value/state for a subject within a scope and validity window."""

    model_config = ConfigDict(frozen=True)

    context_id: str
    fingerprint: str
    context_type: ContextType
    concept_id: str
    concept_key: str
    #: Empty for shared scopes; see :data:`SHARED_SUBJECT`.
    subject_user_id: str
    scope_type: ScopeType
    scope_id: str
    value: str | float | bool | dict[str, Any] | list[Any]
    cardinality: Cardinality = Cardinality.SINGLE
    condition: str | None = None
    visibility: Visibility = Visibility.PRIVATE
    authority: Authority = Authority.USER_ASSERTED
    status: ContextStatus = ContextStatus.ACTIVE
    observed_at: AwareDatetime
    recorded_at: AwareDatetime
    valid_from: AwareDatetime | None = None
    valid_until: AwareDatetime | None = None
    superseded_at: AwareDatetime | None = None
    provenance: Provenance
    lineage: Lineage

    @property
    def value_key(self) -> str:
        """Canonical value form used for equality checks."""
        return canonical_value(self.value)

    @staticmethod
    def fingerprint_for(
        *,
        subject_user_id: str,
        scope_type: ScopeType,
        scope_id: str,
        concept_id: str,
        value: Any,
    ) -> str:
        """Fingerprint used to MERGE a scoped value safely under concurrency."""
        return context_fingerprint(
            subject_user_id=subject_user_id,
            scope_type=scope_type.value,
            scope_id=scope_id,
            concept_id=concept_id,
            value=value,
        )


class EvidenceLink(BaseModel):
    """A ``Context -[:SUPPORTED_BY]-> Episode`` edge.

    The source episode is the idempotency key: retries and re-extraction must
    not increase support (spec section 6).
    """

    model_config = ConfigDict(frozen=True)

    context_id: str
    episode_id: str
    session_id: str
    actor_id: str
    source_type: SourceType
    extraction_confidence: float = Field(ge=0.0, le=1.0)
    observed_at: AwareDatetime
    recorded_at: AwareDatetime

    @property
    def source_trust(self) -> SourceTrust:
        return self.source_type.trust


class ContextSupport(BaseModel):
    """Aggregated, de-duplicated support for a context instance."""

    model_config = ConfigDict(frozen=True)

    context_id: str
    evidence_count: int = Field(ge=0)
    distinct_sessions: int = Field(ge=0)
    distinct_users: int = Field(ge=0)
    first_observed_at: datetime | None = None
    last_observed_at: datetime | None = None
    has_authoritative_source: bool = False
