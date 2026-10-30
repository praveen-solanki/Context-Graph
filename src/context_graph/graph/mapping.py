"""Conversion between domain models and flat Neo4j properties.

Neo4j stores scalars, so nested provenance and lineage are flattened here - in
one place - rather than in every query.
"""

from __future__ import annotations

import json
from typing import Any

from context_graph.domain.context import (
    ContextInstance,
    ContextSupport,
    EvidenceLink,
    Lineage,
    MemoryConcept,
    Provenance,
)
from context_graph.domain.conversation import Episode, SourceSpan
from context_graph.domain.enums import (
    Authority,
    Cardinality,
    ContextStatus,
    ContextType,
    Role,
    ScopeType,
    SourceType,
    Visibility,
)

_JSON_PREFIX = "json:"


def encode_value(value: Any) -> str | float | bool:
    """Store scalars natively; encode structured values as tagged JSON."""
    if isinstance(value, str | bool | int | float):
        return value
    return _JSON_PREFIX + json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def decode_value(value: Any) -> Any:
    """Inverse of :func:`encode_value`."""
    if isinstance(value, str) and value.startswith(_JSON_PREFIX):
        return json.loads(value[len(_JSON_PREFIX) :])
    return value


def episode_properties(episode: Episode) -> dict[str, Any]:
    """Flat properties for an ``:Episode`` node.

    Source spans are stored as one JSON property rather than as nodes: an
    episode's members are only ever read with the episode itself, so a node per
    message would add no traversal and a great many nodes.
    """
    return {
        "episode_id": episode.episode_id,
        "session_id": episode.session_id,
        # First-span fields, kept so existing readers and queries still work.
        "message_id": episode.message_id,
        "role": episode.role.value,
        "sender_id": episode.sender_id,
        "timestamp": episode.timestamp,
        "source_type": episode.source_type.value,
        "sequence": episode.sequence,
        "text": episode.text,
        "ended_at": episode.ended_at,
        "span_count": len(episode.spans),
        "message_ids": list(episode.message_ids),
        "roles": list(episode.roles),
        "senders": list(episode.senders),
        "user_senders": list(episode.user_senders),
        "agent_senders": list(episode.agent_senders),
        "continues_episode_id": episode.continues_episode_id,
        "spans": json.dumps(
            [span.model_dump(mode="json") for span in episode.spans], default=str
        ),
        "metadata": json.dumps(episode.metadata, sort_keys=True, default=str),
    }


def concept_properties(concept: MemoryConcept, embedding: list[float] | None) -> dict[str, Any]:
    """Flat properties for a ``:MemoryConcept`` node."""
    return {
        "concept_id": concept.concept_id,
        "concept": concept.concept,
        "concept_key": concept.concept_key,
        "namespace": concept.namespace,
        "description": concept.description,
        "cardinality": concept.cardinality.value,
        "value_type": concept.value_type,
        "embedding": embedding,
    }


def context_properties(context: ContextInstance, embedding: list[float] | None) -> dict[str, Any]:
    """Flat properties for a ``:Context`` node, including provenance and lineage."""
    return {
        "context_id": context.context_id,
        "fingerprint": context.fingerprint,
        "context_type": context.context_type.value,
        "concept_id": context.concept_id,
        "concept_key": context.concept_key,
        "subject_user_id": context.subject_user_id,
        "scope_type": context.scope_type.value,
        "scope_id": context.scope_id,
        "value": encode_value(context.value),
        "value_key": context.value_key,
        "cardinality": context.cardinality.value,
        "condition": context.condition,
        "visibility": context.visibility.value,
        "authority": context.authority.value,
        "status": context.status.value,
        "observed_at": context.observed_at,
        "recorded_at": context.recorded_at,
        "valid_from": context.valid_from,
        "valid_until": context.valid_until,
        "superseded_at": context.superseded_at,
        "source_episode_id": context.provenance.source_episode_id,
        "source_session_id": context.provenance.source_session_id,
        "source_actor": context.provenance.source_actor,
        "source_type": context.provenance.source_type.value,
        "source_trust": context.provenance.source_trust.value,
        "extraction_confidence": context.provenance.extraction_confidence,
        "run_id": context.lineage.run_id,
        "extractor_model": context.lineage.extractor_model,
        "prompt_version": context.lineage.prompt_version,
        "policy_version": context.lineage.policy_version,
        "embedding_model": context.lineage.embedding_model,
        "embedding_dimension": context.lineage.embedding_dimension,
        "adjudicator_model": context.lineage.adjudicator_model,
        "embedding": embedding,
    }


def evidence_properties(evidence: EvidenceLink) -> dict[str, Any]:
    """Flat properties for a ``SUPPORTED_BY`` relationship."""
    return {
        "session_id": evidence.session_id,
        "actor_id": evidence.actor_id,
        "source_type": evidence.source_type.value,
        "source_trust": evidence.source_trust.value,
        "extraction_confidence": evidence.extraction_confidence,
        "observed_at": evidence.observed_at,
        "recorded_at": evidence.recorded_at,
    }


def episode_from_node(node: dict[str, Any]) -> Episode:
    """Rebuild an :class:`Episode` from stored properties.

    Episodes written before episodes held spans carry no ``spans`` property;
    they are read back as a single span covering the whole message, so stored
    data from earlier runs stays usable.
    """
    stored = node.get("spans")
    if stored:
        spans = tuple(SourceSpan.model_validate(span) for span in json.loads(stored))
    else:
        text = node["text"]
        spans = (
            SourceSpan(
                message_id=node["message_id"],
                sequence=int(node["sequence"]),
                role=Role(node["role"]),
                sender_id=node["sender_id"],
                timestamp=_to_datetime(node["timestamp"]),
                source_type=SourceType(node["source_type"]),
                char_start=0,
                char_end=len(text),
                text=text,
                metadata=json.loads(node.get("metadata") or "{}"),
            ),
        )
    return Episode(
        episode_id=node["episode_id"],
        session_id=node["session_id"],
        sequence=int(node["sequence"]),
        spans=spans,
        continues_episode_id=node.get("continues_episode_id"),
        metadata=json.loads(node.get("metadata") or "{}"),
    )


def concept_from_node(node: dict[str, Any]) -> MemoryConcept:
    """Rebuild a :class:`MemoryConcept` from stored properties."""
    return MemoryConcept(
        concept_id=node["concept_id"],
        concept=node["concept"],
        concept_key=node["concept_key"],
        namespace=node["namespace"],
        description=node.get("description"),
        cardinality=Cardinality(node["cardinality"]),
        value_type=node.get("value_type"),
    )


def context_from_node(node: dict[str, Any]) -> ContextInstance:
    """Rebuild a :class:`ContextInstance` from stored properties."""
    return ContextInstance(
        context_id=node["context_id"],
        fingerprint=node["fingerprint"],
        context_type=ContextType(node["context_type"]),
        concept_id=node["concept_id"],
        concept_key=node["concept_key"],
        subject_user_id=node["subject_user_id"],
        scope_type=ScopeType(node["scope_type"]),
        scope_id=node["scope_id"],
        value=decode_value(node["value"]),
        cardinality=Cardinality(node["cardinality"]),
        condition=node.get("condition"),
        visibility=Visibility(node["visibility"]),
        authority=Authority(node["authority"]),
        status=ContextStatus(node["status"]),
        observed_at=_to_datetime(node["observed_at"]),
        recorded_at=_to_datetime(node["recorded_at"]),
        valid_from=_optional_datetime(node.get("valid_from")),
        valid_until=_optional_datetime(node.get("valid_until")),
        superseded_at=_optional_datetime(node.get("superseded_at")),
        provenance=Provenance(
            source_episode_id=node["source_episode_id"],
            source_session_id=node["source_session_id"],
            source_actor=node["source_actor"],
            source_type=SourceType(node["source_type"]),
            extraction_confidence=node["extraction_confidence"],
        ),
        lineage=Lineage(
            run_id=node["run_id"],
            extractor_model=node["extractor_model"],
            prompt_version=node["prompt_version"],
            policy_version=node["policy_version"],
            embedding_model=node["embedding_model"],
            embedding_dimension=node["embedding_dimension"],
            adjudicator_model=node.get("adjudicator_model"),
        ),
    )


def support_from_record(record: dict[str, Any]) -> ContextSupport:
    """Build aggregated support from a Cypher aggregation row."""
    return ContextSupport(
        context_id=record["context_id"],
        evidence_count=record["evidence_count"],
        distinct_sessions=record["distinct_sessions"],
        distinct_users=record["distinct_users"],
        first_observed_at=_optional_datetime(record.get("first_observed_at")),
        last_observed_at=_optional_datetime(record.get("last_observed_at")),
        has_authoritative_source=record["has_authoritative_source"],
    )


def _to_datetime(value: Any) -> Any:
    """Convert a Neo4j temporal value into a native ``datetime``."""
    to_native = getattr(value, "to_native", None)
    return to_native() if callable(to_native) else value


def _optional_datetime(value: Any) -> Any:
    return None if value is None else _to_datetime(value)
