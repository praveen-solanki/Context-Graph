"""Normative graph schema: labels, relationship types, constraints and indexes.

This is the single normative ontology for the POC (spec sections 8-9). The
relationship types are spelled out in the Cypher statements that create them
(see ``neo4j_store``), so there is exactly one definition of each edge. Scope is
stored as ``scope_type``/``scope_id`` properties on Context; ``APPLIES_TO`` is
the traversal edge. No per-scope relationship types exist.
"""

from __future__ import annotations

from enum import StrEnum
from typing import LiteralString, cast


class Label(StrEnum):
    """Node labels."""

    USER = "User"
    AGENT = "Agent"
    SESSION = "Session"
    EPISODE = "Episode"
    MEMORY_CONCEPT = "MemoryConcept"
    CONTEXT = "Context"
    TASK = "Task"
    PROJECT = "Project"
    TEAM = "Team"
    DEPARTMENT = "Department"
    ORGANIZATION = "Organization"


CONCEPT_VECTOR_INDEX = "concept_embedding_index"
CONTEXT_VECTOR_INDEX = "context_embedding_index"

#: Uniqueness constraints. Episode identity is the (session_id, episode_id) pair.
CONSTRAINTS: tuple[LiteralString, ...] = (
    "CREATE CONSTRAINT user_id_unique IF NOT EXISTS "
    "FOR (n:User) REQUIRE n.user_id IS UNIQUE",
    "CREATE CONSTRAINT agent_id_unique IF NOT EXISTS "
    "FOR (n:Agent) REQUIRE n.agent_id IS UNIQUE",
    "CREATE CONSTRAINT session_id_unique IF NOT EXISTS "
    "FOR (n:Session) REQUIRE n.session_id IS UNIQUE",
    "CREATE CONSTRAINT episode_identity_unique IF NOT EXISTS "
    "FOR (n:Episode) REQUIRE (n.session_id, n.episode_id) IS UNIQUE",
    "CREATE CONSTRAINT concept_id_unique IF NOT EXISTS "
    "FOR (n:MemoryConcept) REQUIRE n.concept_id IS UNIQUE",
    "CREATE CONSTRAINT concept_key_unique IF NOT EXISTS "
    "FOR (n:MemoryConcept) REQUIRE n.concept_key IS UNIQUE",
    "CREATE CONSTRAINT context_id_unique IF NOT EXISTS "
    "FOR (n:Context) REQUIRE n.context_id IS UNIQUE",
    "CREATE CONSTRAINT context_fingerprint_unique IF NOT EXISTS "
    "FOR (n:Context) REQUIRE n.fingerprint IS UNIQUE",
    "CREATE CONSTRAINT task_id_unique IF NOT EXISTS "
    "FOR (n:Task) REQUIRE n.task_id IS UNIQUE",
    "CREATE CONSTRAINT project_id_unique IF NOT EXISTS "
    "FOR (n:Project) REQUIRE n.project_id IS UNIQUE",
    "CREATE CONSTRAINT team_id_unique IF NOT EXISTS "
    "FOR (n:Team) REQUIRE n.team_id IS UNIQUE",
    "CREATE CONSTRAINT department_id_unique IF NOT EXISTS "
    "FOR (n:Department) REQUIRE n.department_id IS UNIQUE",
    "CREATE CONSTRAINT organization_id_unique IF NOT EXISTS "
    "FOR (n:Organization) REQUIRE n.organization_id IS UNIQUE",
)

#: Lookup indexes for the incremental-detection and resolution read paths.
INDEXES: tuple[LiteralString, ...] = (
    "CREATE INDEX episode_session_sequence IF NOT EXISTS "
    "FOR (n:Episode) ON (n.session_id, n.sequence)",
    "CREATE INDEX context_scope_lookup IF NOT EXISTS "
    "FOR (n:Context) ON (n.subject_user_id, n.scope_type, n.scope_id, n.concept_id)",
    "CREATE INDEX context_status IF NOT EXISTS FOR (n:Context) ON (n.status)",
)

_VECTOR_INDEX_TEMPLATE = (
    "CREATE VECTOR INDEX {name} IF NOT EXISTS FOR (n:{label}) ON (n.embedding) "
    "OPTIONS {{indexConfig: {{"
    "`vector.dimensions`: {dimension}, `vector.similarity_function`: 'cosine'}}}}"
)


def vector_index_statements(dimension: int) -> tuple[LiteralString, ...]:
    """Vector index DDL for concept and context embeddings."""
    # The dimension comes from configuration, so the statement cannot be a
    # literal; it is built only from our own template and an int.
    return tuple(
        cast(
            LiteralString,
            _VECTOR_INDEX_TEMPLATE.format(name=name, label=label.value, dimension=int(dimension)),
        )
        for name, label in (
            (CONCEPT_VECTOR_INDEX, Label.MEMORY_CONCEPT),
            (CONTEXT_VECTOR_INDEX, Label.CONTEXT),
        )
    )
