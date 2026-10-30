"""Neo4j persistence for episodes, concepts, context and evidence.

All writes are idempotent: episodes MERGE on ``(session_id, episode_id)``,
concepts on ``concept_key``, context on its deterministic ``fingerprint`` and
evidence on its source episode. Retries therefore never duplicate a node or
inflate evidence counts.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from datetime import datetime
from types import TracebackType
from typing import Any, LiteralString, Self

from neo4j import AsyncDriver, AsyncGraphDatabase, AsyncManagedTransaction, Record
from neo4j import exceptions as neo4j_exceptions

from context_graph.config import Neo4jSettings
from context_graph.domain.context import (
    SHARED_SUBJECT,
    ContextInstance,
    ContextSupport,
    EvidenceLink,
    MemoryConcept,
)
from context_graph.domain.conversation import Episode, SessionInput
from context_graph.domain.enums import ContextStatus, ScopeType
from context_graph.errors import ConfigurationError, GraphWriteError, TransientError
from context_graph.graph import mapping
from context_graph.graph.schema import (
    CONCEPT_VECTOR_INDEX,
    CONSTRAINTS,
    CONTEXT_VECTOR_INDEX,
    INDEXES,
    vector_index_statements,
)
from context_graph.observability import get_logger, log_event

_logger = get_logger("graph.neo4j")

_PERSIST_SESSION: LiteralString = """
MERGE (u:User {user_id: $user_id})
MERGE (s:Session {session_id: $session_id})
SET s.user_id = $user_id,
    s.state = $state,
    s.session_start_time = $session_start_time,
    s.session_end_time = $session_end_time
MERGE (u)-[:PARTICIPATED_IN]->(s)
WITH s
UNWIND $agent_ids AS agent_id
MERGE (a:Agent {agent_id: agent_id})
MERGE (s)-[:INVOLVES_AGENT]->(a)
"""

_PERSIST_EPISODES: LiteralString = """
UNWIND $episodes AS ep
MATCH (s:Session {session_id: ep.session_id})
MERGE (e:Episode {session_id: ep.session_id, episode_id: ep.episode_id})
ON CREATE SET e += ep, e.ingested_at = datetime()
MERGE (s)-[:CONTAINS]->(e)
FOREACH (speaker IN ep.user_senders |
    MERGE (actor:User {user_id: speaker})
    MERGE (e)-[:SENT_BY]->(actor))
FOREACH (speaker IN ep.agent_senders |
    MERGE (actor:Agent {agent_id: speaker})
    MERGE (e)-[:GENERATED_BY]->(actor))
RETURN count(e) AS persisted
"""

_EPISODES_AFTER: LiteralString = """
MATCH (e:Episode {session_id: $session_id})
WHERE e.sequence > $after_sequence
RETURN e ORDER BY e.sequence ASC LIMIT $limit
"""

_MERGE_CONCEPT: LiteralString = """
MERGE (c:MemoryConcept {concept_key: $props.concept_key})
ON CREATE SET c += $props
ON MATCH SET c.description = coalesce(c.description, $props.description),
             c.embedding = coalesce(c.embedding, $props.embedding)
RETURN c
"""

_CONCEPT_BY_KEY: LiteralString = """
MATCH (c:MemoryConcept {concept_key: $concept_key}) RETURN c
"""

# A cosine vector index reports (1 + cosine) / 2; similarity thresholds are raw
# cosine, so both searches convert the score back.
_SEARCH_CONCEPTS: LiteralString = """
CALL db.index.vector.queryNodes($index, $top_k, $embedding) YIELD node, score
RETURN node, 2 * score - 1 AS score
"""

_SEARCH_CONTEXTS: LiteralString = """
CALL db.index.vector.queryNodes($index, $top_k, $embedding) YIELD node, score
WHERE node.subject_user_id IN $subjects AND node.status IN $statuses
RETURN node, 2 * score - 1 AS score
"""

_ACTIVE_CONTEXTS: LiteralString = """
MATCH (c:Context)
WHERE c.subject_user_id = $subject_user_id
  AND c.scope_type = $scope_type
  AND c.scope_id = $scope_id
  AND c.concept_id = $concept_id
  AND c.status IN $statuses
RETURN c ORDER BY c.observed_at DESC
"""

_MERGE_CONTEXT: LiteralString = """
MERGE (c:Context {fingerprint: $props.fingerprint})
ON CREATE SET c += $props, c.created_at = datetime()
// A repeat of the same scoped value re-activates the instance: the newest
// assertion decides the status, while history survives as SUPERSEDES edges.
ON MATCH SET c.last_seen_at = datetime(),
             c.observed_at = CASE WHEN $props.observed_at > c.observed_at
                                  THEN $props.observed_at ELSE c.observed_at END,
             c.valid_until = coalesce($props.valid_until, c.valid_until),
             c.status = $props.status,
             c.superseded_at = null
WITH c
MATCH (concept:MemoryConcept {concept_id: $props.concept_id})
MERGE (c)-[:INSTANTIATES]->(concept)
RETURN c
"""

_MERGE_EVIDENCE: LiteralString = """
MATCH (c:Context {context_id: $context_id})
MATCH (e:Episode {session_id: $session_id, episode_id: $episode_id})
MERGE (c)-[r:SUPPORTED_BY]->(e)
ON CREATE SET r += $props
SET c.evidence_count = count { (c)-[:SUPPORTED_BY]->(:Episode) }
RETURN c.evidence_count AS evidence_count
"""

_SUPERSEDE: LiteralString = """
MATCH (new:Context {context_id: $new_context_id})
MATCH (old:Context {context_id: $old_context_id})
MERGE (new)-[:SUPERSEDES]->(old)
SET old.status = $superseded_status, old.superseded_at = $superseded_at
"""

_REVOKE: LiteralString = """
MATCH (c:Context {context_id: $context_id})
MATCH (e:Episode {session_id: $session_id, episode_id: $episode_id})
MERGE (e)-[:REVOKES]->(c)
SET c.status = $revoked_status, c.revoked_at = $revoked_at
RETURN c
"""

_SUPPORT: LiteralString = """
UNWIND $context_ids AS context_id
MATCH (c:Context {context_id: context_id})
OPTIONAL MATCH (c)-[r:SUPPORTED_BY]->(:Episode)
RETURN context_id AS context_id,
       count(r) AS evidence_count,
       count(DISTINCT r.session_id) AS distinct_sessions,
       count(DISTINCT CASE WHEN r.source_trust = 'first_party' THEN r.actor_id END)
           AS distinct_users,
       min(r.observed_at) AS first_observed_at,
       max(r.observed_at) AS last_observed_at,
       c.authority = 'authoritative' AS has_authoritative_source
"""

_CANDIDATES: LiteralString = """
MATCH (c:Context {status: $candidate_status})
RETURN c ORDER BY c.recorded_at ASC LIMIT $limit
"""

_SET_STATUS: LiteralString = """
MATCH (c:Context {context_id: $context_id})
SET c.status = $status, c.promoted_at = $now
RETURN c
"""

_VECTOR_INDEX_DIMENSIONS: LiteralString = """
SHOW VECTOR INDEXES YIELD name, options
WHERE name IN $names
RETURN name, options['indexConfig']['vector.dimensions'] AS dimensions
"""

#: Text to re-embed for every Context, rebuilt from what the node already holds.
_CONTEXT_VECTOR_BASIS: LiteralString = """
MATCH (c:Context)
RETURN c.context_id AS id, c.concept_key AS concept_key,
       c.context_type AS context_type, c.value AS value
ORDER BY id
"""

#: A concept is re-embedded from one of its instances, matching how it was made.
_CONCEPT_VECTOR_BASIS: LiteralString = """
MATCH (m:MemoryConcept)
OPTIONAL MATCH (c:Context)-[:INSTANTIATES]->(m)
WITH m, c ORDER BY c.recorded_at
WITH m, head(collect(c)) AS sample
RETURN m.concept_id AS id, m.concept AS concept,
       sample.context_type AS context_type, sample.value AS value
ORDER BY id
"""

_SET_CONTEXT_EMBEDDING: LiteralString = """
UNWIND $rows AS row
MATCH (c:Context {context_id: row.id})
SET c.embedding = row.embedding
"""

_SET_CONCEPT_EMBEDDING: LiteralString = """
UNWIND $rows AS row
MATCH (m:MemoryConcept {concept_id: row.id})
SET m.embedding = row.embedding
"""

_COUNT_EPISODES: LiteralString = """
MATCH (e:Episode {session_id: $session_id}) RETURN count(e) AS episodes
"""

#: ``APPLIES_TO`` targets one ontology node per scope; labels cannot be parameters.
_APPLIES_TO: dict[ScopeType, LiteralString] = {
    ScopeType.SESSION: """
        MATCH (c:Context {context_id: $context_id})
        MERGE (n:Session {session_id: $scope_id})
        MERGE (c)-[:APPLIES_TO]->(n)""",
    ScopeType.USER: """
        MATCH (c:Context {context_id: $context_id})
        MERGE (n:User {user_id: $scope_id})
        MERGE (c)-[:APPLIES_TO]->(n)""",
    ScopeType.TASK: """
        MATCH (c:Context {context_id: $context_id})
        MERGE (n:Task {task_id: $scope_id})
        MERGE (c)-[:APPLIES_TO]->(n)""",
    ScopeType.PROJECT: """
        MATCH (c:Context {context_id: $context_id})
        MERGE (n:Project {project_id: $scope_id})
        MERGE (c)-[:APPLIES_TO]->(n)""",
    ScopeType.TEAM: """
        MATCH (c:Context {context_id: $context_id})
        MERGE (n:Team {team_id: $scope_id})
        MERGE (c)-[:APPLIES_TO]->(n)""",
    ScopeType.DEPARTMENT: """
        MATCH (c:Context {context_id: $context_id})
        MERGE (n:Department {department_id: $scope_id})
        MERGE (c)-[:APPLIES_TO]->(n)""",
    ScopeType.ORGANIZATION: """
        MATCH (c:Context {context_id: $context_id})
        MERGE (n:Organization {organization_id: $scope_id})
        MERGE (c)-[:APPLIES_TO]->(n)""",
}

_RESOLVABLE_STATUSES = [ContextStatus.ACTIVE.value, ContextStatus.CANDIDATE.value]


class Neo4jGraphStore:
    """Durable graph store for Episodes and derived Context."""

    def __init__(self, settings: Neo4jSettings, driver: AsyncDriver | None = None) -> None:
        self._settings = settings
        self._driver = driver or AsyncGraphDatabase.driver(
            settings.uri, auth=(settings.user, settings.password.get_secret_value())
        )

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.close()

    async def close(self) -> None:
        """Close the driver's connection pool."""
        await self._driver.close()

    async def ensure_schema(self, embedding_dimension: int) -> None:
        """Create constraints and indexes; safe to run on every startup."""
        await self._check_vector_dimensions(embedding_dimension)
        for statement in (*CONSTRAINTS, *INDEXES, *vector_index_statements(embedding_dimension)):
            await self._run(statement, {})
        log_event(_logger, "graph.schema_ready", dimension=embedding_dimension)

    async def _check_vector_dimensions(self, embedding_dimension: int) -> None:
        """Refuse to run against vector indexes built for another model.

        ``CREATE VECTOR INDEX ... IF NOT EXISTS`` silently keeps an existing
        index, so a changed embedding model would leave stored vectors and new
        ones in different spaces - searches would return nonsense rather than
        fail. Say so instead.
        """
        records = await self._run(
            _VECTOR_INDEX_DIMENSIONS,
            {"names": [CONCEPT_VECTOR_INDEX, CONTEXT_VECTOR_INDEX]},
        )
        mismatched = {
            str(record["name"]): int(record["dimensions"])
            for record in records
            if record["dimensions"] is not None
            and int(record["dimensions"]) != embedding_dimension
        }
        if mismatched:
            details = ", ".join(f"{name} is {dims}" for name, dims in mismatched.items())
            raise ConfigurationError(
                f"vector indexes were built for a different embedding size "
                f"({details}) but the configured dimension is {embedding_dimension}. "
                f"Drop them and re-embed, or point CG_NEO4J__DATABASE at a fresh "
                f"graph: DROP INDEX {' ; DROP INDEX '.join(mismatched)}"
            )

    async def persist_session(self, session: SessionInput) -> None:
        """Upsert the Session, its User and its Agents."""
        await self._run(
            _PERSIST_SESSION,
            {
                "session_id": session.session_id,
                "user_id": session.user_id,
                "state": session.state.value,
                "session_start_time": session.session_start_time,
                "session_end_time": session.session_end_time,
                "agent_ids": list(session.agent_ids),
            },
        )

    async def persist_episodes(self, episodes: Sequence[Episode]) -> int:
        """Upsert Episodes immutably; repeated identities are not duplicated."""
        if not episodes:
            return 0
        records = await self._run(
            _PERSIST_EPISODES,
            {"episodes": [mapping.episode_properties(episode) for episode in episodes]},
        )
        persisted = int(records[0]["persisted"]) if records else 0
        if persisted != len(episodes):
            # The query attaches Episodes to their Session; a missing Session
            # would drop rows, so refuse to report a partial write as success.
            raise GraphWriteError(
                f"persisted {persisted} of {len(episodes)} episodes for "
                f"session {episodes[0].session_id}"
            )
        log_event(
            _logger,
            "graph.episodes_persisted",
            session_id=episodes[0].session_id,
            persisted=persisted,
        )
        return persisted

    async def count_episodes(self, session_id: str) -> int:
        """Number of Episodes stored for a session."""
        records = await self._run(_COUNT_EPISODES, {"session_id": session_id})
        return int(records[0]["episodes"]) if records else 0

    async def episodes_after(self, session_id: str, after_sequence: int, limit: int) -> list[Episode]:
        """Episodes with a sequence greater than the checkpoint, in order."""
        records = await self._run(
            _EPISODES_AFTER,
            {"session_id": session_id, "after_sequence": after_sequence, "limit": limit},
        )
        return [mapping.episode_from_node(dict(record["e"])) for record in records]

    async def merge_concept(
        self, concept: MemoryConcept, embedding: list[float] | None
    ) -> MemoryConcept:
        """Create or reuse a canonical concept, keyed by ``concept_key``."""
        records = await self._run(
            _MERGE_CONCEPT, {"props": mapping.concept_properties(concept, embedding)}
        )
        return mapping.concept_from_node(dict(records[0]["c"]))

    async def concept_by_key(self, concept_key: str) -> MemoryConcept | None:
        """Deterministic concept lookup (exact-duplicate protection)."""
        records = await self._run(_CONCEPT_BY_KEY, {"concept_key": concept_key})
        return mapping.concept_from_node(dict(records[0]["c"])) if records else None

    async def search_concepts(
        self, embedding: Sequence[float], top_k: int
    ) -> list[tuple[MemoryConcept, float]]:
        """Semantic concept candidates ordered by cosine similarity."""
        records = await self._run(
            _SEARCH_CONCEPTS,
            {"index": CONCEPT_VECTOR_INDEX, "top_k": top_k, "embedding": list(embedding)},
        )
        return [
            (mapping.concept_from_node(dict(record["node"])), float(record["score"]))
            for record in records
        ]

    async def search_contexts(
        self, *, embedding: Sequence[float], top_k: int, subject_user_id: str
    ) -> list[tuple[ContextInstance, float]]:
        """Context a user can already see, ranked by semantic similarity.

        Used for the targeted lookup of long-range references (A.6) instead of
        replaying an entire session to the model.
        """
        records = await self._run(
            _SEARCH_CONTEXTS,
            {
                "index": CONTEXT_VECTOR_INDEX,
                "top_k": top_k,
                "embedding": list(embedding),
                "subjects": [subject_user_id, SHARED_SUBJECT],
                "statuses": _RESOLVABLE_STATUSES,
            },
        )
        return [
            (mapping.context_from_node(dict(record["node"])), float(record["score"]))
            for record in records
        ]

    async def active_contexts(
        self, *, subject_user_id: str, scope_type: ScopeType, scope_id: str, concept_id: str
    ) -> list[ContextInstance]:
        """Live instances of one concept in one scope, newest first."""
        records = await self._run(
            _ACTIVE_CONTEXTS,
            {
                "subject_user_id": subject_user_id,
                "scope_type": scope_type.value,
                "scope_id": scope_id,
                "concept_id": concept_id,
                "statuses": _RESOLVABLE_STATUSES,
            },
        )
        return [mapping.context_from_node(dict(record["c"])) for record in records]

    async def write_context(
        self,
        *,
        context: ContextInstance,
        embedding: list[float] | None,
        evidence: EvidenceLink,
        supersedes_context_ids: Sequence[str] = (),
    ) -> ContextInstance:
        """Write a context instance, its scope edge and its evidence atomically."""
        scope_query = _APPLIES_TO[context.scope_type]
        props = mapping.context_properties(context, embedding)
        evidence_props = mapping.evidence_properties(evidence)

        async def unit_of_work(tx: AsyncManagedTransaction) -> dict[str, Any]:
            result = await tx.run(_MERGE_CONTEXT, props=props)
            record = await result.single()
            if record is None:
                raise GraphWriteError(f"context merge returned no node: {context.fingerprint}")
            stored = dict(record["c"])
            context_id = str(stored["context_id"])

            await tx.run(scope_query, context_id=context_id, scope_id=context.scope_id)
            await tx.run(
                _MERGE_EVIDENCE,
                context_id=context_id,
                session_id=evidence.session_id,
                episode_id=evidence.episode_id,
                props=evidence_props,
            )
            for old_context_id in supersedes_context_ids:
                if old_context_id == context_id:
                    continue
                await tx.run(
                    _SUPERSEDE,
                    new_context_id=context_id,
                    old_context_id=old_context_id,
                    superseded_status=ContextStatus.SUPERSEDED.value,
                    superseded_at=context.recorded_at,
                )
            return stored

        stored = await self._write_transaction(unit_of_work, "write_context")
        log_event(
            _logger,
            "graph.context_written",
            context_id=stored["context_id"],
            concept_key=context.concept_key,
            scope_type=context.scope_type.value,
            status=stored["status"],
            supersedes=list(supersedes_context_ids),
        )
        return mapping.context_from_node(stored)

    async def add_evidence(self, evidence: EvidenceLink) -> int:
        """Link one more supporting Episode; repeats do not increase support."""
        records = await self._run(
            _MERGE_EVIDENCE,
            {
                "context_id": evidence.context_id,
                "session_id": evidence.session_id,
                "episode_id": evidence.episode_id,
                "props": mapping.evidence_properties(evidence),
            },
        )
        return int(records[0]["evidence_count"]) if records else 0

    async def revoke_context(
        self, *, context_id: str, session_id: str, episode_id: str, revoked_at: datetime
    ) -> ContextInstance | None:
        """Record an explicit withdrawal; outranks ordinary recency."""
        records = await self._run(
            _REVOKE,
            {
                "context_id": context_id,
                "session_id": session_id,
                "episode_id": episode_id,
                "revoked_status": ContextStatus.REVOKED.value,
                "revoked_at": revoked_at,
            },
        )
        if not records:
            return None
        log_event(_logger, "graph.context_revoked", context_id=context_id, episode_id=episode_id)
        return mapping.context_from_node(dict(records[0]["c"]))

    async def support_for(self, context_ids: Sequence[str]) -> dict[str, ContextSupport]:
        """De-duplicated evidence aggregates used by the promotion policy."""
        if not context_ids:
            return {}
        records = await self._run(_SUPPORT, {"context_ids": list(context_ids)})
        supports = [mapping.support_from_record(dict(record)) for record in records]
        return {support.context_id: support for support in supports}

    async def candidates(self, limit: int) -> list[ContextInstance]:
        """Shared-scope candidates awaiting a governed promotion decision."""
        records = await self._run(
            _CANDIDATES, {"candidate_status": ContextStatus.CANDIDATE.value, "limit": limit}
        )
        return [mapping.context_from_node(dict(record["c"])) for record in records]

    async def set_status(
        self, context_id: str, status: ContextStatus, now: datetime
    ) -> ContextInstance | None:
        """Apply a governed status change (used by promotion)."""
        records = await self._run(
            _SET_STATUS, {"context_id": context_id, "status": status.value, "now": now}
        )
        return mapping.context_from_node(dict(records[0]["c"])) if records else None

    async def vector_basis(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Rows describing what every Context and MemoryConcept should embed."""
        contexts = [dict(record) for record in await self._run(_CONTEXT_VECTOR_BASIS, {})]
        concepts = [dict(record) for record in await self._run(_CONCEPT_VECTOR_BASIS, {})]
        return contexts, concepts

    async def set_embeddings(
        self,
        *,
        contexts: Sequence[dict[str, Any]] = (),
        concepts: Sequence[dict[str, Any]] = (),
    ) -> None:
        """Write recomputed vectors back onto existing nodes."""
        if contexts:
            await self._run(_SET_CONTEXT_EMBEDDING, {"rows": list(contexts)})
        if concepts:
            await self._run(_SET_CONCEPT_EMBEDDING, {"rows": list(concepts)})

    async def drop_vector_indexes(self) -> None:
        """Remove the vector indexes so they can be rebuilt at a new dimension."""
        for name in (CONCEPT_VECTOR_INDEX, CONTEXT_VECTOR_INDEX):
            # An index name cannot be a parameter; these two are our own constants.
            await self._run(f"DROP INDEX {name} IF EXISTS", {})
        log_event(_logger, "graph.vector_indexes_dropped")

    async def _run(self, query: LiteralString, parameters: dict[str, Any]) -> list[Record]:
        """Execute one query, translating driver failures into pipeline errors."""
        try:
            result = await self._driver.execute_query(
                query, parameters, database_=self._settings.database
            )
        except neo4j_exceptions.TransientError as error:
            raise TransientError(f"neo4j transient failure: {error}") from error
        except neo4j_exceptions.ServiceUnavailable as error:
            raise TransientError(f"neo4j unavailable: {error}") from error
        except neo4j_exceptions.Neo4jError as error:
            raise GraphWriteError(f"neo4j query failed: {error}") from error
        return list(result.records)

    async def _write_transaction[T](
        self,
        unit_of_work: Callable[[AsyncManagedTransaction], Awaitable[T]],
        description: str,
    ) -> T:
        """Run a multi-statement write in a single transaction."""
        try:
            async with self._driver.session(database=self._settings.database) as session:
                return await session.execute_write(unit_of_work)
        except neo4j_exceptions.TransientError as error:
            raise TransientError(f"{description} failed transiently: {error}") from error
        except neo4j_exceptions.ServiceUnavailable as error:
            raise TransientError(f"neo4j unavailable during {description}: {error}") from error
        except neo4j_exceptions.Neo4jError as error:
            raise GraphWriteError(f"{description} failed: {error}") from error
