"""Read-only Context retrieval for the retrieval tester.

This module only ever reads. Queries run with READ routing, the client exposes
no write operation, and nothing here touches the ingestion pipeline or the
processing ledger.

Selection follows the metadata ingestion stored: only ``active`` Context inside
its validity window, the user's own private Context, and governed shared
Context the user is a member of. Session-scoped Context belongs to the session
it was stated in and is never carried into a new one (spec A.9).
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from time import perf_counter
from types import TracebackType
from typing import Any, LiteralString, Protocol, Self, runtime_checkable

from neo4j import AsyncDriver, AsyncGraphDatabase, Record, RoutingControl
from neo4j import exceptions as neo4j_exceptions

from context_graph.config import Neo4jSettings
from context_graph.domain.context import ContextInstance
from context_graph.domain.enums import ContextStatus, ScopeType
from context_graph.errors import ContextGraphError
from context_graph.graph import mapping
from context_graph.providers.base import EmbeddingProvider
from retrieval_tester.config import AgentSettings
from retrieval_tester.errors import RetrievalError
from retrieval_tester.models import (
    Colleague,
    Directory,
    RetrievalDebug,
    RetrievalResult,
    RetrievedContext,
    SessionRecall,
    ValueChange,
    utcnow,
)

#: Scopes whose Context is private to one user. SESSION is deliberately absent:
#: SESSION-scoped Context belongs to the session it was stated in and is never
#: carried into a new one (spec A.9).
_PRIVATE_SCOPES = [ScopeType.USER.value, ScopeType.TASK.value, ScopeType.PROJECT.value]
#: Governed shared scopes, readable only through a MEMBER_OF edge.
_SHARED_SCOPES = [
    ScopeType.TEAM.value,
    ScopeType.DEPARTMENT.value,
    ScopeType.ORGANIZATION.value,
]

_SESSIONS: LiteralString = """
MATCH (:User {user_id: $user_id})-[:PARTICIPATED_IN]->(s:Session)-[:CONTAINS]->(e:Episode)
WITH s, collect(e) AS episodes
OPTIONAL MATCH (c:Context {source_session_id: s.session_id})
WITH s, episodes, collect(DISTINCT c.concept_key) AS concepts
WITH s, episodes, concepts,
     [ep IN episodes WHERE ep.role = 'user'] AS user_episodes
WITH s, concepts, size(episodes) AS episode_count,
     reduce(earliest = null, ep IN episodes |
            CASE WHEN earliest IS NULL OR ep.timestamp < earliest
                 THEN ep.timestamp ELSE earliest END) AS started,
     reduce(opener = null, ep IN user_episodes |
            CASE WHEN opener IS NULL OR ep.sequence < opener.sequence
                 THEN ep ELSE opener END) AS opener
RETURN s.session_id AS session_id, s.state AS state, started,
       s.session_end_time AS ended, episode_count,
       left(opener.text, $quote_chars) AS opening_message, concepts
ORDER BY started DESC LIMIT $session_limit
"""

_DIRECTORY: LiteralString = """
MATCH (me:User {user_id: $user_id})-[:MEMBER_OF]->(team:Team)
OPTIONAL MATCH (team)-[:BELONGS_TO]->(department:Department)
OPTIONAL MATCH (department)-[:BELONGS_TO]->(organization:Organization)
OPTIONAL MATCH (team)<-[:PART_OF]-(team_project:Project)
OPTIONAL MATCH (me)-[:WORKS_ON]->(my_project:Project)
OPTIONAL MATCH (team)<-[:MEMBER_OF]-(colleague:User)
WHERE colleague.user_id <> $user_id
OPTIONAL MATCH (colleague)-[:WORKS_ON]->(colleague_project:Project)
WITH team, department, organization,
     collect(DISTINCT team_project.name) AS team_projects,
     collect(DISTINCT my_project.name) AS my_projects,
     colleague, collect(DISTINCT colleague_project.name) AS colleague_projects
WITH team, department, organization, team_projects, my_projects,
     collect(CASE WHEN colleague IS NULL THEN NULL ELSE {
        user_id: colleague.user_id, name: colleague.name,
        title: colleague.title, projects: colleague_projects} END) AS colleagues
RETURN team.team_id AS team_id, team.name AS team_name,
       department.name AS department_name, organization.name AS organization_name,
       team_projects, my_projects,
       [c IN colleagues WHERE c IS NOT NULL][..$colleague_limit] AS colleagues
"""

_VALUE_CHANGES: LiteralString = """
MATCH (new:Context)-[:SUPERSEDES]->(old:Context)
WHERE old.subject_user_id = $user_id
RETURN old.concept_key AS concept_key, old.value AS previous_value,
       new.value AS current_value, old.superseded_at AS changed_at
ORDER BY old.superseded_at DESC LIMIT $change_limit
"""

_RETRIEVE: LiteralString = """
MATCH (c:Context)
WHERE c.status = $active_status
  AND (c.valid_from IS NULL OR c.valid_from <= $now)
  AND (c.valid_until IS NULL OR c.valid_until >= $now)
  AND (
        (c.subject_user_id = $user_id AND c.scope_type IN $private_scopes)
        OR ($include_shared AND c.scope_type IN $shared_scopes AND EXISTS {
              MATCH (:User {user_id: $user_id})-[:MEMBER_OF]->(:Team)
                    -[:BELONGS_TO*0..3]->(scope)
              WHERE (scope:Team AND scope.team_id = c.scope_id)
                 OR (scope:Department AND scope.department_id = c.scope_id)
                 OR (scope:Organization AND scope.organization_id = c.scope_id)
           })
      )
OPTIONAL MATCH (c)-[:SUPPORTED_BY]->(e:Episode)
WITH c, reduce(first = null, ep IN collect(e) |
        CASE WHEN first IS NULL OR ep.timestamp < first.timestamp THEN ep ELSE first END) AS evidence
WITH c, left(evidence.text, $quote_chars) AS evidence_quote
ORDER BY c.evidence_count DESC, c.observed_at DESC
WITH collect({node: c, quote: evidence_quote, embedding: c.embedding}) AS matches
RETURN size(matches) AS considered, matches AS rows
"""


@runtime_checkable
class ContextRetriever(Protocol):
    """Source of personalization Context for a user."""

    async def retrieve(self, user_id: str, query: str | None = None) -> RetrievalResult:
        """Return what applies to ``user_id``, ranked against ``query`` when given."""
        ...


class Neo4jContextRetriever:
    """Reads applicable Context, past conversations and value history.

    All four reads are independent, so they run concurrently. Relevance ranking
    is applied only when a current message and an embedding provider are both
    available; standing instructions are always included regardless of topic.
    """

    def __init__(
        self,
        settings: Neo4jSettings,
        *,
        agent_settings: AgentSettings | None = None,
        embeddings: EmbeddingProvider | None = None,
        driver: AsyncDriver | None = None,
        max_items: int | None = None,
        include_shared_scopes: bool | None = None,
    ) -> None:
        base = agent_settings or AgentSettings()
        self._settings = settings
        self._agent = base.model_copy(
            update={
                key: value
                for key, value in (
                    ("max_context_items", max_items),
                    ("include_shared_scopes", include_shared_scopes),
                )
                if value is not None
            }
        )
        self._embeddings = embeddings
        self._driver = driver or AsyncGraphDatabase.driver(
            settings.uri,
            auth=(settings.user, settings.password.get_secret_value()),
            # Keep the chat UI clean: unused-property hints are expected here.
            notifications_min_severity="OFF",
        )

    @property
    def driver(self) -> AsyncDriver:
        """The graph connection, shared with the read-only lookup tool."""
        return self._driver

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Close the graph connection."""
        await self._driver.close()

    async def retrieve(self, user_id: str, query: str | None = None) -> RetrievalResult:
        """Select what applies to ``user_id`` and explain the choice."""
        now = utcnow()
        started = perf_counter()
        agent = self._agent

        context_rows, session_rows, change_rows, directory_rows, query_vector = await asyncio.gather(
            self._read(
                _RETRIEVE,
                {
                    "user_id": user_id,
                    "now": now,
                    "active_status": ContextStatus.ACTIVE.value,
                    "private_scopes": _PRIVATE_SCOPES,
                    "shared_scopes": _SHARED_SCOPES,
                    "include_shared": agent.include_shared_scopes,
                    "quote_chars": agent.evidence_quote_chars,
                },
            ),
            self._read(
                _SESSIONS,
                {
                    "user_id": user_id,
                    "session_limit": agent.max_sessions_recalled,
                    "quote_chars": agent.evidence_quote_chars,
                },
            ),
            self._read(
                _VALUE_CHANGES,
                {"user_id": user_id, "change_limit": agent.max_value_changes},
            ),
            self._read(
                _DIRECTORY,
                {"user_id": user_id, "colleague_limit": agent.max_colleagues},
            ),
            self._embed_query(query),
        )

        record = context_rows[0] if context_rows else None
        considered = int(record["considered"]) if record else 0
        rows = list(record["rows"]) if record else []
        candidates = tuple(
            self._to_item(dict(row["node"]), row.get("quote"), row.get("embedding"), query_vector)
            for row in rows
        )
        items = self._select(candidates, ranked=query_vector is not None)

        return RetrievalResult(
            items=items,
            directory=_to_directory(directory_rows[0]) if directory_rows else Directory(),
            sessions=tuple(_to_session(row) for row in session_rows),
            history=tuple(_to_change(row) for row in change_rows),
            debug=RetrievalDebug(
                user_id=user_id,
                duration_ms=round((perf_counter() - started) * 1000, 1),
                candidates_considered=considered,
                returned=len(items),
                truncated=considered > len(items),
                by_scope=_count_by_scope(items),
                query=query,
                relevance_used=query_vector is not None,
                sessions_recalled=len(session_rows),
                changes_recalled=len(change_rows),
                directory_found=bool(directory_rows),
                notes=self._notes(user_id, items, now, query_vector is not None, query),
            ),
        )

    def _select(
        self, candidates: tuple[RetrievedContext, ...], *, ranked: bool
    ) -> tuple[RetrievedContext, ...]:
        """Order by relevance; drop items only when there are too many to send.

        Standing instructions (preferences, constraints, working style) always
        stay: they apply whatever the user just asked about.
        """
        limit = self._agent.max_context_items
        if not ranked:
            return candidates[:limit]

        standing = [item for item in candidates if item.is_standing_instruction]
        topical = sorted(
            (item for item in candidates if not item.is_standing_instruction),
            key=lambda item: item.similarity or 0.0,
            reverse=True,
        )
        if len(candidates) > limit:
            # Only now does relevance have to exclude anything.
            topical = [
                item for item in topical if (item.similarity or 0.0) >= self._agent.min_similarity
            ]
        return tuple((standing + topical)[:limit])

    async def _embed_query(self, query: str | None) -> list[float] | None:
        """Embed the current message, when relevance ranking is possible."""
        if not query or self._embeddings is None:
            return None
        try:
            return (await self._embeddings.embed([query]))[0]
        except ContextGraphError as error:
            raise RetrievalError(f"could not embed the message: {error}") from error

    async def _read(self, query: LiteralString, parameters: dict[str, Any]) -> list[Record]:
        """Run one read-only query."""
        try:
            result = await self._driver.execute_query(
                query,
                parameters,
                database_=self._settings.database,
                # Read routing: this client can never write to the graph.
                routing_=RoutingControl.READ,
            )
        except neo4j_exceptions.Neo4jError as error:
            raise RetrievalError(f"context retrieval failed: {error}") from error
        except OSError as error:
            raise RetrievalError(f"could not reach the context graph: {error}") from error
        return list(result.records)

    def _notes(
        self,
        user_id: str,
        items: tuple[RetrievedContext, ...],
        now: datetime,
        ranked: bool,
        query: str | None,
    ) -> tuple[str, ...]:
        """Human-readable reasons for what was included or left out."""
        notes = [
            f"status=active only; validity window checked against {now.isoformat(timespec='seconds')}",
            "SESSION-scoped context is excluded: it belongs to the session it was stated in",
        ]
        if ranked:
            notes.append(
                "ranked against the current message; standing instructions "
                "(Preference, Constraint, WorkingStyle) are always included"
            )
        elif query:
            notes.append("no embedding provider configured: ordered by evidence, not relevance")
        else:
            notes.append("no message yet: ordered by evidence and recency")
        if not self._agent.include_shared_scopes:
            notes.append("shared scopes disabled by configuration")
        elif not any(item.scope_type.is_shared for item in items):
            notes.append(
                "no shared-scope context: it requires a MEMBER_OF edge, which ingestion does not write"
            )
        if not items:
            notes.append(f"no active context stored for user '{user_id}'")
        return tuple(notes)

    @staticmethod
    def _to_item(
        node: dict[str, Any],
        quote: Any,
        embedding: Any,
        query_vector: list[float] | None,
    ) -> RetrievedContext:
        """Convert a stored Context node into the agent's view of it."""
        instance: ContextInstance = mapping.context_from_node(node)
        return RetrievedContext(
            context_id=instance.context_id,
            concept_key=instance.concept_key,
            context_type=instance.context_type,
            value=str(instance.value),
            scope_type=instance.scope_type,
            scope_id=instance.scope_id,
            condition=instance.condition,
            evidence_count=int(node.get("evidence_count") or 0),
            observed_at=instance.observed_at,
            valid_until=instance.valid_until,
            evidence_quote=str(quote) if quote else None,
            similarity=_cosine(query_vector, embedding),
        )


def _to_directory(record: Record) -> Directory:
    """Build the user's organizational placement from a directory row."""
    colleagues = tuple(
        Colleague(
            user_id=row["user_id"],
            name=row["name"] or row["user_id"],
            title=row.get("title"),
            projects=tuple(name for name in (row.get("projects") or []) if name),
        )
        for row in (record["colleagues"] or [])
    )
    return Directory(
        team_id=record["team_id"],
        team_name=record["team_name"],
        department_name=record["department_name"],
        organization_name=record["organization_name"],
        colleagues=tuple(sorted(colleagues, key=lambda item: item.name)),
        team_projects=tuple(name for name in (record["team_projects"] or []) if name),
        my_projects=tuple(name for name in (record["my_projects"] or []) if name),
    )


def _to_session(record: Record) -> SessionRecall:
    """Build episodic recall from a session row."""
    return SessionRecall(
        session_id=record["session_id"],
        state=record["state"] or "unknown",
        started_at=_native(record["started"]),
        ended_at=_native(record["ended"]),
        episode_count=int(record["episode_count"] or 0),
        opening_message=record["opening_message"],
        concepts=tuple(
            key.split(":")[-1] for key in (record["concepts"] or []) if key is not None
        ),
    )


def _to_change(record: Record) -> ValueChange:
    """Build a value-change entry from a supersession row."""
    return ValueChange(
        concept_key=record["concept_key"],
        previous_value=str(record["previous_value"]),
        current_value=str(record["current_value"]) if record["current_value"] else None,
        changed_at=_native(record["changed_at"]),
    )


def _native(value: Any) -> Any:
    """Convert a Neo4j temporal value into a native datetime."""
    to_native = getattr(value, "to_native", None)
    return to_native() if callable(to_native) else value


def _cosine(left: list[float] | None, right: Any) -> float | None:
    """Cosine similarity of two stored unit-length embeddings."""
    if left is None or not isinstance(right, list) or len(left) != len(right):
        return None
    dot = sum(a * float(b) for a, b in zip(left, right, strict=True))
    return max(-1.0, min(1.0, dot))


def _count_by_scope(items: tuple[RetrievedContext, ...]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        counts[item.scope_type.value] = counts.get(item.scope_type.value, 0) + 1
    return counts
