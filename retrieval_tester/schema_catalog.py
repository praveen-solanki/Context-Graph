"""Live discovery of the graph schema the planner is allowed to use.

Nothing here assumes an ontology: labels, relationship types, their endpoints
and every property name are read from the database itself with the built-in
``db.schema.*`` procedures, so the planner can only ever propose structure that
actually exists. The result is cached for a configurable time because the
schema changes far more slowly than queries arrive.

Read-only: both procedures run with READ routing under an explicit timeout.
"""

from __future__ import annotations

import asyncio
import re
from time import monotonic
from typing import Any, LiteralString

from neo4j import AsyncDriver, RoutingControl
from neo4j import exceptions as neo4j_exceptions
from pydantic import BaseModel, ConfigDict

from context_graph.config import Neo4jSettings
from retrieval_tester.errors import RetrievalError

#: Identifiers are interpolated into Cypher, so they must look like identifiers
#: *and* exist in the discovered schema. Both checks are enforced, always.
IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

#: Properties never handed to the model or returned: large, and not readable.
_HIDDEN_PROPERTIES = frozenset({"embedding"})

_NODE_PROPERTIES: LiteralString = """
CALL db.schema.nodeTypeProperties() YIELD nodeLabels, propertyName
RETURN nodeLabels AS labels, propertyName AS property
"""

_VISUALIZATION: LiteralString = "CALL db.schema.visualization()"


class Edge(BaseModel):
    """One relationship type as it is actually used between two labels."""

    model_config = ConfigDict(frozen=True)

    start: str
    type: str
    end: str

    def render(self) -> str:
        return f"({self.start})-[:{self.type}]->({self.end})"


class GraphSchema(BaseModel):
    """The labels, properties and connections discovered in the database."""

    model_config = ConfigDict(frozen=True)

    properties: dict[str, tuple[str, ...]]
    edges: tuple[Edge, ...]

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(sorted(self.properties))

    def has_label(self, label: str) -> bool:
        return label in self.properties

    def has_property(self, label: str, name: str) -> bool:
        return name in self.properties.get(label, ())

    def has_edge(self, start: str, relationship: str, end: str) -> bool:
        return any(
            edge.start == start and edge.type == relationship and edge.end == end
            for edge in self.edges
        )

    def neighbours(self, label: str) -> tuple[str, ...]:
        """Labels reachable from ``label`` in either direction."""
        found = {edge.end for edge in self.edges if edge.start == label}
        found |= {edge.start for edge in self.edges if edge.end == label}
        return tuple(sorted(found))

    def outline(self) -> str:
        """Level 1 view: labels and how they connect, without properties."""
        return (
            "Node labels: " + ", ".join(self.labels) + "\n"
            "Relationships:\n"
            + "\n".join(f"  {edge.render()}" for edge in self.edges)
        )

    def detail(self, labels: set[str]) -> str:
        """Level 2+ view: properties and edges of the selected labels only."""
        selected = sorted(label for label in labels if self.has_label(label))
        lines = [
            f"({label}) properties: " + ", ".join(self.properties[label])
            for label in selected
        ]
        touching = [
            edge.render()
            for edge in self.edges
            if edge.start in labels or edge.end in labels
        ]
        return "\n".join(lines) + "\nRelationships touching these labels:\n" + "\n".join(
            f"  {edge}" for edge in touching
        )


class SchemaCatalog:
    """Discovers the schema once and reuses it until the cache expires."""

    def __init__(
        self,
        settings: Neo4jSettings,
        driver: AsyncDriver,
        *,
        cache_seconds: float,
        timeout_seconds: float,
    ) -> None:
        self._settings = settings
        self._driver = driver
        self._cache_seconds = cache_seconds
        self._timeout = timeout_seconds
        self._cached: GraphSchema | None = None
        self._fetched_at = 0.0
        self._lock = asyncio.Lock()

    @property
    def cached(self) -> GraphSchema | None:
        """The schema already discovered, without triggering a read."""
        return self._cached

    async def get(self) -> GraphSchema:
        """Return the discovered schema, reading it only when the cache is cold."""
        if self._cached is not None and monotonic() - self._fetched_at < self._cache_seconds:
            return self._cached
        async with self._lock:
            # Another caller may have refreshed it while we waited.
            if self._cached is not None and monotonic() - self._fetched_at < self._cache_seconds:
                return self._cached
            schema = await self._discover()
            self._cached, self._fetched_at = schema, monotonic()
            return schema

    async def _discover(self) -> GraphSchema:
        property_rows, visualization = await asyncio.gather(
            self._read(_NODE_PROPERTIES), self._read(_VISUALIZATION)
        )

        properties: dict[str, set[str]] = {}
        for row in property_rows:
            name = str(row["property"])
            if name in _HIDDEN_PROPERTIES or not IDENTIFIER.match(name):
                continue
            for label in row["labels"] or []:
                properties.setdefault(str(label), set()).add(name)

        edges: set[Edge] = set()
        for relationship in (visualization[0]["relationships"] if visualization else []):
            start = _sole_label(relationship.start_node.labels)
            end = _sole_label(relationship.end_node.labels)
            if start and end and IDENTIFIER.match(relationship.type):
                edges.add(Edge(start=start, type=relationship.type, end=end))
                properties.setdefault(start, set())
                properties.setdefault(end, set())

        if not properties:
            raise RetrievalError("the graph reported no schema: nothing can be planned")
        return GraphSchema(
            properties={
                label: tuple(sorted(names)) for label, names in sorted(properties.items())
            },
            edges=tuple(sorted(edges, key=lambda edge: (edge.start, edge.type, edge.end))),
        )

    async def _read(self, query: LiteralString) -> list[dict[str, Any]]:
        """One bounded, read-only schema query."""
        try:
            result = await asyncio.wait_for(
                self._driver.execute_query(
                    query,
                    {},
                    database_=self._settings.database,
                    routing_=RoutingControl.READ,
                ),
                timeout=self._timeout,
            )
        except TimeoutError as error:
            raise RetrievalError("schema discovery timed out") from error
        except neo4j_exceptions.Neo4jError as error:
            raise RetrievalError(f"schema discovery failed: {error}") from error
        except OSError as error:
            raise RetrievalError(f"could not reach the context graph: {error}") from error
        return [dict(record) for record in result.records]


def _sole_label(labels: frozenset[str]) -> str | None:
    """A schema node carries exactly one label; anything else is unusable."""
    return next(iter(labels)) if len(labels) == 1 else None
