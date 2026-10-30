"""A read-only graph lookup the agent can call when it needs structure.

Pre-fetched retrieval answers "what is true about this user". Structural
questions - which teams exist, who is on a given team, who works on a given
project - are open-ended, so instead of pre-fetching every path this tool lets
the agent ask the graph directly, and only when a question needs it.

Safety: the model never writes a query. It picks one of a fixed set of lookups
and supplies a name; every query is parameterised, read-only, and runs with READ
routing. Personal Context is deliberately unreachable here - this tool sees the
organizational directory only.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, LiteralString

from neo4j import AsyncDriver, RoutingControl
from neo4j import exceptions as neo4j_exceptions
from pydantic import BaseModel, ConfigDict

from context_graph.config import Neo4jSettings
from retrieval_tester.errors import RetrievalError

TOOL_NAME = "graph_lookup"


class Lookup(StrEnum):
    """The questions this tool can answer."""

    TEAMS = "teams"
    TEAM_MEMBERS = "team_members"
    PERSON = "person"
    PROJECT = "project"


_QUERIES: dict[Lookup, LiteralString] = {
    Lookup.TEAMS: """
        MATCH (t:Team)
        OPTIONAL MATCH (t)-[:BELONGS_TO]->(d:Department)
        OPTIONAL MATCH (d)-[:BELONGS_TO]->(o:Organization)
        OPTIONAL MATCH (t)<-[:MEMBER_OF]-(u:User)
        RETURN t.name AS team, d.name AS department, o.name AS organization,
               count(DISTINCT u) AS members
        ORDER BY department, team
    """,
    Lookup.TEAM_MEMBERS: """
        MATCH (t:Team)
        WHERE toLower(t.name) CONTAINS toLower($target)
           OR toLower(t.team_id) CONTAINS toLower($target)
        MATCH (t)<-[:MEMBER_OF]-(u:User)
        OPTIONAL MATCH (u)-[:WORKS_ON]->(p:Project)
        RETURN t.name AS team, u.name AS person, u.title AS title,
               collect(DISTINCT p.name) AS projects
        ORDER BY team, person
    """,
    Lookup.PERSON: """
        MATCH (u:User)
        // Match on any word of the name, so a full name still surfaces everyone
        // who shares part of it. Name collisions must be visible to the caller,
        // however precisely the name was typed.
        WHERE any(word IN split(toLower(trim($target)), ' ')
                  WHERE word <> '' AND toLower(coalesce(u.name, u.user_id)) CONTAINS word)
        OPTIONAL MATCH (u)-[:MEMBER_OF]->(t:Team)
        OPTIONAL MATCH (t)-[:BELONGS_TO]->(d:Department)
        OPTIONAL MATCH (u)-[:WORKS_ON]->(p:Project)
        RETURN coalesce(u.name, u.user_id) AS person, u.title AS title,
               t.name AS team, d.name AS department,
               collect(DISTINCT p.name) AS projects
        ORDER BY person
    """,
    Lookup.PROJECT: """
        MATCH (p:Project)
        WHERE toLower(coalesce(p.name, p.project_id)) CONTAINS toLower($target)
           OR toLower(p.project_id) CONTAINS toLower($target)
        OPTIONAL MATCH (p)-[:PART_OF]->(t:Team)
        OPTIONAL MATCH (p)<-[:WORKS_ON]-(u:User)
        RETURN coalesce(p.name, p.project_id) AS project, t.name AS team,
               collect(DISTINCT coalesce(u.name, u.user_id)) AS people
        ORDER BY project
    """,
}

#: Definition handed to the model. One tool, a closed set of lookups.
TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "description": (
            "Look up the company's organizational structure: which teams exist, "
            "who is on a team, where a person sits, or who works on a project. "
            "Use it for questions about teams, colleagues, projects and reporting "
            "structure. It cannot see anyone's private preferences."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "lookup": {
                    "type": "string",
                    "enum": [item.value for item in Lookup],
                    "description": (
                        "teams: every team in the organization. "
                        "team_members: people on a named team. "
                        "person: every person in the organization whose name matches, "
                        "with their team and projects. Use it for any question about "
                        "someone by name, including finding out whether more than one "
                        "person shares that name. "
                        "project: a project's owning team and the people on it."
                    ),
                },
                "target": {
                    "type": "string",
                    "description": (
                        "Team, person or project name, exactly as the user wrote it. "
                        "Never expand or complete a partial name: a first name given "
                        "on its own must be searched on its own, or everyone else who "
                        "shares it is missed. Empty string for 'teams'."
                    ),
                },
            },
            "required": ["lookup", "target"],
            "additionalProperties": False,
        },
    },
}


class GraphLookupCall(BaseModel):
    """One lookup the agent performed, recorded so the UI can show it."""

    model_config = ConfigDict(frozen=True)

    lookup: str
    target: str
    rows: int
    result: str


class GraphLookupTool:
    """Executes the fixed set of organizational lookups, read-only."""

    def __init__(self, settings: Neo4jSettings, driver: AsyncDriver) -> None:
        self._settings = settings
        self._driver = driver
        self.calls: list[GraphLookupCall] = []

    def reset(self) -> None:
        """Forget the lookups recorded so far."""
        self.calls.clear()

    async def run(self, lookup: str, target: str = "") -> GraphLookupCall:
        """Run one lookup and return a compact, model-readable answer."""
        try:
            selected = Lookup(lookup)
        except ValueError:
            return self._record(lookup, target, [], f"unknown lookup '{lookup}'")

        if selected is not Lookup.TEAMS and not target.strip():
            return self._record(lookup, target, [], f"'{lookup}' needs a name to look up")

        try:
            result = await self._driver.execute_query(
                _QUERIES[selected],
                {"target": target.strip()},
                database_=self._settings.database,
                routing_=RoutingControl.READ,
            )
        except neo4j_exceptions.Neo4jError as error:
            raise RetrievalError(f"graph lookup failed: {error}") from error

        rows = [dict(record) for record in result.records]
        return self._record(lookup, target, rows, _render(selected, rows))

    def _record(
        self, lookup: str, target: str, rows: list[dict[str, Any]], result: str
    ) -> GraphLookupCall:
        call = GraphLookupCall(lookup=lookup, target=target, rows=len(rows), result=result)
        self.calls.append(call)
        return call


def _render(lookup: Lookup, rows: list[dict[str, Any]]) -> str:
    """Turn rows into short lines the model can read directly."""
    if not rows:
        return "no match found"
    if lookup is Lookup.TEAMS:
        return "\n".join(
            f"- {row['team']} ({row['department'] or 'no department'}, "
            f"{row['members']} members)"
            for row in rows
        )
    if lookup is Lookup.TEAM_MEMBERS:
        return "\n".join(
            f"- {row['person']}"
            f"{f' ({row['title']})' if row['title'] else ''}"
            f" on {row['team']}"
            f"{f' — works on {', '.join(row['projects'])}' if row['projects'] else ''}"
            for row in rows
        )
    if lookup is Lookup.PERSON:
        return "\n".join(
            f"- {row['person']}"
            f"{f' ({row['title']})' if row['title'] else ''}"
            f" — team {row['team'] or 'unknown'}"
            f"{f', {row['department']}' if row['department'] else ''}"
            f"{f', works on {', '.join(row['projects'])}' if row['projects'] else ''}"
            for row in rows
        )
    return "\n".join(
        f"- {row['project']} — owned by {row['team'] or 'no team'}"
        f"{f', people: {', '.join(row['people'])}' if row['people'] else ''}"
        for row in rows
    )
