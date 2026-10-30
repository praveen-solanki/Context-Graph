"""Organizational structure: the governed writer for scope membership.

Conversation can never create these edges - a user saying "I'm on the Search
team" is a claim, not an authorization (spec section 5). This module is the
authoritative path: it takes a curated structure and writes ``MEMBER_OF``,
``BELONGS_TO`` and ``PART_OF``, which retrieval then traverses to decide which
shared scopes apply to a user.

It writes only structure. It never touches Context, Episodes or the ledger.
"""

from __future__ import annotations

from types import TracebackType
from typing import LiteralString, Self

from neo4j import AsyncDriver, AsyncGraphDatabase
from neo4j import exceptions as neo4j_exceptions
from pydantic import BaseModel, ConfigDict, Field

from context_graph.config import Neo4jSettings
from context_graph.errors import GraphWriteError
from context_graph.observability import get_logger, log_event

_logger = get_logger("graph.org")


class Organization(BaseModel):
    """The top-level governed scope."""

    model_config = ConfigDict(frozen=True)

    organization_id: str
    name: str


class Department(BaseModel):
    """A department inside the organization."""

    model_config = ConfigDict(frozen=True)

    department_id: str
    name: str


class Team(BaseModel):
    """A team, owned by a department."""

    model_config = ConfigDict(frozen=True)

    team_id: str
    name: str
    department_id: str


class Person(BaseModel):
    """A member of exactly one team, identified as the graph knows them."""

    model_config = ConfigDict(frozen=True)

    user_id: str
    name: str
    team_id: str
    title: str | None = None
    #: Projects this person currently works on (directory fact, not a claim).
    projects: tuple[str, ...] = ()


class Project(BaseModel):
    """A project governed by a team."""

    model_config = ConfigDict(frozen=True)

    project_id: str
    name: str
    team_id: str


class OrgStructure(BaseModel):
    """A complete, curated snapshot of who belongs where."""

    model_config = ConfigDict(frozen=True)

    organization: Organization
    departments: tuple[Department, ...] = ()
    teams: tuple[Team, ...] = ()
    people: tuple[Person, ...] = ()
    projects: tuple[Project, ...] = Field(default=())

    def validate_references(self) -> None:
        """Fail loudly if the snapshot points at things it does not define."""
        if not self.organization.organization_id.strip() or not self.organization.name.strip():
            raise ValueError(
                "org structure has no organization; set organization_id and name before applying"
            )
        departments = {department.department_id for department in self.departments}
        teams = {team.team_id for team in self.teams}
        problems = [
            f"team '{team.team_id}' references unknown department '{team.department_id}'"
            for team in self.teams
            if team.department_id not in departments
        ]
        problems += [
            f"person '{person.user_id}' references unknown team '{person.team_id}'"
            for person in self.people
            if person.team_id not in teams
        ]
        problems += [
            f"project '{project.project_id}' references unknown team '{project.team_id}'"
            for project in self.projects
            if project.team_id not in teams
        ]
        known_projects = {project.project_id for project in self.projects}
        problems += [
            f"person '{person.user_id}' works on unknown project '{project_id}'"
            for person in self.people
            for project_id in person.projects
            if project_id not in known_projects
        ]
        if problems:
            raise ValueError("; ".join(problems))


_DEPARTMENTS: LiteralString = """
MERGE (o:Organization {organization_id: $organization_id})
SET o.name = $organization_name
WITH o
UNWIND $departments AS row
MERGE (d:Department {department_id: row.department_id})
SET d.name = row.name
MERGE (d)-[:BELONGS_TO]->(o)
"""

#: A department dropped from the snapshot, or left pointing at an older
#: organization, loses that edge. The current snapshot's own edges stay.
_STALE_DEPARTMENTS: LiteralString = """
MATCH (d:Department)-[r:BELONGS_TO]->(o:Organization)
WHERE o.organization_id <> $organization_id OR NOT d.department_id IN $department_ids
DELETE r
"""

_TEAMS: LiteralString = """
UNWIND $teams AS row
MATCH (d:Department {department_id: row.department_id})
MERGE (t:Team {team_id: row.team_id})
SET t.name = row.name
MERGE (t)-[:BELONGS_TO]->(d)
WITH t, d
OPTIONAL MATCH (t)-[old:BELONGS_TO]->(stale:Department)
WHERE stale.department_id <> d.department_id
DELETE old
"""

#: A team dropped from the snapshot loses its edge to its old department.
_STALE_TEAMS: LiteralString = """
MATCH (t:Team)-[r:BELONGS_TO]->(:Department)
WHERE NOT t.team_id IN $team_ids
DELETE r
"""

_PEOPLE: LiteralString = """
UNWIND $people AS row
MATCH (t:Team {team_id: row.team_id})
MERGE (u:User {user_id: row.user_id})
SET u.name = row.name, u.title = row.title
MERGE (u)-[:MEMBER_OF]->(t)
WITH u, t
OPTIONAL MATCH (u)-[old:MEMBER_OF]->(stale:Team)
WHERE stale.team_id <> t.team_id
DELETE old
"""

#: A person dropped from the snapshot loses their team membership. The User
#: node itself is never touched - it may exist only because a conversation
#: created it, and this writer never deletes nodes, only its own edges.
_STALE_PEOPLE: LiteralString = """
MATCH (u:User)-[r:MEMBER_OF]->(:Team)
WHERE NOT u.user_id IN $user_ids
DELETE r
"""

_PROJECTS: LiteralString = """
UNWIND $projects AS row
MATCH (t:Team {team_id: row.team_id})
MERGE (p:Project {project_id: row.project_id})
SET p.name = row.name
MERGE (p)-[:PART_OF]->(t)
WITH p, t
OPTIONAL MATCH (p)-[old:PART_OF]->(stale:Team)
WHERE stale.team_id <> t.team_id
DELETE old
"""

#: A project dropped from the snapshot loses its edge to its old team.
_STALE_PROJECTS: LiteralString = """
MATCH (p:Project)-[r:PART_OF]->(:Team)
WHERE NOT p.project_id IN $project_ids
DELETE r
"""

#: Directory-level assignment, written after projects exist.
_ASSIGNMENTS: LiteralString = """
UNWIND $assignments AS row
MATCH (u:User {user_id: row.user_id})
MATCH (p:Project {project_id: row.project_id})
MERGE (u)-[:WORKS_ON]->(p)
"""

#: A person's assignment list is a set, not a single edge: drop any of their
#: WORKS_ON edges that point at a project no longer in their current list.
_STALE_ASSIGNMENTS: LiteralString = """
UNWIND $people AS row
MATCH (u:User {user_id: row.user_id})
OPTIONAL MATCH (u)-[old:WORKS_ON]->(stale:Project)
WHERE NOT stale.project_id IN row.projects
DELETE old
"""

#: A person or project dropped from the snapshot entirely loses every
#: WORKS_ON edge involving it.
_REMOVED_ASSIGNMENTS: LiteralString = """
MATCH (u:User)-[r:WORKS_ON]->(p:Project)
WHERE NOT u.user_id IN $user_ids OR NOT p.project_id IN $project_ids
DELETE r
"""


class OrgStructureStore:
    """Writes organizational structure into the graph, idempotently."""

    def __init__(self, settings: Neo4jSettings, driver: AsyncDriver | None = None) -> None:
        self._settings = settings
        self._driver = driver or AsyncGraphDatabase.driver(
            settings.uri,
            auth=(settings.user, settings.password.get_secret_value()),
            notifications_min_severity="OFF",
        )

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

    async def apply(self, structure: OrgStructure) -> dict[str, int]:
        """Write the structure so the graph's org edges exactly match it.

        MERGE keeps an unchanged edge untouched, so re-applying the same
        snapshot changes nothing. A membership, team, project or assignment
        that changed, or an entity dropped from the snapshot entirely, has
        its stale edge removed - only edges this writer owns (``MEMBER_OF``,
        ``BELONGS_TO``, ``PART_OF``, ``WORKS_ON``); Context, Episodes, the
        ledger and nodes themselves are never touched.
        """
        structure.validate_references()
        department_ids = [item.department_id for item in structure.departments]
        team_ids = [item.team_id for item in structure.teams]
        project_ids = [item.project_id for item in structure.projects]
        user_ids = [item.user_id for item in structure.people]

        await self._run(
            _DEPARTMENTS,
            {
                "organization_id": structure.organization.organization_id,
                "organization_name": structure.organization.name,
                "departments": [item.model_dump() for item in structure.departments],
            },
        )
        await self._run(
            _STALE_DEPARTMENTS,
            {
                "organization_id": structure.organization.organization_id,
                "department_ids": department_ids,
            },
        )
        await self._run(_TEAMS, {"teams": [item.model_dump() for item in structure.teams]})
        await self._run(_STALE_TEAMS, {"team_ids": team_ids})
        await self._run(
            _PEOPLE,
            {"people": [item.model_dump(exclude={"projects"}) for item in structure.people]},
        )
        await self._run(_STALE_PEOPLE, {"user_ids": user_ids})
        await self._run(
            _PROJECTS,
            {"projects": [item.model_dump(exclude={"projects"}) for item in structure.projects]},
        )
        await self._run(_STALE_PROJECTS, {"project_ids": project_ids})
        assignments = [
            {"user_id": person.user_id, "project_id": project_id}
            for person in structure.people
            for project_id in person.projects
        ]
        await self._run(_ASSIGNMENTS, {"assignments": assignments})
        await self._run(
            _STALE_ASSIGNMENTS,
            {
                "people": [
                    {"user_id": person.user_id, "projects": list(person.projects)}
                    for person in structure.people
                ]
            },
        )
        await self._run(_REMOVED_ASSIGNMENTS, {"user_ids": user_ids, "project_ids": project_ids})

        applied = {
            "departments": len(structure.departments),
            "teams": len(structure.teams),
            "people": len(structure.people),
            "projects": len(structure.projects),
            "assignments": len(assignments),
        }
        log_event(_logger, "org.structure_applied", **applied)
        return applied

    async def _run(self, query: LiteralString, parameters: dict[str, object]) -> None:
        try:
            await self._driver.execute_query(
                query, parameters, database_=self._settings.database
            )
        except neo4j_exceptions.Neo4jError as error:
            raise GraphWriteError(f"org structure write failed: {error}") from error
