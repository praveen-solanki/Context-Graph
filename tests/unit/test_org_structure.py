"""Organizational structure: the governed writer for scope membership."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import pytest

from context_graph import org
from context_graph.config import Neo4jSettings
from context_graph.org import OrgStructure, OrgStructureStore
from tests.factories import REPO_ROOT


def structure(**overrides: Any) -> OrgStructure:
    payload: dict[str, Any] = {
        "organization": {"organization_id": "acme", "name": "Acme"},
        "departments": [{"department_id": "eng", "name": "Engineering"}],
        "teams": [{"team_id": "search", "name": "Search", "department_id": "eng"}],
        "people": [
            {"user_id": "U1", "name": "Praveen", "team_id": "search", "projects": ["apollo"]}
        ],
        "projects": [{"project_id": "apollo", "name": "Apollo", "team_id": "search"}],
    }
    payload.update(overrides)
    return OrgStructure.model_validate(payload)


class StubDriver:
    """Captures the structural writes instead of running them."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def execute_query(
        self, query: str, parameters: dict[str, Any], **kwargs: Any
    ) -> None:
        self.calls.append((query, parameters))

    async def close(self) -> None:
        return None


class FakeOrgGraphDriver:
    """A minimal in-memory graph that replays ``org.py``'s writes.

    Matched by identity against ``org.py``'s module-level query constants -
    this is a test double for this module's small, fixed set of writes, not a
    general Cypher engine. Edges are ``(type, from_id, to_id)`` triples; node
    ids are ``"Label:key"`` so unrelated labels never collide.
    """

    def __init__(self) -> None:
        self.edges: set[tuple[str, str, str]] = set()
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self._handlers: dict[str, Callable[[dict[str, Any]], None]] = {
            org._DEPARTMENTS: self._apply_departments,  # pyright: ignore[reportPrivateUsage]
            org._STALE_DEPARTMENTS: self._apply_stale_departments,  # pyright: ignore[reportPrivateUsage]
            org._TEAMS: self._apply_teams,  # pyright: ignore[reportPrivateUsage]
            org._STALE_TEAMS: self._apply_stale_teams,  # pyright: ignore[reportPrivateUsage]
            org._PEOPLE: self._apply_people,  # pyright: ignore[reportPrivateUsage]
            org._STALE_PEOPLE: self._apply_stale_people,  # pyright: ignore[reportPrivateUsage]
            org._PROJECTS: self._apply_projects,  # pyright: ignore[reportPrivateUsage]
            org._STALE_PROJECTS: self._apply_stale_projects,  # pyright: ignore[reportPrivateUsage]
            org._ASSIGNMENTS: self._apply_assignments,  # pyright: ignore[reportPrivateUsage]
            org._STALE_ASSIGNMENTS: self._apply_stale_assignments,  # pyright: ignore[reportPrivateUsage]
            org._REMOVED_ASSIGNMENTS: self._apply_removed_assignments,  # pyright: ignore[reportPrivateUsage]
        }

    async def execute_query(
        self, query: str, parameters: dict[str, Any], **kwargs: Any
    ) -> None:
        self.calls.append((query, parameters))
        handler = self._handlers.get(query)
        if handler is None:  # pragma: no cover - guards a query this fake does not model
            raise AssertionError(f"FakeOrgGraphDriver has no handler for query: {query!r}")
        handler(parameters)

    async def close(self) -> None:
        return None

    def _apply_departments(self, parameters: dict[str, Any]) -> None:
        organization = f"Organization:{parameters['organization_id']}"
        for row in parameters["departments"]:
            self.edges.add(("BELONGS_TO", f"Department:{row['department_id']}", organization))

    def _apply_stale_departments(self, parameters: dict[str, Any]) -> None:
        organization = f"Organization:{parameters['organization_id']}"
        keep = {f"Department:{item}" for item in parameters["department_ids"]}
        self.edges = {
            edge
            for edge in self.edges
            if not (
                edge[0] == "BELONGS_TO"
                and edge[1].startswith("Department:")
                and (edge[2] != organization or edge[1] not in keep)
            )
        }

    def _apply_teams(self, parameters: dict[str, Any]) -> None:
        for row in parameters["teams"]:
            team = f"Team:{row['team_id']}"
            self.edges = {
                edge for edge in self.edges if not (edge[0] == "BELONGS_TO" and edge[1] == team)
            }
            self.edges.add(("BELONGS_TO", team, f"Department:{row['department_id']}"))

    def _apply_stale_teams(self, parameters: dict[str, Any]) -> None:
        keep = {f"Team:{item}" for item in parameters["team_ids"]}
        self.edges = {
            edge
            for edge in self.edges
            if not (edge[0] == "BELONGS_TO" and edge[1].startswith("Team:") and edge[1] not in keep)
        }

    def _apply_people(self, parameters: dict[str, Any]) -> None:
        for row in parameters["people"]:
            user = f"User:{row['user_id']}"
            self.edges = {
                edge for edge in self.edges if not (edge[0] == "MEMBER_OF" and edge[1] == user)
            }
            self.edges.add(("MEMBER_OF", user, f"Team:{row['team_id']}"))

    def _apply_stale_people(self, parameters: dict[str, Any]) -> None:
        keep = {f"User:{item}" for item in parameters["user_ids"]}
        self.edges = {
            edge for edge in self.edges if not (edge[0] == "MEMBER_OF" and edge[1] not in keep)
        }

    def _apply_projects(self, parameters: dict[str, Any]) -> None:
        for row in parameters["projects"]:
            project = f"Project:{row['project_id']}"
            self.edges = {
                edge for edge in self.edges if not (edge[0] == "PART_OF" and edge[1] == project)
            }
            self.edges.add(("PART_OF", project, f"Team:{row['team_id']}"))

    def _apply_stale_projects(self, parameters: dict[str, Any]) -> None:
        keep = {f"Project:{item}" for item in parameters["project_ids"]}
        self.edges = {
            edge for edge in self.edges if not (edge[0] == "PART_OF" and edge[1] not in keep)
        }

    def _apply_assignments(self, parameters: dict[str, Any]) -> None:
        for row in parameters["assignments"]:
            self.edges.add(
                ("WORKS_ON", f"User:{row['user_id']}", f"Project:{row['project_id']}")
            )

    def _apply_stale_assignments(self, parameters: dict[str, Any]) -> None:
        for row in parameters["people"]:
            user = f"User:{row['user_id']}"
            keep = {f"Project:{item}" for item in row["projects"]}
            self.edges = {
                edge
                for edge in self.edges
                if not (edge[0] == "WORKS_ON" and edge[1] == user and edge[2] not in keep)
            }

    def _apply_removed_assignments(self, parameters: dict[str, Any]) -> None:
        keep_users = {f"User:{item}" for item in parameters["user_ids"]}
        keep_projects = {f"Project:{item}" for item in parameters["project_ids"]}
        self.edges = {
            edge
            for edge in self.edges
            if not (
                edge[0] == "WORKS_ON"
                and (edge[1] not in keep_users or edge[2] not in keep_projects)
            )
        }


async def test_apply_writes_membership_hierarchy_and_project_ownership() -> None:
    driver = StubDriver()
    store = OrgStructureStore(Neo4jSettings(), driver=driver)  # pyright: ignore[reportArgumentType]

    applied = await store.apply(structure())

    assert applied == {
        "departments": 1,
        "teams": 1,
        "people": 1,
        "projects": 1,
        "assignments": 1,
    }
    written = " ".join(query for query, _ in driver.calls)
    assert "MEMBER_OF" in written  # user -> team
    assert "BELONGS_TO" in written  # team -> department -> organization
    assert "PART_OF" in written  # project -> team
    assert "WORKS_ON" in written  # user -> project
    # Structure only: this writer must never touch derived context or evidence.
    assert "Context" not in written
    assert "Episode" not in written


async def test_apply_is_idempotent() -> None:
    driver = FakeOrgGraphDriver()
    store = OrgStructureStore(Neo4jSettings(), driver=driver)  # pyright: ignore[reportArgumentType]

    await store.apply(structure())
    edges_after_first_run = set(driver.edges)
    await store.apply(structure())

    assert driver.edges == edges_after_first_run


async def test_changing_the_organization_drops_the_old_department_link() -> None:
    driver = FakeOrgGraphDriver()
    store = OrgStructureStore(Neo4jSettings(), driver=driver)  # pyright: ignore[reportArgumentType]
    await store.apply(structure())
    assert ("BELONGS_TO", "Department:eng", "Organization:acme") in driver.edges

    await store.apply(structure(organization={"organization_id": "other", "name": "Other"}))

    assert ("BELONGS_TO", "Department:eng", "Organization:other") in driver.edges
    assert ("BELONGS_TO", "Department:eng", "Organization:acme") not in driver.edges


async def test_a_blank_organization_is_rejected_before_anything_is_written() -> None:
    driver = StubDriver()
    store = OrgStructureStore(Neo4jSettings(), driver=driver)  # pyright: ignore[reportArgumentType]
    blank = structure(organization={"organization_id": "", "name": ""})

    with pytest.raises(ValueError, match="no organization"):
        await store.apply(blank)

    assert driver.calls == []


async def test_moving_a_person_to_a_new_team_drops_the_old_membership() -> None:
    driver = FakeOrgGraphDriver()
    store = OrgStructureStore(Neo4jSettings(), driver=driver)  # pyright: ignore[reportArgumentType]
    await store.apply(structure())
    assert ("MEMBER_OF", "User:U1", "Team:search") in driver.edges

    moved = structure(
        teams=[
            {"team_id": "search", "name": "Search", "department_id": "eng"},
            {"team_id": "platform", "name": "Platform", "department_id": "eng"},
        ],
        people=[
            {"user_id": "U1", "name": "Praveen", "team_id": "platform", "projects": ["apollo"]}
        ],
        projects=[{"project_id": "apollo", "name": "Apollo", "team_id": "platform"}],
    )
    await store.apply(moved)

    assert ("MEMBER_OF", "User:U1", "Team:platform") in driver.edges
    assert ("MEMBER_OF", "User:U1", "Team:search") not in driver.edges


async def test_removing_a_project_assignment_drops_the_stale_works_on_edge() -> None:
    driver = FakeOrgGraphDriver()
    store = OrgStructureStore(Neo4jSettings(), driver=driver)  # pyright: ignore[reportArgumentType]
    await store.apply(structure())
    assert ("WORKS_ON", "User:U1", "Project:apollo") in driver.edges

    unassigned = structure(
        people=[{"user_id": "U1", "name": "Praveen", "team_id": "search", "projects": []}]
    )
    await store.apply(unassigned)

    assert ("WORKS_ON", "User:U1", "Project:apollo") not in driver.edges


async def test_a_person_removed_from_the_snapshot_loses_their_membership_and_assignments() -> None:
    driver = FakeOrgGraphDriver()
    store = OrgStructureStore(Neo4jSettings(), driver=driver)  # pyright: ignore[reportArgumentType]
    await store.apply(structure())

    without_u1 = structure(people=[])
    await store.apply(without_u1)

    assert not any(edge[1] == "User:U1" for edge in driver.edges)


async def test_a_reapplied_unchanged_file_leaves_edges_exactly_as_they_were() -> None:
    driver = FakeOrgGraphDriver()
    store = OrgStructureStore(Neo4jSettings(), driver=driver)  # pyright: ignore[reportArgumentType]
    await store.apply(structure())
    expected = {
        ("BELONGS_TO", "Department:eng", "Organization:acme"),
        ("BELONGS_TO", "Team:search", "Department:eng"),
        ("MEMBER_OF", "User:U1", "Team:search"),
        ("PART_OF", "Project:apollo", "Team:search"),
        ("WORKS_ON", "User:U1", "Project:apollo"),
    }
    assert driver.edges == expected

    await store.apply(structure())

    assert driver.edges == expected


def test_unknown_project_assignment_is_rejected() -> None:
    broken = structure(
        people=[{"user_id": "U1", "name": "X", "team_id": "search", "projects": ["ghost"]}]
    )
    with pytest.raises(ValueError, match="works on unknown project 'ghost'"):
        broken.validate_references()


def test_unknown_team_reference_is_rejected() -> None:
    broken = structure(people=[{"user_id": "U1", "name": "X", "team_id": "ghost"}])
    with pytest.raises(ValueError, match="unknown team 'ghost'"):
        broken.validate_references()


def test_unknown_department_reference_is_rejected() -> None:
    broken = structure(teams=[{"team_id": "search", "name": "S", "department_id": "ghost"}])
    with pytest.raises(ValueError, match="unknown department 'ghost'"):
        broken.validate_references()


async def test_invalid_structure_is_rejected_before_anything_is_written() -> None:
    driver = StubDriver()
    store = OrgStructureStore(Neo4jSettings(), driver=driver)  # pyright: ignore[reportArgumentType]

    with pytest.raises(ValueError):
        await store.apply(structure(projects=[{"project_id": "p", "name": "P", "team_id": "ghost"}]))
    assert driver.calls == []


def test_shipped_org_file_is_an_empty_template() -> None:
    loaded = json.loads((REPO_ROOT / "data" / "org_structure.json").read_text(encoding="utf-8"))

    assert loaded["organization"] == {"organization_id": "", "name": ""}
    assert loaded["departments"] == []
    assert loaded["teams"] == []
    assert loaded["people"] == []
    assert loaded["projects"] == []
