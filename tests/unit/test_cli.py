"""The ``context-graph`` CLI: the only way to load data into the graph."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from types import TracebackType
from typing import Any, Self

import pytest
from pydantic import ValidationError

from context_graph import cli
from context_graph.config import DEFAULT_INPUT, Settings
from context_graph.org import OrgStructure
from context_graph.pipeline.orchestrator import IngestionReport
from context_graph.tool import IngestionRequest
from tests.factories import REPO_ROOT


def request_document(session_id: str) -> dict[str, Any]:
    return {
        "session": {
            "session_id": session_id,
            "user_id": "U1",
            "agent_ids": ["agent-1"],
            "messages": [
                {
                    "message_id": f"{session_id}-m1",
                    "timestamp": "2026-09-12T10:00:00+00:00",
                    "role": "user",
                    "sequence": 1,
                    "text": "keep answers short",
                }
            ],
        }
    }


def org_document(*, user_id: str = "U1", team_id: str = "search") -> dict[str, Any]:
    return {
        "organization": {"organization_id": "acme", "name": "Acme"},
        "departments": [{"department_id": "eng", "name": "Engineering"}],
        "teams": [{"team_id": team_id, "name": "Search", "department_id": "eng"}],
        "people": [{"user_id": user_id, "name": "Person", "team_id": team_id}],
    }


class FakeIngestion:
    """Stands in for ``ContextGraphIngestion``, recording what it was given."""

    instances: list[FakeIngestion] = []

    def __init__(self) -> None:
        self.received: list[list[str]] = []
        self.closed = False
        FakeIngestion.instances.append(self)

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> FakeIngestion:
        return cls()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.closed = True

    async def ingest_conversations(
        self, requests: Sequence[IngestionRequest]
    ) -> list[IngestionReport]:
        self.received.append([request.session.session_id for request in requests])
        return [
            IngestionReport(run_id="run-1", session_id=request.session.session_id)
            for request in requests
        ]


class FakeOrgStore:
    """Stands in for ``OrgStructureStore``, recording what it was asked to apply."""

    instances: list[FakeOrgStore] = []
    applied: list[OrgStructure] = []

    def __init__(self, neo4j_settings: object) -> None:
        FakeOrgStore.instances.append(self)

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        return None

    async def apply(self, structure: OrgStructure) -> dict[str, int]:
        FakeOrgStore.applied.append(structure)
        return {}


@pytest.fixture
def fake_ingestion(monkeypatch: pytest.MonkeyPatch) -> type[FakeIngestion]:
    FakeIngestion.instances = []
    monkeypatch.setattr(cli, "ContextGraphIngestion", FakeIngestion)
    return FakeIngestion


@pytest.fixture
def fake_org_store(monkeypatch: pytest.MonkeyPatch) -> type[FakeOrgStore]:
    FakeOrgStore.instances = []
    FakeOrgStore.applied = []
    monkeypatch.setattr(cli, "OrgStructureStore", FakeOrgStore)
    return FakeOrgStore


def write_json(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# -- settings defaults and CLI overrides -----------------------------------


def test_the_default_input_is_the_prepared_convomem_demo() -> None:
    expected = REPO_ROOT / "data" / "convomem_demo" / "sessions.json"
    assert DEFAULT_INPUT == expected
    assert Settings().input == expected


def test_org_structure_defaults_to_unset() -> None:
    assert Settings().org_structure is None


def test_an_empty_org_structure_setting_means_skip_it() -> None:
    assert Settings.model_validate({"org_structure": ""}).org_structure is None


def test_input_and_org_structure_are_overridable_from_the_command_line(tmp_path: Path) -> None:
    arguments = cli._parse_args(  # pyright: ignore[reportPrivateUsage]
        ["--input", str(tmp_path / "a.json"), "--org-structure", str(tmp_path / "b.json")]
    )
    assert arguments.input == tmp_path / "a.json"
    assert arguments.org_structure == tmp_path / "b.json"


def test_omitted_flags_keep_the_configured_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configured = Settings(input=tmp_path / "configured.json")
    monkeypatch.setattr(cli, "get_settings", lambda: configured)

    resolved = cli._resolved_settings(cli._parse_args([]))  # pyright: ignore[reportPrivateUsage]

    assert resolved.input == tmp_path / "configured.json"
    assert resolved.org_structure is None


def test_a_given_flag_overrides_only_that_setting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configured = Settings(input=tmp_path / "configured.json")
    monkeypatch.setattr(cli, "get_settings", lambda: configured)

    resolved = cli._resolved_settings(  # pyright: ignore[reportPrivateUsage]
        cli._parse_args(["--org-structure", str(tmp_path / "org.json")])  # pyright: ignore[reportPrivateUsage]
    )

    assert resolved.input == tmp_path / "configured.json"
    assert resolved.org_structure == tmp_path / "org.json"


# -- load_requests ----------------------------------------------------------


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (request_document("S1"), ["S1"]),
        ([request_document("S1"), request_document("S2")], ["S1", "S2"]),
    ],
    ids=["single-object", "list"],
)
def test_a_file_may_hold_one_request_or_a_list(
    tmp_path: Path, payload: object, expected: list[str]
) -> None:
    requests = cli.load_requests(write_json(tmp_path / "in.json", payload))
    assert [request.session.session_id for request in requests] == expected


def test_an_invalid_request_is_rejected_before_anything_is_ingested(tmp_path: Path) -> None:
    broken = request_document("S1")
    del broken["session"]["user_id"]
    with pytest.raises(ValidationError):
        cli.load_requests(write_json(tmp_path / "in.json", broken))


# -- run: ingestion -----------------------------------------------------


async def test_every_session_is_ingested_in_one_concurrent_call_and_reported(
    tmp_path: Path, fake_ingestion: type[FakeIngestion], capsys: pytest.CaptureFixture[str]
) -> None:
    path = write_json(tmp_path / "in.json", [request_document("S1"), request_document("S2")])
    settings = Settings(input=path)

    assert await cli.run(settings) == 0

    (ingestion,) = fake_ingestion.instances
    assert ingestion.received == [["S1", "S2"]]
    assert ingestion.closed
    printed = capsys.readouterr().out
    assert '"session_id": "S1"' in printed and '"session_id": "S2"' in printed


async def test_a_missing_input_is_a_clear_error_and_nothing_is_ingested(
    tmp_path: Path,
    fake_ingestion: type[FakeIngestion],
    fake_org_store: type[FakeOrgStore],
    capsys: pytest.CaptureFixture[str],
) -> None:
    org_path = write_json(tmp_path / "org.json", org_document())
    settings = Settings(input=tmp_path / "absent.json", org_structure=org_path)

    assert await cli.run(settings) == 1

    assert "input not found" in capsys.readouterr().err
    assert fake_ingestion.instances == []
    assert fake_org_store.instances == []


# -- run: org structure ---------------------------------------------------


async def test_org_structure_is_applied_before_ingestion_when_set(
    tmp_path: Path, fake_ingestion: type[FakeIngestion], fake_org_store: type[FakeOrgStore]
) -> None:
    input_path = write_json(tmp_path / "in.json", request_document("S1"))
    org_path = write_json(tmp_path / "org.json", org_document())
    settings = Settings(input=input_path, org_structure=org_path)

    assert await cli.run(settings) == 0

    (structure,) = fake_org_store.applied
    assert structure.organization.organization_id == "acme"
    assert fake_ingestion.instances[0].received == [["S1"]]


async def test_org_structure_is_skipped_when_unset(
    tmp_path: Path, fake_ingestion: type[FakeIngestion], fake_org_store: type[FakeOrgStore]
) -> None:
    input_path = write_json(tmp_path / "in.json", request_document("S1"))
    settings = Settings(input=input_path, org_structure=None)

    assert await cli.run(settings) == 0

    assert fake_org_store.instances == []
    assert fake_ingestion.instances[0].received == [["S1"]]


async def test_a_blank_org_file_is_a_clear_error_and_nothing_is_applied_or_ingested(
    tmp_path: Path,
    fake_ingestion: type[FakeIngestion],
    fake_org_store: type[FakeOrgStore],
    capsys: pytest.CaptureFixture[str],
) -> None:
    input_path = write_json(tmp_path / "in.json", request_document("S1"))
    org_path = write_json(tmp_path / "org.json", json.loads(
        (REPO_ROOT / "data" / "org_structure.json").read_text(encoding="utf-8")
    ))
    settings = Settings(input=input_path, org_structure=org_path)

    assert await cli.run(settings) == 1

    assert "no organization" in capsys.readouterr().err
    assert fake_org_store.instances == []
    assert fake_ingestion.instances == []


async def test_an_invalid_sessions_file_does_not_apply_org_structure(
    tmp_path: Path,
    fake_ingestion: type[FakeIngestion],
    fake_org_store: type[FakeOrgStore],
    capsys: pytest.CaptureFixture[str],
) -> None:
    broken = request_document("S1")
    del broken["session"]["user_id"]
    input_path = write_json(tmp_path / "in.json", broken)
    org_path = write_json(tmp_path / "org.json", org_document())
    settings = Settings(input=input_path, org_structure=org_path)

    assert await cli.run(settings) == 1

    assert "error:" in capsys.readouterr().err
    assert fake_org_store.instances == []
    assert fake_ingestion.instances == []


async def test_a_missing_org_structure_is_a_clear_error_and_nothing_is_applied_or_ingested(
    tmp_path: Path, fake_ingestion: type[FakeIngestion], fake_org_store: type[FakeOrgStore],
    capsys: pytest.CaptureFixture[str],
) -> None:
    input_path = write_json(tmp_path / "in.json", request_document("S1"))
    settings = Settings(input=input_path, org_structure=tmp_path / "absent-org.json")

    assert await cli.run(settings) == 1

    assert "org structure not found" in capsys.readouterr().err
    assert fake_org_store.instances == []
    assert fake_ingestion.instances == []


# -- main: end-to-end CLI flag wiring ---------------------------------------


def test_main_applies_cli_overrides_and_ingests(
    tmp_path: Path,
    fake_ingestion: type[FakeIngestion],
    fake_org_store: type[FakeOrgStore],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = write_json(tmp_path / "mine.json", request_document("S-mine"))
    org_path = write_json(tmp_path / "org.json", org_document(user_id="U-mine"))
    monkeypatch.setattr(cli, "get_settings", lambda: Settings())

    exit_code = cli.main(["--input", str(input_path), "--org-structure", str(org_path)])

    assert exit_code == 0
    assert fake_ingestion.instances[0].received == [["S-mine"]]
    assert len(fake_org_store.applied) == 1


def test_main_uses_configured_defaults_when_no_flags_are_given(
    tmp_path: Path,
    fake_ingestion: type[FakeIngestion],
    fake_org_store: type[FakeOrgStore],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = write_json(tmp_path / "default.json", request_document("S-default"))
    monkeypatch.setattr(cli, "get_settings", lambda: Settings(input=input_path))

    assert cli.main([]) == 0

    assert fake_ingestion.instances[0].received == [["S-default"]]
    assert fake_org_store.instances == []
