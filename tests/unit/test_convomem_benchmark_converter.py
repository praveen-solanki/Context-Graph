"""ConvoMem benchmark converter: raw benchmark files become valid ingestion input."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from context_graph.cli import load_requests
from context_graph.tool import IngestionRequest
from convomem_benchmark_converter import prepare
from tests.factories import REPO_ROOT


def conversation(conv_id: str, turns: int) -> dict[str, Any]:
    speakers = ("User", "Assistant")
    return {
        "id": conv_id,
        "messages": [
            {"speaker": speakers[index % 2], "text": f"{conv_id} turn {index}"}
            for index in range(turns)
        ],
    }


def write_person(
    src: Path, folder: str, person_id: str, conversations: list[dict[str, Any]], title: str = "Analyst"
) -> None:
    target = src / folder / f"{person_id}_{title}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps({"evidence_items": [{"personId": person_id, "conversations": conversations}]}),
        encoding="utf-8",
    )


@pytest.fixture
def benchmark(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """Point the preparation step at an empty benchmark and output folder."""
    src, out = tmp_path / "evidence_questions", tmp_path / "convomem_demo"
    src.mkdir()
    monkeypatch.setattr(prepare, "SRC", src)
    monkeypatch.setattr(prepare, "OUT", out)
    return src, out


@pytest.mark.parametrize(
    ("values", "expected"), [([3, 1, 2], 2), ([4, 1, 3, 2], 3), ([7], 7)]
)
def test_median_takes_the_upper_middle_value(values: list[int], expected: int) -> None:
    assert prepare.median(values) == expected


def test_a_session_is_converted_into_a_valid_ingestion_request(
    benchmark: tuple[Path, Path],
) -> None:
    src, _ = benchmark
    person = "46c13060-46dd-448f-b6c7-5e932c5e8a06"
    write_person(src, "user_evidence", person, [], title="SEO_Specialist")

    document = prepare.to_session(
        session_id="conv-1",
        user_id=person,
        messages=conversation("conv-1", 3)["messages"],
        folder="user_evidence",
    )
    session = IngestionRequest.model_validate(document).session

    assert session.user_id == "U-46c13060"
    assert [message.role.value for message in session.messages] == ["user", "agent", "user"]
    assert [message.message_id for message in session.messages] == [
        "conv-1-m001", "conv-1-m002", "conv-1-m003",
    ]
    stamps = [message.timestamp for message in session.messages]
    assert all(later - earlier == prepare.TURN_GAP for earlier, later in zip(stamps, stamps[1:]))
    assert document["session"]["metadata"]["job_title"] == "SEO_Specialist"
    assert document["session"]["metadata"]["evidence_folder"] == "user_evidence"


def test_a_conversation_seen_twice_keeps_its_longest_copy(benchmark: tuple[Path, Path]) -> None:
    src, _ = benchmark
    write_person(src, "user_evidence", "p1", [conversation("c1", 2)], title="A")
    write_person(src, "user_evidence", "p1", [conversation("c1", 6)], title="B")

    store = prepare.load_person_sessions()

    count, messages = store["p1"]["user_evidence"]["c1"]
    assert count == 6 and len(messages) == 6


def test_only_users_present_in_every_folder_are_selected(benchmark: tuple[Path, Path]) -> None:
    src, _ = benchmark
    for folder in prepare.FOLDERS:
        write_person(src, folder, "everywhere", [conversation(f"{folder}-e", 4)])
    write_person(src, prepare.FOLDERS[0], "partial", [conversation("p", 40)])

    users, medians = prepare.pick_users(prepare.load_person_sessions())

    assert users == ["everywhere"]
    assert set(medians) == set(prepare.FOLDERS)


def test_selection_ranks_users_and_is_capped(
    benchmark: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    src, _ = benchmark
    monkeypatch.setattr(prepare, "NUM_USERS", 1)
    for folder in prepare.FOLDERS:
        write_person(src, folder, "short", [conversation(f"{folder}-s", 2)])
        write_person(src, folder, "long", [conversation(f"{folder}-l", 9)])

    users, _ = prepare.pick_users(prepare.load_person_sessions())

    assert users == ["long"]


def test_preparation_writes_ingestible_files_and_a_manifest(
    benchmark: tuple[Path, Path],
) -> None:
    src, out = benchmark
    for folder in prepare.FOLDERS:
        write_person(src, folder, "p1", [conversation(f"{folder}-a", 4), conversation(f"{folder}-b", 6)])

    prepare.main()

    combined = load_requests(out / "sessions.json")
    per_folder = [load_requests(out / folder / "sessions.json") for folder in prepare.FOLDERS]
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert len(combined) == sum(len(requests) for requests in per_folder) == len(manifest)
    assert {entry["folder"] for entry in manifest} == set(prepare.FOLDERS)
    # Longest sessions first within a folder, never below that folder's median.
    assert all(entry["message_count"] >= entry["median_threshold"] for entry in manifest)


def test_preparation_refuses_to_run_without_the_benchmark(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(prepare, "SRC", tmp_path / "missing")
    with pytest.raises(SystemExit, match="missing source data"):
        prepare.main()


def test_the_shipped_demo_dataset_is_valid_ingestion_input() -> None:
    path = REPO_ROOT / "data" / "convomem_demo" / "sessions.json"
    if not path.is_file():  # pragma: no cover - data/ is not versioned
        pytest.skip("prepared demo dataset not present in this checkout")

    requests = load_requests(path)

    assert requests
    assert len({request.session.session_id for request in requests}) == len(requests)
    for request in requests:
        stamps: list[datetime] = [message.timestamp for message in request.session.ordered_messages()]
        assert stamps == sorted(stamps)
