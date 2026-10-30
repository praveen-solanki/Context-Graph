"""Select ConvoMem sessions and convert them to Context Graph ingestion format.

Selection (demo set):
  - users present in all 4 evidence folders
  - top 5 users by above-median session count
  - per folder: 3 longest sessions per user with msg count >= folder median
  - 5 users x 3 sessions x 4 folders = 60 sessions

Writes to data/convomem_demo/
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "convomem" / "core_benchmark" / "evidence_questions"
OUT = ROOT / "data" / "convomem_demo"

FOLDERS = (
    "preference_evidence",
    "changing_evidence",
    "user_evidence",
    "implicit_connection_evidence",
)
SESSIONS_PER_USER = 3
NUM_USERS = 5
AGENT_ID = "agent-1"
BASE_TIME = datetime(2026, 9, 1, 9, 0, 0, tzinfo=timezone.utc)
TURN_GAP = timedelta(seconds=45)

SPEAKER_TO_ROLE = {"User": "user", "Assistant": "agent"}


_PersonSessions = dict[str, dict[str, dict[str, tuple[int, list[dict[str, Any]]]]]]


def load_person_sessions() -> _PersonSessions:
    """person -> folder -> conv_id -> (msg_count, messages)."""
    store: _PersonSessions = defaultdict(lambda: {f: {} for f in FOLDERS})
    for folder in FOLDERS:
        for path in (SRC / folder).rglob("*.json"):
            data = json.loads(path.read_text(encoding="utf-8"))
            for item in data.get("evidence_items", []):
                pid = item["personId"]
                for conv in item.get("conversations") or []:
                    cid = conv["id"]
                    msgs = conv.get("messages") or []
                    n = len(msgs)
                    prev = store[pid][folder].get(cid)
                    if prev is None or n > prev[0]:
                        store[pid][folder][cid] = (n, msgs)
    return store


def median(values: list[int]) -> int:
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def pick_users(store: _PersonSessions) -> tuple[list[str], dict[str, int]]:
    in_all = [pid for pid, folders in store.items() if all(folders[f] for f in FOLDERS)]
    medians = {
        f: median([n for pid in in_all for n, _ in store[pid][f].values()])
        for f in FOLDERS
    }

    def score(pid: str) -> int:
        return sum(
            sum(1 for n, _ in store[pid][f].values() if n >= medians[f])
            for f in FOLDERS
        )

    ranked = sorted(in_all, key=score, reverse=True)
    return ranked[:NUM_USERS], medians


def job_title(person_id: str) -> str | None:
    for folder in FOLDERS:
        matches = list((SRC / folder).rglob(f"{person_id}_*.json"))
        if matches:
            stem = matches[0].stem
            if "_" in stem:
                return stem.split("_", 1)[1]
    return None


def to_session(
    *,
    session_id: str,
    user_id: str,
    messages: list[dict[str, Any]],
    folder: str,
) -> dict[str, Any]:
    short_user = f"U-{user_id[:8]}"
    out_msgs: list[dict[str, Any]] = []
    t = BASE_TIME
    for i, message in enumerate(messages, start=1):
        speaker = message.get("speaker", "User")
        role = SPEAKER_TO_ROLE.get(speaker, "user")
        msg: dict[str, Any] = {
            "message_id": f"{session_id}-m{i:03d}",
            "timestamp": t.isoformat().replace("+00:00", "Z"),
            "role": role,
            "sequence": i,
            "text": message.get("text", ""),
            "sender_id": short_user if role == "user" else None,
            "agent_id": AGENT_ID if role == "agent" else None,
            "episode_id": None,
            "source_type": None,
            "metadata": {},
        }
        out_msgs.append(msg)
        t += TURN_GAP

    return {
        "session": {
            "session_id": session_id,
            "user_id": short_user,
            "agent_ids": [AGENT_ID],
            "messages": out_msgs,
            "session_start_time": out_msgs[0]["timestamp"],
            "session_end_time": out_msgs[-1]["timestamp"],
            "state": "closed",
            "metadata": {
                "source": "Salesforce/ConvoMem",
                "evidence_folder": folder,
                "original_person_id": user_id,
                "original_conversation_id": session_id,
                "job_title": job_title(user_id),
            },
        }
    }


def main() -> None:
    if not SRC.exists():
        raise SystemExit(f"missing source data: {SRC}")

    print("Scanning ConvoMem…")
    store = load_person_sessions()
    users, medians = pick_users(store)
    print(f"medians: {medians}")
    print(f"selected users ({len(users)}):")
    for user in users:
        print(f"  {user}  ({job_title(user) or 'unknown'})")

    OUT.mkdir(parents=True, exist_ok=True)
    all_sessions: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []

    for folder in FOLDERS:
        folder_out = OUT / folder
        folder_out.mkdir(parents=True, exist_ok=True)
        med = medians[folder]
        folder_sessions: list[dict[str, Any]] = []

        for pid in users:
            candidates: list[tuple[str, int, list[dict[str, Any]]]] = [
                (cid, n, msgs)
                for cid, (n, msgs) in store[pid][folder].items()
                if n >= med
            ]
            candidates.sort(key=lambda x: -x[1])
            for cid, n, msgs in candidates[:SESSIONS_PER_USER]:
                session = to_session(
                    session_id=cid,
                    user_id=pid,
                    messages=msgs,
                    folder=folder,
                )
                folder_sessions.append(session)
                all_sessions.append(session)
                manifest.append(
                    {
                        "folder": folder,
                        "person_id": pid,
                        "user_id": session["session"]["user_id"],
                        "job_title": job_title(pid),
                        "session_id": cid,
                        "message_count": n,
                        "median_threshold": med,
                    }
                )

        path = folder_out / "sessions.json"
        path.write_text(json.dumps(folder_sessions, indent=2) + "\n", encoding="utf-8")
        print(f"{folder}: wrote {len(folder_sessions)} sessions -> {path}")

    combined = OUT / "sessions.json"
    combined.write_text(json.dumps(all_sessions, indent=2) + "\n", encoding="utf-8")
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\nTotal: {len(all_sessions)} sessions")
    print(f"Combined: {combined}")
    print(f"Manifest: {OUT / 'manifest.json'}")
    print(f"\nIngest with:\n  uv run context-graph --input {combined}")


if __name__ == "__main__":
    main()
