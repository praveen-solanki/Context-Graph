"""Local JSON storage for retrieval-tester sessions.

This is the only persistence in the application, and it is one-way: this
application writes session files and never reads them back into the Context
Graph. Each file is written in the ingestion tool's input shape, so you can
choose to ingest one yourself later; nothing here does that for you.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from retrieval_tester.errors import SessionStoreError
from retrieval_tester.models import AgentSession

_UNSAFE = re.compile(r"[^A-Za-z0-9_.-]+")


class SessionStore:
    """Writes each retrieval-tester session to its own JSON file."""

    def __init__(self, directory: Path) -> None:
        self._directory = directory

    @property
    def directory(self) -> Path:
        return self._directory

    def path_for(self, session: AgentSession) -> Path:
        """Stable file path for a session."""
        return self._directory / f"{_UNSAFE.sub('-', session.session_id)}.json"

    def save(self, session: AgentSession) -> Path:
        """Write the session, replacing any earlier snapshot of it."""
        path = self.path_for(session)
        try:
            self._directory.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(session.to_document(), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError as error:
            raise SessionStoreError(f"could not save session to {path}: {error}") from error
        return path

    def load(self, path: Path) -> AgentSession:
        """Read a previously saved session."""
        try:
            return AgentSession.from_document(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError, KeyError) as error:
            raise SessionStoreError(f"could not read session from {path}: {error}") from error

    def sessions_for(self, user_id: str) -> list[Path]:
        """Saved sessions for one user, oldest first."""
        if not self._directory.exists():
            return []
        prefix = _UNSAFE.sub("-", user_id)
        return sorted(self._directory.glob(f"DA-{prefix}-*.json"))
