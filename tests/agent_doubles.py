"""Test doubles for the retrieval tester: a canned model transport, a canned
retriever, and a controller wired from them over a temporary session store."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from context_graph.config import ChatBackendSettings
from retrieval_tester.agent import RetrievalAgent
from retrieval_tester.chat import ChatController
from retrieval_tester.config import AgentSettings
from retrieval_tester.models import RetrievalDebug, RetrievalResult, RetrievedContext
from retrieval_tester.session_store import SessionStore


def sample_retrieval(*items: RetrievedContext) -> RetrievalResult:
    return RetrievalResult(
        items=items,
        debug=RetrievalDebug(
            user_id="U1", duration_ms=1.0, candidates_considered=len(items), returned=len(items)
        ),
    )


def sample_item(concept: str = "response_detail", value: str = "concise") -> RetrievedContext:
    return RetrievedContext(
        context_id=f"ctx_{concept}",
        concept_key=f"personalization:{concept}",
        context_type="Preference",  # pyright: ignore[reportArgumentType]
        value=value,
        scope_type="USER",  # pyright: ignore[reportArgumentType]
        scope_id="U1",
        evidence_count=1,
    )


class StubTransport:
    """Chat transport double that captures the request payload."""

    def __init__(self, content: str | None = "hello there") -> None:
        self._content = content
        self.payloads: list[dict[str, Any]] = []

    async def post(self, path: str, payload: dict[str, Any], label: str) -> dict[str, Any]:
        self.payloads.append(payload)
        return {
            "choices": [
                {"finish_reason": "stop", "message": {"content": self._content, "role": "assistant"}}
            ]
        }


def chat_settings() -> ChatBackendSettings:
    from pydantic import SecretStr

    return ChatBackendSettings(
        api_key=SecretStr("k"),
        base_url="https://example.test/v1",
        chat_model="chat-test",
    )


def agent_for(transport: Any, **overrides: Any) -> RetrievalAgent:
    return RetrievalAgent(
        transport=transport, backend=chat_settings(), settings=AgentSettings(**overrides)
    )


class StubRetriever:
    def __init__(self, result: RetrievalResult) -> None:
        self._result = result
        self.calls: list[str] = []
        self.queries: list[str | None] = []

    async def retrieve(self, user_id: str, query: str | None = None) -> RetrievalResult:
        self.calls.append(user_id)
        self.queries.append(query)
        return self._result


def controller_for(tmp_path: Path, transport: Any, retriever: Any) -> ChatController:
    return ChatController(
        retriever=retriever,
        agent=agent_for(transport),
        store=SessionStore(tmp_path),
        settings=AgentSettings(sessions_dir=tmp_path),
    )

