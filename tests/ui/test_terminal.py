"""Terminal chat UI: commands, replies and clean exits, over the real controller."""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any

import pytest

from retrieval_tester.chat import ChatController
from retrieval_tester.session_store import SessionStore
from tests.agent_doubles import (
    StubRetriever,
    StubTransport,
    controller_for,
    sample_item,
    sample_retrieval,
)
from ui.terminal import TerminalUI

Ask = Callable[[str], Coroutine[Any, Any, str]]


def scripted(*answers: str | type[BaseException]) -> Ask:
    """Feed the UI's prompts from a script; an exception type is raised instead."""
    queue = list(answers)

    async def ask(prompt: str) -> str:
        answer = queue.pop(0) if queue else EOFError
        if isinstance(answer, type):
            raise answer
        return answer

    return ask


@pytest.fixture
def retriever() -> StubRetriever:
    return StubRetriever(sample_retrieval(sample_item()))


@pytest.fixture
def controller(tmp_path: Path, retriever: StubRetriever) -> ChatController:
    return controller_for(tmp_path, StubTransport(), retriever)


def ui_for(
    controller: ChatController, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *answers: str | type[BaseException]
) -> TerminalUI:
    monkeypatch.setattr(TerminalUI, "_ask", staticmethod(scripted(*answers)))
    return TerminalUI(controller, SessionStore(tmp_path))


async def test_a_message_gets_a_reply_and_the_session_is_saved_on_quit(
    controller: ChatController, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    ui = ui_for(controller, tmp_path, monkeypatch, "hi there", "/quit")

    assert await ui.run("U1") == 0

    out = capsys.readouterr().out
    assert "read-only" in out
    assert "response_detail = concise" in out
    assert "agent: hello there" in out
    assert "session saved to" in out
    assert list(tmp_path.glob("DA-U1-*.json"))


async def test_the_user_id_is_asked_for_when_not_given(
    controller: ChatController, retriever: StubRetriever, tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ui = ui_for(controller, tmp_path, monkeypatch, "U-asked", "/quit")

    assert await ui.run() == 0
    assert retriever.calls == ["U-asked"]


async def test_a_session_that_cannot_start_exits_with_an_error(
    controller: ChatController, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    ui = ui_for(controller, tmp_path, monkeypatch, "   ")

    assert await ui.run() == 1
    assert "error: user_id must not be empty" in capsys.readouterr().out


@pytest.mark.parametrize("interrupt", [EOFError, KeyboardInterrupt])
async def test_end_of_input_ends_the_session_cleanly(
    interrupt: type[BaseException], controller: ChatController, tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    ui = ui_for(controller, tmp_path, monkeypatch, interrupt)

    assert await ui.run("U1") == 0
    assert "session saved to" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("/help", "/context   show the Context retrieved"),
        ("/context", "retrieved context (1)"),
        ("/debug", "retrieval: 1/1 items"),
        ("/prompt", "-- system prompt"),
        ("/sessions", "saved session(s) for U1"),
        ("/bogus", "unknown command '/bogus'"),
    ],
)
async def test_slash_commands(
    command: str, expected: str, controller: ChatController, tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    ui = ui_for(controller, tmp_path, monkeypatch, command, "/quit")

    assert await ui.run("U1") == 0
    assert expected in capsys.readouterr().out


async def test_new_and_user_commands_start_fresh_sessions(
    controller: ChatController, retriever: StubRetriever, tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ui = ui_for(controller, tmp_path, monkeypatch, "/new", "/user U2", "/quit")

    assert await ui.run("U1") == 0
    assert retriever.calls == ["U1", "U1", "U2"]


async def test_a_failed_reply_is_reported_and_the_chat_continues(
    tmp_path: Path, retriever: StubRetriever, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    controller = controller_for(tmp_path, StubTransport(content=None), retriever)
    ui = ui_for(controller, tmp_path, monkeypatch, "hello?", "", "/quit")

    assert await ui.run("U1") == 0
    out = capsys.readouterr().out
    assert "  error:" in out
    assert "session saved to" in out
