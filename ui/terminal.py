"""Terminal UI for the retrieval tester.

Rendering and input only: every action is delegated to
:class:`~retrieval_tester.chat.ChatController`.
"""

from __future__ import annotations

import asyncio
import textwrap

from retrieval_tester.chat import ChatController
from retrieval_tester.errors import RetrievalTesterError
from retrieval_tester.models import RetrievalResult
from retrieval_tester.session_store import SessionStore

_WIDTH = 88
_HELP = """commands:
  /context   show the Context retrieved for this session
  /debug     show retrieval diagnostics
  /prompt    show the exact system prompt the agent is running with
  /new       start a new session for the same user (re-runs retrieval)
  /user ID   switch to another user_id and start a session
  /sessions  list saved sessions for this user
  /help      show this help
  /quit      end the session and exit
"""


def _rule(title: str = "") -> str:
    if not title:
        return "-" * _WIDTH
    return f"-- {title} " + "-" * max(0, _WIDTH - len(title) - 4)


def _wrap(text: str, indent: str = "  ") -> str:
    return "\n".join(
        textwrap.fill(
            line, width=_WIDTH, initial_indent=indent, subsequent_indent=indent + "  "
        )
        or indent
        for line in text.splitlines()
    )


class TerminalUI:
    """Interactive console for chatting with the retrieval tester."""

    def __init__(self, controller: ChatController, store: SessionStore) -> None:
        self._controller = controller
        self._store = store
        self._user_id = ""

    async def run(self, user_id: str | None = None) -> int:
        """Run the chat loop until the user quits. Returns a process exit code."""
        print(_rule("Context Graph - retrieval tester (read-only)"))
        print("  Retrieves stored Context for a user and chats with it.")
        print("  Nothing from this conversation is written back to the graph.")
        print(_rule())

        identifier = user_id or await self._ask("user_id: ")
        if not await self._start(identifier):
            return 1

        while True:
            try:
                entry = (await self._ask(f"\n[{self._user_id}] you: ")).strip()
            except (EOFError, KeyboardInterrupt):
                print()
                entry = "/quit"

            if not entry:
                continue
            if entry.startswith("/"):
                if await self._command(entry):
                    continue
                break
            await self._say(entry)

        path = self._controller.end_session()
        if path is not None:
            print(f"\nsession saved to {path}")
        return 0

    async def _command(self, entry: str) -> bool:
        """Handle a slash command. Returns False when the UI should exit."""
        command, _, argument = entry.partition(" ")
        match command:
            case "/help":
                print(_HELP)
            case "/context":
                self._show_context(self._controller.retrieval)
            case "/debug":
                self._show_debug(self._controller.retrieval)
            case "/prompt":
                print(_rule("system prompt"))
                print(_wrap(self._controller.system_prompt))
                print(_rule())
            case "/new":
                await self._start(self._user_id)
            case "/user":
                target = argument.strip() or await self._ask("user_id: ")
                await self._start(target)
            case "/sessions":
                self._show_sessions()
            case "/quit" | "/exit":
                return False
            case _:
                print(f"  unknown command '{command}'. /help lists the commands.")
        return True

    async def _start(self, user_id: str) -> bool:
        """Start a session, reporting retrieval results or a clear error."""
        try:
            session = await self._controller.start_session(user_id)
        except RetrievalTesterError as error:
            print(f"  error: {error}")
            return False

        self._user_id = session.user_id
        print(f"\n  session {session.session_id}  (model {session.model})")
        self._show_context(self._controller.retrieval)
        self._show_debug(self._controller.retrieval)
        print(f"  saved to: {self._controller.saved_to}")
        print("  type a message, or /help for commands")
        return True

    async def _say(self, message: str) -> None:
        """Send one message and print the reply."""
        try:
            reply = await self._controller.send(message)
        except RetrievalTesterError as error:
            print(f"  error: {error}")
            return
        for call in self._controller.last_lookups:
            print(f"  [graph lookup] {call.lookup}({call.target!r}) -> {call.rows} rows")
        print(f"\n  agent: {_wrap(reply, indent='').strip()}")
        print(f"  [saved: {self._controller.saved_to}]")

    def _show_context(self, retrieval: RetrievalResult) -> None:
        directory = retrieval.directory
        if directory.is_known:
            print(_rule("organization (shared with the team)"))
            place = f"  {directory.team_name}"
            if directory.department_name:
                place += f" · {directory.department_name}"
            if directory.organization_name:
                place += f" · {directory.organization_name}"
            print(place)
            if directory.my_projects:
                print(f"  works on: {', '.join(directory.my_projects)}")
            for colleague in directory.colleagues:
                role = f" ({colleague.title})" if colleague.title else ""
                projects = f" — {', '.join(colleague.projects)}" if colleague.projects else ""
                print(f"    {colleague.name}{role}{projects}")

        print(_rule(f"retrieved context ({len(retrieval.items)})"))
        if not retrieval.items:
            print("  (none - the agent has no stored context for this user)")
        for item in retrieval.items:
            print(
                f"  [{item.scope_type.value}:{item.scope_id}] "
                f"{item.concept_key.split(':')[-1]} = {item.value}"
            )
            detail = f"      type={item.context_type.value} evidence={item.evidence_count}"
            if item.similarity is not None:
                detail += f" relevance={item.similarity:.2f}"
            if item.condition:
                detail += f" condition={item.condition}"
            print(detail)
            if item.evidence_quote:
                print(f'      said: "{item.evidence_quote}"')

        if retrieval.sessions:
            print(_rule(f"earlier conversations ({len(retrieval.sessions)})"))
            for session in retrieval.sessions:
                print(f"  {session.as_prompt_line()[2:]}")
        if retrieval.history:
            print(_rule(f"changed preferences ({len(retrieval.history)})"))
            for change in retrieval.history:
                print(f"  {change.as_prompt_line()[2:]}")
        print(_rule())

    def _show_debug(self, retrieval: RetrievalResult) -> None:
        debug = retrieval.debug
        print(
            f"  retrieval: {debug.returned}/{debug.candidates_considered} items in "
            f"{debug.duration_ms}ms  by_scope={debug.by_scope or '{}'}  "
            f"sessions={debug.sessions_recalled} changes={debug.changes_recalled} "
            f"directory={'yes' if debug.directory_found else 'no'} "
            f"relevance={'on' if debug.relevance_used else 'off'}"
            f"{'  (truncated)' if debug.truncated else ''}"
        )
        for note in debug.notes:
            print(f"    - {note}")

    def _show_sessions(self) -> None:
        paths = self._store.sessions_for(self._user_id)
        print(f"  {len(paths)} saved session(s) for {self._user_id}:")
        for path in paths:
            print(f"    {path}")

    @staticmethod
    async def _ask(prompt: str) -> str:
        """Read a line without blocking the event loop."""
        return await asyncio.to_thread(input, prompt)
