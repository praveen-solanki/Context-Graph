"""Chat orchestration: retrieve Context, run the conversation, save it locally.

This is the single place that sequences a session, so the UI holds no
orchestration logic and nothing else needs to repeat it.
"""

from __future__ import annotations

from pathlib import Path

from retrieval_tester.agent import AgentReply, RetrievalAgent
from retrieval_tester.config import AgentSettings
from retrieval_tester.errors import RetrievalTesterError
from retrieval_tester.graph_tools import GraphLookupCall
from context_graph.domain.enums import Role
from retrieval_tester.models import AgentSession, RetrievalResult, utcnow
from retrieval_tester.retrieval import ContextRetriever
from retrieval_tester.session_store import SessionStore


class ChatController:
    """Owns one live retrieval-tester session at a time."""

    def __init__(
        self,
        *,
        retriever: ContextRetriever,
        agent: RetrievalAgent,
        store: SessionStore,
        settings: AgentSettings,
    ) -> None:
        self._retriever = retriever
        self._agent = agent
        self._store = store
        self._settings = settings
        self._session: AgentSession | None = None
        self._retrieval: RetrievalResult | None = None
        self._saved_to: Path | None = None
        self._last_reply: AgentReply | None = None

    @property
    def session(self) -> AgentSession:
        """The live session, or an error if none has been started."""
        if self._session is None:
            raise RetrievalTesterError("no session is active; start one first")
        return self._session

    @property
    def retrieval(self) -> RetrievalResult:
        """Context retrieved at the start of the live session."""
        if self._retrieval is None:
            raise RetrievalTesterError("no session is active; start one first")
        return self._retrieval

    @property
    def last_lookups(self) -> tuple[GraphLookupCall, ...]:
        """Graph lookups the agent made for the most recent message."""
        return self._last_reply.lookups if self._last_reply else ()

    @property
    def saved_to(self) -> Path | None:
        """Where the live session was last written."""
        return self._saved_to

    @property
    def system_prompt(self) -> str:
        """The prompt the agent is running with, for inspection in the UI."""
        return self._agent.system_prompt(self.retrieval)

    def replace_agent_settings(self, settings: AgentSettings) -> None:
        """Apply changed agent settings to the live session."""
        self._settings = settings
        self._agent.replace_settings(settings)

    async def start_session(self, user_id: str) -> AgentSession:
        """Retrieve Context for ``user_id`` and open a new session."""
        identifier = user_id.strip()
        if not identifier:
            raise RetrievalTesterError("user_id must not be empty")

        # A session ends when the next one starts, so session-scoped context
        # from the old one stops being applicable (spec A.9).
        self.end_session()
        self._retrieval = await self._retriever.retrieve(identifier)
        self._session = AgentSession.start(
            session_id=f"DA-{identifier}-{utcnow():%Y%m%dT%H%M%SZ}",
            user_id=identifier,
            model=self._agent.model,
            retrieval=self._retrieval,
        )
        self._saved_to = self._store.save(self._session)
        return self._session

    async def send(self, message: str) -> str:
        """Send a user message and record both turns in the saved session."""
        text = message.strip()
        if not text:
            raise RetrievalTesterError("message must not be empty")

        session = self.session
        # Retrieve again for this message: relevance ranking needs the question,
        # and context may have changed since the session opened.
        self._retrieval = await self._retriever.retrieve(session.user_id, text)
        self._last_reply = await self._agent.reply(
            retrieval=self._retrieval, history=session.turns, message=text
        )
        reply = self._last_reply.text
        self._session = session.with_message(Role.USER, text).with_message(Role.AGENT, reply)
        self._session = self._session.model_copy(
            update={
                "retrieved_context": self._retrieval.items,
                "retrieval_debug": self._retrieval.debug,
                "directory": self._retrieval.directory,
                "graph_lookups": (
                    *self._session.graph_lookups,
                    *(call.model_dump(mode="json") for call in self.last_lookups),
                ),
            }
        )
        self._saved_to = self._store.save(self._session)
        return reply

    def end_session(self) -> Path | None:
        """Close and persist the live session, if there is one."""
        if self._session is None:
            return None
        self._session = self._session.closed()
        self._saved_to = self._store.save(self._session)
        return self._saved_to
