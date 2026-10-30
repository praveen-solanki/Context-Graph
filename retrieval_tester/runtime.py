"""Application runtime shared by the UIs.

Owns the async resources (graph connection, model transport) and the event loop
they were created on, so a synchronous UI such as Streamlit can drive the same
async :class:`~retrieval_tester.chat.ChatController` without reconnecting on every
interaction.
"""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from types import TracebackType
from typing import Any, Self

from neo4j import AsyncGraphDatabase

from context_graph.providers.chat_backend import (
    ChatLLMProvider,
    ChatTransport,
    EmbeddingProviderImpl,
)
from retrieval_tester.agent import RetrievalAgent
from retrieval_tester.chat import ChatController
from retrieval_tester.config import AgentSettings, RetrievalTesterConfig
from retrieval_tester.context_tool import ContextGraphRetrievalTool
from retrieval_tester.graph_tools import GraphLookupTool
from retrieval_tester.models import RetrievalResult
from retrieval_tester.retrieval import Neo4jContextRetriever
from retrieval_tester.session_store import SessionStore


class AppRuntime:
    """Long-lived resources plus the loop that owns them."""

    def __init__(self, config: RetrievalTesterConfig, settings: AgentSettings | None = None) -> None:
        self._settings = settings or config.agent
        self._neo4j_uri = config.neo4j.uri
        self._neo4j_database = config.neo4j.database
        self._loop = asyncio.new_event_loop()
        self._transport = ChatTransport(config.chat, retry=config.retry)
        # One graph connection serves both retrieval and the lookup tool.
        self._driver = AsyncGraphDatabase.driver(
            config.neo4j.uri,
            auth=(config.neo4j.user, config.neo4j.password.get_secret_value()),
            notifications_min_severity="OFF",
        )
        embeddings = EmbeddingProviderImpl(embedding=config.embedding, retry=config.retry)
        self._retriever = Neo4jContextRetriever(
            config.neo4j,
            agent_settings=self._settings,
            embeddings=embeddings,
            driver=self._driver,
        )
        self.graph_tool = GraphLookupTool(config.neo4j, self._driver)
        # Planned Context Graph retrieval: its own tool, sharing the same graph
        # connection, model transport and embeddings.
        self.context_graph_tool = ContextGraphRetrievalTool(
            config.neo4j,
            self._driver,
            llm=ChatLLMProvider(
                backend=config.chat, llm=config.llm, transport=self._transport
            ),
            fallback=self._retriever,
            agent_settings=self._settings,
            embeddings=embeddings,
        )
        # Answers the planned-retrieval tab. Deliberately without the lookup
        # tool: the answer must come from what the Context Graph returned.
        self._answerer = RetrievalAgent(
            transport=self._transport,
            backend=config.chat,
            settings=self._settings,
            graph_tool=None,
        )
        self.store = SessionStore(self._settings.sessions_dir)
        self.controller = ChatController(
            retriever=self._retriever,
            agent=RetrievalAgent(
                transport=self._transport,
                backend=config.chat,
                settings=self._settings,
                graph_tool=self.graph_tool,
            ),
            store=self.store,
            settings=self._settings,
        )

    @classmethod
    def create(cls, settings: AgentSettings | None = None) -> Self:
        """Build the runtime from environment configuration."""
        return cls(RetrievalTesterConfig.load(), settings)

    @property
    def settings(self) -> AgentSettings:
        return self._settings

    @property
    def neo4j_uri(self) -> str:
        """Bolt URI this runtime is connected to (from pipeline settings)."""
        return self._neo4j_uri

    @property
    def neo4j_database(self) -> str:
        """Neo4j database name this runtime is using."""
        return self._neo4j_database

    def run[T](self, coroutine: Coroutine[Any, Any, T]) -> T:
        """Run ``coroutine`` on this runtime's loop and return its result."""
        return self._loop.run_until_complete(coroutine)

    async def answer(self, query: str, retrieval: RetrievalResult) -> str:
        """Answer ``query`` from retrieved Context only, using the same agent prompt."""
        reply = await self._answerer.reply(retrieval=retrieval, history=(), message=query)
        return reply.text

    def set_graph_tool_enabled(self, enabled: bool) -> None:
        """Turn the agent's graph lookups on or off for the next message."""
        self._settings = self._settings.model_copy(update={"enable_graph_tool": enabled})
        self.controller.replace_agent_settings(self._settings)

    def close(self) -> None:
        """Release the graph and model connections, then the loop."""
        try:
            self._loop.run_until_complete(self._driver.close())
            self._loop.run_until_complete(self._transport.aclose())
        finally:
            self._loop.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
