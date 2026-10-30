#!/usr/bin/env python
"""Chat with the retrieval tester using Context retrieved from the Context Graph.

The Context Graph is read-only here: this application never writes to it and
never calls the ingestion pipeline. Sessions are saved as local JSON.

    uv run python -m ui.chat_with_agent --user U-praveen
"""

from __future__ import annotations

import argparse
import asyncio

from context_graph.providers.chat_backend import EmbeddingProviderImpl, ChatTransport
from retrieval_tester.agent import RetrievalAgent
from retrieval_tester.chat import ChatController
from retrieval_tester.config import RetrievalTesterConfig
from retrieval_tester.errors import RetrievalTesterError
from retrieval_tester.graph_tools import GraphLookupTool
from retrieval_tester.retrieval import Neo4jContextRetriever
from retrieval_tester.session_store import SessionStore
from ui.terminal import TerminalUI


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--user", help="user_id to start with; prompted for if omitted.")
    parser.add_argument(
        "--max-context", type=int, help="Maximum Context items given to the agent."
    )
    parser.add_argument(
        "--no-shared",
        action="store_true",
        help="Exclude governed shared-scope Context from retrieval.",
    )
    parser.add_argument(
        "--no-graph-tool",
        action="store_true",
        help="Stop the agent querying the graph for structural questions.",
    )
    return parser.parse_args()


async def _run(arguments: argparse.Namespace) -> int:
    config = RetrievalTesterConfig.load()
    settings = config.agent.model_copy(
        update={
            key: value
            for key, value in (
                ("max_context_items", arguments.max_context),
                ("include_shared_scopes", False if arguments.no_shared else None),
                ("enable_graph_tool", False if arguments.no_graph_tool else None),
            )
            if value is not None
        }
    )

    store = SessionStore(settings.sessions_dir)
    async with ChatTransport(config.chat, retry=config.retry) as transport:
        retriever = Neo4jContextRetriever(
            config.neo4j,
            agent_settings=settings,
            embeddings=EmbeddingProviderImpl(
                embedding=config.embedding, retry=config.retry
            ),
        )
        graph_tool = GraphLookupTool(config.neo4j, retriever.driver)
        controller = ChatController(
            retriever=retriever,
            agent=RetrievalAgent(
                transport=transport,
                backend=config.chat,
                settings=settings,
                graph_tool=graph_tool,
            ),
            store=store,
            settings=settings,
        )
        return await TerminalUI(controller, store).run(arguments.user)


def main() -> int:
    """Entry point for ``python -m ui.chat_with_agent``."""
    try:
        return asyncio.run(_run(_parse_args()))
    except RetrievalTesterError as error:
        print(f"error: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
