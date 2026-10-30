"""Retrieval tester: retrieval, prompt assembly, session storage and boundaries."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import pytest
from neo4j import RoutingControl
from neo4j import exceptions as neo4j_exceptions

from context_graph.config import EmbeddingSettings, Neo4jSettings, RetrySettings
from context_graph.domain.enums import SessionState
from retrieval_tester.config import AgentSettings
from retrieval_tester.errors import AgentError, RetrievalTesterError, RetrievalError
from context_graph.domain.enums import Role
from retrieval_tester.models import AgentSession
from retrieval_tester.retrieval import Neo4jContextRetriever
from retrieval_tester.session_store import SessionStore
from retrieval_tester import retrieval as retrieval_module
from tests.agent_doubles import (
    StubRetriever,
    StubTransport,
    agent_for,
    chat_settings,
    controller_for,
    sample_item,
    sample_retrieval,
)
from tests.factories import NOW, REPO_ROOT, context_node


class StubResult:
    def __init__(self, records: list[dict[str, Any]]) -> None:
        self.records = records


class StubDriver:
    """Answers each retrieval query with canned rows, recording the calls."""

    def __init__(
        self,
        context_rows: list[dict[str, Any]] | Exception,
        sessions: list[dict[str, Any]] | None = None,
        changes: list[dict[str, Any]] | None = None,
        directory: list[dict[str, Any]] | None = None,
    ) -> None:
        self._context = context_rows
        self._sessions = sessions or []
        self._changes = changes or []
        self._directory = directory if directory is not None else []
        self.calls: list[dict[str, Any]] = []
        self.closed = False

    async def execute_query(
        self, query: str, parameters: dict[str, Any], **kwargs: Any
    ) -> StubResult:
        self.calls.append({"query": query, "parameters": parameters, **kwargs})
        if "PARTICIPATED_IN" in query:
            return StubResult(self._sessions)
        if "SUPERSEDES" in query:
            return StubResult(self._changes)
        if "WORKS_ON" in query:
            return StubResult(self._directory)
        if isinstance(self._context, Exception):
            raise self._context
        return StubResult(self._context)

    async def close(self) -> None:
        self.closed = True


def retriever_for(
    driver: Any, max_items: int = 10, shared: bool = True, embeddings: Any = None
) -> Neo4jContextRetriever:
    return Neo4jContextRetriever(
        Neo4jSettings(),
        agent_settings=AgentSettings(max_context_items=max_items, include_shared_scopes=shared),
        embeddings=embeddings,
        driver=driver,
    )


async def test_retrieval_maps_nodes_and_reports_debug_information() -> None:
    rows = [
        {"node": context_node(), "quote": "keep it short"},
        {"node": context_node("tone", "formal", "PROJECT", "apollo", evidence=1), "quote": None},
    ]
    driver = StubDriver([{"considered": 5, "rows": rows}])

    result = await retriever_for(driver).retrieve("U1")

    assert [item.concept_key for item in result.items] == [
        "personalization:response_detail",
        "personalization:tone",
    ]
    assert result.items[0].value == "concise" and result.items[0].evidence_count == 2
    assert result.items[0].evidence_quote == "keep it short"
    assert result.debug.returned == 2
    assert result.debug.candidates_considered == 5
    assert result.debug.truncated is True
    assert result.debug.by_scope == {"USER": 1, "PROJECT": 1}
    block = result.as_prompt_block()
    assert "What is true about this user:" in block
    assert "- [USER] response_detail: concise" in block


async def test_retrieval_is_read_only_and_scoped_to_the_user() -> None:
    driver = StubDriver([{"considered": 0, "rows": []}])
    await retriever_for(driver).retrieve("U1")

    call = next(c for c in driver.calls if "$private_scopes" in c["query"])
    assert call["routing_"] is RoutingControl.READ
    assert call["parameters"]["user_id"] == "U1"
    assert call["parameters"]["active_status"] == "active"
    # No write clause may ever appear in any retrieval query.
    for recorded in driver.calls:
        assert recorded["routing_"] is RoutingControl.READ
        text = recorded["query"].upper()
        assert not any(c in text for c in (" MERGE ", " CREATE ", " SET ", " DELETE "))
    assert "SESSION" not in call["parameters"]["private_scopes"]


async def test_empty_result_explains_itself() -> None:
    driver = StubDriver([{"considered": 0, "rows": []}])
    result = await retriever_for(driver).retrieve("U-unknown")

    assert result.items == ()
    assert any("no active context stored" in note for note in result.debug.notes)


async def test_graph_failures_become_clear_retrieval_errors() -> None:
    driver = StubDriver(neo4j_exceptions.ClientError("boom"))
    with pytest.raises(RetrievalError, match="context retrieval failed"):
        await retriever_for(driver).retrieve("U1")


async def test_agent_prompt_carries_the_retrieved_context() -> None:
    transport = StubTransport()
    retrieval = sample_retrieval(sample_item(), sample_item("tone", "formal"))

    reply = await agent_for(transport).reply(retrieval=retrieval, history=(), message="hi")

    system = transport.payloads[0]["messages"][0]["content"]
    assert "response_detail: concise" in system and "tone: formal" in system
    assert reply.text == "hello there"
    assert reply.lookups == ()
    # Free-form chat: the agent must not constrain the reply to a schema.
    assert "response_format" not in transport.payloads[0]


async def test_agent_reports_when_no_context_was_retrieved() -> None:
    transport = StubTransport()
    await agent_for(transport).reply(retrieval=sample_retrieval(), history=(), message="hi")
    assert "no stored context was retrieved" in transport.payloads[0]["messages"][0]["content"]


async def test_agent_sends_only_recent_history() -> None:
    transport = StubTransport()
    history = [
        AgentSession.start(
            session_id="DA-U1-x", user_id="U1", model="gpt-test", retrieval=sample_retrieval()
        )
        .with_message(Role.USER, f"turn {index}")
        .turns[0]
        for index in range(10)
    ]

    await agent_for(transport, max_history_turns=4).reply(
        retrieval=sample_retrieval(), history=history, message="latest"
    )

    messages = transport.payloads[0]["messages"]
    assert len(messages) == 1 + 4 + 1  # system + history window + current message
    assert messages[1]["content"] == "turn 6"


async def test_empty_model_reply_is_an_error_not_a_blank_turn() -> None:
    with pytest.raises(AgentError):
        await agent_for(StubTransport(content=None)).reply(
            retrieval=sample_retrieval(), history=(), message="hi"
        )


async def test_session_is_saved_on_start_and_after_every_turn(tmp_path: Path) -> None:
    retriever = StubRetriever(sample_retrieval(sample_item()))
    controller = controller_for(tmp_path, StubTransport(), retriever)
    store = SessionStore(tmp_path)

    session = await controller.start_session("U1")
    assert retriever.calls == ["U1"]
    assert controller.saved_to is not None
    on_disk = store.load(controller.saved_to)
    assert on_disk.session_id == session.session_id
    assert on_disk.turns == ()
    assert [item.concept_key for item in on_disk.retrieved_context] == [
        "personalization:response_detail"
    ]

    await controller.send("what do you know about me?")
    on_disk = store.load(controller.saved_to)
    assert [turn.role for turn in on_disk.turns] == ["user", "agent"]
    assert on_disk.turns[1].text == "hello there"


async def test_new_session_for_the_same_user_re_runs_retrieval(tmp_path: Path) -> None:
    retriever = StubRetriever(sample_retrieval(sample_item()))
    controller = controller_for(tmp_path, StubTransport(), retriever)

    first = await controller.start_session("U1")
    await controller.send("hello")
    second = await controller.start_session("U1")

    assert retriever.calls == ["U1", "U1", "U1"]  # start, message, start
    assert first.session_id != second.session_id or second.turns == ()
    assert second.turns == ()  # a new session starts empty
    assert len(SessionStore(tmp_path).sessions_for("U1")) >= 1


async def test_ending_a_session_marks_it_closed(tmp_path: Path) -> None:
    controller = controller_for(tmp_path, StubTransport(), StubRetriever(sample_retrieval()))
    await controller.start_session("U1")

    path = controller.end_session()

    assert path is not None
    assert SessionStore(tmp_path).load(path).ended_at is not None


async def test_actions_without_a_session_fail_clearly(tmp_path: Path) -> None:
    controller = controller_for(tmp_path, StubTransport(), StubRetriever(sample_retrieval()))
    with pytest.raises(RetrievalTesterError, match="no session is active"):
        await controller.send("hello")


async def test_empty_input_is_rejected(tmp_path: Path) -> None:
    controller = controller_for(tmp_path, StubTransport(), StubRetriever(sample_retrieval()))
    with pytest.raises(RetrievalTesterError, match="user_id must not be empty"):
        await controller.start_session("   ")
    await controller.start_session("U1")
    with pytest.raises(RetrievalTesterError, match="message must not be empty"):
        await controller.send("  ")


def test_session_store_round_trip_and_listing(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    session = AgentSession.start(
        session_id="DA-U1-20260913T090000Z",
        user_id="U1",
        model="gpt-test",
        retrieval=sample_retrieval(sample_item()),
    ).with_message(Role.USER, "hi")

    path = store.save(session)

    assert path.exists()
    assert store.load(path) == session
    assert store.sessions_for("U1") == [path]
    assert store.sessions_for("U2") == []


def test_saved_session_is_valid_input_for_the_ingestion_tool(tmp_path: Path) -> None:
    """Requirement: the file can be ingested later, by a deliberate manual step."""
    from context_graph.tool import IngestionRequest  # test-only: the app never imports this

    store = SessionStore(tmp_path)
    session = (
        AgentSession.start(
            session_id="DA-U1-20260913T090000Z",
            user_id="U1",
            model="gpt-test",
            retrieval=sample_retrieval(sample_item()),
        )
        .with_message(Role.USER, "keep my answers short")
        .with_message(Role.AGENT, "understood")
        .closed()
    )
    path = store.save(session)

    document = json.loads(path.read_text(encoding="utf-8"))
    request = IngestionRequest.model_validate(document)

    assert request.session.session_id == "DA-U1-20260913T090000Z"
    assert [message.role for message in request.session.messages] == [Role.USER, Role.AGENT]
    assert [message.sequence for message in request.session.messages] == [1, 2]
    assert request.session.state is SessionState.CLOSED
    assert request.session.metadata["source"] == "retrieval_tester"
    # The agent's own view is kept alongside, and ignored by ingestion.
    assert document["retrieval_tester"]["retrieved_context"][0]["concept_key"]


#: Ingestion-side modules this application must never reach.
FORBIDDEN_IMPORTS = (
    "context_graph.tool",
    "context_graph.pipeline",
    "context_graph.ledger",
    "context_graph.graph.neo4j_store",
)


def test_retrieval_tester_never_imports_the_ingestion_pipeline() -> None:
    """The Context Graph is read-only here: no path back into ingestion."""
    sources = [*(REPO_ROOT / "retrieval_tester").glob("*.py"), *(REPO_ROOT / "ui").glob("*.py")]
    assert sources

    for source in sources:
        tree = ast.parse(source.read_text(encoding="utf-8"))
        imported: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module)
        offending = [
            name
            for name in imported
            for forbidden in FORBIDDEN_IMPORTS
            if name == forbidden or name.startswith(f"{forbidden}.")
        ]
        assert not offending, f"{source.name} imports ingestion code: {offending}"


def test_retrieval_query_contains_no_write_clause() -> None:
    query = retrieval_module._RETRIEVE.upper()  # pyright: ignore[reportPrivateUsage]
    assert query.count("MATCH") >= 1
    assert not any(clause in query for clause in ("MERGE", "CREATE", "DELETE", " SET "))


def test_retry_settings_are_shared_with_the_pipeline() -> None:
    assert isinstance(RetrySettings(), RetrySettings)


def test_runtime_shares_one_loop_and_closes_cleanly() -> None:
    """The UIs drive async orchestration through one long-lived loop."""
    from retrieval_tester.config import RetrievalTesterConfig
    from retrieval_tester.runtime import AppRuntime

    config = RetrievalTesterConfig(
        neo4j=Neo4jSettings(),
        chat=chat_settings(),
        embedding=EmbeddingSettings(),
        retry=RetrySettings(),
        agent=AgentSettings(),
    )
    runtime = AppRuntime(config)

    async def identity(value: int) -> int:
        return value

    try:
        # Connections are lazy, so no network is touched here.
        assert runtime.run(identity(7)) == 7
        assert runtime.run(identity(8)) == 8
        assert runtime.controller is not None
    finally:
        runtime.close()


def session_row(session_id: str, concepts: list[str], episodes: int = 4) -> dict[str, Any]:
    return {
        "session_id": session_id,
        "state": "closed",
        "started": NOW,
        "ended": NOW,
        "episode_count": episodes,
        "opening_message": "hi there",
        "concepts": [f"personalization:{concept}" for concept in concepts],
    }


def change_row(concept: str, previous: str, current: str) -> dict[str, Any]:
    return {
        "concept_key": f"personalization:{concept}",
        "previous_value": previous,
        "current_value": current,
        "changed_at": NOW,
    }


async def test_earlier_conversations_are_retrieved_for_episodic_recall() -> None:
    driver = StubDriver(
        [{"considered": 0, "rows": []}],
        sessions=[session_row("S-1", ["coffee_preference", "vehicle_interest"])],
    )

    result = await retriever_for(driver).retrieve("U1")

    assert len(result.sessions) == 1
    recall = result.sessions[0]
    assert recall.session_id == "S-1"
    assert recall.concepts == ("coffee_preference", "vehicle_interest")
    assert recall.episode_count == 4
    assert "Earlier conversations" in result.as_prompt_block()
    assert result.debug.sessions_recalled == 1


async def test_superseded_values_are_retrieved_as_history() -> None:
    driver = StubDriver(
        [{"considered": 0, "rows": []}],
        changes=[change_row("response_detail", "verbose", "concise")],
    )

    result = await retriever_for(driver).retrieve("U1")

    assert result.history[0].previous_value == "verbose"
    assert result.history[0].current_value == "concise"
    assert "previously verbose, now concise" in result.as_prompt_block()


class StubEmbedder:
    """Returns a fixed query vector, so similarity is predictable."""

    def __init__(self, vector: list[float]) -> None:
        self._vector = vector

    @property
    def model(self) -> str:
        return "stub-embedder"

    @property
    def dimension(self) -> int:
        return len(self._vector)

    async def embed(self, texts: Any) -> list[list[float]]:
        return [self._vector for _ in texts]


def scored_row(concept: str, context_type: str, embedding: list[float]) -> dict[str, Any]:
    node = context_node(concept)
    node["context_type"] = context_type
    return {"node": node, "quote": None, "embedding": embedding}


async def test_relevance_orders_by_similarity_and_keeps_standing_instructions() -> None:
    driver = StubDriver(
        [
            {
                "considered": 3,
                "rows": [
                    scored_row("response_detail", "Preference", [0.0, 1.0]),  # standing, unrelated
                    scored_row("vehicle_interest", "Interest", [0.1, 0.995]),  # weak match
                    scored_row("coffee_preference", "Interest", [1.0, 0.0]),  # strong match
                ],
            }
        ]
    )
    retriever = retriever_for(driver, embeddings=StubEmbedder([1.0, 0.0]))

    result = await retriever.retrieve("U1", "tell me about coffee")

    assert result.debug.relevance_used is True
    assert result.debug.query == "tell me about coffee"
    concepts = [item.concept for item in result.items]
    # Standing instruction first, then topical items by similarity.
    assert concepts[0] == "response_detail"
    assert concepts[1] == "coffee_preference"
    assert result.items[1].similarity == pytest.approx(1.0)


async def test_nothing_is_dropped_while_it_all_fits() -> None:
    """Relevance reorders; it only excludes when there is too much to send."""
    driver = StubDriver(
        [
            {
                "considered": 2,
                "rows": [
                    scored_row("coffee_preference", "Interest", [1.0, 0.0]),
                    scored_row("vehicle_interest", "Interest", [0.0, 1.0]),  # similarity 0.0
                ],
            }
        ]
    )
    retriever = retriever_for(driver, max_items=10, embeddings=StubEmbedder([1.0, 0.0]))

    result = await retriever.retrieve("U1", "coffee")

    assert {item.concept for item in result.items} == {"coffee_preference", "vehicle_interest"}


async def test_irrelevant_context_is_dropped_only_when_over_the_limit() -> None:
    driver = StubDriver(
        [
            {
                "considered": 2,
                "rows": [
                    scored_row("coffee_preference", "Interest", [1.0, 0.0]),
                    scored_row("vehicle_interest", "Interest", [0.0, 1.0]),
                ],
            }
        ]
    )
    retriever = retriever_for(driver, max_items=1, embeddings=StubEmbedder([1.0, 0.0]))

    result = await retriever.retrieve("U1", "coffee")

    assert [item.concept for item in result.items] == ["coffee_preference"]
    assert result.debug.truncated is True


async def test_message_retrieval_runs_per_turn(tmp_path: Path) -> None:
    retriever = StubRetriever(sample_retrieval(sample_item()))
    controller = controller_for(tmp_path, StubTransport(), retriever)

    await controller.start_session("U1")
    await controller.send("what do i like?")

    # Once at session start with no query, once for the message itself.
    assert retriever.queries == [None, "what do i like?"]


def directory_row(colleagues: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "team_id": "search",
        "team_name": "Search",
        "department_name": "Engineering",
        "organization_name": "Acme Corp",
        "team_projects": ["Apollo"],
        "my_projects": ["Apollo"],
        "colleagues": colleagues
        if colleagues is not None
        else [
            {"user_id": "U-aman-verma", "name": "Aman Verma", "title": "Engineer", "projects": ["Apollo"]},
            {"user_id": "U003", "name": "Sofia Duarte", "title": "Tech Lead", "projects": []},
        ],
    }


async def test_directory_is_retrieved_so_team_questions_can_be_answered() -> None:
    driver = StubDriver([{"considered": 0, "rows": []}], directory=[directory_row()])

    result = await retriever_for(driver).retrieve("U-praveen")

    directory = result.directory
    assert directory.is_known
    assert directory.team_name == "Search"
    assert directory.department_name == "Engineering"
    assert [c.name for c in directory.colleagues] == ["Aman Verma", "Sofia Duarte"]
    assert directory.colleagues[0].projects == ("Apollo",)
    assert result.debug.directory_found is True

    block = result.as_prompt_block()
    assert "Search team" in block and "Aman Verma" in block


async def test_directory_exposes_only_shared_facts_about_colleagues() -> None:
    """Names, roles and projects are shared; preferences are never included."""
    driver = StubDriver([{"considered": 0, "rows": []}], directory=[directory_row()])

    result = await retriever_for(driver).retrieve("U-praveen")

    colleague = result.directory.colleagues[0]
    assert set(colleague.model_dump()) == {"user_id", "name", "title", "projects"}


async def test_user_without_a_team_gets_no_directory() -> None:
    driver = StubDriver([{"considered": 0, "rows": []}], directory=[])

    result = await retriever_for(driver).retrieve("U-unknown")

    assert result.directory.is_known is False
    assert result.directory.as_prompt_block() == ""
    assert result.debug.directory_found is False


async def test_shared_scope_follows_the_team_department_organization_chain() -> None:
    driver = StubDriver([{"considered": 0, "rows": []}])
    await retriever_for(driver).retrieve("U1")

    context_query = next(c["query"] for c in driver.calls if "$private_scopes" in c["query"])
    # A user reaches department and organization scope through their team.
    assert "MEMBER_OF]->(:Team)" in context_query
    assert "BELONGS_TO*0..3" in context_query
