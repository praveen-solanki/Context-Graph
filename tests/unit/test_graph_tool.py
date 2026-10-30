"""The agent's read-only graph lookup tool."""

from __future__ import annotations

from typing import Any

import pytest
from neo4j import RoutingControl

from context_graph.config import Neo4jSettings
from retrieval_tester.agent import RetrievalAgent
from retrieval_tester.config import AgentSettings
from retrieval_tester.errors import AgentError
from retrieval_tester.graph_tools import TOOL_NAME, TOOL_SCHEMA, GraphLookupTool, Lookup
from retrieval_tester.models import RetrievalDebug, RetrievalResult
from tests.agent_doubles import chat_settings


class StubDriver:
    def __init__(self, rows: list[dict[str, Any]] | None = None) -> None:
        self._rows = rows or []
        self.calls: list[dict[str, Any]] = []

    async def execute_query(self, query: str, parameters: dict[str, Any], **kwargs: Any) -> Any:
        self.calls.append({"query": query, "parameters": parameters, **kwargs})
        return type("Result", (), {"records": [dict(row) for row in self._rows]})()

    async def close(self) -> None:
        return None


def tool_for(rows: list[dict[str, Any]] | None = None) -> tuple[GraphLookupTool, StubDriver]:
    driver = StubDriver(rows)
    return GraphLookupTool(Neo4jSettings(), driver), driver  # pyright: ignore[reportArgumentType]


async def test_teams_lookup_needs_no_target_and_renders_rows() -> None:
    tool, driver = tool_for(
        [
            {"team": "Search", "department": "Engineering", "organization": "Acme", "members": 4},
            {"team": "Payments", "department": "Engineering", "organization": "Acme", "members": 3},
        ]
    )

    call = await tool.run(Lookup.TEAMS.value, "")

    assert call.rows == 2
    assert "Search (Engineering, 4 members)" in call.result
    assert driver.calls[0]["routing_"] is RoutingControl.READ


async def test_lookup_that_needs_a_name_refuses_without_one() -> None:
    tool, driver = tool_for()

    call = await tool.run(Lookup.TEAM_MEMBERS.value, "  ")

    assert "needs a name" in call.result
    assert driver.calls == []  # nothing was queried


async def test_unknown_lookup_is_reported_not_raised() -> None:
    tool, driver = tool_for()

    call = await tool.run("delete_everything", "x")

    assert "unknown lookup" in call.result
    assert driver.calls == []


async def test_no_match_is_stated_plainly() -> None:
    tool, _ = tool_for([])
    call = await tool.run(Lookup.PERSON.value, "nobody")
    assert call.result == "no match found"


async def test_every_lookup_is_read_only_and_parameterised() -> None:
    for lookup in Lookup:
        tool, driver = tool_for([])
        await tool.run(lookup.value, "x")
        query = driver.calls[0]["query"].upper()
        assert not any(clause in query for clause in ("MERGE", "CREATE", "DELETE", " SET "))
        # The model supplies a name, never a query fragment.
        assert "$TARGET" in query or lookup is Lookup.TEAMS


async def test_tool_cannot_reach_personal_context() -> None:
    """The lookup tool sees the directory only; Context is retrieval's job."""
    for lookup in Lookup:
        tool, driver = tool_for([])
        await tool.run(lookup.value, "x")
        assert "Context" not in driver.calls[0]["query"]
        assert "Episode" not in driver.calls[0]["query"]


class StubTransport:
    """Replays scripted model responses and records what was sent."""

    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self._responses = responses
        self.payloads: list[dict[str, Any]] = []

    async def post(self, path: str, payload: dict[str, Any], label: str) -> dict[str, Any]:
        self.payloads.append(payload)
        return {"choices": [self._responses[min(len(self.payloads) - 1, len(self._responses) - 1)]]}


def text_response(content: str) -> dict[str, Any]:
    return {"finish_reason": "stop", "message": {"role": "assistant", "content": content}}


def tool_response(lookup: str, target: str) -> dict[str, Any]:
    return {
        "finish_reason": "tool_calls",
        "message": {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": TOOL_NAME,
                        "arguments": f'{{"lookup": "{lookup}", "target": "{target}"}}',
                    },
                }
            ],
        },
    }


def empty_retrieval() -> RetrievalResult:
    return RetrievalResult(
        debug=RetrievalDebug(user_id="U1", duration_ms=0.0, candidates_considered=0, returned=0)
    )


def agent_for(
    transport: Any, tool: GraphLookupTool | None, *, token_parameter: str = "max_tokens", **overrides: Any
) -> RetrievalAgent:
    return RetrievalAgent(
        transport=transport,
        backend=chat_settings().model_copy(update={"token_parameter": token_parameter}),
        settings=AgentSettings(**overrides),
        graph_tool=tool,
    )


async def test_agent_runs_the_lookup_and_answers_from_its_result() -> None:
    tool, _ = tool_for([{"team": "Payments", "department": "Engineering", "organization": "Acme", "members": 3}])
    transport = StubTransport([tool_response("teams", ""), text_response("Payments and Search.")])

    reply = await agent_for(transport, tool).reply(
        retrieval=empty_retrieval(), history=(), message="what teams exist?"
    )

    assert reply.text == "Payments and Search."
    assert [call.lookup for call in reply.lookups] == ["teams"]
    # The lookup result was fed back to the model as a tool message.
    assert transport.payloads[1]["messages"][-1]["role"] == "tool"
    assert "Payments" in transport.payloads[1]["messages"][-1]["content"]


async def test_tool_is_not_offered_when_switched_off() -> None:
    tool, driver = tool_for()
    transport = StubTransport([text_response("no lookups here")])

    reply = await agent_for(transport, tool, enable_graph_tool=False).reply(
        retrieval=empty_retrieval(), history=(), message="what teams exist?"
    )

    assert "tools" not in transport.payloads[0]
    assert reply.lookups == ()
    assert driver.calls == []


async def test_tool_is_offered_when_switched_on() -> None:
    tool, _ = tool_for()
    transport = StubTransport([text_response("hello")])

    await agent_for(transport, tool).reply(
        retrieval=empty_retrieval(), history=(), message="hi"
    )

    assert transport.payloads[0]["tools"] == [TOOL_SCHEMA]


@pytest.mark.parametrize("token_parameter", ["max_tokens", "max_completion_tokens"])
async def test_the_reply_cap_uses_the_configured_token_parameter(token_parameter: str) -> None:
    tool, _ = tool_for()
    transport = StubTransport([text_response("hi")])

    await agent_for(transport, tool, token_parameter=token_parameter, max_output_tokens=321).reply(
        retrieval=empty_retrieval(), history=(), message="hi"
    )

    other = {"max_tokens", "max_completion_tokens"} - {token_parameter}
    assert transport.payloads[0][token_parameter] == 321
    assert not other & transport.payloads[0].keys()


async def test_endless_lookups_are_stopped() -> None:
    tool, _ = tool_for([])
    transport = StubTransport([tool_response("teams", "")])

    with pytest.raises(AgentError, match="kept asking for lookups"):
        await agent_for(transport, tool, max_tool_iterations=2).reply(
            retrieval=empty_retrieval(), history=(), message="loop forever"
        )
