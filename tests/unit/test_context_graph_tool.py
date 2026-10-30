"""Planned Context Graph retrieval: schema discovery, levels, guards, failures."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from neo4j import RoutingControl
from neo4j import exceptions as neo4j_exceptions
from pydantic import BaseModel

from context_graph.config import Neo4jSettings
from context_graph.errors import ProviderError
from retrieval_tester.config import AgentSettings
from retrieval_tester.context_tool import (
    TOOL_NAME,
    TOOL_SCHEMA,
    AllowedScopes,
    ContextGraphRetrievalTool,
    Hop,
    SchemaFocus,
    GraphEdge,
    GraphNode,
    TraceStep,
    TraversalGraph,
    TraversalPlan,
    compile_plan,
    repair_directions,
    validate_plan,
)
from retrieval_tester.errors import RetrievalError
from retrieval_tester.models import RetrievalDebug, RetrievalResult
from retrieval_tester.schema_catalog import Edge, GraphSchema, SchemaCatalog
from tests.factories import REPO_ROOT, context_node

# -- test doubles ---------------------------------------------------------

#: The schema the stub database reports, mirroring the real ontology.
EDGES = (
    Edge(start="User", type="MEMBER_OF", end="Team"),
    Edge(start="Team", type="BELONGS_TO", end="Department"),
    Edge(start="Context", type="APPLIES_TO", end="Team"),
    Edge(start="Context", type="APPLIES_TO", end="User"),
    Edge(start="Context", type="SUPPORTED_BY", end="Episode"),
    Edge(start="Session", type="CONTAINS", end="Episode"),
    Edge(start="Session", type="INVOLVES_AGENT", end="Agent"),
    Edge(start="User", type="PARTICIPATED_IN", end="Session"),
)
SCHEMA = GraphSchema(
    properties={
        "User": ("name", "title", "user_id"),
        "Team": ("name", "team_id"),
        "Department": ("department_id", "name"),
        "Episode": ("episode_id", "text"),
        "Session": ("session_id", "state"),
        "Agent": ("agent_id",),
        "Context": ("context_id", "context_type", "scope_id", "scope_type", "value"),
    },
    edges=EDGES,
)


class SchemaRow:
    """A ``db.schema.visualization()`` relationship, as the driver returns it."""

    def __init__(self, start: str, kind: str, end: str) -> None:
        self.type = kind
        self.start_node = type("Node", (), {"labels": frozenset({start})})()
        self.end_node = type("Node", (), {"labels": frozenset({end})})()


class StubResult:
    def __init__(self, records: list[dict[str, Any]]) -> None:
        self.records = records


class StubDriver:
    """Answers each query by what it asks for, and records every call."""

    def __init__(
        self,
        *,
        traversal: list[dict[str, Any]] | Exception | None = None,
        shared: list[list[str]] | None = None,
        user_exists: bool = True,
        delay: float = 0.0,
    ) -> None:
        self._traversal = traversal if traversal is not None else []
        self._shared = shared or []
        self._user_exists = user_exists
        self._delay = delay
        self.calls: list[dict[str, Any]] = []

    async def execute_query(
        self, query: str, parameters: dict[str, Any], **kwargs: Any
    ) -> StubResult:
        self.calls.append({"query": query, "parameters": parameters, **kwargs})
        if self._delay:
            await asyncio.sleep(self._delay)
        if "nodeTypeProperties" in query:
            return StubResult(
                [
                    {"labels": [label], "property": name}
                    for label, names in SCHEMA.properties.items()
                    for name in names
                ]
                + [{"labels": ["Context"], "property": "embedding"}]
            )
        if "visualization" in query:
            return StubResult(
                [{"relationships": [SchemaRow(e.start, e.type, e.end) for e in EDGES]}]
            )
        if "AS shared" in query:
            return StubResult([{"shared": self._shared}] if self._user_exists else [])
        if isinstance(self._traversal, Exception):
            raise self._traversal
        return StubResult(self._traversal)


class StubLLM:
    """Replays scripted structured replies and records the prompts it saw."""

    def __init__(self, replies: list[Any]) -> None:
        self._replies = list(replies)
        self.prompts: list[tuple[str, str]] = []

    @property
    def model(self) -> str:
        return "stub"

    def estimate_tokens(self, text: str) -> int:
        return len(text)

    async def count_tokens(self, *, system: str, user: str) -> int:
        return len(system) + len(user)

    async def structured[M: BaseModel](
        self, *, system: str, user: str, schema: type[M], max_output_tokens: int
    ) -> M:
        self.prompts.append((system, user))
        if not self._replies:
            raise AssertionError("the planner asked for more levels than were scripted")
        reply = self._replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return schema.model_validate(reply.model_dump())


class StubFallback:
    """Stands in for the existing retriever."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str | None]] = []

    async def retrieve(self, user_id: str, query: str | None = None) -> RetrievalResult:
        self.calls.append((user_id, query))
        return RetrievalResult(
            debug=RetrievalDebug(
                user_id=user_id,
                duration_ms=0.0,
                candidates_considered=0,
                returned=0,
                notes=("standard retrieval",),
            )
        )


def tool_for(
    replies: list[Any],
    driver: StubDriver | None = None,
    **overrides: Any,
) -> tuple[ContextGraphRetrievalTool, StubDriver, StubLLM, StubFallback]:
    stub_driver = driver or StubDriver()
    llm, fallback = StubLLM(replies), StubFallback()
    settings = AgentSettings(**overrides)
    tool = ContextGraphRetrievalTool(
        Neo4jSettings(),
        stub_driver,  # pyright: ignore[reportArgumentType]
        llm=llm,
        fallback=fallback,
        agent_settings=settings,
        catalog=SchemaCatalog(
            Neo4jSettings(),
            stub_driver,  # pyright: ignore[reportArgumentType]
            cache_seconds=settings.schema_cache_seconds,
            timeout_seconds=settings.graph_timeout_seconds,
        ),
    )
    return tool, stub_driver, llm, fallback


def rows(*nodes: dict[str, Any], walked: bool = False) -> list[dict[str, Any]]:
    """Traversal rows, each carrying the path the stub 'walked' to reach it."""
    built: list[dict[str, Any]] = []
    for node in nodes:
        context = {"id": node["context_id"], "label": "Context", "caption": node["value"]}
        path = [{"id": "u1", "label": "User", "caption": "Riya"}, context] if walked else [context]
        edges = (
            [{"type": "APPLIES_TO", "source": node["context_id"], "target": "u1"}]
            if walked
            else []
        )
        built.append(
            {"node": node, "quote": "because I said so", "path_nodes": path, "path_edges": edges}
        )
    return built


# -- schema discovery -----------------------------------------------------


async def test_schema_is_read_from_the_database_not_assumed() -> None:
    driver = StubDriver()
    catalog = SchemaCatalog(Neo4jSettings(), driver, cache_seconds=60.0, timeout_seconds=5.0)  # pyright: ignore[reportArgumentType]

    schema = await catalog.get()

    assert set(schema.labels) == set(SCHEMA.properties)
    assert schema.has_edge("User", "MEMBER_OF", "Team")
    assert not schema.has_edge("Team", "MEMBER_OF", "User")  # direction matters
    assert all(call["routing_"] is RoutingControl.READ for call in driver.calls)


async def test_embeddings_are_never_offered_to_the_model() -> None:
    catalog = SchemaCatalog(Neo4jSettings(), StubDriver(), cache_seconds=60.0, timeout_seconds=5.0)  # pyright: ignore[reportArgumentType]
    schema = await catalog.get()
    assert not schema.has_property("Context", "embedding")
    assert "embedding" not in schema.detail({"Context"})


async def test_schema_is_cached_and_reused() -> None:
    driver = StubDriver()
    catalog = SchemaCatalog(Neo4jSettings(), driver, cache_seconds=300.0, timeout_seconds=5.0)  # pyright: ignore[reportArgumentType]

    await catalog.get()
    await catalog.get()

    assert len(driver.calls) == 2  # two discovery queries, once


async def test_schema_discovery_timeout_is_reported() -> None:
    driver = StubDriver(delay=0.05)
    catalog = SchemaCatalog(Neo4jSettings(), driver, cache_seconds=60.0, timeout_seconds=0.01)  # pyright: ignore[reportArgumentType]

    with pytest.raises(RetrievalError, match="timed out"):
        await catalog.get()


def test_level_one_view_hides_properties_and_level_two_shows_only_what_was_chosen() -> None:
    outline = SCHEMA.outline()
    assert "(User)-[:MEMBER_OF]->(Team)" in outline
    assert "user_id" not in outline

    detail = SCHEMA.detail({"Team", "Context"})
    assert "(Team) properties: name, team_id" in detail
    assert "(User) properties" not in detail


# -- level-by-level planning ---------------------------------------------


async def test_a_question_about_the_user_stops_after_one_level() -> None:
    tool, _, llm, _ = tool_for(
        [SchemaFocus(answered_by_user_context=True, keywords=["concise"], reason="own context")],
        StubDriver(traversal=rows(context_node())),
    )

    result = await tool.run("how should you write to me?", "U1")

    assert len(llm.prompts) == 1  # no second call was needed
    assert "planned in 1 level(s)" in " ".join(result.debug.notes)
    assert [item.value for item in result.items] == ["concise"]
    # Level 1 saw the outline only.
    assert "Node labels:" in llm.prompts[0][1]


async def test_a_traversal_question_adds_a_second_level_with_deeper_schema() -> None:
    plan = TraversalPlan(
        anchor_label="User",
        anchor_property="user_id",
        anchor_value="U1",
        hops=[Hop(relationship="MEMBER_OF", direction="out", target_label="Team"),
              Hop(relationship="APPLIES_TO", direction="in", target_label="Context")],
        reason="team context",
    )
    tool, _, llm, _ = tool_for(
        [SchemaFocus(labels=["User", "Team"], keywords=["deploy"]), plan],
        StubDriver(traversal=rows(context_node(scope_type="TEAM", scope_id="T1")), shared=[["TEAM", "T1"]]),
    )

    result = await tool.run("what did my team decide about deploys?", "U1")

    assert len(llm.prompts) == 2
    # Level 2 was given properties, and only for the labels level 1 chose.
    assert "(Team) properties" in llm.prompts[1][1]
    assert "(Episode) properties" not in llm.prompts[1][1]
    assert "planned in 2 level(s)" in " ".join(result.debug.notes)
    assert result.items[0].scope_type.value == "TEAM"


async def test_a_third_level_runs_only_when_the_plan_asks_to_go_deeper() -> None:
    shallow = TraversalPlan(
        anchor_label="User",
        hops=[Hop(relationship="MEMBER_OF", direction="out", target_label="Team")],
        needs_deeper_traversal=True,
    )
    complete = TraversalPlan(
        anchor_label="User",
        hops=[
            Hop(relationship="MEMBER_OF", direction="out", target_label="Team"),
            Hop(relationship="APPLIES_TO", direction="in", target_label="Context"),
        ],
    )
    tool, _, llm, _ = tool_for(
        [SchemaFocus(labels=["User"]), shallow, complete],
        StubDriver(traversal=rows(context_node()), shared=[["TEAM", "T1"]]),
    )

    result = await tool.run("what does my department expect?", "U1")

    assert len(llm.prompts) == 3
    assert "planned in 3 level(s)" in " ".join(result.debug.notes)


async def test_planning_levels_are_capped() -> None:
    needy = TraversalPlan(
        anchor_label="Context", needs_deeper_traversal=True
    )
    tool, _, llm, _ = tool_for(
        [SchemaFocus(labels=["Context"]), needy], planner_max_levels=2
    )

    await tool.run("something vague", "U1")

    assert len(llm.prompts) == 2  # the third level was never requested


# -- plan validation ------------------------------------------------------


def _scopes() -> AllowedScopes:
    return AllowedScopes(user_exists=True, shared=(("TEAM", "T1"),))


@pytest.mark.parametrize(
    ("plan", "expected"),
    [
        (TraversalPlan(anchor_label="Wizard"), "unknown label"),
        (
            TraversalPlan(anchor_label="Context", anchor_property="secret", anchor_value="x"),
            "no property",
        ),
        (
            TraversalPlan(
                anchor_label="User",
                hops=[Hop(relationship="OWNS", direction="out", target_label="Context")],
            ),
            "no OWNS between",
        ),
        (
            TraversalPlan(
                anchor_label="User",
                hops=[Hop(relationship="MEMBER_OF", direction="in", target_label="Team")],
            ),
            "no MEMBER_OF between",
        ),
        (
            TraversalPlan(
                anchor_label="User",
                hops=[Hop(relationship="MEMBER_OF", direction="out", target_label="Team")],
            ),
            "never reaches a Context",
        ),
    ],
)
def test_invalid_plans_are_rejected_against_the_real_schema(
    plan: TraversalPlan, expected: str
) -> None:
    assert expected in (validate_plan(plan, SCHEMA, AgentSettings()) or "")


def test_a_filter_the_user_may_not_read_is_dropped_not_obeyed() -> None:
    """The guard decides access; an unreadable or invented filter is removed."""
    from retrieval_tester.context_tool import _sanitise_filters  # pyright: ignore[reportPrivateUsage]

    plan = TraversalPlan(
        anchor_label="Context",
        scope_types=["ORGANIZATION", "TEAM", "SESSION"],
        context_types=["Preference", "status"],
    )
    cleaned = _sanitise_filters(plan, _scopes())

    assert cleaned.scope_types == ["TEAM"]  # no ORGANIZATION membership, never SESSION
    assert cleaned.context_types == ["Preference"]
    # ORGANIZATION is unreachable regardless of what the plan asked for.
    _, parameters = compile_plan(cleaned, "U1", _scopes(), AgentSettings())
    assert parameters["allowed_shared"] == [["TEAM", "T1"]]


def test_a_plan_within_the_schema_and_the_user_s_scopes_is_accepted() -> None:
    plan = TraversalPlan(
        anchor_label="User",
        anchor_property="user_id",
        anchor_value="U1",
        hops=[
            Hop(relationship="MEMBER_OF", direction="out", target_label="Team"),
            Hop(relationship="APPLIES_TO", direction="in", target_label="Context"),
        ],
        scope_types=["TEAM", "USER"],
    )
    assert validate_plan(plan, SCHEMA, AgentSettings()) is None


def test_traversal_depth_is_bounded() -> None:
    plan = TraversalPlan(
        anchor_label="User",
        hops=[Hop(relationship="MEMBER_OF", direction="out", target_label="Team")] * 4,
    )
    assert "exceeds the limit" in (validate_plan(plan, SCHEMA, AgentSettings(planner_max_hops=2)) or "")


async def test_an_invalid_plan_falls_back_to_standard_retrieval() -> None:
    tool, _, _, fallback = tool_for(
        [SchemaFocus(labels=["User"]), TraversalPlan(anchor_label="Wizard")]
    )

    result = await tool.run("who is on my team?", "U1")

    assert fallback.calls == [("U1", "who is on my team?")]
    assert "planned retrieval unavailable" in result.debug.notes[0]
    assert "rejected plan" not in " ".join(result.debug.notes)  # it never ran


# -- compiled query: lifecycle, authorization, bounds ---------------------


def _compiled(plan: TraversalPlan, scopes: AllowedScopes | None = None, **overrides: Any) -> tuple[str, dict[str, Any]]:
    return compile_plan(plan, "U1", scopes or _scopes(), AgentSettings(**overrides))


def test_the_compiled_query_is_read_only_and_parameterised() -> None:
    cypher, parameters = _compiled(
        TraversalPlan(anchor_label="Context", anchor_property="scope_id", anchor_value="T1")
    )
    assert not any(word in cypher.upper() for word in ("MERGE", "CREATE", "DELETE", " SET "))
    assert "T1" not in cypher and parameters["anchor_value"] == "T1"
    assert "LIMIT $row_limit" in cypher


def test_lifecycle_and_authorization_are_always_applied() -> None:
    cypher, parameters = _compiled(TraversalPlan(anchor_label="Context"))
    assert "c.status = $active_status" in cypher
    assert "c.valid_until IS NULL OR c.valid_until >= $now" in cypher
    assert "c.subject_user_id = $user_id AND c.scope_type IN $private_scopes" in cypher
    assert "[c.scope_type, c.scope_id] IN $allowed_shared" in cypher
    assert parameters["allowed_shared"] == [["TEAM", "T1"]]
    assert parameters["active_status"] == "active"
    # SESSION belongs to the session it was stated in and is in neither list.
    assert "SESSION" not in parameters["private_scopes"] + parameters["shared_scopes"]


def test_a_user_with_no_membership_can_reach_no_shared_scope() -> None:
    _, parameters = _compiled(TraversalPlan(anchor_label="Context"), AllowedScopes(user_exists=True))
    assert parameters["allowed_shared"] == []


def test_shared_scopes_can_be_switched_off_by_configuration() -> None:
    _, parameters = _compiled(TraversalPlan(anchor_label="Context"), include_shared_scopes=False)
    assert parameters["shared_scopes"] == []


def test_hops_are_compiled_in_the_direction_the_plan_gave() -> None:
    cypher, _ = _compiled(
        TraversalPlan(
            anchor_label="User",
            hops=[
                Hop(relationship="MEMBER_OF", direction="out", target_label="Team"),
                Hop(relationship="APPLIES_TO", direction="in", target_label="Context"),
            ],
        )
    )
    assert "(n0:User)-[:MEMBER_OF]->(n1:Team)<-[:APPLIES_TO]-(n2:Context)" in cypher


# -- execution and failure handling ---------------------------------------


async def test_an_unknown_user_is_reported_and_nothing_is_planned() -> None:
    tool, _, llm, fallback = tool_for([], StubDriver(user_exists=False))

    result = await tool.run("anything", "ghost")

    assert result.items == ()
    assert "no user 'ghost'" in result.debug.notes[0]
    assert llm.prompts == [] and fallback.calls == []


async def test_empty_and_duplicate_results_are_handled() -> None:
    duplicate = context_node()
    tool, _, _, _ = tool_for(
        [SchemaFocus(answered_by_user_context=True)],
        StubDriver(traversal=rows(duplicate, duplicate, context_node("timezone", "CET"))),
    )

    result = await tool.run("what do you know?", "U1")

    assert len(result.items) == 2  # the duplicate was collapsed
    assert result.debug.candidates_considered == 3


async def test_no_match_is_stated_plainly() -> None:
    tool, _, _, _ = tool_for([SchemaFocus(answered_by_user_context=True)], StubDriver(traversal=[]))
    result = await tool.run("what about nothing?", "U1")
    assert result.items == ()
    assert any("matched no Context" in note for note in result.debug.notes)


async def test_a_graph_failure_falls_back_to_standard_retrieval() -> None:
    tool, _, _, fallback = tool_for(
        [SchemaFocus(answered_by_user_context=True)],
        StubDriver(traversal=neo4j_exceptions.Neo4jError("graph is down")),
    )

    result = await tool.run("anything", "U1")

    assert fallback.calls and "planned retrieval unavailable" in result.debug.notes[0]
    assert "rejected plan" in " ".join(result.debug.notes)


async def test_a_model_failure_falls_back_to_standard_retrieval() -> None:
    tool, _, _, fallback = tool_for([ProviderError("model unavailable")])

    result = await tool.run("anything", "U1")

    assert fallback.calls == [("U1", "anything")]
    assert "model unavailable" in result.debug.notes[0]


async def test_planning_timeout_falls_back() -> None:
    class SlowLLM(StubLLM):
        async def structured[M: BaseModel](
            self, *, system: str, user: str, schema: type[M], max_output_tokens: int
        ) -> M:
            await asyncio.sleep(0.2)
            raise AssertionError("should have timed out")

    driver = StubDriver()
    fallback = StubFallback()
    tool = ContextGraphRetrievalTool(
        Neo4jSettings(),
        driver,  # pyright: ignore[reportArgumentType]
        llm=SlowLLM([]),
        fallback=fallback,
        agent_settings=AgentSettings(planner_timeout_seconds=0.01),
    )

    result = await tool.run("anything", "U1")

    assert "timed out" in result.debug.notes[0]
    assert fallback.calls == [("U1", "anything")]


async def test_empty_input_is_refused() -> None:
    tool, _, _, _ = tool_for([])
    for query, user in (("", "U1"), ("q", "  ")):
        with pytest.raises(RetrievalError):
            await tool.run(query, user)


# -- tool registration and UI integration ---------------------------------


def test_the_readable_scopes_match_the_existing_retriever() -> None:
    """Derived from the ontology, so the two paths can never diverge."""
    from retrieval_tester import context_tool, retrieval

    assert getattr(context_tool, "_PRIVATE_SCOPES") == getattr(retrieval, "_PRIVATE_SCOPES")
    assert getattr(context_tool, "_SHARED_SCOPES") == getattr(retrieval, "_SHARED_SCOPES")


def test_the_tool_declares_query_and_user_id() -> None:
    function = TOOL_SCHEMA["function"]
    assert function["name"] == TOOL_NAME
    assert set(function["parameters"]["required"]) == {"query", "user_id"}
    assert function["parameters"]["additionalProperties"] is False


def test_the_runtime_exposes_the_tool_without_changing_the_existing_one() -> None:
    source = (REPO_ROOT / "retrieval_tester" / "runtime.py").read_text()
    assert "self.context_graph_tool = ContextGraphRetrievalTool(" in source
    assert "self.graph_tool = GraphLookupTool(config.neo4j, self._driver)" in source


def test_a_hop_written_backwards_is_corrected_not_invented() -> None:
    """The schema decides direction; a flip is a correction, never a widening."""
    backwards = TraversalPlan(
        anchor_label="Team",
        hops=[Hop(relationship="APPLIES_TO", direction="out", target_label="Context")],
    )
    repaired = repair_directions(backwards, SCHEMA)

    assert repaired.hops[0].direction == "in"
    assert validate_plan(repaired, SCHEMA, AgentSettings()) is None
    # A relationship that exists in neither direction is left alone and rejected.
    invented = TraversalPlan(
        anchor_label="Team",
        hops=[Hop(relationship="OWNS", direction="out", target_label="Context")],
    )
    assert repair_directions(invented, SCHEMA) == invented


def test_the_anchor_is_filtered_before_the_traversal_collapses_to_context() -> None:
    """The anchor variable is out of scope after WITH DISTINCT, so filter first."""
    cypher, _ = _compiled(
        TraversalPlan(
            anchor_label="User",
            anchor_property="user_id",
            anchor_value="U1",
            hops=[Hop(relationship="APPLIES_TO", direction="in", target_label="Context")],
        )
    )
    assert cypher.index("n0.user_id") < cypher.index("collect(trail)[0]")
    assert "CONTAINS toLower($anchor_value)" in cypher


async def test_an_over_narrow_plan_is_retried_without_its_filters() -> None:
    """Filters are the model's guess; the guard is not, so only filters are dropped."""

    class NarrowThenWide(StubDriver):
        def __init__(self) -> None:
            super().__init__(shared=[["TEAM", "T1"]])
            self.traversals: list[dict[str, Any]] = []

        async def execute_query(
            self, query: str, parameters: dict[str, Any], **kwargs: Any
        ) -> StubResult:
            result = await super().execute_query(query, parameters, **kwargs)
            if "path_nodes" not in query:
                return result
            self.traversals.append(parameters)
            if "keywords" in parameters:
                return StubResult([])
            return StubResult(rows(context_node()))

    driver = NarrowThenWide()
    tool, _, _, _ = tool_for(
        [SchemaFocus(answered_by_user_context=True, keywords=["nothing-matches"])], driver
    )

    result = await tool.run("what do I prefer?", "U1")

    assert len(driver.traversals) == 2
    assert any("without them" in note for note in result.debug.notes)
    assert len(result.items) == 1
    # The guard is identical on both attempts: only the filters were dropped.
    for parameters in driver.traversals:
        assert parameters["allowed_shared"] == [["TEAM", "T1"]]
        assert parameters["active_status"] == "active"


# -- retrieval trace ------------------------------------------------------


def stages(trace: tuple[TraceStep, ...]) -> list[str]:
    return [step.stage for step in trace]


async def test_the_trace_records_every_step_of_a_planned_traversal() -> None:
    plan = TraversalPlan(
        anchor_label="User",
        anchor_property="user_id",
        anchor_value="U1",
        hops=[
            Hop(relationship="MEMBER_OF", direction="out", target_label="Team"),
            Hop(relationship="APPLIES_TO", direction="in", target_label="Context"),
        ],
    )
    tool, _, _, _ = tool_for(
        [SchemaFocus(labels=["User", "Team"], keywords=["deploy"]), plan],
        StubDriver(traversal=rows(context_node()), shared=[["TEAM", "T1"]]),
    )

    await tool.run("what did my team decide?", "U1")
    trace = tool.last_trace

    assert stages(trace) == [
        "request",
        "schema discovery",
        "scope resolution",
        "locate the answer in the schema",
        "narrow the schema for level 2",
        "build the traversal plan",
        "validate the plan against the graph and this user's scopes",
        "planned traversal",
        "deduplicate, rank and cap",
    ]
    # Each planning level is attributed to the model and carries its own depth.
    calls = [step for step in trace if step.actor == "llm"]
    assert [step.level for step in calls] == [1, 2]
    # What the model was given, and what it chose, are both kept in full.
    assert "Node labels:" in calls[0].given and "[system]" in calls[0].given
    assert '"answered_by_user_context"' in calls[0].returned
    assert "(Team) properties" in calls[1].given
    assert '"anchor_label": "User"' in calls[1].returned
    # The executed query and its parameters are kept verbatim.
    executed = next(step for step in trace if step.stage == "planned traversal")
    assert "MATCH trail = (n0:User)-[:MEMBER_OF]->(n1:Team)<-[:APPLIES_TO]-(n2:Context)" in executed.given
    assert "$allowed_shared = [['TEAM', 'T1']]" in executed.given
    assert executed.detail["rows"] == 1


async def test_the_trace_shows_where_planning_stopped_and_why() -> None:
    tool, _, _, _ = tool_for(
        [SchemaFocus(answered_by_user_context=True, keywords=["concise"])],
        StubDriver(traversal=rows(context_node())),
    )

    await tool.run("how should you answer me?", "U1")

    stop = next(step for step in tool.last_trace if step.stage == "stop planning")
    assert "no further level needed" in stop.returned
    assert "levels used: 1 of 3" in stop.returned
    assert [step.level for step in tool.last_trace if step.actor == "llm"] == [1]


async def test_the_trace_shows_a_corrected_direction_and_a_dropped_filter() -> None:
    backwards = TraversalPlan(
        anchor_label="Team",
        hops=[Hop(relationship="APPLIES_TO", direction="out", target_label="Context")],
        scope_types=["ORGANIZATION", "TEAM"],
        context_types=["Preference", "nonsense"],
    )
    tool, _, _, _ = tool_for(
        [SchemaFocus(labels=["Team"]), backwards],
        StubDriver(traversal=rows(context_node()), shared=[["TEAM", "T1"]]),
    )

    await tool.run("what does my team expect?", "U1")

    check = next(step for step in tool.last_trace if step.stage.startswith("validate"))
    assert "directions corrected" in check.returned
    assert "filters dropped" in check.returned
    assert "ORGANIZATION" in check.returned and "nonsense" in check.returned


async def test_a_rejected_plan_and_the_fallback_are_both_traced() -> None:
    tool, _, _, _ = tool_for(
        [SchemaFocus(labels=["User"]), TraversalPlan(anchor_label="Wizard")]
    )

    await tool.run("who is on my team?", "U1")
    trace = tool.last_trace

    check = next(step for step in trace if step.stage.startswith("validate"))
    assert check.selected is False and "rejected: unknown label" in check.returned
    served = next(step for step in trace if step.actor == "fallback")
    assert "StubFallback.retrieve(user_id, query)" in served.given
    assert served.stage == "fall back to standard retrieval"


async def test_a_failed_model_call_is_traced_before_the_fallback() -> None:
    tool, _, _, _ = tool_for([ProviderError("model unavailable")])

    await tool.run("anything", "U1")

    failed = next(step for step in tool.last_trace if step.actor == "llm")
    assert failed.selected is False and "model unavailable" in failed.returned


async def test_each_retrieval_starts_a_fresh_trace() -> None:
    tool, _, _, _ = tool_for(
        [SchemaFocus(answered_by_user_context=True), SchemaFocus(answered_by_user_context=True)],
        StubDriver(traversal=rows(context_node())),
    )

    await tool.run("first", "U1")
    first = len(tool.last_trace)
    await tool.run("second", "U1")

    assert len(tool.last_trace) == first
    assert tool.last_trace[0].given.endswith("user_id: U1")
    assert "second" in tool.last_trace[0].given


# -- the drawable traversal ----------------------------------------------


async def test_the_walked_subgraph_is_captured_for_drawing() -> None:
    plan = TraversalPlan(
        anchor_label="User",
        hops=[Hop(relationship="APPLIES_TO", direction="in", target_label="Context")],
    )
    tool, _, _, _ = tool_for(
        [SchemaFocus(labels=["User"]), plan],
        StubDriver(traversal=rows(context_node(), walked=True), shared=[["TEAM", "T1"]]),
    )

    await tool.run("what do I prefer?", "U1")
    graph = tool.last_graph

    assert not graph.is_empty
    anchor = next(node for node in graph.nodes if node.stage == 0)
    landed = next(node for node in graph.nodes if node.stage == 1)
    assert (anchor.label, anchor.caption) == ("User", "Riya")
    assert landed.label == "Context" and landed.is_result  # it survived ranking
    assert [edge.type for edge in graph.edges] == ["APPLIES_TO"]
    # One caption per hop, so a viewer can step through the walk.
    assert graph.stages == ("start at User", "follow APPLIES_TO to Context")


async def test_the_drawing_is_bounded() -> None:
    many = rows(*[context_node(f"c{index}", f"v{index}") for index in range(20)], walked=True)
    tool, _, _, _ = tool_for(
        [SchemaFocus(answered_by_user_context=True)], StubDriver(traversal=many), max_graph_nodes=5
    )

    await tool.run("everything", "U1")

    assert len(tool.last_graph.nodes) == 5
    assert tool.last_graph.truncated


async def test_there_is_nothing_to_draw_when_the_fallback_answered() -> None:
    tool, _, _, _ = tool_for(
        [SchemaFocus(labels=["User"]), TraversalPlan(anchor_label="Wizard")]
    )

    await tool.run("who is on my team?", "U1")

    assert tool.last_graph.is_empty


async def test_every_step_carries_a_sentence_a_newcomer_can_read() -> None:
    tool, _, _, _ = tool_for(
        [SchemaFocus(answered_by_user_context=True, reason="it is about them")],
        StubDriver(traversal=rows(context_node())),
    )

    await tool.run("how should you answer me?", "U1")

    assert all(step.summary for step in tool.last_trace)
    assert all(not step.summary.startswith("_") for step in tool.last_trace)
    said = " ".join(step.summary for step in tool.last_trace)
    assert "You asked" in said
    assert "allowed to see" in said
    # No jargon leaks into the plain-language line; the raw text stays in `given`.
    assert "MATCH" not in said and "$" not in said


def test_the_drawing_reveals_the_walk_hop_by_hop() -> None:
    graph = TraversalGraph(
        nodes=(
            GraphNode(id="a", label="User", caption="Riya", stage=0),
            GraphNode(id="b", label="Context", caption="concise", stage=1, is_result=True),
        ),
        edges=(GraphEdge(source="a", target="b", type="APPLIES_TO", stage=1),),
        stages=("start at User", "follow APPLIES_TO to Context"),
    )

    first, second = graph.to_dot(0), graph.to_dot(1)

    assert "Riya" in first and "concise" not in first and "APPLIES_TO" not in first
    assert "Riya" in second and "concise" in second and "APPLIES_TO" in second


async def test_context_that_is_not_about_the_question_is_set_aside() -> None:
    """One explicit question: off-topic context is noise, not background."""

    class Embeddings:
        model = "stub-embed"
        dimension = 2

        async def embed(self, texts: Any) -> list[list[float]]:
            return [[1.0, 0.0]]

    on_topic = context_node("event_schema", "one schema per event")
    on_topic["embedding"] = [1.0, 0.0]  # similarity 1.0
    off_topic = context_node("coffee", "flat white")
    off_topic["embedding"] = [0.1, 0.995]  # similarity 0.1
    off_topic["context_type"] = "Fact"  # not a standing instruction

    driver = StubDriver(traversal=rows(on_topic, off_topic))
    tool = ContextGraphRetrievalTool(
        Neo4jSettings(),
        driver,  # pyright: ignore[reportArgumentType]
        llm=StubLLM([SchemaFocus(answered_by_user_context=True)]),
        fallback=StubFallback(),
        agent_settings=AgentSettings(),
        embeddings=Embeddings(),  # pyright: ignore[reportArgumentType]
    )

    result = await tool.run("what did we decide about event schemas?", "U1")

    assert [item.concept for item in result.items] == ["event_schema"]
    assert any("not about your question" in step.summary for step in tool.last_trace)


async def test_a_standing_instruction_survives_the_relevance_threshold() -> None:
    """A preference applies whatever the topic, exactly as pre-fetch keeps it."""

    class Embeddings:
        model = "stub-embed"
        dimension = 2

        async def embed(self, texts: Any) -> list[list[float]]:
            return [[1.0, 0.0]]

    standing = context_node("response_detail", "concise")  # Preference
    standing["embedding"] = [0.1, 0.995]

    tool = ContextGraphRetrievalTool(
        Neo4jSettings(),
        StubDriver(traversal=rows(standing)),  # pyright: ignore[reportArgumentType]
        llm=StubLLM([SchemaFocus(answered_by_user_context=True)]),
        fallback=StubFallback(),
        agent_settings=AgentSettings(),
        embeddings=Embeddings(),  # pyright: ignore[reportArgumentType]
    )

    result = await tool.run("what did we decide about event schemas?", "U1")

    assert [item.concept for item in result.items] == ["response_detail"]


def test_a_misnamed_anchor_property_is_pointed_at_the_real_identity() -> None:
    """A model that names a relationship still said which node to start from."""
    from retrieval_tester.context_tool import repair_anchor

    proposed = TraversalPlan(
        anchor_label="User",
        anchor_property="MEMBER_OF",  # a relationship, not a property
        anchor_value="U1",
        hops=[
            Hop(relationship="MEMBER_OF", direction="out", target_label="Team"),
            Hop(relationship="APPLIES_TO", direction="in", target_label="Context"),
        ],
    )
    repaired = repair_anchor(proposed, SCHEMA)

    assert repaired.anchor_property == "user_id" and repaired.anchor_value == "U1"
    assert validate_plan(repaired, SCHEMA, AgentSettings()) is None


def test_an_anchor_is_never_invented() -> None:
    """Nothing is guessed: without a real identity property the plan is untouched."""
    from retrieval_tester.context_tool import repair_anchor

    # Episode has no identity property in the ontology's scope map.
    unfixable = TraversalPlan(anchor_label="Episode", anchor_property="nonsense", anchor_value="x")
    assert repair_anchor(unfixable, SCHEMA) == unfixable
    assert "no property" in (validate_plan(unfixable, SCHEMA, AgentSettings()) or "")

    # A property that is already real is left exactly as the model gave it.
    fine = TraversalPlan(anchor_label="User", anchor_property="name", anchor_value="Riya")
    assert repair_anchor(fine, SCHEMA) == fine


# -- tracing a stored fact back to its source ----------------------------


def source_trace() -> TraversalPlan:
    """Context first, then on through Episode and Session to the Agent."""
    return TraversalPlan(
        anchor_label="Context",
        hops=[
            Hop(relationship="SUPPORTED_BY", direction="out", target_label="Episode"),
            Hop(relationship="CONTAINS", direction="in", target_label="Session"),
            Hop(relationship="INVOLVES_AGENT", direction="out", target_label="Agent"),
        ],
    )


def test_a_route_may_pass_through_context_and_carry_on_to_its_source() -> None:
    plan = source_trace()

    assert validate_plan(plan, SCHEMA, AgentSettings()) is None
    assert plan.context_position == 0 and plan.traces_source

    # The guard is applied to the Context, wherever it sits on the route.
    cypher, _ = compile_plan(plan, "U1", _scopes(), AgentSettings())
    assert "(n0:Context)-[:SUPPORTED_BY]->(n1:Episode)" in cypher
    assert "<-[:CONTAINS]-(n2:Session)-[:INVOLVES_AGENT]->(n3:Agent)" in cypher
    assert "WITH n0 AS c" in cypher and "c.status = $active_status" in cypher


def test_a_route_must_still_reach_exactly_one_context() -> None:
    assert "never reaches a Context" in (
        validate_plan(
            TraversalPlan(
                anchor_label="User",
                hops=[Hop(relationship="MEMBER_OF", direction="out", target_label="Team")],
            ),
            SCHEMA,
            AgentSettings(),
        )
        or ""
    )
    assert "must be one" in (
        validate_plan(
            TraversalPlan(
                anchor_label="Context",
                hops=[Hop(relationship="SUPERSEDES", direction="out", target_label="Context")],
            ),
            GraphSchema(
                properties={"Context": ("context_id",)},
                edges=(Edge(start="Context", type="SUPERSEDES", end="Context"),),
            ),
            AgentSettings(),
        )
        or ""
    )


async def test_the_drawing_marks_the_context_not_the_last_node() -> None:
    """On a source trace the answer sits at the start, not the end of the path."""
    node = context_node()
    node["context_id"] = "c1"
    row: dict[str, Any] = {
        "node": node,
        "quote": "because I said so",
        "path_nodes": [
            {"id": "c1", "label": "Context", "caption": "concise", "props": []},
            {"id": "e1", "label": "Episode", "caption": "I prefer...", "props": []},
            {"id": "s1", "label": "Session", "caption": "S1", "props": []},
        ],
        "path_edges": [
            {"type": "SUPPORTED_BY", "source": "c1", "target": "e1"},
            {"type": "CONTAINS", "source": "s1", "target": "e1"},
        ],
    }
    plan = TraversalPlan(
        anchor_label="Context",
        hops=[
            Hop(relationship="SUPPORTED_BY", direction="out", target_label="Episode"),
            Hop(relationship="CONTAINS", direction="in", target_label="Session"),
        ],
    )
    tool, _, _, _ = tool_for([SchemaFocus(labels=["Context"]), plan], StubDriver(traversal=[row]))

    await tool.run("where did that come from?", "U1")
    graph = tool.last_graph

    marked = [node for node in graph.nodes if node.is_result]
    assert [node.label for node in marked] == ["Context"]
    assert graph.stages[2].endswith("(tracing the source)")


def test_the_source_chain_is_walked_from_the_schema_not_guessed() -> None:
    """Which route reaches the answer is a judgement; how evidence links is not."""
    from retrieval_tester.context_tool import extend_with_source

    plan = TraversalPlan(
        anchor_label="User",
        anchor_property="user_id",
        anchor_value="U1",
        hops=[
            Hop(relationship="MEMBER_OF", direction="out", target_label="Team"),
            Hop(relationship="APPLIES_TO", direction="in", target_label="Context"),
        ],
    )
    traced = extend_with_source(plan, SCHEMA, AgentSettings())

    assert [(hop.relationship, hop.direction, hop.target_label) for hop in traced.hops] == [
        ("MEMBER_OF", "out", "Team"),
        ("APPLIES_TO", "in", "Context"),
        # Directions come from the discovered schema, not from the chain order.
        ("SUPPORTED_BY", "out", "Episode"),
        ("CONTAINS", "in", "Session"),
        ("INVOLVES_AGENT", "out", "Agent"),
    ]
    assert traced.context_position == 2 and traced.traces_source
    assert validate_plan(traced, SCHEMA, AgentSettings()) is None


def test_the_source_chain_stops_where_the_graph_stops() -> None:
    """A graph without sessions traces as far as it can and no further."""
    from retrieval_tester.context_tool import extend_with_source

    partial = GraphSchema(
        properties={"Context": ("context_id",), "Episode": ("episode_id",)},
        edges=(Edge(start="Context", type="SUPPORTED_BY", end="Episode"),),
    )
    traced = extend_with_source(TraversalPlan(anchor_label="Context"), partial, AgentSettings())

    assert [hop.target_label for hop in traced.hops] == ["Episode"]
    assert validate_plan(traced, partial, AgentSettings()) is None


def test_the_source_chain_respects_the_depth_limit_and_never_doubles_up() -> None:
    from retrieval_tester.context_tool import extend_with_source

    capped = extend_with_source(
        TraversalPlan(anchor_label="Context"), SCHEMA, AgentSettings(planner_max_hops=2)
    )
    assert len(capped.hops) == 2

    # A route the model already carried past its Context is left alone.
    already = extend_with_source(source_trace(), SCHEMA, AgentSettings())
    assert already == source_trace()
