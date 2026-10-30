"""Planned Context Graph retrieval: query + user_id in, applicable Context out.

The pre-fetch retriever answers "what is true about this user" with one fixed
query. This tool answers a *question* by planning a traversal over the schema
the database actually reports, one level at a time:

  level 1  the query plus a compact outline of the discovered schema; the model
           says where the answer is likely to live, and whether the user's own
           Context already settles it
  level 2  only the properties and edges of the labels level 1 chose; the model
           returns a bounded traversal plan
  level 3  runs only when level 2 says the plan needs to go deeper, and sees
           only the schema around the plan's terminal label

A question that level 1 can already settle never reaches level 2, so the common
case costs a single model call.

The model never writes Cypher. It returns a validated plan, that plan is checked
against the discovered schema and against the scopes this user may read, and
only then is it compiled into a parameterised, read-only query. Lifecycle
(status, validity window) and authorization are applied by this module, never by
the model: a plan cannot widen them. Anything that fails - no schema, an invalid
plan, a model failure, a graph timeout - falls back to the existing retriever.
"""

from __future__ import annotations

import asyncio
import re
from time import perf_counter
from typing import Any, Literal, cast

from neo4j import AsyncDriver, RoutingControl
from neo4j import exceptions as neo4j_exceptions
from pydantic import BaseModel, ConfigDict, Field

from context_graph.config import Neo4jSettings
from context_graph.domain.enums import ContextStatus, ContextType, ScopeType
from context_graph.errors import ContextGraphError
from context_graph.graph import mapping
from context_graph.graph.schema import Label
from context_graph.providers.base import EmbeddingProvider, LLMProvider
from retrieval_tester.config import AgentSettings
from retrieval_tester.errors import RetrievalError
from retrieval_tester.models import (
    RetrievalDebug,
    RetrievalResult,
    RetrievedContext,
    utcnow,
)
from retrieval_tester.retrieval import (
    ContextRetriever,
    _cosine,  # pyright: ignore[reportPrivateUsage] - one definition of similarity
)
from retrieval_tester.schema_catalog import IDENTIFIER, GraphSchema, SchemaCatalog

TOOL_NAME = "context_graph_retrieval"

#: Readable scopes, derived from the ontology rather than restated: SESSION
#: context belongs to the session it was stated in and is never applicable.
_PRIVATE_SCOPES = [
    scope.value
    for scope in ScopeType
    if not scope.is_shared and scope is not ScopeType.SESSION
]
_SHARED_SCOPES = [scope.value for scope in ScopeType if scope.is_shared]

def _values(values: Any) -> str:
    """Render a controlled vocabulary for a prompt."""
    return ", ".join(values)


#: Words carried into keyword matching from the query when the model gives none.
_STOPWORDS = frozenset(
    """a an and any are as at be by can did do does for from has have how i in is it me my of
    on or our should show that the their them there they to us was we what when where which who
    whom why will with you your""".split()
)

#: Definition handed to a model or UI that registers this tool.
TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "description": (
            "Answer a question from the Context Graph by planning a traversal over "
            "the stored Context: preferences, constraints, decisions, goals and "
            "facts, including the ones that belong to a team, department or project "
            "this user is entitled to read. Use it when the pre-fetched context does "
            "not already contain the answer."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The question to answer, in the user's own words.",
                },
                "user_id": {
                    "type": "string",
                    "description": "The user the question is asked on behalf of.",
                },
            },
            "required": ["query", "user_id"],
            "additionalProperties": False,
        },
    },
}


class SchemaFocus(BaseModel):
    """Level 1: where in the real schema the answer is likely to live."""

    model_config = ConfigDict(frozen=True)

    labels: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    answered_by_user_context: bool = False
    #: The question asks where something came from, not only what is stored.
    needs_source_trace: bool = False
    reason: str = ""


class Hop(BaseModel):
    """One relationship step of the plan."""

    model_config = ConfigDict(frozen=True)

    relationship: str
    direction: Literal["out", "in"]
    target_label: str

    def render(self) -> str:
        arrow = f"-[:{self.relationship}]->" if self.direction == "out" else f"<-[:{self.relationship}]-"
        return f"{arrow}({self.target_label})"


class TraversalPlan(BaseModel):
    """A bounded path from an anchor node to the Context that answers the query."""

    model_config = ConfigDict(frozen=True)

    anchor_label: str
    #: Identity property of the anchor; empty means "every node of that label".
    anchor_property: str = ""
    anchor_value: str = ""
    hops: list[Hop] = Field(default_factory=list)
    scope_types: list[str] = Field(default_factory=list)
    context_types: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    needs_deeper_traversal: bool = False
    reason: str = ""

    @property
    def context_position(self) -> int:
        """Index along the path where the Context sits: 0 is the anchor."""
        labels = [self.anchor_label, *(hop.target_label for hop in self.hops)]
        return labels.index(Label.CONTEXT.value) if Label.CONTEXT.value in labels else -1

    @property
    def traces_source(self) -> bool:
        """Whether the route carries on past the Context to where it came from."""
        return 0 <= self.context_position < len(self.hops)

    @property
    def has_filters(self) -> bool:
        """Whether the plan narrows the Context it lands on beyond the path."""
        return bool(self.scope_types or self.context_types or self.keywords)

    def without_filters(self) -> TraversalPlan:
        """The same traversal, unfiltered. Access is unchanged: the guard decides."""
        return self.model_copy(
            update={"scope_types": [], "context_types": [], "keywords": []}
        )

    def render(self) -> str:
        anchor = self.anchor_label
        if self.anchor_property:
            anchor += f" {{{self.anchor_property}: '{self.anchor_value}'}}"
        path = f"({anchor})" + "".join(hop.render() for hop in self.hops)
        filters = [
            f"scope_type IN {self.scope_types}" if self.scope_types else "",
            f"context_type IN {self.context_types}" if self.context_types else "",
            f"keywords {self.keywords}" if self.keywords else "",
        ]
        return path + "".join(f"  [{item}]" for item in filters if item)


class GraphNode(BaseModel):
    """One node the traversal actually touched, as the UI should draw it."""

    model_config = ConfigDict(frozen=True)

    id: str
    label: str
    caption: str
    #: Hop number it sits at: 0 is the anchor, the last one holds the answers.
    stage: int
    #: Whether this node is one of the Context items that was returned.
    is_result: bool = False
    #: A few of the node's own properties, so it can be inspected like a
    #: database node rather than just a dot. Never includes embeddings.
    properties: dict[str, str] = Field(default_factory=dict)


class GraphEdge(BaseModel):
    """One relationship the traversal actually followed."""

    model_config = ConfigDict(frozen=True)

    source: str
    target: str
    type: str
    #: Hop number this edge completes.
    stage: int


class TraversalGraph(BaseModel):
    """The real subgraph that was walked, ready to draw hop by hop."""

    model_config = ConfigDict(frozen=True)

    nodes: tuple[GraphNode, ...] = ()
    edges: tuple[GraphEdge, ...] = ()
    #: One caption per hop, so a viewer can step through the traversal.
    stages: tuple[str, ...] = ()
    truncated: bool = False

    @property
    def is_empty(self) -> bool:
        return not self.nodes

    @property
    def last_hop(self) -> int:
        return max(len(self.stages) - 1, 0)

    def to_payload(self) -> dict[str, Any]:
        """The walked graph as plain data for an interactive viewer."""
        return {
            "nodes": [
                {
                    "id": node.id,
                    "label": node.label,
                    "caption": node.caption,
                    "level": node.stage,
                    "isResult": node.is_result,
                    "properties": node.properties,
                }
                for node in self.nodes
            ],
            "edges": [
                {
                    "source": edge.source,
                    "target": edge.target,
                    "type": edge.type,
                    "level": edge.stage,
                }
                for edge in self.edges
            ],
            "stages": list(self.stages),
            "labels": sorted({node.label for node in self.nodes}),
            "truncated": self.truncated,
        }

    def to_dot(self, upto: int | None = None) -> str:
        """Graphviz for the walk, revealed up to hop ``upto`` (all hops by default).

        Rendering lives with the data rather than in a UI module so it can be
        tested without starting an application.
        """
        limit = self.last_hop if upto is None else upto
        lines = [
            "digraph traversal {",
            "  rankdir=LR; bgcolor=transparent; pad=0.3; nodesep=0.35; ranksep=0.7;",
            '  node [shape=box style="rounded,filled" fontname="Helvetica" fontsize=10'
            ' color="#00000000" fontcolor=white];',
            '  edge [fontname="Helvetica" fontsize=9 color="#64748b" fontcolor="#64748b"];',
        ]
        visible: set[str] = set()
        for node in self.nodes:
            if node.stage > limit:
                continue
            visible.add(node.id)
            colour = (
                _RESULT_COLOUR
                if node.is_result
                else _ANCHOR_COLOUR
                if node.stage == 0
                else _STEP_COLOUR
            )
            caption = node.caption.replace('"', "'")[:38]
            lines.append(
                f'  "{node.id}" [label="{node.label}\\n{caption}" fillcolor="{colour}"];'
            )
        for edge in self.edges:
            if edge.stage <= limit and {edge.source, edge.target} <= visible:
                lines.append(f'  "{edge.source}" -> "{edge.target}" [label="{edge.type}"];')
        lines.append("}")
        return "\n".join(lines)


class TraceStep(BaseModel):
    """One recorded step of a retrieval, so the UI can show the whole path.

    Every decision is here in full: what was handed to the model, what it chose,
    what was rejected or corrected, the exact query that ran and what came back.
    """

    model_config = ConfigDict(frozen=True)

    #: Planning level this belongs to; 0 for work that is not a planning level.
    level: int
    #: Short name of the step, e.g. "locate the answer in the schema".
    stage: str
    #: One plain sentence, for a reader who has never seen this system.
    summary: str = ""
    #: What performed it: the model, the graph, or this module's own rules.
    actor: Literal["llm", "graph", "policy", "embeddings", "fallback"]
    #: Exactly what was given to it.
    given: str = ""
    #: Exactly what came back.
    returned: str = ""
    duration_ms: float = 0.0
    #: Whether this step produced the outcome the retrieval went on to use.
    selected: bool = True
    detail: dict[str, Any] = Field(default_factory=dict)


class AllowedScopes(BaseModel):
    """What this user may read, resolved from the governed membership edges."""

    model_config = ConfigDict(frozen=True)

    user_exists: bool
    #: ``[scope_type, scope_id]`` pairs reachable through MEMBER_OF / BELONGS_TO.
    shared: tuple[tuple[str, str], ...] = ()

    @property
    def scope_types(self) -> tuple[str, ...]:
        """Scope types this user may read at all; SESSION is never among them."""
        return tuple(_PRIVATE_SCOPES) + tuple({pair[0] for pair in self.shared})


_LEVEL1_SYSTEM = """You plan retrieval over a Context Graph. You never write queries.

You are given a user's question and the labels and relationships that exist in
the graph. Say where the answer is most likely to live.

Rules:
- Only name labels from the list you were given. Never invent one.
- Set answered_by_user_context to true ONLY when the answer is durable Context
  about the asking user themselves (preferences, constraints, goals, habits,
  working style, personal facts) and needs no walk through Team, Project,
  Department, Session or Agent. Prefer it for "what do I prefer / what are my
  goals" style questions.
- Set answered_by_user_context to false when the question is about a team,
  department, organization, project, colleague, membership ("what team am I
  on", "who is on my team", "what projects do I work on"), a past conversation,
  or anything reached through MEMBER_OF, BELONGS_TO, PART_OF, WORKS_ON or
  PARTICIPATED_IN. Those need a real traversal, not a Context-only lookup.
- Set needs_source_trace to true when the question asks where something came
  from, who said it, when or in which conversation it was said, which assistant
  was involved, or asks to trace, prove or show the evidence behind something.
  A source question needs a route even when it is about the asker themselves.
- keywords: the few content words of the question that a stored value or concept
  name would contain. Lowercase, no filler words.
- reason: one short sentence.
"""

_LEVEL2_SYSTEM = f"""You turn a question into a bounded traversal plan. You never write queries.

You are given the question, the labels chosen in the previous step, their real
properties and the relationships that touch them.

Rules:
- The question is asked by a named user. When it says "my", "our", "my team" or
  anything else about the asker's own surroundings, start the route at that user
  and walk the relationship that connects them to the group. Starting at the
  group itself matches every group in the company, which is not what was asked.
- anchor_label must be one of the labels you were given.
- anchor_property must be a real property of anchor_label - an identifier or a
  name from the property list you were given, never a relationship type - or
  empty to match every node of that label. anchor_value is the value to match, and it may be
  a name or identifier taken from the question.
- hops walk from the anchor through the graph. Every hop must be a relationship
  that actually exists between those two labels, in the direction you give.
- The route must pass through Context exactly once: that is what is returned.
- Context does not have to be the last node. Reach it and carry on when the
  question asks where something came from, who said it, in which conversation,
  or which assistant was involved - for example Context to the Episode that
  supports it, on to the Session that contains it, on to the Agent involved in
  that session. That is how a stored fact is traced back to its source.
- Otherwise keep the path as short as it can be. Never add a hop that the
  question does not require.
- scope_types, context_types and keywords are optional filters on the Context
  you land on. Leave them empty unless the question clearly calls for one.
  scope_types must come from: {_values(scope.value for scope in ScopeType)}.
  context_types must come from: {_values(kind.value for kind in ContextType)}.
- You cannot grant access. Anything the user may not read is removed after
  planning, whatever you put in the plan.
- needs_deeper_traversal is true only when the path you produced cannot reach
  the answer and you need to see the schema beyond its last label.
- reason: one short sentence.
"""

_LEVEL3_SYSTEM = """Your previous plan said it needed to go deeper.

You are now given the schema around the last label of that plan. Produce the
complete plan, from the anchor, with the extra hops included. The same rules
apply: real labels, real properties, real relationships in the right direction,
the last hop landing on Context, and no hop the question does not require.
Set needs_deeper_traversal to false: this is the final plan.
"""


class ContextGraphRetrievalTool:
    """Level-by-level planned retrieval from the Context Graph."""

    def __init__(
        self,
        settings: Neo4jSettings,
        driver: AsyncDriver,
        *,
        llm: LLMProvider,
        fallback: ContextRetriever,
        agent_settings: AgentSettings,
        catalog: SchemaCatalog | None = None,
        embeddings: EmbeddingProvider | None = None,
    ) -> None:
        self._settings = settings
        self._driver = driver
        self._llm = llm
        self._fallback = fallback
        self._agent = agent_settings
        self._embeddings = embeddings
        self._catalog = catalog or SchemaCatalog(
            settings,
            driver,
            cache_seconds=agent_settings.schema_cache_seconds,
            timeout_seconds=agent_settings.graph_timeout_seconds,
        )
        self._trace: list[TraceStep] = []
        self._graph = TraversalGraph()

    @property
    def catalog(self) -> SchemaCatalog:
        """The schema cache, shared so a UI can show what was discovered."""
        return self._catalog

    @property
    def last_trace(self) -> tuple[TraceStep, ...]:
        """Every step of the most recent retrieval, in order."""
        return tuple(self._trace)

    @property
    def last_graph(self) -> TraversalGraph:
        """The subgraph the most recent retrieval actually walked."""
        return self._graph

    def reset(self) -> None:
        """Forget the recorded trace and traversal."""
        self._trace.clear()
        self._graph = TraversalGraph()

    def _record(
        self,
        *,
        level: int,
        stage: str,
        actor: Literal["llm", "graph", "policy", "embeddings", "fallback"],
        summary: str = "",
        given: str = "",
        returned: str = "",
        started: float | None = None,
        selected: bool = True,
        **detail: Any,
    ) -> None:
        """Add one step to the trace of the retrieval in progress."""
        self._trace.append(
            TraceStep(
                level=level,
                stage=stage,
                summary=summary,
                actor=actor,
                given=given,
                returned=returned,
                duration_ms=_elapsed(started) if started is not None else 0.0,
                selected=selected,
                detail=detail,
            )
        )

    async def run(self, query: str, user_id: str) -> RetrievalResult:
        """Plan and execute retrieval for ``query`` on behalf of ``user_id``."""
        question, identifier = query.strip(), user_id.strip()
        if not identifier:
            raise RetrievalError("user_id must not be empty")
        if not question:
            raise RetrievalError("query must not be empty")

        started = perf_counter()
        notes: list[str] = []
        self.reset()
        self._record(
            level=0,
            stage="request",
            actor="policy",
            summary=f"You asked: \u201c{question}\u201d, as user {identifier}.",
            given=f"query: {question}\nuser_id: {identifier}",
            returned=f"tool: {TOOL_NAME}",
        )

        step = perf_counter()
        cached = self._catalog.cached is not None
        try:
            schema, scopes = await asyncio.gather(
                self._catalog.get(), self._allowed_scopes(identifier)
            )
        except RetrievalError as error:
            self._record(
                level=0, stage="schema discovery", actor="graph",
                summary="Asked the database what it contains, but it could not answer.",
                given="db.schema.nodeTypeProperties(), db.schema.visualization()",
                returned=f"failed: {error}", started=step, selected=False,
            )
            return await self._fall_back(question, identifier, started, f"{error}")

        self._record(
            level=0,
            stage="schema discovery",
            actor="graph",
            summary=(
                f"Read the real shape of the graph from the database: "
                f"{len(schema.labels)} kinds of node and {len(schema.edges)} kinds of "
                "connection between them."
                + ("" if not cached else " (reused from cache, no query needed.)")
            ),
            given=(
                "CALL db.schema.nodeTypeProperties()\nCALL db.schema.visualization()"
                if not cached
                else f"cached schema, reused for up to {self._agent.schema_cache_seconds}s"
            ),
            returned=schema.outline(),
            started=step,
            source="cache" if cached else "database",
            labels=len(schema.labels),
            relationships=len(schema.edges),
        )
        self._record(
            level=0,
            stage="scope resolution",
            actor="graph",
            summary=(
                f"Worked out what {identifier} is allowed to see: their own private "
                "context, plus "
                + (
                    "the shared context of "
                    + ", ".join(f"{kind.lower()} {name}" for kind, name in scopes.shared)
                    if scopes.shared
                    else "no shared group, because they are not a member of any"
                )
                + "."
                if scopes.user_exists
                else f"No user called {identifier} exists in the graph."
            ),
            given=_ALLOWED_SCOPES.strip(),
            returned=(
                f"user exists: {scopes.user_exists}\n"
                + (
                    "shared scopes: "
                    + (
                        ", ".join(f"{kind}:{name}" for kind, name in scopes.shared)
                        or "none"
                    )
                )
                + f"\nreadable scope types: {', '.join(scopes.scope_types)}"
            ),
            user_exists=scopes.user_exists,
            shared=[list(pair) for pair in scopes.shared],
        )

        if not scopes.user_exists:
            return self._empty(
                question,
                identifier,
                started,
                (f"no user '{identifier}' exists in the graph; nothing was retrieved",),
            )
        if not schema.has_label(Label.CONTEXT.value):
            return await self._fall_back(
                question, identifier, started, "the graph holds no Context nodes"
            )

        try:
            plan, levels = await self._plan(question, identifier, schema, scopes)
        except (ContextGraphError, RetrievalError) as error:
            return await self._fall_back(question, identifier, started, f"planning failed: {error}")

        notes.append(f"planned in {levels} level(s): {plan.render()}")
        if plan.reason:
            notes.append(f"plan reason: {plan.reason}")

        try:
            rows = await self._execute(plan, identifier, scopes, attempt="planned traversal")
            if not rows and plan.has_filters:
                # The structure was right but the model's optional filters were
                # too narrow. One more read, same path, same guard, no filters.
                plan = plan.without_filters()
                rows = await self._execute(
                    plan, identifier, scopes, attempt="retry without the model's filters"
                )
                notes.append("no match with the planned filters; retried the same path without them")
        except RetrievalError as error:
            return await self._fall_back(question, identifier, started, f"{error}", plan=plan)

        items = await self._rank(rows, question)
        self._graph = _build_graph(plan, rows, items, self._agent)
        # Context-only plans return islands of Context nodes with no edges; attach
        # each Context's APPLIES_TO neighbour so the journey view stays connected.
        if self._graph.nodes and not self._graph.edges and rows:
            self._graph = await self._connect_context_scopes(self._graph, rows)
        notes.append(
            "ranked against the query by embedding similarity"
            if any(item.similarity is not None for item in items)
            else "not ranked by relevance: ordered by evidence and recency"
        )
        notes.extend(self._scope_notes(scopes))
        if not items:
            notes.append("the planned traversal matched no Context this user may read")
        return RetrievalResult(
            items=items,
            debug=RetrievalDebug(
                user_id=identifier,
                duration_ms=_elapsed(started),
                candidates_considered=len(rows),
                returned=len(items),
                truncated=len(rows) > len(items),
                by_scope=_count_by_scope(items),
                query=question,
                relevance_used=any(item.similarity is not None for item in items),
                notes=tuple(notes),
            ),
        )

    # -- planning ---------------------------------------------------------

    async def _plan(
        self, query: str, user_id: str, schema: GraphSchema, scopes: AllowedScopes
    ) -> tuple[TraversalPlan, int]:
        """Decide the traversal, adding a level only when one is needed."""
        focus = await self._ask(
            1,
            "locate the answer in the schema",
            _LEVEL1_SYSTEM,
            f"Question: {query}\nAsked by user_id: {user_id}\n\n{schema.outline()}",
            SchemaFocus,
        )
        keywords = _keywords(focus.keywords or _fallback_keywords(query), self._agent)

        if focus.answered_by_user_context:
            # Level 1 settled it: the user's own Context, no second model call.
            # The route still starts at the User. Anchoring on Context instead
            # produced a plan with no anchor at all, so the walk began mid-graph
            # and the drawing had no User node in it.
            plan = TraversalPlan(
                anchor_label=Label.USER.value,
                anchor_property="user_id",
                anchor_value=user_id,
                hops=_USER_CONTEXT_ROUTE,
                keywords=keywords,
                reason=focus.reason or "answered from this user's own context",
            )
            # This route already passes through the Episode and Session that
            # produced the Context, so the source is traced by construction;
            # extending again would walk back out to a duplicate Episode.
            self._record(
                level=1,
                stage="stop planning",
                actor="policy",
                summary=(
                    "The question is about you, so no second model call is needed. "
                    "The walk starts at you and follows your conversations to the "
                    "turns that produced your context, so every answer arrives with "
                    "the session and turn it came from."
                ),
                given="level 1 said the user's own context answers the question",
                returned=(
                    f"no further level needed; route: {plan.render()}\n"
                    "levels used: 1 of "
                    f"{self._agent.planner_max_levels}"
                ),
                keywords=keywords,
            )
            return plan, 1

        labels = {label for label in focus.labels if schema.has_label(label)}
        dropped = sorted(set(focus.labels) - labels)
        labels.add(Label.CONTEXT.value)
        self._record(
            level=1,
            stage="narrow the schema for level 2",
            actor="policy",
            summary=(
                "Checked the model's choice against the real graph and kept only what "
                f"exists: {', '.join(sorted(labels))}."
                + (f" Ignored {', '.join(dropped)} - no such node in this graph." if dropped else "")
            ),
            given=f"labels the model named: {', '.join(focus.labels) or 'none'}",
            returned=(
                f"kept: {', '.join(sorted(labels))}"
                + (f"\ndropped (not in the graph): {', '.join(dropped)}" if dropped else "")
            ),
            dropped=dropped,
        )
        detail = schema.detail(labels)
        plan = await self._ask(
            2,
            "build the traversal plan",
            _LEVEL2_SYSTEM,
            f"Question: {query}\nAsked by user_id: {user_id}\n\n"
            f"Labels chosen: {', '.join(sorted(labels))}\n\n{detail}",
            TraversalPlan,
        )
        levels = 2

        if plan.needs_deeper_traversal and levels < self._agent.planner_max_levels:
            terminal = plan.hops[-1].target_label if plan.hops else plan.anchor_label
            deeper = set(schema.neighbours(terminal)) | {terminal, Label.CONTEXT.value}
            self._record(
                level=3,
                stage="go deeper",
                actor="policy",
                summary=(
                    f"The plan could not reach an answer from {terminal}, so the model "
                    "is shown what lies beyond it and asked again."
                ),
                given=f"level 2 asked for more depth beyond ({terminal})",
                returned=f"opening the schema around: {', '.join(sorted(deeper))}",
            )
            plan = await self._ask(
                3,
                "complete the traversal plan",
                _LEVEL3_SYSTEM,
                f"Question: {query}\nAsked by user_id: {user_id}\n\n"
                f"Previous plan: {plan.render()}\n\n{schema.detail(deeper)}",
                TraversalPlan,
            )
            levels = 3
        elif plan.needs_deeper_traversal:
            self._record(
                level=levels,
                stage="depth limit reached",
                actor="policy",
                summary=(
                    "The model wanted yet another planning round; it was refused, so "
                    "planning can never run away."
                ),
                given=f"level {levels} asked for another level",
                returned=f"refused: planner_max_levels is {self._agent.planner_max_levels}",
                selected=False,
            )

        proposed = plan
        plan = _sanitise_filters(repair_anchor(repair_directions(plan, schema), schema), scopes)
        if focus.needs_source_trace:
            plan = extend_with_source(plan, schema, self._agent)
        error = validate_plan(plan, schema, self._agent)
        self._record(
            level=levels,
            stage="validate the plan against the graph and this user's scopes",
            actor="policy",
            summary=(
                f"The model's route was rejected: {error}. Nothing it suggested was run."
                if error
                else "Checked the model's route against the real graph"
                + (" and corrected a connection it had pointing the wrong way" if plan.hops != proposed.hops else "")
                + (
                    f", and started from {plan.anchor_label}.{plan.anchor_property} - the real "
                    f"way to identify a {plan.anchor_label}"
                    if plan.anchor_property != proposed.anchor_property
                    else ""
                )
                + (
                    ", and removed filters that are not real or not yours to read"
                    if (set(proposed.scope_types) - set(plan.scope_types))
                    or (set(proposed.context_types) - set(plan.context_types))
                    else ""
                )
                + ". The route is safe to run."
            ),
            given=f"as proposed: {proposed.render()}",
            returned=(
                f"rejected: {error}"
                if error
                else (
                    f"accepted: {plan.render()}"
                    + ("\ndirections corrected against the schema" if plan.hops != proposed.hops else "")
                    + (
                        f"\nanchor property corrected: {proposed.anchor_property or '(none)'} "
                        f"-> {plan.anchor_property}"
                        if plan.anchor_property != proposed.anchor_property
                        else ""
                    )
                    + (
                        "\nfilters dropped (not real, or not readable by this user): "
                        + ", ".join(
                            sorted(
                                (set(proposed.scope_types) - set(plan.scope_types))
                                | (set(proposed.context_types) - set(plan.context_types))
                            )
                        )
                        if (set(proposed.scope_types) - set(plan.scope_types))
                        or (set(proposed.context_types) - set(plan.context_types))
                        else ""
                    )
                )
            ),
            selected=error is None,
            levels_used=levels,
            max_hops=self._agent.planner_max_hops,
        )
        if error is not None:
            raise RetrievalError(f"the plan does not fit the graph: {error}")
        return plan.model_copy(update={"keywords": plan.keywords or keywords}), levels

    async def _ask[M: BaseModel](
        self, level: int, stage: str, system: str, user: str, schema: type[M]
    ) -> M:
        """One structured, validated model call under the planner's timeout."""
        started = perf_counter()
        given = f"[system]\n{system}\n[user]\n{user}"
        try:
            reply = await asyncio.wait_for(
                self._llm.structured(
                    system=system,
                    user=user,
                    schema=schema,
                    max_output_tokens=self._agent.planner_max_output_tokens,
                ),
                timeout=self._agent.planner_timeout_seconds,
            )
        except TimeoutError as error:
            self._record(
                level=level, stage=stage, actor="llm",
                summary=f"The model did not answer within {self._agent.planner_timeout_seconds}s.",
                given=given,
                returned=f"timed out after {self._agent.planner_timeout_seconds}s",
                started=started, selected=False, schema=schema.__name__,
            )
            raise RetrievalError(
                f"planning timed out after {self._agent.planner_timeout_seconds}s"
            ) from error
        except ContextGraphError as error:
            self._record(
                level=level, stage=stage, actor="llm",
                summary=f"The model call failed: {error}",
                given=given,
                returned=f"failed: {error}", started=started, selected=False,
                schema=schema.__name__,
            )
            raise
        self._record(
            level=level,
            stage=stage,
            actor="llm",
            summary=_describe(reply),
            given=given,
            returned=reply.model_dump_json(indent=2),
            started=started,
            schema=schema.__name__,
            model=self._llm.model,
        )
        return reply

    # -- authorization ----------------------------------------------------

    async def _allowed_scopes(self, user_id: str) -> AllowedScopes:
        """Resolve the shared scopes this user is a governed member of."""
        rows = await self._read(_ALLOWED_SCOPES, {"user_id": user_id})
        if not rows:
            return AllowedScopes(user_exists=False)
        record = rows[0]
        if not self._agent.include_shared_scopes:
            return AllowedScopes(user_exists=True)
        return AllowedScopes(
            user_exists=True,
            shared=tuple(
                (str(pair[0]), str(pair[1]))
                for pair in (record["shared"] or [])
                if pair and pair[0] and pair[1]
            ),
        )

    # -- execution --------------------------------------------------------

    async def _execute(
        self, plan: TraversalPlan, user_id: str, scopes: AllowedScopes, *, attempt: str
    ) -> list[dict[str, Any]]:
        """Compile the validated plan and run it, read-only and bounded."""
        cypher, parameters = compile_plan(plan, user_id, scopes, self._agent)
        started = perf_counter()
        try:
            rows = await self._read(cypher, parameters)
        except RetrievalError as error:
            self._record(
                level=0, stage=attempt, actor="graph",
                summary=f"The database could not run the route: {error}",
                given=f"{cypher}\n\n-- parameters --\n{_render_parameters(parameters)}",
                returned=f"failed: {error}", started=started, selected=False,
            )
            raise
        self._record(
            level=0,
            stage=attempt,
            actor="graph",
            summary=(
                f"Walked the route {plan.render().split('  [')[0]} and found "
                f"{len(rows)} piece(s) of context you are allowed to read."
                if rows
                else "Walked the route and found nothing you are allowed to read."
            ),
            given=f"{cypher}\n\n-- parameters --\n{_render_parameters(parameters)}",
            returned=f"{len(rows)} row(s), capped at {self._agent.planner_max_rows}",
            started=started,
            selected=bool(rows),
            rows=len(rows),
            plan=plan.render(),
        )
        return rows

    async def _read(self, query: str, parameters: dict[str, Any]) -> list[dict[str, Any]]:
        """One bounded, read-only graph read."""
        try:
            result = await asyncio.wait_for(
                self._driver.execute_query(
                    # Compiled here from validated identifiers only; every value
                    # the model supplied travels as a parameter.
                    cast(Any, query),
                    parameters,
                    database_=self._settings.database,
                    routing_=RoutingControl.READ,
                ),
                timeout=self._agent.graph_timeout_seconds,
            )
        except TimeoutError as error:
            raise RetrievalError("the planned traversal timed out") from error
        except neo4j_exceptions.Neo4jError as error:
            raise RetrievalError(f"the planned traversal failed: {error}") from error
        except OSError as error:
            raise RetrievalError(f"could not reach the context graph: {error}") from error
        return [dict(record) for record in result.records]

    async def _rank(self, rows: list[dict[str, Any]], query: str) -> tuple[RetrievedContext, ...]:
        """Deduplicate, rank against the query when possible, and cap."""
        seen: dict[str, dict[str, Any]] = {}
        for row in rows:
            node = dict(row["node"])
            seen.setdefault(str(node.get("context_id")), {"node": node, "quote": row.get("quote")})

        duplicates = len(rows) - len(seen)
        vector = await self._embed(query)
        items = [
            _to_item(entry["node"], entry["quote"], vector) for entry in seen.values()
        ]
        dropped = 0
        if vector is not None:
            items.sort(key=lambda item: item.similarity or 0.0, reverse=True)
            # This tool answers one explicit question, so context that is not
            # about it is noise rather than useful background. Standing
            # instructions stay whatever the topic, exactly as the pre-fetch
            # retriever keeps them.
            relevant = [
                item
                for item in items
                if item.is_standing_instruction
                or (item.similarity or 0.0) >= self._agent.min_similarity
            ]
            dropped, items = len(items) - len(relevant), relevant
        kept = tuple(items[: self._agent.max_context_items])
        ranked = any(item.similarity is not None for item in kept)
        self._record(
            level=0,
            stage="deduplicate, rank and cap",
            actor="embeddings" if ranked else "policy",
            summary=(
                f"Removed {duplicates} duplicate(s)"
                + (
                    f", set aside {dropped} item(s) that are not about your question"
                    if dropped
                    else ""
                )
                + ", "
                + ("sorted by how close each one is to your question" if ranked
                   else "kept the order the graph gave (best-evidenced first)")
                + f", and returned the top {len(kept)}."
            ),
            given=(
                f"{len(rows)} row(s) -> {len(seen)} distinct Context node(s)"
                + (
                    f", {dropped} below the relevance threshold "
                    f"({self._agent.min_similarity})"
                    if dropped
                    else ""
                )
                + "\n"
                + (
                    f"embedded the query with {self._embeddings.model}"
                    + (
                        ""
                        if ranked
                        else " (no stored vector was comparable: the graph holds "
                        "embeddings of a different size, so relevance is unavailable)"
                    )
                    if vector is not None and self._embeddings is not None
                    else "no embedding provider: keeping the graph's own order"
                )
            ),
            returned=(
                f"{len(kept)} item(s) returned, capped at {self._agent.max_context_items}\n"
                + (
                    "ordered by similarity to the query"
                    if ranked
                    else "ordered by evidence count, then recency"
                )
                + "\n"
                + "\n".join(
                    f"  [{item.scope_type.value}:{item.scope_id}] {item.concept} = {item.value}"
                    + (f"  (relevance {item.similarity:.2f})" if item.similarity is not None else "")
                    for item in kept
                )
            ),
            duplicates=duplicates,
            off_topic_dropped=dropped,
            ranked=ranked,
        )
        return kept

    async def _embed(self, query: str) -> list[float] | None:
        """Embed the query for ranking; ranking is optional, so failure is not fatal."""
        if self._embeddings is None:
            return None
        try:
            return (await self._embeddings.embed([query]))[0]
        except ContextGraphError as error:
            self._record(
                level=0, stage="rank by relevance", actor="embeddings",
                summary=f"ranking skipped: {error}", selected=False,
            )
            return None

    async def _connect_context_scopes(
        self, graph: TraversalGraph, rows: list[dict[str, Any]]
    ) -> TraversalGraph:
        """Attach each Context's APPLIES_TO neighbour when the plan had no hops.

        A Context-only plan returns islands of Context nodes. Linking each to the
        User / Project / Team it applies to makes the journey panel readable.
        """
        context_ids = [
            str(dict(row["node"]).get("context_id") or "")
            for row in rows
            if row.get("node") is not None
        ]
        context_ids = [identifier for identifier in context_ids if identifier]
        if not context_ids:
            return graph
        query = """
        MATCH (c:Context)-[r:APPLIES_TO]->(t)
        WHERE c.context_id IN $ids
        RETURN elementId(c) AS context_eid, c.context_id AS context_id,
               elementId(t) AS target_eid, head(labels(t)) AS target_label,
               coalesce(t.user_id, t.project_id, t.team_id, t.department_id,
                        t.organization_id, t.session_id, t.name, head(labels(t))) AS caption,
               type(r) AS rel
        """
        try:
            records = await self._read(query, {"ids": context_ids})
        except RetrievalError as error:
            self._record(
                level=0, stage="draw scope links", actor="graph",
                summary=f"scope links not drawn: {error}", selected=False,
            )
            return graph
        if not records or "context_eid" not in records[0]:
            return graph

        nodes = {node.id: node for node in graph.nodes}
        edges = {(edge.source, edge.type, edge.target): edge for edge in graph.edges}
        for record in records:
            context_eid = str(record["context_eid"])
            target_eid = str(record["target_eid"])
            if context_eid not in nodes:
                continue
            if target_eid not in nodes and len(nodes) < self._agent.max_graph_nodes:
                nodes[target_eid] = GraphNode(
                    id=target_eid,
                    label=str(record["target_label"] or "?"),
                    caption=str(record["caption"] or "?")[:40],
                    stage=0,
                    is_result=False,
                    properties={},
                )
                # Context that answered sits one hop out from its scope target.
                existing = nodes[context_eid]
                nodes[context_eid] = existing.model_copy(update={"stage": 1})
            if target_eid in nodes:
                key = (context_eid, str(record["rel"]), target_eid)
                edges.setdefault(
                    key,
                    GraphEdge(
                        source=context_eid,
                        target=target_eid,
                        type=str(record["rel"]),
                        stage=1,
                    ),
                )
        if not edges:
            return graph
        return TraversalGraph(
            nodes=tuple(nodes.values()),
            edges=tuple(edges.values()),
            stages=("start at scope", "follow APPLIES_TO to Context"),
            truncated=graph.truncated,
        )

    async def _graph_from_items(
        self, user_id: str, items: tuple[RetrievedContext, ...]
    ) -> TraversalGraph:
        """Build a User→Context drawing when planned retrieval fell back."""
        if not items:
            return TraversalGraph()
        context_ids = [item.context_id for item in items]
        query = """
        MATCH (u:User {user_id: $user_id})
        MATCH (c:Context)-[:APPLIES_TO]->(t)
        WHERE c.context_id IN $ids
        RETURN elementId(u) AS user_eid, u.user_id AS user_caption,
               elementId(c) AS context_eid, c.context_id AS context_id,
               coalesce(toString(c.value), c.concept_key, c.context_type) AS caption,
               elementId(t) AS target_eid, head(labels(t)) AS target_label,
               coalesce(t.user_id, t.project_id, t.team_id, t.session_id,
                        t.name, head(labels(t))) AS target_caption
        """
        try:
            records = await self._read(query, {"user_id": user_id, "ids": context_ids})
        except RetrievalError as error:
            self._record(
                level=0, stage="draw fallback graph", actor="graph",
                summary=f"fallback graph not drawn: {error}", selected=False,
            )
            return TraversalGraph()
        if not records or "user_eid" not in records[0]:
            return TraversalGraph()

        returned = {item.context_id for item in items}
        nodes: dict[str, GraphNode] = {}
        edges: dict[tuple[str, str, str], GraphEdge] = {}
        for record in records:
            user_eid = str(record["user_eid"])
            context_eid = str(record["context_eid"])
            target_eid = str(record["target_eid"])
            nodes.setdefault(
                user_eid,
                GraphNode(
                    id=user_eid,
                    label="User",
                    caption=str(record["user_caption"]),
                    stage=0,
                ),
            )
            nodes.setdefault(
                context_eid,
                GraphNode(
                    id=context_eid,
                    label="Context",
                    caption=str(record["caption"] or "?")[:40],
                    stage=1,
                    is_result=str(record["context_id"]) in returned,
                ),
            )
            if target_eid == user_eid:
                edges.setdefault(
                    (context_eid, "APPLIES_TO", user_eid),
                    GraphEdge(
                        source=context_eid,
                        target=user_eid,
                        type="APPLIES_TO",
                        stage=1,
                    ),
                )
            else:
                nodes.setdefault(
                    target_eid,
                    GraphNode(
                        id=target_eid,
                        label=str(record["target_label"] or "?"),
                        caption=str(record["target_caption"] or "?")[:40],
                        stage=2,
                    ),
                )
                edges.setdefault(
                    (context_eid, "APPLIES_TO", target_eid),
                    GraphEdge(
                        source=context_eid,
                        target=target_eid,
                        type="APPLIES_TO",
                        stage=1,
                    ),
                )
        return TraversalGraph(
            nodes=tuple(nodes.values()),
            edges=tuple(edges.values()),
            stages=("start at User", "Context via APPLIES_TO"),
            truncated=False,
        )

    # -- results ----------------------------------------------------------

    def _scope_notes(self, scopes: AllowedScopes) -> list[str]:
        notes = [
            "lifecycle and authorization are applied after planning: "
            f"status={ContextStatus.ACTIVE.value}, inside its validity window, "
            "own private context or a shared scope this user is a member of",
            f"{ScopeType.SESSION.value}-scoped context is excluded: it belongs to "
            "the session it was stated in",
        ]
        if not self._agent.include_shared_scopes:
            notes.append("shared scopes disabled by configuration")
        elif not scopes.shared:
            notes.append("this user is a member of no shared scope, so none was traversed")
        else:
            notes.append(
                "shared scopes allowed: "
                + ", ".join(f"{kind}:{identifier}" for kind, identifier in scopes.shared)
            )
        return notes

    def _empty(
        self, query: str, user_id: str, started: float, notes: tuple[str, ...]
    ) -> RetrievalResult:
        return RetrievalResult(
            debug=RetrievalDebug(
                user_id=user_id,
                duration_ms=_elapsed(started),
                candidates_considered=0,
                returned=0,
                query=query,
                notes=notes,
            )
        )

    async def _fall_back(
        self,
        query: str,
        user_id: str,
        started: float,
        reason: str,
        *,
        plan: TraversalPlan | None = None,
    ) -> RetrievalResult:
        """Planned retrieval failed: answer with the existing retriever instead."""
        notes = [f"planned retrieval unavailable ({reason}); used standard retrieval instead"]
        if plan is not None:
            notes.append(f"rejected plan: {plan.render()}")
        step = perf_counter()
        try:
            result = await self._fallback.retrieve(user_id, query)
        except RetrievalError as error:
            self._record(
                level=0, stage="fall back to standard retrieval", actor="fallback",
                summary=f"The safety net failed too: {error}",
                given=f"reason: {reason}", returned=f"failed: {error}",
                started=step, selected=False,
            )
            return self._empty(query, user_id, started, (*notes, f"fallback failed: {error}"))
        self._record(
            level=0,
            stage="fall back to standard retrieval",
            actor="fallback",
            summary=(
                f"Planning could not be used ({reason}), so the ordinary retriever "
                f"answered instead and returned {result.debug.returned} item(s). "
                "Nothing was guessed."
            ),
            given=(
                f"reason: {reason}\n"
                + (f"rejected plan: {plan.render()}\n" if plan is not None else "")
                + f"tool: {type(self._fallback).__name__}.retrieve(user_id, query)"
            ),
            returned=(
                f"{result.debug.returned}/{result.debug.candidates_considered} item(s)\n"
                + "\n".join(f"  - {note}" for note in result.debug.notes)
            ),
            started=step,
            reason=reason,
        )
        # Standard retrieval has no planned trail; still draw User→Context so the
        # journey panel is not blank when the answer came from the safety net.
        self._graph = await self._graph_from_items(user_id, result.items)
        return result.model_copy(
            update={
                "debug": result.debug.model_copy(
                    update={"notes": (*notes, *result.debug.notes)}
                )
            }
        )


# -- validation -----------------------------------------------------------


def repair_directions(plan: TraversalPlan, schema: GraphSchema) -> TraversalPlan:
    """Flip a hop the model wrote backwards, when the schema allows only one way.

    This corrects a direction; it can never reach a relationship that does not
    exist, and an ambiguous pair (usable both ways) is left exactly as given.
    """
    hops: list[Hop] = []
    current = plan.anchor_label
    for hop in plan.hops:
        given = (
            (current, hop.target_label) if hop.direction == "out" else (hop.target_label, current)
        )
        reverse = (given[1], given[0])
        if not schema.has_edge(given[0], hop.relationship, given[1]) and schema.has_edge(
            reverse[0], hop.relationship, reverse[1]
        ):
            hop = hop.model_copy(update={"direction": "in" if hop.direction == "out" else "out"})
        hops.append(hop)
        current = hop.target_label
    return plan.model_copy(update={"hops": hops})


def extend_with_source(
    plan: TraversalPlan, schema: GraphSchema, settings: AgentSettings
) -> TraversalPlan:
    """Continue a validated route past its Context to where that Context came from.

    Which route reaches the answer is a judgement, so the model makes it. How a
    stored fact links back to its evidence is not: it is one fixed chain in the
    ontology (``CG`` writes it on every ingest), so it is walked mechanically
    here, with no model call. Each link is resolved against the *discovered*
    schema - its direction and its target label come from the database - and the
    walk stops at the first link the graph does not have.
    """
    if plan.context_position < 0 or plan.traces_source:
        return plan
    hops = list(plan.hops)
    current = Label.CONTEXT.value
    for relationship in settings.provenance_chain:
        if len(hops) >= settings.planner_max_hops:
            break
        step = next(
            (
                Hop(relationship=relationship, direction="out", target_label=edge.end)
                if edge.start == current
                else Hop(relationship=relationship, direction="in", target_label=edge.start)
            )
            for edge in schema.edges
            if edge.type == relationship and current in (edge.start, edge.end)
        ) if any(
            edge.type == relationship and current in (edge.start, edge.end)
            for edge in schema.edges
        ) else None
        if step is None:
            break
        hops.append(step)
        current = step.target_label
    return plan.model_copy(update={"hops": hops})


#: The route from a user to their own Context, used when level 1 decides the
#: question is answered by the asker's own context. It walks through the
#: conversation that produced the Context rather than straight down APPLIES_TO,
#: so the drawn graph shows where each answer came from without a second plan.
#: The access guard still applies at the Context, so SESSION-scoped Context is
#: filtered out exactly as it is on the direct path (spec A.9).
_USER_CONTEXT_ROUTE = [
    Hop(relationship="PARTICIPATED_IN", direction="out", target_label=Label.SESSION.value),
    Hop(relationship="CONTAINS", direction="out", target_label=Label.EPISODE.value),
    Hop(relationship="SUPPORTED_BY", direction="in", target_label=Label.CONTEXT.value),
]

#: Identity property of each node label, from the ontology's own scope map.
_IDENTITY_PROPERTY = {scope.node_label: scope.id_property for scope in ScopeType}


def repair_anchor(plan: TraversalPlan, schema: GraphSchema) -> TraversalPlan:
    """Point a misnamed anchor property at the label's real identity property.

    A model that names a relationship, or a property of the wrong label, still
    told us *which node to start from*. When the ontology knows that label's
    identity property and the graph really has it, use that instead of throwing
    the whole route away. Nothing is invented: an unknown label, a missing
    identity property or an empty value leaves the plan exactly as proposed, to
    be rejected by validation.
    """
    if not plan.anchor_property or schema.has_property(plan.anchor_label, plan.anchor_property):
        return plan
    identity = _IDENTITY_PROPERTY.get(plan.anchor_label)
    if identity is None or not schema.has_property(plan.anchor_label, identity):
        return plan
    return plan.model_copy(update={"anchor_property": identity})


def _sanitise_filters(plan: TraversalPlan, scopes: AllowedScopes) -> TraversalPlan:
    """Drop optional filter values that are not real, or not this user's to read.

    Dropping a filter can only widen the *candidate* set, never the *authorized*
    one: the guard compiled around every traversal decides access on its own. A
    filter naming a scope this user may not read is therefore removed rather
    than treated as an attempt to reach it.
    """
    allowed = set(scopes.scope_types)
    known = {kind.value for kind in ContextType}
    return plan.model_copy(
        update={
            "scope_types": [scope for scope in plan.scope_types if scope in allowed],
            "context_types": [kind for kind in plan.context_types if kind in known],
        }
    )


def validate_plan(
    plan: TraversalPlan, schema: GraphSchema, settings: AgentSettings
) -> str | None:
    """Reject any structure the graph or the traversal limits forbid."""
    if not schema.has_label(plan.anchor_label):
        return f"unknown label '{plan.anchor_label}'"
    if plan.anchor_property and not schema.has_property(plan.anchor_label, plan.anchor_property):
        return f"'{plan.anchor_label}' has no property '{plan.anchor_property}'"
    if plan.anchor_property and not plan.anchor_value.strip():
        return f"no value given for '{plan.anchor_property}'"
    if len(plan.hops) > settings.planner_max_hops:
        return f"{len(plan.hops)} hops exceeds the limit of {settings.planner_max_hops}"

    current = plan.anchor_label
    for hop in plan.hops:
        if not IDENTIFIER.match(hop.relationship) or not schema.has_label(hop.target_label):
            return f"unknown relationship or label in {hop.render()}"
        start, end = (
            (current, hop.target_label) if hop.direction == "out" else (hop.target_label, current)
        )
        if not schema.has_edge(start, hop.relationship, end):
            return f"no {hop.relationship} between ({start}) and ({end})"
        current = hop.target_label

    # The route must pass through exactly one Context - that is what the tool
    # returns and what the access guard is applied to. It may reach it at the
    # end, or reach it early and carry on to the Episode, Session or Agent that
    # produced it, which is how a source is traced.
    visited = [plan.anchor_label, *(hop.target_label for hop in plan.hops)]
    found = visited.count(Label.CONTEXT.value)
    if found == 0:
        return f"the route never reaches a {Label.CONTEXT.value} node"
    if found > 1:
        return f"the route passes through {Label.CONTEXT.value} {found} times; it must be one"
    return None


# -- compilation ----------------------------------------------------------

_ALLOWED_SCOPES = """
MATCH (me:User {user_id: $user_id})
OPTIONAL MATCH (me)-[:MEMBER_OF]->(:Team)-[:BELONGS_TO*0..3]->(scope)
WITH me, collect(DISTINCT CASE
    WHEN scope:Team THEN ['TEAM', scope.team_id]
    WHEN scope:Department THEN ['DEPARTMENT', scope.department_id]
    WHEN scope:Organization THEN ['ORGANIZATION', scope.organization_id]
    ELSE NULL END) AS scopes
RETURN [s IN scopes WHERE s IS NOT NULL] AS shared
"""

#: Colours for the traversal drawing: where it started, what it passed
#: through, and what it actually found.
_ANCHOR_COLOUR = "#2563eb"
_STEP_COLOUR = "#94a3b8"
_RESULT_COLOUR = "#16a34a"

#: Identity-bearing properties, in the order a human would read them. Taken
#: from the ontology's own scope definitions rather than guessed, so a caption
#: can never name a property the graph does not have.
_CAPTION_PROPERTIES = (
    "name",
    "concept",
    "value",
    # An Episode is best recognised by what was said in it.
    "text",
    *dict.fromkeys(scope.id_property for scope in ScopeType),
    # The identities the ontology constrains that no scope maps to.
    "agent_id",
    "episode_id",
)
#: ``toStringOrNull`` rather than ``toString``: a node property may be an array
#: (an Episode's senders, for instance) and ``toString`` fails on those rather
#: than returning null, which would take the whole traversal down.
_CAPTION = (
    "coalesce("
    + ", ".join(f"left(toStringOrNull(x.{name}), 48)" for name in _CAPTION_PROPERTIES)
    + ", head(labels(x)))"
)

#: Applied to every planned traversal, after the model is done. Lifecycle first,
#: then the same access rule the pre-fetch retriever uses, expressed against the
#: scopes already resolved from the governed membership edges.
_GUARD = """
  c.status = $active_status
  AND (c.valid_from IS NULL OR c.valid_from <= $now)
  AND (c.valid_until IS NULL OR c.valid_until >= $now)
  AND (
        (c.subject_user_id = $user_id AND c.scope_type IN $private_scopes)
        OR (c.scope_type IN $shared_scopes AND [c.scope_type, c.scope_id] IN $allowed_shared)
      )
"""


def compile_plan(
    plan: TraversalPlan, user_id: str, scopes: AllowedScopes, settings: AgentSettings
) -> tuple[str, dict[str, Any]]:
    """Render a validated plan as a parameterised, read-only Cypher query.

    Only labels, relationship types and property names that were checked against
    the discovered schema are interpolated; every value is a parameter.
    """
    parameters: dict[str, Any] = {
        "user_id": user_id,
        "now": utcnow(),
        "active_status": ContextStatus.ACTIVE.value,
        "private_scopes": _PRIVATE_SCOPES,
        "shared_scopes": _SHARED_SCOPES if settings.include_shared_scopes else [],
        "allowed_shared": [list(pair) for pair in scopes.shared],
        "row_limit": settings.planner_max_rows,
        "quote_chars": settings.evidence_quote_chars,
        "property_limit": settings.max_node_properties,
    }

    path = f"(n0:{plan.anchor_label})"
    for index, hop in enumerate(plan.hops, start=1):
        arrow = (
            f"-[:{hop.relationship}]->" if hop.direction == "out" else f"<-[:{hop.relationship}]-"
        )
        path += f"{arrow}(n{index}:{hop.target_label})"
    terminal = f"n{plan.context_position}"

    # The anchor is filtered while its variable is still in scope; everything
    # after the WITH speaks only about the Context that was landed on.
    anchor_where = ""
    if plan.anchor_property:
        parameters["anchor_value"] = plan.anchor_value
        # Matched the way the existing lookup tool matches a name the user typed:
        # a partial name must still find the node. Access is unaffected - the
        # guard below decides what may be read, however many anchors match.
        anchor_where = (
            f"WHERE toLower(toString(n0.{plan.anchor_property})) "
            "CONTAINS toLower($anchor_value)\n"
        )

    where = [_GUARD.strip()]
    if plan.scope_types:
        parameters["scope_types"] = plan.scope_types
        where.append("c.scope_type IN $scope_types")
    if plan.context_types:
        parameters["context_types"] = plan.context_types
        where.append("c.context_type IN $context_types")
    if plan.keywords:
        parameters["keywords"] = plan.keywords
        where.append(
            "any(word IN $keywords WHERE toLower(c.concept_key) CONTAINS word "
            "OR toLower(toString(c.value)) CONTAINS word)"
        )

    # One representative path per Context, so the UI can draw exactly what was
    # walked without changing how many candidates the traversal considers.
    return (
        f"MATCH trail = {path}\n"
        f"{anchor_where}"
        f"WITH {terminal} AS c, collect(trail)[0] AS trail\n"
        "WHERE " + "\n  AND ".join(where) + "\n"
        "OPTIONAL MATCH (c)-[:SUPPORTED_BY]->(e:Episode)\n"
        "WITH c, trail, reduce(first = null, ep IN collect(e) |\n"
        "        CASE WHEN first IS NULL OR ep.timestamp < first.timestamp\n"
        "             THEN ep ELSE first END) AS evidence\n"
        "RETURN c AS node, left(evidence.text, $quote_chars) AS quote,\n"
        f"       [x IN nodes(trail) | {{id: elementId(x), label: head(labels(x)),\n"
        f"        caption: {_CAPTION},\n"
        "        props: [k IN [key IN keys(x) WHERE key <> 'embedding'\n"
        "                       AND toStringOrNull(x[key]) IS NOT NULL][..$property_limit]\n"
        "                | [k, left(toStringOrNull(x[k]), 80)]]}] AS path_nodes,\n"
        "       [r IN relationships(trail) | {type: type(r),\n"
        "        source: elementId(startNode(r)), target: elementId(endNode(r))}] AS path_edges\n"
        "ORDER BY c.evidence_count DESC, c.observed_at DESC\n"
        "LIMIT $row_limit",
        parameters,
    )


# -- helpers --------------------------------------------------------------


def _to_item(
    node: dict[str, Any], quote: Any, query_vector: list[float] | None
) -> RetrievedContext:
    """Convert a stored Context node into the agent's view of it."""
    instance = mapping.context_from_node(node)
    return RetrievedContext(
        context_id=instance.context_id,
        concept_key=instance.concept_key,
        context_type=instance.context_type,
        value=str(instance.value),
        scope_type=instance.scope_type,
        scope_id=instance.scope_id,
        condition=instance.condition,
        evidence_count=int(node.get("evidence_count") or 0),
        observed_at=instance.observed_at,
        valid_until=instance.valid_until,
        evidence_quote=str(quote) if quote else None,
        similarity=_cosine(query_vector, node.get("embedding")),
    )


def _keywords(words: list[str], settings: AgentSettings) -> list[str]:
    """Normalise and bound the keywords used for value matching."""
    cleaned = [
        word.strip().lower()
        for word in words
        if len(word.strip()) > 2 and word.strip().lower() not in _STOPWORDS
    ]
    return list(dict.fromkeys(cleaned))[: settings.planner_max_keywords]


def _fallback_keywords(query: str) -> list[str]:
    """Content words of the query, when the model supplied none."""
    return re.findall(r"[A-Za-z0-9_]+", query.lower())


def _build_graph(
    plan: TraversalPlan,
    rows: list[dict[str, Any]],
    items: tuple[RetrievedContext, ...],
    settings: AgentSettings,
) -> TraversalGraph:
    """Turn the paths the traversal walked into a drawable, hop-by-hop graph."""
    returned = {item.context_id for item in items}
    nodes: dict[str, GraphNode] = {}
    edges: dict[tuple[str, str, str], GraphEdge] = {}
    depth = 0

    for row in rows:
        path = list(row.get("path_nodes") or [])
        position = {str(entry["id"]): index for index, entry in enumerate(path)}
        depth = max(depth, len(path) - 1)
        for index, entry in enumerate(path):
            identifier = str(entry["id"])
            if identifier in nodes or len(nodes) >= settings.max_graph_nodes:
                continue
            caption = str(entry.get("caption") or entry.get("label") or "?")
            nodes[identifier] = GraphNode(
                id=identifier,
                label=str(entry.get("label") or "?"),
                caption=caption,
                properties={
                    str(pair[0]): str(pair[1]) for pair in (entry.get("props") or []) if pair
                },
                stage=index,
                # The last node of the path is the Context itself; mark the ones
                # that survived ranking so the drawing shows what was answered with.
                is_result=index == plan.context_position
                and str(dict(row["node"]).get("context_id")) in returned,
            )
        for edge in row.get("path_edges") or []:
            source, target = str(edge["source"]), str(edge["target"])
            if source not in nodes or target not in nodes:
                continue
            key = (source, str(edge["type"]), target)
            edges.setdefault(
                key,
                GraphEdge(
                    source=source,
                    target=target,
                    type=str(edge["type"]),
                    stage=max(position.get(source, 0), position.get(target, 0)),
                ),
            )

    stages = [f"start at {plan.anchor_label}"]
    for index, hop in enumerate(plan.hops, start=1):
        trace = " (tracing the source)" if index > plan.context_position else ""
        stages.append(f"follow {hop.relationship} to {hop.target_label}{trace}")
    return TraversalGraph(
        nodes=tuple(nodes.values()),
        edges=tuple(edges.values()),
        stages=tuple(stages[: depth + 1]) or (f"start at {plan.anchor_label}",),
        truncated=len(nodes) >= settings.max_graph_nodes,
    )


def _describe(reply: BaseModel) -> str:
    """One sentence saying what the model chose, for a first-time reader."""
    if isinstance(reply, SchemaFocus):
        if reply.answered_by_user_context:
            return (
                "The model read the map of the graph and decided the answer is in "
                "your own stored context, so no journey is needed."
                + (f" Reason: {reply.reason}" if reply.reason else "")
            )
        return (
            "The model read the map of the graph and thinks the answer lives around: "
            + (", ".join(reply.labels) or "nothing it could name")
            + (f". Reason: {reply.reason}" if reply.reason else ".")
        )
    if isinstance(reply, TraversalPlan):
        route = reply.render().split("  [")[0]
        return (
            f"The model proposed a route through the graph: {route}"
            + (f". Reason: {reply.reason}" if reply.reason else ".")
            + (" It says it still needs to look further." if reply.needs_deeper_traversal else "")
        )
    return "The model replied."


def _render_parameters(parameters: dict[str, Any]) -> str:
    """Query parameters, readable, with the long vectors of values kept short."""
    return "\n".join(
        f"  ${name} = {value if len(str(value)) <= 200 else str(value)[:200] + ' ...'}"
        for name, value in sorted(parameters.items())
    )


def _count_by_scope(items: tuple[RetrievedContext, ...]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        counts[item.scope_type.value] = counts.get(item.scope_type.value, 0) + 1
    return counts


def _elapsed(started: float) -> float:
    return round((perf_counter() - started) * 1000, 1)
