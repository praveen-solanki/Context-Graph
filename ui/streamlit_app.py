"""Streamlit UI for the retrieval tester.

Rendering only: retrieval, agent calls and session saving all go through
:class:`~retrieval_tester.chat.ChatController`, the same orchestration the terminal
UI uses. The Context Graph stays read-only here.

    uv run python -m streamlit run ui/streamlit_app.py
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from context_graph.config import get_settings
from context_graph.domain.enums import Role
from retrieval_tester.context_tool import TOOL_NAME as CONTEXT_GRAPH_TOOL
from retrieval_tester.context_tool import TraceStep, TraversalGraph
from retrieval_tester.errors import RetrievalTesterError
from ui.graph_view import render_html
from retrieval_tester.models import RetrievalResult
from retrieval_tester.runtime import AppRuntime

_STATE_RUNTIME = "runtime"
_STATE_STARTED = "session_started"


def runtime() -> AppRuntime:
    """Process-wide runtime, recreated when ``CG_NEO4J__URI`` changes.

    Streamlit keeps ``st.session_state`` across reruns, and ``get_settings`` is
    cached for the process. After switching Neo4j in ``.env``, clear that cache
    and rebuild the driver so the UI does not stay on the old graph.
    """
    get_settings.cache_clear()
    configured_uri = get_settings().neo4j.uri
    existing: AppRuntime | None = st.session_state.get(_STATE_RUNTIME)
    if existing is not None and existing.neo4j_uri != configured_uri:
        existing.close()
        del st.session_state[_STATE_RUNTIME]
        st.session_state[_STATE_STARTED] = False
        existing = None
    if existing is None:
        st.session_state[_STATE_RUNTIME] = AppRuntime.create()
    value: Any = st.session_state[_STATE_RUNTIME]
    return value


def start_session(user_id: str) -> None:
    """Open a new session, surfacing any failure in the UI."""
    app = runtime()
    try:
        app.run(app.controller.start_session(user_id))
        st.session_state[_STATE_STARTED] = True
    except RetrievalTesterError as error:
        st.session_state[_STATE_STARTED] = False
        st.error(f"Could not start the session: {error}")


def render_sidebar() -> None:
    """User selection, session control and save status."""
    app = runtime()
    with st.sidebar:
        st.header("Session")
        st.caption(
            f"Neo4j\n\n`{app.neo4j_uri}` · db `{app.neo4j_database}`"
        )
        user_id = (
            st.text_input(
                "user_id",
                value=st.session_state.get("user_id", ""),
                placeholder="U-46c13060",
                help="A user that already exists in the Context Graph.",
            )
            or ""
        )
        st.session_state["user_id"] = user_id

        started = st.session_state.get(_STATE_STARTED, False)
        label = "New session" if started else "Start session"
        if st.button(label, type="primary", use_container_width=True, disabled=not user_id.strip()):
            start_session(user_id)
            st.rerun()

        if not started:
            st.info("Enter a user_id and start a session.")
            return

        session = app.controller.session
        st.caption(f"session_id\n\n`{session.session_id}`")
        st.caption(f"model: `{session.model}`  ·  turns: {len(session.turns)}")

        use_tool = st.toggle(
            "Let the agent query the graph",
            value=app.settings.enable_graph_tool,
            help=(
                "On: the agent can look up teams, people and projects itself. "
                "Off: it only sees what retrieval pre-fetched."
            ),
        )
        if use_tool != app.settings.enable_graph_tool:
            app.set_graph_tool_enabled(use_tool)

        if st.button("End session", use_container_width=True):
            app.controller.end_session()
            st.session_state[_STATE_STARTED] = False
            st.rerun()

        st.subheader("Saved")
        if app.controller.saved_to is not None:
            st.success(f"Saved to `{app.controller.saved_to}`")
            st.download_button(
                "Download session JSON",
                data=app.controller.saved_to.read_bytes(),
                file_name=app.controller.saved_to.name,
                mime="application/json",
                use_container_width=True,
            )
        previous = app.store.sessions_for(session.user_id)
        if previous:
            with st.expander(f"{len(previous)} saved session(s) for this user"):
                for path in previous:
                    st.write(f"`{path.name}`")

        st.caption(
            "Read-only: nothing from this conversation is written to the Context Graph."
        )


def render_context(retrieval: RetrievalResult, *, show_prompt: bool = True) -> None:
    """The Context handed to the agent, and why."""
    debug = retrieval.debug
    directory = retrieval.directory
    if directory.is_known:
        st.subheader("Organization")
        place = " · ".join(
            part
            for part in (directory.team_name, directory.department_name, directory.organization_name)
            if part
        )
        st.markdown(f"**{place}**")
        if directory.my_projects:
            st.caption(f"works on: {', '.join(directory.my_projects)}")
        for colleague in directory.colleagues:
            role = f" · {colleague.title}" if colleague.title else ""
            projects = f" · {', '.join(colleague.projects)}" if colleague.projects else ""
            st.caption(f"{colleague.name}{role}{projects}")
        st.caption("_Shared with the team; personal preferences stay private._")

    st.subheader(f"Retrieved context ({len(retrieval.items)})")
    if not retrieval.items:
        st.warning("No stored context was retrieved for this user.")
    for item in retrieval.items:
        with st.container(border=True):
            st.markdown(f"**{item.concept_key.split(':')[-1]}** — {item.value}")
            detail = (
                f"`{item.scope_type.value}:{item.scope_id}` · {item.context_type.value} · "
                f"evidence {item.evidence_count}"
            )
            if item.similarity is not None:
                detail += f" · relevance {item.similarity:.2f}"
            if item.condition:
                detail += f" · only when _{item.condition}_"
            st.caption(detail)
            if item.evidence_quote:
                st.caption(f'> _{item.evidence_quote}_')

    if retrieval.sessions:
        st.subheader(f"Earlier conversations ({len(retrieval.sessions)})")
        for session in retrieval.sessions:
            with st.container(border=True):
                when = session.started_at.strftime("%Y-%m-%d %H:%M") if session.started_at else "?"
                st.markdown(f"**{when}** · {session.episode_count} messages · `{session.state}`")
                st.caption(", ".join(session.concepts) or "nothing durable was recorded")
                if session.opening_message:
                    st.caption(f'opened with: _{session.opening_message}_')

    if retrieval.history:
        st.subheader(f"Changed preferences ({len(retrieval.history)})")
        for change in retrieval.history:
            st.caption(change.as_prompt_line()[2:])

    with st.expander("Retrieval debug"):
        st.json(
            {
                "user_id": debug.user_id,
                "duration_ms": debug.duration_ms,
                "candidates_considered": debug.candidates_considered,
                "returned": debug.returned,
                "truncated": debug.truncated,
                "by_scope": debug.by_scope,
                "query": debug.query,
                "relevance_used": debug.relevance_used,
                "sessions_recalled": debug.sessions_recalled,
                "changes_recalled": debug.changes_recalled,
                "directory_found": debug.directory_found,
            }
        )
        for note in debug.notes:
            st.caption(f"· {note}")

    if show_prompt:
        with st.expander("System prompt given to the agent"):
            st.code(runtime().controller.system_prompt, language="text")


#: How each actor in a trace step is shown: icon, and who it is in plain words.
_ACTORS = {
    "llm": ("🧠", "the model decided"),
    "graph": ("🗄️", "the database answered"),
    "policy": ("🛡️", "safety rules checked"),
    "embeddings": ("📐", "relevance was measured"),
    "fallback": ("↩️", "the safety net answered"),
}

def render_trace(trace: tuple[TraceStep, ...]) -> None:
    """What happened, in order, in plain words. Raw detail stays folded away."""
    if not trace:
        return

    levels = [step.level for step in trace if step.level]
    calls = [step for step in trace if step.actor == "llm"]
    st.subheader("What happened, step by step")
    st.caption(
        f"{len(trace)} steps · {len(calls)} question(s) to the model · "
        f"planning depth {max(levels, default=0)} · "
        f"{sum(step.duration_ms for step in trace) / 1000:.1f}s in total"
    )

    for index, step in enumerate(trace, start=1):
        icon, actor = _ACTORS[step.actor]
        timing = f"{step.duration_ms:.0f}ms" if step.duration_ms else ""
        unused = " · this one was not used" if not step.selected else ""
        with st.container(border=True):
            st.markdown(f"**{index}. {icon} {step.summary or step.stage}**")
            level = f"planning level {step.level} · " if step.level else ""
            st.caption(f"{level}{actor}{f' · {timing}' if timing else ''}{unused}")
            with st.expander("show the exact detail"):
                if step.detail:
                    st.json(step.detail, expanded=False)
                if step.given:
                    st.caption("what went in")
                    st.code(step.given, language="text")
                if step.returned:
                    st.caption("what came back")
                    st.code(step.returned, language="text")


def render_traversal(graph: TraversalGraph) -> None:
    """The real path through the graph, walkable hop by hop."""
    st.subheader("The journey through the graph")
    if graph.is_empty:
        st.info(
            "No journey to draw: planning fell back before a trail was walked, "
            "or no Context nodes were returned."
        )
        return

    if graph.edges:
        caption = "  →  ".join(graph.stages)
    else:
        caption = (
            "Context hits only (no relationship hops in this plan) — "
            "nodes may look disconnected until a multi-hop route is used"
        )
    st.caption(
        caption
        + f"  ·  {len(graph.nodes)} nodes and {len(graph.edges)} relationships were visited"
        + ("  ·  trimmed to keep the picture readable" if graph.truncated else "")
    )
    # Interactive viewer (needs the vis-network CDN). Always also show Graphviz
    # below so a blocked CDN or iframe quirk never leaves a blank panel.
    st.iframe(render_html(graph), height=620)
    with st.expander("static picture (always available)", expanded=not graph.edges):
        st.graphviz_chart(graph.to_dot(), width="stretch")


def render_context_graph_tool() -> None:
    """The planned Context Graph retrieval tool: query + user_id in, Context out.

    Independent of the chat session: the tool takes the user it retrieves for as
    an argument, exactly as the agent-facing tool definition declares.
    """
    st.subheader("Planned Context Graph retrieval")
    st.caption(
        f"`{CONTEXT_GRAPH_TOOL}` — plans a bounded traversal over the graph schema "
        "that was discovered from the database, level by level, then executes it "
        "with this user's lifecycle and scope rules applied."
    )

    query_column, user_column = st.columns([3, 1], gap="medium")
    with query_column:
        query = (
            st.text_input(
                "query",
                key="cg_query",
                placeholder="What has my team decided about deployments?",
            )
            or ""
        )
    with user_column:
        user_id = (
            st.text_input(
                "user_id",
                key="cg_user_id",
                value=st.session_state.get("user_id", ""),
                placeholder="U-praveen",
            )
            or ""
        )

    if st.button(
        "Run retrieval",
        key="cg_run",
        type="primary",
        disabled=not (query.strip() and user_id.strip()),
    ):
        app = runtime()
        st.session_state.pop("cg_result", None)
        st.session_state.pop("cg_answer", None)
        try:
            with st.spinner("Planning the traversal..."):
                result = app.run(app.context_graph_tool.run(query, user_id))
            st.session_state["cg_result"] = result
            st.session_state["cg_graph"] = app.context_graph_tool.last_graph
            st.session_state["cg_trace"] = app.context_graph_tool.last_trace
            with st.spinner("Answering from what was retrieved..."):
                st.session_state["cg_answer"] = app.run(app.answer(query, result))
        except RetrievalTesterError as error:
            st.error(f"{error}")

    result: Any = st.session_state.get("cg_result")
    if result is None:
        st.info("Enter a query and a user_id, then run the tool.")
        return

    answer: Any = st.session_state.get("cg_answer")
    if answer:
        st.markdown("### Answer")
        with st.container(border=True):
            st.markdown(answer)
        st.caption(
            f"Written by the model from the {len(result.items)} item(s) below and "
            "nothing else - it has no other access to the graph."
        )

    render_traversal(st.session_state.get("cg_graph") or TraversalGraph())

    result_column, trace_column = st.columns([2, 3], gap="large")
    with result_column:
        render_context(result, show_prompt=False)
    with trace_column:
        render_trace(st.session_state.get("cg_trace") or ())


def render_chat() -> None:
    """Conversation history and the input box."""
    app = runtime()
    for message in app.controller.session.turns:
        with st.chat_message("user" if message.role is Role.USER else "assistant"):
            st.markdown(message.text)

    prompt = st.chat_input("Message the agent")
    if not prompt:
        return
    with st.chat_message("user"):
        st.markdown(prompt)
    try:
        with st.spinner("Thinking..."):
            reply = app.run(app.controller.send(prompt))
    except RetrievalTesterError as error:
        st.error(f"The agent could not reply: {error}")
        return
    with st.chat_message("assistant"):
        st.markdown(reply)
    st.rerun()


def main() -> None:
    """Render the page."""
    st.set_page_config(page_title="Context Graph · retrieval tester", page_icon="🧠", layout="wide")
    st.title("Retrieval tester")
    st.caption(
        "Chats using Context retrieved from the Context Graph. The graph is read-only here: "
        "this app never writes to it and never calls the ingestion pipeline."
    )

    render_sidebar()
    chat_tab, tool_tab = st.tabs(["Chat", "Context Graph retrieval"])

    with tool_tab:
        render_context_graph_tool()

    with chat_tab:
        if not st.session_state.get(_STATE_STARTED, False):
            st.info("Start a session from the sidebar to retrieve this user's stored Context.")
            return

        chat_column, context_column = st.columns([3, 2], gap="large")
        with chat_column:
            render_chat()
            lookups = runtime().controller.last_lookups
            if lookups:
                with st.expander(
                    f"Graph lookups for the last message ({len(lookups)})", expanded=True
                ):
                    for call in lookups:
                        st.caption(f"`{call.lookup}({call.target!r})` → {call.rows} rows")
                        st.code(call.result, language="text")
        with context_column:
            render_context(runtime().controller.retrieval)


main()
