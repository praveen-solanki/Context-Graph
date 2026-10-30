"""Streamlit page: it renders offline, the context graph tool is wired in, and
the live page runs it."""

from __future__ import annotations

import ast

import pytest
from streamlit.testing.v1 import AppTest

from tests.factories import REPO_ROOT

PAGE = REPO_ROOT / "ui" / "streamlit_app.py"


def test_the_page_renders_before_any_service_is_contacted() -> None:
    """Opening the page needs no Neo4j or model: retrieval waits for a user action."""
    app = AppTest.from_file(str(PAGE), default_timeout=60).run()

    assert [exception.value for exception in app.exception] == []
    assert [error.value for error in app.error] == []
    assert [tab.label for tab in app.tabs] == ["Chat", "Context Graph retrieval"]
    assert any("Start a session from the sidebar" in info.value for info in app.info)


def test_the_ui_wires_the_tool_end_to_end() -> None:
    tree = ast.parse(PAGE.read_text())
    functions = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    source = PAGE.read_text()

    assert "render_context_graph_tool" in functions
    # query and user_id are collected, and retrieval runs through the runtime.
    assert 'key="cg_query"' in source and 'key="cg_user_id"' in source
    assert "app.context_graph_tool.run(query, user_id)" in source
    # The existing chat tool's UI is untouched.
    assert "Let the agent query the graph" in source
    assert "render_context(runtime().controller.retrieval)" in source


def test_the_ui_renders_the_answer_the_journey_and_the_trace() -> None:
    source = PAGE.read_text()
    # The retrieved context is handed to the model, and its answer is shown.
    assert "app.answer(query, result)" in source
    assert 'st.markdown("### Answer")' in source
    # The walked subgraph is drawn interactively, with a static fallback.
    assert "def render_traversal(graph: TraversalGraph) -> None:" in source
    assert "st.iframe(render_html(graph)" in source
    assert "st.graphviz_chart(graph.to_dot()" in source
    # Plain-language trace first, exact detail folded away behind it.
    assert "def render_trace(trace: tuple[TraceStep, ...]) -> None:" in source
    assert "step.summary or step.stage" in source
    assert 'st.expander("show the exact detail")' in source
    # The chat tab is untouched.
    assert 'st.subheader(f"Retrieved context ({len(retrieval.items)})")' in source
    assert "render_context(runtime().controller.retrieval)" in source


@pytest.mark.integration
def test_the_ui_runs_the_tool_and_shows_the_result() -> None:
    """Drive the real page: type a query and a user_id, run it, read the result."""
    app = AppTest.from_file(str(PAGE), default_timeout=180).run()
    assert [tab.label for tab in app.tabs] == ["Chat", "Context Graph retrieval"]

    app.text_input(key="cg_query").set_value("what do you know about me?")
    app.text_input(key="cg_user_id").set_value("U-praveen").run()
    app.button(key="cg_run").click().run()

    assert [error.value for error in app.error] == []
    headings = [heading.value for heading in app.subheader]
    assert "Planned Context Graph retrieval" in headings
    assert any(heading.startswith("Retrieved context") for heading in headings)
    assert "What happened, step by step" in headings
    assert "The journey through the graph" in headings
    assert any("question(s) to the model" in caption.value for caption in app.caption)
    # The retrieved context reached the model and an answer came back.
    rendered = [item.value for item in app.markdown]
    assert "### Answer" in rendered
    assert rendered[rendered.index("### Answer") + 1].strip()
    # Every step is stated in a sentence, not as raw prompt text.
    assert any(item.startswith("**1. ") and "You asked" in item for item in rendered)
    # Raw prompts and Cypher stay folded away behind each step.
    labels = [str(getattr(item, "label", "")) for item in app.get("expander")]
    assert labels.count("show the exact detail") >= 3

    # A question about the user's team must walk User -> Team -> Context, and
    # that whole walk must reach the interactive viewer.
    app.text_input(key="cg_query").set_value("what has my team decided about deployments?")
    app.text_input(key="cg_user_id").set_value("U-praveen").run()
    app.button(key="cg_run").click().run()

    assert [error.value for error in app.error] == []
    journey = [caption.value for caption in app.caption if "relationships were visited" in caption.value]
    assert journey and "MEMBER_OF" in journey[0] and "APPLIES_TO" in journey[0]
    frames = app.get("iframe")
    assert len(frames) == 1
    page = str(frames[0].proto)
    for token in ("vis-network", "MEMBER_OF", "APPLIES_TO", "Praveen Solanki", "Search"):
        assert token in page, token
    assert '\\"level\\": 2' in page  # the walk really reached a third level
