"""Interactive traversal viewer: one self-contained, injection-safe HTML page."""

from __future__ import annotations

from retrieval_tester.context_tool import GraphEdge, GraphNode, TraversalGraph
from ui.graph_view import render_html


def walked_graph() -> TraversalGraph:
    """A two-hop walk: a user, their team, and the context that applies to it."""
    return TraversalGraph(
        nodes=(
            GraphNode(
                id="u", label="User", caption="Riya Menon", stage=0,
                properties={"user_id": "U001", "name": "Riya Menon"},
            ),
            GraphNode(id="t", label="Team", caption="Search", stage=1, properties={"team_id": "search"}),
            GraphNode(
                id="c", label="Context", caption="Thursday", stage=2, is_result=True,
                properties={"concept_key": "deployment_day", "value": "Thursday"},
            ),
        ),
        edges=(
            GraphEdge(source="u", target="t", type="MEMBER_OF", stage=1),
            GraphEdge(source="c", target="t", type="APPLIES_TO", stage=2),
        ),
        stages=("start at User", "follow MEMBER_OF to Team", "follow APPLIES_TO to Context"),
    )


def test_the_viewer_carries_the_whole_walk_and_nothing_unresolved() -> None:
    html = render_html(walked_graph())

    # Every placeholder was filled: no template markers survive into the page.
    assert "__DATA__" not in html and "__COLOURS__" not in html and "HEIGHTpx" not in html
    # Nodes, relationships and their captions are all present for the viewer.
    for text in ("Riya Menon", "Search", "Thursday", "MEMBER_OF", "APPLIES_TO"):
        assert text in html
    # Properties travel with the nodes so one can be inspected like a real node.
    assert "deployment_day" in html and "U001" in html
    # Multi-level traversals lay out by hop, which is what makes the levels visible.
    assert '"level": 0' in html and '"level": 2' in html
    assert 'direction: "LR"' in html and 'sortMethod: "directed"' in html


def test_the_viewer_marks_where_the_walk_started_and_what_it_found() -> None:
    payload = walked_graph().to_payload()
    assert [node["level"] for node in payload["nodes"]] == [0, 1, 2]
    assert [node["isResult"] for node in payload["nodes"]] == [False, False, True]
    assert payload["labels"] == ["Context", "Team", "User"]
    # Each label gets its own colour, as a graph browser assigns them.
    html = render_html(walked_graph())
    colours = {line for line in html.splitlines() if '"Context":' in line}
    assert colours


def test_the_viewer_survives_a_missing_cdn() -> None:
    html = render_html(walked_graph())
    assert 'if (typeof vis === "undefined")' in html
    assert "could not load" in html


def test_page_content_cannot_break_out_of_the_script_block() -> None:
    hostile = TraversalGraph(
        nodes=(
            GraphNode(
                id="x", label="Context", caption="</script><script>alert(1)</script>", stage=0
            ),
        ),
        stages=("start at Context",),
    )
    html = render_html(hostile)
    assert "</script><script>alert(1)" not in html
    assert "<\\/script>" in html
