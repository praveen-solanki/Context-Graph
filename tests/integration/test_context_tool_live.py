"""Compiled traversal plans are valid Cypher against the live graph."""

from __future__ import annotations

from typing import Any, cast

import pytest
from neo4j import RoutingControl

from retrieval_tester.config import AgentSettings
from retrieval_tester.context_tool import AllowedScopes, Hop, TraversalPlan, compile_plan

SCOPES = AllowedScopes(user_exists=True, shared=(("TEAM", "T1"),))


@pytest.mark.integration
async def test_compiled_plans_are_valid_cypher_against_the_live_graph() -> None:
    """Every shape the compiler can emit is EXPLAINed by the real database."""
    from neo4j import AsyncGraphDatabase

    from context_graph.config import get_settings

    settings = get_settings().neo4j
    plans = [
        TraversalPlan(anchor_label="Context", keywords=["deploy"]),
        TraversalPlan(
            anchor_label="User",
            anchor_property="user_id",
            anchor_value="U1",
            hops=[Hop(relationship="APPLIES_TO", direction="in", target_label="Context")],
            scope_types=["USER"],
            context_types=["Preference"],
        ),
        TraversalPlan(
            anchor_label="User",
            anchor_property="name",
            anchor_value="Riya",
            hops=[
                Hop(relationship="MEMBER_OF", direction="out", target_label="Team"),
                Hop(relationship="APPLIES_TO", direction="in", target_label="Context"),
            ],
        ),
    ]
    driver = AsyncGraphDatabase.driver(
        settings.uri, auth=(settings.user, settings.password.get_secret_value())
    )
    try:
        for plan in plans:
            cypher, parameters = compile_plan(plan, "U1", SCOPES, AgentSettings())
            await driver.execute_query(
                cast(Any, f"EXPLAIN {cypher}"),
                parameters,
                database_=settings.database,
                routing_=RoutingControl.READ,
            )
    finally:
        await driver.close()
