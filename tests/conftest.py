"""Shared fixtures and test-category wiring.

Each suite folder (``unit``, ``e2e``, ``integration``, ``ui``) is also a marker,
applied automatically, so ``pytest -m unit`` selects a category. Integration
tests need live Neo4j/MongoDB and only run when ``CG_INTEGRATION=1``.
"""

from __future__ import annotations

import os

import pytest

from context_graph.domain.conversation import SessionInput
from context_graph.domain.enums import Role
from tests.factories import make_message

_CATEGORIES = frozenset({"unit", "e2e", "integration", "ui"})
_INTEGRATION_ENABLED = os.environ.get("CG_INTEGRATION") == "1"


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Mark each test with its folder's category and gate live-service tests."""
    skip_live = pytest.mark.skip(reason="set CG_INTEGRATION=1 to run against live services")
    for item in items:
        category = item.path.parent.name
        if category in _CATEGORIES:
            item.add_marker(category)
        if not _INTEGRATION_ENABLED and item.get_closest_marker("integration"):
            item.add_marker(skip_live)


@pytest.fixture
def session() -> SessionInput:
    """A short two-turn session used across pipeline tests."""
    return SessionInput(
        session_id="S1",
        user_id="U123",
        agent_ids=("agent-1",),
        messages=(
            make_message(1, "Please keep my replies concise, always."),
            make_message(2, "Sure, I will keep answers short.", Role.AGENT),
        ),
    )
