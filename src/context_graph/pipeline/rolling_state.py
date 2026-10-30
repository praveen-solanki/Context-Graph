"""Temporary rolling session state (A.6).

Working context for the extraction LLM only: never written to Neo4j or
MongoDB, never a source of truth, bounded in both item count and tokens, and
discarded when the session's batches are done.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

from context_graph.domain.extraction import ExtractedSignal


class RollingSessionState:
    """Bounded, session-scoped summary of what earlier batches established."""

    def __init__(
        self,
        *,
        max_items: int,
        token_budget: int,
        estimate_tokens: Callable[[str], int],
    ) -> None:
        self._max_items = max_items
        self._token_budget = token_budget
        self._estimate_tokens = estimate_tokens
        #: Ordered by recency; keyed by concept so a restated concept replaces itself.
        self._items: dict[str, str] = {}

    def update(self, signals: Iterable[ExtractedSignal]) -> None:
        """Fold a batch's signals into the state, dropping the stalest items."""
        for signal in signals:
            line = self._render(signal)
            self._items.pop(signal.concept, None)
            self._items[signal.concept] = line
        while len(self._items) > self._max_items:
            self._items.pop(next(iter(self._items)))

    def recall(self, reference: str, summary: str) -> None:
        """Record context found by a targeted lookup for a stale reference."""
        key = f"ref:{reference.casefold()}"
        self._items.pop(key, None)
        self._items[key] = f"- {reference} refers to: {summary}"
        while len(self._items) > self._max_items:
            self._items.pop(next(iter(self._items)))

    def render(self) -> str:
        """Render the state, trimming the oldest items to fit the token budget."""
        lines = list(self._items.values())
        while lines and self._estimate_tokens("\n".join(lines)) > self._token_budget:
            lines.pop(0)
        return "\n".join(lines)

    @staticmethod
    def _render(signal: ExtractedSignal) -> str:
        scope = signal.scope_cue.scope_type.value if signal.scope_cue else "unscoped"
        condition = f" when {signal.condition}" if signal.condition else ""
        return (
            f"- {signal.concept}={signal.value} "
            f"({signal.context_type.value}, {scope}, {signal.operation.value}){condition}"
        )
