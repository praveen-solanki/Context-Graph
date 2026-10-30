"""Cheap relevance gate (A.7).

The gate only decides whether an Episode is worth an expensive extraction call.
It never deletes an Episode and never changes the historical record, and when
in doubt it sends the Episode to the LLM.
"""

from __future__ import annotations

import re

from context_graph.config import GateSettings
from context_graph.domain.conversation import Episode
from context_graph.domain.enums import GateDecision

#: Words that carry no durable context on their own.
_ACKNOWLEDGEMENTS = frozenset(
    """
    thanks thank thx ty ok okay okey k sure cool nice great good got it yes yeah yep no nope
    hi hello hey bye goodbye please sorry np welcome fine perfect awesome done alright right
    understood noted sounds very much you your my me i a the that this
    """.split()
)
_WORD = re.compile(r"[a-z0-9']+")


class RelevanceGate:
    """Deterministic pre-filter in front of the extraction LLM."""

    def __init__(self, settings: GateSettings) -> None:
        self._settings = settings

    def decide(self, episode: Episode) -> GateDecision:
        """Decide whether ``episode`` should be sent to extraction.

        Judged on the spoken content only: speaker labels are a rendering
        detail and must not make an acknowledgement look like content.
        """
        text = episode.content.strip()
        if len(text) < self._settings.min_content_chars:
            return GateDecision.SKIP
        if len(text) > self._settings.max_acknowledgement_chars:
            return GateDecision.EXTRACT
        words = _WORD.findall(text.casefold())
        if words and all(word in _ACKNOWLEDGEMENTS for word in words):
            return GateDecision.SKIP
        return GateDecision.EXTRACT
