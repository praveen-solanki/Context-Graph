"""Group a session's messages into bounded episodes.

One message is no longer one episode. A session is processed in token-bounded
windows; the model proposes topic boundaries over ordered source references,
and this module reconstructs episodes from the original records. The model
never supplies text, identity, timestamps or metadata - only boundaries.

Guarantees enforced here, not trusted from the model:
- every source span belongs to exactly one finalized episode;
- spans keep their exact text, offsets, speaker, timestamp and metadata;
- episodes stay within a hard token limit, continuing into further episodes
  when a discussion outgrows it;
- a window's unfinished tail carries into the next window, and bounded earlier
  content is replayed as read-only context;
- when semantic grouping fails repeatedly, a size-based fallback still produces
  bounded episodes, so content is never discarded.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

from context_graph.config import LLMSettings, RetrySettings, SegmentationSettings
from context_graph.domain.conversation import Episode, MessageInput, SessionInput, SourceSpan
from context_graph.domain.extraction import SegmentationProposal
from context_graph.errors import ContextGraphError, ExtractionValidationError
from context_graph.observability import get_logger, log_event
from context_graph.prompts import SEGMENTATION_SYSTEM, render_segmentation
from context_graph.providers.base import LLMProvider
from context_graph.retry import retry_async

_logger = get_logger("pipeline.segmentation")

#: Split points for an oversized message, tried in order of preference.
_PARAGRAPH = re.compile(r"\n\s*\n")
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


class SegmentationError(ExtractionValidationError):
    """A grouping proposal did not satisfy the coverage rules."""


class EpisodeBuilder:
    """Turns a session's ordered messages into bounded, span-based episodes."""

    def __init__(
        self,
        *,
        llm: LLMProvider,
        llm_settings: LLMSettings,
        settings: SegmentationSettings,
        retry_settings: RetrySettings,
    ) -> None:
        self._llm = llm
        self._settings = settings
        self._retry = retry_settings
        self._window_budget = settings.window_budget(llm_settings.input_token_budget)

    async def build(self, session: SessionInput) -> list[Episode]:
        """Group the session into episodes, in order, covering every message."""
        spans = self.spans_for(session)
        if not spans:
            return []

        episodes: list[Episode] = []
        carried: list[SourceSpan] = []
        index = 0

        while index < len(spans) or carried:
            window, next_index = self._next_window(spans, index, carried)
            # Finalize everything when the supplied session ends, and also when a
            # window could not take in new content - otherwise an always-open
            # tail would carry forever.
            must_finalize = next_index >= len(spans) or next_index == index
            context = self._context_spans(spans, window)
            index = next_index
            groups, carried = await self._group(
                session, window, context, must_finalize=must_finalize
            )
            for group in groups:
                episodes.extend(self._finalize(session, group, len(episodes), episodes))

        log_event(
            _logger,
            "segmentation.session_segmented",
            session_id=session.session_id,
            messages=len(session.messages),
            spans=len(spans),
            episodes=len(episodes),
        )
        return episodes

    # ------------------------------------------------------------------ spans

    def spans_for(self, session: SessionInput) -> list[SourceSpan]:
        """Every message as one span, splitting any message that is too large."""
        spans: list[SourceSpan] = []
        for message in session.ordered_messages():
            whole = SourceSpan.whole_message(message, session.speaker_of(message))
            if self._tokens(whole.text) <= self._settings.max_span_tokens:
                spans.append(whole)
                continue
            spans.extend(self._split_message(whole, message))
        return spans

    def _split_message(self, whole: SourceSpan, message: MessageInput) -> list[SourceSpan]:
        """Split one oversized message into bounded spans with exact offsets."""
        bounds = self._split_points(message.text)
        parts = [
            whole.slice(start, end, part, len(bounds))
            for part, (start, end) in enumerate(bounds)
        ]
        log_event(
            _logger,
            "segmentation.message_split",
            message_id=message.message_id,
            chars=len(message.text),
            parts=len(parts),
        )
        return parts

    def _split_points(self, text: str) -> list[tuple[int, int]]:
        """Offset pairs covering ``text`` exactly, preferring natural breaks."""
        limit = self._char_budget(self._settings.max_span_tokens)
        cuts = self._candidate_cuts(text)

        bounds: list[tuple[int, int]] = []
        start = 0
        while start < len(text):
            hard_end = min(start + limit, len(text))
            if hard_end >= len(text):
                bounds.append((start, len(text)))
                break
            # The furthest natural break that still fits, else an exact cut.
            end = max((cut for cut in cuts if start < cut <= hard_end), default=hard_end)
            bounds.append((start, end))
            start = end
        return bounds or [(0, len(text))]

    @staticmethod
    def _candidate_cuts(text: str) -> list[int]:
        """Paragraph breaks first, then sentence breaks; may be empty."""
        cuts = [match.end() for match in _PARAGRAPH.finditer(text)]
        cuts.extend(match.end() for match in _SENTENCE.finditer(text))
        return sorted(set(cuts))

    # ---------------------------------------------------------------- windows

    def _next_window(
        self, spans: Sequence[SourceSpan], index: int, carried: Sequence[SourceSpan]
    ) -> tuple[list[SourceSpan], int]:
        """Carried tail plus as many new spans as the token budget allows."""
        window = list(carried)
        used = sum(self._tokens(span.text) for span in window)
        while index < len(spans):
            cost = self._tokens(spans[index].text)
            if window and used + cost > self._window_budget:
                break
            window.append(spans[index])
            used += cost
            index += 1
        return window, index

    def _context_spans(
        self, spans: Sequence[SourceSpan], window: Sequence[SourceSpan]
    ) -> list[SourceSpan]:
        """Bounded, already-assigned content shown only for continuity."""
        if not self._settings.context_spans or not window:
            return []
        first = window[0]
        position = next(
            (
                offset
                for offset, span in enumerate(spans)
                if (span.sequence, span.part) == (first.sequence, first.part)
            ),
            0,
        )
        start = max(0, position - self._settings.context_spans)
        return list(spans[start:position])

    # --------------------------------------------------------------- grouping

    async def _group(
        self,
        session: SessionInput,
        window: Sequence[SourceSpan],
        context: Sequence[SourceSpan],
        *,
        must_finalize: bool,
    ) -> tuple[list[list[SourceSpan]], list[SourceSpan]]:
        """Return finalized groups for this window, plus the unfinished tail."""
        by_ref = {span.ref: span for span in window}

        proposal: SegmentationProposal | None = None
        rejection: str | None = None
        for attempt in range(1, self._settings.max_attempts + 1):
            try:
                candidate = await self._propose(window, context, rejection)
                self._validate(candidate, by_ref)
                proposal = candidate
                break
            except ContextGraphError as error:
                rejection = str(error)
                log_event(
                    _logger,
                    "segmentation.proposal_rejected",
                    session_id=session.session_id,
                    attempt=attempt,
                    spans=len(window),
                    error=str(error),
                )

        if proposal is None:
            log_event(
                _logger,
                "segmentation.size_fallback",
                session_id=session.session_id,
                spans=len(window),
            )
            return [list(group) for group in self._size_groups(window)], []

        groups = [[by_ref[ref] for ref in episode.span_refs] for episode in proposal.episodes]
        if must_finalize or not proposal.episodes[-1].open_ended or len(groups) == 1:
            return groups, []
        log_event(
            _logger,
            "segmentation.tail_carried",
            session_id=session.session_id,
            spans=len(groups[-1]),
        )
        return groups[:-1], groups[-1]

    async def _propose(
        self,
        window: Sequence[SourceSpan],
        context: Sequence[SourceSpan],
        rejection: str | None = None,
    ) -> SegmentationProposal:
        """Ask the model for boundaries over this window's references."""
        user = render_segmentation(
            [(span.ref, span.role.value, span.text) for span in window],
            [(span.ref, span.role.value, span.text) for span in context],
            rejection,
        )
        return await retry_async(
            lambda: self._llm.structured(
                system=SEGMENTATION_SYSTEM,
                user=user,
                schema=SegmentationProposal,
                max_output_tokens=self._settings.max_output_tokens,
            ),
            settings=self._retry,
            description="segment-window",
        )

    @staticmethod
    def _validate(proposal: SegmentationProposal, by_ref: dict[str, SourceSpan]) -> None:
        """Every reference used exactly once, in order, and nothing invented."""
        assigned = [ref for episode in proposal.episodes for ref in episode.span_refs]
        unknown = [ref for ref in assigned if ref not in by_ref]
        if unknown:
            raise SegmentationError(f"proposal references unknown spans: {unknown[:5]}")
        if len(set(assigned)) != len(assigned):
            raise SegmentationError("proposal assigns a span to more than one episode")
        missing = [ref for ref in by_ref if ref not in set(assigned)]
        if missing:
            raise SegmentationError(f"proposal leaves spans unassigned: {missing[:5]}")

        order = [(by_ref[ref].sequence, by_ref[ref].part) for ref in assigned]
        if order != sorted(order):
            raise SegmentationError("proposal reorders the conversation")

    def _size_groups(self, window: Sequence[SourceSpan]) -> list[list[SourceSpan]]:
        """Deterministic fallback: fill episodes to the size limit, in order."""
        groups: list[list[SourceSpan]] = []
        current: list[SourceSpan] = []
        used = 0
        for span in window:
            cost = self._tokens(span.text)
            if current and used + cost > self._settings.max_episode_tokens:
                groups.append(current)
                current, used = [], 0
            current.append(span)
            used += cost
        if current:
            groups.append(current)
        return groups

    # ------------------------------------------------------------ finalizing

    def _finalize(
        self,
        session: SessionInput,
        group: Sequence[SourceSpan],
        ordinal: int,
        existing: Sequence[Episode],
    ) -> list[Episode]:
        """Build episodes for one group, continuing it if it exceeds the limit."""
        episodes: list[Episode] = []
        previous_id = existing[-1].episode_id if existing else None
        for index, bounded in enumerate(self._bounded(group)):
            episode = Episode.from_spans(
                session_id=session.session_id,
                sequence=ordinal + index,
                spans=bounded,
                continues_episode_id=previous_id if index else None,
            )
            previous_id = episode.episode_id
            episodes.append(episode)
        return episodes

    def _bounded(self, group: Sequence[SourceSpan]) -> list[list[SourceSpan]]:
        """Split a group that outgrew the episode limit into continuations."""
        if self._tokens_of(group) <= self._settings.max_episode_tokens:
            return [list(group)]
        return self._size_groups(group)

    # -------------------------------------------------------------- utilities

    def _tokens(self, text: str) -> int:
        return self._llm.estimate_tokens(text)

    def _tokens_of(self, spans: Iterable[SourceSpan]) -> int:
        return sum(self._tokens(span.text) for span in spans)

    def _char_budget(self, tokens: int) -> int:
        """Characters that fit in ``tokens``, using the model's own estimate."""
        probe = "x" * 1_000
        per_char = max(self._tokens(probe), 1) / len(probe)
        return max(1, int(tokens / per_char))
