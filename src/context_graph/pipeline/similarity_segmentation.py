"""Group a session's messages into bounded episodes using text size and
embedding similarity, as an alternative to the LLM-boundary strategy.

Character count decides *when* a split is required; embedding similarity
between neighbouring text decides *where*. Exactly like the LLM-boundary
strategy (:mod:`context_graph.pipeline.segmentation`), no message text is ever
rewritten - only exact, contiguous slices of the original messages, and a
message that is split still contributes every character to exactly one span.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from context_graph.config import SimilaritySegmentationSettings
from context_graph.domain.conversation import Episode, SessionInput, SourceSpan
from context_graph.observability import get_logger, log_event
from context_graph.pipeline.segmentation import SegmentationError
from context_graph.providers.base import EmbeddingProvider

_logger = get_logger("pipeline.similarity_segmentation")

# -- protected regions -----------------------------------------------------
# Structured content (code, URLs, stack traces, tables, ...) that should stay
# intact when a message is long enough to need intra-message analysis.

_CODE_FENCE = re.compile(r"```[\s\S]*?```")
_INLINE_CODE = re.compile(r"`[^`\n]+`")
_URL = re.compile(r"https?://[^\s<>\"']+|www\.[^\s<>\"']+", re.IGNORECASE)
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_UUID = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)
_FILE_PATH = re.compile(
    r"(?:/(?:[A-Za-z0-9._-]+/)+[A-Za-z0-9._-]+"
    r"|(?:[A-Za-z]:\\(?:[^\\/:*?\"<>|\r\n]+\\)*[^\\/:*?\"<>|\r\n]+)"
    r"|(?:\.\.?/[\w./-]+))"
)
_JSONISH = re.compile(r"\{[^{}]{8,}\}|(\[[^\[\]]{8,}\])")
_QUERY = re.compile(
    r"\b(?:MATCH|MERGE|CREATE\s+(?:INDEX|CONSTRAINT|NODE)?"
    r"|SELECT|INSERT|UPDATE|DELETE|WITH|RETURN|WHERE|UNWIND)\b"
    r"[\s\S]{10,}?(?:;|\n\n|\Z)",
    re.IGNORECASE,
)
_STACK = re.compile(
    r"(?:Traceback \(most recent call last\):"
    r"|(?:^|\n)\s+File \"[^\"]+\", line \d+"
    r"|(?:^|\n)\s+at [\w.$]+\([^)]*\))",
    re.MULTILINE,
)
_TABLE = re.compile(r"(?:^|\n)(?:\|[^\n]+\|(?:\n\|[^\n]+\|)*)")
_XML = re.compile(r"</?[A-Za-z][^>]{0,200}>")

_PROTECTED_PATTERNS = (
    _CODE_FENCE,
    _INLINE_CODE,
    _URL,
    _EMAIL,
    _UUID,
    _FILE_PATH,
    _JSONISH,
    _QUERY,
    _STACK,
    _TABLE,
    _XML,
)

_LINE_CUT = re.compile(r"\n")
_WORD_CUT = re.compile(r"\S+")
_KEY_CUT = re.compile(r",\s*|\n")


def _protected_regions(text: str) -> list[tuple[int, int]]:
    """Merged half-open intervals that should stay intact when possible."""
    hits = [
        (match.start(), match.end())
        for pattern in _PROTECTED_PATTERNS
        for match in pattern.finditer(text)
    ]
    return _merge(hits)


def _segment_by_protected(text: str) -> list[tuple[int, int, bool]]:
    """Partition ``text`` into (start, end, is_protected) segments covering it."""
    regions = _protected_regions(text)
    segments: list[tuple[int, int, bool]] = []
    cursor = 0
    for start, end in regions:
        if cursor < start:
            segments.append((cursor, start, False))
        segments.append((start, end, True))
        cursor = end
    if cursor < len(text):
        segments.append((cursor, len(text), False))
    if not segments and text:
        segments.append((0, len(text), False))
    return segments


def _split_oversized(text: str, max_size: int) -> list[tuple[int, int]]:
    """Cover ``text`` with contiguous slices, preferring structural cuts."""
    if len(text) <= max_size:
        return [(0, len(text))]
    preferred = [match.end() for match in _LINE_CUT.finditer(text)]
    preferred.extend(match.end() for match in _KEY_CUT.finditer(text))
    fallback = [match.end() for match in _WORD_CUT.finditer(text)]
    return _pack_cuts(text, preferred, fallback, max_size)


def _pack_cuts(
    text: str, preferred: Sequence[int], fallback: Sequence[int], max_size: int
) -> list[tuple[int, int]]:
    """Walk ``text`` taking the furthest allowed cut that still fits ``max_size``."""
    preferred_set = {cut for cut in preferred if 0 < cut < len(text)}
    fallback_set = {cut for cut in fallback if 0 < cut < len(text)}
    bounds: list[tuple[int, int]] = []
    start = 0
    while start < len(text):
        hard_end = min(start + max_size, len(text))
        if hard_end >= len(text):
            bounds.append((start, len(text)))
            break
        preferred_cut = max((cut for cut in preferred_set if start < cut <= hard_end), default=None)
        fallback_cut = max((cut for cut in fallback_set if start < cut <= hard_end), default=None)
        end = preferred_cut or fallback_cut or hard_end
        if end <= start:
            end = hard_end
        bounds.append((start, end))
        start = end
    return bounds or [(0, len(text))]


def _merge(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not intervals:
        return []
    ordered = sorted(intervals)
    merged = [ordered[0]]
    for start, end in ordered[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


_WORD = re.compile(r"\S+")


def _word_blocks(text: str, block_words: int) -> list[tuple[int, int]]:
    """Contiguous word groups covering ``text``, including surrounding whitespace."""
    words = list(_WORD.finditer(text))
    if not words:
        return [(0, len(text))]
    bounds: list[tuple[int, int]] = []
    cursor = 0
    for start in range(0, len(words), block_words):
        next_index = start + block_words
        end = words[next_index].start() if next_index < len(words) else len(text)
        bounds.append((cursor, end))
        cursor = end
    if bounds and bounds[-1][1] < len(text):
        last_start, _ = bounds[-1]
        bounds[-1] = (last_start, len(text))
    return bounds


# -- conversation-dependency scoring ----------------------------------------
# Keeps tightly coupled turns (a question and its short answer) together.

_ACK = re.compile(
    r"^(yes|yeah|yep|yup|ok|okay|sure|no|nope|nah|thanks|thank you|"
    r"got it|right|correct|exactly|please|cool|great|perfect|done)[.!?]*$",
    re.IGNORECASE,
)
_FOLLOWUP = re.compile(
    r"^(why|how|what|when|where|who|and|then|which)[?.!\s]*$", re.IGNORECASE
)
_QUESTION_START = re.compile(
    r"^(what|why|how|when|where|who|should|could|would|can|is|are|"
    r"do|does|did|have|has|will|shall)\b",
    re.IGNORECASE,
)


def _is_tiny(text: str, limit: int) -> bool:
    return len(text.strip()) <= limit


def _is_question(text: str) -> bool:
    stripped = text.strip()
    return stripped.endswith("?") or bool(_QUESTION_START.match(stripped))


def _is_acknowledgement(text: str) -> bool:
    return bool(_ACK.match(text.strip()))


def _is_followup(text: str) -> bool:
    return bool(_FOLLOWUP.match(text.strip()))


def _dependency_penalty(previous: SourceSpan, nxt: SourceSpan, tiny_limit: int) -> float:
    """0 means independent; 1 means do not split here unless size forces it."""
    penalty = 0.0
    prev_text, next_text = previous.text, nxt.text
    if previous.role != nxt.role:
        if _is_question(prev_text):
            penalty = max(penalty, 0.9)
        if _is_question(prev_text) and _is_tiny(next_text, tiny_limit):
            penalty = 1.0
        elif not _is_tiny(prev_text, tiny_limit):
            penalty = max(penalty, 0.25)
    if _is_tiny(next_text, tiny_limit) and _is_acknowledgement(next_text):
        penalty = 1.0
    if _is_tiny(prev_text, tiny_limit) and _is_followup(prev_text):
        penalty = 1.0
    return penalty


# -- analysis blocks ---------------------------------------------------------
# Temporary embedding units used only to locate a boundary; never stored.


class _AnalysisBlock(BaseModel):
    model_config = ConfigDict(frozen=True)

    piece_index: int = Field(ge=0)
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)
    text: str
    protected: bool = False


def _analysis_blocks(
    pieces: Sequence[SourceSpan], settings: SimilaritySegmentationSettings
) -> list[_AnalysisBlock]:
    """Cover every piece with contiguous blocks. Gaps and overlaps are bugs."""
    blocks: list[_AnalysisBlock] = []
    for index, piece in enumerate(pieces):
        blocks.extend(_blocks_for_piece(index, piece, settings))
    return blocks


def _blocks_for_piece(
    piece_index: int, piece: SourceSpan, settings: SimilaritySegmentationSettings
) -> list[_AnalysisBlock]:
    text = piece.text
    # A message that already fits in the episode stays one block. Intra-message
    # word blocks exist only when the message itself is longer than max_size.
    if len(text) <= settings.max_size or len(text) < settings.min_intra_message_chars:
        return [_AnalysisBlock(piece_index=piece_index, char_start=0, char_end=len(text), text=text)]

    blocks: list[_AnalysisBlock] = []
    for start, end, is_protected in _segment_by_protected(text):
        segment = text[start:end]
        spans = (
            _split_oversized(segment, settings.max_size)
            if is_protected
            else _word_blocks(segment, settings.block_words)
        )
        for relative_start, relative_end in spans:
            abs_start, abs_end = start + relative_start, start + relative_end
            blocks.append(
                _AnalysisBlock(
                    piece_index=piece_index,
                    char_start=abs_start,
                    char_end=abs_end,
                    text=text[abs_start:abs_end],
                    protected=is_protected,
                )
            )
    return blocks or [
        _AnalysisBlock(piece_index=piece_index, char_start=0, char_end=len(text), text=text)
    ]


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    """Cosine similarity in ``[-1, 1]``. Empty vectors score 0."""
    dot = left_norm = right_norm = 0.0
    for a, b in zip(left, right, strict=True):
        dot += a * b
        left_norm += a * a
        right_norm += b * b
    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0
    return dot / math.sqrt(left_norm * right_norm)


# -- builder -----------------------------------------------------------------


class SimilarityEpisodeBuilder:
    """Turns a session's ordered messages into bounded episodes using size and
    embedding similarity instead of a model-proposed boundary."""

    def __init__(
        self, *, embeddings: EmbeddingProvider, settings: SimilaritySegmentationSettings
    ) -> None:
        self._embeddings = embeddings
        self._settings = settings

    async def build(self, session: SessionInput) -> list[Episode]:
        """Group the session into episodes, in order, covering every message."""
        pieces = [
            SourceSpan.whole_message(message, session.speaker_of(message))
            for message in session.ordered_messages()
        ]
        if not pieces:
            return []

        groups = await self._segment(pieces)
        episodes = _finalize(session.session_id, groups)
        log_event(
            _logger,
            "segmentation.session_segmented",
            strategy="similarity",
            session_id=session.session_id,
            messages=len(session.messages),
            episodes=len(episodes),
        )
        return episodes

    async def _segment(self, pieces: list[SourceSpan]) -> list[list[SourceSpan]]:
        cache: dict[str, list[float]] = {}
        remaining = list(pieces)
        episodes: list[list[SourceSpan]] = []
        while remaining:
            window = self._accumulate(remaining)
            window_size = _size(window)
            if not remaining and window_size <= self._settings.max_size:
                episodes.append(window)
                break
            left, right = await self._choose_split(window, cache)
            if not left:
                raise SegmentationError("similarity split produced an empty left side")
            episodes.append(left)
            if not right:
                continue
            remaining = right + remaining
        return episodes

    def _accumulate(self, remaining: list[SourceSpan]) -> list[SourceSpan]:
        """Pull messages until the candidate is at/over target (or a hard max)."""
        window: list[SourceSpan] = []
        window_size = 0
        settings = self._settings

        def take() -> SourceSpan:
            return remaining.pop(0)

        while remaining:
            nxt_size = len(remaining[0].text)
            if not window:
                window.append(take())
                window_size += nxt_size
                if window_size > settings.max_size:
                    break
                continue

            projected = window_size + nxt_size
            if projected <= settings.target_size:
                window.append(take())
                window_size = projected
                continue

            window.append(take())
            self._absorb_tinies(window, remaining)
            break
        return window

    def _absorb_tinies(self, window: list[SourceSpan], remaining: list[SourceSpan]) -> None:
        """Tiny replies inherit the episode that just reached the size region."""
        limit = self._settings.tiny_reply_chars
        max_size = self._settings.max_size
        window_size = _size(window)
        while remaining and _is_tiny(remaining[0].text, limit):
            nxt_size = len(remaining[0].text)
            if window_size + nxt_size > max_size:
                break
            piece = remaining.pop(0)
            window.append(piece)
            window_size += nxt_size

    async def _choose_split(
        self, window: list[SourceSpan], cache: dict[str, list[float]]
    ) -> tuple[list[SourceSpan], list[SourceSpan]]:
        blocks = _analysis_blocks(window, self._settings)
        if len(blocks) < 2:
            return _force_split(window, self._settings)

        texts = [block.text if block.text.strip() else " " for block in blocks]
        vectors = await self._embedded(texts, cache)
        scored: list[tuple[float, int, list[SourceSpan], list[SourceSpan]]] = []
        for index in range(len(blocks) - 1):
            if not _allows_intra_message_cut(window, blocks, index, self._settings.max_size):
                continue
            left, right = _split_after_block(window, blocks, index)
            if not left or not right:
                continue
            left_size = _size(left)
            if left_size < self._settings.min_size or left_size > self._settings.max_size:
                continue
            change = 1.0 - _cosine(vectors[index], vectors[index + 1])
            size_fit = _size_fit(left_size, self._settings)
            dep = _boundary_dependency(window, blocks[index], blocks[index + 1], self._settings)
            score = (
                self._settings.semantic_weight * change
                + self._settings.size_weight * size_fit
                - self._settings.dependency_weight * dep
            )
            scored.append((score, index, left, right))
        if scored:
            scored.sort(key=lambda item: (item[0], -abs(_size(item[2]) - self._settings.target_size)))
            best = scored[-1]
            return best[2], best[3]
        return _force_split(window, self._settings)

    async def _embedded(
        self, texts: Sequence[str], cache: dict[str, list[float]]
    ) -> list[list[float]]:
        """Embed ``texts``, reusing one call's results across repeated text."""
        missing = [text for text in dict.fromkeys(texts) if text not in cache]
        if missing:
            vectors = await self._embeddings.embed(missing)
            for text, vector in zip(missing, vectors, strict=True):
                cache[text] = vector
        return [cache[text] for text in texts]


def _allows_intra_message_cut(
    window: Sequence[SourceSpan], blocks: Sequence[_AnalysisBlock], index: int, max_size: int
) -> bool:
    """Keep any message that already fits in ``max_size`` in one piece."""
    left_block, right_block = blocks[index], blocks[index + 1]
    if left_block.piece_index != right_block.piece_index:
        return True
    return len(window[left_block.piece_index].text) > max_size


def _boundary_dependency(
    window: Sequence[SourceSpan],
    left_block: _AnalysisBlock,
    right_block: _AnalysisBlock,
    settings: SimilaritySegmentationSettings,
) -> float:
    if left_block.piece_index == right_block.piece_index:
        return 0.0
    return _dependency_penalty(
        window[left_block.piece_index], window[right_block.piece_index], settings.tiny_reply_chars
    )


def _size_fit(left_size: int, settings: SimilaritySegmentationSettings) -> float:
    span = max(settings.max_size - settings.min_size, 1)
    return 1.0 - abs(left_size - settings.target_size) / span


def _relative_slice(piece: SourceSpan, start: int, end: int) -> SourceSpan:
    """An exact sub-slice of ``piece``, offsets relative to ``piece.text`` itself.

    ``part``/``part_count`` are placeholders: :func:`_finalize` assigns the real
    values once every episode is known, since a message split here may end up
    spread across episodes decided much later.
    """
    return piece.slice(piece.char_start + start, piece.char_start + end, 0, 1)


def _split_after_block(
    window: Sequence[SourceSpan], blocks: Sequence[_AnalysisBlock], after_index: int
) -> tuple[list[SourceSpan], list[SourceSpan]]:
    left_block, right_block = blocks[after_index], blocks[after_index + 1]
    left: list[SourceSpan] = []
    right: list[SourceSpan] = []
    for index, piece in enumerate(window):
        if index < left_block.piece_index:
            left.append(piece)
        elif index > right_block.piece_index:
            right.append(piece)
        elif left_block.piece_index == right_block.piece_index == index:
            cut = left_block.char_end
            if 0 < cut < len(piece.text):
                left.append(_relative_slice(piece, 0, cut))
                right.append(_relative_slice(piece, cut, len(piece.text)))
            elif cut <= 0:
                right.append(piece)
            else:
                left.append(piece)
        elif index == left_block.piece_index:
            left.append(piece)
        else:
            right.append(piece)
    return left, right


def _force_split(
    window: list[SourceSpan], settings: SimilaritySegmentationSettings
) -> tuple[list[SourceSpan], list[SourceSpan]]:
    """Split between messages near target. Cut inside a message only if it exceeds max."""
    first = window[0]
    if len(first.text) > settings.max_size:
        left, right_first = _hard_split_piece(first, settings.max_size)
        return left, right_first + window[1:]

    running = 0
    best_cut = 1
    best_distance: int | None = None
    for index, piece in enumerate(window):
        nxt = running + len(piece.text)
        if nxt > settings.max_size and running > 0:
            return window[:index], window[index:]
        if index < len(window) - 1 and settings.min_size <= nxt <= settings.max_size:
            distance = abs(nxt - settings.target_size)
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_cut = index + 1
        running = nxt
    if len(window) >= 2:
        return window[:best_cut], window[best_cut:]
    return window, []


def _hard_split_piece(piece: SourceSpan, max_size: int) -> tuple[list[SourceSpan], list[SourceSpan]]:
    """Last-resort structural cut so an oversized message still yields two sides."""
    if len(piece.text) <= max_size:
        return [piece], []
    cuts = _split_oversized(piece.text, max_size)
    if not cuts or cuts[0][1] >= len(piece.text):
        cut = max(1, min(max_size, len(piece.text) - 1))
        return [_relative_slice(piece, 0, cut)], [_relative_slice(piece, cut, len(piece.text))]
    _, end = cuts[0]
    end = min(end, len(piece.text) - 1) if end >= len(piece.text) else max(end, 1)
    return [_relative_slice(piece, 0, end)], [_relative_slice(piece, end, len(piece.text))]


def _size(pieces: Sequence[SourceSpan]) -> int:
    return sum(len(piece.text) for piece in pieces)


def _finalize(session_id: str, groups: Sequence[Sequence[SourceSpan]]) -> list[Episode]:
    """Assign real part/part_count to any message split across groups, then
    build one Episode per group with a deterministic identity."""
    counts: Counter[str] = Counter(span.message_id for group in groups for span in group)
    seen: Counter[str] = Counter()
    episodes: list[Episode] = []
    for sequence, group in enumerate(groups):
        spans: list[SourceSpan] = []
        for span in group:
            seen[span.message_id] += 1
            total = counts[span.message_id]
            spans.append(
                span
                if total == 1
                else span.model_copy(update={"part": seen[span.message_id] - 1, "part_count": total})
            )
        episodes.append(Episode.from_spans(session_id=session_id, sequence=sequence, spans=spans))
    return episodes
