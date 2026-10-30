"""Token-safe batching of Episodes for extraction (A.1).

Batches are sized by token budget, never by message count. An Episode larger
than the budget stays intact in Neo4j and is split into temporary chunks that
keep the original ``episode_id`` plus character offsets.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from context_graph.config import BatchingSettings, LLMSettings
from context_graph.domain.conversation import Episode
from context_graph.domain.extraction import EpisodeChunk, ExtractionBatch
from context_graph.providers.base import LLMProvider


class TokenBatcher:
    """Group Episodes into batches that fit the model's input budget."""

    def __init__(
        self, llm: LLMProvider, llm_settings: LLMSettings, batching: BatchingSettings
    ) -> None:
        self._llm = llm
        self._budget = llm_settings.input_token_budget
        self._batching = batching

    def chunk(self, episode: Episode) -> list[EpisodeChunk]:
        """Split one Episode into chunks that each fit the input budget."""
        tokens = self._llm.estimate_tokens(episode.text)
        if tokens <= self._budget:
            return [
                EpisodeChunk(
                    episode=episode,
                    char_end=len(episode.text),
                    token_estimate=tokens,
                )
            ]

        # Split on characters: the budget is a token estimate, and the chunk
        # offsets must map back onto the original Episode text exactly.
        chars_per_chunk = max(1, len(episode.text) * self._budget // max(tokens, 1))
        bounds = list(range(0, len(episode.text), chars_per_chunk))
        return [
            EpisodeChunk(
                episode=episode,
                chunk_index=index,
                chunk_count=len(bounds),
                char_start=start,
                char_end=min(start + chars_per_chunk, len(episode.text)),
                token_estimate=self._llm.estimate_tokens(
                    episode.text[start : start + chars_per_chunk]
                ),
            )
            for index, start in enumerate(bounds)
        ]

    def batches(self, run_id: str, session_id: str, episodes: Iterable[Episode]) -> list[ExtractionBatch]:
        """Build ordered batches, carrying a small overlap for continuity."""
        chunks = [chunk for episode in episodes for chunk in self.chunk(episode)]
        batches: list[ExtractionBatch] = []
        current: list[EpisodeChunk] = []
        current_tokens = 0

        for chunk in chunks:
            exceeds_budget = current and current_tokens + chunk.token_estimate > self._budget
            exceeds_count = len(current) >= self._batching.max_episodes_per_batch
            if exceeds_budget or exceeds_count:
                batches.append(self._build(run_id, session_id, current, batches))
                current, current_tokens = [], 0
            current.append(chunk)
            current_tokens += chunk.token_estimate

        if current:
            batches.append(self._build(run_id, session_id, current, batches))
        return batches

    def _build(
        self,
        run_id: str,
        session_id: str,
        chunks: Sequence[EpisodeChunk],
        previous: Sequence[ExtractionBatch],
    ) -> ExtractionBatch:
        overlap: tuple[EpisodeChunk, ...] = ()
        if previous and self._batching.overlap_episodes:
            overlap = previous[-1].chunks[-self._batching.overlap_episodes :]
        return ExtractionBatch(
            run_id=run_id,
            session_id=session_id,
            chunks=tuple(chunks),
            overlap_chunks=overlap,
            token_estimate=sum(chunk.token_estimate for chunk in chunks),
        )
