"""Strategy selection: ``SegmentationSettings.strategy`` picks the segmenter."""

from __future__ import annotations

from pydantic import BaseModel

from context_graph.config import SegmentationSettings, Settings, SimilaritySegmentationSettings
from context_graph.domain.conversation import SessionInput
from context_graph.domain.enums import SegmentationStrategy
from context_graph.domain.extraction import SegmentationProposal
from context_graph.pipeline.orchestrator import IngestionPipeline
from context_graph.pipeline.segmentation import EpisodeBuilder
from context_graph.pipeline.similarity_segmentation import SimilarityEpisodeBuilder
from tests.factories import make_message
from tests.fakes import FakeEmbeddingProvider, FakeLLMProvider, InMemoryGraphStore, InMemoryLedger


class NoSegmentationLLM(FakeLLMProvider):
    """Fails the test if the pipeline still asks the model to propose a boundary."""

    async def structured[M: BaseModel](
        self, *, system: str, user: str, schema: type[M], max_output_tokens: int
    ) -> M:
        if schema is SegmentationProposal:
            raise AssertionError("the similarity strategy must never ask the model to segment")
        return await super().structured(
            system=system, user=user, schema=schema, max_output_tokens=max_output_tokens
        )


def session(message_count: int = 5) -> SessionInput:
    return SessionInput(
        session_id="S1",
        user_id="U1",
        messages=tuple(
            make_message(index, f"message {index} with a bit of real content in it")
            for index in range(1, message_count + 1)
        ),
    )


def pipeline(settings: Settings, llm: FakeLLMProvider) -> IngestionPipeline:
    return IngestionPipeline(
        settings=settings,
        store=InMemoryGraphStore(),
        ledger=InMemoryLedger(),
        llm=llm,
        embeddings=FakeEmbeddingProvider(),
        run_id="run-1",
    )


async def test_the_default_strategy_builds_the_llm_boundary_segmenter() -> None:
    built = pipeline(Settings(), FakeLLMProvider())

    assert isinstance(built._episodes, EpisodeBuilder)  # pyright: ignore[reportPrivateUsage]


async def test_the_similarity_strategy_builds_the_similarity_segmenter() -> None:
    settings = Settings(
        segmentation=SegmentationSettings(strategy=SegmentationStrategy.SIMILARITY)
    )

    built = pipeline(settings, FakeLLMProvider())

    assert isinstance(built._episodes, SimilarityEpisodeBuilder)  # pyright: ignore[reportPrivateUsage]


async def test_the_similarity_strategy_segments_without_ever_asking_the_model() -> None:
    settings = Settings(
        segmentation=SegmentationSettings(
            strategy=SegmentationStrategy.SIMILARITY,
            similarity=SimilaritySegmentationSettings(min_size=10, target_size=40, max_size=90),
        )
    )
    built = pipeline(settings, NoSegmentationLLM())

    report = await built.ingest(session())

    assert report.episodes_persisted > 0


async def test_similarity_settings_are_used_not_the_llm_boundary_defaults() -> None:
    """A deliberately tiny max_size forces several episodes out of one session."""
    settings = Settings(
        segmentation=SegmentationSettings(
            strategy=SegmentationStrategy.SIMILARITY,
            similarity=SimilaritySegmentationSettings(min_size=5, target_size=15, max_size=30),
        )
    )
    built = pipeline(settings, FakeLLMProvider())

    report = await built.ingest(session(message_count=10))

    assert report.episodes_persisted > 1
