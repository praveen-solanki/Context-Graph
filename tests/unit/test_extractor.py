"""Extraction validation, batch shrinking and result reuse."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from context_graph.config import LLMSettings, RetrySettings
from context_graph.domain.conversation import Episode
from context_graph.domain.extraction import (
    EpisodeChunk,
    ExtractedSignal,
    ExtractionBatch,
    ExtractionEnvelope,
)
from context_graph.errors import ExtractionValidationError
from context_graph.pipeline.extractor import ExtractionService
from tests.factories import make_message, single_span_episode
from tests.fakes import FakeLLMProvider, InMemoryLedger


def chunk(sequence: int) -> EpisodeChunk:
    episode: Episode = single_span_episode("S1", make_message(sequence, f"message {sequence}"))
    return EpisodeChunk(episode=episode, char_end=len(episode.text), token_estimate=10)


def batch(*sequences: int) -> ExtractionBatch:
    chunks = tuple(chunk(sequence) for sequence in sequences)
    return ExtractionBatch(
        run_id="run-1",
        session_id="S1",
        chunks=chunks,
        token_estimate=sum(item.token_estimate for item in chunks),
    )


def signal_for(batch_item: EpisodeChunk, concept: str = "response_detail") -> ExtractedSignal:
    return ExtractedSignal(
        concept=concept, value="concise", source_episode_ids=(batch_item.episode_id,)
    )


def service(
    llm: FakeLLMProvider, ledger: InMemoryLedger, max_signals_per_batch: int = 60
) -> ExtractionService:
    return ExtractionService(
        llm=llm,
        ledger=ledger,
        llm_settings=LLMSettings(max_signals_per_batch=max_signals_per_batch),
        retry_settings=RetrySettings(max_attempts=1, base_backoff_seconds=0.0),
    )


async def test_validated_result_is_reused_instead_of_calling_the_model_again() -> None:
    ledger = InMemoryLedger()
    target = batch(1)
    envelope = ExtractionEnvelope(signals=(signal_for(target.chunks[0]),))
    await ledger.start_batch(target)
    await ledger.save_extraction(target.run_id, target.batch_key, envelope)
    llm = FakeLLMProvider([])

    result = await service(llm, ledger).extract(target)

    assert result == envelope
    assert llm.calls == []


async def test_invalid_output_splits_the_batch_and_never_stores_a_partial_result() -> None:
    ledger = InMemoryLedger()
    target = batch(1, 2)
    first, second = target.split()
    responses: dict[str, BaseModel] = {
        first.chunks[0].episode_id: ExtractionEnvelope(signals=(signal_for(first.chunks[0]),)),
        second.chunks[0].episode_id: ExtractionEnvelope(signals=(signal_for(second.chunks[0], "tone"),)),
    }

    def responder(user: str, schema: type[BaseModel]) -> BaseModel:
        matches = [value for key, value in responses.items() if key in user]
        if len(matches) != 1:
            raise ExtractionValidationError("model output was truncated at max_tokens")
        return matches[0]

    llm = FakeLLMProvider(responder=responder)
    result = await service(llm, ledger).extract(target)

    assert {signal.concept for signal in result.signals} == {"response_detail", "tone"}
    assert len(llm.calls) == 3  # full batch, then each half
    assert ledger.batches[(target.run_id, target.batch_key)].result == result


async def test_single_chunk_batch_that_stays_invalid_is_recorded_as_failed() -> None:
    ledger = InMemoryLedger()
    target = batch(1)
    llm = FakeLLMProvider(
        responder=lambda user, schema: (_ for _ in ()).throw(
            ExtractionValidationError("truncated")
        )
    )

    with pytest.raises(ExtractionValidationError):
        await service(llm, ledger).extract(target)

    record = ledger.batches[(target.run_id, target.batch_key)]
    assert record.result is None
    assert record.error is not None


async def test_too_many_signals_is_treated_as_an_invalid_extraction() -> None:
    ledger = InMemoryLedger()
    target = batch(1)
    oversized = ExtractionEnvelope(
        signals=tuple(signal_for(target.chunks[0], f"concept_{index}") for index in range(5))
    )
    llm = FakeLLMProvider(responder=lambda user, schema: oversized)

    with pytest.raises(ExtractionValidationError):
        await service(llm, ledger, max_signals_per_batch=2).extract(target)
