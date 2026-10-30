"""LLM extraction with schema validation, retry and batch shrinking (A.8, A.13).

A partial or invalid extraction is never stored as success: the batch is split
and retried, and a batch whose validated result is already in the ledger is
never sent to the model again.
"""

from __future__ import annotations

from context_graph.config import LLMSettings, RetrySettings
from context_graph.domain.extraction import (
    ExtractedSignal,
    ExtractionBatch,
    ExtractionEnvelope,
)
from context_graph.errors import ExtractionValidationError
from context_graph.ledger.models import ExtractionBatchRecord
from context_graph.ledger.mongo import MongoLedger
from context_graph.observability import get_logger, log_event
from context_graph.prompts import EXTRACTION_SYSTEM, render_batch
from context_graph.providers.base import LLMProvider
from context_graph.retry import retry_async

_logger = get_logger("pipeline.extractor")


class ExtractionService:
    """Turn a batch of Episodes into a validated extraction envelope."""

    def __init__(
        self,
        *,
        llm: LLMProvider,
        ledger: MongoLedger,
        llm_settings: LLMSettings,
        retry_settings: RetrySettings,
    ) -> None:
        self._llm = llm
        self._ledger = ledger
        self._settings = llm_settings
        self._retry = retry_settings

    async def extract(self, batch: ExtractionBatch, rolling_state: str = "") -> ExtractionEnvelope:
        """Extract signals for ``batch``, reusing stored work where possible."""
        stored = await self._ledger.batch_record(batch.run_id, batch.batch_key)
        if (reused := self._reusable(stored)) is not None:
            log_event(
                _logger,
                "extraction.result_reused",
                batch_key=batch.batch_key,
                signals=len(reused.signals),
            )
            return reused

        await self._ledger.start_batch(batch)
        try:
            envelope = await self._extract_with_shrinking(batch, rolling_state)
        except ExtractionValidationError as error:
            await self._ledger.mark_failed(batch.run_id, batch.batch_key, str(error))
            raise
        await self._ledger.save_extraction(batch.run_id, batch.batch_key, envelope)
        return envelope

    async def _extract_with_shrinking(
        self, batch: ExtractionBatch, rolling_state: str
    ) -> ExtractionEnvelope:
        """Call the model; on invalid or oversized output, halve and retry."""
        try:
            return await retry_async(
                lambda: self._call(batch, rolling_state),
                settings=self._retry,
                description=f"extract:{batch.batch_key}",
            )
        except ExtractionValidationError as error:
            halves = batch.split()
            if not halves:
                raise
            log_event(
                _logger,
                "extraction.batch_split",
                batch_key=batch.batch_key,
                chunks=len(batch.chunks),
                reason=str(error),
            )
            signals: list[ExtractedSignal] = []
            for half in halves:
                signals.extend((await self.extract(half, rolling_state)).signals)
            return ExtractionEnvelope(signals=tuple(signals)).merged()

    async def _call(self, batch: ExtractionBatch, rolling_state: str) -> ExtractionEnvelope:
        user = render_batch(
            batch.chunks, overlap=batch.overlap_chunks, rolling_state=rolling_state
        )
        await self._verify_token_budget(user)

        envelope = await self._llm.structured(
            system=EXTRACTION_SYSTEM,
            user=user,
            schema=ExtractionEnvelope,
            max_output_tokens=self._settings.max_output_tokens,
        )
        if len(envelope.signals) > self._settings.max_signals_per_batch:
            raise ExtractionValidationError(
                f"extraction returned {len(envelope.signals)} signals, "
                f"limit is {self._settings.max_signals_per_batch}"
            )

        result = envelope.for_episodes(frozenset(batch.episode_ids)).merged()
        log_event(
            _logger,
            "extraction.completed",
            batch_key=batch.batch_key,
            episodes=len(batch.episode_ids),
            returned_signals=len(envelope.signals),
            kept_signals=len(result.signals),
            token_estimate=batch.token_estimate,
        )
        return result

    async def _verify_token_budget(self, user: str) -> None:
        """Confirm a near-budget batch against the model's own tokenizer."""
        budget = self._settings.input_token_budget
        estimate = self._llm.estimate_tokens(user)
        if estimate < budget * self._settings.token_verification_ratio:
            return
        counted = await self._llm.count_tokens(system=EXTRACTION_SYSTEM, user=user)
        if counted > budget:
            raise ExtractionValidationError(
                f"batch needs {counted} input tokens, budget is {budget}"
            )

    @staticmethod
    def _reusable(record: ExtractionBatchRecord | None) -> ExtractionEnvelope | None:
        """Return a previously validated result, if the ledger holds one."""
        if record is None or record.result is None:
            return None
        return record.result
