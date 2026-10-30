"""Concept resolution: deterministic first, then semantic, then adjudication.

Deterministic identity protects against exact duplicates; embeddings find
semantically equivalent concepts; the LLM adjudicates only the genuinely
ambiguous band. A deterministic id is never used as a substitute for semantic
matching (spec section 3, A.13).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from context_graph.config import ResolutionSettings, RetrySettings
from context_graph.domain.context import DEFAULT_NAMESPACE, MemoryConcept
from context_graph.domain.enums import Cardinality
from context_graph.domain.extraction import ConceptAdjudication, ExtractedSignal
from context_graph.domain.ids import new_id, slugify
from context_graph.errors import ProviderError
from context_graph.graph.neo4j_store import Neo4jGraphStore
from context_graph.observability import get_logger, log_event
from context_graph.prompts import ADJUDICATION_SYSTEM
from context_graph.providers.base import LLMProvider
from context_graph.retry import retry_async

_logger = get_logger("pipeline.resolution")

#: Output budget for adjudication; the verdict is a handful of tokens.
_ADJUDICATION_MAX_TOKENS = 512


class ConceptResolution(BaseModel):
    """The concept a signal resolved onto, and how it was resolved."""

    model_config = ConfigDict(frozen=True)

    concept: MemoryConcept
    method: str
    adjudicator_model: str | None = None


def embedding_text_for(*, concept: str, context_type: str, value: Any) -> str:
    """Text embedded for candidate discovery, for both concept and context.

    A bare concept name embeds its surface form, not its meaning:
    ``option_a_approach`` and ``option_b_approach`` score 0.95 while genuine
    synonyms score 0.3-0.6. Including the type and the value moves the signal
    into the text, and anything still ambiguous goes to the adjudicator.

    Kept callable without a signal so stored nodes can be re-embedded exactly
    the way they were embedded originally.
    """
    return f"{concept.replace('_', ' ')} ({context_type}): {value}"


def embedding_text(signal: ExtractedSignal) -> str:
    """Text embedded for one extracted signal."""
    return embedding_text_for(
        concept=signal.concept, context_type=signal.context_type.value, value=signal.value
    )


class ConceptResolver:
    """Resolve an extracted signal onto a canonical :class:`MemoryConcept`."""

    def __init__(
        self,
        *,
        store: Neo4jGraphStore,
        llm: LLMProvider,
        settings: ResolutionSettings,
        retry_settings: RetrySettings,
    ) -> None:
        self._store = store
        self._llm = llm
        self._settings = settings
        self._retry = retry_settings

    async def resolve(
        self, signal: ExtractedSignal, concept_embedding: list[float]
    ) -> ConceptResolution:
        """Return the concept for ``signal``, creating one only when needed."""
        key = MemoryConcept.key_for(signal.concept)
        if (exact := await self._store.concept_by_key(key)) is not None:
            return ConceptResolution(concept=exact, method="deterministic")

        candidates = await self._store.search_concepts(
            concept_embedding, self._settings.candidate_top_k
        )
        if candidates:
            best, score = candidates[0]
            if score >= self._settings.accept_similarity:
                log_event(
                    _logger,
                    "resolution.concept_matched",
                    concept=signal.concept,
                    matched=best.concept_key,
                    score=round(score, 4),
                    method="embedding",
                )
                return ConceptResolution(concept=best, method="embedding")

            ambiguous = [
                (candidate, candidate_score)
                for candidate, candidate_score in candidates
                if candidate_score >= self._settings.ambiguous_similarity
            ]
            if ambiguous:
                resolved = await self._adjudicate(signal, ambiguous)
                if resolved is not None:
                    return resolved

        return ConceptResolution(
            concept=await self._create(signal, concept_embedding), method="created"
        )

    async def _adjudicate(
        self, signal: ExtractedSignal, candidates: list[tuple[MemoryConcept, float]]
    ) -> ConceptResolution | None:
        """Ask the LLM to decide between genuinely ambiguous candidates."""
        listing = "\n".join(
            f"- {candidate.concept_key}: {candidate.concept}"
            f"{f' ({candidate.description})' if candidate.description else ''}"
            for candidate, _ in candidates
        )
        user = (
            f"New concept: {signal.concept}\n"
            f"Example value: {signal.value}\n\n"
            f"Known concepts:\n{listing}"
        )
        try:
            verdict = await retry_async(
                lambda: self._llm.structured(
                    system=ADJUDICATION_SYSTEM,
                    user=user,
                    schema=ConceptAdjudication,
                    max_output_tokens=_ADJUDICATION_MAX_TOKENS,
                ),
                settings=self._retry,
                description=f"adjudicate:{signal.concept}",
            )
        except ProviderError as error:
            # An adjudicator failure must not invent an identity: fall back to
            # creating a distinct concept, which stays reviewable.
            log_event(
                _logger,
                "resolution.adjudication_failed",
                concept=signal.concept,
                error=str(error),
            )
            return None

        if not verdict.is_match or verdict.concept_key is None:
            log_event(
                _logger,
                "resolution.adjudicated_as_distinct",
                concept=signal.concept,
                candidates=[candidate.concept_key for candidate, _ in candidates],
                reason=verdict.reason,
            )
            return None
        match = next(
            (candidate for candidate, _ in candidates if candidate.concept_key == verdict.concept_key),
            None,
        )
        if match is None:
            log_event(
                _logger,
                "resolution.adjudication_rejected",
                concept=signal.concept,
                returned=verdict.concept_key,
            )
            return None
        log_event(
            _logger,
            "resolution.concept_matched",
            concept=signal.concept,
            matched=match.concept_key,
            method="adjudicated",
        )
        return ConceptResolution(
            concept=match, method="adjudicated", adjudicator_model=self._llm.model
        )

    async def _create(self, signal: ExtractedSignal, embedding: list[float]) -> MemoryConcept:
        """Create the canonical concept; concurrent writers converge on one node."""
        name = slugify(signal.concept)
        concept = MemoryConcept(
            concept_id=new_id("concept"),
            concept=name,
            concept_key=MemoryConcept.key_for(name),
            namespace=DEFAULT_NAMESPACE,
            cardinality=signal.cardinality or Cardinality.SINGLE,
            value_type=_value_type(signal.value),
        )
        stored = await self._store.merge_concept(concept, embedding)
        log_event(
            _logger,
            "resolution.concept_created",
            concept_key=stored.concept_key,
            concept_id=stored.concept_id,
        )
        return stored


def _value_type(value: Any) -> str:
    """Coarse value type recorded on the concept."""
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int | float):
        return "number"
    if isinstance(value, str):
        return "string"
    return "structured"
