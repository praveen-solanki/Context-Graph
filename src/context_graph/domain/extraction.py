"""Extraction envelope: what the LLM may return, and the batches it runs on.

The envelope is deliberately small (schema-growth rule, spec section 3):
concepts grow without the prompt turning into a global attribute list.
"""

from __future__ import annotations

from typing import Self

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_validator,
    model_validator,
)

from context_graph.domain.conversation import Episode
from context_graph.domain.enums import Cardinality, ContextType, ScopeType, SignalOperation
from context_graph.domain.ids import canonical_value, deterministic_id

#: Spellings of "no value" models write as strings instead of JSON null.
_NULL_LITERALS = frozenset({"", "null", "none"})


def _null_literal_to_none(value: object) -> object:
    """Read a stringified null as absent, so it never becomes a stored value."""
    if isinstance(value, str) and value.strip().casefold() in _NULL_LITERALS:
        return None
    return value


class ScopeCue(BaseModel):
    """A scope the LLM believes a statement concerns.

    A cue is an observation only: the writer policy decides whether the source
    may create or publish context at that scope.
    """

    model_config = ConfigDict(frozen=True)

    scope_type: ScopeType
    scope_ref: str | None = Field(
        default=None,
        description=(
            "Name mentioned in the conversation for that scope, e.g. 'Apollo' or "
            "'Search'. Required for TASK, PROJECT, TEAM, DEPARTMENT and "
            "ORGANIZATION; null for USER and SESSION."
        ),
    )

    _absent_scope_ref = field_validator("scope_ref", mode="before")(_null_literal_to_none)


class ExtractedSignal(BaseModel):
    """One structured signal interpreted from conversation text."""

    model_config = ConfigDict(frozen=True)

    concept: str = Field(
        min_length=1,
        max_length=80,
        description=(
            "Short reusable lowercase snake_case anchor for what this is about, "
            "e.g. response_detail, deployment_day, review_requirement. Not a "
            "sentence and not the value."
        ),
    )
    #: Scalar only. A structured payload cannot be expressed as a strict JSON
    #: schema, and a scoped value is a state ("concise"), not a document;
    #: ``ContextInstance.value`` still accepts richer values from other writers.
    value: str | float | bool
    context_type: ContextType = ContextType.OTHER
    operation: SignalOperation = SignalOperation.ASSERT
    subject: str = Field(
        default="user",
        description=(
            "WHO the context is about, never what it is about. Use 'user' whenever "
            "it concerns the speaker, including things they report about their own "
            "team. Use a person's name only when the context is explicitly about "
            "someone else."
        ),
    )
    scope_cue: ScopeCue | None = None
    condition: str | None = None
    cardinality: Cardinality | None = None
    valid_from: AwareDatetime | None = None
    valid_until: AwareDatetime | None = None
    temporal_cue: str | None = None
    #: Explicit durability language ("always", "from now on", "remember").
    durability_cue: bool = False
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    evidence_quote: str | None = None
    source_episode_ids: tuple[str, ...] = Field(min_length=1)

    _absent_optionals = field_validator(
        "scope_cue",
        "condition",
        "cardinality",
        "valid_from",
        "valid_until",
        "temporal_cue",
        "evidence_quote",
        mode="before",
    )(_null_literal_to_none)

    def identity_key(self) -> tuple[str, ...]:
        """Meaning-equality key used to merge duplicate extractions."""
        cue = self.scope_cue
        return (
            self.concept.casefold(),
            canonical_value(self.value),
            self.operation.value,
            self.subject.casefold(),
            cue.scope_type.value if cue else "",
            (cue.scope_ref or "").casefold() if cue else "",
            (self.condition or "").casefold(),
        )

    @model_validator(mode="after")
    def _validate_window(self) -> Self:
        if (
            self.valid_from is not None
            and self.valid_until is not None
            and self.valid_until < self.valid_from
        ):
            raise ValueError("valid_until must not precede valid_from")
        return self


class ExtractionEnvelope(BaseModel):
    """Validated result of a single extraction call."""

    model_config = ConfigDict(frozen=True)

    signals: tuple[ExtractedSignal, ...] = ()
    #: References the model could not resolve from this batch alone, e.g.
    #: "the project we discussed"; used for a targeted lookup (A.6).
    unresolved_references: tuple[str, ...] = ()

    def for_episodes(self, episode_ids: frozenset[str]) -> ExtractionEnvelope:
        """Drop signals whose provenance is not in ``episode_ids``.

        Guards against a model attributing a signal to an episode that was not
        part of the batch (overlap episodes are context, not new evidence).
        """
        return ExtractionEnvelope(
            unresolved_references=self.unresolved_references,
            signals=tuple(
                signal.model_copy(
                    update={
                        "source_episode_ids": tuple(
                            episode_id
                            for episode_id in signal.source_episode_ids
                            if episode_id in episode_ids
                        )
                    }
                )
                for signal in self.signals
                if episode_ids.intersection(signal.source_episode_ids)
            )
        )


    def merged(self) -> ExtractionEnvelope:
        """Collapse duplicate signals, e.g. from chunks of one large Episode.

        Signals are equal when their meaning is equal; the merged signal keeps
        the union of supporting episodes and the strongest confidence (A.1).
        """
        merged: dict[tuple[str, ...], ExtractedSignal] = {}
        for signal in self.signals:
            key = signal.identity_key()
            existing = merged.get(key)
            if existing is None:
                merged[key] = signal
                continue
            merged[key] = existing.model_copy(
                update={
                    "source_episode_ids": tuple(
                        dict.fromkeys(existing.source_episode_ids + signal.source_episode_ids)
                    ),
                    "confidence": max(existing.confidence, signal.confidence),
                    "durability_cue": existing.durability_cue or signal.durability_cue,
                }
            )
        return ExtractionEnvelope(
            signals=tuple(merged.values()),
            unresolved_references=tuple(dict.fromkeys(self.unresolved_references)),
        )


class ConceptAdjudication(BaseModel):
    """LLM verdict on whether an extracted concept matches a known concept."""

    model_config = ConfigDict(frozen=True)

    is_match: bool
    #: ``concept_key`` of the matching candidate, when ``is_match`` is true.
    concept_key: str | None = None
    reason: str | None = None


class ProposedEpisode(BaseModel):
    """One episode boundary proposed by the model, as source references."""

    model_config = ConfigDict(frozen=True)

    span_refs: tuple[str, ...] = Field(min_length=1)
    #: True when the discussion clearly continues past the end of this window.
    open_ended: bool = False


class SegmentationProposal(BaseModel):
    """The model's grouping of one window's source spans into episodes."""

    model_config = ConfigDict(frozen=True)

    episodes: tuple[ProposedEpisode, ...] = Field(min_length=1)


class EpisodeChunk(BaseModel):
    """A unit of extraction input.

    A normal Episode is one whole chunk. An oversized Episode is split into
    temporary chunks that keep the original ``episode_id`` plus offsets, so all
    extracted signals still reference the original Episode (A.1).
    """

    model_config = ConfigDict(frozen=True)

    episode: Episode
    chunk_index: int = Field(default=0, ge=0)
    chunk_count: int = Field(default=1, ge=1)
    char_start: int = Field(default=0, ge=0)
    char_end: int = Field(ge=0)
    token_estimate: int = Field(ge=0)

    @computed_field
    @property
    def text(self) -> str:
        """Content of this chunk of the original Episode."""
        return self.episode.text[self.char_start : self.char_end]

    @property
    def episode_id(self) -> str:
        return self.episode.episode_id

    @property
    def is_partial(self) -> bool:
        return self.chunk_count > 1


class ExtractionBatch(BaseModel):
    """A token-budgeted set of chunks sent to the LLM in one call."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    session_id: str
    chunks: tuple[EpisodeChunk, ...] = Field(min_length=1)
    #: Chunks repeated from the previous batch purely for continuity.
    overlap_chunks: tuple[EpisodeChunk, ...] = ()
    token_estimate: int = Field(ge=0)

    @computed_field
    @property
    def batch_key(self) -> str:
        """Deterministic key so a retried batch reuses its stored result (A.3)."""
        return deterministic_id(
            "batch",
            self.run_id,
            self.session_id,
            *(f"{chunk.episode_id}#{chunk.chunk_index}" for chunk in self.chunks),
        )

    @property
    def episode_ids(self) -> tuple[str, ...]:
        """Distinct episodes carried by this batch, in order."""
        return tuple(dict.fromkeys(chunk.episode_id for chunk in self.chunks))

    @property
    def sequences(self) -> tuple[int, ...]:
        return tuple(dict.fromkeys(chunk.episode.sequence for chunk in self.chunks))

    def split(self) -> tuple[ExtractionBatch, ...]:
        """Halve the batch for retry after an oversized or invalid output.

        Splitting never divides an Episode's chunks across batches unless the
        batch is a single Episode, whose chunks are merged before context writes.
        """
        if len(self.chunks) == 1:
            return ()
        midpoint = len(self.chunks) // 2
        halves = (self.chunks[:midpoint], self.chunks[midpoint:])
        return tuple(
            self.model_copy(
                update={
                    "chunks": half,
                    "token_estimate": sum(chunk.token_estimate for chunk in half),
                }
            )
            for half in halves
        )
