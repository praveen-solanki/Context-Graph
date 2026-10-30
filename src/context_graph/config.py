"""Single source of truth for configuration, budgets, thresholds and policies.

Values are environment-overridable with the ``CG_`` prefix and ``__`` as the
nested delimiter, e.g. ``CG_LLM__MODEL`` or ``CG_NEO4J__URI``.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Self

from enum import StrEnum

from pydantic import (
    AliasChoices,
    BaseModel,
    Field,
    SecretStr,
    computed_field,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

from context_graph.domain.enums import ScopeType, SegmentationStrategy

#: Repository root: src/context_graph/config.py -> src/context_graph -> src -> root.
_REPO_ROOT = Path(__file__).resolve().parents[2]
#: The ConvoMem demo set prepared by ``python -m convomem_benchmark_converter.prepare``.
DEFAULT_INPUT = _REPO_ROOT / "data" / "convomem_demo" / "sessions.json"


class Neo4jSettings(BaseModel):
    """Connection settings for the durable graph store."""

    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: SecretStr = SecretStr("neo4j")
    database: str = "neo4j"


class MongoSettings(BaseModel):
    """Connection settings for the processing ledger."""

    uri: str = "mongodb://localhost:27017"
    database: str = "context_graph"


class AuthScheme(StrEnum):
    """How the chat backend expects its credential."""

    BEARER = "bearer"
    API_KEY = "api-key"


class StructuredOutputMode(StrEnum):
    """How structured results are obtained from the chat backend."""

    #: Server-enforced JSON schema. Needs a backend that supports it.
    JSON_SCHEMA = "json_schema"
    #: JSON mode with the schema in the prompt, validated and repaired here.
    JSON_OBJECT = "json_object"


class ChatBackendSettings(BaseSettings):
    """The chat model used for extraction, adjudication and the retrieval tester.

    Defaults target NVIDIA NIM (``nvidia/nemotron-3.5-bolt-e4b``). Every
    difference between backends - credential header, token-cap parameter name,
    structured-output support, extra gateway headers - is configuration, so
    swapping backends never means changing provider code.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", populate_by_name=True)

    base_url: str = Field(
        default="https://integrate.api.nvidia.com/v1",
        validation_alias=AliasChoices("LLM_BASE_URL", "OPENAI_BASE_URL"),
    )
    api_key: SecretStr = Field(
        default=SecretStr("local-dev"),
        validation_alias=AliasChoices("LLM_API_KEY", "OPENAI_API_KEY", "NVIDIA_API_KEY"),
    )
    chat_model: str = Field(
        default="nvidia/nemotron-3.5-bolt-e4b",
        validation_alias=AliasChoices("LLM_MODEL", "MODEL_NAME"),
    )
    auth_scheme: AuthScheme = Field(default=AuthScheme.BEARER, validation_alias=AliasChoices("LLM_AUTH_SCHEME"))
    #: gpt-5.x rejects ``max_tokens``; the orchestrator rejects the other name.
    token_parameter: str = Field(default="max_tokens", validation_alias=AliasChoices("LLM_TOKEN_PARAMETER"))
    structured_output: StructuredOutputMode = Field(
        default=StructuredOutputMode.JSON_OBJECT, validation_alias=AliasChoices("LLM_STRUCTURED_OUTPUT")
    )
    #: Gateway header the orchestrator requires, e.g. ``ai-model-service-code``.
    service_code_header: str = Field(
        default="ai-model-service-code", validation_alias=AliasChoices("LLM_SERVICE_CODE_HEADER")
    )
    service_code: str | None = Field(
        default=None, validation_alias=AliasChoices("LLM_SERVICE_CODE")
    )
    #: Attempts to repair invalid JSON before the call is treated as failed.
    repair_attempts: int = Field(default=2, ge=0, validation_alias=AliasChoices("LLM_REPAIR_ATTEMPTS"))
    #: Long extractions need room; short calls answer in about a second.
    request_timeout_seconds: float = Field(
        default=360.0, validation_alias=AliasChoices("LLM_REQUEST_TIMEOUT_SECONDS")
    )
    #: The orchestrator pod's throughput knee is around 7-8 in-flight requests.
    max_concurrent_requests: int = Field(
        default=8, gt=0, validation_alias=AliasChoices("LLM_MAX_CONCURRENCY")
    )
    temperature: float = Field(default=0.0, ge=0.0, le=2.0, validation_alias=AliasChoices("LLM_TEMPERATURE"))

    @property
    def chat_completions_url(self) -> str:
        """Absolute chat completions URL for the configured backend."""
        return f"{self.base_url.rstrip('/')}/chat/completions"

    def headers(self) -> dict[str, str]:
        """Credential and gateway headers for every chat request."""
        key = self.api_key.get_secret_value()
        headers = (
            {"Authorization": f"Bearer {key}"}
            if self.auth_scheme is AuthScheme.BEARER
            else {"api-key": key}
        )
        if self.service_code:
            headers[self.service_code_header] = self.service_code
        return headers


class LLMSettings(BaseModel):
    """Extraction budgets and limits; the backend lives in :class:`ChatBackendSettings`."""

    #: Deliberately below the default model's window: it bounds batch size, and
    #: so extraction quality, not just what the model accepts.
    context_limit_tokens: int = Field(default=10_240, gt=0)
    #: Must hold ``max_signals_per_batch`` signals (~130 tokens each), or full
    #: batches are truncated and re-run as halves.
    max_output_tokens: int = Field(default=4_000, gt=0)
    prompt_overhead_tokens: int = Field(default=800, ge=0)
    safety_margin_tokens: int = Field(default=400, ge=0)
    rolling_state_token_budget: int = Field(default=600, ge=0)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    #: Characters per token used by the local token estimator.
    chars_per_token: float = Field(default=3.5, gt=0)
    #: Hard cap on signals accepted from one extraction call (A.13 budget rule).
    max_signals_per_batch: int = Field(default=30, gt=0)
    #: Verify the batch with the model tokenizer once the local estimate reaches
    #: this fraction of the input budget.
    token_verification_ratio: float = Field(default=0.8, gt=0.0, le=1.0)

    @computed_field
    @property
    def input_token_budget(self) -> int:
        """Tokens available for episode content in a single extraction call."""
        budget = (
            self.context_limit_tokens
            - self.max_output_tokens
            - self.prompt_overhead_tokens
            - self.safety_margin_tokens
            - self.rolling_state_token_budget
        )
        if budget <= 0:
            raise ValueError("LLM token budget leaves no room for episode content")
        return budget


class EmbeddingBackend(StrEnum):
    """Which embedding service is used."""

    #: OpenAI-compatible ``/embeddings`` endpoint (NVIDIA NIM, Azure OpenAI and others).
    OPENAI = "openai"
    #: The in-house embeddings gateway, which has its own payload and headers.
    IN_HOUSE = "in_house"


class EmbeddingSettings(BaseSettings):
    """Embedding service and the vector shape it must produce."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", populate_by_name=True)

    backend: EmbeddingBackend = Field(
        default=EmbeddingBackend.OPENAI, validation_alias=AliasChoices("EMBEDDING_BACKEND")
    )
    model: str = Field(
        default="nvidia/nemotron-3.5-embed-1b",
        validation_alias=AliasChoices("EMBEDDING_MODEL", "EMBEDDING_MODEL_NAME"),
    )
    #: Must match the vector index dimension; changing it requires a fresh index.
    #: 2048 is the width of nvidia/nemotron-3.5-embed-1b.
    dimension: int = Field(
        default=2048, gt=0, validation_alias=AliasChoices("EMBEDDING_DIMENSION")
    )
    batch_size: int = Field(default=64, gt=0, validation_alias=AliasChoices("EMBEDDING_BATCH_SIZE"))
    base_url: str = Field(
        default="https://integrate.api.nvidia.com/v1",
        validation_alias=AliasChoices("EMBEDDING_BASE_URL", "IN_HOUSE_EMBEDDINGS_URL"),
    )
    api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "EMBEDDING_API_KEY", "NVIDIA_API_KEY", "LLM_API_KEY", "OPENAI_API_KEY"
        ),
    )
    auth_scheme: AuthScheme = Field(default=AuthScheme.BEARER, validation_alias=AliasChoices("EMBEDDING_AUTH_SCHEME"))
    #: Only used by the in-house gateway.
    service_code: str | None = Field(default="graph", validation_alias=AliasChoices("EMBEDDING_SERVICE_CODE"))
    #: Required by the in-house gateway (``x-smtip-tid``); no default.
    tenant_id: str | None = Field(
        default=None, validation_alias=AliasChoices("EMBEDDING_TENANT_ID", "IN_HOUSE_EMBEDDINGS_TID")
    )
    priority: str = Field(default="async", validation_alias=AliasChoices("EMBEDDING_PRIORITY"))
    quantize: bool = Field(default=False, validation_alias=AliasChoices("EMBEDDING_QUANTIZE"))

    @property
    def embeddings_url(self) -> str:
        """Absolute endpoint for the selected backend."""
        base = self.base_url.rstrip("/")
        if self.backend is EmbeddingBackend.IN_HOUSE:
            return base
        return f"{base}/embeddings"

    def headers(self) -> dict[str, str]:
        """Credential and gateway headers for the selected backend."""
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.backend is EmbeddingBackend.IN_HOUSE:
            if self.service_code:
                headers["x-smtip-service-code"] = self.service_code
            if self.tenant_id:
                headers["x-smtip-tid"] = self.tenant_id
            headers["x-smtip-embedding-priority"] = self.priority
            return headers
        if self.api_key is not None:
            key = self.api_key.get_secret_value()
            headers.update(
                {"Authorization": f"Bearer {key}"}
                if self.auth_scheme is AuthScheme.BEARER
                else {"api-key": key}
            )
        return headers


class SimilaritySegmentationSettings(BaseModel):
    """Size and scoring knobs for the ``similarity`` segmentation strategy.

    Character count decides *when* a split is required; embedding similarity
    between neighbouring text decides *where*. Unused while
    :attr:`SegmentationSettings.strategy` is ``llm_boundary``.
    """

    min_size: int = Field(default=600, gt=0)
    target_size: int = Field(default=1_500, gt=0)
    max_size: int = Field(default=2_800, gt=0)
    #: Word-block width for intra-message analysis. Temporary; never stored.
    block_words: int = Field(default=6, ge=1)
    #: Oversized messages are analysed in word blocks only above this length.
    min_intra_message_chars: int = Field(default=80, ge=1)
    #: Short acknowledgements / follow-ups inherit neighbouring context.
    tiny_reply_chars: int = Field(default=40, ge=0)
    semantic_weight: float = Field(default=1.0, ge=0.0)
    #: Size only ranks in-range candidates; it must not choose the cut by itself.
    size_weight: float = Field(default=0.15, ge=0.0)
    dependency_weight: float = Field(default=0.85, ge=0.0)

    @model_validator(mode="after")
    def _ordered_bounds(self) -> Self:
        if not (self.min_size <= self.target_size <= self.max_size):
            raise ValueError("require min_size <= target_size <= max_size")
        return self


class SegmentationSettings(BaseModel):
    """Grouping conversation messages into bounded episodes."""

    #: Which strategy groups messages into episodes.
    strategy: SegmentationStrategy = Field(default=SegmentationStrategy.LLM_BOUNDARY)
    #: Knobs for the ``similarity`` strategy; unused by ``llm_boundary``.
    similarity: SimilaritySegmentationSettings = Field(
        default_factory=SimilaritySegmentationSettings
    )
    #: Hard ceiling on one episode, in estimated tokens. A discussion that grows
    #: past this is continued in a further bounded episode.
    max_episode_tokens: int = Field(default=1_200, gt=0)
    #: A single message larger than this is split into bounded source spans.
    max_span_tokens: int = Field(default=600, gt=0)
    #: Upper bound on one grouping request's source content.
    max_window_tokens: int = Field(default=2_500, gt=0)
    #: Tokens reserved for the grouping instructions.
    prompt_overhead_tokens: int = Field(default=900, ge=0)
    #: Tokens reserved for the model's boundary proposal.
    max_output_tokens: int = Field(default=1_200, gt=0)
    #: Finalized spans replayed as read-only context in the next window.
    context_spans: int = Field(default=4, ge=0)
    #: Attempts at semantic grouping before falling back to size-based grouping.
    max_attempts: int = Field(default=2, ge=1)
    #: Sessions segmented and ingested concurrently.
    max_concurrent_sessions: int = Field(default=4, gt=0)

    @model_validator(mode="after")
    def _validate_bounds(self) -> Self:
        """A span must fit inside an episode, or the episode limit is not a limit."""
        if self.max_span_tokens > self.max_episode_tokens:
            raise ValueError(
                "max_span_tokens must not exceed max_episode_tokens: an episode "
                "cannot be smaller than the smallest span it must hold"
            )
        return self

    def window_budget(self, model_input_budget: int) -> int:
        """Tokens available for source content in one grouping request."""
        available = model_input_budget - self.prompt_overhead_tokens - self.max_output_tokens
        return max(self.max_span_tokens, min(self.max_window_tokens, available))


class GateSettings(BaseModel):
    """Cheap relevance gate (A.7). The gate never deletes an Episode."""

    min_content_chars: int = Field(default=12, ge=0)
    #: Episodes at or below this length that are pure acknowledgements are skipped.
    max_acknowledgement_chars: int = Field(default=40, ge=0)


class ResolutionSettings(BaseModel):
    """Thresholds for embedding-first semantic resolution (spec section 3)."""

    candidate_top_k: int = Field(default=5, gt=0)
    #: Thresholds are raw cosine. Measured on nvidia/nemotron-3.5-embed-1b:
    #: genuine synonyms reach 0.76-0.90 and related-but-distinct concepts reach
    #: 0.73, so only near-identical text may auto-accept and the overlap band is
    #: adjudicated. Recalibrate when the embedding model changes.
    accept_similarity: float = Field(default=0.95, ge=0.0, le=1.0)
    #: Between ``ambiguous_similarity`` and ``accept_similarity`` the LLM adjudicates.
    ambiguous_similarity: float = Field(default=0.70, ge=0.0, le=1.0)


class RetrySettings(BaseModel):
    """Retry policy for transient failures (A.13)."""

    max_attempts: int = Field(default=3, ge=1)
    base_backoff_seconds: float = Field(default=1.0, ge=0.0)
    max_backoff_seconds: float = Field(default=10.0, ge=0.0)


class PromotionRule(BaseModel):
    """Promotion requirement for one scope (thresholds are type + scope aware)."""

    min_evidence: int = Field(default=1, ge=1)
    min_sessions: int = Field(default=1, ge=1)
    min_users: int = Field(default=1, ge=1)
    #: Shared scopes that must not be published from ordinary conversation.
    requires_authoritative_source: bool = False


class PromotionSettings(BaseModel):
    """Type and scope aware promotion policy (spec section 4, A.13)."""

    rules: dict[ScopeType, PromotionRule] = Field(
        default_factory=lambda: {
            ScopeType.SESSION: PromotionRule(),
            ScopeType.TASK: PromotionRule(),
            ScopeType.PROJECT: PromotionRule(min_evidence=2, min_sessions=2),
            ScopeType.USER: PromotionRule(min_evidence=2, min_sessions=2),
            ScopeType.TEAM: PromotionRule(min_evidence=3, min_sessions=3, min_users=3),
            ScopeType.DEPARTMENT: PromotionRule(min_evidence=5, min_sessions=5, min_users=4),
            ScopeType.ORGANIZATION: PromotionRule(
                min_evidence=5,
                min_sessions=5,
                min_users=5,
                requires_authoritative_source=True,
            ),
        }
    )
    #: Explicit durability cues ("always", "from now on") accelerate user promotion.
    explicit_cue_evidence: int = Field(default=1, ge=1)

    def rule_for(self, scope_type: ScopeType) -> PromotionRule:
        """Return the promotion rule for ``scope_type`` (never a global default)."""
        return self.rules.get(scope_type, PromotionRule())


class BatchingSettings(BaseModel):
    """Token-safe batching (A.1) and cross-batch continuity (A.6)."""

    #: Episodes repeated from the previous batch for local continuity.
    overlap_episodes: int = Field(default=1, ge=0)
    #: Maximum episodes per extraction call, regardless of token budget.
    max_episodes_per_batch: int = Field(default=40, gt=0)
    #: Maximum items retained in the temporary rolling session state.
    rolling_state_max_items: int = Field(default=20, gt=0)
    #: Episodes read from the graph per incremental-detection page.
    fetch_limit: int = Field(default=500, gt=0)


class MaintenanceSettings(BaseModel):
    """Asynchronous graph maintenance (promotion, aggregation)."""

    #: Candidates examined per promotion pass.
    candidate_batch_size: int = Field(default=200, gt=0)


class Settings(BaseSettings):
    """Root settings object. Build once per process via :func:`get_settings`."""

    model_config = SettingsConfigDict(
        env_prefix="CG_",
        env_nested_delimiter="__",
        env_file=".env",
        extra="ignore",
    )

    run_id: str | None = None
    log_level: str = "INFO"
    log_file: str = "logs/context_graph.log"

    #: Sessions JSON the ``context-graph`` command ingests. Overridable with
    #: ``--input`` or ``CG_INPUT``.
    input: Path = Field(default=DEFAULT_INPUT)
    #: Org structure JSON applied in the same run before ingesting
    #: conversations, or ``None`` to skip it. Overridable with
    #: ``--org-structure`` or ``CG_ORG_STRUCTURE``.
    org_structure: Path | None = Field(default=None)

    neo4j: Neo4jSettings = Field(default_factory=Neo4jSettings)
    chat: ChatBackendSettings = Field(default_factory=ChatBackendSettings)
    mongo: MongoSettings = Field(default_factory=MongoSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    segmentation: SegmentationSettings = Field(default_factory=SegmentationSettings)
    gate: GateSettings = Field(default_factory=GateSettings)
    resolution: ResolutionSettings = Field(default_factory=ResolutionSettings)
    retry: RetrySettings = Field(default_factory=RetrySettings)
    promotion: PromotionSettings = Field(default_factory=PromotionSettings)
    batching: BatchingSettings = Field(default_factory=BatchingSettings)
    maintenance: MaintenanceSettings = Field(default_factory=MaintenanceSettings)

    @field_validator("org_structure", mode="before")
    @classmethod
    def _blank_org_structure_means_skip_it(cls, value: object) -> object:
        """An empty string (e.g. from an unset ``.env`` entry) means "none"."""
        return None if value == "" else value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings instance."""
    return Settings()
