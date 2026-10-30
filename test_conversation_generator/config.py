"""Configuration for the test conversation generator.

The chat backend is the pipeline's own :class:`ChatBackendSettings`, so both
read the same environment names. Generation knobs use the ``SYNTH_`` prefix and
never collide with the pipeline's ``CG_`` settings.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from pydantic import AwareDatetime, BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from context_graph.config import ChatBackendSettings, LLMSettings, RetrySettings


class GenerationSettings(BaseSettings):
    """How much data to generate, and how hard to push the API."""

    model_config = SettingsConfigDict(env_prefix="SYNTH_", env_file=".env", extra="ignore")

    seed: int = 20260912
    output_path: Path = Path("data/synthetic_sessions.json")
    #: Distinct users the generated sessions are spread across.
    user_count: int = Field(default=8, gt=0)
    #: Overrides the per-scenario session count when set.
    sessions_per_scenario: int | None = Field(default=None, gt=0)
    #: Overrides the per-scenario turn count when set.
    turns_per_session: int | None = Field(default=None, gt=0)
    #: Only generate these scenario keys; empty means every scenario.
    scenarios: tuple[str, ...] = ()
    #: Concurrent model requests; keep modest to respect rate limits.
    concurrency: int = Field(default=4, gt=0)
    #: Attempts per session when the model returns unusable content.
    attempts_per_session: int = Field(default=3, gt=0)
    #: Extra calls allowed to grow an oversized episode to its target length.
    max_expansion_calls: int = Field(default=2, ge=0)
    #: Output budget per generation call; long sessions need room.
    max_output_tokens: int = Field(default=16_000, gt=0)
    #: Sampling temperature; generated data should vary, unlike extraction.
    temperature: float = Field(default=0.9, ge=0.0, le=2.0)
    #: Wall-clock anchor for generated timestamps; fixed for reproducibility.
    start_time: AwareDatetime = datetime(2026, 9, 1, 9, 0, tzinfo=UTC)
    #: Seconds between consecutive messages in a session.
    seconds_between_messages: int = Field(default=45, gt=0)
    retry: RetrySettings = Field(default_factory=lambda: RetrySettings(max_attempts=4))


class ConversationGeneratorConfig(BaseModel):
    """Everything the generator needs, assembled once at startup."""

    chat: ChatBackendSettings
    generation: GenerationSettings

    @classmethod
    def load(cls) -> ConversationGeneratorConfig:
        """Build configuration from the environment and ``.env``."""
        return cls(
            chat=ChatBackendSettings(),
            generation=GenerationSettings(),
        )

    def llm_settings(self) -> LLMSettings:
        """Generation-tuned model settings for the shared chat provider."""
        return LLMSettings(
            temperature=self.generation.temperature,
            max_output_tokens=self.generation.max_output_tokens,
        )
