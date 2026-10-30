"""Plan, generate and validate test ingestion requests.

The model supplies conversation text only. Identity, ordering and timestamps are
assigned deterministically here, so a run with the same seed produces the same
session_ids, message_ids and sequences. Every session is validated against the
POC's own input schema before it can reach the dataset.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from datetime import datetime, timedelta
from hashlib import blake2b
from typing import Literal

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
)

from context_graph.domain.conversation import MessageInput, SessionInput
from context_graph.domain.enums import Role
from context_graph.errors import ContextGraphError
from context_graph.observability import get_logger, log_event
from context_graph.tool import IngestionRequest
from context_graph.providers.chat_backend import ChatLLMProvider
from test_conversation_generator.config import GenerationSettings
from test_conversation_generator.scenarios import Scenario

_logger = get_logger("test_conversation_generator.generator")

_SYSTEM_PROMPT = """You write realistic multi-turn conversations between a user and an AI assistant.

Rules:
- Write natural workplace dialogue, not a list of facts or scripted Q&A.
- Alternate roles, starting with the user. The assistant replies like a real assistant.
- Only the user states their own preferences, goals, constraints and decisions.
- Never invent a scene label, speaker name or markdown heading: return plain speech.
- Return exactly the requested number of turns.

The speakers have never seen your instructions. Never quote, restate or allude to
them, and never use their vocabulary ("durability language", "durable context",
"scenario", "turns"). Convey what is asked through what people would actually say.
"""

#: Phrases that only appear when the model echoes its own instructions.
_INSTRUCTION_LEAKS: tuple[str, ...] = (
    "durability language",
    "durable context",
    "this scenario",
    "the scenario",
    "these instructions",
    "as instructed",
    "requested number of turns",
    "pasted document must",
)


class GenerationQualityError(ContextGraphError):
    """Generated text is structurally valid but not usable as realistic data."""


class ConversationTurn(BaseModel):
    """One generated turn."""

    model_config = ConfigDict(frozen=True)

    role: Literal["user", "agent"]
    text: str

    @field_validator("text")
    @classmethod
    def _reject_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("turn text must not be blank")
        return value


class GeneratedConversation(BaseModel):
    """The model's full response for one session."""

    model_config = ConfigDict(frozen=True)

    turns: tuple[ConversationTurn, ...] = Field(min_length=2)


class ConversationContinuation(BaseModel):
    """Additional body text used to grow an oversized episode."""

    model_config = ConfigDict(frozen=True)

    text: str


class SessionPlan(BaseModel):
    """Deterministic identity and shape of one session, fixed before generation."""

    model_config = ConfigDict(frozen=True)

    session_id: str
    user_id: str
    agent_id: str
    scenario: Scenario
    turns: int = Field(gt=1)
    start_time: AwareDatetime
    seed: int


class SessionManifest(BaseModel):
    """What a generated session is meant to exercise, for downstream testing."""

    model_config = ConfigDict(frozen=True)

    session_id: str
    scenario: str
    title: str
    user_id: str
    exercises: tuple[str, ...]
    message_count: int
    max_message_chars: int
    total_chars: int


class ConversationDataset(BaseModel):
    """The generated dataset plus its manifest."""

    model_config = ConfigDict(frozen=True)

    requests: tuple[IngestionRequest, ...]
    manifest: tuple[SessionManifest, ...]
    failures: tuple[str, ...] = ()


class ConversationDatasetGenerator:
    """Turns the scenario catalogue into validated ingestion requests."""

    def __init__(
        self,
        *,
        client: ChatLLMProvider,
        settings: GenerationSettings,
        scenarios: Sequence[Scenario],
    ) -> None:
        self._client = client
        self._settings = settings
        self._scenarios = tuple(scenarios)

    def plan(self) -> list[SessionPlan]:
        """Lay out every session deterministically before any API call."""
        plans: list[SessionPlan] = []
        for scenario in self._scenarios:
            sessions = self._settings.sessions_per_scenario or scenario.sessions
            turns = self._settings.turns_per_session or scenario.turns
            for index in range(sessions):
                session_id = f"S-{scenario.key}-{index + 1:02d}"
                plans.append(
                    SessionPlan(
                        session_id=session_id,
                        user_id=self._user_id(scenario, index),
                        agent_id=f"agent-{(index % 2) + 1}",
                        scenario=scenario,
                        turns=turns,
                        start_time=self._settings.start_time
                        + timedelta(hours=len(plans) * 3),
                        seed=self._derive_seed(session_id),
                    )
                )
        return plans

    async def generate(self) -> ConversationDataset:
        """Generate every planned session concurrently, keeping only valid ones."""
        plans = self.plan()
        log_event(_logger, "generation.started", sessions=len(plans), seed=self._settings.seed)
        results = await asyncio.gather(
            *(self._generate_session(plan) for plan in plans), return_exceptions=True
        )

        requests: list[IngestionRequest] = []
        manifest: list[SessionManifest] = []
        failures: list[str] = []
        for plan, result in zip(plans, results, strict=True):
            if isinstance(result, BaseException):
                failures.append(f"{plan.session_id}: {result}")
                log_event(
                    _logger,
                    "generation.session_failed",
                    session_id=plan.session_id,
                    error=str(result),
                )
                continue
            requests.append(result)
            manifest.append(_manifest_for(plan, result.session))

        log_event(
            _logger,
            "generation.completed",
            generated=len(requests),
            failed=len(failures),
        )
        return ConversationDataset(
            requests=tuple(requests), manifest=tuple(manifest), failures=tuple(failures)
        )

    async def _generate_session(self, plan: SessionPlan) -> IngestionRequest:
        """Generate one session, retrying unusable model output a bounded number of times."""
        last_error: Exception | None = None
        for attempt in range(1, self._settings.attempts_per_session + 1):
            try:
                conversation = await self._client.complete_json(
                    system=_SYSTEM_PROMPT,
                    user=self._prompt(plan),
                    schema=GeneratedConversation,
                    max_output_tokens=self._settings.max_output_tokens,
                    # A different seed per attempt, still reproducible run to run.
                    seed=plan.seed + attempt,
                    label=plan.session_id,
                )
                _reject_instruction_leaks(conversation.turns)
                turns = await self._expand_if_needed(plan, self._trim(plan, conversation.turns))
                return self._build_request(plan, turns)
            except (ContextGraphError, ValidationError) as error:
                last_error = error
                log_event(
                    _logger,
                    "generation.attempt_failed",
                    session_id=plan.session_id,
                    attempt=attempt,
                    error=str(error),
                )
        assert last_error is not None
        raise last_error

    def _trim(
        self, plan: SessionPlan, turns: Sequence[ConversationTurn]
    ) -> tuple[ConversationTurn, ...]:
        """Hold the model to the planned turn count, so counts stay configurable."""
        if len(turns) <= plan.turns:
            return tuple(turns)
        log_event(
            _logger,
            "generation.turns_trimmed",
            session_id=plan.session_id,
            generated=len(turns),
            planned=plan.turns,
        )
        return tuple(turns[: plan.turns])

    async def _expand_if_needed(
        self, plan: SessionPlan, turns: Sequence[ConversationTurn]
    ) -> tuple[ConversationTurn, ...]:
        """Grow the longest user turn until it is genuinely oversized."""
        target = plan.scenario.oversized_chars
        if not target:
            return tuple(turns)

        index = max(
            (position for position, turn in enumerate(turns) if turn.role == "user"),
            key=lambda position: len(turns[position].text),
            default=0,
        )
        text = turns[index].text
        for call in range(self._settings.max_expansion_calls):
            if len(text) >= target:
                break
            continuation = await self._client.complete_json(
                system=_SYSTEM_PROMPT,
                max_output_tokens=self._settings.max_output_tokens,
                user=(
                    "Continue the following pasted incident postmortem with further numbered "
                    f"sections, at least {target - len(text)} characters, same document, no "
                    "preamble and no closing remarks. Return the continuation only.\n\n"
                    f"{text[-2000:]}"
                ),
                schema=ConversationContinuation,
                seed=plan.seed + 100 + call,
                label=f"{plan.session_id}:expand{call + 1}",
            )
            text = f"{text}\n\n{continuation.text.strip()}"

        if len(text) < target:
            log_event(
                _logger,
                "generation.oversized_target_missed",
                session_id=plan.session_id,
                chars=len(text),
                target=target,
            )
        updated = list(turns)
        updated[index] = turns[index].model_copy(update={"text": text})
        return tuple(updated)

    def _build_request(
        self, plan: SessionPlan, turns: Sequence[ConversationTurn]
    ) -> IngestionRequest:
        """Assign deterministic identity and validate against the POC schema."""
        messages = tuple(
            MessageInput(
                message_id=f"{plan.session_id}-m{sequence:03d}",
                timestamp=self._timestamp(plan, sequence),
                role=Role.USER if turn.role == "user" else Role.AGENT,
                sequence=sequence,
                text=turn.text,
                sender_id=plan.user_id if turn.role == "user" else None,
                agent_id=None if turn.role == "user" else plan.agent_id,
            )
            for sequence, turn in enumerate(turns, start=1)
        )
        session = SessionInput(
            session_id=plan.session_id,
            user_id=plan.user_id,
            agent_ids=(plan.agent_id,),
            messages=messages,
        )
        return IngestionRequest(session=session)

    def _prompt(self, plan: SessionPlan) -> str:
        """Instructions for one session, derived from its scenario."""
        scenario = plan.scenario
        lines = [
            f"Conversation: {scenario.title}.",
            f"Write exactly {plan.turns} turns, starting with the user.",
            scenario.guidance,
        ]
        if scenario.include_acknowledgements:
            lines.append(
                "Include at least two throwaway turns that carry no durable context, "
                "such as 'thanks', 'ok' or 'got it'."
            )
        if scenario.oversized_chars:
            lines.append(
                f"One single user turn must be at least {scenario.oversized_chars} characters "
                "long: a full pasted document with numbered sections and real detail."
            )
        if scenario.distinct_users:
            lines.append(
                "This speaker is a different person from other speakers describing the same "
                "thing: use their own phrasing, role and examples."
            )
        return "\n".join(lines)

    def _user_id(self, scenario: Scenario, index: int) -> str:
        """Stable user assignment; scenarios needing independent voices get distinct users."""
        if scenario.distinct_users:
            offset = self._derive_seed(scenario.key) + index
        else:
            offset = self._derive_seed(scenario.key)
        return f"U{offset % self._settings.user_count + 1:03d}"

    def _timestamp(self, plan: SessionPlan, sequence: int) -> datetime:
        return plan.start_time + timedelta(
            seconds=self._settings.seconds_between_messages * (sequence - 1)
        )

    def _derive_seed(self, token: str) -> int:
        """Deterministic per-token seed derived from the run seed."""
        digest = blake2b(
            f"{self._settings.seed}:{token}".encode(), digest_size=4
        ).digest()
        return int.from_bytes(digest, "big")


def _reject_instruction_leaks(turns: Sequence[ConversationTurn]) -> None:
    """Fail the attempt when the model repeats its own brief back as dialogue."""
    for turn in turns:
        lowered = turn.text.casefold()
        leak = next((phrase for phrase in _INSTRUCTION_LEAKS if phrase in lowered), None)
        if leak is not None:
            raise GenerationQualityError(f"generated text echoed the instructions: '{leak}'")


def _manifest_for(plan: SessionPlan, session: SessionInput) -> SessionManifest:
    lengths = [len(message.text) for message in session.messages]
    return SessionManifest(
        session_id=session.session_id,
        scenario=plan.scenario.key,
        title=plan.scenario.title,
        user_id=session.user_id,
        exercises=plan.scenario.exercises,
        message_count=len(session.messages),
        max_message_chars=max(lengths, default=0),
        total_chars=sum(lengths),
    )
