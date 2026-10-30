"""Test conversation generator: determinism, validation and quality guards."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from context_graph.config import RetrySettings
from context_graph.domain.enums import Role
from context_graph.providers.chat_backend import ChatLLMProvider, strict_schema
from test_conversation_generator.config import GenerationSettings
from test_conversation_generator.generator import (
    ConversationContinuation,
    ConversationTurn,
    GeneratedConversation,
    ConversationDatasetGenerator,
)
from test_conversation_generator.scenarios import CATALOG, Scenario, select


class StubClient(ChatLLMProvider):
    """Provider double: returns scripted conversations without a network call."""

    def __init__(self, turns: int = 4, continuation: str = "more text") -> None:  # noqa: D107
        self._turns = turns
        self._continuation = continuation
        self.calls: list[str] = []

    async def complete_json[M: BaseModel](
        self,
        *,
        system: str,
        user: str,
        schema: type[M],
        max_output_tokens: int,
        label: str,
        seed: int | None = None,
    ) -> M:
        self.calls.append(label)
        if schema is ConversationContinuation:
            return ConversationContinuation(text=self._continuation)  # pyright: ignore[reportReturnType]
        conversation = GeneratedConversation(
            turns=tuple(
                ConversationTurn(
                    role="user" if index % 2 == 0 else "agent",
                    text=f"turn {index} of {label}",
                )
                for index in range(self._turns)
            )
        )
        return conversation  # pyright: ignore[reportReturnType]


def generator(scenarios: tuple[Scenario, ...], **overrides: Any) -> ConversationDatasetGenerator:
    return ConversationDatasetGenerator(
        client=StubClient(**overrides.pop("client", {})),
        settings=GenerationSettings(**overrides),
        scenarios=scenarios,
    )


def test_plan_is_deterministic_for_a_seed_and_changes_with_it() -> None:
    first = generator(CATALOG).plan()
    again = generator(CATALOG).plan()
    other_seed = generator(CATALOG, seed=7).plan()

    assert [plan.session_id for plan in first] == [plan.session_id for plan in again]
    assert [plan.user_id for plan in first] == [plan.user_id for plan in again]
    assert [plan.user_id for plan in first] != [plan.user_id for plan in other_seed]


def test_session_and_message_identity_is_unique_and_sequential() -> None:
    plans = generator(CATALOG).plan()
    assert len({plan.session_id for plan in plans}) == len(plans)

    scenario = select(("preference_response_style",))
    request = generator(scenario)._build_request(  # pyright: ignore[reportPrivateUsage]
        generator(scenario).plan()[0],
        [ConversationTurn(role="user", text="a"), ConversationTurn(role="agent", text="b")],
    )
    messages = request.session.messages
    assert [message.sequence for message in messages] == [1, 2]
    assert len({message.message_id for message in messages}) == 2
    assert messages[0].timestamp < messages[1].timestamp
    assert messages[0].role is Role.USER and messages[1].role is Role.AGENT
    assert messages[1].agent_id == request.session.agent_ids[0]


async def test_generated_sessions_validate_against_the_poc_schema() -> None:
    dataset = await generator(select(("preference_response_style", "team_convention"))).generate()

    assert dataset.failures == ()
    assert len(dataset.requests) == 5  # 1 + 4 independent team reports
    assert {entry.session_id for entry in dataset.manifest} == {
        request.session.session_id for request in dataset.requests
    }


async def test_scenarios_needing_independent_voices_use_distinct_users() -> None:
    dataset = await generator(select(("team_convention",))).generate()
    users = {request.session.user_id for request in dataset.requests}
    assert len(users) == len(dataset.requests)


async def test_repeated_scenario_keeps_one_user_across_sessions() -> None:
    dataset = await generator(select(("repeated_evidence",))).generate()
    assert len({request.session.user_id for request in dataset.requests}) == 1


async def test_turn_count_is_capped_by_configuration() -> None:
    dataset = await generator(
        select(("preference_response_style",)), turns_per_session=2, client={"turns": 9}
    ).generate()
    assert len(dataset.requests[0].session.messages) == 2


async def test_oversized_scenario_is_expanded_until_it_needs_chunking() -> None:
    dataset = await generator(
        select(("oversized_episode",)), client={"turns": 4, "continuation": "x" * 9_000}
    ).generate()
    longest = max(len(message.text) for message in dataset.requests[0].session.messages)
    assert longest >= CATALOG[-1].oversized_chars


async def test_instruction_leakage_fails_the_session_instead_of_being_written() -> None:
    class LeakyClient(StubClient):
        async def complete_json[M: BaseModel](
            self,
            *,
            system: str,
            user: str,
            schema: type[M],
            max_output_tokens: int,
            label: str,
            seed: int | None = None,
        ) -> M:
            conversation = GeneratedConversation(
                turns=(
                    ConversationTurn(role="user", text="Use explicit durability language here."),
                    ConversationTurn(role="agent", text="ok"),
                )
            )
            return conversation  # pyright: ignore[reportReturnType]

    dataset = await ConversationDatasetGenerator(
        client=LeakyClient(),
        settings=GenerationSettings(attempts_per_session=2),
        scenarios=select(("preference_response_style",)),
    ).generate()

    assert dataset.requests == ()
    assert len(dataset.failures) == 1
    assert "echoed the instructions" in dataset.failures[0]


def test_strict_schema_forbids_extra_fields_everywhere() -> None:
    schema = strict_schema(GeneratedConversation)
    assert schema["additionalProperties"] is False
    assert schema["$defs"]["ConversationTurn"]["additionalProperties"] is False
    assert schema["$defs"]["ConversationTurn"]["required"] == ["role", "text"]


def test_unknown_scenario_key_is_rejected() -> None:
    with pytest.raises(KeyError):
        select(("does_not_exist",))


def test_every_scenario_declares_what_it_exercises() -> None:
    assert all(scenario.exercises for scenario in CATALOG)
    assert len({scenario.key for scenario in CATALOG}) == len(CATALOG)


def test_retry_settings_are_shared_with_the_pipeline() -> None:
    assert isinstance(GenerationSettings().retry, RetrySettings)
