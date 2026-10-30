"""Scenario catalogue.

One declarative entry per Context Graph behaviour the dataset must exercise.
Scenario definitions live only here; the generator turns them into sessions
without knowing what any individual scenario means.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class Scenario(BaseModel):
    """A controlled conversation shape to generate."""

    model_config = ConfigDict(frozen=True)

    key: str
    title: str
    #: What the conversation must contain, handed to the model as instructions.
    guidance: str
    #: Pipeline behaviours this scenario is meant to exercise (recorded in the manifest).
    exercises: tuple[str, ...]
    turns: int = Field(default=8, gt=1)
    #: Independent sessions generated for this scenario.
    sessions: int = Field(default=1, gt=0)
    #: Give every session of this scenario a different user.
    distinct_users: bool = False
    #: Include acknowledgements and small talk the relevance gate should skip.
    include_acknowledgements: bool = False
    #: Minimum characters for one pasted turn, forcing temporary chunking.
    oversized_chars: int = Field(default=0, ge=0)


CATALOG: tuple[Scenario, ...] = (
    Scenario(
        key="preference_response_style",
        title="Durable response-style preference",
        guidance=(
            "The user states a lasting preference for concise, bullet-point answers "
            "using explicit durability language such as 'always' or 'from now on', "
            "and the assistant confirms it."
        ),
        exercises=("context_type:Preference", "scope:USER", "durability_cue"),
    ),
    Scenario(
        key="preference_response_style_paraphrase",
        title="Same preference, different wording",
        guidance=(
            "The user expresses the same underlying preference as 'always keep my "
            "answers concise', but with completely different wording: they complain "
            "about long-winded replies and ask for just the key points, permanently. "
            "Do not reuse the words 'concise' or 'brief'."
        ),
        exercises=("semantic_resolution", "concept_reuse", "context_type:Preference"),
        sessions=2,
        distinct_users=True,
    ),
    Scenario(
        key="goal_quarter",
        title="Personal goal for the quarter",
        guidance=(
            "The user describes a concrete goal they are working toward this quarter, "
            "with a target and a rough deadline, and the assistant asks one clarifying question."
        ),
        exercises=("context_type:Goal", "temporal_validity"),
    ),
    Scenario(
        key="task_in_progress",
        title="Task the user is working on",
        guidance=(
            "The user describes a specific task they are in the middle of, including "
            "its current status and what is blocking it."
        ),
        exercises=("context_type:Task", "context_type:State"),
    ),
    Scenario(
        key="decision_architecture",
        title="Architectural decision with rationale",
        guidance=(
            "The user and assistant discuss two options and the user settles on one, "
            "stating the decision and the reason for it."
        ),
        exercises=("context_type:Decision",),
    ),
    Scenario(
        key="constraint_conditional",
        title="Conditional constraint",
        guidance=(
            "The user states a hard constraint that only applies under a specific "
            "condition, for example 'when I am on call, never suggest changes that "
            "need a deploy'. The condition must be explicit."
        ),
        exercises=("context_type:Constraint", "condition"),
    ),
    Scenario(
        key="working_style",
        title="Working style and collaboration habits",
        guidance=(
            "The user explains how they prefer to work: review cadence, how they like "
            "feedback delivered, and when they do focused work."
        ),
        exercises=("context_type:WorkingStyle", "multi_signal_batch"),
        turns=10,
    ),
    Scenario(
        key="activity_and_event",
        title="Recurring activity and a dated event",
        guidance=(
            "The user mentions a recurring weekly activity and one specific upcoming "
            "event with a date, then the assistant acknowledges both."
        ),
        exercises=("context_type:Activity", "context_type:Event", "temporal_validity"),
    ),
    Scenario(
        key="interests_and_facts",
        title="Interests and stable personal facts",
        guidance=(
            "The user mentions two technical interests they follow and two stable facts "
            "about how they work, such as their timezone and their primary language."
        ),
        exercises=("context_type:Interest", "context_type:Fact", "multi_signal_batch"),
    ),
    Scenario(
        key="project_context",
        title="User-private project context",
        guidance=(
            "The user talks about their own work on a named project called Apollo: how "
            "they want its release notes written and which branch they work from. The "
            "preference is about their own work, not a team rule."
        ),
        exercises=("scope:PROJECT", "user_private_project"),
    ),
    Scenario(
        key="team_convention",
        title="Team convention reported independently",
        guidance=(
            "The user mentions, in passing, a convention that their Search team follows: "
            "the team deploys on Thursdays and requires two reviewers. They state it as an "
            "existing team norm, not as a personal preference or a new rule they are making."
        ),
        exercises=("scope:TEAM", "shared_candidate", "promotion_evidence"),
        sessions=4,
        distinct_users=True,
    ),
    Scenario(
        key="repeated_evidence",
        title="Same preference restated in a later session",
        guidance=(
            "The user restates a preference they clearly already have - they want design "
            "documents to always start with a summary - in a new conversation, in their own "
            "words, and the assistant confirms."
        ),
        exercises=("evidence_accumulation", "idempotent_evidence", "concept_reuse"),
        sessions=3,
    ),
    Scenario(
        key="conflicting_value",
        title="User changes their mind",
        guidance=(
            "The user says they have changed their mind about an earlier preference: they "
            "used to want short summaries, but from now on they want detailed explanations "
            "with examples. The contradiction must be explicit."
        ),
        exercises=("supersession", "conflicting_values"),
    ),
    Scenario(
        key="explicit_withdrawal",
        title="Explicit withdrawal of a preference",
        guidance=(
            "The user explicitly withdraws a preference they set earlier: they ask the "
            "assistant to forget and stop applying their rule about always adding code "
            "comments. Use clear withdrawal language such as 'forget', 'drop that' or "
            "'stop doing that'."
        ),
        exercises=("revocation", "retraction_outranks_recency"),
    ),
    Scenario(
        key="ambiguous_entity",
        title="Ambiguous person reference",
        guidance=(
            "The user refers to a colleague only by the first name Rahul, mentions a "
            "preference that belongs to that colleague rather than themselves, and never "
            "gives enough detail to identify which Rahul they mean."
        ),
        exercises=("unresolved_subject", "ambiguous_entity"),
    ),
    Scenario(
        key="irrelevant_chatter",
        title="Acknowledgements and small talk",
        guidance=(
            "A short exchange that carries no durable context at all: greetings, thanks, "
            "'ok', 'got it', and a single throwaway question about the weather."
        ),
        exercises=("relevance_gate", "no_context_expected"),
        turns=8,
        include_acknowledgements=True,
    ),
    Scenario(
        key="long_working_session",
        title="Long session with continuity",
        guidance=(
            "A long working session about migrating a service. Early turns introduce the "
            "project name, the deadline and two constraints; much later turns refer back to "
            "them only as 'the migration' and 'that constraint we agreed', without repeating "
            "the details. Include several preferences stated along the way."
        ),
        exercises=(
            "token_batching",
            "rolling_session_state",
            "long_range_reference",
            "multi_signal_batch",
        ),
        turns=40,
        include_acknowledgements=True,
    ),
    Scenario(
        key="oversized_episode",
        title="Pasted document in a single turn",
        guidance=(
            "The user pastes a very long incident postmortem document in one single message, "
            "with numbered sections and detailed paragraphs, then asks the assistant to note "
            "two durable preferences that the document implies about how they want incidents "
            "handled. The pasted document must be one message, not split across turns."
        ),
        exercises=("oversized_episode", "chunk_provenance", "chunk_signal_merge"),
        turns=6,
        oversized_chars=14_000,
    ),
)

CATALOG_BY_KEY: dict[str, Scenario] = {scenario.key: scenario for scenario in CATALOG}


def select(keys: tuple[str, ...] = ()) -> tuple[Scenario, ...]:
    """Return the requested scenarios, or the whole catalogue when none are named."""
    if not keys:
        return CATALOG
    unknown = sorted(set(keys) - CATALOG_BY_KEY.keys())
    if unknown:
        raise KeyError(f"unknown scenario(s): {', '.join(unknown)}")
    return tuple(CATALOG_BY_KEY[key] for key in keys)
