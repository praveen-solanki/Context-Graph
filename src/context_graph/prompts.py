"""Centralized prompts and their version.

Prompt text lives only here. The controlled vocabulary is rendered from the
domain enums so the prompt can never drift from the ontology.
"""

from __future__ import annotations

from collections.abc import Iterable

from context_graph.domain.enums import ContextType, ScopeType, SignalOperation
from context_graph.domain.extraction import EpisodeChunk

#: Bump when the prompt text changes; stored in every Context's lineage.
PROMPT_VERSION = "extraction-v1"
ADJUDICATION_PROMPT_VERSION = "adjudication-v1"


def _values(values: Iterable[str]) -> str:
    return ", ".join(values)


EXTRACTION_SYSTEM = f"""You extract durable personalization context from conversation turns.

You interpret language only. You never decide identity, permissions or storage.

Return one signal per distinct, durable piece of context. A signal must be
supported by the quoted text of the episodes it cites.

Rules:
- Cite the episode_id(s) that support each signal in source_episode_ids.
- Do NOT extract small talk, acknowledgements, or restatements of your own replies.
- Do NOT invent a subject, scope, value or time that the text does not support.
- concept is a short reusable snake_case anchor (for example: response_detail,
  meeting_schedule, deployment_strategy). Prefer an existing common concept name
  over inventing a synonym, and keep it coarse: 'response_detail', never
  'response_style_always_from_now_on_language'.
- value is the concrete state for that concept (for example: concise, weekly).
- subject says WHO the context is about, not what it is about. It is "user" for
  anything the speaker says about themselves or reports about their own team;
  use a person's name only when the context belongs to someone else. A user
  describing their own team's convention still has subject "user" - the team
  goes in scope_cue, never in subject.
- context_type must be one of: {_values(ContextType)}. Use "Other" when no type fits.
- scope_cue records the scope the statement appears to concern
  ({_values(ScopeType)}) plus the name mentioned, if any. A cue is an observation:
  it does not grant permission to write at that scope.
- Cue TASK, PROJECT, TEAM, DEPARTMENT or ORGANIZATION when the statement
  describes how that named group or thing works, and put its name in scope_ref:
  a convention the user's Search team follows is TEAM with scope_ref "Search".
  Never cue a scope whose name the text does not give - fall back to USER when
  the statement is durable for the speaker, or SESSION when it applies only to
  this conversation. Do not widen a named team to DEPARTMENT or ORGANIZATION.
- Use SESSION only when the user limits it to this conversation ("for this chat",
  "just now"). "Always", "from now on" and "never" mean USER, not SESSION.
- operation is "{SignalOperation.REVOKE}" only when the user explicitly withdraws or
  cancels earlier context; otherwise "{SignalOperation.ASSERT}".
- durability_cue is true only for explicit durability language such as "always",
  "from now on", "remember", "never".
- condition carries a trigger ("when I am on call") instead of forcing a
  conditional instruction into an unconditional preference.
- confidence is your confidence in the interpretation of the text, not the
  probability that the preference is durable.

- unresolved_references lists short phrases that point at information introduced
  earlier in the session and missing here ("the project we discussed"). List the
  phrase only; never guess what it refers to.

Episodes labelled CONTEXT-ONLY are provided for continuity. Never cite them.
"""

ADJUDICATION_SYSTEM = """You decide whether two personalization concepts mean the same thing.

Answer only from the given names and descriptions. Choose the candidate that is
semantically identical, not merely related. If none is identical, return
is_match=false. Never guess.
"""


SEGMENTATION_PROMPT_VERSION = "segmentation-v2"

SEGMENTATION_SYSTEM = """You group consecutive conversation content into episodes.

An episode is a bounded stretch of conversation about one thing: a question and
the work of answering it, a decision and its discussion, one topic and the
replies that belong to it.

Rules:
- Group by topic, not by turn. Do not assume any user/agent alternation.
- A short reply ("ok", "thanks", "yes do that") belongs with the discussion it
  answers, never alone.
- Start a new episode when the subject genuinely changes. If the conversation
  keeps alternating between two subjects, keep the alternating stretch together
  rather than splitting every turn.
- Cover every SOURCE reference exactly once, in the order given. Never invent,
  repeat, reorder or skip a reference.
- Some references are parts of one long message. You may put a boundary between
  those parts when they cover different subjects.
- Mark the final episode open_ended=true only when the discussion is clearly
  unfinished and continues past the last reference given.
- References labelled CONTEXT are earlier content shown only so you can judge
  continuity. Never include a CONTEXT reference in any episode.
"""


def render_segmentation(
    spans: Iterable[tuple[str, str, str]],
    context: Iterable[tuple[str, str, str]] = (),
    rejection: str | None = None,
) -> str:
    """Render one grouping request: ordered references, roles and text.

    ``rejection`` is why the previous proposal for this window was refused; at
    temperature 0 the identical request would return the identical proposal.
    """
    sections: list[str] = []
    if rejection:
        sections.append(
            f"Your previous grouping of this window was rejected: {rejection}. "
            "Return a grouping that covers every SOURCE reference exactly once, in order."
        )
    context_lines = [f"[CONTEXT {ref} {role}] {text}" for ref, role, text in context]
    if context_lines:
        sections.append(
            "CONTEXT (already assigned to earlier episodes - never group these):\n"
            + "\n".join(context_lines)
        )
    sections.append(
        "SOURCE (group all of these):\n"
        + "\n".join(f"[{ref} {role}] {text}" for ref, role, text in spans)
    )
    return "\n\n".join(sections)


def render_batch(
    chunks: Iterable[EpisodeChunk],
    *,
    overlap: Iterable[EpisodeChunk] = (),
    rolling_state: str = "",
) -> str:
    """Render the user message for one extraction call."""
    sections: list[str] = []
    if rolling_state:
        sections.append(f"SESSION STATE (continuity only, do not cite):\n{rolling_state}")
    context_only = [_render_chunk(chunk, context_only=True) for chunk in overlap]
    if context_only:
        sections.append("CONTEXT-ONLY EPISODES:\n" + "\n".join(context_only))
    sections.append(
        "EPISODES TO EXTRACT FROM:\n"
        + "\n".join(_render_chunk(chunk, context_only=False) for chunk in chunks)
    )
    return "\n\n".join(sections)


def _render_chunk(chunk: EpisodeChunk, *, context_only: bool) -> str:
    episode = chunk.episode
    marker = "CONTEXT-ONLY " if context_only else ""
    part = f" part {chunk.chunk_index + 1}/{chunk.chunk_count}" if chunk.is_partial else ""
    return (
        f"[{marker}episode_id={episode.episode_id} role={episode.role.value} "
        f"sender={episode.sender_id} seq={episode.sequence} "
        f"time={episode.timestamp.isoformat()}{part}]\n{chunk.text}"
    )
