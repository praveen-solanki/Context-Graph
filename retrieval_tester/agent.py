"""The retrieval agent: retrieved Context in, conversational reply out.

The agent owns prompt assembly and the model call only. It never writes to the
Context Graph and never touches the ingestion pipeline; the Context it receives
is already selected by the retriever.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import json

from pydantic import BaseModel, ConfigDict

from context_graph.config import ChatBackendSettings
from context_graph.domain.conversation import MessageInput
from context_graph.domain.enums import Role
from context_graph.errors import ContextGraphError
from context_graph.providers.chat_backend import ChatTransport
from retrieval_tester.config import AgentSettings
from retrieval_tester.errors import AgentError
from retrieval_tester.graph_tools import TOOL_SCHEMA, GraphLookupCall, GraphLookupTool
from retrieval_tester.models import RetrievalResult

_SYSTEM_PROMPT = """You are a helpful workplace assistant talking to one colleague.

You are given personalization context retrieved for this user: durable
preferences, constraints, goals and facts they established in earlier
conversations. Apply it silently and naturally.

Rules:
- Follow the retrieved preferences without being asked and without announcing
  that you are following them.
- Respect any condition attached to a context item; it applies only then.
- Never ask the user to restate or confirm something the context already
  settles: act on it. Asking again defeats the point of remembering it.
- If the user's current message contradicts stored context, the current message
  wins for this conversation.
- If the user asks what you know about them, answer honestly from the context.
- You are given a list of earlier conversations with this user. Use it to answer
  questions about what was discussed before, and refer to those conversations
  naturally by what they were about and when they happened.
- You are given preferences the user has changed. Follow the current value, and
  mention the old one only if asked what changed.
- The organizational details below (team, department, teammates, projects) are
  shared information this user is entitled to see. Answer questions about their
  team and colleagues from it.
- When the user names a person, look that name up exactly as they said it before
  answering. Then, in this order:
  1. If what is true about this user below names the person they mean, or gives
     a rule for choosing between people who share a name, follow it, say nothing
     about the ambiguity and do not ask again - the point of having stored it is
     that they never have to repeat it.
  2. Otherwise, if exactly one person matches, answer from their details.
  3. Otherwise, say that more than one person shares the name, list each with
     their team and role, and ask which one they mean. Never choose for them:
     sharing the user's team is not an answer unless the user has said it is.
- Never reveal another person's preferences or private context: you are only
  given their name, role and projects.
- Never claim to remember something that is not in the context below.
- If no context was retrieved, behave like a competent assistant meeting them
  for the first time.
"""

_NO_CONTEXT = "(no stored context was retrieved for this user)"


class AgentReply(BaseModel):
    """What the agent answered, and any graph lookups it made to get there."""

    model_config = ConfigDict(frozen=True)

    text: str
    lookups: tuple[GraphLookupCall, ...] = ()


class RetrievalAgent:
    """Generates replies grounded in the Context retrieved for a user."""

    def __init__(
        self,
        *,
        transport: ChatTransport,
        backend: ChatBackendSettings,
        settings: AgentSettings,
        graph_tool: GraphLookupTool | None = None,
    ) -> None:
        self._transport = transport
        self._backend = backend
        self._settings = settings
        self._graph_tool = graph_tool

    @property
    def model(self) -> str:
        return self._backend.chat_model

    def system_prompt(self, retrieval: RetrievalResult) -> str:
        """The exact system prompt the model receives, for display in the UI."""
        context = retrieval.as_prompt_block() or _NO_CONTEXT
        return f"{_SYSTEM_PROMPT}\n{context}\n"

    def replace_settings(self, settings: AgentSettings) -> None:
        """Swap in changed settings (used by the UI toggle)."""
        self._settings = settings

    @property
    def graph_tool_enabled(self) -> bool:
        """Whether the agent may query the graph for structural questions."""
        return self._graph_tool is not None and self._settings.enable_graph_tool

    async def reply(
        self, *, retrieval: RetrievalResult, history: Sequence[MessageInput], message: str
    ) -> AgentReply:
        """Answer ``message``, optionally looking structure up in the graph."""
        recent = list(history)[-self._settings.max_history_turns :]
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt(retrieval)},
            *(
                {
                    "role": "user" if turn.role is Role.USER else "assistant",
                    "content": turn.text,
                }
                for turn in recent
            ),
            {"role": "user", "content": message},
        ]

        lookups: list[GraphLookupCall] = []
        for _ in range(self._settings.max_tool_iterations + 1):
            choice = await self._call(messages)
            tool_calls = choice["message"].get("tool_calls") or []
            if not tool_calls:
                content = choice["message"].get("content")
                if not content:
                    raise AgentError(
                        f"the model returned no reply (finish_reason={choice.get('finish_reason')})"
                    )
                return AgentReply(text=str(content).strip(), lookups=tuple(lookups))

            messages.append(choice["message"])
            for call in tool_calls:
                lookups.append(await self._run_tool(call))
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "content": lookups[-1].result,
                    }
                )
        raise AgentError("the model kept asking for lookups without answering")

    async def _call(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        """One chat completion, with the graph tool offered when enabled."""
        payload: dict[str, Any] = {
            "model": self._backend.chat_model,
            "messages": messages,
            self._backend.token_parameter: self._settings.max_output_tokens,
            "temperature": self._settings.temperature,
        }
        if self.graph_tool_enabled:
            payload["tools"] = [TOOL_SCHEMA]
        try:
            body = await self._transport.post("chat/completions", payload, "retrieval-tester")
        except ContextGraphError as error:
            raise AgentError(f"the model call failed: {error}") from error
        return dict(body["choices"][0])

    async def _run_tool(self, call: dict[str, Any]) -> GraphLookupCall:
        """Execute one requested lookup; a bad request is reported, not raised."""
        assert self._graph_tool is not None
        try:
            arguments = json.loads(call["function"]["arguments"] or "{}")
        except json.JSONDecodeError:
            arguments = {}
        return await self._graph_tool.run(
            str(arguments.get("lookup", "")), str(arguments.get("target", ""))
        )
