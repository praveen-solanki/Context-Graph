"""Chat and embedding providers for any OpenAI-compatible backend.

The only module that talks to a model service. The default is NVIDIA NIM;
any other OpenAI-compatible deployment is the same client with different settings.
everything that differs between them - credential header, token-cap parameter,
structured-output support, gateway headers, embedding payload shape - comes from
configuration rather than from a second implementation.

Structured results are always returned as validated Pydantic models. When the
backend cannot enforce a JSON schema, the schema is put in the prompt and the
reply is validated here, with a bounded repair attempt that shows the model its
own error. A reply that never validates is a failure, never a partial result.
"""

from __future__ import annotations

import asyncio
import json
import math
from collections.abc import Sequence
from types import TracebackType
from typing import Any, Self

import httpx
from pydantic import BaseModel, ValidationError

from context_graph.config import (
    ChatBackendSettings,
    EmbeddingBackend,
    EmbeddingSettings,
    LLMSettings,
    RetrySettings,
    StructuredOutputMode,
)
from context_graph.errors import ExtractionValidationError, ProviderError, TransientError
from context_graph.observability import get_logger, log_event
from context_graph.retry import retry_async

_logger = get_logger("providers.azure")

_RETRYABLE_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504})


def strict_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Render a Pydantic model as a JSON schema the backend will enforce strictly."""
    return _tighten(model.model_json_schema())


def _tighten(node: Any) -> Any:
    """Require every property and forbid extras, as strict mode demands."""
    if isinstance(node, list):
        return [_tighten(item) for item in node]
    if not isinstance(node, dict):
        return node
    schema: dict[str, Any] = {
        key: _tighten(value) for key, value in node.items() if key != "default"
    }
    if "$ref" in schema:
        # Strict mode forbids keywords beside a $ref; every field is required,
        # so a default carries no meaning for the model anyway.
        return {"$ref": schema["$ref"]}
    if schema.get("type") != "object":
        return schema
    if "properties" not in schema:
        # Strict mode cannot express a free-form object, and the endpoint would
        # reject it at request time; fail here, where the cause is obvious.
        raise ValueError(
            "strict schema cannot contain a free-form object; give the field explicit fields"
        )
    schema["additionalProperties"] = False
    schema["required"] = list(schema["properties"])
    return schema


class ChatTransport:
    """Shared HTTP transport: bounded concurrency, retries, typed failures."""

    def __init__(
        self,
        settings: ChatBackendSettings,
        *,
        retry: RetrySettings,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._settings = settings
        self._retry = retry
        self._limit = asyncio.Semaphore(settings.max_concurrent_requests)
        self._client = client or httpx.AsyncClient(timeout=settings.request_timeout_seconds)

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Release the HTTP connection pool."""
        await self._client.aclose()

    async def post(self, path: str, payload: dict[str, Any], label: str) -> dict[str, Any]:
        """POST to ``path`` under the configured base URL, retrying transient failures."""
        return await retry_async(
            lambda: self._post_once(path, payload, label),
            settings=self._retry,
            description=f"azure:{label}",
        )

    async def _post_once(self, path: str, payload: dict[str, Any], label: str) -> dict[str, Any]:
        url = f"{self._settings.base_url.rstrip('/')}/{path.lstrip('/')}"
        async with self._limit:
            try:
                response = await self._client.post(
                    url, headers=self._settings.headers(), json=payload
                )
            except httpx.HTTPError as error:
                raise TransientError(f"{label}: request failed: {error}") from error

        if response.status_code in _RETRYABLE_STATUS:
            raise TransientError(f"{label}: backend returned {response.status_code}")
        if response.status_code >= 400:
            raise ProviderError(
                f"{label}: backend returned {response.status_code}: {response.text[:300]}"
            )
        body: dict[str, Any] = response.json()
        return body


class ChatLLMProvider:
    """OpenAI-compatible :class:`~context_graph.providers.base.LLMProvider`."""

    def __init__(
        self,
        *,
        backend: ChatBackendSettings,
        llm: LLMSettings,
        transport: ChatTransport | None = None,
        retry: RetrySettings | None = None,
    ) -> None:
        self._backend = backend
        self._llm = llm
        self._transport = transport or ChatTransport(backend, retry=retry or RetrySettings())

    @property
    def model(self) -> str:
        return self._backend.chat_model

    def estimate_tokens(self, text: str) -> int:
        """Local estimate used to assemble batches without a network round-trip."""
        return math.ceil(len(text) / self._llm.chars_per_token)

    async def count_tokens(self, *, system: str, user: str) -> int:
        """Token count for an assembled request.

        Neither backend exposes a token-counting endpoint, so this is the same
        local estimate; the configured safety margin absorbs the error.
        """
        return self.estimate_tokens(system) + self.estimate_tokens(user)

    async def structured[M: BaseModel](
        self, *, system: str, user: str, schema: type[M], max_output_tokens: int
    ) -> M:
        """Return a validated instance of ``schema``."""
        return await self.complete_json(
            system=system,
            user=user,
            schema=schema,
            max_output_tokens=max_output_tokens,
            label=schema.__name__,
        )

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
        """Chat completion constrained to ``schema``; ``seed`` aids reproducibility."""
        json_schema_mode = (
            self._backend.structured_output is StructuredOutputMode.JSON_SCHEMA
        )
        instructions = system if json_schema_mode else _with_schema(system, schema)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": instructions},
            {"role": "user", "content": user},
        ]

        last_error: str | None = None
        for attempt in range(self._backend.repair_attempts + 1):
            if last_error is not None:
                # Show the model its own mistake rather than silently retrying.
                messages = [
                    *messages,
                    {"role": "user", "content": _repair_request(last_error, schema)},
                ]
            content = await self._chat(messages, schema, max_output_tokens, seed, label)
            try:
                return schema.model_validate_json(_json_payload(content))
            except (ValidationError, ValueError) as error:
                last_error = str(error)[:400]
                log_event(
                    _logger,
                    "llm.output_invalid",
                    label=label,
                    schema=schema.__name__,
                    attempt=attempt + 1,
                    error=last_error[:200],
                )
        raise ExtractionValidationError(
            f"{label}: output did not satisfy {schema.__name__} after "
            f"{self._backend.repair_attempts + 1} attempts"
        )

    async def _chat(
        self,
        messages: list[dict[str, Any]],
        schema: type[BaseModel],
        max_output_tokens: int,
        seed: int | None,
        label: str,
    ) -> str:
        """One chat completion, returning the raw message content."""
        payload: dict[str, Any] = {
            "model": self._backend.chat_model,
            "messages": messages,
            self._backend.token_parameter: max_output_tokens,
            "temperature": self._llm.temperature,
        }
        if self._backend.structured_output is StructuredOutputMode.JSON_SCHEMA:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__.lower(),
                    "strict": True,
                    "schema": strict_schema(schema),
                },
            }
        else:
            payload["response_format"] = {"type": "json_object"}
        if seed is not None:
            payload["seed"] = seed

        body = await self._transport.post("chat/completions", payload, label)
        choice = body["choices"][0]
        usage: dict[str, Any] = body.get("usage") or {}
        log_event(
            _logger,
            "llm.call_completed",
            model=self._backend.chat_model,
            schema=schema.__name__,
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
            finish_reason=choice.get("finish_reason"),
        )
        if choice.get("finish_reason") == "length":
            # A truncated body is not valid JSON: never treat it as success.
            raise ExtractionValidationError(f"{label}: output truncated at the token cap")
        content = choice["message"].get("content")
        if not content:
            raise ExtractionValidationError(f"{label}: model returned an empty message")
        return str(content)

    async def aclose(self) -> None:
        """Release the transport owned by this provider."""
        await self._transport.aclose()


class EmbeddingProviderImpl:
    """Embeddings from either backend, behind one provider interface."""

    def __init__(
        self,
        *,
        embedding: EmbeddingSettings,
        retry: RetrySettings | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._embedding = embedding
        self._retry = retry or RetrySettings()
        self._client = client or httpx.AsyncClient(timeout=180.0)

    @property
    def model(self) -> str:
        return self._embedding.model

    @property
    def dimension(self) -> int:
        return self._embedding.dimension

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed ``texts`` in provider-sized batches, preserving order."""
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._embedding.batch_size):
            chunk = texts[start : start + self._embedding.batch_size]
            vectors.extend(await self._embed_chunk(chunk))
        return vectors

    async def _embed_chunk(self, texts: Sequence[str]) -> list[list[float]]:
        in_house = self._embedding.backend is EmbeddingBackend.IN_HOUSE
        # NIM embed models take input_type and reject the OpenAI "dimensions" field.
        nvidia = "api.nvidia.com" in self._embedding.base_url
        payload = (
            {
                "inputs": [
                    {"id": str(index), "text": text if text.strip() else " "}
                    for index, text in enumerate(texts)
                ],
                "include_metadata": False,
                "model": self._embedding.model,
                "dim": self._embedding.dimension,
                "quantize_int8": self._embedding.quantize,
            }
            if in_house
            else {
                "model": self._embedding.model,
                "input": list(texts),
                **(
                    {"input_type": "passage", "encoding_format": "float", "truncate": "END"}
                    if nvidia
                    else {"dimensions": self._embedding.dimension}
                ),
            }
        )
        body = await retry_async(
            lambda: self._post(payload),
            settings=self._retry,
            description="embeddings",
        )
        return (
            _in_house_vectors(body, len(texts)) if in_house else _openai_vectors(body, len(texts))
        )

    async def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        """One embeddings call, translating failures into pipeline errors."""
        try:
            response = await self._client.post(
                self._embedding.embeddings_url,
                headers=self._embedding.headers(),
                json=payload,
            )
        except httpx.HTTPError as error:
            raise TransientError(f"embeddings request failed: {error}") from error
        if response.status_code in _RETRYABLE_STATUS:
            raise TransientError(f"embeddings returned {response.status_code}")
        if response.status_code >= 400:
            raise ProviderError(
                f"embeddings returned {response.status_code}: {response.text[:300]}"
            )
        body: dict[str, Any] = response.json()
        return body

    async def aclose(self) -> None:
        """Release the HTTP connection pool."""
        await self._client.aclose()


def _openai_vectors(body: dict[str, Any], expected: int) -> list[list[float]]:
    """Vectors from an OpenAI-compatible response, ordered by ``index``."""
    data = body.get("data")
    if not isinstance(data, list) or len(data) != expected:
        raise ProviderError("embedding response did not match the request size")
    items: list[dict[str, Any]] = [dict(item) for item in data]
    items.sort(key=lambda item: int(item["index"]))
    return [list(item["embedding"]) for item in items]


def _in_house_vectors(body: dict[str, Any], expected: int) -> list[list[float]]:
    """Vectors from the in-house gateway, restored to the input order."""
    results = body.get("results")
    if not isinstance(results, list) or len(results) != expected:
        raise ProviderError("embedding response did not match the request size")
    by_id = {str(item["id"]): list(item["embedding"]) for item in results}
    missing = [str(index) for index in range(expected) if str(index) not in by_id]
    if missing:
        raise ProviderError(f"embedding response is missing inputs: {missing[:5]}")
    return [by_id[str(index)] for index in range(expected)]


def _with_schema(system: str, schema: type[BaseModel]) -> str:
    """Put the required JSON schema in the prompt for JSON-mode backends."""
    rendered = json.dumps(strict_schema(schema), separators=(",", ":"))
    return (
        f"{system}\n\n"
        "Reply with a single JSON object and nothing else - no prose, no code\n"
        "fences. It must satisfy this JSON schema exactly, including every\n"
        f"required field:\n{rendered}"
    )


def _repair_request(error: str, schema: type[BaseModel]) -> str:
    """Ask for a corrected object, showing the model what was wrong."""
    return (
        f"Your previous reply did not satisfy {schema.__name__}:\n{error}\n"
        "Reply again with only the corrected JSON object."
    )


def _json_payload(content: str) -> str:
    """The JSON object in a reply, tolerating code fences and stray prose."""
    text = content.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        text = text[4:].lstrip() if text.lower().startswith("json") else text.lstrip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("reply contained no JSON object")
    return text[start : end + 1]
