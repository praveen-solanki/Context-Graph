"""Chat backend behaviour, error translation and the shared retry policy."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import BaseModel, SecretStr

from context_graph.config import (
    AuthScheme,
    ChatBackendSettings,
    EmbeddingBackend,
    EmbeddingSettings,
    LLMSettings,
    RetrySettings,
    StructuredOutputMode,
)
from context_graph.errors import ExtractionValidationError, ProviderError, TransientError
from context_graph.providers.chat_backend import (
    EmbeddingProviderImpl,
    ChatLLMProvider,
    ChatTransport,
    strict_schema,
)
from context_graph.retry import retry_async


class Reply(BaseModel):
    text: str


def backend_settings(**overrides: Any) -> ChatBackendSettings:
    return ChatBackendSettings(
        api_key=SecretStr("test-key"),
        base_url="https://example.test/v1",
        chat_model="chat-test",
        **overrides,
    )


Handler = Callable[[httpx.Request], httpx.Response]


def transport_for(
    handler: Handler,
    retry: RetrySettings | None = None,
    settings: ChatBackendSettings | None = None,
) -> ChatTransport:
    """Transport wired to an in-process handler instead of the network."""
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return ChatTransport(
        settings or backend_settings(),
        retry=retry or RetrySettings(max_attempts=2, base_backoff_seconds=0.0),
        client=client,
    )


def llm_for(handler: Handler, retry: RetrySettings | None = None) -> ChatLLMProvider:
    return ChatLLMProvider(
        backend=backend_settings(), llm=LLMSettings(), transport=transport_for(handler, retry)
    )


def chat_response(content: str, finish_reason: str = "stop") -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [
                {"finish_reason": finish_reason, "message": {"content": content, "role": "assistant"}}
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        },
    )


async def test_json_schema_backend_constrains_the_request() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        captured.update(json.loads(request.content))
        captured["url"] = str(request.url)
        captured["api_key"] = request.headers.get("api-key")
        return chat_response('{"text": "ok"}')

    settings = backend_settings(
        structured_output=StructuredOutputMode.JSON_SCHEMA,
        token_parameter="max_completion_tokens",
        auth_scheme=AuthScheme.API_KEY,
    )
    provider = ChatLLMProvider(
        backend=settings,
        llm=LLMSettings(),
        transport=transport_for(handler, settings=settings),
    )
    result = await provider.structured(system="s", user="u", schema=Reply, max_output_tokens=100)

    assert result == Reply(text="ok")
    assert captured["url"].endswith("/v1/chat/completions")
    assert captured["api_key"] == "test-key"
    # The token-cap parameter name is configuration, not a hardcoded choice.
    assert captured["max_completion_tokens"] == 100
    assert captured["response_format"]["json_schema"]["strict"] is True
    assert "seed" not in captured  # only the generator pins a seed


async def test_json_object_backend_puts_the_schema_in_the_prompt() -> None:
    """ministral-3b has no server-side schema enforcement, so we validate here."""
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        captured.update(json.loads(request.content))
        captured["auth"] = request.headers.get("authorization")
        captured["service_code"] = request.headers.get("ai-model-service-code")
        return chat_response('```json\n{"text": "ok"}\n```')

    settings = backend_settings(service_code="graph")
    provider = ChatLLMProvider(
        backend=settings, llm=LLMSettings(), transport=transport_for(handler, settings=settings)
    )
    result = await provider.structured(system="s", user="u", schema=Reply, max_output_tokens=100)

    # A fenced reply still parses, and the schema was given in the system prompt.
    assert result == Reply(text="ok")
    assert captured["response_format"] == {"type": "json_object"}
    assert captured["max_tokens"] == 100
    assert "JSON schema" in captured["messages"][0]["content"]
    assert captured["auth"] == "Bearer test-key"
    assert captured["service_code"] == "graph"


async def test_invalid_json_is_repaired_with_the_error_shown_to_the_model() -> None:
    replies = ['{"wrong": 1}', '{"text": "second try"}']

    def handler(request: httpx.Request) -> httpx.Response:
        return chat_response(replies.pop(0))

    result = await llm_for(handler).structured(
        system="s", user="u", schema=Reply, max_output_tokens=100
    )

    assert result == Reply(text="second try")


async def test_output_that_never_validates_is_a_failure() -> None:
    with pytest.raises(ExtractionValidationError, match="did not satisfy"):
        await llm_for(lambda _: chat_response('{"wrong": 1}')).structured(
            system="s", user="u", schema=Reply, max_output_tokens=100
        )


async def test_truncated_output_is_never_treated_as_success() -> None:
    with pytest.raises(ExtractionValidationError):
        await llm_for(lambda _: chat_response('{"text": "partial', "length")).structured(
            system="s", user="u", schema=Reply, max_output_tokens=100
        )


async def test_rate_limits_are_retried_and_then_succeed() -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(429, json={"error": "slow down"})
        return chat_response('{"text": "ok"}')

    result = await llm_for(handler).structured(
        system="s", user="u", schema=Reply, max_output_tokens=100
    )

    assert result == Reply(text="ok")
    assert attempts == 2


async def test_bad_requests_are_not_retried() -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(400, json={"error": "bad model"})

    with pytest.raises(ProviderError):
        await llm_for(handler).structured(system="s", user="u", schema=Reply, max_output_tokens=100)
    assert attempts == 1


async def test_embeddings_preserve_input_order_and_request_the_configured_dimension() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        captured.update(json.loads(request.content))
        # Deliberately out of order: `index` is authoritative.
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": 1, "embedding": [0.2, 0.2]},
                    {"index": 0, "embedding": [0.1, 0.1]},
                ]
            },
        )

    provider = EmbeddingProviderImpl(
        embedding=EmbeddingSettings(
            backend=EmbeddingBackend.OPENAI, dimension=2, base_url="https://example.test/v1"
        ),
        retry=RetrySettings(max_attempts=1, base_backoff_seconds=0.0),
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    vectors = await provider.embed(["first", "second"])

    assert vectors == [[0.1, 0.1], [0.2, 0.2]]
    assert captured["dimensions"] == 2
    assert captured["model"] == EmbeddingSettings().model
    assert provider.dimension == 2


async def test_embedding_response_of_the_wrong_size_is_an_error() -> None:
    provider = EmbeddingProviderImpl(
        embedding=EmbeddingSettings(
            backend=EmbeddingBackend.OPENAI, dimension=2, base_url="https://example.test/v1"
        ),
        retry=RetrySettings(max_attempts=1, base_backoff_seconds=0.0),
        client=httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda _: httpx.Response(200, json={"data": [{"index": 0, "embedding": [0.1]}]})
            )
        ),
    )
    with pytest.raises(ProviderError):
        await provider.embed(["first", "second"])


def test_local_token_estimate_scales_with_length() -> None:
    provider = ChatLLMProvider(
        backend=backend_settings(),
        llm=LLMSettings(chars_per_token=3.5),
        transport=transport_for(lambda _: chat_response("{}")),
    )
    assert provider.estimate_tokens("x" * 35) == 10
    assert provider.estimate_tokens("") == 0
    assert provider.model == "chat-test"


def test_strict_schema_forbids_extra_fields() -> None:
    schema = strict_schema(Reply)
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["text"]


async def test_retry_stops_after_the_configured_attempts() -> None:
    attempts = 0

    async def failing() -> None:
        nonlocal attempts
        attempts += 1
        raise TransientError("boom")

    with pytest.raises(TransientError):
        await retry_async(
            failing,
            settings=RetrySettings(max_attempts=3, base_backoff_seconds=0.0),
            description="test",
        )
    assert attempts == 3


async def test_retry_does_not_swallow_non_transient_errors() -> None:
    async def failing() -> None:
        raise ProviderError("permanent")

    with pytest.raises(ProviderError):
        await retry_async(
            failing,
            settings=RetrySettings(max_attempts=3, base_backoff_seconds=0.0),
            description="test",
        )


async def test_in_house_embedding_payload_and_ordering() -> None:
    """The in-house gateway has its own payload, headers and response shape."""
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        captured.update(json.loads(request.content))
        captured["service_code"] = request.headers.get("x-smtip-service-code")
        captured["tenant"] = request.headers.get("x-smtip-tid")
        captured["priority"] = request.headers.get("x-smtip-embedding-priority")
        # Deliberately out of order: ids are authoritative.
        return httpx.Response(
            200,
            json={
                "results": [
                    {"id": "1", "embedding": [0.2, 0.2]},
                    {"id": "0", "embedding": [0.1, 0.1]},
                ]
            },
        )

    provider = EmbeddingProviderImpl(
        embedding=EmbeddingSettings(
            backend=EmbeddingBackend.IN_HOUSE,
            base_url="https://example.test/v1/embeddings/embed",
            model="all-MiniLM-L6-v2",
            dimension=2,
            service_code="graph",
            tenant_id="tenant-1",
            priority="sync",
        ),
        retry=RetrySettings(max_attempts=1, base_backoff_seconds=0.0),
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    vectors = await provider.embed(["first", "second"])

    assert vectors == [[0.1, 0.1], [0.2, 0.2]]
    assert captured["inputs"] == [{"id": "0", "text": "first"}, {"id": "1", "text": "second"}]
    assert captured["dim"] == 2
    assert captured["service_code"] == "graph"
    assert captured["tenant"] == "tenant-1"
    assert captured["priority"] == "sync"


async def test_in_house_embedding_response_missing_an_input_is_an_error() -> None:
    provider = EmbeddingProviderImpl(
        embedding=EmbeddingSettings(
            backend=EmbeddingBackend.IN_HOUSE,
            base_url="https://example.test/v1/embeddings/embed",
            dimension=2,
        ),
        retry=RetrySettings(max_attempts=1, base_backoff_seconds=0.0),
        client=httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda _: httpx.Response(
                    200,
                    json={"results": [{"id": "0", "embedding": [0.1]}, {"id": "9", "embedding": [0.2]}]},
                )
            )
        ),
    )
    with pytest.raises(ProviderError, match="missing inputs"):
        await provider.embed(["first", "second"])


def test_backend_headers_follow_the_configured_scheme() -> None:
    bearer = backend_settings(service_code="recognition").headers()
    assert bearer["Authorization"] == "Bearer test-key"
    # The orchestrator's gateway header travels with the credential.
    assert bearer["ai-model-service-code"] == "recognition"

    api_key = backend_settings(auth_scheme=AuthScheme.API_KEY, service_code=None).headers()
    assert api_key == {"api-key": "test-key"}


def test_defaults_target_nvidia(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Chat and embeddings default to NVIDIA NIM. Another OpenAI-compatible host is config."""
    from context_graph.config import LLMSettings, Settings

    # No .env and no overriding variables: these are the code defaults.
    monkeypatch.chdir(tmp_path)
    for name in (
        "LLM_BASE_URL", "OPENAI_BASE_URL", "LLM_MODEL", "MODEL_NAME", "LLM_TOKEN_PARAMETER",
        "LLM_STRUCTURED_OUTPUT", "LLM_SERVICE_CODE", "EMBEDDING_BACKEND", "EMBEDDING_MODEL",
        "EMBEDDING_MODEL_NAME", "EMBEDDING_DIMENSION", "EMBEDDING_BASE_URL", "IN_HOUSE_EMBEDDINGS_URL",
        "EMBEDDING_AUTH_SCHEME", "EMBEDDING_API_KEY", "NVIDIA_API_KEY", "LLM_API_KEY", "OPENAI_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")

    settings = Settings()
    assert settings.chat.chat_model == "nvidia/nemotron-3.5-bolt-e4b"
    assert settings.chat.base_url == "https://integrate.api.nvidia.com/v1"
    assert settings.chat.chat_completions_url.endswith("/chat/completions")
    assert settings.chat.token_parameter == "max_tokens"
    assert settings.chat.structured_output is StructuredOutputMode.JSON_OBJECT
    assert settings.chat.service_code is None
    assert settings.embedding.backend is EmbeddingBackend.OPENAI
    assert settings.embedding.model == "nvidia/nemotron-3.5-embed-1b"
    assert settings.embedding.dimension == 2048
    assert settings.embedding.embeddings_url == "https://integrate.api.nvidia.com/v1/embeddings"
    assert settings.chat.headers()["Authorization"] == "Bearer nvapi-test"
    assert settings.embedding.headers()["Authorization"] == "Bearer nvapi-test"
    llm = LLMSettings()
    assert llm.context_limit_tokens == 10_240
    assert llm.max_output_tokens >= llm.max_signals_per_batch * 130
