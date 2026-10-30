"""Strict-schema rendering rules a json_schema backend enforces at request time."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel, Field

from context_graph.domain.extraction import ConceptAdjudication, ExtractionEnvelope
from context_graph.providers.chat_backend import strict_schema


class WithDefault(BaseModel):
    name: str = "unnamed"
    count: int = 3


class FreeForm(BaseModel):
    payload: dict[str, Any]


def test_defaults_are_stripped_and_every_field_is_required() -> None:
    schema = strict_schema(WithDefault)
    assert schema["required"] == ["name", "count"]
    assert all("default" not in value for value in schema["properties"].values())


def test_ref_keeps_no_sibling_keywords() -> None:
    # A $ref carrying a sibling `default` is rejected by the endpoint.
    schema = strict_schema(ExtractionEnvelope)
    signal = schema["$defs"]["ExtractedSignal"]["properties"]
    assert signal["context_type"] == {"$ref": "#/$defs/ContextType"}
    assert schema["$defs"]["ExtractedSignal"]["additionalProperties"] is False


def test_free_form_objects_fail_locally_instead_of_at_the_endpoint() -> None:
    with pytest.raises(ValueError, match="free-form object"):
        strict_schema(FreeForm)


def test_pipeline_schemas_render() -> None:
    for model in (ExtractionEnvelope, ConceptAdjudication):
        assert strict_schema(model)["additionalProperties"] is False


class Nested(BaseModel):
    inner: WithDefault = Field(default_factory=WithDefault)


def test_nested_models_are_tightened_too() -> None:
    schema = strict_schema(Nested)
    assert schema["$defs"]["WithDefault"]["additionalProperties"] is False
    assert schema["required"] == ["inner"]
