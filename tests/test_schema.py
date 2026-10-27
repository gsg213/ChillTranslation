import copy
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from registre.schema import (
    TranslationInput,
    TranslationOutput,
    json_schema,
    register_ids,
    tag_ids,
)

FIXTURE = json.loads((Path(__file__).parent / "fixtures/example_output.json").read_text())


def test_golden_example_validates():
    TranslationInput.model_validate(FIXTURE["input"])
    TranslationOutput.model_validate(FIXTURE["output"])


def test_unknown_source_address_rejected():
    bad = copy.deepcopy(FIXTURE["output"])
    bad["source_address"] = "vosotros"
    with pytest.raises(ValidationError):
        TranslationOutput.model_validate(bad)


def test_too_many_annotations_rejected():
    bad = copy.deepcopy(FIXTURE["output"])
    anns = bad["variants"]["amical"]["annotations"]
    bad["variants"]["amical"]["annotations"] = anns + anns[:1]  # 5 > MAX_ANNOTATIONS
    with pytest.raises(ValidationError):
        TranslationOutput.model_validate(bad)


def test_ungrounded_span_rejected():
    bad = copy.deepcopy(FIXTURE["output"])
    bad["variants"]["amical"]["annotations"][0]["span"] = "Vous pouvez"
    with pytest.raises(ValidationError, match="spans not found"):
        TranslationOutput.model_validate(bad)


def test_unknown_tag_rejected():
    bad = copy.deepcopy(FIXTURE["output"])
    bad["variants"]["soutenu"]["annotations"][0]["tag"] = "not_a_tag"
    with pytest.raises(ValidationError):
        TranslationOutput.model_validate(bad)


def test_missing_register_rejected():
    bad = copy.deepcopy(FIXTURE["output"])
    del bad["variants"]["familier"]
    with pytest.raises(ValidationError):
        TranslationOutput.model_validate(bad)


def test_json_schema_enumerates_tags_and_orders_fields():
    schema = json.dumps(json_schema())
    assert all(t in schema for t in tag_ids())
    assert list(json_schema()["properties"]) == ["source_address", "assumptions", "variants"]
    variants = json_schema()["$defs"]["Variants"]
    assert list(variants["properties"]) == register_ids()
    assert variants["required"] == register_ids()


def test_taxonomy_ids_unique():
    assert len(tag_ids()) == len(set(tag_ids()))
