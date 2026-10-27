"""Input/output schema: single source of truth for teacher, filters, training and serving.

Register ids and tag ids are read from spec/*.yaml, so the JSON schema exported
from here constrains guided decoding to valid registers and tags.

    python -m registre.schema > spec/output.schema.json
"""

from __future__ import annotations

import json
from enum import Enum
from functools import cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, create_model, model_validator

SPEC_DIR = Path(__file__).resolve().parents[2] / "spec"

MAX_ANNOTATIONS = 4  # per variant; bounds output length (latency)
MAX_NOTE_CHARS = 200
MAX_ASSUMPTIONS = 3


@cache
def load_spec(name: str) -> dict[str, Any]:
    return yaml.safe_load((SPEC_DIR / f"{name}.yaml").read_text(encoding="utf-8"))


def register_ids() -> list[str]:
    return [r["id"] for r in load_spec("registers")["registers"]]


def tag_ids() -> list[str]:
    return [t["id"] for t in load_spec("taxonomy")["tags"]]


TagId = Enum("TagId", {t: t for t in tag_ids()}, type=str)


class TranslationInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str = Field(min_length=1, max_length=500)
    source_lang: Literal["es", "en"] = "es"  # "en" kept for a possible later extension
    speaker_gender: Literal["m", "f"]


SourceAddress = Literal["tu", "usted", "vos", "plural", "none", "ambiguous"]


class Annotation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    span: str = Field(min_length=1, description="Exact substring of the variant text.")
    tag: TagId
    note: str = Field(min_length=1, max_length=MAX_NOTE_CHARS)


class Variant(BaseModel):
    """`text` comes before `annotations` so the model writes the sentence first."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)
    annotations: list[Annotation] = Field(max_length=MAX_ANNOTATIONS)

    @model_validator(mode="after")
    def spans_are_grounded(self) -> Variant:
        # Not expressible in JSON schema, so it lives here (filters + serving).
        missing = [a.span for a in self.annotations if a.span not in self.text]
        if missing:
            raise ValueError(f"spans not found in text: {missing}")
        return self


# One field per register, in spec order: guided decoding then forces every
# register to be present and generated in that order.
Variants = create_model(
    "Variants",
    __config__=ConfigDict(extra="forbid"),
    **{rid: (Variant, ...) for rid in register_ids()},
)


class TranslationOutput(BaseModel):
    """Analysis of the source first, so the variants are conditioned on it.

    `source_address` is what the Spanish source uses to address someone. It is
    reported, never obeyed: all four registers are produced regardless. On
    CoCoA-MT it can be scored against the human formal/informal labels.
    """

    model_config = ConfigDict(extra="forbid")

    source_address: SourceAddress
    assumptions: list[str] = Field(default_factory=list, max_length=MAX_ASSUMPTIONS)
    variants: Variants  # type: ignore[valid-type]


def json_schema() -> dict[str, Any]:
    return TranslationOutput.model_json_schema()


if __name__ == "__main__":
    print(json.dumps(json_schema(), ensure_ascii=False, indent=2))
