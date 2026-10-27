"""Day-1 smoke test: does the vLLM server enforce our JSON schema?

Run with ./run vllm-up, then ./run smoke. The dev container reaches the server by its
compose service name ("vllm"), not localhost.
"""

import json
import os

from openai import OpenAI
from pydantic import ValidationError

from registre.schema import TranslationOutput, json_schema

BASE_URL = os.environ.get("VLLM_URL", "http://vllm:8000/v1")
MODEL = os.environ.get("SMOKE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
PROMPT = (
    "Traduce al francés en cuatro registros (soutenu, courant, amical, familier) "
    "y explica cada uno: ¿Me puedes mandar el archivo?"
)

client = OpenAI(base_url=BASE_URL, api_key="unused")
resp = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": PROMPT}],
    response_format={
        "type": "json_schema",
        "json_schema": {"name": "translation", "schema": json_schema()},
    },
    temperature=0,
    max_tokens=1500,
)
choice = resp.choices[0]
raw = choice.message.content or ""
print(raw)
print(f"\nfinish_reason={choice.finish_reason}  tokens={resp.usage.completion_tokens}")
if choice.finish_reason == "length":
    print("✗ truncated at max_tokens: the JSON is cut off, not the grammar's fault")

try:
    json.loads(raw)
    print("✓ valid JSON: syntax is guaranteed by constrained decoding")
    TranslationOutput.model_validate_json(raw)
    print("✓ passes Pydantic, including span grounding")
except json.JSONDecodeError as e:
    print(f"✗ not valid JSON: {e}")
except ValidationError as e:
    print(f"✗ valid JSON but rejected by Pydantic (expected with a tiny model):\n{e}")
