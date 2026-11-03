"""Run a model over input sentences and save its structured outputs.

Used for the day-2 teacher run (and the day-1 bake-off). Two backends:
  vllm    offline batch inference on a GPU node, inside the vLLM image (Grid5000)
  openai  any OpenAI-compatible HTTP endpoint: your own vLLM server or an API provider

Input JSONL rows: {"id", "source", "speaker_gender"}. Output JSONL: one record per input with
the raw model text, finish_reason, token counts and provenance (model, prompt hash, settings).
Re-running skips ids already in the output file, so an interrupted run resumes where it stopped.

Example (Grid5000, inside the vLLM image):
  python3 -m registre.generate --backend vllm --model <teacher> \\
      --inputs data/teacher/inputs.jsonl --out data/teacher/raw.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

from registre.prompt import messages, prompt_hash
from registre.schema import json_schema


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def append_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


class VLLMBackend:
    """Offline batch inference: vLLM schedules the whole batch itself (continuous batching)."""

    def __init__(self, args: argparse.Namespace):
        from vllm import LLM, SamplingParams
        from vllm.sampling_params import StructuredOutputsParams

        self.llm = LLM(
            model=args.model,
            tensor_parallel_size=args.tp,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_mem,
            seed=args.seed,
        )
        guided = StructuredOutputsParams(json=json_schema()) if args.guided else None
        self.params = SamplingParams(
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            seed=args.seed,
            structured_outputs=guided,
        )
        self.template_kwargs = {"enable_thinking": False} if args.no_thinking else None

    def generate(self, conversations: list[list[dict]]) -> list[dict]:
        outs = self.llm.chat(
            conversations,
            self.params,
            use_tqdm=True,
            chat_template_kwargs=self.template_kwargs,
        )
        return [
            {
                "text": o.outputs[0].text,
                "finish_reason": o.outputs[0].finish_reason,
                "prompt_tokens": len(o.prompt_token_ids or []),
                "completion_tokens": len(o.outputs[0].token_ids),
            }
            for o in outs
        ]


class OpenAIBackend:
    """HTTP endpoint, several requests in flight at once, retries with backoff."""

    def __init__(self, args: argparse.Namespace):
        from openai import OpenAI

        self.client = OpenAI(base_url=args.base_url, api_key=os.environ.get(args.api_key_env, "-"))
        self.args = args

    def _one(self, conversation: list[dict]) -> dict:
        extra = {}
        if self.args.guided:
            schema = {"name": "translation", "schema": json_schema()}
            extra["response_format"] = {"type": "json_schema", "json_schema": schema}
        if self.args.no_thinking:
            extra["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        for attempt in range(4):
            try:
                r = self.client.chat.completions.create(
                    model=self.args.model,
                    messages=conversation,
                    temperature=self.args.temperature,
                    max_tokens=self.args.max_tokens,
                    seed=self.args.seed,
                    **extra,
                )
                c = r.choices[0]
                return {
                    "text": c.message.content or "",
                    "finish_reason": c.finish_reason,
                    "prompt_tokens": r.usage.prompt_tokens if r.usage else None,
                    "completion_tokens": r.usage.completion_tokens if r.usage else None,
                }
            except Exception as e:  # network errors, rate limits: retry, then record the error
                if attempt == 3:
                    return {"text": "", "finish_reason": "error", "error": repr(e)}
                time.sleep(2**attempt)
        raise AssertionError("unreachable")

    def generate(self, conversations: list[list[dict]]) -> list[dict]:
        with ThreadPoolExecutor(self.args.workers) as pool:
            return list(pool.map(self._one, conversations))


def run(args: argparse.Namespace, backend) -> int:
    inputs = [r for r in read_jsonl(args.inputs) if args.split in (None, r.get("split"))]
    done = {r["id"] for r in read_jsonl(args.out)}
    todo = [r for r in inputs if r["id"] not in done][: args.limit]
    print(f"{len(inputs)} inputs, {len(done)} already done, {len(todo)} to generate")
    provenance = {
        "model": args.model,
        "backend": args.backend,
        "prompt_hash": prompt_hash(),
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "guided": args.guided,
        "seed": args.seed,
    }
    for start in range(0, len(todo), args.chunk):  # write per chunk: a crash loses one chunk
        chunk = todo[start : start + args.chunk]
        convs = [messages(r["source"], r["speaker_gender"], not args.no_system_role) for r in chunk]
        results = backend.generate(convs)
        now = datetime.now(UTC).isoformat(timespec="seconds")
        append_jsonl(
            args.out,
            [
                {"id": r["id"], **res, **provenance, "created": now}
                for r, res in zip(chunk, results, strict=True)
            ],
        )
        print(f"  wrote {start + len(chunk)}/{len(todo)}")
    return len(todo)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--backend", choices=["vllm", "openai"], default="vllm")
    p.add_argument("--model", required=True)
    p.add_argument("--inputs", type=Path, default=Path("data/teacher/inputs.jsonl"))
    p.add_argument("--out", type=Path, default=Path("data/teacher/raw.jsonl"))
    p.add_argument(
        "--limit", type=int, default=None, help="only the first N pending inputs (pilots)"
    )
    p.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default=None,
        help="only inputs from this split (pilots and bake-off: val)",
    )
    p.add_argument("--chunk", type=int, default=512)
    p.add_argument(
        "--no-guided",
        dest="guided",
        action="store_false",
        help="disable schema-constrained decoding (bake-off: measure native ability)",
    )
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=1500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--no-system-role",
        action="store_true",
        help="merge the system prompt into the user turn (for templates without one)",
    )
    p.add_argument(
        "--no-thinking",
        action="store_true",
        help="disable 'thinking' for reasoning models (it multiplies output tokens)",
    )
    # vllm backend
    p.add_argument("--tp", type=int, default=1, help="tensor parallel size (number of GPUs)")
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--gpu-mem", type=float, default=0.9)
    # openai backend
    p.add_argument("--base-url", default="http://vllm:8000/v1")
    p.add_argument("--api-key-env", default="TEACHER_API_KEY")
    p.add_argument("--workers", type=int, default=8)
    return p.parse_args(argv)


def main() -> None:
    args = parse_args()
    backend = VLLMBackend(args) if args.backend == "vllm" else OpenAIBackend(args)
    run(args, backend)


if __name__ == "__main__":
    main()
