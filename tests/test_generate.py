import json

from registre.generate import llm_kwargs, parse_args, read_jsonl, run
from registre.schema import SPEC_DIR

GOLDEN = json.loads((SPEC_DIR / "golden_example.json").read_text())["output"]


class FakeBackend:
    """Stands in for vLLM or an API: lets us test everything except the model itself."""

    def __init__(self):
        self.calls = 0

    def generate(self, conversations):
        self.calls += len(conversations)
        return [
            {"text": json.dumps(GOLDEN), "finish_reason": "stop", "prompt_tokens": 1,
             "completion_tokens": 1}
            for _ in conversations
        ]  # fmt: skip


def _inputs(tmp_path, n):
    path = tmp_path / "inputs.jsonl"
    rows = [
        {"id": f"x{i}", "source": f"Frase {i}.", "speaker_gender": "m",
         "split": "val" if i % 2 else "train"}
        for i in range(n)
    ]  # fmt: skip
    path.write_text("\n".join(json.dumps(r) for r in rows))
    return path


def test_run_writes_records_with_provenance(tmp_path):
    out = tmp_path / "raw.jsonl"
    args = parse_args(["--model", "m", "--inputs", str(_inputs(tmp_path, 3)), "--out", str(out)])
    run(args, FakeBackend())
    rows = read_jsonl(out)
    assert [r["id"] for r in rows] == ["x0", "x1", "x2"]
    assert {"model", "prompt_hash", "temperature", "guided", "created"} <= rows[0].keys()


def test_run_resumes_and_respects_limit(tmp_path):
    inputs, out = _inputs(tmp_path, 5), tmp_path / "raw.jsonl"
    base = ["--model", "m", "--inputs", str(inputs), "--out", str(out), "--chunk", "2"]
    first = FakeBackend()
    run(parse_args([*base, "--limit", "2"]), first)
    second = FakeBackend()
    run(parse_args(base), second)
    assert (first.calls, second.calls) == (2, 3)  # the second run only does what's missing
    assert len(read_jsonl(out)) == 5


def test_split_filter(tmp_path):
    out = tmp_path / "raw.jsonl"
    argv = ["--model", "m", "--inputs", str(_inputs(tmp_path, 6)), "--out", str(out)]
    run(parse_args([*argv, "--split", "val"]), FakeBackend())
    assert [r["id"] for r in read_jsonl(out)] == ["x1", "x3", "x5"]


def test_mistral_and_text_only_flags():
    plain = llm_kwargs(parse_args(["--model", "m"]))
    assert "tokenizer_mode" not in plain and "limit_mm_per_prompt" not in plain
    mistral = llm_kwargs(parse_args(["--model", "m", "--mistral-format", "--text-only"]))
    assert mistral["tokenizer_mode"] == mistral["config_format"] == mistral["load_format"]
    assert mistral["tokenizer_mode"] == "mistral"
    assert mistral["limit_mm_per_prompt"] == {"image": 0}
