"""CometKiwi quality estimation for every teacher variant.   Usage: ./run qe

Runs in its own container (Dockerfile.qe): unbabel-comet pins old versions of numpy,
transformers and huggingface-hub that would break the dev environment and vLLM.

CometKiwi is *reference-free*: it scores a (Spanish source, French translation) pair directly,
roughly between 0 and 1, so it can check meaning preservation when no human reference exists.
The model is gated on Hugging Face: accept its license with the account of your HF_TOKEN.
"""

import json
import os
from pathlib import Path

RAW = Path(os.environ.get("QE_RAW", "data/teacher/raw.jsonl"))
INPUTS = Path("data/teacher/inputs.jsonl")
OUT = Path(os.environ.get("QE_OUT", "data/teacher/qe.jsonl"))
MODEL = os.environ.get("QE_MODEL", "Unbabel/wmt22-cometkiwi-da")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    import torch
    from comet import download_model, load_from_checkpoint

    sources = {r["id"]: r["source"] for r in read_jsonl(INPUTS)}
    pairs = []  # (id, register, source, translation)
    for r in read_jsonl(RAW):
        try:
            variants = json.loads(r["text"])["variants"]
        except (json.JSONDecodeError, KeyError, TypeError):
            continue  # unparseable output: the schema stage of the filter rejects it anyway
        for register, v in variants.items():
            if isinstance(v, dict) and v.get("text") and r["id"] in sources:
                pairs.append((r["id"], register, sources[r["id"]], v["text"]))
    print(f"scoring {len(pairs):,} (source, variant) pairs with {MODEL}")

    model = load_from_checkpoint(download_model(MODEL))
    gpus = 1 if torch.cuda.is_available() else 0
    print("device:", torch.cuda.get_device_name(0) if gpus else "CPU (slower but fine)")
    data = [{"src": src, "mt": mt} for _, _, src, mt in pairs]
    scores = model.predict(data, batch_size=64, gpus=gpus, progress_bar=True).scores

    with OUT.open("w", encoding="utf-8") as f:
        for (row_id, register, _, _), score in zip(pairs, scores, strict=True):
            f.write(json.dumps({"id": row_id, "register": register, "qe": round(float(score), 4)}))
            f.write("\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
