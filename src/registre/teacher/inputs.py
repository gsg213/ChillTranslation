"""Build data/teacher/inputs.jsonl from the processed splits.   Usage: ./run teacher-inputs

Every split gets teacher outputs: train/val become training data (after filtering); the test
sets get them too, as the "teacher ceiling" you compare the student against on day 4.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from registre.data.splits import stable_hash

PROCESSED = Path("data/processed")
OUT = Path("data/teacher/inputs.jsonl")
FILES = ["train", "val", "test_tatoeba", "test_cocoa", "test_own"]
COLUMNS = ["id", "source", "split", "origin", "addresses_someone"]


def speaker_gender(row_id: str, origin: str) -> str:
    """Your own sentences use your gender (SPEAKER_GENDER in .env). Everything else is split
    50/50 by hash, so the model sees both values and learns to use the field."""
    if origin == "own" and os.environ.get("SPEAKER_GENDER") in ("m", "f"):
        return os.environ["SPEAKER_GENDER"]
    return "f" if stable_hash(row_id) % 2 else "m"


def main() -> None:
    frames = []
    for name in FILES:
        path = PROCESSED / f"{name}.parquet"
        if path.exists():
            frames.append(pd.read_parquet(path, columns=COLUMNS).assign(file=name))
    if not frames:
        raise SystemExit("data/processed is empty: run ./run data first")
    df = pd.concat(frames, ignore_index=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for r in df.itertuples(index=False):
            row = {
                "id": r.id,
                "source": r.source,
                "speaker_gender": speaker_gender(r.id, r.origin),
                "split": r.split,
                "origin": r.origin,
                "file": r.file,
                "addresses_someone": bool(r.addresses_someone),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(df["file"].value_counts().to_string())
    print(f"wrote {len(df):,} inputs to {OUT}")


if __name__ == "__main__":
    main()
