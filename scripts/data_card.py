"""Rewrite the generated block of DATA_CARD.md from data/processed/{stats,manifest}.json.
 
Run with ./run card after ./run data. Only the text between the GENERATED markers changes,
so the hand-written sections are never touched and the numbers can't drift from the files.
"""
 
import datetime
import json
import subprocess
import sys
from pathlib import Path
 
CARD = Path("DATA_CARD.md")
OUT = Path("data/processed")
BEGIN = "<!-- BEGIN GENERATED: ./run card -->"
END = "<!-- END GENERATED -->"
 
 
def git_commit() -> str:
    try:
        cmd = ["git", "rev-parse", "--short", "HEAD"]
        return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
 
 
def render(stats: list[dict], manifest: dict) -> str:
    lines = [
        "### Rows and features by split and origin",
        "",
        "| split | origin | rows | addresses someone | questions | median words |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in stats:
        lines.append(
            f"| {r['split']} | {r['origin']} | {r['rows']:,} | {r['addresses_someone']:.0%} "
            f"| {r['is_question']:.0%} | {r['median_words']:g} |"
        )
    lines += ["", "### Files (`data/processed/`)", "", "| file | rows | sha256 |", "|---|---:|---|"]
    for name, info in manifest.items():
        lines.append(f"| {name}.parquet | {info['rows']:,} | `{info['sha256'][:16]}…` |")
    today = datetime.date.today().isoformat()
    lines += ["", f"Generated on {today} from commit `{git_commit()}`."]
    return "\n".join(lines)
 
 
def main() -> None:
    try:
        stats = json.loads((OUT / "stats.json").read_text())
        manifest = json.loads((OUT / "manifest.json").read_text())
    except FileNotFoundError:
        sys.exit("data/processed has no stats.json / manifest.json: run ./run data first")
 
    card = CARD.read_text(encoding="utf-8")
    if BEGIN not in card or END not in card:
        sys.exit(f"{CARD} is missing the GENERATED markers")
    head, rest = card.split(BEGIN, 1)
    _, tail = rest.split(END, 1)
    CARD.write_text(f"{head}{BEGIN}\n{render(stats, manifest)}\n{END}{tail}", encoding="utf-8")
    print(f"Updated the generated section of {CARD}")
 
 
if __name__ == "__main__":
    main()
 