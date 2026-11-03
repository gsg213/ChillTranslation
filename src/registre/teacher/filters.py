"""Day-2 filtering funnel: which teacher outputs are good enough to train on.

Usage: ./run filter   (or ./run filter ARGS="--raw data/teacher/pilot.jsonl" for a pilot)

Each item goes through the stages in order and is labelled with the first stage it fails:

  finished   the model stopped by itself (not cut off at max_tokens)
  schema     valid JSON, valid tags, every span copied from its variant (Pydantic)
  rules      every variant obeys its register profile (rules.py)
  grammar    no spelling/grammar errors in soutenu and courant (LanguageTool; skipped if down)
  collapse   soutenu and familier really differ when the sentence addresses someone (chrF)
  qe         every variant scores above its register's quality threshold (CometKiwi; skipped
             if data/teacher/qe.jsonl doesn't exist)

Train/val items that fail any stage are dropped. Test items are never dropped (that would
cherry-pick the teacher's best work for the comparison); their flags are kept for analysis.
"""

from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from pydantic import ValidationError
from sacrebleu.metrics import CHRF

from registre.rules import check
from registre.schema import TranslationOutput, register_ids

STAGES = ["finished", "schema", "rules", "grammar", "collapse", "qe"]
GRAMMAR_REGISTERS = ["soutenu", "courant"]  # LanguageTool would flag intended colloquialisms
GRAMMAR_ISSUE_TYPES = {"misspelling", "grammar"}  # ignore style and typography suggestions
_CHRF = CHRF()


# ------------------------------------------------------------------ loading


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def load(raw: Path, inputs: Path, qe: Path | None) -> pd.DataFrame:
    """One row per item: the model output, its input metadata, and one QE column per register."""
    out = pd.DataFrame(read_jsonl(raw)).drop_duplicates("id", keep="last")
    meta = pd.DataFrame(read_jsonl(inputs))[
        ["id", "source", "speaker_gender", "split", "origin", "file", "addresses_someone"]
    ]
    df = out.merge(meta, on="id", how="left", validate="one_to_one")
    if qe is not None and qe.exists():
        scores = pd.DataFrame(read_jsonl(qe)).pivot(index="id", columns="register", values="qe")
        df = df.merge(scores.add_prefix("qe_"), left_on="id", right_index=True, how="left")
    return df


# ------------------------------------------------------------------ stages


def parse_output(text: str) -> TranslationOutput | None:
    try:
        return TranslationOutput.model_validate_json(text)
    except ValidationError:
        return None


def rule_violations(out: TranslationOutput) -> dict[str, list[str]]:
    plural = out.source_address == "plural"
    found = {}
    for rid in register_ids():
        v = check(getattr(out.variants, rid).text, rid, plural_addressee=plural)
        if v:
            found[rid] = sorted(v)
    return found


class LanguageToolClient:
    """Minimal client for the LanguageTool HTTP API (the `languagetool` compose service)."""

    def __init__(self, url: str = "http://languagetool:8010"):
        self.url = url.rstrip("/")

    def available(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.url}/v2/languages", timeout=3):
                return True
        except OSError:
            return False

    def errors(self, text: str) -> list[str]:
        data = urllib.parse.urlencode({"language": "fr", "text": text}).encode()
        with urllib.request.urlopen(f"{self.url}/v2/check", data=data, timeout=30) as r:
            matches = json.load(r)["matches"]
        return [
            m["rule"]["id"] for m in matches if m["rule"].get("issueType") in GRAMMAR_ISSUE_TYPES
        ]


def chrf(a: str, b: str) -> float:
    return _CHRF.sentence_score(a, [b]).score


def qe_thresholds(df: pd.DataFrame, percentile: float, overrides: dict[str, float]) -> dict:
    """Per register: an explicit override, or the given percentile of the train/val scores.

    Per register because QE models are trained on standard written text: they score correct
    texting French ("J'pense pas, stp") lower than correct formal French. One global threshold
    would throw away the familier variants first.
    """
    pool = df[df["split"].isin(["train", "val"])]
    out = {}
    for rid in register_ids():
        col = f"qe_{rid}"
        if rid in overrides:
            out[rid] = overrides[rid]
        elif col in pool and pool[col].notna().any():
            out[rid] = float(pool[col].quantile(percentile))
    return out


# ------------------------------------------------------------------ funnel


def run_funnel(
    df: pd.DataFrame,
    lt: LanguageToolClient | None,
    collapse_threshold: float = 90.0,
    qe_percentile: float = 0.02,
    qe_overrides: dict[str, float] | None = None,
    workers: int = 8,
) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    failed: dict[str, str | None] = dict.fromkeys(df["id"])
    skipped: list[str] = []

    def fail(mask: pd.Series, stage: str) -> None:
        for i in df.loc[mask & df["id"].map(lambda x: failed[x] is None), "id"]:
            failed[i] = stage

    # finished
    fail(df["finish_reason"].ne("stop"), "finished")

    # schema
    parsed = df["text"].map(parse_output)
    df["output"] = parsed.map(lambda o: o.model_dump_json() if o is not None else None)
    fail(parsed.isna(), "schema")

    # rules
    df["violations"] = parsed.map(lambda o: json.dumps(rule_violations(o)) if o else None)
    fail(df["violations"].map(lambda v: v not in (None, "{}")), "rules")

    # grammar
    if lt is not None and lt.available():
        jobs = [
            (i, rid, getattr(o.variants, rid).text)
            for i, o in zip(df["id"], parsed, strict=True)
            if o is not None and failed[i] is None
            for rid in GRAMMAR_REGISTERS
        ]
        with ThreadPoolExecutor(workers) as pool:
            results = list(pool.map(lambda j: lt.errors(j[2]), jobs))
        errors: dict[str, list[str]] = {}
        for (i, rid, _), errs in zip(jobs, results, strict=True):
            errors.setdefault(i, []).extend(f"{rid}:{e}" for e in errs)
        df["grammar_errors"] = df["id"].map(lambda i: json.dumps(errors.get(i, [])))
        fail(df["id"].map(lambda i: bool(errors.get(i))), "grammar")
    else:
        skipped.append("grammar")
        print("  LanguageTool not reachable (./run lt-up): grammar stage skipped")

    # collapse
    df["chrf_soutenu_familier"] = [
        chrf(o.variants.soutenu.text, o.variants.familier.text) if o is not None else None
        for o in parsed
    ]
    collapsed = df["addresses_someone"].fillna(False).astype(bool) & (
        df["chrf_soutenu_familier"].fillna(0) >= collapse_threshold
    )
    fail(collapsed, "collapse")

    # qe
    thresholds: dict[str, float] = {}
    qe_cols = [f"qe_{rid}" for rid in register_ids() if f"qe_{rid}" in df]
    if qe_cols:
        thresholds = qe_thresholds(df, qe_percentile, qe_overrides or {})
        low = pd.Series(False, index=df.index)
        for rid, t in thresholds.items():
            low |= df[f"qe_{rid}"].fillna(-1) < t
        fail(low, "qe")
    else:
        skipped.append("qe")
        print("  no data/teacher/qe.jsonl (./run qe): qe stage skipped")

    df["failed_stage"] = df["id"].map(failed)
    df["keep"] = df["failed_stage"].isna() | df["split"].eq("test")  # tests: never dropped
    report = funnel_report(df, skipped, thresholds, collapse_threshold)
    return df, report


def funnel_report(df: pd.DataFrame, skipped: list[str], thresholds: dict, collapse: float) -> dict:
    report: dict = {"skipped_stages": skipped, "qe_thresholds": thresholds, "collapse": collapse}
    for split, part in df.groupby("split"):
        alive = len(part)
        steps = {"generated": alive}
        for stage in STAGES:
            alive -= int(part["failed_stage"].eq(stage).sum())
            steps[f"after_{stage}"] = alive
        report[split] = steps
    viol = Counter()
    for v in df["violations"].dropna():
        for rid, codes in json.loads(v).items():
            viol.update(f"{rid}:{c}" for c in codes)
    report["rule_violations"] = dict(viol.most_common())
    if "grammar_errors" in df:
        gram = Counter(e for v in df["grammar_errors"].dropna() for e in json.loads(v))
        report["top_grammar_rules"] = dict(gram.most_common(15))
    return report


def print_report(report: dict) -> None:
    splits = [s for s in ("train", "val", "test") if s in report]
    print(f"\n{'stage':<16}" + "".join(f"{s:>10}" for s in splits))
    for key in ["generated", *[f"after_{s}" for s in STAGES]]:
        print(f"{key:<16}" + "".join(f"{report[s][key]:>10,}" for s in splits))
    if report["skipped_stages"]:
        print(f"(skipped: {', '.join(report['skipped_stages'])})")
    print("\nrule violations (register:code):", report["rule_violations"] or "none")
    if report.get("top_grammar_rules"):
        print("top LanguageTool rules:", report["top_grammar_rules"])
    if report["qe_thresholds"]:
        print("QE thresholds:", {k: round(v, 3) for k, v in report["qe_thresholds"].items()})


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--raw", type=Path, default=Path("data/teacher/raw.jsonl"))
    p.add_argument("--inputs", type=Path, default=Path("data/teacher/inputs.jsonl"))
    p.add_argument("--qe", type=Path, default=Path("data/teacher/qe.jsonl"))
    p.add_argument("--out-dir", type=Path, default=Path("data/teacher"))
    p.add_argument("--lt-url", default="http://languagetool:8010")
    p.add_argument("--collapse-threshold", type=float, default=90.0)
    p.add_argument("--qe-percentile", type=float, default=0.02)
    p.add_argument(
        "--qe-threshold",
        action="append",
        default=[],
        metavar="REGISTER=VALUE",
        help="fixed QE threshold for a register, e.g. familier=0.62 (repeatable)",
    )
    args = p.parse_args()

    overrides = {k: float(v) for k, v in (t.split("=") for t in args.qe_threshold)}
    df = load(args.raw, args.inputs, args.qe)
    df, report = run_funnel(
        df,
        LanguageToolClient(args.lt_url),
        collapse_threshold=args.collapse_threshold,
        qe_percentile=args.qe_percentile,
        qe_overrides=overrides,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = "filtered" if args.raw.stem == "raw" else f"{args.raw.stem}_filtered"
    df.drop(columns=["text"]).to_parquet(args.out_dir / f"{stem}.parquet", index=False)
    (args.out_dir / f"{stem}_funnel.json").write_text(json.dumps(report, indent=2))
    print_report(report)
    print(f"\nwrote {args.out_dir / f'{stem}.parquet'} and {stem}_funnel.json")


if __name__ == "__main__":
    main()
