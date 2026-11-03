"""Day-2 human review (checkpoint 1).

  ./run review-sheet   → review/sheet.csv (for the reviewer) + review/key.csv (for you)
  ./run review-score   → reads the filled review/sheet.csv, writes review/results.json

The sheet is *blind*: it shows the Spanish, one French variant and its annotations, but not
whether the item passed the filters or what the automatic checks said. A reviewer who sees
"rules: OK" judges differently. The key file links each row back to that information.

Each row is one (item, register) pair: 20 accepted training items per register (the data you
will train on) plus up to 5 per register that rules.py rejected (to measure whether its flags
are right). Rows are shuffled so accepted and rejected ones are mixed.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import pandas as pd

from registre.data.splits import stable_hash
from registre.schema import load_spec, register_ids

SHEET = Path("review/sheet.csv")
KEY = Path("review/key.csv")
RESULTS = Path("review/results.json")
YES = {"y", "yes", "o", "oui", "1", "x", "true"}
NO = {"n", "no", "non", "0", "false"}


def _variant(output_json: str, rid: str) -> dict:
    return json.loads(output_json)["variants"][rid]


def _format_annotations(annotations: list[dict]) -> str:
    return "\n".join(
        f"{k}. «{a['span']}» [{a['tag']}] {a['note']}" for k, a in enumerate(annotations, 1)
    )


def make_sheet(filtered: pd.DataFrame, n_accepted: int = 20, n_rejected: int = 5) -> tuple:
    train = filtered[filtered["split"].eq("train")]
    use_when = {r["id"]: r["use_when"] for r in load_spec("registers")["registers"]}
    rows = []
    for rid in register_ids():
        order = train["id"].map(lambda i, r=rid: stable_hash(f"{i}:{r}"))
        ranked = train.assign(_h=order).sort_values("_h")
        accepted = ranked[ranked["failed_stage"].isna()].head(n_accepted)
        flagged = ranked["violations"].fillna("{}").map(lambda v, r=rid: r in json.loads(v))
        rejected = ranked[ranked["failed_stage"].eq("rules") & flagged].head(n_rejected)
        for group, part in (("accepted", accepted), ("rejected_rules", rejected)):
            for r in part.itertuples():
                v = _variant(r.output, rid)
                qe = getattr(r, f"qe_{rid}", None)
                viol = json.loads(r.violations or "{}").get(rid, [])
                rows.append(
                    {
                        "id": r.id,
                        "register": rid,
                        "group": group,
                        "rules_violations": ",".join(viol),
                        "qe": None if qe is None or pd.isna(qe) else float(qe),
                        "tags": ",".join(a["tag"] for a in v["annotations"]),
                        "n_annotations": len(v["annotations"]),
                        "source_es": r.source,
                        "register_use": use_when[rid],
                        "french": v["text"],
                        "annotations": _format_annotations(v["annotations"]),
                    }
                )
    rows.sort(key=lambda x: stable_hash(f"{x['id']}:{x['register']}:shuffle"))
    df = pd.DataFrame(rows)
    df.insert(0, "review_id", range(1, len(df) + 1))
    sheet = df[["review_id", "source_es", "register", "register_use", "french", "annotations"]]
    sheet = sheet.assign(translation_ok="", register_ok="", wrong_annotations="", comment="")
    key = df[["review_id", "id", "register", "group", "rules_violations", "qe", "tags",
              "n_annotations"]]  # fmt: skip
    return sheet, key


# ------------------------------------------------------------------ scoring


def _yes_no(value) -> bool | None:
    v = str(value).strip().lower()
    return True if v in YES else False if v in NO else None


def _numbers(value) -> list[int]:
    """'1, 3' → [1, 3]. Robust to spreadsheets turning '2' into 2.0."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    out = []
    for token in re.split(r"[,; ]+", str(value).strip()):
        try:
            out.append(int(float(token)))
        except ValueError:
            continue
    return out


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% confidence interval for a proportion k/n. With n=20, expect it to be wide."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (round(max(0.0, centre - half), 3), round(min(1.0, centre + half), 3))


def _rate(k: int, n: int) -> dict:
    return {"errors": k, "n": n, "rate": round(k / n, 3) if n else None, "ci95": wilson(k, n)}


def auc(scores: list[float], positives: list[bool]) -> float | None:
    """Probability that a random good translation scores higher than a random bad one."""
    pos = [s for s, p in zip(scores, positives, strict=True) if p]
    neg = [s for s, p in zip(scores, positives, strict=True) if not p]
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return round(wins / (len(pos) * len(neg)), 3)


def score(sheet: pd.DataFrame, key: pd.DataFrame, limits: dict[str, float]) -> dict:
    sheet = sheet.assign(review_id=sheet["review_id"].astype(int))
    df = key.merge(sheet[["review_id", "translation_ok", "register_ok", "wrong_annotations"]],
                   on="review_id")  # fmt: skip
    df["t_ok"] = df["translation_ok"].map(_yes_no)
    df["r_ok"] = df["register_ok"].map(_yes_no)
    df["n_wrong"] = df["wrong_annotations"].map(lambda v: len(_numbers(v)))
    unanswered = int(df["t_ok"].isna().sum() + df["r_ok"].isna().sum())
    df = df.dropna(subset=["t_ok", "r_ok"])

    acc = df[df["group"].eq("accepted")]
    results: dict = {"unanswered_cells": unanswered, "teacher": {}, "limits": limits}
    for rid, part in [("all", acc), *acc.groupby("register")]:
        results["teacher"][rid] = {
            "translation": _rate(int((~part["t_ok"].astype(bool)).sum()), len(part)),
            "register": _rate(int((~part["r_ok"].astype(bool)).sum()), len(part)),
            "annotations": _rate(int(part["n_wrong"].sum()), int(part["n_annotations"].sum())),
        }

    # Is rules.py right? Flagged = it found a violation; bad = the reviewer said register not ok.
    flagged = df["rules_violations"].fillna("").ne("")
    bad = ~df["r_ok"].astype(bool)
    tp, fp, fn = (
        int((flagged & bad).sum()),
        int((flagged & ~bad).sum()),
        int((~flagged & bad).sum()),
    )
    results["rules_checker"] = {
        "precision": round(tp / (tp + fp), 3) if tp + fp else None,
        "recall": round(tp / (tp + fn), 3) if tp + fn else None,
        "tp": tp, "fp": fp, "fn": fn,
    }  # fmt: skip

    # Does CometKiwi separate good from bad translations?
    with_qe = df.dropna(subset=["qe"])
    results["qe_auc"] = auc(with_qe["qe"].tolist(), with_qe["t_ok"].astype(bool).tolist())

    # Which tags does the teacher misapply? (only tags seen at least 5 times)
    per_tag: dict[str, list[int]] = {}
    for r in acc.itertuples():
        wrong = set(_numbers(r.wrong_annotations))
        for k, tag in enumerate(str(r.tags).split(","), 1):
            if tag:
                per_tag.setdefault(tag, [0, 0])
                per_tag[tag][0] += k in wrong
                per_tag[tag][1] += 1
    results["tag_error_rates"] = {
        t: round(w / n, 2) for t, (w, n) in sorted(per_tag.items()) if n >= 5 and w
    }

    # Decide on the pooled rates (n≈80): with 20 rows per register, one or two errors move a
    # register's rate by 5-10 points, so per-register overruns are warnings to inspect.
    over = [
        (rid, kind, m[kind]["rate"])
        for rid, m in results["teacher"].items()
        for kind in ("translation", "register", "annotations")
        if m[kind]["rate"] is not None and m[kind]["rate"] > limits[kind]
    ]
    results["problems"] = [
        f"all: {k} error rate {r} > {limits[k]}" for g, k, r in over if g == "all"
    ]
    results["warnings"] = [
        f"{g}: {k} error rate {r} > {limits[k]}" for g, k, r in over if g != "all"
    ]
    results["decision"] = "FIX" if results["problems"] else "GO"
    return results


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("command", choices=["sheet", "score"])
    p.add_argument("--filtered", type=Path, default=Path("data/teacher/filtered.parquet"))
    p.add_argument("--max-translation", type=float, default=0.10)
    p.add_argument("--max-register", type=float, default=0.10)
    p.add_argument("--max-annotations", type=float, default=0.15)
    args = p.parse_args()

    if args.command == "sheet":
        sheet, key = make_sheet(pd.read_parquet(args.filtered))
        SHEET.parent.mkdir(parents=True, exist_ok=True)
        sheet.to_csv(SHEET, index=False, encoding="utf-8-sig")  # BOM: Excel/LibreOffice read UTF-8
        key.to_csv(KEY, index=False)
        print(f"wrote {SHEET} ({len(sheet)} rows) for the reviewer and {KEY} for you")
        return

    limits = {"translation": args.max_translation, "register": args.max_register,
              "annotations": args.max_annotations}  # fmt: skip
    sheet = pd.read_csv(SHEET, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    results = score(sheet, pd.read_csv(KEY), limits)
    RESULTS.write_text(json.dumps(results, indent=2))
    for rid, m in results["teacher"].items():
        line = "  ".join(f"{k} {v['rate']} {v['ci95']}" for k, v in m.items())
        print(f"{rid:<9} {line}")
    print("rules.py:", results["rules_checker"])
    print("CometKiwi AUC:", results["qe_auc"])
    print("tags most often wrong:", results["tag_error_rates"] or "none")
    print(f"\nCHECKPOINT 1: {results['decision']}")
    for problem in results["problems"]:
        print("  - problem:", problem)
    for warning in results["warnings"]:
        print("  - inspect:", warning)


if __name__ == "__main__":
    main()
