import copy
import json

import pandas as pd

from registre.schema import SPEC_DIR
from registre.teacher.filters import run_funnel

GOLDEN = json.loads((SPEC_DIR / "golden_example.json").read_text())["output"]


def _item(i, output=GOLDEN, finish="stop", split="train", addresses=True, **qe):
    row = {
        "id": i, "text": json.dumps(output), "finish_reason": finish, "split": split,
        "addresses_someone": addresses, "source": "¿Me puedes mandar el archivo?",
    }  # fmt: skip
    row.update({f"qe_{k}": v for k, v in qe.items()})
    return row


def _same_text_everywhere(text):
    out = copy.deepcopy(GOLDEN)
    for v in out["variants"].values():
        v["text"], v["annotations"] = text, []
    return out


def _bad_span():
    out = copy.deepcopy(GOLDEN)
    out["variants"]["amical"]["annotations"][0]["span"] = "absent"
    return out


def _soutenu_with_tu():
    out = copy.deepcopy(GOLDEN)
    out["variants"]["soutenu"] = {"text": "Tu peux m'envoyer le fichier ?", "annotations": []}
    return out


def test_each_item_fails_at_the_right_stage():
    df = pd.DataFrame([
        _item("good"),
        _item("cut", finish="length"),
        _item("span", output=_bad_span()),
        _item("rules", output=_soutenu_with_tu()),
        _item("same", output=_same_text_everywhere("Il pleut depuis ce matin.")),
        _item("same_ok", output=_same_text_everywhere("Il pleut depuis ce matin."),
              addresses=False),
    ])  # fmt: skip
    out, report = run_funnel(df, lt=None)
    stage = {
        i: (None if pd.isna(s) else s) for i, s in zip(out["id"], out["failed_stage"], strict=True)
    }
    assert stage == {"good": None, "cut": "finished", "span": "schema", "rules": "rules",
                     "same": "collapse", "same_ok": None}  # fmt: skip
    assert report["train"]["generated"] == 6 and report["train"]["after_qe"] == 2
    assert set(report["skipped_stages"]) == {"grammar", "qe"}
    assert report["rule_violations"] == {"soutenu:address": 1, "soutenu:question_form": 1}


def test_qe_threshold_is_per_register():
    scores = {"soutenu": 0.8, "courant": 0.8, "amical": 0.8}
    df = pd.DataFrame([_item(f"i{k}", familier=0.6 + k / 100, **scores) for k in range(20)])
    out, report = run_funnel(df, lt=None, qe_percentile=0.10)
    # familier scores are lower overall, but only its own bottom 10% is dropped
    assert out["failed_stage"].eq("qe").sum() == 2
    assert round(report["qe_thresholds"]["familier"], 3) == 0.619


def test_test_items_are_never_dropped():
    df = pd.DataFrame(
        [_item("t", finish="length", split="test"), _item("v", finish="length", split="val")]
    )
    out, _ = run_funnel(df, lt=None)
    assert dict(zip(out["id"], out["keep"], strict=True)) == {"t": True, "v": False}
