import json

import pandas as pd

from registre.schema import SPEC_DIR
from registre.teacher.review import _numbers, auc, make_sheet, score, wilson

GOLDEN = json.loads((SPEC_DIR / "golden_example.json").read_text())["output"]


def _filtered(n=30):
    rows = []
    for i in range(n):
        rejected = i % 6 == 0
        rows.append({
            "id": f"x{i}", "split": "train", "source": "¿Me puedes mandar el archivo?",
            "output": json.dumps(GOLDEN),
            "failed_stage": "rules" if rejected else None,
            "violations": json.dumps({"soutenu": ["address"]}) if rejected else "{}",
            "qe_soutenu": 0.5 + i / 100,
        })  # fmt: skip
    return pd.DataFrame(rows)


def test_sheet_is_blind_and_mixed():
    sheet, key = make_sheet(_filtered(), n_accepted=4, n_rejected=2)
    assert "group" not in sheet and "qe" not in sheet and "rules_violations" not in sheet
    assert set(key["group"]) == {"accepted", "rejected_rules"}
    assert len(key[key["register"].eq("soutenu")]) == 6  # 4 accepted + 2 rejected
    assert sheet["review_id"].tolist() == key["review_id"].tolist()


def test_score_rates_rules_and_decision():
    sheet, key = make_sheet(_filtered(), n_accepted=4, n_rejected=2)
    rejected = key["group"].eq("rejected_rules")
    sheet["translation_ok"] = "y"
    sheet["register_ok"] = ["n" if r else "y" for r in rejected]  # reviewer agrees with rules.py
    sheet["wrong_annotations"] = ""
    sheet.loc[0, "wrong_annotations"] = "1, 2"
    res = score(sheet, key, {"translation": 0.1, "register": 0.1, "annotations": 0.15})
    assert res["rules_checker"]["precision"] == 1.0 and res["rules_checker"]["recall"] == 1.0
    assert res["teacher"]["all"]["translation"]["rate"] == 0.0
    assert res["decision"] == "GO"


def test_wilson_and_auc():
    low, high = wilson(2, 20)
    assert low < 0.1 < high < 0.35  # 2 errors in 20 is compatible with a true rate near 30%
    assert auc([0.9, 0.8, 0.2], [True, True, False]) == 1.0


def test_wrong_annotation_numbers_survive_spreadsheets():
    assert _numbers("1, 3") == [1, 3]
    assert _numbers(2.0) == [2]  # a spreadsheet saved "2" as a number
    assert _numbers("") == [] and _numbers(float("nan")) == []
