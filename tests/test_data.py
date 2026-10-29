"""Tests for the dataset step (day 1, step 6).

The helper tests pass immediately. The tests for clean / cocoa_rows / assign_splits /
sample_train / remove_near_duplicates pass once you implement those functions.
Run: ./run shell → uv run pytest tests/test_data.py
"""

from pathlib import Path

import pandas as pd
import pytest

from registre.data import cocoa, splits, tatoeba
from registre.data.splits import bucket, normalize, stable_hash

COCOA_ROOT = Path("data/raw/contrastive-controlled-mt-main/CoCoA-MT")

# ---------------------------------------------------------- given helpers


def test_normalize_keeps_accents():
    assert normalize("¿Tú vienes?") != normalize("¿Tu vienes?")  # "tú" (you) ≠ "tu" (your)


def test_normalize_ignores_case_and_punctuation():
    assert normalize("¡Hola, Juan!") == normalize("hola   juan")


def test_hash_and_bucket_are_deterministic():
    assert stable_hash("abc") == stable_hash("abc")
    assert 0 <= bucket("abc") < 100


# ------------------------------------------------------- 6.2 tatoeba.clean


def _tat(sentences):
    return pd.DataFrame({"tatoeba_id": range(len(sentences)), "source": sentences, "fr_ref": pd.NA})


def test_clean_length_filter():
    out = tatoeba.clean(_tat(["Hola amigo", "Me gusta mucho el café.", " ".join(["palabra"] * 26)]))
    assert out["source"].tolist() == ["Me gusta mucho el café."]


def test_clean_dedup_on_normalized_form():
    out = tatoeba.clean(
        _tat(["¿Vienes a la fiesta?", "vienes a la fiesta", "¿Vienes tú a la fiesta?"])
    )
    assert len(out) == 2  # first two normalize to the same string; the third differs


def test_clean_caps_tom_and_mary():
    plain = [f"Esta es la frase número {i}." for i in range(90)]
    toms = [f"Tom compró {i} manzanas ayer." for i in range(60)]
    out = tatoeba.clean(_tat(plain + toms), max_name_share=0.10)
    share = out["source"].str.contains(tatoeba.NAMES).mean()
    assert 0 < share <= 0.10 + 1e-9


# --------------------------------------------------- 6.4 cocoa.cocoa_rows


@pytest.mark.skipif(not COCOA_ROOT.exists(), reason="run ./run download first")
def test_cocoa_rows_shape():
    rows = cocoa.cocoa_rows(COCOA_ROOT)
    counts = rows["split_hint"].value_counts().to_dict()
    assert counts == {"test": 2 * 489, "train": 2 * 306}
    assert set(rows["cocoa_formality"]) == {"formal", "informal"}
    assert rows.groupby("group_id").size().eq(2).all()  # each item: one formal + one informal
    assert rows["id"].is_unique


# ------------------------------------------------- 6.6 splits / sampling


def _rows():
    tat = pd.DataFrame(
        {
            "id": [f"t{i}" for i in range(2000)],
            "source": [f"frase {i}" for i in range(2000)],
            "origin": "tatoeba",
            "group_id": [f"tat-{i}" for i in range(2000)],
            "split_hint": pd.NA,
            "addresses_someone": [i % 4 == 0 for i in range(2000)],
        }
    )
    coc = pd.DataFrame(
        {
            "id": [f"c{i}-{f}" for i in range(100) for f in ("formal", "informal")],
            "source": [f"fuente {i} {f}" for i in range(100) for f in ("formal", "informal")],
            "origin": "cocoa",
            "group_id": [f"cocoa-{i}" for i in range(100) for _ in range(2)],
            "split_hint": ["train" if i < 60 else "test" for i in range(100) for _ in range(2)],
            "addresses_someone": True,
        }
    )
    own = pd.DataFrame(
        {
            "id": ["own-1"],
            "source": ["¿Me ayudas?"],
            "origin": "own",
            "group_id": ["own-own-1"],
            "split_hint": "test",
            "addresses_someone": True,
        }
    )
    return pd.concat([tat, coc, own], ignore_index=True)


def test_split_groups_stay_together():
    out = splits.assign_splits(_rows())
    assert out.groupby("group_id")["split"].nunique().eq(1).all()


def test_split_fixed_destinations():
    out = splits.assign_splits(_rows())
    assert out.loc[out["origin"].eq("own"), "split"].eq("test").all()
    cocoa_test = out["origin"].eq("cocoa") & out["split_hint"].eq("test")
    assert out.loc[cocoa_test, "split"].eq("test").all()
    assert (
        not out.loc[out["origin"].eq("cocoa") & out["split_hint"].eq("train"), "split"]
        .eq("test")
        .any()
    )


def test_split_tatoeba_sizes():
    out = splits.assign_splits(_rows())
    tat = out[out["origin"].eq("tatoeba")]["split"].value_counts()
    assert tat["test"] <= splits.TATOEBA_TEST and tat["val"] <= splits.TATOEBA_VAL
    assert tat["train"] > 0


def test_split_is_deterministic():
    a, b = splits.assign_splits(_rows()), splits.assign_splits(_rows())
    assert a["split"].tolist() == b["split"].tolist()


def test_sample_train_size_and_stratification():
    out = splits.sample_train(splits.assign_splits(_rows()), n_tatoeba=500, min_address_share=0.5)
    tat_train = out[out["origin"].eq("tatoeba") & out["split"].eq("train")]
    assert len(tat_train) == 500
    assert tat_train["addresses_someone"].mean() >= 0.5 - 1e-9
    assert out["origin"].eq("cocoa").sum() == 200  # other origins untouched


# ------------------------------------------------------ 6.7 leakage check


def test_near_duplicates_removed_from_train_only():
    df = pd.DataFrame(
        {
            "id": ["test-1", "train-dup", "train-ok"],
            "source": [
                "¿Puedes venir mañana por la tarde a la reunión del equipo en la oficina?",
                "¿Puedes venir mañana por la tarde a la reunión del equipo en la oficina central?",
                "El tren sale a las ocho de la mañana.",
            ],
            "split": ["test", "train", "train"],
        }
    )
    out = splits.remove_near_duplicates(df, threshold=0.8)
    assert out["id"].tolist() == ["test-1", "train-ok"]
