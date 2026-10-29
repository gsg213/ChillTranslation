"""Contract for src/registre/rules.py — write the implementation to make these pass.

Expected API:

    features(text: str) -> Features   # a dataclass with at least:
        address: str                  # "tu" | "vous" | "mixed" | "none"
        ne_kept: int                  # clauses with a negation that keeps ne / n'
        ne_dropped: int               # clauses with pas/jamais/rien/aucun and no ne
        question_forms: list[str]     # one per question sentence, see spec/registers.yaml
        oral_contractions: list[str]  # t'es, t'as, chuis, bare "y a", j' before a consonant…
        contraction_slots: list[str]  # "tu es", "tu as", "il y a", "il n'y a" written in full
        sms: list[str]                # stp, svp, bcp, jsp, mdr…
        colloquial: list[str]         # boulot, bouquin, sympa, ouais…

    check(text: str, register_id: str, plural_addressee: bool = False) -> set[str]
        # violation codes: "address", "ne_drop", "question_form",
        #                  "oral_contractions", "sms", "colloquial"

Rules are heuristics: these tests pin down the behaviour that matters and the
traps that break naive regexes. Add a test for every bug you find.
"""

import json
from pathlib import Path

import pytest

from registre.rules import check, features
from registre.schema import load_spec, register_ids

# ---------------------------------------------------------------- address


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Tu peux m'aider ?", "tu"),
        ("Je t'aime.", "tu"),  # t' = te
        ("Tes clés sont sur la table.", "tu"),
        ("Pourriez-vous m'aider ?", "vous"),
        ("Merci, s'il vous plaît.", "vous"),
        ("On se retrouve au rendez-vous.", "none"),  # trap: rendez-vous
        ("Il pleut depuis ce matin.", "none"),
        ("Tu peux venir svp ?", "mixed"),  # svp is the vous form
    ],
)
def test_address(text, expected):
    assert features(text).address == expected


# --------------------------------------------------------------- negation


@pytest.mark.parametrize(
    "text, kept, dropped",
    [
        ("Je ne sais pas.", 1, 0),
        ("Je sais pas.", 0, 1),
        ("Il n'y a pas de souci.", 1, 0),
        ("Y a pas de souci.", 0, 1),
        ("Je n'ai rien vu.", 1, 0),
        ("J'ai rien vu.", 0, 1),
        ("Cette personne est très gentille.", 0, 0),  # trap: personne as a noun
        ("Il est plus grand que moi.", 0, 0),  # trap: comparative plus
        ("J'ai pas mal de travail.", 0, 0),  # trap: "pas mal de" = a lot of
        ("Si jamais tu viens, appelle-moi.", 0, 0),  # trap: "si jamais" = if ever
        # Regressions found by running the checker on CoCoA's human references:
        ("Vous avez fait pas mal d'économies.", 0, 0),  # "pas mal d'" (elided)
        ("Pas de problème.", 0, 0),  # verbless: no ne to drop
        ("Pas d'autre question ?", 0, 0),
        ("Sûrement pas.", 0, 0),
        ("Je sais pas si ça marche ou pas.", 0, 1),  # "ou pas" ignored, "je sais pas" counts
    ],
)
def test_negation(text, kept, dropped):
    f = features(text)
    assert (f.ne_kept, f.ne_dropped) == (kept, dropped)


def test_negative_subject_is_not_dropped_ne():
    assert features("Personne n'est venu.").ne_dropped == 0
    assert features("Rien ne marche.").ne_dropped == 0


# -------------------------------------------------------------- questions


@pytest.mark.parametrize(
    "text, forms",
    [
        ("Pourriez-vous m'envoyer le fichier ?", ["inversion"]),
        ("A-t-il fini ?", ["inversion"]),
        ("Viens-tu ce soir ?", ["inversion"]),
        ("Est-ce que vous pouvez m'aider ?", ["est_ce_que"]),
        ("Qu'est-ce que tu veux ?", ["est_ce_que"]),
        ("Tu peux m'aider ?", ["intonation"]),
        ("On se retrouve au rendez-vous ?", ["intonation"]),  # trap: rendez-vous
        ("Où se trouve la gare ?", ["wh_fronted"]),
        ("Pourquoi tu pars ?", ["wh_svo"]),
        ("Tu vas où ?", ["in_situ"]),
        ("C'est où, la gare ?", ["in_situ"]),
        ("Il viendra peut-être demain.", []),  # not a question; trap: peut-être
        ("Tu peux m'envoyer le fichier ? Je pense pas avoir le temps.", ["intonation"]),
        # Regressions found on CoCoA's human references:
        ("Alors, où se situent vos locaux ?", ["wh_fronted"]),  # leading filler
        ("Eh bien, quel âge a votre petit-fils ?", ["wh_fronted"]),
        ("D'où vient votre famille ?", ["wh_fronted"]),  # preposition + question word
        ("Je me suis dit : « Quelle carrière ?", ["wh_fronted"]),  # reported speech
    ],
)
def test_question_forms(text, forms):
    assert features(text).question_forms == forms


# ----------------------------------------------------------- spoken forms


@pytest.mark.parametrize(
    "text, n",
    [
        ("T'es où ?", 1),
        ("Chuis crevé.", 1),
        ("J'pense pas.", 1),
        ("J'habite à Grenoble.", 0),  # trap: h is not a consonant slot (j'habite is standard)
        ("Il y a un problème.", 0),
        ("Il n'y a pas de souci.", 0),  # trap: "y a" inside "n'y a"
        ("Y a un problème.", 1),
        ("Y a-t-il autre chose ?", 0),  # regression: formal inversion of "il y a"
    ],
)
def test_oral_contractions(text, n):
    assert len(features(text).oral_contractions) == n


@pytest.mark.parametrize(
    "text, has_slot",
    [("Tu es où ?", True), ("Il y a un problème.", True), ("T'es où ?", False)],
)
def test_contraction_slots(text, has_slot):
    assert bool(features(text).contraction_slots) == has_slot


def test_typographic_apostrophes_and_spaces():
    # LLM output often uses ’ and a narrow no-break space before "?"
    assert features("T’es où\u202f?").oral_contractions == ["T'es"]
    assert features("T’es où\u202f?").question_forms == ["in_situ"]
    f = features("Je n’ai rien vu.")
    assert (f.ne_kept, f.ne_dropped) == (1, 0)
    assert features("Je t’aime.").address == "tu"


def test_sms_and_colloquial():
    assert len(features("Tu peux venir stp ?").sms) == 1
    assert len(features("J'ai trouvé un boulot sympa.").colloquial) == 2


# ------------------------------------------------------------------ check

REGISTERS = load_spec("registers")["registers"]
SPEC_EXAMPLES = [(r["id"], ex["fr"]) for r in REGISTERS for ex in r["examples"]]


@pytest.mark.parametrize("register_id, text", SPEC_EXAMPLES)
def test_spec_examples_obey_their_own_register(register_id, text):
    assert check(text, register_id) == set()


def test_golden_example_obeys_registers():
    fixture = json.loads((Path(__file__).parent / "fixtures/example_output.json").read_text())
    for rid in register_ids():
        assert check(fixture["output"]["variants"][rid]["text"], rid) == set(), rid


@pytest.mark.parametrize(
    "text, register_id, plural, expected",
    [
        ("Tu peux m'envoyer le fichier ?", "soutenu", False, {"address", "question_form"}),
        ("Pourquoi tu pars ?", "soutenu", False, {"address", "question_form"}),
        ("J'ai trouvé un boulot.", "soutenu", False, {"colloquial"}),
        ("Je sais pas si je peux venir.", "courant", False, {"ne_drop"}),
        ("Est-ce que vous pouvez venir stp ?", "courant", False, {"address", "sms"}),
        ("Vous venez ce soir ?", "amical", False, {"address"}),
        ("Vous venez tous ce soir ?", "amical", True, set()),  # plural vous is fine anywhere
        ("Viens-tu ce soir ?", "amical", False, {"question_form"}),
        ("Tu es où ?", "familier", False, {"oral_contractions"}),  # slot left unused
        ("T'es où ?", "familier", False, set()),
        ("Il n'y a pas de souci.", "familier", False, {"oral_contractions", "ne_drop"}),
    ],
)
def test_check_violations(text, register_id, plural, expected):
    assert check(text, register_id, plural_addressee=plural) == expected