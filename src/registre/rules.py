"""Rule-based register checks for French text (day 1, step 7).

features(text) extracts surface markers of register (address form, negation, question form,
spoken contractions, texting abbreviations, colloquial words). check(text, register_id)
compares them with that register's profile in spec/registers.yaml and returns violation codes.

These are heuristics. tests/test_rules.py pins down the behaviour that matters and the traps
that break naive regexes; on day 2 you measure their precision/recall against the human review.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import cache

from registre.schema import load_spec

# ---------------------------------------------------------------- input clean-up

# LLMs (and phones) write ’ instead of ' and put a narrow no-break space before ? ! : ;
# Without this, "t’es" or "n’ai" would slip past every pattern below.
_NORMALIZE = str.maketrans({"’": "'", "ʼ": "'", "‘": "'", "\u00a0": " ", "\u202f": " "})

# Hyphenated words that look like inversion or contain "vous" but are neither.
STOPWORDS = re.compile(r"\b(?:rendez-vous|peut-être|c'est-à-dire|là-bas|week-end)\b", re.I)
TOKEN = re.compile(r"\w+'|\w+")  # keeps elided forms whole: "c'", "qu'", "j'"

# ---------------------------------------------------------------- address

TU = re.compile(r"\b(?:tu|te|toi|ton|ta|tes|stp)\b|\bt'(?=\w)", re.I)  # t' = te or tu
VOUS = re.compile(r"\b(?:vous|votre|vos|svp)\b", re.I)

# ---------------------------------------------------------------- negation

NE = re.compile(r"\bne\b|\bn'", re.I)
NEGATION_WORDS = re.compile(r"\b(?:pas|jamais|rien|aucun|aucune|plus|personne)\b", re.I)
# Only these can reveal a *dropped* ne. "plus" (comparative) and "personne" (a noun) are too
# often not negations at all, so they only count when a ne is present.
DROPPABLE = {"pas", "jamais", "rien", "aucun", "aucune"}
# Fixed expressions that contain a negation word but negate nothing (or have no verb).
NOT_NEGATIONS = re.compile(r"\bpas mal d(?:e\b|')|\bsi jamais\b|\b(?:ou|pourquoi|non) pas\b", re.I)
# A dropped ne needs a verb with a subject before the negation: "je sais pas", "y a pas".
# Verbless phrases ("Pas de problème", "Sûrement pas", "Pas d'autre question ?") have no ne
# to drop, and are normal even in polite speech.
SUBJECTS_BEFORE_NEGATION = {
    "je", "j'", "tu", "t'", "il", "elle", "on", "nous", "vous", "ils", "elles",
    "c'", "ce", "ça", "y",
}  # fmt: skip
CLAUSE_BREAK = re.compile(r"[,.;:!?…]")

# ---------------------------------------------------------------- questions

# word-hyphen-subject pronoun, optionally with the euphonic -t- (a-t-il)
INVERSION = re.compile(r"\w-(?:t-)?(?:je|tu|il|elle|on|nous|vous|ils|elles)\b", re.I)
QUESTION_WORDS = {
    "où", "quand", "comment", "pourquoi", "qui", "que", "qu'", "quoi",
    "quel", "quelle", "quels", "quelles", "combien",
}  # fmt: skip
# Mid-sentence "que"/"qui" are usually relative pronouns or conjunctions, not questions.
IN_SITU_WORDS = QUESTION_WORDS - {"que", "qu'", "qui"}
SUBJECT_PRONOUNS = {
    "je", "j'", "tu", "t'", "il", "elle", "on", "nous", "vous", "ils", "elles", "ça", "c'",
}  # fmt: skip
# Skipped at the start of a question: "Alors, où… ?", "Eh bien, quel… ?", "Et toi, tu… ?"
LEADING_FILLERS = {
    "alors", "et", "mais", "donc", "bon", "ben", "eh", "bien", "oh", "ah", "ok",
    "sinon", "puis", "enfin", "voyons", "voir",
}  # fmt: skip
# A question word right after one of these is still fronted: "D'où… ?", "À quelle heure… ?"
PREPOSITIONS = {"d'", "de", "à", "pour", "avec", "par", "en", "chez", "depuis", "dans", "sur"}
# Reported speech: classify only what follows the last ":" or opening quote.
QUOTE_START = re.compile(r".*[:«“\"]")
SENTENCE_END = re.compile(r"(?<=[.!?…])\s+")

# ---------------------------------------------------------------- spoken forms

ORAL = re.compile(
    r"\b(?:t'es|t'as|t'étais|t'avais|chuis|j'suis)\b"
    r"|(?<!n')(?<!il )\by'?\s?a\b(?!-t-)"  # bare "y a" — not in "il y a", "n'y a", "y a-t-il"
    r"|\bj'(?=[bcdfgjklmnpqrstvwxzç])",  # j' before a consonant; not h ("j'habite" is standard)
    re.I,
)
# Full forms that a texting register would contract (familier: "expected" slots).
CONTRACTION_SLOTS = re.compile(r"\btu es\b|\btu as\b|\bil y a\b|\bil n'y a\b", re.I)
SMS = re.compile(r"\b(?:stp|svp|bcp|jsp|mdr|ptdr|slt|dsl|tkt|pk|pq|qd|auj|rdv|mtn)\b", re.I)
COLLOQUIAL = re.compile(
    r"\b(?:boulot|bouquin|bagnole|bosser|bouffer|truc|machin|ouais|sympa|crevée?|fringues"
    r"|mec|nana|potes?|kiffer|kiffe|ouf|relou|chelou)\b",
    re.I,
)


@dataclass
class Features:
    address: str  # "tu" | "vous" | "mixed" | "none"
    ne_kept: int  # clauses with a negation that keeps ne / n'
    ne_dropped: int  # clauses with pas/jamais/rien/aucun, a subject, and no ne
    question_forms: list[str] = field(default_factory=list)  # one per question sentence
    oral_contractions: list[str] = field(default_factory=list)
    contraction_slots: list[str] = field(default_factory=list)
    sms: list[str] = field(default_factory=list)
    colloquial: list[str] = field(default_factory=list)


def _address(text: str) -> str:
    text = STOPWORDS.sub(" ", text)  # "rendez-vous" is not vous
    tu, vous = bool(TU.search(text)), bool(VOUS.search(text))
    if tu and vous:
        return "mixed"
    return "tu" if tu else "vous" if vous else "none"


def _has_subject_before_negation(clause: str) -> bool:
    tokens = TOKEN.findall(clause)
    first_neg = next(i for i, t in enumerate(tokens) if t in DROPPABLE)
    return any(t in SUBJECTS_BEFORE_NEGATION for t in tokens[:first_neg])


def _negations(text: str) -> tuple[int, int]:
    """(kept, dropped), counted per clause: one clause = one negation decision."""
    kept = dropped = 0
    for clause in CLAUSE_BREAK.split(NOT_NEGATIONS.sub(" ", text.lower())):
        words = NEGATION_WORDS.findall(clause)
        if not words:
            continue
        if NE.search(clause):  # also covers "Personne n'est venu", where ne comes after
            kept += 1
        elif DROPPABLE.intersection(words) and _has_subject_before_negation(clause):
            dropped += 1
    return kept, dropped


def _question_form(sentence: str) -> str:
    """Classify one sentence ending in "?". The order of the tests matters."""
    s = STOPWORDS.sub(" ", sentence.lower())  # "peut-être", "rendez-vous" are not inversion
    s = QUOTE_START.sub(" ", s)  # « … Quelle carrière ? » → "Quelle carrière ?"
    if "est-ce qu" in s:  # also "qu'est-ce que", "est-ce qu'il"
        return "est_ce_que"
    if INVERSION.search(s):
        return "inversion"
    tokens = TOKEN.findall(s)
    while tokens and tokens[0] in LEADING_FILLERS:
        tokens = tokens[1:]
    if len(tokens) > 1 and tokens[0] in PREPOSITIONS and tokens[1] in QUESTION_WORDS:
        tokens = tokens[1:]
    if tokens and tokens[0] in QUESTION_WORDS:
        # "Pourquoi tu pars ?" (spoken) vs "Où se trouve la gare ?" (standard)
        return "wh_svo" if len(tokens) > 1 and tokens[1] in SUBJECT_PRONOUNS else "wh_fronted"
    if any(t in IN_SITU_WORDS for t in tokens[1:]):  # "Tu vas où ?", "C'est où, la gare ?"
        return "in_situ"
    return "intonation"


def features(text: str) -> Features:
    text = text.translate(_NORMALIZE)
    kept, dropped = _negations(text)
    sentences = SENTENCE_END.split(text.strip())
    return Features(
        address=_address(text),
        ne_kept=kept,
        ne_dropped=dropped,
        question_forms=[_question_form(s) for s in sentences if s.rstrip().endswith("?")],
        oral_contractions=[m.group(0) for m in ORAL.finditer(text)],
        contraction_slots=CONTRACTION_SLOTS.findall(text),
        sms=SMS.findall(text),
        colloquial=COLLOQUIAL.findall(text),
    )


@cache
def _profiles() -> dict[str, dict]:
    return {r["id"]: r["profile"] for r in load_spec("registers")["registers"]}


def check(text: str, register_id: str, plural_addressee: bool = False) -> set[str]:
    """Violation codes of `text` against the register's profile; empty set = compliant.

    plural_addressee: the variant speaks to several people, so "vous" is correct in every
    register (taken from the model's `assumptions` at evaluation time).
    """
    profiles = _profiles()
    if register_id not in profiles:
        raise ValueError(f"unknown register {register_id!r}; known: {sorted(profiles)}")
    p = profiles[register_id]
    f = features(text)
    v: set[str] = set()

    expected = p["address"]
    wrong_form = f.address in ("tu", "vous") and expected != "any" and f.address != expected
    plural_ok = plural_addressee and f.address == "vous"
    if f.address == "mixed" or (wrong_form and not plural_ok):
        v.add("address")

    if p["ne_drop"] == "forbidden" and f.ne_dropped:
        v.add("ne_drop")
    if p["ne_drop"] == "expected" and f.ne_kept:
        v.add("ne_drop")

    if any(q not in p["question_forms"] for q in f.question_forms):
        v.add("question_form")

    if p["oral_contractions"] == "forbidden" and f.oral_contractions:
        v.add("oral_contractions")
    if p["oral_contractions"] == "expected" and f.contraction_slots:
        v.add("oral_contractions")
    if p["sms_abbreviations"] == "forbidden" and f.sms:
        v.add("sms")
    if p["colloquial_lexicon"] == "forbidden" and f.colloquial:
        v.add("colloquial")
    return v