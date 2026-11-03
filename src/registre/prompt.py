"""The task definition sent to a model: used by the day-1 bake-off and the day-2 teacher run.

Everything is rendered from spec/ (registers, taxonomy, golden example), so the prompt, the
data filters and the evaluation all read the same definition. prompt_hash() identifies the
exact prompt that produced a dataset; it is stored with every generated record.
"""

from __future__ import annotations

import hashlib
import json
from functools import cache

from registre.schema import MAX_ANNOTATIONS, MAX_NOTE_CHARS, SPEC_DIR, load_spec

_LEVEL = {
    "forbidden": "never",
    "allowed": "allowed",
    "expected": "use them wherever possible",
}
# The ids used by rules.py, rendered so a model knows what each one means.
_QUESTION_FORMS = {
    "inversion": "inversion (Pouvez-vous… ?)",
    "est_ce_que": "est-ce que (Est-ce que vous… ?)",
    "intonation": "intonation only (Tu peux… ?)",
    "wh_fronted": "question word first (Où est… ?)",
    "wh_svo": "question word + subject pronoun (Pourquoi tu pars ?)",
    "in_situ": "question word at the end (Tu vas où ?)",
}


def _registers_section() -> list[str]:
    lines = ["## The four registers (always produce all four, in this order)"]
    for r in load_spec("registers")["registers"]:
        p = r["profile"]
        lines += [
            f"### {r['id']} — {r['label']}",
            f"Use when: {r['use_when']}",
            f"- Address: {p['address']}",
            f"- Dropping 'ne' in negations: {_LEVEL[p['ne_drop']]}",
            "- Question forms allowed: "
            + "; ".join(_QUESTION_FORMS[q] for q in p["question_forms"]),
            f"- Spoken contractions (t'es, y a, j'sais): {_LEVEL[p['oral_contractions']]}",
            f"- Texting abbreviations (stp, bcp): {_LEVEL[p['sms_abbreviations']]}",
            f"- Colloquial words (boulot, truc): {_LEVEL[p['colloquial_lexicon']]}",
            *[f"- {rule}" for rule in r.get("judge_only", [])],
            "Examples:",
            *[f"- {ex['es']} → {ex['fr']}" for ex in r["examples"]],
            "",
        ]
    return lines


def _conventions_section() -> list[str]:
    conv = load_spec("registers")["conventions"]
    return ["## Conventions", *[f"- {k}: {' '.join(v.split())}" for k, v in conv.items()], ""]


def _tags_section() -> list[str]:
    tags = load_spec("taxonomy")["tags"]
    ordered = sorted(tags, key=lambda t: not t["es_contrast"])  # Spanish-contrast tags first
    return [
        "## Annotation tags (use only these ids)",
        "Prefer the tags marked [ES], where Spanish works differently from French.",
        *[
            f"- {t['id']}{' [ES]' if t['es_contrast'] else ''}: {t['description']} "
            f"(e.g. {t['example']})"
            for t in ordered
        ],
        "",
    ]


def _output_section() -> list[str]:
    golden = json.loads((SPEC_DIR / "golden_example.json").read_text(encoding="utf-8"))
    return [
        "## Output",
        "Reply with JSON only, matching the provided schema:",
        "- source_address: how the Spanish addresses someone (tu, usted, vos, plural, none, "
        "or ambiguous when the verb form could be usted or él/ella). Report it; never let it "
        "change the registers.",
        "- assumptions: short notes on choices the source left open (who is addressed, "
        "gender, singular or plural).",
        f"- For each variant: the French text, then at most {MAX_ANNOTATIONS} annotations. "
        "Each span must be copied exactly from that variant's text.",
        f"- Notes are in English, at most {MAX_NOTE_CHARS} characters, and explain the French "
        "choice, contrasting with Spanish when that helps a Spanish speaker.",
        "",
        "## Example",
        "Input:",
        json.dumps(golden["input"], ensure_ascii=False),
        "Output:",
        json.dumps(golden["output"], ensure_ascii=False),
    ]


@cache
def system_prompt() -> str:
    intro = [
        "You translate Spanish into French as spoken in France, in four registers, and explain "
        "each translation to a Spanish-speaking learner of French.",
        "",
    ]
    sections = _registers_section() + _conventions_section() + _tags_section()
    return "\n".join(intro + sections + _output_section())


def user_message(source: str, speaker_gender: str) -> str:
    payload = {"source": source, "source_lang": "es", "speaker_gender": speaker_gender}
    return json.dumps(payload, ensure_ascii=False)


def messages(source: str, speaker_gender: str, system_role: bool = True) -> list[dict]:
    """Chat messages. system_role=False merges the system prompt into the user turn, for
    models whose chat template rejects a system message."""
    if system_role:
        return [
            {"role": "system", "content": system_prompt()},
            {"role": "user", "content": user_message(source, speaker_gender)},
        ]
    merged = f"{system_prompt()}\n\n## Input\n{user_message(source, speaker_gender)}"
    return [{"role": "user", "content": merged}]


def prompt_hash() -> str:
    return hashlib.sha256(system_prompt().encode("utf-8")).hexdigest()[:12]


if __name__ == "__main__":
    print(system_prompt())
    print(f"\n[prompt_hash={prompt_hash()}  ~{len(system_prompt()) // 4} tokens]")
