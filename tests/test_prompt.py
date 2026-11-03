import json

from registre.prompt import messages, prompt_hash, system_prompt
from registre.schema import TranslationOutput, load_spec, register_ids


def test_prompt_mentions_every_register_and_tag():
    prompt = system_prompt()
    assert all(f"### {rid}" in prompt for rid in register_ids())
    assert all(f"- {t['id']}" in prompt for t in load_spec("taxonomy")["tags"])


def test_spanish_contrast_tags_come_first():
    prompt = system_prompt()
    tags = load_spec("taxonomy")["tags"]
    last_es = max(prompt.index(f"- {t['id']} [ES]") for t in tags if t["es_contrast"])
    first_other = min(prompt.index(f"- {t['id']}:") for t in tags if not t["es_contrast"])
    assert last_es < first_other


def test_example_in_prompt_is_valid_output():
    example = system_prompt().split("Output:\n", 1)[1].strip()
    TranslationOutput.model_validate_json(example)


def test_messages_with_and_without_system_role():
    with_system = messages("¿Vienes?", "f")
    assert [m["role"] for m in with_system] == ["system", "user"]
    assert json.loads(with_system[1]["content"])["speaker_gender"] == "f"
    merged = messages("¿Vienes?", "f", system_role=False)
    assert [m["role"] for m in merged] == ["user"]
    assert system_prompt() in merged[0]["content"]


def test_prompt_hash_is_stable():
    assert prompt_hash() == prompt_hash() and len(prompt_hash()) == 12
