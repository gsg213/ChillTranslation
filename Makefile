.PHONY: setup test test-schema lint schema data bakeoff

setup:            ## install the environment and git hooks
	uv sync --all-groups
	uv run pre-commit install

test:             ## all tests (test_rules fails until you write src/registre/rules.py)
	uv run pytest -q

test-schema:
	uv run pytest -q tests/test_schema.py

lint:
	uv run ruff check . && uv run ruff format --check .

schema:           ## export the JSON schema used by the teacher and vLLM
	uv run python -m registre.schema > spec/output.schema.json

data:             ## build processed splits from data/raw (you write registre.data.build)
	uv run python -m registre.data.build

bakeoff:          ## run on a GPU node (you write registre.bakeoff)
	uv run python -m registre.bakeoff --config configs/bakeoff.yaml
