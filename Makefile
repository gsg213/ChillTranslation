# Runs INSIDE the dev container. From the host, call ./run <target>.
.PHONY: help test test-schema lint format precommit schema download data smoke upload-data

help:          ## list targets
	@grep -E '^[a-z-]+:.*##' Makefile | sed -E 's/:.*## / — /'

test:          ## all tests (test_rules fails until src/registre/rules.py exists)
	uv run pytest -q

test-schema:   ## schema tests only
	uv run pytest -q tests/test_schema.py

lint:          ## ruff lint + format check
	uv run ruff check .
	uv run ruff format --check .

format:        ## auto-fix lint and formatting
	uv run ruff check --fix .
	uv run ruff format .

precommit: lint test-schema  ## what the git hook runs before each commit

schema:        ## export the JSON schema used by the teacher and vLLM
	uv run python -m registre.schema > spec/output.schema.json

download:      ## fetch Tatoeba + CoCoA-MT into data/raw
	sh scripts/download.sh

data:          ## build processed splits (you write registre.data.build)
	uv run python -m registre.data.build

smoke:         ## structured-output check against the vllm service (./run vllm-up first)
	uv run python scripts/smoke_vllm.py

upload-data:   ## push data/processed to a private HF dataset repo (HF_DATASET_REPO in .env)
	uv run hf upload $(HF_DATASET_REPO) data/processed . --repo-type dataset --private
