# Decision log

One entry per decision: what, why, and what it rules out. Add the date when you make it.
Decisions are hypotheses — if later evidence contradicts one, add a new entry; don't edit the old one.

## D1 — Translation direction: Spanish → French
- Source: Spanish as I write it. Target: French from France.
- Why: I compose in Spanish; the goal is producing French that says what I mean.

## D2 — Four registers, always
- soutenu, courant, amical, familier — every request returns all four.
- The source's tú/usted is reported in `source_address`, never obeyed.
- Rules out: inferring "the" register from the source.

## D3 — Explanations in English, from a closed tag taxonomy
- Notes in English (readable by reviewers and the LLM judge); tags from spec/taxonomy.yaml.
- Rules out: free-form explanations that can't be evaluated.

## D4 — Infrastructure split
- Training, evaluation and benchmarks: Grid5000.
- Serving: local PC with an RTX 3090 (24 GB), vLLM in a container.
- Interface: a Telegram bot (long polling, so no open ports), restricted to my Telegram user id.
- Consequence: weights at serving precision must fit in ~16 GB, leaving room for KV cache,
  CUDA context and the desktop. bf16 by default; a 4-bit variant is measured on day 5 as a
  latency/quality trade-off, not a necessity.

## D5 — Scope cuts
- No per-register LoRA adapters: one adapter, merged into the base model for serving.
- No cloud deployment: the project is trained on a cluster and served locally.

## D6 — Base model (fill in BEFORE running the bake-off)
- Candidates:
- Hard constraints: serving weights ≤ ~16 GB (≈ 8B in bf16, larger only if quantized);
  supported by vLLM; license compatible with a public repo.
- Primary criterion:
- Tie-breaker:
- Disqualifier:
- Result and choice:

## D7 — Everything in containers
- Host needs only Docker (and, on GPU machines, the NVIDIA driver + container toolkit).
- Dev tools run in the `dev` image (./run <target>); serving runs in the official vLLM image.
- Why: nothing installed on my machines, and the same environment everywhere.
