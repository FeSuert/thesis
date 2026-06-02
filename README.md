# thesis-code

Code for the thesis **"Protecting User Privacy in Long-Term AI Chats: A System to Automatically Rewrite Prompts."**

## Overview

This repository contains everything needed to train, evaluate, and deploy the three-model adversarial framework (Attacker, Evaluator, Defender) described in the thesis, plus the browser extension that ships the Defender on-device.

## Environments

Code runs in two environments:

| Environment | Purpose | CUDA |
|---|---|---|
| **Local** | Edit, lint, small tests, data annotation tooling, writeup | No (CPU / MPS) |
| **Remote H200** | SFT, DPO, preference generation, full evaluation | Yes |

**Code is shared via git.** Data, checkpoints, and logs are never committed. See `docs/local-dev.md` and `docs/remote-h200.md` for the full workflow.

## Quick start

### Local

```bash
# Install uv if you don't have it: https://docs.astral.sh/uv/
uv sync --extra dev
uv run pytest tests/
```

### Remote (H200)

```bash
./scripts/setup_env.sh       # one-time; installs uv + syncs dependencies with CUDA
cp .env.example .env         # then edit for cluster-specific paths
uv run python -m thesis.cli --help
```

## Repository layout

```
thesis-code/
├── src/thesis/              # Python package (import path: `thesis.*`)
│   ├── data/                # dataset loaders (PersonaMem-v2, SFT corpus, WildChat fallback)
│   ├── models/
│   │   ├── attacker/        # LLM and encoder-based Attacker candidates
│   │   ├── evaluator/       # semantic similarity + task faithfulness
│   │   └── defender/        # Defender base + SFT + DPO heads
│   ├── training/
│   │   ├── sft/             # supervised fine-tuning on paraphrase corpus
│   │   ├── dpo/             # Direct Preference Optimization
│   │   └── preference_gen/  # Algorithm 1 Phase 1 — generate D_pref
│   ├── evaluation/
│   │   ├── rq1/             # longitudinal risk (ASR vs. context length)
│   │   ├── rq2/             # privacy mitigation + cross-model transfer
│   │   ├── rq3/             # utility preservation
│   │   └── baselines/       # Presidio, zero-shot LLM, IncogniText-style
│   └── utils/               # shared helpers (config, logging, seeds)
├── configs/                 # YAML configs (Hydra-compatible)
├── scripts/                 # env setup, job submission templates
├── tests/                   # pytest
├── docs/
│   ├── local-dev.md
│   ├── remote-h200.md
│   └── reproducibility.md
├── extension/               # browser extension (Phase 5 — empty for now)
├── notebooks/               # exploratory notebooks (not committed by default)
├── pyproject.toml
├── .env.example             # template for environment variables
├── .gitignore
└── README.md
```

## Experiment tracking

- **W&B** is the default (configured via `WANDB_PROJECT` in `.env`).
- Every run is logged with: config hash, dataset version, seed, git SHA.
- The `experiments.md` log is an append-only human-readable index of runs.

## Data, checkpoints, and secrets

- **Never committed.** `.gitignore` excludes `data/`, `checkpoints/`, `runs/`, `wandb/`, `.env`.
- Paths on the cluster are set via environment variables in `.env` (see `.env.example`).
- The repo stays small and portable.

## Decisions and planning

All design decisions are logged in `../thesis/thoughts/decisions.md`. See also:
- `../thesis/thoughts/proposal-analysis.md`
- `../thesis/thoughts/phase1-kickoff-checklist.md`
- `../thesis/thoughts/research-log.md`

## Open TODOs for setup

- [ ] Pick a git hosting target (GitHub / GitLab / university git) and add as `origin`.
- [ ] Configure pre-commit hooks: `uv run pre-commit install`.
- [ ] Confirm H200 cluster job submission mechanism (SLURM? Bash?) — update `scripts/`.
- [ ] First sync on H200: clone repo, run `scripts/setup_env.sh`, verify `uv run python -c "import torch; print(torch.cuda.is_available())"` returns `True`.
