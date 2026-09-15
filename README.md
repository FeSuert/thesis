# thesis

Code and artifacts for the master's thesis **"Protecting User Privacy in Long-Term AI Chats: A System to Automatically Rewrite Prompts"** (Ivan Urdenko, Ruhr-University Bochum).

A small, on-device **Defender** model rewrites each user message to hide personal attributes (location, profession, age, sex, relationship status, income) before it is sent to an external chatbot, while preserving the meaning of the request. It is trained with a three-model adversarial framework — an **Attacker** that scores privacy leakage and an **Evaluator** that scores meaning kept — using supervised fine-tuning (SFT) followed by Direct Preference Optimization (DPO).

- **Model (Hugging Face):** [`oberus/qwen3.5-4b-privacy-defender`](https://huggingface.co/oberus/qwen3.5-4b-privacy-defender)
- **Base model:** [`Qwen/Qwen3.5-4B`](https://huggingface.co/Qwen/Qwen3.5-4B) (Apache-2.0)
- **Benchmark:** [PersonaMem-v2](https://huggingface.co/datasets/bowen-upenn/PersonaMem-v2)

## Results

Evaluated on 200 held-out personas; all headline numbers have 95% bootstrap confidence intervals in the thesis.

**Risk grows with chat length (RQ1).** Undefended attack success rate (ASR) nearly triples as a chat grows — 0.30 (1 turn) → 0.84 (64 turns) — then levels off. The Defender lowers the whole curve (0.26 → 0.65).

**The Defender lowers risk and keeps meaning (RQ2, 15-turn operating point, Qwen3.5-9B attacker):**

| Method | ASR ↓ | Meaning kept (S_sem) ↑ | % messages changed |
|---|---|---|---|
| Undefended | 0.688 | — | 0% |
| Presidio (redaction baseline) | 0.590 | 0.964 | 39% |
| Staab et al. (adversarial anonymization) | 0.633 | 0.992 | 8% |
| Gemini 3.5 Flash (cloud teacher) | 0.562 | 0.985 | 28% |
| **Defender (SFT+DPO)** | **0.522** | **0.994** | 17% |

The largest, statistically real drops are on **location** (0.645 → 0.230, −64%) and **sex** (0.812 → 0.452, −44%). Attributes that build up across turns (age, income) are harder for a memoryless rewriter — this is the thesis's central finding.

**Protection transfers to unseen attackers (RQ2):** relative ASR reduction of −24% against Gemma 4 31B and −16% against Kimi K3, close to the −24% seen in-distribution.

**Answers stay useful (RQ3):** the answer to a rewritten prompt is as good or better ~90% of the time (LLM-as-a-judge), with far less quality loss than the Presidio baseline.

## Repository layout

```
thesis/
├── src/thesis/              # Python package (import path: `thesis.*`)
│   ├── data/                # PersonaMem-v2 loader
│   ├── models/
│   │   ├── attacker/        # LLM attacker + verbalized-confidence calibration
│   │   ├── evaluator/       # semantic-similarity utility scorer
│   │   └── defender/        # Defender prompt + inference
│   ├── training/
│   │   ├── sft/             # supervised fine-tuning
│   │   ├── preference_gen/  # build the DPO preference pairs
│   │   └── dpo/             # Direct Preference Optimization
│   ├── evaluation/
│   │   ├── rq1/             # longitudinal risk vs. chat length
│   │   ├── rq2/             # privacy mitigation + cross-model transfer
│   │   ├── rq3/             # answer-quality (LLM-as-a-judge)
│   │   ├── baselines/       # Presidio, Gemini cloud teacher, Staab et al.
│   │   └── bootstrap_ci.py  # paired bootstrap confidence intervals
│   └── utils/               # config, API client, reproducibility
├── configs/                 # SFT / DPO YAML configs
├── scripts/                 # SLURM (.sbatch) jobs + data-generation scripts
├── data-public/             # synthetic SFT corpus + rebuilt ground-truth labels
├── extension/               # on-device browser extension (see below)
├── tests/                   # pytest
├── notebooks/
├── pyproject.toml
├── .env.example
└── README.md
```

Data, model checkpoints, logs, and secrets are **never committed** (see `.gitignore`); they live on the cluster or are downloaded on demand. `data-public/` holds only synthetic SFT pairs and derived attribute labels for reproducibility.

## Quick start

```bash
# install uv: https://docs.astral.sh/uv/
uv sync --extra dev
uv run pytest tests/
```

Training and evaluation ran on the Ruhr-University Bochum **Elysium** HPC cluster via the SLURM scripts in `scripts/`. Copy `.env.example` to `.env` and set the cluster paths and API keys before running.

## Browser extension (on-device deployment)

`extension/` ships the Defender as a Chrome (Manifest V3) extension for ChatGPT and Claude. A content script intercepts the message on send and shows a review box with the rewritten version; the user chooses what to send. The model runs **locally** — `extension/local_server/` is a small Flask server (`127.0.0.1`) that serves the merged Defender via native transformers, so raw text never leaves the machine.

A fully in-browser (WebGPU/WebLLM) version was attempted but is not yet possible: the `qwen3.5` architecture (a multimodal, hybrid linear-attention/SSM model) is not supported by current browser inference runtimes. See `extension/README.md` for setup.

## Citation

```bibtex
@mastersthesis{urdenko2026privacy,
  title  = {Protecting User Privacy in Long-Term AI Chats: A System to Automatically Rewrite Prompts},
  author = {Urdenko, Ivan},
  school = {Ruhr-University Bochum},
  year   = {2026}
}
```

## License

Code released under the MIT License. The released model inherits the Apache-2.0 license of its base model, Qwen3.5-4B.
