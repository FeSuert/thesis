# Configs

Base config: `base.yaml`. Phase-specific configs live under subdirectories:

```
configs/
  base.yaml
  sft/           # SFT warm-up (Phase 2) — paraphrase corpus
  dpo/           # DPO training (Phase 3)
  eval/          # evaluation configs (Phase 4)
```

Subdirectories are empty for now; they are populated per phase.

Paths in configs are resolved from environment variables (see `.env.example`), so the same config works on Mac and on H200.
