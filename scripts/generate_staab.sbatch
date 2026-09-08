#!/bin/bash
# Staab adversarial-anonymization baseline — generation only (fully offline).
# Loads Qwen3.5-4B (anonymizer) + Qwen3.5-9B (in-loop adversary) on one fat GPU.
# Needs outputs/eval/v2/turns_undefended.jsonl from a prior run_eval_v2 run.
# Crash-safe/resumable: if it times out, just `sbatch` again and it continues.
#
#SBATCH --job-name=gen-staab
#SBATCH --nodes=1
#SBATCH --partition=fat_gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --time=20:00:00
#SBATCH --account=haberi2x_0000
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

cd "$SLURM_SUBMIT_DIR"
module load elysium/2026
export HF_HOME="${HF_HOME:-/lustre/urdeniw5/hf_cache}"
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Both models must already be in $HF_HOME (same models as eval). --no-sync so the
# compute node never contacts pypi. Add --thinking for the slower thinking A/B.
uv run --no-sync python scripts/generate_staab_benchmark.py "$@"
