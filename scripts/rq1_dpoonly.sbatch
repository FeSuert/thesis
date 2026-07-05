#!/bin/bash
# RQ1 longitudinal — DPO-only variant, to complete the deep curve.
# Mirrors job 5502602 settings exactly (limit 80, checkpoints 1..64, full-history 32k
# attacker Qwen3.5-9B, benchmark split) so the numbers line up with outputs/eval/rq1-qwen-deep.
# 'undefended' is included as an alignment check: it should reproduce the existing
# undefended curve (~0.30 @k1 -> ~0.83 @k64). Expect n_attr=431 per full checkpoint.
#   sbatch scripts/rq1_dpoonly.sbatch
#SBATCH --job-name=rq1-dpoonly
#SBATCH --nodes=1
#SBATCH --partition=fat_gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --time=06:00:00
#SBATCH --account=haberi2x_0000
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

cd "$SLURM_SUBMIT_DIR"
module load elysium/2026
export HF_HOME="${HF_HOME:-/lustre/urdeniw5/hf_cache}"
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

uv run python -m thesis.evaluation.rq1_longitudinal \
  --defenders undefended=none dpoonly=outputs/dpo-only-qwen3_5-4b/merged \
  --attacker Qwen/Qwen3.5-9B \
  --split benchmark \
  --limit 80 \
  --checkpoints 1,3,5,8,12,16,24,32,48,64 \
  --max-context-tokens 32000 \
  --out outputs/eval/rq1-qwen-deep-dpoonly \
  "$@"
