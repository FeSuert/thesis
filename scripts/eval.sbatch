#!/bin/bash
# Phase 4 evaluation — all Defender variants under one attacker. Needs the 9B attacker +
# one 4B defender resident together, so fat_gpu (H100). Extra args pass through, e.g.:
#   sbatch scripts/eval.sbatch --limit 60                          # PASS 1 quick (in-dist Qwen attacker)
#   sbatch scripts/eval.sbatch                                     # PASS 1 full benchmark
#   sbatch scripts/eval.sbatch --attacker <non-Qwen> --out outputs/eval/pass2   # PASS 2 cross-model
#SBATCH --job-name=eval-pass
#SBATCH --nodes=1
#SBATCH --partition=fat_gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --time=10:00:00
#SBATCH --account=haberi2x_0000
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

cd "$SLURM_SUBMIT_DIR"
module load elysium/2026
export HF_HOME="${HF_HOME:-/lustre/urdeniw5/hf_cache}"
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

uv run python -m thesis.evaluation.run_eval \
  --defenders \
    undefended=none \
    base=Qwen/Qwen3.5-4B \
    sft=outputs/sft-qwen3_5-4b/merged \
    sftdpo_v1=outputs/dpo-sftdpo-qwen3_5-4b/merged \
    sftdpo_v2=outputs/dpo-sftdpo-qwen3_5-4b-v2/merged \
    dpoonly=outputs/dpo-only-qwen3_5-4b/merged \
  "$@"
