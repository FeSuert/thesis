#!/bin/bash
# RQ1 longitudinal (ASR vs conversation length). Defender(4B) + attacker resident together.
# Default attacker Qwen3.5-9B → fat_gpu (H100). For Gemma cross-model curve add
# --partition=fat_gpu_h200 and --attacker google/gemma-4-31b-it. Extra args pass through:
#   sbatch scripts/rq1.sbatch --limit 60
#   sbatch --partition=fat_gpu_h200 scripts/rq1.sbatch --attacker google/gemma-4-31b-it --limit 60 --out outputs/eval/rq1-gemma
#SBATCH --job-name=rq1
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

uv run --no-sync python -m thesis.evaluation.rq1_longitudinal \
  --defenders undefended=none sft=outputs/sft-qwen3_5-4b/merged \
              sftdpo_v2=outputs/dpo-sftdpo-qwen3_5-4b-v2/merged \
  "$@"
