#!/bin/bash
# DPO training for the Defender (Phase 3; D-021). Runs on an H100 (fat_gpu).
# Config is arg 1 (default below); extra overrides pass through, e.g.:
#   sbatch scripts/dpo.sbatch configs/dpo/qwen3_5-4b.yaml                                  # SFT+DPO
#   sbatch scripts/dpo.sbatch configs/dpo/qwen3_5-4b.yaml --override defender.base_model=Qwen/Qwen3.5-4B run_name=dpo-only-qwen3_5-4b   # DPO-only
#SBATCH --job-name=dpo-qwen3_5-4b
#SBATCH --nodes=1
#SBATCH --partition=fat_gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --time=04:00:00
#SBATCH --account=haberi2x_0000
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

cd "$SLURM_SUBMIT_DIR"
module load elysium/2026

export HF_HOME="${HF_HOME:-/lustre/urdeniw5/hf_cache}"
export HF_HUB_OFFLINE=1
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CONFIG="${1:-configs/dpo/qwen3_5-4b.yaml}"
shift 2>/dev/null || true
echo "[dpo.sbatch] config: $CONFIG  extra: $*"
uv run python -m thesis.training.dpo.train --config "$CONFIG" "$@"
