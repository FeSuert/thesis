#!/bin/bash
# Full SFT run for the Defender (Qwen3.5-4B) on Elysium.
# Submit from the repo root:  sbatch scripts/sft.sbatch
#SBATCH --job-name=sft-qwen35-4b
#SBATCH --nodes=1
#SBATCH --partition=fat_gpu          # H100 80GB; use fat_gpu_h200 for the 141GB H200
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --time=04:00:00
#SBATCH --account=haberi2x_0000
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

cd "$SLURM_SUBMIT_DIR"
module load elysium/2026

# Point the HF cache at the Lustre location where models are pre-downloaded (login node).
# Respect an already-exported HF_HOME, else default to the known cache root.
export HF_HOME="${HF_HOME:-/lustre/urdeniw5/hf_cache}"
# Compute nodes have no internet: load the pre-downloaded model from the cache only.
export HF_HUB_OFFLINE=1
# W&B can't reach the internet from compute nodes: log locally, sync later from login node
# with `wandb sync wandb/offline-run-*`. (Use WANDB_MODE=disabled to skip logging entirely.)
export WANDB_MODE=offline
# Reduce CUDA memory fragmentation (helps with the large vocab logits).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Config is the first arg; defaults to the validated Qwen3-4B-Instruct-2507 run.
#   sbatch scripts/sft.sbatch configs/sft/qwen3_5-4b.yaml   # Qwen3.5-4B (merged, Fix A)
CONFIG="${1:-configs/sft/qwen35-4b.yaml}"
echo "[sft.sbatch] using config: $CONFIG"
uv run python -m thesis.training.sft.train --config "$CONFIG"
