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

# Compute nodes have no internet: load the pre-downloaded model from the cache only.
export HF_HUB_OFFLINE=1
# Reduce CUDA memory fragmentation (helps with the large 248K-vocab logits).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

uv run python -m thesis.training.sft.train --config configs/sft/qwen35-4b.yaml
