#!/bin/bash
# Phase 3 preference generation. Needs the 4B Defender + 9B Attacker + mpnet
# resident together (~26 GB+), so it runs on an H100 (fat_gpu), not the A30.
# Submit from repo root; extra args pass through to the generator:
#   sbatch scripts/prefgen.sbatch --sessions 12 --max-turns 6 --k 4
#SBATCH --job-name=prefgen
#SBATCH --nodes=1
#SBATCH --partition=fat_gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --time=02:00:00
#SBATCH --account=haberi2x_0000
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

cd "$SLURM_SUBMIT_DIR"
module load elysium/2026

export HF_HOME="${HF_HOME:-/lustre/urdeniw5/hf_cache}"
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

uv run python -m thesis.training.preference_gen.generate \
    --defender outputs/sft-qwen3_5-4b/merged --data-root data/personamem-v2 "$@"
