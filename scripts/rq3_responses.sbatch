#!/bin/bash
# RQ3 step 1 — generate responses for original vs. rewritten prompts.
# Needs the turns file from a finished eval-v2 run.
#
# Usage:
#   sbatch scripts/rq3_responses.sbatch                # default: sftdpo_v2 turns
#   sbatch scripts/rq3_responses.sbatch outputs/eval/v2-qwen/turns_sft.jsonl outputs/eval/rq3/responses_sft.jsonl
#
#SBATCH --job-name=rq3-resp
#SBATCH --nodes=1
#SBATCH --partition=fat_gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --time=12:00:00
#SBATCH --account=haberi2x_0000
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

cd "$SLURM_SUBMIT_DIR"
module load elysium/2026
export HF_HOME="${HF_HOME:-/lustre/urdeniw5/hf_cache}"
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TURNS="${1:-outputs/eval/v2-qwen/turns_sftdpo_v2.jsonl}"
OUT="${2:-outputs/eval/rq3/responses_sftdpo_v2.jsonl}"

uv run python -m thesis.evaluation.rq3_responses \
  --turns "$TURNS" \
  --out "$OUT"
