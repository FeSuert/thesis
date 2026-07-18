#!/bin/bash
# Instrumented eval run — full benchmark, Presidio baseline included.
# Produces per-turn/per-persona artifacts for distribution analysis and bootstrap CIs.
#
# Usage:
#   sbatch scripts/eval_v2.sbatch                         # Qwen attacker (default)
#   sbatch scripts/eval_v2.sbatch --attacker google/gemma-4-31b-it --out outputs/eval/v2-gemma
#
#SBATCH --job-name=eval-v2
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

# Presidio needs a spacy model — download on the login node before submitting:
#   uv run python -m spacy download en_core_web_lg
# (compute nodes have no internet, so this must already be cached)
uv run python -c "import spacy; spacy.load('en_core_web_lg')" || {
  echo "ERROR: en_core_web_lg not found. Run 'uv run python -m spacy download en_core_web_lg' on the login node first."
  exit 1
}

uv run python -m thesis.evaluation.run_eval_v2 \
  --defenders \
    undefended=none \
    sft=outputs/sft-qwen3_5-4b/merged \
    sftdpo_v2=outputs/dpo-sftdpo-qwen3_5-4b-v2/merged \
  --presidio \
  "$@"
