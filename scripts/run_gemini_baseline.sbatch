#!/bin/bash
# RQ2/RQ3 Baseline generation using Gemini API
# Does not require a GPU.
#   sbatch scripts/run_gemini_baseline.sbatch
#SBATCH --job-name=gemini_baseline
#SBATCH --nodes=1
#SBATCH --partition=cpu       # Use CPU partition to save GPU quota
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --account=haberi2x_0000
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

cd "$SLURM_SUBMIT_DIR"
module load elysium/2026

# Load the .env file to expose GEMINI_API_KEY
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Ensure the SDK is available
uv pip install google-generativeai

# Run your standard evaluation script, pointing to the new baseline class
# Assuming your eval script accepts a --baseline flag or similar router
uv run --no-sync python -m thesis.evaluation.rq2_privacy \
  --defender gemini \
  --out outputs/eval/baselines/gemini-flash
