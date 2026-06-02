#!/usr/bin/env bash
# SLURM job submission template. Adapt to the actual cluster once confirmed.
#
# Usage:
#   sbatch scripts/submit_example.sh <training-entry-point> [args...]
#
# Example:
#   sbatch scripts/submit_example.sh thesis.training.sft --config configs/sft/base.yaml

#SBATCH --job-name=thesis
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

mkdir -p "$REPO_ROOT/outputs/slurm"

cd "$REPO_ROOT"

# Ensure uv is on PATH inside the job.
# shellcheck disable=SC1091
source "$HOME/.local/bin/env" 2>/dev/null || true

# Load .env for run-time config.
# shellcheck disable=SC1091
[ -f .env ] && set -a && source .env && set +a

# Record run metadata.
echo "== Job: $SLURM_JOB_NAME ($SLURM_JOB_ID)"
echo "== Host: $(hostname)"
echo "== Git SHA: $(git rev-parse --short HEAD 2>/dev/null || echo 'n/a')"
echo "== Date: $(date -Is)"
echo "== Args: $*"

uv run python -m "$@"
