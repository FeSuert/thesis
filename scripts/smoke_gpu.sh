#!/usr/bin/env bash
# GPU smoke test for Elysium. Confirms the env sees a GPU and torch+bf16 work.
# Cheap on purpose: 1 GPU on the filler partition (NOT the costly H200 standard partition).
#
# Before submitting: set --account from `rub-acclist`.
#   sbatch scripts/smoke_gpu.sh
#
#SBATCH --job-name=smoke-gpu
#SBATCH --partition=gpu_filler          # A30 24GB, low FairShare cost, 1h limit
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=4
#SBATCH --time=00:15:00
#SBATCH --account=haberi2x_0000            # <-- from `rub-acclist`
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
mkdir -p "$REPO_ROOT/outputs/slurm"
cd "$REPO_ROOT"
source "$HOME/.local/bin/env" 2>/dev/null || true

echo "== host: $(hostname) | date: $(date -Is)"
nvidia-smi

uv run python - <<'PY'
import torch
print("torch:", torch.__version__, "| cuda available:", torch.cuda.is_available())
assert torch.cuda.is_available(), "no CUDA visible in the job"
print("device:", torch.cuda.get_device_name(0))
x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
print("bf16 matmul ok:", (x @ x).shape)
print("SMOKE OK")
PY
