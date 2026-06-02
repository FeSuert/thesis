#!/usr/bin/env bash
# One-time environment setup on the H200 cluster.
# Safe to re-run: it's idempotent.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$REPO_ROOT"

echo "==> Repo: $REPO_ROOT"

# 1. Ensure uv is available.
if ! command -v uv >/dev/null 2>&1; then
    echo "==> Installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # shellcheck disable=SC1091
    source "$HOME/.local/bin/env" 2>/dev/null || true
fi

echo "==> uv: $(uv --version)"

# 2. Sync dependencies. Let uv pick the right torch wheel for the platform.
echo "==> Syncing dependencies"
uv sync --extra dev

# 3. Verify PyTorch sees CUDA.
echo "==> CUDA check"
uv run python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device count:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"  [{i}] {torch.cuda.get_device_name(i)}")
PY

# 4. Copy .env.example to .env if .env is missing.
if [ ! -f "$REPO_ROOT/.env" ]; then
    echo "==> Creating .env from .env.example (EDIT IT before running training)"
    cp "$REPO_ROOT/.env.example" "$REPO_ROOT/.env"
fi

echo "==> Done. Edit .env, then you're ready to run jobs."
