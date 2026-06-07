"""Inspect a saved LoRA adapter: how many tensors, which projection types, which layers.

Usage:
    uv run python scripts/inspect_adapter.py outputs/sft-qwen35-4b/final
"""
import collections
import sys
from pathlib import Path

from safetensors import safe_open

adapter = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/sft-qwen35-4b/final")
f = safe_open(str(adapter / "adapter_model.safetensors"), "pt")
keys = list(f.keys())

proj = collections.Counter()
layers = set()
for k in keys:
    parts = k.split(".")
    # find the projection name (…_proj) and the layer index
    for i, p in enumerate(parts):
        if p.endswith("_proj"):
            proj[p] += 1
        if p == "layers" and i + 1 < len(parts):
            layers.add(int(parts[i + 1]))

print(f"adapter dir       : {adapter}")
print(f"total LoRA tensors: {len(keys)}")
print(f"by projection type: {dict(proj)}")
print(f"layers with LoRA  : {len(layers)} -> {sorted(layers)}")
print(f"example keys      :")
for k in keys[:6]:
    print("   ", k)
