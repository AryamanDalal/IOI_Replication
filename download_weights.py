"""Download + cache GPT-2 small weights while you have internet.

Run once, online:  python download_weights.py

Then `verify_offline.py` proves the cached copy loads with no network.
"""

from __future__ import annotations

import sys

import torch
from transformer_lens import HookedTransformer


def main() -> int:
    torch.set_grad_enabled(False)
    print("Downloading + caching gpt2 (~500MB on first run) ...", flush=True)
    model = HookedTransformer.from_pretrained("gpt2", device="cpu")
    logits = model("Hello, world")
    print(f"OK  cached and loaded. {model.cfg.n_layers} layers, "
          f"logits shape {tuple(logits.shape)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
