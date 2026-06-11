"""Prove the model weights are cached and load WITHOUT network access.

The first `HookedTransformer.from_pretrained("gpt2")` downloads ~500MB and
caches it under ~/.cache/huggingface. This script forces offline mode so a
successful load proves the cache works -- i.e. the replication runs on a plane.

Run:  python verify_offline.py
"""

from __future__ import annotations

import os
import sys

# Must be set BEFORE importing transformers/transformer_lens so the libraries
# pick up offline mode at import time.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch  # noqa: E402
from transformer_lens import HookedTransformer  # noqa: E402


def main() -> int:
    torch.set_grad_enabled(False)
    print("Loading gpt2 with HF_HUB_OFFLINE=1 (no network allowed) ...", flush=True)
    try:
        model = HookedTransformer.from_pretrained("gpt2", device="cpu")
    except Exception as exc:  # pragma: no cover - diagnostic path
        print(f"FAIL: offline load raised: {exc}")
        print("The weights are probably not cached yet. Run download_weights.py "
              "once while online.")
        return 1

    # A real forward pass, fully offline, to prove the weights are intact.
    logits = model("The quick brown fox")
    print(f"OK  offline load + forward pass succeeded. logits shape: {tuple(logits.shape)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
