"""Executable 'setup is done' criterion for the IOI replication environment.

Run:  python smoke_test.py

Checks, in order, exiting non-zero on the first failure:
  (a) GPT-2 small loads via TransformerLens.
  (b) One forward pass on an IOI prompt runs (no dtype/device errors).
  (c) Logit difference logit[IO] - logit[S] is POSITIVE on clean prompts.
  (d) One activation patch (single layer + position) runs end to end and
      returns a number.

(d) is a PLUMBING check -- it proves the patching machinery works, not that
any particular circuit claim holds. The real patching experiment stays in the
notebook.

Device defaults to CPU: GPT-2 small is tiny, and CPU sidesteps MPS dtype quirks
that would otherwise eat the afternoon. Set IOI_DEVICE=mps to try the Apple GPU.
"""

from __future__ import annotations

import os
import sys
from functools import partial

import torch
from transformer_lens import HookedTransformer

from ioi_prompts import build_prompts, verify_alignment

DEVICE = os.environ.get("IOI_DEVICE", "cpu")


def logit_diff(model: HookedTransformer, tokens: torch.Tensor, io_id: int, s_id: int) -> float:
    """logit[IO] - logit[S] at the final sequence position, for one prompt."""
    logits = model(tokens)  # [batch, seq, d_vocab]
    final = logits[0, -1]   # [d_vocab]
    return (final[io_id] - final[s_id]).item()


def patch_resid_at_pos(activation, hook, clean_act, pos):
    """Replace the residual stream at a single position with the clean value."""
    activation[:, pos, :] = clean_act[:, pos, :]
    return activation


def main() -> int:
    torch.set_grad_enabled(False)

    # (a) model loads -------------------------------------------------------
    print(f"[a] loading gpt2 on device={DEVICE} ...", flush=True)
    model = HookedTransformer.from_pretrained("gpt2", device=DEVICE)
    print(f"[a] OK  ({model.cfg.n_layers} layers, d_model={model.cfg.d_model})")

    # Position alignment is a precondition for (d); fail loudly if broken.
    verify_alignment(model)
    print("[a] prompt tokenization aligned (single-token names, equal lengths)")

    prompts = build_prompts()
    p0 = prompts[0]

    # (b) one forward pass --------------------------------------------------
    clean_tokens = model.to_tokens(p0.clean)
    _ = model(clean_tokens)
    print(f"[b] OK  forward pass on {clean_tokens.shape[1]} tokens: {p0.clean!r}")

    # (c) logit diff positive on clean prompts ------------------------------
    diffs = []
    for p in prompts:
        io_id = model.to_single_token(p.io)
        s_id = model.to_single_token(p.s)
        diffs.append(logit_diff(model, model.to_tokens(p.clean), io_id, s_id))
    mean_diff = sum(diffs) / len(diffs)
    print(f"[c] per-prompt logit diffs: {[round(d, 3) for d in diffs]}")
    print(f"[c] mean clean logit diff (IO - S) = {mean_diff:+.4f}")
    if mean_diff <= 0:
        print("[c] FAIL: expected positive logit diff on clean prompts")
        return 1
    print("[c] OK  model prefers the indirect object on clean prompts")

    # (d) one activation patch, end to end ----------------------------------
    io_id = model.to_single_token(p0.io)
    s_id = model.to_single_token(p0.s)
    clean_tokens = model.to_tokens(p0.clean)
    corrupted_tokens = model.to_tokens(p0.corrupted)

    # Cache clean activations, then patch one site into the corrupted run.
    _, clean_cache = model.run_with_cache(clean_tokens)
    layer, pos = 9, -1                      # arbitrary single site; plumbing only
    hook_name = f"blocks.{layer}.hook_resid_pre"
    clean_act = clean_cache[hook_name]

    corrupted_diff = logit_diff(model, corrupted_tokens, io_id, s_id)
    patched_logits = model.run_with_hooks(
        corrupted_tokens,
        fwd_hooks=[(hook_name, partial(patch_resid_at_pos, clean_act=clean_act, pos=pos))],
    )
    patched_diff = (patched_logits[0, -1, io_id] - patched_logits[0, -1, s_id]).item()
    print(f"[d] corrupted logit diff           = {corrupted_diff:+.4f}")
    print(f"[d] patched  logit diff (L{layer}, pos {pos}) = {patched_diff:+.4f}")
    print(f"[d] OK  activation patch returned a number (delta {patched_diff - corrupted_diff:+.4f})")

    print("\nALL CHECKS PASSED -- environment is ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
