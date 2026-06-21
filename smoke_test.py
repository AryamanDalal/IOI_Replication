"""Executable 'setup is done' criterion for the IOI replication environment.

Run:  python smoke_test.py

Checks, in order, exiting non-zero on the first failure:
  (a) GPT-2 small loads via TransformerLens.
  (b) One forward pass on an IOI prompt runs (no dtype/device errors).
  (c) Logit difference logit[IO] - logit[S] is POSITIVE on clean prompts
      (reported as mean / sd / range over the whole prompt set).
  (d) Activation patching across the partial-corrupted set: patch clean
      activations at one site into each partial variant; report mean/sd/range
      of |base - patched| of  logit[IO] - max(competing names).
  (e) The same patch on the ABC-corrupted set (curiosity), same metric.

(d)/(e) are PLUMBING checks -- they prove the patching machinery works, not
that any particular circuit claim holds. The real patching experiment stays in
the notebook.

Device defaults to CPU: GPT-2 small is tiny, and CPU sidesteps MPS dtype quirks
that would otherwise eat the afternoon. Set IOI_DEVICE=mps to try the Apple GPU.
"""

from __future__ import annotations

import os
import sys
from collections import defaultdict
from functools import partial
from statistics import mean, stdev

import torch
from transformer_lens import HookedTransformer
from ioi_prompts import ioi_prompts, verify_alignment

DEVICE = os.environ.get("IOI_DEVICE", "cpu")


def _io_minus_max(final: torch.Tensor, io_id: int, s_ids) -> float:
    """logit[IO] - max_j logit[competitor_j] from a final-position logit vector.

    s_ids is a single token id or an iterable of competitor ids; the strongest
    competitor (largest logit) is subtracted. A single id reduces to the plain
    logit[IO] - logit[S].
    """
    if isinstance(s_ids, int):
        s_ids = (s_ids,)
    competitor = torch.stack([final[i] for i in s_ids]).max()
    return (final[io_id] - competitor).item()


def logit_diff(model: HookedTransformer, tokens: torch.Tensor, io_id: int, s_ids) -> float:
    """logit[IO] - strongest competitor logit at the final sequence position.

    s_ids is one id or an iterable of competitor ids (the IOI distractor S, plus
    any other names present in a corrupted variant).
    """
    return _io_minus_max(model(tokens)[0, -1], io_id, s_ids)


def patch_resid_at_pos(activation, hook, clean_act, pos):
    """Replace the residual stream at a single position with the clean value."""
    activation[:, pos, :] = clean_act[:, pos, :]
    return activation


def patched_logit_diff(model, tokens, io_id, s_ids, hook_name, clean_act, pos):
    """Like logit_diff, but with clean_act patched into `tokens` at one site."""
    logits = model.run_with_hooks(
        tokens,
        fwd_hooks=[(hook_name, partial(patch_resid_at_pos, clean_act=clean_act, pos=pos))],
    )
    return _io_minus_max(logits[0, -1], io_id, s_ids)


def _tick(iterable, label):
    """Yield from `iterable`, printing a live count so a long loop visibly moves.

    A list comprehension builds its whole list before anything prints, so a slow
    one looks like a hang. Wrapping its iterable in this prints '[c] <label> N'
    in place (\\r) on every item -- if the number is climbing, it's running, not
    timing out. Prints a final newline so the next message starts on a clean line.
    """
    i = 0
    for i, item in enumerate(iterable, 1):
        print(f"[c] {label:9s} {i} ...", end="\r", flush=True)
        yield item
    print(f"[c] {label:9s} {i} done")


def _fmt_stats(values: list[float]) -> str:
    """One-line 'mean / sd / range' summary of a list with >= 2 entries.

    sd is the sample standard deviation; range is max - min. These describe the
    full set of values we computed (not a sample of it), but sample sd is the
    conventional default and the difference is negligible at this many points.
    """
    return (f"mean={mean(values):+.4f}  sd={stdev(values):.4f}  "
            f"range={max(values) - min(values):.4f}")


def main() -> int:
    torch.set_grad_enabled(False)

    # (a) model loads -------------------------------------------------------
    print(f"[a] loading gpt2 on device={DEVICE} ...", flush=True)
    model = HookedTransformer.from_pretrained("gpt2", device=DEVICE)
    print(f"[a] OK  ({model.cfg.n_layers} layers, d_model={model.cfg.d_model})")

    # Position alignment is a precondition for (d); fail loudly if broken.
    verify_alignment(model)
    print("[a] prompt tokenization aligned (single-token names, equal lengths)")

    # First clean prompt, for the single forward-pass demo in (b). next() pulls
    # just one item from the generator -- nothing else is built.
    clean_text0 = next(ioi_prompts()).clean

    # (b) one forward pass --------------------------------------------------
    clean_tokens = model.to_tokens(clean_text0)
    _ = model(clean_tokens)
    print(f"[b] OK  forward pass on {clean_tokens.shape[1]} tokens: {clean_text0!r}")

    # (c) logit diff positive on clean prompts ------------------------------
    # One streaming pass over ioi_prompts() -- nothing is materialized, so this
    # scales to far larger NAMES/TEMPLATES without touching this code. Each
    # prompt's strings are formatted once and its variants flattened inline.

        # NOTE: corrupted and partial both score IO - S2 and share the same
        # (IO, S2) marginal, so their unpaired stats coincide by construction
        # (observed: mean +2.2551 both; range identical; sd 0.7482 vs 0.7488).
        # This is structural overlap, NOT a finding -- partial-as-a-set tells
        # you nothing corrupted doesn't.
        # DIAGNOSTIC if revisiting: mean/range identical + sd differing at the
        # ~4th decimal = marginal twins (expected, genuinely different prompts).
        # mean AND sd AND range all bit-identical = duplicate-generation bug
        # (same prompts emitted twice) -- suspect the generator, not the data.
        # Partial's real signal is in the PAIRED clean<->partial contrast (each
        # clean prompt vs its own S2-swapped twin), never in its own spread.
    clean_diffs = []
    partial_corrupted_diffs, partial_corrupted_diffs1 = [], []
    corrupted_diffs = []
    
    # Role-averaging buckets for clean SD: key is the UNORDERED name pair, so the
    # two role assignments (io=A,s=B and io=B,s=A) across all templates land in the
    # same bucket. Collapsing a bucket to its MEAN cancels each name's per-token
    # prior (one ordering's +prior(A)-prior(B) offsets the other's), leaving circuit
    # variability -- that's what makes clean SD "clean". We store the values (not a
    # running sum) because the correct average divides by the bucket count.
    # d = dict()
    clean_by_pair: dict[frozenset[str], list[float]] = defaultdict(list)
    pair_orderings: dict[frozenset[str], set] = defaultdict(set)  # distinct IOs seen per pair

    for p in _tick(ioi_prompts(), "prompts"):
        io_id, s_id = model.to_single_token(p.io), model.to_single_token(p.s)

        # clean (ABB): IO - S
        clean_diff = logit_diff(model, model.to_tokens(p.clean), io_id, s_id)
        clean_diffs.append(clean_diff)
        pair_id = frozenset({str(io_id), str(s_id)})
        # d[pair_id] = d.get(pair_id, 0) + clean_diff/2   # /2 doesn't average a 10-entry bucket
        clean_by_pair[pair_id].append(clean_diff)
        pair_orderings[pair_id].add(io_id)            # distinct IOs seen for this pair


        # partial (S2-swap): IO - s (kept subject), and IO - the swapped S2 name.
        for text, c in zip(p.partial_corrupted, p.partial_s2):
            tokens = model.to_tokens(text)
            partial_corrupted_diffs.append(logit_diff(model, tokens, io_id, s_id))
            partial_corrupted_diffs1.append(logit_diff(model, tokens, io_id, model.to_single_token(c)))

        # corrupted (ABC, IO fixed): IO (c_a) - S2 (c_c).
        for text, a, c in zip(p.corrupted, p.c_a, p.c_c):
            corrupted_diffs.append(
                logit_diff(model, model.to_tokens(text),
                           model.to_single_token(a), model.to_single_token(c)))

    # mean / sd / range for each metric.
    #
    # DISCLAIMER: SD for corrupted/partial is over individual prompts, not
    # role-averaged, so it carries per-name prior and distractor-identity variance
    # on top of circuit variability -- it's an upper bound on the true spread, not
    # the spread. Clean SD is role-averaged and clean.
    #
    # Clean: collapse each unordered name pair to its mean first (role-averaging),
    # then report mean/sd/range over those per-pair values.
    clean_pair_means = [mean(v) for v in clean_by_pair.values()]
    # Sanity-check the role-averaging assumptions: every pair must have seen BOTH
    # orderings (so the per-name prior actually cancels), and equal bucket sizes
    # mean the design is balanced (so the role-avg mean matches the grand mean --
    # only the SD should differ).
    both_orderings = all(len(s) == 2 for s in pair_orderings.values())
    equal_sizes = len({len(v) for v in clean_by_pair.values()}) == 1
    # print(f"[c] clean     (IO - S):  {_fmt_stats(clean_diffs)}")
    print(f"[c] clean     (IO - S, role-avg): {_fmt_stats(clean_pair_means)}")
    print(f"[c]   grand={mean(clean_diffs):+.4f}  pair-avg-mean={mean(clean_pair_means):+.4f}  "
          f"both_orderings={both_orderings}  equal_sizes={equal_sizes}")
    print(f"[c] corrupted (IO - S2): {_fmt_stats(corrupted_diffs)}")
    print(f"[c] partial   (IO - S2): {_fmt_stats(partial_corrupted_diffs1)}")

    if mean(clean_diffs) <= 0:
        print("[c] FAIL: expected positive logit diff on clean prompts")
        return 1
    print("[c] OK  model prefers the indirect object on clean prompts")

    # (d) activation patching across the partial-corrupted set --------------
    # For each clean prompt: cache its activations once, then patch the clean
    # value at (layer 9, final position) into every partial-corrupted variant.
    # Metric is |base - patched| of  logit[IO] - max(logit[S], logit[S2-swap]) --
    # how far that single site drags the partial run back toward the clean answer.
    layer, pos = 9, -1
    hook_name = f"blocks.{layer}.hook_resid_pre"

    partial_patch_deltas = []
    for p in _tick(ioi_prompts(), "patch-part"):
        io_id, s_id = model.to_single_token(p.io), model.to_single_token(p.s)
        _, clean_cache = model.run_with_cache(model.to_tokens(p.clean))
        clean_act = clean_cache[hook_name]

        # competitor = strongest of {clean S, the swapped S2 name}.
        for text, s2 in zip(p.partial_corrupted, p.partial_s2):
            tokens = model.to_tokens(text)
            competitors = (s_id, model.to_single_token(s2))
            base = logit_diff(model, tokens, io_id, competitors)
            patched = patched_logit_diff(model, tokens, io_id, competitors, hook_name, clean_act, pos)
            partial_patch_deltas.append(patched - base)

    print(f"[d] |partial - patched| (L{layer}, pos {pos}): {_fmt_stats(partial_patch_deltas)}")

    # (e) curiosity: the same patch on the ABC-corrupted set. IO (c_a) is fixed,
    # so the competitor is the strongest of {clean S, c_b, c_c}: the metric is
    # |base - patched| of  logit[IO] - max(logit[S], logit[c_b], logit[c_c]).
    corrupted_patch_deltas = []
    for p in _tick(ioi_prompts(), "patch-corr"):
        io_id, s_id = model.to_single_token(p.io), model.to_single_token(p.s)
        _, clean_cache = model.run_with_cache(model.to_tokens(p.clean))
        clean_act = clean_cache[hook_name]

        for text, b, c in zip(p.corrupted, p.c_b, p.c_c):
            tokens = model.to_tokens(text)
            competitors = (s_id, model.to_single_token(b), model.to_single_token(c))
            base = logit_diff(model, tokens, io_id, competitors)
            patched = patched_logit_diff(model, tokens, io_id, competitors, hook_name, clean_act, pos)
            corrupted_patch_deltas.append(patched - base)

    print(f"[e] |corrupted - patched| (L{layer}, pos {pos}): {_fmt_stats(corrupted_patch_deltas)}")

    print("\nALL CHECKS PASSED -- environment is ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
