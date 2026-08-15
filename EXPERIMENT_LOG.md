# Experiment log

A record of exploratory runs against the IOI dataset in this repository. **This is a
lab notebook, not a results document.** Everything below was produced by a throwaway
analysis harness, not by the `path_patching.py` published here, and none of it has
been through a faithfulness / completeness / minimality check. Treat the numbers as
observations that motivated the next piece of work, not as claims.

Raw outputs (~1.2 GB of tensors, plus figures) are kept local and deliberately not
committed.

---

## Setup

Common to everything below unless stated otherwise.

| | |
|---|---|
| Model | GPT-2 small, CPU, offline weights |
| Dataset seed | `base_seed = 20260716` |
| Orderings | both `IO_S1_S2` and `S1_IO_S2`, pooled |
| Prompt count | 200 for `small` templates, 25 for `large` |
| Metric | `logit[IO] - logit[S]` at the final position |
| Clean baseline | **+3.417** (small, 200 prompts) |

Note the dataset config differs from the published default: these runs used
`NUMBER_OF_BATCHES = 20`, where `dataset.py` currently ships `5`. The `large`
templates were subsampled to 25 prompts because they carry ~200 tokens of padding
between the name mentions and are correspondingly expensive on CPU.

---

## The overnight sweep — 2026-08-12, 8.39 h

A full sweep over every node and every receiver channel, chunked and logged.

**Node sweeps** (12 configurations): `corrupt` / `negative` / `scrambled` ×
`small` / `large` × `direct_effect` / `total_effect`.

**Path sweeps** (18 configurations): `corrupt` / `negative` / `scrambled` ×
`small` / `large` × `q_input` / `k_input` / `v_input`.

135 units recorded in `manifest.jsonl` — 54 node, 81 path — across 30 distinct
configurations. Fifteen configurations ran as a single unit at full prompt count; the
other fifteen were split into 8 chunks each. Each logged chunk covered 144 nodes with
12 skipped, at roughly 166 s per chunk. Every unit carries its own provenance record
(seed, orderings, size, prompt count, model).

Two further `denoise_small` node runs (direct and total effect) were added on
2026-08-14, after the sweep, and are on disk but absent from the manifest.

The cost is the finding here: **8.4 hours for one pass.** That is the constraint that
makes the interface worth getting right before re-running, and it is why the
aggregation layer is the next thing to build rather than another sweep.

---

## Targeted probes

Small, focused runs at 200 prompts on `small` templates. These are the ones with
numbers worth recording.

### One hop into the Name Movers

Patching each sender into the residual input of the three heads at
**L9H6, L9H9, L10H0**, measuring the change in `logit[IO] - logit[S]`.

| sender | effect |
|---|---|
| **L8H6** | **+2.238** |
| **L8H10** | **+1.069** |
| **L7H9** | **+0.667** |
| **L7H3** | **+0.414** |
| L9H6 | −0.456 |
| L9H9 | −0.165 |

Four heads account for essentially all of it, and they sit two layers upstream of the
receivers. Everything below L7H3 is under 0.15 — MLP3 at +0.124 and MLP5 at +0.116
are the largest of the remainder.

### Which channel carries it

The same four senders, split by receiver channel. This is the sharpest result in the
log.

| sender | q_input | k_input | v_input |
|---|---|---|---|
| L8H6 | **2.234** | 0.001 | 0.000 |
| L8H10 | **1.053** | 0.007 | −0.002 |
| L7H9 | **0.671** | −0.006 | 0.001 |
| L7H3 | **0.418** | −0.004 | −0.000 |

The effect is carried entirely by the **query**. Keys and values are at noise level —
three orders of magnitude smaller. Mechanistically: these senders change *what the
receiver head is looking for*, not what it finds or what it copies.

Patching into just the three receivers reproduces the full downstream effect to
within 0.005 (`L8H6`: 2.238 via the movers alone vs 2.243 unrestricted), so at this
depth the three heads are the whole story.

### Two hops

Patching each sender into **L8H6, L8H10, L7H9, L7H3**, then measuring the effect at
the Name Movers.

| sender | effect |
|---|---|
| **L5H5** | **+2.774** |
| L3H0 | +0.806 |
| L6H9 | +0.711 |
| MLP5 | +0.677 |
| MLP3 | −0.277 |
| L6H6 | −0.236 |

One head, L5H5, dominates the second hop as decisively as L8H6 dominated the first.

### Attention to S2

Baseline attention paid to the S2 position (sequence position 10) by each of the four
middle-layer senders:

| head | attention to S2 |
|---|---|
| L8H6 | 0.777 |
| L7H9 | 0.380 |
| L8H10 | 0.342 |
| L7H3 | 0.138 |

Sweeping what changes that attention, **L5H5 is the largest positive contributor for
all four**: L8H6 +0.103, L8H10 +0.087, L7H9 +0.062, L7H3 +0.037. That is the same
head that dominates the two-hop result, reached by a different measurement. MLP0 is
the largest negative contributor for L7H9 (−0.092).

### Residual stream sanity

Residual norm grows monotonically with depth across the 13 measured points, 5.17 →
537.58, with the jump concentrated in the last layer. A decomposition identity check
drifts by 2.6e−07, so the per-component split is exact to float precision.

---

## Reading this against the paper

The heads that fall out above — L9H6, L9H9, L10H0 as the output-writing group; L7H3,
L7H9, L8H6, L8H10 one hop upstream acting through the query; L5H5 and L3H0 upstream
of those — line up with the roles Wang et al. describe as Name Movers, S-Inhibition
heads, and the induction / duplicate-token machinery feeding them.

That correspondence is an encouraging sign, not a replication. Nothing here has been
validated for faithfulness, completeness, or minimality, the sweeps were exploratory,
and the `large` runs were at 25 prompts. Establishing that a recovered head set
actually *is* the circuit is the work that comes next.

---

## What this log argues for next

1. The aggregation layer — turning a sweep of patched runs into ranked effects — is
   the bottleneck, not more compute.
2. Any re-run should be worth 8+ hours before it starts.
3. The query-channel result is specific enough to be worth confirming through the
   published `path_patching.py`, as a cross-check on both implementations.
