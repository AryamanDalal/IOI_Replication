# IOI Replication

A from-scratch replication of **"Interpretability in the Wild: A Circuit for Indirect
Object Identification in GPT-2 Small"** (Wang et al., 2022), built on
[TransformerLens](https://github.com/TransformerLensOrg/TransformerLens).

The goal is not to re-run the authors' code. It is to rebuild the circuit-discovery
pipeline — dataset, direct logit attribution, path patching, circuit validation —
independently, and see which of the paper's findings survive contact with my own
implementation.

GPT-2 small, CPU only, offline weights.

## The task

Indirect object identification. Given

> When **Mary** and **John** went to the store, **John** gave a drink to ___

the model should prefer `" Mary"` (the indirect object, **IO**) over `" John"` (the
repeated subject, **S**). The readout throughout is the **logit difference**
`logit[IO] - logit[S]` at the final position — a signed, per-prompt scalar that
stays interpretable under ablation and patching.

## Pipeline

| stage | file | state |
|---|---|---|
| Dataset construction | `dataset.py` | done |
| Tokenizer / alignment verification | `dataset_verification.py` | done |
| Forward passes + cache records | `process_dataset.py` | done |
| Direct logit attribution | `direct_logit_attribution.py` | done |
| Path patching | *(not published)* | **~half done** |
| Circuit validation | — | not started |

### `dataset.py` — the prompt set

Builds a dict keyed by `(template_ordering, template_size)`, each holding four
prompt variants that share a name triplet and a sentence frame:

- **clean** — IO appears once, S twice (the IOI task proper)
- **corrupt** — ABC: three distinct names, so no name is repeated
- **negative** — clean with the IO and S roles exchanged
- **scrambled** — same names, same length, but the tokens between the second name
  and the final clause are shuffled into word salad

Two orderings (`IO_S1_S2`, `S1_IO_S2`) cross with two sizes (`small`, `large`).
`large` pads ~200 tokens of distractor text between the name mentions to test
whether the circuit's behaviour is distance-dependent.

The design constraint that makes everything downstream tractable: **within a size,
every template tokenizes to the same length, and the name tokens land at identical
sequence positions.** That is what allows caches from different prompts to be
stacked and compared position-by-position. Names are single-token under GPT-2's
BPE with a leading space, and sampled in reproducible batches from a fixed seed.

### `dataset_verification.py` — proving the constraint holds

Asserts what `dataset.py` assumes rather than trusting it: every name is a single
token, and every sentence of a given size tokenizes to the same length with names
at the same positions. Also the single place the model is configured —
`set_use_attn_result`, `set_use_split_qkv_input`, `set_use_hook_mlp_in`, all
required for per-head decomposition and for patching individual Q/K/V inputs later.

### `process_dataset.py` — one forward pass, reused everywhere

Runs each prompt once and freezes the result into a `Run_Details` record: the
prompt, the `ActivationCache`, the logits, the IO/S token ids, and the logit
difference.

This exists because the analyses downstream are *not* independent. Attribution,
patching, and validation all want the same caches, and re-running the model per
analysis is both slow on CPU and a correctness hazard — a stale or mismatched
cache is invisible until the numbers quietly disagree. Computing once and passing
records around makes cache provenance explicit.

### `direct_logit_attribution.py` — who writes the answer?

Decomposes the final residual stream into its writers via
`get_full_resid_decomposition` and projects each onto the IO−S unembedding
direction, giving a per-component contribution to the logit difference.

Components: 144 attention heads (12 layers × 12), 12 MLPs, token embeddings,
positional embeddings, and biases. A **faithfulness check** confirms the
decomposition sums back to the measured logit difference — if the parts don't
reconstruct the whole, the attribution is meaningless and everything after it is
too.

Outputs five figures: composition breakdown, a 12×12 per-head heatmap, top heads
ranked, per-MLP contributions, and a layerwise cumulative trace.

## Path patching — halfway

**Not in this repository yet.** It exists locally as a working notebook and is
roughly half finished; it will be published once the interface settles.

Direct logit attribution answers *who writes to the output*. It cannot answer *who
feeds whom* — a head with near-zero direct contribution may still be load-bearing
because it supplies the input another head depends on. Path patching isolates the
causal edge between a sender and a receiver by patching the sender's contribution
along one path while freezing everything else, so the measured change is
attributable to that path alone.

Current state:

- **Working** — patch/freeze pairing over `Run_Details`, patching head outputs
  (`z`) and MLP outputs, and the direct-effect vs total-effect distinction
- **In progress** — the receiver-side hooks (`q_input` / `k_input` / `v_input`)
  and validation of the patch specification
- **Not started** — indirect-effect composition across a sender→receiver chain,
  and the statistics layer aggregating patched vs unpatched logit differences

The expensive part is that a full per-head sweep is ~27 minutes per effect on CPU,
so the interface needs to be right before the sweep is worth running.

Once path patching lands, the remaining work is circuit validation: faithfulness,
completeness, and minimality of the recovered head set against the paper's.

## Running it

```bash
python dataset_verification.py       # verify tokenization and alignment
python process_dataset.py            # build the cached run records
python direct_logit_attribution.py   # attribution + the five figures
```

Weights are read from a local Hugging Face cache; `HF_HUB_OFFLINE` and
`TRANSFORMERS_OFFLINE` are set at import time, so no network access is required
or attempted.

## Reference

Wang, K., Variengien, A., Conmy, A., Shlegeris, B., & Steinhardt, J. (2022).
*Interpretability in the Wild: a Circuit for Indirect Object Identification in
GPT-2 small.* [arXiv:2211.00593](https://arxiv.org/abs/2211.00593)
