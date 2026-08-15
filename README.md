# IOI Replication

A from-scratch replication and extension of **"Interpretability in the Wild: A
Circuit for Indirect Object Identification in GPT-2 Small"** (Wang et al., 2022),
built on [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens).

This is not a walkthrough of the paper and not a re-run of the authors' code. It is
an independent rederivation of the IOI circuit, built to produce two things:

1. **A general framework for identifying and characterizing a circuit from the
   ground up** — a sequence of measurements where each stage is motivated by the
   specific question the previous stage could not answer, and where the
   decomposition is verified to be exact at every step.
2. **An answer to a question the original paper leaves open:** is the IOI circuit a
   property of the *task*, or of the *prompt templates* it was discovered on?

Along the way, it tests which of the paper's findings survive contact with an
independent implementation.

GPT-2 small, CPU only, offline weights.

## The task

Indirect object identification. Given

> When **Mary** and **John** went to the store, **John** gave a drink to ___

the model should prefer `" Mary"` (the indirect object, **IO**) over `" John"` (the
repeated subject, **S**). The readout throughout is the **logit difference**
`logit[IO] - logit[S]` at the final position — a signed, per-prompt scalar that stays
interpretable under ablation and patching.

The dataset is built so that, within a template size, every prompt tokenizes to the
same length and the name tokens land at identical sequence positions. That constraint
is what makes caches from different prompts stackable and comparable position by
position, and it is asserted rather than assumed.

## The framework

The core methodological arc is a causal-mediation story, rebuilt from first
principles rather than reproduced from the paper's code:

- **Direct logit attribution** decomposes the prediction into one signed
  contribution per head, surfacing candidate heads — but it is correlational.
- **Direct effects** (path patching each head straight to the logits) confirm the
  candidates causally.
- **Total effects** (activation patching) reveal a larger set of heads and a gap:
  heads whose influence doesn't reach the logits directly. The circuit is not a
  list of heads; it is heads talking to each other.
- **Per-head ΔDLA between conditions** localizes exactly where each upstream head's
  influence lands downstream, with a conservation check — the ΔDLA contributions
  must sum to the direct/total gap, so the decomposition is provably complete
  rather than assumed.
- **Q/K/V decomposition** names the mechanism of each interaction: an upstream
  head's effect on a downstream head is pinned to the query, key, or value route it
  travels through, with the joint three-path patch as the ceiling (the routes
  interact through the attention softmax and are not additive).
- **Dedicated ablation tests** then establish faithfulness, completeness, and
  minimality — properties the narrative above motivates but does not prove.

Each stage is designed to ship with diagrams — what is attended to, where effects
flow — because a circuit claim should be inspectable, not just reported.

None of this machinery is IOI-specific. The prompt harness, the patching classes,
the attribution and mediation tooling, and the validation suite are being built so
the same sequence of measurements — and the same diagrams — can be re-run on a
different model, a different task, or the same model at a different point in
training. Building the instrument once and reusing it is the point of the design.

## The research questions

Wang et al. discovered the circuit on a narrow family of hand-written templates.
This project runs the full framework across a systematically varied prompt suite —
scrambled context, shorter and longer prompts, varied name pools — and measures
circuit survival at the *circuit* level, not just the behavioral level: do the same
heads keep the same roles, the same attention signatures, the same mediation
structure? The underlying question is what the circuit is actually keying on — the
sentence's structure, or the names themselves and their positional rhythm. The two
possibilities imply very different circuits, and the sweep is designed to tell them
apart rather than to confirm either.

A second question sits one level inside the interactions themselves: **through which
channel does each upstream head act on the head it feeds?** An influence arriving at
a receiver's query changes *what that head looks for*; at its key, *what can be
found*; at its value, *what gets copied once found*. These are different mechanisms,
and the paper's account implies a specific answer — the S-inhibition heads should
act on the name movers' queries. Preliminary exploratory runs (see the
[experiment log](EXPERIMENT_LOG.md)) find exactly that, with key- and value-route
effects at noise level, three orders of magnitude below the query route. Confirming
that channel assignment through the published pipeline and its validation layer, and
then testing whether the assignment itself survives the prompt sweep — whether the
*routing* of the circuit is as stable as its membership — is the second axis of the
circuit-survival question.

## Rigor

Much of this work is exploratory — probing a mechanism to see what it does, not
betting on an outcome — and it is reported as such. Where an outcome *is*
confidently expected, the expectation is encoded as an assertion in the code so a
violation fails loudly instead of being explained away. Every reported number
carries where it was measured, measured results are distinguished from inferred
ones, and divergences from the paper's values are logged rather than reconciled
away. Structural invariants — tokenization, position alignment, frozen-run no-ops —
are asserted rather than assumed, and the DLA decomposition is gated by a
faithfulness check confirming the per-component contributions reconstruct the
measured logit difference. Measured values for the known name-mover heads serve as
regression anchors as the code evolves.

## Authorship

The experimental design and the logic of this codebase are mine: the algorithms,
the hooks, the assertions, and the architecture of every module were designed and
first written by me.

Claude Code was used, always against logic I had already specified, in a few
bounded ways. It edited my code for clarity — restructuring what I had written
into cleaner form while preserving its behavior. It wrote occasional small helpers
(two or three lines, one or two per file at most) implementing steps I had already
defined. Under my direct oversight it wrote the mechanical validation code in
`Patch_Dict_Verification` and the matplotlib charting code. And it drafted
documentation prose, including [`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md), against
numbers it extracted and verified from the saved artifacts rather than recalled.

Where behavior-preservation mattered it was checked rather than assumed — polish
passes were verified against the prior revision (identical output hashes;
assertion and success paths cross-checked on matched specifications) — and those
checks are recorded in the commit messages. Earlier commit messages state this
division of labor more coarsely; this section is the authoritative version.

## Where this is

The measurement pipeline is built and published. The analysis layer on top of it is
not finished.

**Working end to end.** Dataset construction and its tokenizer/alignment
verification; a forward-pass layer that runs each prompt once and freezes the result
into a reusable record; direct logit attribution with a faithfulness check that
confirms the per-component contributions reconstruct the measured logit difference.

**Working, newly published.** Path patching: specification validation, whole-node
patching of head and MLP outputs, sender→receiver edge patching into a receiver's
Q/K/V or MLP input, and the direct-effect vs total-effect distinction. This is the
part that answers *who feeds whom* rather than *who writes the answer* — a head with
near-zero direct contribution can still be load-bearing because it supplies the
input another head depends on.

**Not built yet.** The aggregation layer that turns a sweep of patched runs into
ranked effects, and circuit validation proper — faithfulness, completeness, and
minimality of a recovered head set against the paper's.

So the published code can currently *perform* an intervention and hand back the
patched runs, but does not yet score a circuit for you.

## What has actually been run

Exploratory sweeps have been run against this dataset, including an 8.4-hour
overnight pass over every node and receiver channel. Those runs used a separate
throwaway harness, **not** the `path_patching.py` published here, and their outputs
are deliberately not in this repository — they are 1.2 GB of tensors.

[`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md) records what was run and what it showed,
with the measured numbers. Read it as a lab notebook: the runs were exploratory,
some at small prompt counts, and nothing in it has been through the validation layer
described above. It is a log, not a result.

## Running it

```bash
python dataset_verification.py       # verify tokenization and alignment
python process_dataset.py            # build the cached run records
python direct_logit_attribution.py   # attribution + five figures
```

`path_patching.py` is importable and its classes are usable, but its `main()` only
assembles the freeze/patch pairs — there is no end-to-end runner yet, pending the
aggregation layer.

Weights are read from a local Hugging Face cache; `HF_HUB_OFFLINE` and
`TRANSFORMERS_OFFLINE` are set before TransformerLens is imported, so no network
access is required or attempted.

## Where this goes

The cross-model leg is the second axis of the same question — whether the answer
survives a change of substrate, not just a change of prompt. One level deeper sits a
question this project sets up but does not answer: whether a name mover's
selectivity lives in *where it attends* (QK) or in *what it copies once attending*
(OV) — measurable via OV coverage across name vs. non-name tokens, and via OV-swap
interventions with attention frozen. Beyond that, the framework is built to support
a natural research program it does not yet execute: the circuit after fine-tuning,
across training checkpoints, and on other tasks entirely. This repo is the first
project in a longer mechanistic interpretability program working toward decoding
internal model representations into explicit relational structure; the instrument
built here is its foundation.

## Reference

Wang, K., Variengien, A., Conmy, A., Shlegeris, B., & Steinhardt, J. (2022).
*Interpretability in the Wild: a Circuit for Indirect Object Identification in
GPT-2 small.* [arXiv:2211.00593](https://arxiv.org/abs/2211.00593)