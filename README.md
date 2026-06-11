# IOI Replication — Name-Mover Heads in GPT-2 Small

A learning-in-public replication of one component of
[Wang et al. (2022), *Interpretability in the Wild*](https://arxiv.org/abs/2211.00593):
the **name-mover heads** in the indirect object identification (IOI) circuit, via
logit difference and activation patching. Built on
[TransformerLens](https://github.com/TransformerLensOrg/TransformerLens).

> **Provenance note:** The initial scaffold in this repo was generated with Claude
> Code. I'm working through it piece by piece — predicting behavior before running,
> modifying, and ultimately rewriting the core experiment code by hand. The commit
> history documents that progression. The final state I'm working toward: every line
> of the core pipeline (prompt generation, logit-diff metric, activation caching,
> patching loop) written and owned by me.

Part of a longer mechanistic interpretability research program. Writeup to follow.

**Status:** environment set up and verified. The activation-patching experiment
itself lives in [`notebooks/ioi_replication.ipynb`](notebooks/ioi_replication.ipynb).

## Setup

Requires Python 3.10–3.12 (developed on 3.11). On Apple Silicon, PyTorch's MPS
backend works, but GPT-2 small is small enough that CPU is the default — set
`IOI_DEVICE=mps` to opt in.

```bash
python3.11 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt   # or requirements.lock.txt for an exact env
./.venv/bin/python download_weights.py                  # caches GPT-2 (~500MB), once, online
```

Opening this folder in VSCode auto-selects `.venv` and activates it in new
terminals (see [.vscode/settings.json](.vscode/settings.json)).

## Verify the setup

```bash
./.venv/bin/python smoke_test.py        # the "setup is done" criterion
./.venv/bin/python verify_offline.py    # proves the cached weights load with no network
```

`smoke_test.py` is a real pass/fail gate. It checks, exiting non-zero on the
first failure:

- **(a)** GPT-2 small loads via TransformerLens.
- **(b)** One forward pass on an IOI prompt runs (no dtype/device errors).
- **(c)** The logit difference `logit[IO] - logit[S]` is **positive** on clean
  prompts (the model prefers the indirect object).
- **(d)** One activation patch (single layer + position) runs end to end and
  returns a number. *This is a plumbing check, not the experiment.*

Expected output ends with `ALL CHECKS PASSED -- environment is ready.`

## Files

| File | Purpose |
|------|---------|
| [`ioi_prompts.py`](ioi_prompts.py) | Clean + ABC-corrupted prompt pairs with single-token names and verified position alignment. |
| [`smoke_test.py`](smoke_test.py) | Executable setup criterion (checks a–d above). |
| [`download_weights.py`](download_weights.py) | Cache GPT-2 weights while online. |
| [`verify_offline.py`](verify_offline.py) | Prove the cached weights load offline. |
| [`notebooks/ioi_replication.ipynb`](notebooks/ioi_replication.ipynb) | The replication workspace. |
| `requirements.txt` / `requirements.lock.txt` | Pinned top-level deps / full lock. |

## A note on tokenization

IOI activation patching is only valid if a position index means the same thing
in the clean and corrupted prompts. Two guards enforce this in
[`ioi_prompts.py`](ioi_prompts.py):

- every name is a **single GPT-2 token** (re-checked against the live tokenizer,
  not assumed — and note TransformerLens prepends a BOS token, which is easy to
  miscount), and
- each clean prompt and its corrupted counterpart tokenize to the **same
  length**.

`verify_alignment(model)` raises loudly if either is violated, so a misaligned
prompt fails fast instead of silently producing garbage patch results.
