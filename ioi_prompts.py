"""IOI (Indirect Object Identification) prompt set.

Builds clean prompts and their ABC-corrupted counterparts with the
position-alignment guarantees that activation patching depends on:

  * Every name is a SINGLE GPT-2 token (verified at runtime, not assumed).
  * A clean prompt and its corrupted counterpart tokenize to the SAME length,
    so a position index means the same thing in both. Misaligned positions are
    the classic silent bug that turns patching into noise.

Task structure (Wang et al., 2022, "Interpretability in the Wild"):

  Clean (template ABB):
    "When Mary and John went to the store, John gave a drink to" -> " Mary"
      IO (indirect object) = Mary   <- the correct answer
      S  (subject)         = John   <- the distractor; appears twice

  ABC-corrupted:
    "When Tom and Mike went to the store, Alex gave a drink to" -> (no signal)
      All three name slots get DISTINCT names, so no token is duplicated and
      the duplicate-token / induction signal the circuit relies on is gone.
      This is the baseline you patch *from* when localizing the circuit.

The logit-difference metric is  logit[IO] - logit[S]  at the final position:
positive means the model prefers the indirect object, as it should on a clean
prompt. That single scalar is what activation patching moves around.
"""

from __future__ import annotations

from dataclasses import dataclass


# Names vetted to be single GPT-2 tokens *with a leading space* (the form they
# take mid-sentence). verify_single_token() re-checks this against the live
# tokenizer so a model/tokenizer change can't silently break alignment.
NAMES: list[str] = [
    "Mary", "John", "Tom", "James", "Dan", "Paul", "Mark", "Mike",
    "Alex", "Kevin", "Anna", "Laura", "Sarah", "Scott", "Jack", "Ryan",
]

# Same-length templates. {IO} and {S} are the indirect object and subject; the
# subject appears twice. Keep every template's fixed tokens identical across the
# clean/corrupted pair (we only swap names), which preserves length alignment.
TEMPLATES: list[str] = [
    "When{IO} and{S} went to the store,{S} gave a drink to{END}",
    "When{IO} and{S} went to the park,{S} gave the ball to{END}",
    "Then{IO} and{S} went to the office,{S} gave a pen to{END}",
    "After{IO} and{S} left the bar,{S} handed the keys to{END}",
]


@dataclass(frozen=True)
class IOIPrompt:
    clean: str          # full clean prompt text (no trailing answer)
    corrupted: str      # ABC-corrupted counterpart, same token length
    io: str             # correct answer token text, e.g. " Mary"
    s: str              # distractor token text, e.g. " John"


def _name_tok(name: str) -> str:
    """Mid-sentence GPT-2 form of a name: a leading space + the name."""
    return " " + name


def build_prompts() -> list[IOIPrompt]:
    """Construct clean + ABC-corrupted prompt pairs.

    Cycles names deterministically across templates so the set is reproducible
    (no RNG) and every prompt uses distinct names where required.
    """
    prompts: list[IOIPrompt] = []
    n = len(NAMES)
    for t_idx, template in enumerate(TEMPLATES):
        # Pick three distinct names for IO, S, and the corrupted third slot.
        io = NAMES[(t_idx * 3 + 0) % n]
        s = NAMES[(t_idx * 3 + 1) % n]
        # Corrupted names: three fresh, mutually distinct names for the slots
        # that were IO, S(first), S(second) in the clean prompt.
        c_a = NAMES[(t_idx * 3 + 4) % n]
        c_b = NAMES[(t_idx * 3 + 5) % n]
        c_c = NAMES[(t_idx * 3 + 6) % n]

        clean = template.format(IO=_name_tok(io), S=_name_tok(s), END="")
        # Corrupted: fill the two {S} occurrences with two *different* corrupted
        # names and {IO} with a third, so nothing repeats.
        corrupted = (
            template.replace("{IO}", _name_tok(c_a), 1)
            .replace("{S}", _name_tok(c_b), 1)   # first subject occurrence
            .replace("{S}", _name_tok(c_c), 1)   # second subject occurrence
            .replace("{END}", "")
        )
        prompts.append(
            IOIPrompt(clean=clean, corrupted=corrupted, io=_name_tok(io), s=_name_tok(s))
        )
    return prompts


def verify_single_token(tokenizer, names: list[str] | None = None) -> list[str]:
    """Return names that do NOT tokenize to exactly one token.

    Pass the result of `model.tokenizer`. An empty list means every name is
    safe to use for position-aligned patching.
    """
    names = names or NAMES
    bad = []
    for name in names:
        # TransformerLens sets add_bos_token=True, so a bare encode() prepends
        # the BOS token and every name would look like 2 tokens. We only care
        # about the name's own length, so suppress special tokens here.
        ids = tokenizer.encode(_name_tok(name), add_special_tokens=False)
        if len(ids) != 1:
            bad.append(name)
    return bad


def verify_alignment(model) -> None:
    """Assert clean and corrupted prompts tokenize to identical lengths.

    Raises AssertionError on the first misaligned pair, naming it, so the
    failure is loud instead of silently corrupting patch results.
    """
    bad_names = verify_single_token(model.tokenizer)
    assert not bad_names, f"Multi-token names break alignment: {bad_names}"
    for p in build_prompts():
        n_clean = model.to_tokens(p.clean).shape[1]
        n_corr = model.to_tokens(p.corrupted).shape[1]
        assert n_clean == n_corr, (
            f"Length mismatch ({n_clean} vs {n_corr}):\n"
            f"  clean    : {p.clean!r}\n  corrupted: {p.corrupted!r}"
        )


if __name__ == "__main__":
    for p in build_prompts():
        print("clean    :", repr(p.clean))
        print("corrupted:", repr(p.corrupted))
        print("IO / S   :", repr(p.io), "/", repr(p.s))
        print("-" * 60)
