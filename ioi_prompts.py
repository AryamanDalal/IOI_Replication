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
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import islice
import verify_offline



# Names vetted to be single GPT-2 tokens *with a leading space* (the form they
# take mid-sentence). verify_single_token() re-checks this against the live
# tokenizer so a model/tokenizer change can't silently break alignment.
NAMES: list[str] = [
    "Michael",
    "Anna",
    "Tom",
    "Mike",
    "Jack",]
    # "Christopher",
    # "Jessica",
    # "Matthew",
    # "Ashley",
    # "Jennifer",
    # "Joshua",
    # "Amanda",
    # "Daniel",
    # "David",
    # "James",
    # "Dan",
    # "Robert",
    # "John",
    # "Joseph",
    # "Andrew",
    # "Ryan",
    # "Brandon",
#     "Jason",
#     "Justin",
#     "Sarah",
#     "William",
#     "Jonathan",
#     "Stephanie",
#     "Brian",
#     "Nicole",
#     "Nicholas",
#     "Anthony",
#     "Heather",
#     "Eric",
#     "Elizabeth",
#     "Adam",
#     "Megan",
#     "Melissa",
#     "Kevin",
#     "Steven",
#     "Thomas",
#     "Timothy",
#     "Christina",
#     "Kyle",
#     "Rachel",
#     "Laura",
#     "Lauren",
#     "Amber",
#     "Brittany",
#     "Danielle",
#     "Richard",
#     "Kimberly",
#     "Jeffrey",
#     "Amy",
#     "Crystal",
#     "Michelle",
#     "Tiffany",
#     "Jeremy",
#     "Benjamin",
#     "Mark",
#     "Emily",
#     "Aaron",
#     "Charles",
#     "Rebecca",
#     "Jacob",
#     "Stephen",
#     "Patrick",
#     "Sean",
#     "Erin",
#     "Jamie",
#     "Kelly",
#     "Samantha",
#     "Nathan",
#     "Sara",
#     "Dustin",
#     "Paul",
#     "Angela",
#     "Tyler",
#     "Scott",
#     "Katherine",
#     "Andrea",
#     "Gregory",
#     "Erica",
#     "Mary",
#     "Travis",
#     "Lisa",
#     "Kenneth",
#     "Bryan",
#     "Lindsey",
#     "Kristen",
#     "Jose",
#     "Alexander",
#     "Alex",
#     "Jesse",
#     "Katie",
#     "Lindsay",
#     "Shannon",
#     "Vanessa",
#     "Courtney",
#     "Christine",
#     "Alicia",
#     "Cody",
#     "Allison",
#     "Bradley",
#     "Samuel",
# ]


# Same-length templates. {IO} and {S} are the indirect object and subject; the
# subject appears twice. Keep every template's fixed tokens identical across the
# clean/corrupted pair (we only swap names), which preserves length alignment.
TEMPLATES: list[str] = [
    "When{IO} and{S1} went to the store,{S2} gave a drink to{END}",
    "When{IO} and{S1} went to the park,{S2} gave the ball to{END}",
    "Then{IO} and{S1} went to the office,{S2} gave a pen to{END}",
    "After{IO} and{S1} went to the bar,{S2} gave the keys to{END}",
    "After{IO} and{S1} went to the ball,{S2} gave the money to{END}"
]


@dataclass(frozen=True)
class IOIPrompt:
    clean: str          # full clean prompt text (no trailing answer)
    corrupted: list[str]     # ABC-corrupted counterpart, same token length
    partial_corrupted: list[str]#Clean prompt with different s2
    partial_s2: list[str]    # swapped S2 name of each partial variant (index-aligned)
    io: str             # correct answer token text, e.g. " Mary"
    s: str              # distractor token text, e.g. " John"
    c_a: list[str]            #Corrupted Name A
    c_b: list[str]            #Corrupted Name B
    c_c: list[str]            #Corrupted Name C

def _name_tok(name: str) -> str:
    """Mid-sentence GPT-2 form of a name: a leading space + the name."""
    return " " + name


def verify_single_token(tokenizer, names: list[str] = NAMES) -> list[str]:
    """Return names that do NOT tokenize to exactly one token.

    Pass the result of `model.tokenizer`. An empty list means every name is
    safe to use for position-aligned patching.
    """

    bad = []
    for name in names:
        # TransformerLens sets add_bos_token=True, so a bare encode() prepends
        # the BOS token and every name would look like 2 tokens. We only care
        # about the name's own length, so suppress special tokens here.
        ids = tokenizer.encode(_name_tok(name), add_special_tokens=False)
        if len(ids) != 1:
            bad.append(name)

    assert not bad, f"Multi-token names break alignment: {bad}"
    print("Verification Successful. No multitoken names found")
    return None


def verify_alignment(model) -> None:
    """Assert each clean prompt tokenizes to the same length as every one of its
    corrupted and partial-corrupted variants.
    Raises AssertionError on the first misaligned pair, naming it, so the
    failure is loud instead of silently corrupting patch results.
    """
    #Ensure all names have token length = 1
    verify_single_token(model.tokenizer)

    #Ensures Senteces have the same token lengths
    verify_templates(model)

    # Single-token names + equal-length templates already guarantee alignment;
    # spot-check a sample of prompts against ALL their variants to catch
    # regressions loudly.
    for prompt in islice(ioi_prompts(), 100):
        n_clean = model.to_tokens(prompt.clean).shape[1]
        for variant in (*prompt.corrupted, *prompt.partial_corrupted):
            n_var = model.to_tokens(variant).shape[1]
            assert n_clean == n_var, (
                f"Length mismatch ({n_clean} vs {n_var}):\n"
                f"  clean  : {prompt.clean!r}\n  variant: {variant!r}"
            )
    return None

def verify_templates(model) -> None:
    """
    Ensures Senteces have the same token lengths. Assertion error on the first sentence
    with a different token length in the TEMPLATES list.
    """
    prev = None

    for sentence in TEMPLATES:
        sentence = sentence.format(IO=_name_tok("John"), S1=_name_tok("John"), S2=_name_tok("John"), END="")
        if prev is None:
            prev = model.to_tokens(sentence).shape[1]
        
        new = model.to_tokens(sentence).shape[1]
        assert new == prev, f"Templates are of different token lenghts:{prev} and {new}"

    print(f"All sentences have token length = {prev}")
    return None

def ioi_prompts() -> Iterator[IOIPrompt]:
    """Yield one IOIPrompt per clean prompt, grouping all of its corrupted
    variants into parallel lists.

    For each (template, io, s) with io != s:
      * clean (ABB): IO=io, S1=S2=s -- duplicate-token signal present.
      * partial_corrupted (S2-swap): io and S1=s kept, S2 -> b for every fresh
        name b (not io, not s). Breaks the duplication, keeps io and s in view.
      * corrupted (ABC, IO fixed): io kept at the IO slot, both subject slots
        overwritten with two DISTINCT fresh names (neither io nor s). Removes
        the signal while pinning IO so logit[io]-logit[s] stays like-for-like.

    c_a/c_b/c_c are the (IO, S1, S2) names of each ABC variant, index-aligned
    with `corrupted` and stored in the same leading-space token form as io/s.
    """
    for template in TEMPLATES:
        for i, io in enumerate(NAMES):
            for j, s in enumerate(NAMES):
                if i == j:                       # IO must differ from S
                    continue
                io_tok, s_tok = _name_tok(io), _name_tok(s)
                clean = template.format(IO=io_tok, S1=s_tok, S2=s_tok, END="")

                partial_corrupted, corrupted = [], []
                partial_s2 = []
                c_a, c_b, c_c = [], [], []
                for k, b in enumerate(NAMES):
                    if k in (i, j):              # b is fresh: not io, not s
                        continue
                    b_tok = _name_tok(b)
                    # S2-swap: keep io and s, only the second subject -> b.
                    partial_corrupted.append(
                        template.format(IO=io_tok, S1=s_tok, S2=b_tok, END=""))
                    partial_s2.append(b_tok)

                    for l, c in enumerate(NAMES):    # ABC: second free slot
                        if l in (i, j, k):           # c distinct & fresh too
                            continue
                        c_tok = _name_tok(c)
                        corrupted.append(
                            template.format(IO=io_tok, S1=b_tok, S2=c_tok, END=""))
                        c_a.append(io_tok)
                        c_b.append(b_tok)
                        c_c.append(c_tok)

                yield IOIPrompt(
                    clean=clean,
                    corrupted=corrupted,
                    partial_corrupted=partial_corrupted,
                    partial_s2=partial_s2,
                    io=io_tok,
                    s=s_tok,
                    c_a=c_a,
                    c_b=c_b,
                    c_c=c_c,
                )





if __name__ == "__main__":
    model = verify_offline.gpt2()
    verify_single_token(model.tokenizer)
    #Ensure templates have the same token lengths with inserted names
    verify_alignment(model)

    # Eyeball a few grouped prompts: one clean ABB prompt followed by all of its
    # partial (S2-swap) and ABC (IO-fixed) variants. islice bounds the preview.
    for prompt in islice(ioi_prompts(), 2):
        print("clean             :", repr(prompt.clean))
        print("  IO / S           :", repr(prompt.io), "/", repr(prompt.s))
        print(f"partial_corrupted ({len(prompt.partial_corrupted)}):")
        for text, s2 in zip(prompt.partial_corrupted, prompt.partial_s2):
            print("    ", repr(text),"| IO/S1/S2:", repr(prompt.io), repr(prompt.s), repr(s2))
        print(f"corrupted ABC ({len(prompt.corrupted)}):")
        for text, a, b, c in zip(prompt.corrupted, prompt.c_a, prompt.c_b, prompt.c_c):
            print("    ", repr(text), "| IO/S1/S2:", repr(a), repr(b), repr(c))
        print("-" * 60)
