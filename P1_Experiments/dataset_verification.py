import torch
from dataset import Names, Templates
import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
from transformer_lens import HookedTransformer

def load_model() -> HookedTransformer:
    """Load GPT-2 small onto the CPU with autograd disabled (inference only)."""
    torch.set_grad_enabled(False)
    model = HookedTransformer.from_pretrained("gpt2", device="cpu")
    return model

def verify_names(model: HookedTransformer) -> None:
    """
    Assert that every name in Names().ALL_NAMES is a single GPT-2 token in its
    mid-sentence form (a leading space + the name). dataset.py documents this
    single-token assumption but cannot check it itself — it has no tokenizer.
    """
    bad = []
    names = Names().ALL_NAMES
    for name in names:
        # Mid-sentence form: the leading space is part of the token (" Michael" -> 1 token).
        if len(model.tokenizer.encode(" " + name, add_special_tokens = False)) != 1:
            bad.append(name)

    assert not bad, f"The given names are multitoken: {bad}."
    print("\n","All names are a single token.\n")
    return None

def verify_alignment(model: HookedTransformer) -> None:
    """
    Per template size, verify the two invariants the IOI experiments depend on:
      1. Every template of that size tokenizes to the same number of tokens.
      2. The N1/N2/N3 name tokens occupy identical sequence positions across all
         templates of that size, including the scrambled variants.
    Templates are filled with distinct single-token sentinel names, so the only
    thing that can shift a name's position is the surrounding template text.
    """
    templates = Templates().TEMPLATES
    # One distinct single-token name per slot; END fills to "" (contributes no token).
    fill = {"N1": " John", "N2": " Mary", "N3": " Tom", "END": ""}

    # Column of the three name token ids -> shape [3, 1].
    encoded_names = model.to_tokens([" John", " Mary", " Tom"], prepend_bos=False)
    assert encoded_names.shape[1] == 1, "The provided names are multitoken"

    # Broadcasting [3, 1] against each filled sentence [1, L] gives a [3, L] boolean
    # mask whose row i marks where name i occurs in that sentence.
    small = [encoded_names == model.to_tokens(template.format(**fill), prepend_bos=False) for template in templates["small"]] \
            + [encoded_names == model.to_tokens(template.format(**fill), prepend_bos=False) for template in templates["scrambled_small"]]
    # Equal token length across the group (also required for the torch.stack below).
    assert all(small[0].shape[1] == mask.shape[1] for mask in small), "small length sentences tokenize to different lengths"
    # all(dim=0): keep a cell only if a name sits there in EVERY sentence -> [3, L].
    # sum(dim=1) == 1: each name shares exactly one common column; torch.all folds to a scalar.
    assert torch.all(torch.stack(small).all(dim=0).sum(dim=1) == 1), "Name tokens do not occur at the same sequence position across all small length sentences"

    large = [encoded_names == model.to_tokens(template.format(**fill), prepend_bos=False) for template in templates["large"]] \
            + [encoded_names == model.to_tokens(template.format(**fill), prepend_bos=False) for template in templates["scrambled_large"]]
    assert all(large[0].shape[1] == mask.shape[1] for mask in large), "large length sentences tokenize to different lengths"
    assert torch.all(torch.stack(large).all(dim=0).sum(dim=1) == 1), "Name tokens do not occur at the same sequence position across all large length sentences"

    print("All sentences of the same length type tokenize to the same number of tokens with the name tokens occurring uniformly at the same sequence positions", "\n")
    return None

if __name__ == "__main__":
    model = load_model()
    verify_names(model)
    verify_alignment(model)
