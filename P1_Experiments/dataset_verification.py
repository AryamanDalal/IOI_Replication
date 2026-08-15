# imports
import os
import logging

import torch

from dataset import Names, Templates, IOI, IOIPrompt

# TransformerLens emits one "already cached" record per weight file it finds locally; drop those
# so the verification output below is the only thing printed.
logging.getLogger().addFilter(lambda r: "already cached" not in r.getMessage())

# Offline mode is set BEFORE transformer_lens is imported: HuggingFace reads these flags at
# import time, and the GPT-2 weights are assumed to already sit in the local cache. Every module
# that imports this one therefore inherits an offline-configured transformer_lens for free.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
from transformer_lens import HookedTransformer


# -- Section 1: model loading and cache configuration ---------------------------------------

def load_model() -> HookedTransformer:
    """
    Load GPT-2 small onto the CPU with autograd disabled (inference only). The three set_use_*
    flags turn on hooks TransformerLens leaves off by default to save memory, but which the
    experiments downstream read or patch: per-head attention results, split Q/K/V inputs, and
    the MLP input.
    return -> a HookedTransformer ready for run_with_cache and hook-based patching
    """
    torch.set_grad_enabled(False)
    model = HookedTransformer.from_pretrained("gpt2", device="cpu")
    model.set_use_attn_result(True)
    model.set_use_split_qkv_input(True)
    model.set_use_hook_mlp_in(True)
    return model

def make_cache_filter(model: HookedTransformer):
    """
    Build the names_filter handed to run_with_cache. An unfiltered cache holds every hook of
    every layer, far more than these experiments read; since one cache is retained per prompt
    (see process_dataset.Run_Details), this filter is what bounds total memory.

    model  --> a loaded HookedTransformer; only cfg.n_layers is read, to name the final resid_post
    return -> a predicate (hook name -> keep it?) suitable for run_with_cache(names_filter=...)
    """
    def cache_filter(name: str) -> bool:
        # Per-layer hooks, matched by suffix because every such name carries a "blocks.{layer}." prefix.
        keep_suffix = ("hook_resid_pre", "attn.hook_q", "attn.hook_k", "attn.hook_v", "attn.hook_z",
                       "attn.hook_attn_scores", "hook_mlp_in", "hook_mlp_out",
                       "ln1.hook_scale", "ln2.hook_scale")
        # Whole-model hooks, matched exactly because there is exactly one of each.
        keep_exact = ("hook_embed", "hook_pos_embed", "ln_final.hook_scale",
                      f"blocks.{model.cfg.n_layers - 1}.hook_resid_post")

        return name.endswith(keep_suffix) or name in keep_exact
    return cache_filter


# -- Section 2: dataset invariants the experiments depend on --------------------------------

def verify_names(model: HookedTransformer) -> None:
    """
    Assert that every name in Names().ALL_NAMES is a single GPT-2 token in its
    mid-sentence form (a leading space + the name). dataset.py documents this
    single-token assumption but cannot check it itself — it has no tokenizer.
    Raises AssertionError listing every offender. return -> None
    """
    multitoken_names = []
    names = Names().ALL_NAMES
    for name in names:
        # Mid-sentence form: the leading space is part of the token (" Michael" -> 1 token).
        if len(model.tokenizer.encode(" " + name, add_special_tokens=False)) != 1:
            multitoken_names.append(name)

    assert not multitoken_names, f"The given names are multitoken: {multitoken_names}."
    print("\n", "All names are a single token.\n")
    return None

def verify_alignment(model: HookedTransformer) -> None:
    """
    Per template size, verify the two invariants the IOI experiments depend on:
      1. Every template of that size tokenizes to the same number of tokens.
      2. The N1/N2/N3 name tokens occupy identical sequence positions across all
         templates of that size, including the scrambled variants.
    Templates are filled with distinct single-token sentinel names, so the only
    thing that can shift a name's position is the surrounding template text.
    Raises AssertionError naming the size group that failed. return -> None
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


# -- Section 3: dataset entry point ---------------------------------------------------------

def load_prompts_dict() -> tuple[dict, list[IOIPrompt]]:
    """
    Build the IOI dataset. Kept here so downstream modules reach the prompts through the module
    that has already configured offline mode, rather than importing dataset.py directly.
    return -> (prompts_dict {(ordering, size): {variant: [Prompt]}}, flat list of IOIPrompt)
    """
    prompts_dict, ioi_lst = IOI().create_dataset()
    return prompts_dict, ioi_lst


def main() -> None:
    """Load the model and run both dataset invariants: single-token names, then name alignment."""
    model = load_model()
    verify_names(model)
    verify_alignment(model)
    return None


if __name__ == "__main__":
    main()
