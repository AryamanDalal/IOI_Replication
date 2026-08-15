# imports
import torch
from dataclasses import dataclass

from dataset import Prompt
from dataset_verification import load_prompts_dict, load_model, make_cache_filter
# Imported only for type hints. transformer_lens is already imported (with offline mode
# set) as a side effect of importing dataset_verification above, so this line is safe here.
from transformer_lens import ActivationCache, HookedTransformer


@dataclass(frozen=True)
class Run_Details:
    prompt: Prompt              # prompt.text / prompt.io / prompt.s1 / prompt.s2, all str
    cache: ActivationCache      # {(hook_name, layer): e.g. [batch_index, seq_no, n_head, d_head]}
    logits: torch.Tensor        # [batch_index, seq_no, vocab_id] = [1, small=15|large=60, 50257]
    io_id: int                  # token id for the IO name
    s1_id: int                  # token id for the S1 name
    logit_diff: torch.Tensor    # scalar — logit[IO] - logit[S] at the final position


def process_prompts_dict(model: HookedTransformer) -> dict:
    """
    Run one forward pass per prompt in the dataset and cache what the analyses need, so that no
    downstream module has to re-run the model. Decoupling the forward pass from the analysis this
    way means the same cached activations feed direct logit attribution and path patching alike.
    The nesting of the returned dict mirrors the dataset's own: keyed first by
    (ordering, size), then by variant.

    model  --> a loaded HookedTransformer (see dataset_verification.load_model)
    return -> {(ordering, size): {variant: [Run_Details]}}
    """
    prompts_dict, ioi_lst = load_prompts_dict()
    processed_prompts_dict = {}
    # One filter reused for every pass; without it each retained cache would hold every hook.
    cache_filter = make_cache_filter(model)

    for prompt_ordering, prompt_size in prompts_dict.keys():
        processed_prompts_dict.setdefault((prompt_ordering, prompt_size), {"clean": [], "corrupt": [], "negative": [], "scrambled": []})

        for prompt_type in prompts_dict[(prompt_ordering, prompt_size)].keys():
            for prompt in prompts_dict[(prompt_ordering, prompt_size)][prompt_type]:
                io_id, s1_id = model.to_single_token(prompt.io), model.to_single_token(prompt.s1)
                logits, cache = model.run_with_cache(prompt.text, names_filter=cache_filter)

                # The metric of interest, read off the final position of this same pass.
                logit_diff = logits[0, -1, io_id] - logits[0, -1, s1_id]

                run_details = Run_Details(prompt=prompt,
                                          cache=cache,
                                          logits=logits,
                                          io_id=io_id,
                                          s1_id=s1_id,
                                          logit_diff=logit_diff
                )

                processed_prompts_dict[(prompt_ordering, prompt_size)][prompt_type].append(run_details)
    print("Loaded processed_prompts_dict")
    return processed_prompts_dict

def collect_processed_prompts_lists(processed_prompts_dict: dict, orderings: list[str], sizes: list[str], prompt_types: list[str]) -> list[list[Run_Details]]:
    """
    Flatten the nested dict into one flat prompt list per (prompt_type, size) combination,
    concatenating the requested orderings within each. Pooling the orderings is what lets an
    analysis average over both name positions while still keeping the variants apart, and the
    per-combination lists stay index-aligned so they can be paired elementwise downstream
    (path patching consumes exactly such a pair).

    processed_prompts_dict --> the dict returned by process_prompts_dict
    orderings              --> template orderings to pool, e.g. ["IO_S1_S2", "S1_IO_S2"]
    sizes                  --> template sizes to include, e.g. ["small"]
    prompt_types           --> variants to include: "clean" | "corrupt" | "negative" | "scrambled"
    return -> one list of Run_Details per (prompt_type, size), ordered prompt_types-major
    """
    processed_prompts_lists = []

    for prompt_type in prompt_types:
        for size in sizes:
            temp_list = []

            for ordering in orderings:
                temp_list.extend(processed_prompts_dict[(ordering, size)][prompt_type])
            processed_prompts_lists.append(temp_list)

    print(f"Loaded {len(processed_prompts_lists)} prompt lists each with {len(processed_prompts_lists[0])} prompts")
    return processed_prompts_lists


def main() -> None:
    """Load the model and build the cached Run_Details for the whole dataset (smoke check)."""
    model = load_model()
    processed_prompts_dict = process_prompts_dict(model)
    return None


if __name__ == "__main__":
    main()
