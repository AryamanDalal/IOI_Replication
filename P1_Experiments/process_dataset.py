#imports
import torch
from dataset_verification import load_prompts_dict, load_model
from dataclasses import dataclass
from transformer_lens import ActivationCache
from dataset import Prompt



@dataclass(frozen=True)
class Run_Details:
    prompt: Prompt #prompt.text: str, prompt.io: str, prompt.s1: str
    cache: ActivationCache #{(hook_name, layer): eg. [batch_index, seq_no, n_head, d_head]}
    logits: torch.tensor # [batch_index, seq_no, vocab_id] = [1, small=15|large=60, 50257]
    io_id: int #token_id for the io name
    s1_id: int #token_id for the s1 name
    logit_diff: int 

def process_prompts_dict(model):
    prompts_dict, ioi_lst = load_prompts_dict()
    processed_prompts_dict = {}

    for prompt_ordering, prompt_size in prompts_dict.keys():
        processed_prompts_dict.setdefault((prompt_ordering, prompt_size), {"clean": [], "corrupt": [], "negative": [], "scrambled": []})

        for prompt_type in prompts_dict[(prompt_ordering, prompt_size)].keys():
            for prompt in prompts_dict[(prompt_ordering, prompt_size)][prompt_type]:
                io_id, s1_id = model.to_single_token(prompt.io), model.to_single_token(prompt.s1)
                logits, cache = model.run_with_cache(prompt.text)
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

def collect_processed_prompts_lists(processed_prompts_dict, orderings, sizes, prompt_types):
    processed_prompts_lists = []

    for prompt_type in prompt_types:
        for size in sizes:
            temp_list = []

            for ordering in orderings:
                temp_list.extend(processed_prompts_dict[(ordering, size)][prompt_type])
            processed_prompts_lists.append(temp_list)

    print(f"Loaded {len(processed_prompts_lists)} prompt lists each with {len(processed_prompts_lists[0])} prompts")
    return processed_prompts_lists

def main():
    model = load_model()
    processed_prompts_dict = process_prompts_dict(model)
    return None

if __name__ == "__main__":
    main()
