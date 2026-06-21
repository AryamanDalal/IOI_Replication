import torch
from ioi_prompts import ioi_prompts
import verify_offline

torch.set_grad_enabled(False)
model = verify_offline.gpt2()
prompt = next(ioi_prompts())

print(prompt)