import os
import sys
model_name_or_path = sys.argv[1]
output_path = sys.argv[2]
if os.path.exists(output_path):
    exit(0)

from modeling_llama import LlamaForCausalLM as AutoModelForCausalLM
import transformers
from transformers import (
    AutoConfig,
)
num_experts = 4


config = AutoConfig.from_pretrained(model_name_or_path)
expert_size = int(config.intermediate_size/num_experts)
llama_model = transformers.LlamaForCausalLM.from_pretrained(
    model_name_or_path,
    from_tf=bool(".ckpt" in model_name_or_path),
    config=config,
    low_cpu_mem_usage=False,
)
llama_model.eval()
llama_model_state_dict = llama_model.state_dict()

for key in llama_model_state_dict.keys():
    print(key, llama_model_state_dict[key].size())

model = AutoModelForCausalLM(config)
model_state_dict = model.state_dict()

for key in model_state_dict.keys():
    if key not in llama_model_state_dict:
        print(f"Key {key} not found in llama_model_state_dict: {model_state_dict[key].size()}")
print("="*10)
for key in llama_model_state_dict.keys():
    if key not in model_state_dict:
        print(f"Key {key} not found in model_state_dict: {llama_model_state_dict[key].size()}")

for key in model_state_dict.keys():
    if key in llama_model_state_dict:
        print(f"Key {key} found in both models: {model_state_dict[key].size()} and {llama_model_state_dict[key].size()}")
        model_state_dict[key] = llama_model_state_dict[key]
    else:
        if ".experts" in key:
            expert_pos = key.find(".experts.")
            for i in range(num_experts):
                if f".experts.{i}" in key:
                    new_key = key[:expert_pos] + key[expert_pos+len(f".experts.{i}"):]
                    print(f"Replacing {key} ({model_state_dict[key].size()}) with {new_key} ({llama_model_state_dict[new_key].size()})")
                    model_state_dict[key] = llama_model_state_dict[new_key]
                    print(f"New {key} size: {model_state_dict[key].size()}")
                    print('='*5)
                    break
model.load_state_dict(model_state_dict)
model.save_pretrained(output_path)