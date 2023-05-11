import argparse

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from iGPT.models.husky_src.husky_chat import Blip2LlaMAForConditionalGeneration


def apply_delta(base_model_path, target_model_path, delta_path):
    print("Loading base model")
    base = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True)

    print("Loading delta")
    delta_tokenizer = AutoTokenizer.from_pretrained(delta_path, use_fast=False)
    delta = Blip2LlaMAForConditionalGeneration.from_pretrained(delta_path, torch_dtype=torch.float16, low_cpu_mem_usage=True)

    print("Applying delta")
    for name, param in tqdm(delta.state_dict().items(), desc="Applying delta"):
        if name.startswith('language_model'):
            name = name[len('language_model.'):]
            if param.data.shape == base.state_dict()[name].shape:
                param.data += base.state_dict()[name]
            else:
                bparam = base.state_dict()[name]
                param.data[:bparam.shape[0], :bparam.shape[1]] += bparam
        else:
            pass

    print("Saving target model")
    delta.save_pretrained(target_model_path)
    delta_tokenizer.save_pretrained(target_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-path", type=str, required=True)
    parser.add_argument("--target-model-path", type=str, required=True)
    parser.add_argument("--delta-path", type=str, required=True)

    args = parser.parse_args()

    apply_delta(args.base_model_path, args.target_model_path, args.delta_path)
    # srun -p INTERN2 --gres=gpu:0 python apply_delta.py --base-model-path "/mnt/petrelfs/share_data/wangweiyun/share_hf/llama-7b-hf" --target-model-path "/mnt/petrelfs/share_data/wangweiyun/share_hf/husky-7b-demo-v0_01" --delta-path "/mnt/petrelfs/share_data/wangweiyun/share_hf/husky-7b-delta-v0_01"