
import os
import torch
from typing import Tuple
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    Qwen2ForCausalLM,
    AutoModelForCausalLM,
    AutoTokenizer,
)


def load_llm(config: dict, device: str = "cuda") -> Tuple:
    """
    加载 LLM，返回 (model, tokenizer, model_config)。
    config 字段:
        type: "vicuna" | "llama" | "llama3" | "qwen" | "auto"
        path: 模型路径
        freeze: bool (默认 True)
        torch_dtype: "float16" | "bfloat16" | "float32" (默认 float16)
        lora: dict | null — 若提供则用 LoRA 微调
            r: 8, alpha: 32, dropout: 0.1
            target_modules: ["q_proj", "v_proj"] (默认)
    """
    llm_type = config.get("type", "auto")
    path = config["path"]
    freeze = config.get("freeze", True)
    dtype_str = config.get("torch_dtype", "float16")
    lora_cfg = config.get("lora", None)

    if dtype_str == "float16":
        torch_dtype = torch.float16
    elif dtype_str == "bfloat16":
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32

    print(f"Loading LLM from {path} (type={llm_type}, dtype={dtype_str})...")

    if llm_type in ("vicuna", "llama", "llama3"):
        model = LlamaForCausalLM.from_pretrained(
            path, torch_dtype=torch_dtype, device_map=device
        )
        tokenizer = LlamaTokenizer.from_pretrained(path, use_fast=False)
    elif llm_type == "qwen":
        model = Qwen2ForCausalLM.from_pretrained(
            path, torch_dtype=torch_dtype, device_map=device
        )
        tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False, trust_remote_code=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch_dtype, device_map=device, trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # LoRA
    if lora_cfg is not None:
        from peft import LoraConfig, get_peft_model, TaskType
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=lora_cfg.get("r", 8),
            lora_alpha=lora_cfg.get("alpha", 32),
            lora_dropout=lora_cfg.get("dropout", 0.1),
            target_modules=lora_cfg.get("target_modules", ["q_proj", "v_proj"]),
        )
        model = get_peft_model(model, lora_config)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"LoRA enabled: r={lora_cfg.get('r', 8)}, trainable={trainable:,}/{total:,} ({100*trainable/total:.2f}%)")
    elif freeze:
        for param in model.parameters():
            param.requires_grad = False

    print(f"LLM loaded: vocab_size={model.config.vocab_size}, hidden_size={model.config.hidden_size}")
    return model, tokenizer, model.config
