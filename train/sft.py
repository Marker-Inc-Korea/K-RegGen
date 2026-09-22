# -*- coding: utf-8 -*-
"""LoRA SFT for K-RegGen Guard. The prompt (policy table + request + response) is masked;
only the short verdict block is supervised.

  python train/sft.py --model Qwen/Qwen3-4B-Instruct-2507 --data outputs/k_reggen_train \
      --out checkpoints/guard-4b
"""
import argparse
import json
import os
import random

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, Trainer,
                          TrainingArguments)


def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def end_token(tok):
    for t in ("<|im_end|>", "<|endofturn|>", "<end_of_turn>"):
        if t in tok.get_vocab():
            return t
    return tok.eos_token


def encode(rows, tok, max_len):
    """prompt = chat template up to the assistant turn (thinking disabled);
    completion = answer + end-of-turn token. Loss on the completion only."""
    eot = end_token(tok)
    feats, dropped = [], 0
    for r in rows:
        msgs = r["messages"]
        try:
            prompt = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False)
        except TypeError:
            prompt = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        a_ids = tok(msgs[-1]["content"] + eot, add_special_tokens=False)["input_ids"]
        ids = p_ids + a_ids
        if len(ids) > max_len:
            dropped += 1
            continue
        feats.append({"input_ids": ids, "labels": [-100] * len(p_ids) + a_ids,
                      "attention_mask": [1] * len(ids)})
    print(f"encoded {len(feats)} (dropped {dropped} over {max_len} tokens)", flush=True)
    return Dataset.from_list(feats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--data", default="outputs/k_reggen_train")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    random.seed(a.seed)
    torch.manual_seed(a.seed)
    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    train = read_jsonl(os.path.join(a.data, "sft_train.jsonl"))
    val = read_jsonl(os.path.join(a.data, "sft_val.jsonl"))
    random.Random(a.seed).shuffle(train)
    train_ds, val_ds = encode(train, tok, a.max_len), encode(val, tok, a.max_len)

    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="sdpa")
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=a.r, lora_alpha=a.alpha, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir=a.out, per_device_train_batch_size=a.bs, per_device_eval_batch_size=a.bs,
        gradient_accumulation_steps=a.accum, learning_rate=a.lr, num_train_epochs=a.epochs,
        warmup_ratio=0.03, lr_scheduler_type="cosine", bf16=True, logging_steps=10,
        eval_strategy="steps", eval_steps=100, save_strategy="steps", save_steps=100,
        save_total_limit=2, load_best_model_at_end=True, metric_for_best_model="eval_loss",
        greater_is_better=False, train_sampling_strategy="group_by_length", dataloader_num_workers=4,
        report_to="none", seed=a.seed)
    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
                      data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100))
    trainer.train()
    trainer.save_model(a.out)
    tok.save_pretrained(a.out)
    json.dump(vars(a), open(os.path.join(a.out, "train_args.json"), "w"), indent=1)
    print(f"[done] adapter -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
