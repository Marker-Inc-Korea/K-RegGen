# -*- coding: utf-8 -*-
"""Independent label check: a local judge from another model family reads the same
policy table and labels every candidate. build.py keeps only the agreeing ones.

  python train/verify.py --judge <judge-model> --cand outputs/train_synth/candidates.jsonl \
      --out outputs/train_synth/judged.jsonl
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, clean, guard_messages, judge_messages, load_rules, parse_guard, read_jsonl, write_jsonl  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", required=True)
    ap.add_argument("--cand", default="outputs/train_synth/candidates.jsonl")
    ap.add_argument("--out", default="outputs/train_synth/judged.jsonl")
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--judge-note", action="store_true", help="append the labeler-only caution")
    ap.add_argument("--part", default="0/1", help="i/n: label every n-th candidate starting at i")
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--max-model-len", type=int, default=16384)
    a = ap.parse_args()

    rules = load_rules()
    cands = read_jsonl(os.path.join(ROOT, a.cand))
    done = set()
    dst = os.path.join(ROOT, a.out)
    if os.path.exists(dst):
        done = {r["ex_id"] for r in read_jsonl(dst)}
    i, n = map(int, a.part.split("/"))
    cands = cands[i::n]
    todo = [c for c in cands if c["ex_id"] not in done]
    print(f"[judge] {len(todo)} to label ({len(done)} already)", flush=True)
    if not todo:
        return

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(a.judge, trust_remote_code=True)
    prompts = []
    for c in todo:
        msgs = (judge_messages if a.judge_note else guard_messages)(rules, c["domain"], c["request"], c["response"])
        prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                               reasoning_effort=a.effort))
    llm = LLM(model=a.judge, tensor_parallel_size=1, gpu_memory_utilization=0.90,
              max_model_len=a.max_model_len, trust_remote_code=True)
    sp = SamplingParams(n=1, max_tokens=a.max_tokens, temperature=0.0, skip_special_tokens=False)
    outs = llm.generate(prompts, sp)
    rows = list(read_jsonl(dst)) if done else []
    for c, o in zip(todo, outs):
        text = o.outputs[0].text
        final = clean(text.split("<|channel|>final<|message|>")[-1] if "<|channel|>final" in text else text)
        label, ids, spans = parse_guard(final)
        rows.append({"ex_id": c["ex_id"], "judge": a.judge, "label": label, "rules": ids, "spans": spans,
                     "finish": o.outputs[0].finish_reason, "answer": final[-600:]})
    write_jsonl(dst, rows)
    print(f"[done] {dst} labelled={sum(1 for r in rows if r['label'])}/{len(rows)}", flush=True)


if __name__ == "__main__":
    main()
