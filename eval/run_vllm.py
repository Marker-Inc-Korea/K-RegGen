# -*- coding: utf-8 -*-
"""K-RegGen 벤치를 로컬 vLLM으로 측정한다. 단일 GPU 전용(TP는 이 클러스터에서 행).

  python eval/run_vllm.py --model google/gemma-4-31B-it --sets core,minpairs --n 5 --out outputs/

CLI 하네스(run_codex/run_claude)와 같은 시스템 프롬프트를 쓴다.
"""
import argparse, gc, json, os, re, sys

NEUTRAL_SYS = "당신은 한국어 사용자를 돕는 AI 어시스턴트입니다. 사용자의 메시지에 대화로 답하세요."

# 모델별 공식 권장 샘플링 파라미터
PARAMS = {
    "google/gemma-4-31B-it":                        dict(temperature=1.0, top_p=0.95, top_k=64),
    "Qwen/Qwen3.8-27B":                             dict(temperature=0.7, top_p=0.8,  top_k=20),
    "kakaocorp/kanana-2-30b-a3b-instruct-2601":     dict(temperature=0.7, top_p=0.8,  top_k=20),
    "LGAI-EXAONE/EXAONE-4.5-33B-FP8":               dict(temperature=0.6, top_p=0.95, top_k=-1),
    "zai-org/GLM-4.7-Flash":                        dict(temperature=0.6, top_p=0.95, top_k=-1),
}
DEFAULT_PARAMS = dict(temperature=0.7, top_p=0.9, top_k=-1)


def clean(text):
    """사고 흔적 제거. EXAONE은 템플릿이 <think>를 미리 열어 고아 </think>가 남는다."""
    t = text
    if "</think>" in t and "<think>" not in t:
        t = t.split("</think>", 1)[1]
    t = re.sub(r"<think>.*?</think>", "", t, flags=re.S)
    # gpt-oss 계열 harmony 평문형
    if "assistantfinal" in t:
        t = t.rpartition("assistantfinal")[2]
    return t.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--sets", default="core,minpairs")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--grow-rounds", type=int, default=2,
                    help="절단된 샘플을 예산 2배로 재생성하는 최대 라운드 수")
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--gpu-util", type=float, default=0.90)
    ap.add_argument("--moe-backend", default="")
    ap.add_argument("--max-num-seqs", type=int, default=0)
    a = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    files = {"core": "k_reggen_core.jsonl", "minpairs": "k_reggen_minpairs.jsonl",
             "full": "k_reggen_v0.1.jsonl"}
    items = []
    for s in a.sets.split(","):
        s = s.strip()
        p = os.path.join(root, "dataset", files[s]) if s in files else s   # 이름 또는 파일 경로
        items += [json.loads(l) for l in open(p, encoding="utf-8")]
    print(f"[load] {len(items)} items from {a.sets}", flush=True)

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    prompts = []
    for it in items:
        msgs = [{"role": "system", "content": NEUTRAL_SYS},
                {"role": "user", "content": it["user"]}]
        try:
            p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                        enable_thinking=False)
        except TypeError:
            p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        prompts.append(p)

    kw = dict(model=a.model, tensor_parallel_size=1, gpu_memory_utilization=a.gpu_util,
              max_model_len=a.max_model_len, trust_remote_code=True, enforce_eager=False)
    if a.moe_backend:
        kw["moe_backend"] = a.moe_backend
    if a.max_num_seqs:
        kw["max_num_seqs"] = a.max_num_seqs
    llm = LLM(**kw)

    def gen(prompt_list, budget, n):
        sp = SamplingParams(n=n, max_tokens=budget, seed=20260920,
                            **PARAMS.get(a.model, DEFAULT_PARAMS))
        return llm.generate(prompt_list, sp)

    outs = gen(prompts, a.max_tokens, a.n)
    # 결과를 [문항][샘플] 격자로 펼친 뒤, 절단된 칸만 예산을 2배로 늘려 다시 채운다.
    grid = [[{"text": c.text, "finish": c.finish_reason} for c in o.outputs] for o in outs]
    budget = a.max_tokens
    for rnd in range(a.grow_rounds):
        todo = [(i, j) for i, row in enumerate(grid)
                for j, c in enumerate(row) if c["finish"] == "length"]
        if not todo:
            break
        budget *= 2
        print(f"[grow {rnd+1}] 절단 {len(todo)}칸 → max_tokens={budget}로 재생성", flush=True)
        sub = sorted({i for i, _ in todo})
        re_outs = gen([prompts[i] for i in sub], budget, a.n)
        for i, o in zip(sub, re_outs):
            cands = [{"text": c.text, "finish": c.finish_reason} for c in o.outputs]
            for j, c in enumerate(grid[i]):
                if c["finish"] == "length":
                    # 같은 문항의 재생성 후보 중 절단되지 않은 것을 우선 채택
                    pick = next((x for x in cands if x["finish"] != "length"), cands[j])
                    grid[i][j] = pick
                    if pick in cands:
                        cands.remove(pick)
    ntrunc_final = sum(1 for row in grid for c in row if c["finish"] == "length")

    os.makedirs(os.path.join(root, a.out), exist_ok=True)
    tag = a.model.split("/")[-1]
    suffix = "" if a.sets == "core,minpairs" else "_" + "-".join(
        os.path.splitext(os.path.basename(x.strip()))[0] for x in a.sets.split(","))
    dst = os.path.join(root, a.out, f"{tag}{suffix}.jsonl")
    with open(dst, "w", encoding="utf-8") as f:
        for it, row in zip(items, grid):
            f.write(json.dumps({**it, "model": a.model,
                                "samples": [{"text": clean(c["text"]), "finish": c["finish"]} for c in row]},
                               ensure_ascii=False) + "\n")
    ntrunc = ntrunc_final
    print(f"[done] {dst} | items={len(items)} n={a.n} truncated={ntrunc}", flush=True)
    del llm; gc.collect()


if __name__ == "__main__":
    main()
