# -*- coding: utf-8 -*-
"""Diversity report for guard training data (requests and responses).

  prep   : collect text sets (unique requests per scenario, sampled responses) -> sets.json
  embed  : sentence embeddings with an e5-style encoder (torch + transformers)   -> emb.npz
  report : lexical, n-gram overlap, compression and semantic metrics per set

Size-dependent metrics (distinct-n, Self-BLEU, compression, Vendi, MATTR) are averaged
over equal-size random draws so sets of different sizes are comparable.
Report deps: numpy, scikit-learn, kiwipiepy, sacrebleu, lexicalrichness.

  python train/diversity.py prep --out outputs/diversity
  python train/diversity.py embed --model intfloat/multilingual-e5-large --out outputs/diversity
  python train/diversity.py report --out outputs/diversity
"""
import argparse
import glob
import json
import math
import os
import random
import re
import sys
import zlib
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, read_jsonl  # noqa: E402

norm = lambda s: re.sub(r"\s+", " ", s).strip()


def last_sentence(q):
    parts = re.split(r"(?<=[.?!])\s+", q.strip())
    return parts[-1] if parts else q


def prep(a):
    O = lambda p: os.path.join(ROOT, "outputs", p)
    rmap = json.load(open(O("request_map.json"))) if os.path.exists(O("request_map.json")) else {}

    def reqs(files, apply_map=False):
        seen = {}
        for f in files:
            for r in read_jsonl(f):
                q = rmap.get(r["scenario"], r["request"]) if apply_map else r["request"]
                seen.setdefault(r["scenario"], {"text": norm(q), "style": r.get("style", "")})
        return list(seen.values())

    v02 = [O("k_reggen_train/train.jsonl"), O("k_reggen_train_v02/real_train.jsonl")]
    v021 = [O("k_reggen_train_v021s/train.jsonl"), O("k_reggen_train_v021/real_train.jsonl")]
    test = [O("k_reggen_train_v021s/test.jsonl"), O("k_reggen_train_v021/real_test.jsonl")]
    fr = [O("train_synth/frontier/candidates.jsonl")]
    sets = {
        "요청 · v0.2 train": reqs(v02),
        "요청 · v0.2.1 train": reqs(v021),
        "요청 · v0.2.1 train + 끝맺음 교체": reqs(v021, True),
        "요청 · 프론티어형": reqs(fr),
        "요청 · 프론티어형 + 끝맺음 교체": reqs(fr, True),
        "요청 · v0.2.1 test": reqs(test),
        "요청 · 벤치": [{"text": norm(b["user"]), "style": b["task_type"]}
                      for b in read_jsonl(os.path.join(ROOT, "dataset", "k_reggen_v0.1.jsonl"))],
    }
    rng = random.Random(0)
    for src, f in (("합성", O("k_reggen_train_v021s/train.jsonl")), ("실제", O("k_reggen_train_v021/real_train.jsonl"))):
        rows = read_jsonl(f)
        sets[f"응답 · {src} train"] = [{"text": norm(r["response"])[:1500], "style": ""} for r in rng.sample(rows, 1000)]
    os.makedirs(os.path.join(ROOT, a.out), exist_ok=True)
    json.dump(sets, open(os.path.join(ROOT, a.out, "sets.json"), "w"), ensure_ascii=False)
    print({k: len(v) for k, v in sets.items()})


def embed(a):
    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer
    sets = json.load(open(os.path.join(ROOT, a.out, "sets.json")))
    texts = sorted({x["text"] for v in sets.values() for x in v})
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModel.from_pretrained(a.model, torch_dtype=torch.float16).cuda().eval()
    embs = []
    with torch.no_grad():
        for i in range(0, len(texts), 64):
            b = tok(["query: " + t for t in texts[i:i + 64]], padding=True, truncation=True, max_length=512,
                    return_tensors="pt").to("cuda")
            h = model(**b).last_hidden_state
            m = b["attention_mask"].unsqueeze(-1)
            e = (h * m).sum(1) / m.sum(1)
            embs.append(torch.nn.functional.normalize(e.float(), dim=-1).cpu().numpy())
    np.savez(os.path.join(ROOT, a.out, "emb.npz"), texts=np.array(texts, dtype=object), emb=np.concatenate(embs))
    print("embedded", len(texts))


def report(a):
    import numpy as np
    from kiwipiepy import Kiwi
    from lexicalrichness import LexicalRichness
    from sacrebleu.metrics import BLEU
    from sklearn.cluster import KMeans

    sets = json.load(open(os.path.join(ROOT, a.out, "sets.json")))
    z = np.load(os.path.join(ROOT, a.out, "emb.npz"), allow_pickle=True)
    E = {t: e for t, e in zip(z["texts"], z["emb"])}
    kiwi = Kiwi()
    morph = {}

    def toks(t):
        if t not in morph:
            morph[t] = [f"{x.form}/{x.tag}" for x in kiwi.tokenize(t)]
        return morph[t]

    bleu = BLEU(tokenize="none", effective_order=True)
    rng = random.Random(0)
    M, DRAWS = a.m, 5
    rows = []
    for name, items in sets.items():
        texts = [x["text"] for x in items]
        acc = Counter()
        for d in range(DRAWS):
            s = rng.sample(texts, min(M, len(texts)))
            tk = [toks(t) for t in s]
            for n in (1, 2, 3):
                g = [tuple(t[i:i + n]) for t in tk for i in range(len(t) - n + 1)]
                acc[f"distinct-{n}"] += len(set(g)) / max(1, len(g)) / DRAWS
            # Self-BLEU: 각 문장을 나머지 문장들을 참조로 삼아 채점(낮을수록 다양)
            joined = [" ".join(t) for t in tk]
            sb = [bleu.sentence_score(joined[i], joined[:i] + joined[i + 1:]).score for i in rng.sample(range(len(s)), min(60, len(s)))]
            acc["Self-BLEU"] += sum(sb) / len(sb) / DRAWS
            raw = "\n".join(s).encode()
            acc["압축률"] += len(zlib.compress(raw, 9)) / len(raw) / DRAWS
            acc["MATTR"] += LexicalRichness(" ".join(" ".join(t) for t in tk), preprocessor=None, tokenizer=str.split).mattr(window_size=50) / DRAWS
            X = np.stack([E[t] for t in s])
            K = X @ X.T / len(s)
            ev = np.clip(np.linalg.eigvalsh(K), 1e-12, None)
            acc["Vendi"] += float(np.exp(-(ev * np.log(ev)).sum())) / DRAWS
        X = np.stack([E[t] for t in texts])
        S = X @ X.T
        np.fill_diagonal(S, -1)
        nn = S.max(1)
        iu = np.triu_indices(len(texts), 1)
        k = min(50, len(texts) // 5)
        lab = KMeans(n_clusters=k, n_init=5, random_state=0).fit_predict(X)
        p = np.bincount(lab) / len(lab)
        ends = Counter(last_sentence(t)[-10:] for t in texts)
        rows.append(dict(집합=name, n=len(texts), **{k2: v for k2, v in acc.items()},
                         의미_평균코사인=float(S[iu].mean()), 의미_최근접평균=float(nn.mean()),
                         의미_근접중복_0_95=float((nn >= 0.95).mean()),
                         군집균등도=float(-(p * np.log(p)).sum() / math.log(k)),
                         최빈끝맺음=ends.most_common(1)[0][1] / len(texts)))
    json.dump(rows, open(os.path.join(ROOT, a.out, "report.json"), "w"), ensure_ascii=False, indent=1)
    keys = [k for k in rows[0] if k not in ("집합",)]
    print("| " + " | ".join(["집합"] + keys) + " |")
    print("|---" * (len(keys) + 1) + "|")
    for r in rows:
        print("| " + " | ".join([r["집합"]] + [f"{r[k]:.3f}" if isinstance(r[k], float) else str(r[k]) for k in keys]) + " |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prep", "embed", "report"])
    ap.add_argument("--out", default="outputs/diversity")
    ap.add_argument("--model", default="intfloat/multilingual-e5-large")
    ap.add_argument("--m", type=int, default=250, help="sample size for size-dependent metrics")
    a = ap.parse_args()
    {"prep": prep, "embed": embed, "report": report}[a.stage](a)


if __name__ == "__main__":
    main()
