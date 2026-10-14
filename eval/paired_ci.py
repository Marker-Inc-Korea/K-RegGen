# -*- coding: utf-8 -*-
"""Paired bootstrap of the flag-level F1 difference between two guards on the same examples.

  python eval/paired_ci.py --ref outputs/eval_frozen/real_test.jsonl --a test_guard-4b-v025.jsonl --b test_guard-4b-v05.jsonl
  python eval/paired_ci.py --ref outputs/guard/bench_gold_sample.jsonl --a bench_guard-4b-v025.jsonl --b bench_guard-4b-v05.jsonl

The reference rows are keyed by `key` (benchmark) or `ex_id` (test splits); predictions by `key`.
"""
import argparse
import json
import random

FLAG = {"위반", "확인필요"}


def read(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def f1(rows):
    tp = sum(g and a for g, a in rows)
    pp = sum(a for _, a in rows)
    gp = sum(g for g, _ in rows)
    r, p = tp / max(1, gp), tp / max(1, pp)
    return 2 * r * p / max(1e-9, r + p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--a", required=True, help="baseline predictions")
    ap.add_argument("--b", required=True, help="candidate predictions")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    ref = {r.get("key") or r["ex_id"]: r["label"] in FLAG for r in read(a.ref) if r.get("label")}
    pa = {r["key"]: r["label"] in FLAG for r in read(a.a)}
    pb = {r["key"]: r["label"] in FLAG for r in read(a.b)}
    keys = [k for k in ref if k in pa and k in pb]
    A = [(ref[k], pa[k]) for k in keys]
    B = [(ref[k], pb[k]) for k in keys]
    rng = random.Random(a.seed)
    diffs = []
    for _ in range(a.n):
        idx = [rng.randrange(len(keys)) for _ in keys]
        diffs.append(f1([B[i] for i in idx]) - f1([A[i] for i in idx]))
    diffs.sort()
    lo, hi = diffs[int(a.n * .025)], diffs[int(a.n * .975)]
    print(f"n={len(keys)}  F1 a={f1(A):.3f}  b={f1(B):.3f}  diff={f1(B) - f1(A):+.3f}  95% [{lo:+.3f}, {hi:+.3f}]")


if __name__ == "__main__":
    main()
