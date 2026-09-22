# -*- coding: utf-8 -*-
"""Cluster near-duplicate requests and assign splits per cluster.

The same teacher given the same rule/context/style often writes the same request, so a
scenario-level split let near-identical requests land in both train and test. Here every
request is shingled (char 4-grams), candidate pairs come from MinHash LSH, and pairs with
Jaccard >= --thr are joined into clusters. Each cluster gets one split, and only a few
scenarios per cluster are kept so one phrasing does not dominate.

  python train/dedup.py --cand "outputs/train_synth/*/candidates.jsonl" outputs/train_real/candidates.jsonl \
      --out outputs/split_map.json
"""
import argparse
import glob
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, read_jsonl  # noqa: E402

K, BANDS = 64, 16
ROWS = K // BANDS


def shingles(q):
    t = re.sub(r"\s+", "", q)
    return {t[i:i + 4] for i in range(max(1, len(t) - 3))}


def minhash(sh, seeds):
    return tuple(min(int.from_bytes(hashlib.blake2b((sd + x).encode(), digest_size=6).digest(), "little")
                     for x in sh) for sd in seeds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", nargs="+", required=True, help="candidate files or globs (need scenario, request)")
    ap.add_argument("--out", default="outputs/split_map.json")
    ap.add_argument("--thr", type=float, default=0.7)
    ap.add_argument("--cap-train", type=int, default=2)
    ap.add_argument("--cap-eval", type=int, default=1)
    ap.add_argument("--test", type=float, default=0.08)
    ap.add_argument("--val", type=float, default=0.04)
    a = ap.parse_args()

    req = {}
    for pat in a.cand:
        for f in sorted(glob.glob(os.path.join(ROOT, pat))):
            for r in read_jsonl(f):
                req.setdefault(r["scenario"], re.sub(r"\s+", " ", r["request"]).strip())
    ids = sorted(req)
    sh = [shingles(req[s]) for s in ids]
    seeds = [hashlib.md5(str(k).encode()).hexdigest()[:8] for k in range(K)]
    sig = [minhash(x, seeds) for x in sh]

    parent = list(range(len(ids)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    npairs = 0
    seen = set()
    for b in range(BANDS):
        buck = defaultdict(list)
        for i, s in enumerate(sig):
            buck[s[b * ROWS:(b + 1) * ROWS]].append(i)
        for v in buck.values():
            for x in range(len(v)):
                for y in range(x + 1, len(v)):
                    i, j = v[x], v[y]
                    if (i, j) in seen:
                        continue
                    seen.add((i, j))
                    if len(sh[i] & sh[j]) / len(sh[i] | sh[j]) >= a.thr:
                        npairs += 1
                        parent[find(i)] = find(j)

    clusters = defaultdict(list)
    for i, s in enumerate(ids):
        clusters[find(i)].append(s)
    out = {}
    stat = Counter()
    for members in clusters.values():
        members.sort()
        u = int(hashlib.md5(members[0].encode()).hexdigest(), 16) % 10000 / 10000
        split = "test" if u < a.test else "val" if u < a.test + a.val else "train"
        cap = a.cap_train if split == "train" else a.cap_eval
        for k, s in enumerate(members):
            out[s] = {"split": split, "keep": k < cap, "cluster": members[0], "size": len(members)}
            stat[(split, k < cap)] += 1
    os.makedirs(os.path.dirname(os.path.join(ROOT, a.out)), exist_ok=True)
    json.dump(out, open(os.path.join(ROOT, a.out), "w"), ensure_ascii=False)
    sizes = Counter(len(m) for m in clusters.values())
    print(f"[dedup] scenarios {len(ids)} | near-dup pairs {npairs} | clusters {len(clusters)} "
          f"(size>1: {sum(v for k, v in sizes.items() if k > 1)}, largest {max(sizes)})")
    print("  kept/dropped by split:", {f"{sp}{'' if k else '-dropped'}": v for (sp, k), v in sorted(stat.items())})


if __name__ == "__main__":
    main()
