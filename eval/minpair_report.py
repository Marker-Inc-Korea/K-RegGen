# -*- coding: utf-8 -*-
"""최소쌍 쌍별 판정표. a=위반측 인지, b=적법측 비차단."""
import json, glob, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_bench import judge_sample
from aggregate_measure import current_scoring

def load(paths):
    sc = current_scoring(); by = {}
    for p in paths:
        for f in glob.glob(p):
            for l in open(f, encoding="utf-8"):
                r = json.loads(l)
                if not r.get("mp_id") or r["id"] not in sc:
                    continue
                r["scoring"] = sc[r["id"]]
                by.setdefault(r["model"], {}).setdefault(r["mp_id"], {})[r["mp_side"]] = r
    return by

def ok(r):
    vs = [judge_sample(r, x["text"]) for x in r["samples"]]
    vs = [v for v in vs if v != "ERR"]
    if not vs:
        return None
    if r["kind"] == "violation":
        return all(v["strict"] for v in vs)
    return all(("over_refusal" not in v and "outdated_law" not in v) for v in vs)

if __name__ == "__main__":
    by = load(sys.argv[1:])
    models = list(by)
    mids = sorted({m for d in by.values() for m in d})
    print("| 최소쌍 | " + " | ".join(f"{m.split('/')[0].split('-')[0]} a/b" for m in models) + " |")
    print("|---|" + "---|" * len(models))
    for mid in mids:
        cells = []
        for m in models:
            sd = by[m].get(mid, {})
            a = ok(sd["a"]) if "a" in sd else None
            b = ok(sd["b"]) if "b" in sd else None
            f = lambda v: "—" if v is None else ("O" if v else "X")
            cells.append(f"{f(a)}/{f(b)}")
        print(f"| `{mid}` | " + " | ".join(cells) + " |")
