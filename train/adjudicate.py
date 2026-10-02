# -*- coding: utf-8 -*-
"""Turn two blind readings of candidate examples into training rows.

Used for examples the reference judge disagreed with: those are exactly the hard cases
(acceptance-style violations, violations wrapped in warnings, conditional rules where the
request already reveals the fact), so they are labelled by two independent readings
instead of being dropped.

A row is kept when both readings give the same label; for flagged rows the rule set is the
shared one (identical, or the overlap), and the span is copied verbatim from the response.
Rows both readers marked low-confidence are dropped. Versions that only append a
disclaimer to another version (…_disclaimer, …_flagged) take the base version's reading.

  python train/adjudicate.py --pool pool.json --readers outA outB --idmaps idmap_A.json idmap_B.json \
      --dst outputs/k_reggen_train_v04_adj
"""
import argparse
import glob
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, load_rules, write_jsonl  # noqa: E402
from fixes import restore_span  # noqa: E402

LABELS = ("위반", "확인필요", "적법")


def load_reading(out_dir, idmap_path):
    idmap = json.load(open(idmap_path))
    got = {}
    for f in sorted(glob.glob(os.path.join(out_dir, "*.jsonl"))):
        for line in open(f, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                v = json.loads(line)
            except json.JSONDecodeError:
                continue
            if v.get("vid") in idmap and v.get("label") in LABELS:
                got[idmap[v["vid"]]] = v
    return got


def decide(c, a, b, rules):
    if a is None or b is None:
        return None, "missing_reading"
    if a["label"] != b["label"]:
        return None, "readers_disagree"
    if a.get("conf") == "low" and b.get("conf") == "low":
        return None, "both_low_conf"
    label = a["label"]
    if label == "적법":
        return {**c, "label": "적법", "rules": [], "spans": []}, "ok"
    ra = {x for x in a.get("rules", []) if x in rules["by_id"]}
    rb = {x for x in b.get("rules", []) if x in rules["by_id"]}
    rs = sorted(ra if ra == rb else ra & rb)
    if not rs:
        return None, "rules_disagree"
    if label == "확인필요" and not any(rules["by_id"][x].get("condition") for x in rs):
        return None, "unverified_without_condition_rule"
    spans = [s for s in (restore_span(x, c["response"]) for x in a.get("spans", [])) if s]
    if not spans:
        spans = [s for s in (restore_span(x, c["response"]) for x in b.get("spans", [])) if s]
    if not spans:
        return None, "no_span"
    return {**c, "label": label, "rules": rs, "spans": spans[:3]}, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True, help="json list of candidate rows (with split)")
    ap.add_argument("--readers", nargs=2, required=True)
    ap.add_argument("--idmaps", nargs=2, required=True)
    ap.add_argument("--dst", required=True)
    a = ap.parse_args()
    rules = load_rules()
    pool = json.load(open(a.pool))
    ra, rb = (load_reading(d, m) for d, m in zip(a.readers, a.idmaps))
    by_id = {c["ex_id"]: c for c in pool}
    decided, stat, moves = {}, Counter(), Counter()
    for c in pool:
        if c["version"].endswith(("_disclaimer", "_flagged")):
            continue
        row, why = decide(c, ra.get(c["ex_id"]), rb.get(c["ex_id"]), rules)
        stat[why] += 1
        if row:
            decided[c["ex_id"]] = row
            moves[(c["src"], c["version"], c["label"], row["label"])] += 1
    # 면책 문구만 덧붙인 변형은 원본 판독을 따른다
    for c in pool:
        if not c["version"].endswith(("_disclaimer", "_flagged")):
            continue
        base_ver = c["version"].rsplit("_", 1)[0]
        base = next((d for d in decided.values() if d["scenario"] == c["scenario"] and d["version"] == base_ver), None)
        if base is None:
            stat["variant_without_base"] += 1
            continue
        spans = [s for s in (restore_span(x, c["response"]) for x in base["spans"]) if s]
        if base["label"] != "적법" and not spans:
            stat["variant_no_span"] += 1
            continue
        decided[c["ex_id"]] = {**c, "label": base["label"], "rules": base["rules"], "spans": spans}
        stat["variant_propagated"] += 1
    parts = {"train": [], "val": []}
    for r in decided.values():
        r = {k: v for k, v in r.items() if k not in ("judge", "src")}
        parts[r["split"]].append({**r, "adjudicated": True})
    os.makedirs(os.path.join(ROOT, a.dst), exist_ok=True)
    for sp, rows in parts.items():
        write_jsonl(os.path.join(ROOT, a.dst, f"{sp}.jsonl"), rows)
        print(f"  {sp}: {len(rows)}", dict(Counter(r["label"] for r in rows)))
    print("[adjudicate]", dict(stat))
    for k, v in sorted(moves.items(), key=lambda x: -x[1])[:20]:
        print("  ", k, v)
    json.dump({"stat": dict(stat), "moves": {"|".join(k): v for k, v in moves.items()}},
              open(os.path.join(ROOT, a.dst, "build_meta.json"), "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
