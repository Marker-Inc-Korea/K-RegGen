# -*- coding: utf-8 -*-
"""Merge the audited base splits with newly finalized sets into one training release.

The test split is always the frozen, audited evaluation set.

  --drop-cond-violating  drop the base synthetic violating versions of conditional rules. They
                         were generated with the response stating the breach ('동의 여부와 관계없이',
                         '평가 전이지만'). v0.3 dropped them and lost recall on conditional rules,
                         so it is off by default.
  --lawful-cap 0.487     drop lawful rows from the added sets (never from the base) until the
                         lawful share of train is at most this value.
  --val-real-only        keep only collected model responses in val, so the checkpoint is chosen
                         on real responses rather than on synthetic style.
  --holdout A,B          move every scenario that intends or is labelled with these rules out of
                         train/val (rule hold-out experiment).

  python train/merge.py --base-synth outputs/k_reggen_train_v024s --base-real outputs/k_reggen_train_v025 \
      --add outputs/k_reggen_train_v03_frontier outputs/k_reggen_train_v03_cond5 \
      --frozen-test outputs/eval_frozen --dst outputs/k_reggen_train_v03 --drop-cond-violating
"""
import argparse
import json
import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, guard_messages, guard_target, load_rules, read_jsonl, write_jsonl  # noqa: E402

# 본인부담금형 할인 위반본은 조건과 무관하게 금지 요소를 담으므로 남긴다
KEEP_VIOLATING = {"MA-NONCOVERED-DISCOUNT"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-synth", required=True)
    ap.add_argument("--base-real", required=True)
    ap.add_argument("--add", nargs="*", default=[])
    ap.add_argument("--frozen-test", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--holdout", default="", help="comma-separated rule IDs kept out of train/val (with their scenarios)")
    ap.add_argument("--drop-cond-violating", action="store_true")
    ap.add_argument("--lawful-cap", type=float, default=0.0)
    ap.add_argument("--val-real-only", action="store_true")
    a = ap.parse_args()
    hold = set(x for x in a.holdout.split(",") if x)
    rules = load_rules()
    cond = {r["id"] for r in rules["rules"] if r.get("condition")} - KEEP_VIOLATING
    P = lambda *x: os.path.join(ROOT, *x)
    frozen = read_jsonl(P(a.frozen_test, "synth_test.jsonl")) + read_jsonl(P(a.frozen_test, "real_test.jsonl"))
    frozen_scen = {c["scenario"] for c in frozen}
    stat, parts = Counter(), {}
    for sp in ("train", "val"):
        rows = []
        for c in read_jsonl(P(a.base_synth, f"{sp}.jsonl")):
            if (a.drop_cond_violating and c.get("rule") in cond and c["version"].startswith("violating")
                    and c.get("style") != "inquiry"):
                stat[(sp, "base_synth_cond_violating_dropped")] += 1
                continue
            rows.append({**c, "_base": True})
        stat[(sp, "base_synth")] = len(rows)
        real = read_jsonl(P(a.base_real, f"real_{sp}.jsonl"))
        rows += [{**c, "_base": True} for c in real]
        stat[(sp, "base_real")] = len(real)
        for d in a.add:
            add = [c for c in read_jsonl(P(d, f"{sp}.jsonl")) if c["scenario"] not in frozen_scen]
            rows += add
            stat[(sp, os.path.basename(d))] = len(add)
        seen, uniq = set(), []
        for c in rows:
            if c["ex_id"] not in seen and c["scenario"] not in frozen_scen:
                seen.add(c["ex_id"])
                uniq.append(c)
        if hold:
            held = {c["scenario"] for c in uniq if c.get("rule") in hold or c.get("intended_rule") in hold
                    or set(c["rules"]) & hold}
            parts.setdefault("heldout", []).extend(c for c in uniq if c["scenario"] in held)
            uniq = [c for c in uniq if c["scenario"] not in held]
            stat[(sp, "moved_to_heldout")] = sum(1 for c in parts["heldout"] if c["scenario"] in held)
        if sp == "train" and a.lawful_cap:
            rng = random.Random(7)
            added_lawful = [i for i, c in enumerate(uniq) if c["label"] == "적법" and not c.get("_base")]
            rng.shuffle(added_lawful)
            n_law = sum(c["label"] == "적법" for c in uniq)
            # 적법 비율이 상한 이하가 될 때까지 추가분의 적법 예제만 뺀다
            k = 0
            while added_lawful[k:] and (n_law - k) / (len(uniq) - k) > a.lawful_cap:
                k += 1
            drop = set(added_lawful[:k])
            uniq = [c for i, c in enumerate(uniq) if i not in drop]
            stat[(sp, "added_lawful_dropped_for_cap")] = k
        if sp == "val" and a.val_real_only:
            stat[(sp, "synthetic_dropped_from_val")] = sum("source_model" not in c for c in uniq)
            uniq = [c for c in uniq if "source_model" in c]
        random.Random(1).shuffle(uniq)
        parts[sp] = [{k: v for k, v in c.items() if k != "_base"} for c in uniq]
    parts["test"] = frozen
    os.makedirs(P(a.dst), exist_ok=True)
    for sp, rows in parts.items():
        write_jsonl(P(a.dst, f"{sp}.jsonl"), rows)
        write_jsonl(P(a.dst, f"sft_{sp}.jsonl"),
                    [{"ex_id": c["ex_id"], "messages": guard_messages(rules, c["domain"], c["request"], c["response"])
                      + [{"role": "assistant", "content": guard_target(c["label"], c["rules"], c["spans"])}]}
                     for c in rows])
        stat[(sp, "total")] = len(rows)
        stat[(sp, "labels")] = dict(Counter(c["label"] for c in rows))
    json.dump({"holdout": sorted(hold), "base_synth": a.base_synth, "base_real": a.base_real, "add": a.add,
               "frozen_test": a.frozen_test, "drop_cond_violating": a.drop_cond_violating,
               "lawful_cap": a.lawful_cap, "val_real_only": a.val_real_only,
               "stats": {f"{k[0]}:{k[1]}": v for k, v in stat.items()}},
              open(P(a.dst, "build_meta.json"), "w"), ensure_ascii=False, indent=1)
    for k, v in stat.items():
        print(k, v)


if __name__ == "__main__":
    main()
