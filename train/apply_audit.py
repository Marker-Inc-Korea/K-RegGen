# -*- coding: utf-8 -*-
"""Apply a two-reader audit to a built dataset.

A first reader flagged examples (label/rule/span problems, with a suggested fix); a second
reader re-judged the flagged examples blind (without the current label). Per example:
  - second reader agrees with the current label -> keep; adopt its rules/spans only when the
    first reader flagged rules/spans and the second reader is confident
  - both readers move the label the same way -> relabel to the second reader's verdict
  - readers disagree, or the change is low-confidence -> drop (contested)

  python train/apply_audit.py --first first_pass.json --second verify_out_dir --idmap idmap.json \
      --src outputs/k_reggen_train_v023s outputs/k_reggen_train_v023 \
      --dst outputs/k_reggen_train_v024s outputs/k_reggen_train_v024
"""
import argparse
import glob
import json
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, guard_messages, guard_target, load_rules, read_jsonl, write_jsonl  # noqa: E402
from fixes import restore_span  # noqa: E402

LABEL = re.compile(r"(위반|확인필요|적법)")


def suggested_label(findings):
    """The label the first reader proposed, if its fix text names one."""
    for f in findings:
        if f["cat"] == "label":
            m = LABEL.search(f.get("fix", ""))
            if m:
                return m.group(1)
    return None


def decide(cur, first, second, rules):
    """-> (action, new_example_or_None)."""
    lab2 = second.get("label")
    if lab2 not in ("위반", "확인필요", "적법"):
        return "drop_unreadable", None
    r2 = sorted(x for x in second.get("rules", []) if x in rules["by_id"])
    sp2 = [s for s in (restore_span(x, cur["response"]) for x in second.get("spans", [])) if s]
    if lab2 != "적법" and (not r2 or not sp2):
        return "drop_second_incomplete", None
    if lab2 == "확인필요" and not any(rules["by_id"][x].get("condition") for x in r2):
        return "drop_unverified_without_condition", None
    confident = second.get("conf", "high") != "low"
    cats = {f["cat"] for f in first}
    if lab2 == cur["label"]:
        if lab2 == "적법":
            return "keep", cur
        new = dict(cur)
        if "rule" in cats and confident and r2 != sorted(cur["rules"]):
            new["rules"] = r2
        if "span" in cats and confident:
            new["spans"] = sp2
        return ("fix_rules_spans" if new != cur else "keep"), new
    if not confident:
        return "drop_low_conf_change", None
    if suggested_label(first) != lab2:
        return "drop_readers_disagree", None
    new = {**cur, "label": lab2, "rules": r2 if lab2 != "적법" else [], "spans": sp2 if lab2 != "적법" else []}
    return f"relabel_{cur['label']}->{lab2}", new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", required=True)
    ap.add_argument("--second", required=True, help="directory of *.jsonl verdicts")
    ap.add_argument("--idmap", required=True)
    ap.add_argument("--src", nargs="+", required=True)
    ap.add_argument("--dst", nargs="+", required=True)
    ap.add_argument("--log", default="outputs/audit_applied.jsonl")
    a = ap.parse_args()
    rules = load_rules()
    first = json.load(open(a.first))
    idmap = json.load(open(a.idmap))
    second = {}
    for f in sorted(glob.glob(os.path.join(a.second, "*.jsonl"))):
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            try:
                v = json.loads(line)
            except json.JSONDecodeError:
                continue
            if v.get("vid") in idmap:
                second[idmap[v["vid"]]] = v
    stats, log, rec = Counter(), [], {}
    for src, dst in zip(a.src, a.dst):
        src, dst = os.path.join(ROOT, src), os.path.join(ROOT, dst)
        os.makedirs(dst, exist_ok=True)
        for f in sorted(os.listdir(src)):
            if f.startswith("sft_") or not f.endswith(".jsonl"):
                continue
            keep = []
            for r in read_jsonl(os.path.join(src, f)):
                if r["ex_id"] in first:
                    if r["ex_id"] not in second:
                        act, new = "unverified_kept", r
                    else:
                        act, new = decide(r, first[r["ex_id"]], second[r["ex_id"]], rules)
                    stats[(f, act.split("->")[0] if act.startswith("drop") else act)] += 1
                    log.append({"ex_id": r["ex_id"], "file": f, "action": act,
                                "old": [r["label"], r["rules"]], "new": [new["label"], new["rules"]] if new else None})
                    if new is None:
                        continue
                    r = new
                rec[r["ex_id"]] = r
                keep.append(r)
            write_jsonl(os.path.join(dst, f), keep)
            stats[(f, "kept")] += len(keep)
        if os.path.exists(os.path.join(src, "build_meta.json")):
            meta = json.load(open(os.path.join(src, "build_meta.json")))
            meta["audit"] = "train/apply_audit.py"
            json.dump(meta, open(os.path.join(dst, "build_meta.json"), "w"), ensure_ascii=False, indent=1)
    for src, dst in zip(a.src, a.dst):
        src, dst = os.path.join(ROOT, src), os.path.join(ROOT, dst)
        for f in sorted(os.listdir(src)):
            if not f.startswith("sft_"):
                continue
            out = []
            for s in read_jsonl(os.path.join(src, f)):
                c = rec.get(s["ex_id"])
                if c is not None:
                    out.append({**s, "messages": guard_messages(rules, c["domain"], c["request"], c["response"])
                                + [{"role": "assistant", "content": guard_target(c["label"], c["rules"], c["spans"])}]})
            write_jsonl(os.path.join(dst, f), out)
    write_jsonl(os.path.join(ROOT, a.log), log)
    for k, v in sorted(stats.items(), key=lambda x: str(x[0])):
        print(k, v)


if __name__ == "__main__":
    main()
