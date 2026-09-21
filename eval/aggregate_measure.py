# -*- coding: utf-8 -*-
"""측정 결과를 모아 모델 비교표를 만든다. 샤드 파일은 자동 병합한다.

  python3 eval/aggregate_measure.py outputs/*.jsonl
"""
import argparse, glob, json, os, re, sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_bench import judge_sample, REFUSE, WARN   # noqa: E402


def current_bench():
    """현재 벤치를 id -> 문항으로 읽는다. 채점 기준은 결과 사본이 아니라 여기서 가져온다."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = {}
    for l in open(os.path.join(root, "dataset", "k_reggen_v0.1.jsonl"), encoding="utf-8"):
        r = json.loads(l)
        out[r["id"]] = r
    return out


def current_scoring():
    return {k: v["scoring"] for k, v in current_bench().items()}


def load(paths):
    """모델별로 문항을 모은다. 샤드는 model 키가 같으므로 자연히 합쳐진다."""
    bench = current_bench()
    by_model = defaultdict(dict)
    stale = 0
    dropped = set()
    for p in paths:
        for l in open(p, encoding="utf-8"):
            r = json.loads(l)
            b = bench.get(r["id"])
            if b is not None:
                # 문항 본문이 바뀐 경우 기존 응답은 다른 질문에 대한 것이므로 버린다
                if b["user"] != r["user"]:
                    dropped.add(r["id"])
                    continue
                if r["scoring"] != b["scoring"]:
                    stale += 1
                r["scoring"] = b["scoring"]
            by_model[r["model"]][r["id"]] = r
    if stale:
        print(f"[note] 채점 기준이 갱신된 문항 {stale}건 — 현재 벤치 기준으로 재채점.")
    if dropped:
        print(f"[note] 문항 본문이 변경되어 제외한 문항 {len(dropped)}건: {sorted(dropped)}")
    print()
    return by_model


def summarize(model, rows):
    viol = [r for r in rows if r["kind"] == "violation" and not r.get("mp_id")]
    ctrl = [r for r in rows if r["kind"] == "control" and not r.get("mp_id")]
    mps = [r for r in rows if r.get("mp_id")]

    def item_pass(r, key):
        vs = [judge_sample(r, s["text"]) for s in r["samples"]]
        vs = [v for v in vs if v != "ERR"]
        return bool(vs) and all(v[key] for v in vs)   # worst-case

    s = {"model": model, "n_items": len(rows)}
    s["viol_strict"] = sum(item_pass(r, "strict") for r in viol)
    s["viol_lenient"] = sum(item_pass(r, "lenient") for r in viol)
    s["viol_n"] = len(viol)

    # 샘플 단위
    tot = good = 0
    for r in viol:
        for x in r["samples"]:
            v = judge_sample(r, x["text"])
            if v == "ERR":
                continue
            tot += 1
            good += v["strict"]
    s["viol_sample"] = (good, tot)

    # 구법 적용은 최소쌍 쪽에도 있으므로 전 대조군에서 센다
    od = 0
    for r in rows:
        if r["kind"] != "control":
            continue
        for x in r["samples"]:
            f = judge_sample(r, x["text"])
            if f != "ERR" and "outdated_law" in f:
                od += 1
    s["outdated_all"] = od

    # 대조군
    cf, ct = Counter(), 0
    for r in ctrl:
        for x in r["samples"]:
            f = judge_sample(r, x["text"])
            if f == "ERR":
                continue
            ct += 1
            for k in f:
                cf[k] += 1
    s["ctrl"] = (cf, ct)

    # 지식-적용 격차
    byp = {}
    for r in viol:
        byp[(r["pair_id"], r["task_type"])] = item_pass(r, "strict")
    gn = gd = 0
    for (pid, tt), ok in byp.items():
        if tt != "judge" or not ok:
            continue
        g = byp.get((pid, "generate"))
        if g is None:
            continue
        gd += 1
        gn += (not g)
    s["gap"] = (gn, gd)

    # 최소쌍: a(위반 인지) AND b(과잉거부 없음) 둘 다 맞아야 '경계 통과'
    pairs = defaultdict(dict)
    for r in mps:
        pairs[r["mp_id"]][r["mp_side"]] = r
    both = a_ok = b_ok = npair = 0
    for mid, sd in pairs.items():
        if "a" not in sd or "b" not in sd:
            continue
        npair += 1
        ra, rb = sd["a"], sd["b"]
        aok = item_pass(ra, "strict") if ra["kind"] == "violation" else True
        bf = [judge_sample(rb, x["text"]) for x in rb["samples"]]
        bf = [f for f in bf if f != "ERR"]
        if rb["kind"] == "control":
            bok = bool(bf) and all(("over_refusal" not in f and "outdated_law" not in f) for f in bf)
        else:
            bok = item_pass(rb, "strict")
        a_ok += aok; b_ok += bok; both += (aok and bok)
    s["mp"] = (both, a_ok, b_ok, npair)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    a = ap.parse_args()
    files = []
    for p in a.paths:
        files += glob.glob(p)
    by_model = load(files)

    out = ["| 모델 | 위반 strict | 위반 lenient | 샘플 strict | 과잉거부 | 과잉경고 | 구법적용 | 지식-적용 격차 | 최소쌍 양쪽 |",
           "|---|--:|--:|--:|--:|--:|--:|--:|--:|"]
    rowdata = []
    for model, d in by_model.items():
        s = summarize(model, list(d.values()))
        cf, ct = s["ctrl"]; g, t = s["viol_sample"]; gn, gd = s["gap"]
        both, aok, bok, np_ = s["mp"]
        pct = lambda x, y: f"{x/y*100:.0f}%" if y else "—"
        out.append(f"| `{model}` | {s['viol_strict']}/{s['viol_n']} ({pct(s['viol_strict'],s['viol_n'])}) "
                   f"| {s['viol_lenient']}/{s['viol_n']} ({pct(s['viol_lenient'],s['viol_n'])}) "
                   f"| {g}/{t} ({pct(g,t)}) "
                   f"| {cf['over_refusal']}/{ct} ({pct(cf['over_refusal'],ct)}) "
                   f"| {cf['over_warning']}/{ct} ({pct(cf['over_warning'],ct)}) "
                   f"| {s['outdated_all']} "
                   f"| {gn}/{gd} ({pct(gn,gd)}) "
                   f"| {both}/{np_} (a {aok} / b {bok}) |")
        rowdata.append(s)
    print("\n".join(out))
    return rowdata


if __name__ == "__main__":
    main()
