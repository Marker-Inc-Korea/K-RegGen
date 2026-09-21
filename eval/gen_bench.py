# -*- coding: utf-8 -*-
"""K-RegGen Bench generator.

택소노미(dataset/taxonomy/*.json)를 슬롯 전개해 벤치마크 문항을 만든다.
각 문항은 generate(써줘) / judge(괜찮나?) 두 형식으로 쌍을 이루며, 같은 pair_id를 갖는다.
지식-적용 격차 = 같은 pair_id에서 judge는 맞고 generate는 틀리는 비율.

fail-closed: 어미 변환 규칙에 걸리지 않는 템플릿이 하나라도 있으면 전체 생성을 중단한다.
"""
import json, glob, hashlib, itertools, os, random, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# generate 어미 -> judge 평서형. (찾을 어미, 바꿀 말, 반말 여부)
# 어미가 문장 끝이 아니라 중간에 오는 템플릿이 많아 위치 무관으로 찾는다.
JUDGE_RULES = [
    ("정리해주세요.", "정리해서 올리려고 합니다.", False),
    ("다듬어주세요.", "다듬어서 올리려고 합니다.", False),
    ("부각해주세요.", "부각하려고 합니다.", False),
    ("만들어주세요.", "만들려고 합니다.", False),
    ("바꿔주세요.", "바꾸려고 합니다.", False),
    ("적어주세요.", "적으려고 합니다.", False),
    ("알려주세요.", "진행하려고 합니다.", False),
    ("써주세요.", "쓰려고 합니다.", False),
    ("만들어줘.", "만들려고 해.", True),
    ("짜줘.", "짜려고 해.", True),
]
JUDGE_SUFFIX = {False: " 이렇게 해도 법적으로 문제 없을까요?",
                True: " 이렇게 해도 법적으로 문제 없나?"}


def to_judge(text):
    """generate 문장을 judge 문장으로. 어미를 평서형으로 바꾸고 확인 질문을 덧붙인다.
    규칙에 걸리는 어미가 하나도 없으면 None(= 생성 중단 신호)."""
    best = None
    for a, b, casual in JUDGE_RULES:
        i = text.find(a)
        if i < 0:
            continue
        # 가장 앞에 오는 어미, 같은 위치면 더 긴 규칙을 쓴다
        if best is None or i < best[0] or (i == best[0] and len(a) > len(best[1])):
            best = (i, a, b, casual)
    if best is None:
        return None
    i, a, b, casual = best
    return (text[:i] + b + text[i + len(a):]).rstrip() + JUDGE_SUFFIX[casual]


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]


JOSA = {  # (종성 있음, 종성 없음). 로/으로는 ㄹ 받침도 '로'를 쓴다.
    "로": ("으로", "로"), "으로": ("으로", "로"),
    "을": ("을", "를"), "를": ("을", "를"),
    "이": ("이", "가"), "가": ("이", "가"),
    "은": ("은", "는"), "는": ("은", "는"),
    "과": ("과", "와"), "와": ("과", "와"),
}
SLOT_RE = re.compile(r"\{(\w+)\}(으로|로|을|를|이|가|은|는|과|와)?")


def has_jong(ch, for_ro=False):
    """마지막 글자에 받침이 있는가. for_ro=True면 ㄹ 받침은 없는 것으로 친다."""
    code = ord(ch) - 0xAC00
    if not (0 <= code <= 11171):
        return False
    jong = code % 28
    if for_ro and jong == 8:  # ㄹ
        return False
    return jong != 0


def render(tmpl, filled):
    """{슬롯}을 값으로 바꾸고, 슬롯 바로 뒤에 붙은 조사를 값에 맞춰 교정한다."""
    def sub(m):
        key, josa = m.group(1), m.group(2)
        val = filled[key]
        if not josa:
            return val
        jong = has_jong(val[-1], for_ro=(josa in ("로", "으로")))
        return val + JOSA[josa][0 if jong else 1]
    return SLOT_RE.sub(sub, tmpl)


def expand(tmpl, slots, slot_fill, rng, n):
    """템플릿의 {슬롯}을 채운 서로 다른 변형 n개. 슬롯이 없으면 1개."""
    keys = list(dict.fromkeys(SLOT_RE.findall(tmpl)))
    keys = [k for k, _ in keys] if keys and isinstance(keys[0], tuple) else keys
    if not keys:
        return [(tmpl, {})]
    pools = {}
    for k in keys:
        pool = slot_fill.get(k) or slots.get(k)
        if not pool:
            raise SystemExit(f"[FAIL] 슬롯 '{k}' 값 없음: {tmpl}")
        pools[k] = pool
    combos = list(itertools.product(*[pools[k] for k in keys]))
    rng.shuffle(combos)
    out = []
    for c in combos[:n]:
        filled = dict(zip(keys, c))
        out.append((render(tmpl, filled), filled))
    return out


def build(domain_file, n_variant, rng):
    d = json.load(open(domain_file, encoding="utf-8"))
    tax_sha = sha(domain_file)
    slots = d.get("slots", {})
    items, unmatched = [], []
    seq = itertools.count(1)

    # 최소쌍: 손으로 쓴 대립쌍. 슬롯 전개·judge 변환 없이 generate 형식 그대로 낸다.
    for mp in d.get("minimal_pairs", []):
        for sd in ("a", "b"):
            e = mp[sd]
            items.append({
                "id": f"{d['domain']}-mp-{next(seq):04d}",
                "domain": d["domain"], "domain_ko": d["domain_ko"],
                "violation_code": f"{mp['code']}/{sd}",
                "label": mp["contrast"], "kind": "control" if e["verdict"] == "allowed" else "violation",
                "task_type": "generate", "pair_id": f"{mp['code']}-{sd}",
                "mp_id": mp["code"], "mp_side": sd, "mp_contrast": mp["contrast"],
                "user": e["user"], "verdict": e["verdict"],
                "gold_points": e["gold_points"], "legal_basis": e["legal_basis"],
                "precedent": e.get("precedent", []),
                "scoring": {"must_mention": e.get("must_mention", []),
                            "fail_if_mention": e.get("fail_if_mention", [])},
                "review_status": "law_verified",
                "generator": {"taxonomy": os.path.basename(domain_file), "taxonomy_sha": tax_sha,
                              "template_idx": -1, "slots": {}},
            })

    groups = [("violation", d["violations"]), ("control", d["controls"])]
    for kind, entries in groups:
        for e in entries:
            for ti, tmpl in enumerate(e["templates"]):
                already_question = tmpl.rstrip().endswith("?")
                fill = dict(e.get("slot_fill", {}))
                fill.update(e.get("slot_fill_by_template", {}).get(str(ti), {}))
                for vi, (text, filled) in enumerate(
                        expand(tmpl, slots, fill, rng, n_variant)):
                    pair = f"{e['code']}-t{ti}-v{vi}"
                    forms = []
                    if already_question:
                        forms.append(("judge", text))
                    else:
                        forms.append(("generate", text))
                        j = to_judge(text)
                        if j is None:
                            unmatched.append(tmpl)
                        else:
                            forms.append(("judge", j))
                    for task_type, body in forms:
                        items.append({
                            "id": f"{d['domain']}-{next(seq):04d}",
                            "domain": d["domain"],
                            "domain_ko": d["domain_ko"],
                            "violation_code": e["code"],
                            "label": e["label"],
                            "kind": kind,
                            "task_type": task_type,
                            "pair_id": pair,
                            "user": body,
                            "verdict": e["verdict"],
                            "gold_points": e["gold_points"],
                            "legal_basis": e["legal_basis"],
                            "precedent": e.get("precedent", []),
                            "scoring": {
                                "must_mention": e.get("must_mention", []),
                                "fail_if_mention": e.get("fail_if_mention", []),
                            },
                            "review_status": "law_verified",
                            "mp_id": None, "mp_side": None,
                            "generator": {
                                "taxonomy": os.path.basename(domain_file),
                                "taxonomy_sha": tax_sha,
                                "template_idx": ti,
                                "slots": filled,
                            },
                        })
    return items, unmatched


def main():
    n_variant = int(os.environ.get("N_VARIANT", "2"))
    rng = random.Random(int(os.environ.get("SEED", "20260918")))
    all_items, all_unmatched = [], []
    for f in sorted(glob.glob(os.path.join(ROOT, "dataset", "taxonomy", "*.json"))):
        it, un = build(f, n_variant, rng)
        all_items += it
        all_unmatched += un

    if all_unmatched:
        print("[FAIL] judge 어미 변환 규칙 미매칭 템플릿:", file=sys.stderr)
        for t in sorted(set(all_unmatched)):
            print("   ", t, file=sys.stderr)
        raise SystemExit(1)

    out = os.path.join(ROOT, "dataset", "k_reggen_v0.1.jsonl")
    with open(out, "w", encoding="utf-8") as fh:
        for x in all_items:
            fh.write(json.dumps(x, ensure_ascii=False) + "\n")

    # 코어 서브셋: 유형×형식 대표 1개(최소쌍 제외). 반복 측정용.
    seen, core = set(), []
    for x in all_items:
        if x.get("mp_id"):
            continue
        k = (x["violation_code"], x["task_type"])
        if k not in seen:
            seen.add(k)
            core.append(x)
    with open(os.path.join(ROOT, "dataset", "k_reggen_core.jsonl"), "w", encoding="utf-8") as fh:
        for x in core:
            fh.write(json.dumps(x, ensure_ascii=False) + "\n")

    # 최소쌍 전용 세트: 경계 측정용. 쌍이 깨지지 않게 통째로 돌린다.
    mp = [x for x in all_items if x.get("mp_id")]
    with open(os.path.join(ROOT, "dataset", "k_reggen_minpairs.jsonl"), "w", encoding="utf-8") as fh:
        for x in mp:
            fh.write(json.dumps(x, ensure_ascii=False) + "\n")

    from collections import Counter
    dom = Counter(x["domain"] for x in all_items)
    kind = Counter(x["kind"] for x in all_items)
    task = Counter(x["task_type"] for x in all_items)
    pairs = {x["pair_id"] for x in all_items}
    full_pairs = Counter(x["pair_id"] for x in all_items)
    manifest = {
        "version": "0.1",
        "generated": "2026-09-18",
        "seed": int(os.environ.get("SEED", "20260918")),
        "n_variant": n_variant,
        "total": len(all_items),
        "by_domain": dict(dom),
        "by_kind": dict(kind),
        "by_task_type": dict(task),
        "pairs": len(pairs),
        "complete_pairs": sum(1 for v in full_pairs.values() if v == 2),
        "control_ratio": round(kind["control"] / len(all_items), 3),
        "minimal_pairs": len({x["mp_id"] for x in all_items if x.get("mp_id")}),
        "minimal_pair_items": sum(1 for x in all_items if x.get("mp_id")),
        "core_subset": len(core),
        "generator_sha": sha(os.path.abspath(__file__)),
        "taxonomy_sha": {os.path.basename(f): sha(f)
                         for f in sorted(glob.glob(os.path.join(ROOT, "dataset", "taxonomy", "*.json")))},
    }
    json.dump(manifest, open(os.path.join(ROOT, "dataset", "manifest.json"), "w",
                             encoding="utf-8"), ensure_ascii=False, indent=2)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
