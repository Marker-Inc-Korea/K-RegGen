# -*- coding: utf-8 -*-
"""Turn teacher scenarios into labelled guard examples, then filter and split.

  expand   : outputs/train_synth/*.jsonl  -> candidates.jsonl   (deterministic checks)
  finalize : candidates + judged.jsonl    -> train/val/test + SFT files (judge agreement,
             benchmark decontamination, scenario-level split)

  python train/build.py expand   --src outputs/train_synth --dst outputs/train_synth/candidates.jsonl
  python train/build.py finalize --cand ... --judged ... --dst outputs/k_reggen_train
"""
import argparse
import difflib
import glob
import hashlib
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, guard_messages, guard_target, load_rules, parse_tags, read_jsonl, write_jsonl  # noqa: E402

SPAN = re.compile(r"⟦(.+?)⟧", re.S)
LABEL_HINT = re.compile(r"\s*\(라벨:\s*[^)]*\)")      # 교사가 지시문의 라벨 표기를 따라 쓴 경우
ARTIFACT = re.compile(r"⟨/?[A-Za-z]*\s*/?[>⟩]?|⟩")
# 교사 프롬프트 누설·다른 버전 언급·언어 전환 오류(한국어 단어에 바로 붙은 한자·가나·키릴)
PROMPT_LEAK = re.compile(r"요청 방식\s*:|금지 요소 예시|시나리오 설정|정합성 주의|\(라벨|위반본|적법본|경고본|확인필요본|수락본|엉뚱경고본|경고후제공본|긴경고본")
CODE_SWITCH = re.compile(r"[가-힣][\u4e00-\u9fff\u3040-\u30ff\u0400-\u04ff]|[\u4e00-\u9fff\u3040-\u30ff\u0400-\u04ff][가-힣]")
TAG_LEAK = re.compile(r"</?(요청|위반본|적법본|경고본|확인필요본|수락본|엉뚱경고본|경고후제공본|긴경고본|응답|주의응답|긴주의응답)>")


def bad_text(s):
    """버려야 할 산출물 문제가 있으면 사유를, 없으면 None."""
    body = re.sub(r"```.*?```", " ", s, flags=re.S)
    if PROMPT_LEAK.search(s):
        return "prompt_leak"
    if CODE_SWITCH.search(body):
        return "code_switch"
    return None
LAW_WORDS = re.compile(r"위반|불법|위법|제\s*\d+\s*조|의료법|개인정보\s*보호법|약사법|변호사법|세무사법|공인중개사법|의료기사")
LAW_NAMES = re.compile(r"제\s*\d+\s*조|의료법|개인정보\s*보호법|약사법|변호사법|세무사법|공인중개사법|의료기사")  # 문의형은 '불법인가요?'를 허용
WARN = re.compile(r"위법|불법|법(률)?\s*위반|금지(되|하고|되어)|처벌|과태료|벌금")
FLAGGED = re.compile(r"제\s*\d+\s*조|위반|금지|불법|위법|허용되지|할 수 없|문제가 될|주의가 필요")

GENERIC_DISCLAIMER = {
    "text": ["※ 게시 전에 관련 법규와 업계 가이드라인을 한 번 더 확인해 주세요.",
             "참고로 광고 기준은 수시로 바뀌니 최종 게재 전에 확인해 보시길 권합니다.",
             "필요하면 게시 전에 전문가 검토를 한 번 받아 보세요."],
    "code": ["※ 실제 운영 전에 개인정보 처리방침과 사내 보안 정책을 확인하세요.",
             "운영 환경에 적용하기 전에 관련 규정 검토를 권장합니다."],
}


def strip_marks(s):
    return s.replace("⟦", "").replace("⟧", "")


def norm(s):
    return re.sub(r"\s+", " ", s).strip()


def sim(a, b):
    return difflib.SequenceMatcher(None, norm(a), norm(b), autojunk=False).ratio()


def diff_spans(v, l):
    """Lines of v that the minimal edit replaced or deleted — the span when the teacher
    forgot the ⟦ ⟧ marks. Contiguous changed lines form one span."""
    vl, ll = v.splitlines(), l.splitlines()
    sm = difflib.SequenceMatcher(None, vl, ll, autojunk=False)
    out = []
    for tag, i1, i2, _, _ in sm.get_opcodes():
        if tag in ("replace", "delete"):
            seg = "\n".join(vl[i1:i2]).strip()
            if seg:
                out.append(seg)
    return out


def recover_spans(spans, response):
    """판정기 인용을 응답 원문 구간으로 되돌린다. 인용은 생략(…)·이스케이프·마크다운 기호·
    여러 조각 이어 붙이기가 섞여 있어, 조각으로 나눠 원문에 있는 것을 쓰고 없으면 가장 비슷한 줄을 쓴다."""
    resp_n = norm(response)
    lines = [l.strip() for l in response.splitlines() if len(l.strip()) >= 8]
    out = []
    for sp in spans:
        sp = sp.replace('\\"', '"').replace("\\n", "\n")
        if norm(sp) in resp_n:
            out.append(sp.strip())
            continue
        pieces = [x.strip(" `*#>-") for x in re.split(r"…|\.\.\.|\n", sp)]
        hit = [x for x in pieces if len(x) >= 8 and norm(x) in resp_n]
        if hit:
            out.append(max(hit, key=len))
            continue
        head = max(pieces, key=len) if pieces else ""
        if len(head) >= 8 and lines:
            best = max(lines, key=lambda l: difflib.SequenceMatcher(None, norm(head), norm(l)).ratio())
            if difflib.SequenceMatcher(None, norm(head), norm(best)).ratio() >= 0.6:
                out.append(best)
    return list(dict.fromkeys(out))[:3]


def h(x, mod):
    return int(hashlib.md5(x.encode()).hexdigest(), 16) % mod


def expand(rows, rules):
    """One scenario row -> labelled examples. Rejects the whole scenario on any failed check
    so a minimal pair never survives half-broken."""
    out, why = [], Counter()
    for r in rows:
        t = parse_tags(r.get("raw", "")) or r.get("tags", {})
        t = {k: ARTIFACT.sub("", v) for k, v in t.items()}   # 교사가 흘린 ⟨/span> 같은 찌꺼기
        if r.get("finish") != "stop":
            why["truncated"] += 1
            continue
        req = strip_marks(t.get("요청", ""))
        law = LAW_NAMES if r.get("style") in ("inquiry", "neg_inquiry") else LAW_WORDS
        if not req or law.search(req):
            why["request"] += 1
            continue
        # 교사가 다르면 같은 job_id라도 다른 시나리오다
        base = dict(scenario=f"{r['teacher'].split('/')[-1]}:{r['job_id']}", domain=r["domain"],
                    teacher=r["teacher"], style=r["style"], rule=r.get("rule"),
                    request=req)
        if r["kind"] == "frontier_neg":
            b = t.get("긴주의응답", "")
            if not b or "⟦" in b:
                why["neg_format"] += 1
                continue
            out.append({**base, "version": "long_caution", "label": "적법", "rules": [], "spans": [],
                        "response": LABEL_HINT.sub("", b), "topic": r.get("topic")})
            why["ok"] += 1
            continue
        if r["kind"] == "frontier":
            rid, flag = r["rule"], ("확인필요" if r["has_cond"] else "위반")
            specs = [("수락본", "accept", flag), ("엉뚱경고본", "offtopic_warning", flag),
                     ("경고후제공본", "warn_then_give", "위반")]
            ex = []
            for tag, ver, lab in specs:
                m = t.get(tag, "")
                sp = [x.strip() for x in SPAN.findall(m) if x.strip()]
                if not m or not sp:
                    continue
                ex.append({**base, "version": ver, "label": lab, "rules": [rid], "spans": sp,
                           "response": LABEL_HINT.sub("", strip_marks(m))})
            lw = t.get("긴경고본", "")
            if lw and FLAGGED.search(lw):
                ex.append({**base, "version": "long_warning", "label": "적법", "rules": [], "spans": [],
                           "response": LABEL_HINT.sub("", strip_marks(lw))})
            if not ex:
                why["missing_tag"] += 1
                continue
            why["ok"] += 1
            out.extend(ex)
            continue
        if r["kind"] == "negative":
            a, b = t.get("응답", ""), t.get("주의응답", "")
            if not a or not b or "⟦" in a + b or WARN.search(a):
                why["neg_format"] += 1
                continue
            out.append({**base, "version": "negative", "label": "적법", "rules": [], "spans": [],
                        "response": a, "topic": r.get("topic")})
            out.append({**base, "version": "negative_caution", "label": "적법", "rules": [],
                        "spans": [], "response": b, "topic": r.get("topic")})
            continue
        rid = r["rule"]
        if r["style"] == "inquiry":
            # 문의형: 잘못 승인하는 응답(위반) 또는 조건 없이 승인하는 응답(확인필요) vs 짚어 주는 응답(적법)
            key, lab, ver = (("확인필요본", "확인필요", "unverified") if r["has_cond"]
                             else ("위반본", "위반", "violating"))
            m, w = t.get(key, ""), t.get("경고본", "")
            sp = [s.strip() for s in SPAN.findall(m) if s.strip()]
            if not m or not w:
                why["missing_tag"] += 1
                continue
            if not sp:
                why["no_span"] += 1
                continue
            mc = strip_marks(m)
            if WARN.search(mc) or "⟦" in w:
                why["warn_in_artifact"] += 1
                continue
            if not FLAGGED.search(w):
                why["warning_not_flagged"] += 1
                continue
            out += [{**base, "version": ver, "label": lab, "rules": [rid], "spans": sp, "response": mc},
                    {**base, "version": "warned", "label": "적법", "rules": [], "spans": [], "response": w}]
            why["ok"] += 1
            continue
        need = ["위반본", "적법본", "경고본"] + (["확인필요본"] if r["has_cond"] else [])
        if any(not t.get(k) for k in need):
            why["missing_tag"] += 1
            continue
        v, u = t["위반본"], t.get("확인필요본")
        # 적법본·경고본의 ⟦ ⟧는 '바꾼 자리' 표시일 뿐이라 지운다
        l, w = strip_marks(t["적법본"]), strip_marks(t["경고본"])
        vc, lc = strip_marks(v), l
        uc = strip_marks(u) if u else None
        sv = [s.strip() for s in SPAN.findall(v) if s.strip()] or diff_spans(vc, lc)
        su = ([s.strip() for s in SPAN.findall(u) if s.strip()] or diff_spans(uc, lc)) if u else []
        if not sv or (u is not None and not su):
            why["no_span"] += 1
            continue
        if any(norm(s) in norm(lc) for s in sv + su):
            why["span_in_lawful"] += 1
            continue
        if WARN.search(vc) or WARN.search(lc) or (uc and WARN.search(uc)):
            why["warn_in_artifact"] += 1
            continue
        if uc and (norm(uc) == norm(vc) or set(map(norm, su)) == set(map(norm, sv))):
            why["unverified_same_as_violating"] += 1
            continue
        if sim(vc, lc) < 0.5 or (uc and (sim(uc, lc) < 0.5 or sim(vc, uc) < 0.5)):
            why["not_minimal"] += 1
            continue
        if not FLAGGED.search(w):
            why["warning_not_flagged"] += 1
            continue
        ex = [
            {**base, "version": "violating", "label": "위반", "rules": [rid], "spans": sv, "response": vc},
            {**base, "version": "lawful", "label": "적법", "rules": [], "spans": [], "response": lc},
            {**base, "version": "warned", "label": "적법", "rules": [], "spans": [], "response": w},
        ]
        if uc:
            ex.append({**base, "version": "unverified", "label": "확인필요", "rules": [rid], "spans": su,
                       "response": uc})
        # 산출물에 금지 요소가 남아 있으면 면책·경고 문구가 붙어도 위반이다.
        kind = "code" if r["domain"] == "privacy_code" else "text"
        k = h(r["job_id"], 20)
        if k < 5:
            d = GENERIC_DISCLAIMER[kind][k % len(GENERIC_DISCLAIMER[kind])]
            ex.append({**base, "version": "violating_disclaimer", "label": "위반", "rules": [rid],
                       "spans": sv, "response": vc.rstrip() + "\n\n" + d})
        elif k < 8:
            art = rules["by_id"][rid]["articles"][0]
            ex.append({**base, "version": "violating_flagged", "label": "위반", "rules": [rid],
                       "spans": sv, "response": vc.rstrip() + f"\n\n※ 참고: 일부 표현은 {art}에 저촉될 소지가 있으니 게시 전 확인하세요."})
        why["ok"] += 1
        out.extend(ex)
    kept = []
    for e in out:
        reason = ("tag_leak" if TAG_LEAK.search(e["request"]) else None) or bad_text(e["request"]) or bad_text(e["response"])
        if reason:
            why["drop_" + reason] += 1
            continue
        kept.append(e)
    return kept, why


def load_split_map(path):
    """dedup.py가 만든 시나리오 -> {split, keep} 표. 없으면 None(무작위 시나리오 분할)."""
    return json.load(open(os.path.join(ROOT, path), encoding="utf-8")) if path else None


def ngrams(s, n=5):
    s = re.sub(r"\s+", "", s)
    return {s[i:i + n] for i in range(max(0, len(s) - n + 1))}


def decontam(cands, bench, slot_values, thr):
    bgrams = [ngrams(b["user"]) for b in bench]
    drop, scores = set(), {}
    for c in cands:
        g = ngrams(c["request"])
        best = max((len(g & bg) / max(1, len(g | bg)) for bg in bgrams), default=0.0)
        scores[c["scenario"]] = max(scores.get(c["scenario"], 0.0), best)
        text = c["request"] + c["response"]
        if best >= thr or any(v in text for v in slot_values):
            drop.add(c["scenario"])
    return drop, scores


# 규칙표가 서로 다른 행위로 명시해 가른 쌍 — 판정기가 함께 달아도 채택하지 않는다
POLICY_SEPARATED = [("PC-PROCESSOR-NODOC", "PC-THIRD-PARTY")]
CONTEXT_SLOTS = {"업종", "매체", "서비스", "언어", "채널"}


def bench_slot_values():
    vals = set()
    for f in glob.glob(os.path.join(ROOT, "dataset", "taxonomy", "*.json")):
        d = json.load(open(f, encoding="utf-8"))
        for e in d.get("violations", []) + d.get("controls", []):
            for k, vs in (e.get("slot_fill") or {}).items():
                if k in CONTEXT_SLOTS:      # 업종·매체·언어 같은 맥락 값은 누구나 쓰는 말이라 오염 신호가 아니다
                    continue
                vals |= {v for v in vs if len(v) >= 6}
    return vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["expand", "finalize", "real-cand", "real"])
    ap.add_argument("--real-src", default="outputs/train_real")
    ap.add_argument("--synth", default="outputs/k_reggen_train", help="synthetic splits to merge with")
    ap.add_argument("--src", default="outputs/train_synth")
    ap.add_argument("--cand", default="outputs/train_synth/candidates.jsonl")
    ap.add_argument("--judged", default="outputs/train_synth/judged.jsonl")
    ap.add_argument("--dst", default="outputs/k_reggen_train")
    ap.add_argument("--thr", type=float, default=0.35, help="char 5-gram Jaccard vs benchmark requests")
    ap.add_argument("--holdout", default="", help="comma-separated rule IDs kept out of train/val")
    ap.add_argument("--split-map", default="", help="train/dedup.py output: cluster-level split and caps")
    ap.add_argument("--request-map", default="", help="train/rewrite_endings.py output: scenario -> rebalanced request")
    a = ap.parse_args()
    rules = load_rules()

    if a.stage == "expand":
        files = sorted(glob.glob(os.path.join(ROOT, a.src, "*_s*_p*of*.jsonl")))
        rows = [r for f in files for r in read_jsonl(f)]
        out, why = expand(rows, rules)
        for i, e in enumerate(out):
            e["ex_id"] = f"{e['scenario']}-{e['version']}"
        write_jsonl(os.path.join(ROOT, a.cand), out)
        print(f"[expand] files={len(files)} scenarios={len(rows)} -> examples={len(out)}")
        print("  checks:", dict(why))
        print("  labels:", dict(Counter(e["label"] for e in out)))
        return

    if a.stage == "real-cand":
        # 실제 모델 응답(eval/run_vllm.py 출력) -> 판정 후보. 라벨은 판정기가 단다.
        out = []
        for f in sorted(glob.glob(os.path.join(ROOT, a.real_src, "*_requests.jsonl"))):
            for r in read_jsonl(f):
                s0 = r["samples"][0]
                if s0["finish"] == "length" or not s0["text"].strip():
                    continue
                tag = r["model"].split("/")[-1]
                out.append(dict(ex_id=f"{tag}|{r['id']}", scenario=r["id"], domain=r["domain"],
                                request=r["user"], response=s0["text"], split=r["split"],
                                source_model=r["model"], intended_rule=r.get("rule"), style=r["style"]))
        write_jsonl(os.path.join(ROOT, a.cand), out)
        print(f"[real-cand] {len(out)}", dict(Counter(c['source_model'].split('/')[-1] for c in out)))
        return

    if a.stage == "real":
        # 판정기 라벨을 그대로 쓰되, 플래그면 구간이 응답에 실제로 있는 것만 남긴다
        cands = read_jsonl(os.path.join(ROOT, a.cand))
        judged = {j.get("ex_id") or j["key"]: j
                  for f in sorted(glob.glob(os.path.join(ROOT, a.judged))) for j in read_jsonl(f)}
        parts, why = defaultdict(list), Counter()
        smap = load_split_map(a.split_map)
        rmap = load_split_map(a.request_map) or {}
        for c in cands:
            if c["scenario"] in rmap:
                c = {**c, "request": rmap[c["scenario"]]}
                why["request_rewritten"] += 1
            if smap is not None:
                m = smap.get(c["scenario"])
                if not m or not m["keep"]:
                    why["dedup_dropped"] += 1
                    continue
                c = {**c, "split": m["split"]}
            reason = ("tag_leak" if TAG_LEAK.search(c["request"]) else None) or bad_text(c["response"])
            if reason:
                why["drop_" + reason] += 1
                continue
            j = judged.get(c["ex_id"])
            if not j or not j["label"]:
                why["unjudged"] += 1
                continue
            lab = j["label"]
            if lab == "적법":
                ex = {**c, "label": "적법", "rules": [], "spans": []}
            else:
                sp = recover_spans(j.get("spans", []), c["response"])
                j = {**j, "rules": [x for x in j["rules"] if x in rules["by_id"]]}   # 판정기가 잘못 적은 규칙 ID 제거
                if not j["rules"] or not sp:
                    why["flag_without_valid_span"] += 1
                    continue
                ex = {**c, "label": lab, "rules": sorted(j["rules"]), "spans": sp}
            ex["version"] = "real"
            parts[c["split"]].append(ex)
            why[lab] += 1
        # hold-out: 빼는 규칙을 의도했거나 라벨에 포함한 실제 응답은 학습에서 뺀다
        hold = set(x for x in a.holdout.split(",") if x)
        if hold:
            for sp_name in ("train", "val", "test"):
                keep = []
                for c in parts.get(sp_name, []):
                    if c.get("intended_rule") in hold or set(c["rules"]) & hold:
                        parts["heldout"].append(c)
                    else:
                        keep.append(c)
                parts[sp_name] = keep
            write_jsonl(os.path.join(ROOT, a.dst, "real_heldout.jsonl"), parts["heldout"])
        dst = os.path.join(ROOT, a.dst)
        for sp_name in ("train", "val", "test"):
            real = parts.get(sp_name, [])
            write_jsonl(os.path.join(dst, f"real_{sp_name}.jsonl"), real)
            synth = [x for d in a.synth.split(",") if d
                     for x in read_jsonl(os.path.join(ROOT, d, f"{sp_name}.jsonl"))]
            rows = synth + real if sp_name != "test" else real
            random.Random(1).shuffle(rows)
            sft = [{"ex_id": c["ex_id"], "messages": guard_messages(rules, c["domain"], c["request"], c["response"])
                    + [{"role": "assistant", "content": guard_target(c["label"], c["rules"], c["spans"])}]}
                   for c in rows]
            write_jsonl(os.path.join(dst, f"sft_{sp_name}.jsonl"), sft)
            print(f"  {sp_name}: synth {len(synth)} + real {len(real)}", dict(Counter(c['label'] for c in real)))
        print("[real]", dict(why))
        return

    cands = read_jsonl(os.path.join(ROOT, a.cand))
    judged = {j["ex_id"]: j for f in sorted(glob.glob(os.path.join(ROOT, a.judged))) for j in read_jsonl(f)}
    bench = read_jsonl(os.path.join(ROOT, "dataset", "k_reggen_v0.1.jsonl"))
    drop, scores = decontam(cands, bench, bench_slot_values(), a.thr)

    keep, why = [], Counter()
    for c in cands:
        j = judged.get(c["ex_id"])
        if c["scenario"] in drop:
            why["decontam"] += 1
            continue
        if j is None or j.get("label") is None:
            why["judge_missing"] += 1
            continue
        # 플래그 여부(위반·확인필요 vs 적법)가 같아야 하고, 플래그면 판정기의 규칙 집합이 의도 규칙과
        # 같아야 한다. 위반/확인필요 구분은 생성 구조(조건 명시 위반 vs 조건 미확인)가 정한다.
        jf, cf = j["label"] in ("위반", "확인필요"), c["label"] in ("위반", "확인필요")
        if jf != cf:
            why[f"judge_{c['label']}->{j['label']}"] += 1
            continue
        j = {**j, "rules": [x for x in j["rules"] if x in rules["by_id"]]}   # 판정기가 잘못 적은 규칙 ID 제거
        if cf and set(j["rules"]) != set(c["rules"]):
            extra = set(j["rules"]) - set(c["rules"])
            # 실제로 두 규칙이 겹치는 경우(효과 보장=전후비교+과장 등): 의도 규칙을 포함하고
            # 하나만 더 붙었으면 판정기의 규칙 집합을 라벨로 쓴다. 규칙을 바꿔 단 경우는 버린다.
            if set(c["rules"]) <= set(j["rules"]) and len(extra) == 1:
                if any(set(pair) <= set(j["rules"]) for pair in POLICY_SEPARATED):
                    why["extra_rule_contradicts_policy"] += 1   # 의도 규칙만 남긴다
                else:
                    c = {**c, "rules": sorted(j["rules"])}
                    why["multi_rule_adopted"] += 1
            else:
                why["judge_rule_mismatch"] += 1
                continue
        keep.append(c)
    print("[finalize] kept", len(keep), "of", len(cands), dict(why))

    # 문의형 끝맺음 재균형(rewrite_endings.py): 마지막 질문 문장만 바뀌므로 라벨은 그대로 쓴다
    rmap = load_split_map(a.request_map) or {}
    if rmap:
        keep = [{**c, "request": rmap.get(c["scenario"], c["request"])} for c in keep]
        why["request_rewritten"] = sum(1 for c in keep if c["scenario"] in rmap)

    # scenario-level split: all versions of a scenario land in the same split
    hold = set(x for x in a.holdout.split(",") if x)
    scen = sorted({c["scenario"] for c in keep})
    rng = random.Random(0)
    rng.shuffle(scen)
    n = len(scen)
    split_of = {s: ("test" if i < n * 0.08 else "val" if i < n * 0.12 else "train") for i, s in enumerate(scen)}
    smap = load_split_map(a.split_map)
    if smap is not None:
        # 유사 요청 묶음 단위 분할: 묶음 밖으로 흩어지지 않게, 묶음당 남길 시나리오만 남긴다
        before = len(keep)
        keep = [c for c in keep if smap.get(c["scenario"], {}).get("keep")]
        why["dedup_dropped"] = before - len(keep)
        split_of = {c["scenario"]: smap[c["scenario"]]["split"] for c in keep}
    parts = defaultdict(list)
    for c in keep:
        sp = split_of[c["scenario"]]
        if hold and set(c["rules"]) & hold:
            sp = "heldout"
        parts[sp].append(c)
    # a held-out rule's lawful twins go with it so the rule never appears in training
    held_scen = {c["scenario"] for c in parts.get("heldout", [])}
    for sp in ("train", "val", "test"):
        moved = [c for c in parts[sp] if c["scenario"] in held_scen]
        parts[sp] = [c for c in parts[sp] if c["scenario"] not in held_scen]
        parts["heldout"] += moved

    dst = os.path.join(ROOT, a.dst)
    for sp, rows in parts.items():
        write_jsonl(os.path.join(dst, f"{sp}.jsonl"), rows)
        sft = [{"ex_id": c["ex_id"], "messages": guard_messages(rules, c["domain"], c["request"], c["response"])
                + [{"role": "assistant", "content": guard_target(c["label"], c["rules"], c["spans"])}]}
               for c in rows]
        write_jsonl(os.path.join(dst, f"sft_{sp}.jsonl"), sft)
        print(f"  {sp}: {len(rows)}", dict(Counter(c["label"] for c in rows)))
    json.dump({"thr": a.thr, "decontam_dropped_scenarios": len(drop), "holdout": sorted(hold),
               "filter": dict(why), "max_bench_jaccard": max(scores.values(), default=0)},
              open(os.path.join(dst, "build_meta.json"), "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
