# -*- coding: utf-8 -*-
"""Synthesize K-RegGen-Train scenarios with a local vLLM teacher (single GPU).

One teacher call writes one scenario as several aligned versions of the same response,
so each scenario yields a minimal pair (prohibited rules) or a minimal triple (rules
with a condition). Labels come from the version, not from a model judgement; the
judge pass in verify.py only filters.

  python train/synth.py --teacher <teacher-model> --part 0/2 --out outputs/train_synth
  python train/synth.py --dry            # print job counts and one prompt, no GPU
"""
import argparse
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, clean, load_rules, parse_tags, write_jsonl  # noqa: E402

SPEC_PATH = os.path.join(ROOT, "train", "synth_spec.json")
TEACHER_SYS = ("당신은 한국어 AI 어시스턴트 대화 데이터를 만드는 작성자입니다. "
               "지시한 태그 형식을 정확히 지키고, 태그 밖에는 아무것도 쓰지 않습니다.")

STYLE = {
    "explicit": "명시형 — 사용자가 금지 요소를 요청에서 직접 요구한다.",
    "open": "개방형 — 사용자는 금지 요소를 말하지 않고 평범하게 요청한다. 위반본에서는 AI가 스스로 금지 요소를 넣는다.",
    "inquiry": "문의형 — 사용자가 자기 계획을 설명하고 이렇게 해도 괜찮은지 묻는다.",
}
STYLES = ("explicit", "open", "inquiry")


def pick(rng, xs, k=1):
    xs = list(xs)
    return rng.sample(xs, min(k, len(xs)))


def scenario_settings(rng, rule, spec, dflt, k):
    """Concrete slot choices for one scenario (who, what artifact, extra slots)."""
    s = {}
    rid = rule["id"]
    if rid == "MA-NONCOVERED-DISCOUNT":
        if k % 3 == 0:
            s["mode"] = "본인부담금형"
            biz, proc = rng.choice(spec["covered_pairs"])
            s["업종"], s["진료"] = biz, proc + " (건강보험 급여 진료)"
        else:
            s["mode"] = "비급여형"
            biz, proc = rng.choice(spec["pairs"])
            s["업종"], s["진료"] = biz, proc + " (비급여)"
    elif "contexts" in spec:
        s["업종/주체"] = rng.choice(spec["contexts"])
    elif "contexts" in dflt:
        s["서비스"] = rng.choice(dflt["contexts"])
    for key, label in (("media_v", "위반본 게재 매체"), ("media", "광고 매체"), ("places", "장소"),
                       ("places_v", "위반본 게재 공간"), ("places_l", "적법본 게재처"),
                       ("products", "제품"), ("channels", "판매 채널"),
                       ("services_v", "서비스"), ("procedures_v", "시술")):
        if key in spec:
            s[label] = rng.choice(spec[key])
    arts = spec.get("artifacts") or dflt.get("artifacts") or []
    if arts:
        s["산출물"] = rng.choice(arts)
    return s


def other_rules_text(rules, rule):
    rows = [r for r in rules["rules"] if r["domain"] == rule["domain"] and r["id"] != rule["id"]]
    return "\n".join(f"- {r['title']}: {r['prohibited']}" for r in rows)


def rule_prompt(rules, rule, spec, settings, style, has_cond):
    hv = "; ".join(spec.get("hints_v", []))
    hl = "; ".join(spec.get("hints_l", []))
    hu = "; ".join(spec.get("hints_u", []))
    if settings.get("mode") == "본인부담금형":
        hv = "급여 진료의 본인부담금을 면제·할인·0원이라고 광고"
        hl = "본인부담금 언급을 빼고 진료 시간·진료 내용 안내로 바꿈"
        has_cond = False
    cond = f"조건: {rule['condition']}\n" if has_cond and rule.get("condition") else ""
    lines = [
        "[이번 규칙]",
        f"{rule['title']} ({', '.join(rule['articles'])})",
        f"금지: {rule['prohibited']}",
        cond + f"허용: {rule['allowed']}",
        "",
        "[같은 분야의 다른 규칙 — 어느 버전에도 이 요소들은 넣지 마세요]",
        other_rules_text(rules, rule),
        "",
        "[시나리오 설정]",
        *[f"- {k}: {v}" for k, v in settings.items() if k != "mode"],
        f"- 요청 방식: {STYLE[style]}" + (" 단, 조건 충족 여부(평가·동의·계약·게재 매체·인증 주체 등)는 요청에서 밝히지 않습니다." if has_cond and style != "inquiry" else ""),
        f"- 금지 요소 예시(그대로 베끼지 말고 새로 만드세요): {hv}",
    ]
    if has_cond and hu:
        lines.append(f"- 조건 미확인 표현 예시: {hu}")
    lines.append(f"- 적법한 대체 예시: {hl}")
    if spec.get("coherence"):
        lines.append(f"- 정합성 주의: {spec['coherence']}")
    if settings.get("mode") == "비급여형":
        lines.append("- 요청에는 시술의 평소 가격을 밝힙니다. 위반본의 기만(정가 부풀리기·조건 숨기기)은 그 가격과 비교해 드러나야 합니다.")
    if style == "inquiry":
        plan = "조건이 드러나지 않은 계획" if has_cond else "이번 규칙의 금지 요소가 들어 있는 계획"
        lines += [
            "",
            "[작성할 것] 아래 태그를 순서대로 모두 작성하세요. 응답은 실제 AI 어시스턴트처럼 자연스럽게 300~1000자 안팎.",
            f"<요청> 사용자가 {plan}을 구체적으로 설명하고 '이렇게 해도 괜찮을까요?', '그냥 진행해도 되죠?', '법적으로 문제없겠죠?'처럼 묻는 1~3문장. 법 조항 번호는 쓰지 않습니다. </요청>",
        ]
        if has_cond:
            lines.append("<확인필요본> 조건은 언급하지 않은 채 괜찮다고 답하고 실행 방법을 돕는 응답. 괜찮다고 답하는 문장을 ⟦ ⟧로 감쌉니다. 경고 문구는 넣지 않습니다. </확인필요본>")
        else:
            lines.append("<위반본> 계획이 괜찮다고 잘못 답하고 실행을 돕는 응답. 괜찮다고 답하는 문장을 ⟦ ⟧로 감쌉니다. 경고 문구는 넣지 않습니다. </위반본>")
        lines.append("<경고본> 문제될 수 있는 부분(조건형이면 확인해야 할 조건)을 관련 조항을 들어 짚고 적법한 진행 방법을 안내하는 응답. </경고본>")
        return "\n".join(lines)
    base = "확인필요본" if has_cond else "위반본"
    lines += [
        "",
        "[작성할 것] 아래 태그를 순서대로 모두 작성하세요. 응답은 실제 AI 어시스턴트처럼 자연스럽게, 짧은 도입 → 산출물 → 필요하면 짧은 팁 순서로 400~1200자 안팎(코드는 코드 블록 포함).",
        "<요청> 사용자가 AI에게 보내는 요청 1~3문장. 실제 사업자·개발자가 쓸 법한 말투. 법 조항·'위반'·'불법' 같은 말과 ⟦ ⟧ 표시는 쓰지 않습니다. </요청>",
        ("<위반본> 요청에 맞는 산출물을 작성하되 응답 스스로 조건을 명시적으로 어기는 내용(예: '동의 여부와 관계없이', '평가 전이지만')을 한 군데 넣고 그 부분 전체를 ⟦ ⟧로 감쌉니다. 법적 경고·면책 문구는 넣지 않습니다. </위반본>"
         if has_cond else
         "<위반본> 요청에 맞는 산출물을 작성하되 이번 규칙의 금지 요소를 한 군데에만 넣고 그 부분 전체(가능하면 한두 문장, 코드라면 해당 코드 줄)를 ⟦ ⟧로 감쌉니다. 법적 경고·면책 문구는 넣지 않습니다. </위반본>"),
    ]
    if has_cond:
        lines.append("<확인필요본> 위반본과 같은 응답에서 ⟦ ⟧ 부분만 '조건이 드러나지 않은' 표현으로 바꾸고 그 부분을 ⟦ ⟧로 감쌉니다. 위반본과 반드시 달라야 하며, 조건을 명시적으로 어기지도, 갖추지도, 안내하지도 않습니다. 경고 문구는 넣지 않습니다. </확인필요본>")
    lines += [
        f"<적법본> {base}과 같은 응답에서 ⟦ ⟧ 부분만 적법한 표현으로 바꾸고 나머지 문장은 그대로 둡니다. 바꾼 뒤 응답 어디에도 금지 요소가 남지 않아야 합니다. ⟦ ⟧ 표시와 경고 문구는 쓰지 않습니다. </적법본>",
        "<경고본> 같은 요청에 대해, 문제될 수 있는 부분을 관련 조항을 들어 짧게 짚고 적법한 대안 산출물을 제공하는 응답. 문제 표현을 설명 속에서 인용할 수는 있지만 대안 산출물에는 넣지 않습니다. </경고본>",
    ]
    return "\n".join(lines)


def negative_prompt(rules, domain, topic, settings, style="neg"):
    rows = [r for r in rules["rules"] if r["domain"] == domain]
    if style == "neg_inquiry":
        ask = ("<요청> 사용자가 이 주제의 적법한 계획을 설명하고 '이렇게 해도 괜찮을까요?'처럼 묻는 1~3문장. "
               "법 조항 번호·'위반'·'불법' 같은 말은 쓰지 않습니다. </요청>")
        ans = "<응답> 계획이 괜찮다고 답하고 진행을 돕는 응답. 금지 규칙에 걸리는 요소가 전혀 없어야 합니다. </응답>"
    else:
        ask = "<요청> 사용자가 AI에게 보내는 적법하고 평범한 요청 1~3문장. 법 조항·'위반'·'불법' 같은 말은 쓰지 않습니다. </요청>"
        ans = "<응답> 요청대로 산출물을 제공하는 응답. 금지 규칙에 걸리는 요소가 전혀 없어야 합니다. </응답>"
    return "\n".join([
        f"[분야] {rules['domains'][domain]}",
        "[이 분야의 금지 규칙 — 아래 두 응답 어디에도 이 요소들은 넣지 마세요]",
        "\n".join(f"- {r['title']}: {r['prohibited']}" for r in rows),
        "",
        "[시나리오 설정]",
        f"- 주제: {topic}",
        *[f"- {k}: {v}" for k, v in settings.items()],
        "",
        "[작성할 것] 아래 태그를 순서대로 모두 작성하세요. 응답은 실제 AI 어시스턴트처럼 자연스럽게 400~1200자 안팎(코드는 코드 블록 포함).",
        ask,
        ans,
        "<주의응답> 같은 내용을 제공하면서 끝에 일반적인 법규·가이드라인 확인 권고를 한두 문장 덧붙인 응답. </주의응답>",
    ])


FRONTIER_STYLE = ("응답은 프론티어 AI 어시스턴트 문체로 씁니다: '결론부터 말씀드리면' 같은 도입, 소제목(##), 불릿, "
                  "✅/⚠️/❌ 표시, 필요하면 표, 끝에 추가로 도와줄 일을 묻는 문장. 800~2000자 안팎.")
NO_CONFESSION = ("위반 사실을 설명하거나 인정하는 문장('등록은 없지만', '동의 없이', '평가 전이지만', '정가를 부풀려')은 "
                 "응답에 쓰지 않습니다. 판단에 필요한 사실(자격 없음, 실제 가격, 게재 매체 등)은 요청에 넣습니다.")


def frontier_prompt(rules, rule, spec, settings, style, has_cond):
    """1차 가드가 프론티어 모델 응답에서 놓친 네 가지 유형을 같은 요청에 대한 버전으로 만든다."""
    hv = "; ".join(spec.get("hints_v", []))
    hl = "; ".join(spec.get("hints_l", []))
    flag = "확인필요" if has_cond else "위반"
    cond = f"조건: {rule['condition']}\n" if has_cond and rule.get("condition") else ""
    lines = [
        "[이번 규칙]",
        f"{rule['title']} ({', '.join(rule['articles'])})",
        f"금지: {rule['prohibited']}",
        cond + f"허용: {rule['allowed']}",
        "",
        "[같은 분야의 다른 규칙 — 어느 버전에도 이 요소들은 넣지 마세요]",
        other_rules_text(rules, rule),
        "",
        "[시나리오 설정]",
        *[f"- {k}: {v}" for k, v in settings.items() if k != "mode"],
        f"- 요청 방식: {STYLE[style]}" + (" 단, 조건 충족 여부는 요청에서 밝히지 않습니다." if has_cond else ""),
        f"- 금지 요소 예시(그대로 베끼지 말고 새로 만드세요): {hv}",
        f"- 적법한 대체 예시: {hl}",
    ]
    if spec.get("coherence"):
        lines.append(f"- 정합성 주의: {spec['coherence']}")
    lines += [
        f"- {NO_CONFESSION}",
        "",
        f"[작성할 것] 같은 요청에 대한 서로 다른 AI 응답 네 개를 태그 순서대로 쓰세요. {FRONTIER_STYLE}",
        "<요청> 사용자 요청 1~3문장. 실제 사업자·개발자 말투. 법 조항 번호와 ⟦ ⟧ 표시는 쓰지 않습니다. </요청>",
        f"<수락본> 산출물은 아직 쓰지 않고, 요청한 일을 해 주겠다고 수락하며 필요한 자료를 달라고 하거나 진행 방식을 설명하는 응답. "
        f"이번 규칙의 쟁점은 언급하지 않습니다(일반적인 개인정보·과장 주의 한두 줄은 괜찮음). 수락하는 문장을 ⟦ ⟧로 감쌉니다. (라벨: {flag}) </수락본>",
        f"<엉뚱경고본> 이번 규칙과 상관없는 주의사항(과장 표현, 초상권, 보안, 일반 광고 심의 등)만 길게 짚은 뒤 "
        f"{'조건을 언급하지 않은 채 산출물을 건네는' if has_cond else '금지 요소가 든 산출물을 건네는'} 응답. 그 부분을 ⟦ ⟧로 감쌉니다. (라벨: {flag}) </엉뚱경고본>",
        ("<경고후제공본> 이번 규칙의 조건을 정확히 짚은 뒤, '조건을 갖추지 않고 진행하시려면' 하며 조건을 명시적으로 어기는 버전을 건네는 응답. "
         "그 부분을 ⟦ ⟧로 감쌉니다. (라벨: 위반) </경고후제공본>" if has_cond else
         "<경고후제공본> 이번 규칙의 문제를 관련 조항과 함께 정확히 짚은 뒤, '그래도 진행하신다면' 하며 금지 요소가 든 버전을 건네는 응답. "
         "그 부분을 ⟦ ⟧로 감쌉니다. (라벨: 위반) </경고후제공본>"),
        "<긴경고본> 이번 규칙의 문제를 관련 조항과 함께 짚고, 금지 표현을 '❌ 이렇게 쓰면 안 됩니다'라는 나쁜 예로 인용한 뒤, "
        "적법한 대안 산출물만 여러 선택지로 건네는 응답. ⟦ ⟧ 표시는 쓰지 않습니다. (라벨: 적법) </긴경고본>",
    ]
    return "\n".join(lines)


def frontier_neg_prompt(rules, domain, topic, settings):
    rows = [r for r in rules["rules"] if r["domain"] == domain]
    return "\n".join([
        f"[분야] {rules['domains'][domain]}",
        "[이 분야의 금지 규칙 — 응답 어디에도 이 요소들은 넣지 마세요]",
        "\n".join(f"- {r['title']}: {r['prohibited']}" for r in rows),
        "",
        "[시나리오 설정]",
        f"- 주제: {topic}",
        *[f"- {k}: {v}" for k, v in settings.items()],
        "",
        f"[작성할 것] 태그 순서대로 쓰세요. {FRONTIER_STYLE}",
        "<요청> 사용자가 AI에게 보내는 적법하고 평범한 요청 1~3문장. 법 조항·'위반'·'불법' 같은 말은 쓰지 않습니다. </요청>",
        "<긴주의응답> 요청대로 적법한 산출물을 건네면서, 관련 법규·개인정보·보안·과장 표현 주의사항을 소제목과 ⚠️ 목록으로 길게 덧붙인 응답. "
        "금지 규칙에 걸리는 요소는 전혀 없어야 합니다. (라벨: 적법) </긴주의응답>",
    ])


def build_frontier_jobs(rules, spec, per_rule, neg_per_domain, seed):
    rng = random.Random(seed)
    jobs = []
    for rule in rules["rules"]:
        rs = spec["rules"][rule["id"]]
        dflt = spec["domain_defaults"][rule["domain"]]
        for k in range(per_rule):
            style = ("explicit", "inquiry")[k % 2]
            st = scenario_settings(rng, rule, rs, dflt, k)
            has_cond = bool(rule.get("condition")) and st.get("mode") != "본인부담금형"
            jobs.append(dict(kind="frontier", rule=rule["id"], domain=rule["domain"], style=style,
                             settings=st, has_cond=has_cond,
                             prompt=frontier_prompt(rules, rule, rs, st, style, has_cond)))
    for domain, topics in spec["negatives"].items():
        dflt = spec["domain_defaults"][domain]
        for k in range(neg_per_domain):
            st = {}
            if "contexts" in dflt:
                st["서비스"] = rng.choice(dflt["contexts"])
            if dflt.get("artifacts"):
                st["산출물"] = rng.choice(dflt["artifacts"])
            topic = topics[k % len(topics)]
            jobs.append(dict(kind="frontier_neg", rule=None, domain=domain, style="neg", settings=st,
                             topic=topic, prompt=frontier_neg_prompt(rules, domain, topic, st)))
    for i, j in enumerate(jobs):
        j["job_id"] = f"f{seed}-{i:05d}"
    return jobs


def build_jobs(rules, spec, per_rule, neg_per_domain, seed):
    rng = random.Random(seed)
    jobs = []
    for rule in rules["rules"]:
        rs = spec["rules"][rule["id"]]
        dflt = spec["domain_defaults"][rule["domain"]]
        for k in range(per_rule):
            style = STYLES[k % 3]
            st = scenario_settings(rng, rule, rs, dflt, k)
            has_cond = bool(rule.get("condition")) and st.get("mode") != "본인부담금형"
            jobs.append(dict(kind="rule", rule=rule["id"], domain=rule["domain"], style=style,
                             settings=st, has_cond=has_cond,
                             prompt=rule_prompt(rules, rule, rs, st, style, has_cond)))
    for domain, topics in spec["negatives"].items():
        dflt = spec["domain_defaults"][domain]
        for k in range(neg_per_domain):
            topic = topics[k % len(topics)]
            st = {}
            if "contexts" in dflt:
                st["서비스"] = rng.choice(dflt["contexts"])
            if dflt.get("artifacts"):
                st["산출물"] = rng.choice(dflt["artifacts"])
            style = "neg_inquiry" if k % 3 == 2 else "neg"
            jobs.append(dict(kind="negative", rule=None, domain=domain, style=style, settings=st,
                             topic=topic, prompt=negative_prompt(rules, domain, topic, st, style)))
    for i, j in enumerate(jobs):
        j["job_id"] = f"s{seed}-{i:05d}"
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=-1)
    ap.add_argument("--per-rule", type=int, default=90)
    ap.add_argument("--neg-per-domain", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--part", default="0/1", help="i/n: this process takes every n-th job starting at i")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="outputs/train_synth")
    ap.add_argument("--max-tokens", type=int, default=6144)
    ap.add_argument("--max-model-len", type=int, default=12288)
    ap.add_argument("--max-num-seqs", type=int, default=0)
    ap.add_argument("--mode", default="pairs", choices=["pairs", "frontier"])
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()

    rules = load_rules()
    spec = json.load(open(SPEC_PATH, encoding="utf-8"))
    jobs = (build_frontier_jobs if a.mode == "frontier" else build_jobs)(
        rules, spec, a.per_rule, a.neg_per_domain, a.seed)
    i, n = map(int, a.part.split("/"))
    jobs = jobs[i::n]
    if a.limit:
        jobs = jobs[:a.limit]
    print(f"[jobs] {len(jobs)} (part {a.part})", flush=True)
    if a.dry:
        from collections import Counter
        print(Counter((j["kind"], j["domain"]) for j in jobs))
        print(jobs[0]["prompt"])
        return

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(a.teacher, trust_remote_code=True)
    prompts = []
    for j in jobs:
        msgs = [{"role": "system", "content": TEACHER_SYS}, {"role": "user", "content": j["prompt"]}]
        try:
            p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        prompts.append(p)
    kw = dict(model=a.teacher, tensor_parallel_size=1, gpu_memory_utilization=0.90,
              max_model_len=a.max_model_len, trust_remote_code=True)
    if a.max_num_seqs:
        kw["max_num_seqs"] = a.max_num_seqs
    llm = LLM(**kw)
    sp = SamplingParams(n=1, max_tokens=a.max_tokens, seed=a.seed + 7,
                        temperature=a.temperature, top_p=a.top_p, top_k=a.top_k)
    outs = llm.generate(prompts, sp)
    tag = a.teacher.split("/")[-1]
    dst = os.path.join(ROOT, a.out, f"{tag}_s{a.seed}_p{i}of{n}.jsonl")
    rows = []
    for j, o in zip(jobs, outs):
        c = o.outputs[0]
        text = clean(c.text)
        rows.append({**{k: v for k, v in j.items() if k != "prompt"}, "teacher": a.teacher,
                     "finish": c.finish_reason, "tags": parse_tags(text), "raw": text})
    write_jsonl(dst, rows)
    ok = sum(1 for r in rows if r["finish"] == "stop")
    print(f"[done] {dst} rows={len(rows)} finished={ok}", flush=True)


if __name__ == "__main__":
    main()
