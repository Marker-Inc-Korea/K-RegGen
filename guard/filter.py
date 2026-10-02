# -*- coding: utf-8 -*-
"""K-RegGen Guard as a filter: put it behind any model and check each response.

    from guard.filter import RegGuard
    guard = RegGuard(lora="checkpoints/guard-4b")              # Qwen3-4B-Instruct-2507 + LoRA, vLLM
    result = guard.check(request, response, domain="medical_ad")
    shown = guard.annotate(response, result)                   # response + rule-table notice if flagged

`result` holds the verdict (위반 / 확인필요 / 적법), the rules with their articles, the quoted
span, the notice text, the allowed alternatives looked up from the rule table, and a hazard
label for safety stacks that use the Llama Guard categories (S6/KR-reggen when flagged).

Pass the domain (업종 힌트) when you know it. Without it a keyword router picks the domain from the
request (96~98% correct on the evaluation sets, under 0.5% wrong); on a tie the guard reads the
tied domains' tables and keeps the most severe verdict. Reading every table blindly is not
offered: the guard was trained with the right table only, and on the audited test it doubled
the false-alarm rate (8% -> 18%).

    python3 guard/filter.py --lora checkpoints/guard-4b --input items.jsonl --out results.jsonl
    python3 guard/filter.py --lora checkpoints/guard-4b --domain medical_ad --request "..." --response "..."

Input rows need `request` and `response`; `domain` and `id` are optional.
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "train"))
from common import clean, guard_messages, load_rules, notice, parse_guard, read_jsonl, write_jsonl  # noqa: E402

SEVERITY = {"위반": 2, "확인필요": 1, "적법": 0}

# 요청 문장으로 업종을 고른다(규칙표의 업종과 1:1). 동점이면 동점인 업종을 모두 본다.
DOMAIN_KEYWORDS = {
    "privacy_code": r"코드|스크립트|쿼리|SQL|sql|DB|데이터베이스|API|함수|파이썬|[Pp]ython|자바|Java|코틀린|Kotlin|Go 언어|Golang|"
                    r"PHP|Node|노드|자바스크립트|JavaScript|TypeScript|타입스크립트|스키마|테이블|백엔드|프론트엔드|서버|크롤|로직|"
                    r"구현|개발|CSV|엑셀 추출|주민등록번호|주민번호|개인정보|CCTV|cctv|폐쇄회로|가명|암호화|리전|백업|탈퇴 회원|"
                    r"수신 동의|마케팅 동의|회원가입|가입 폼|동의 체크",
    "medical_ad": r"병원|의원|클리닉|치과|한의원|피부과|성형외과|안과|정형외과|산부인과|비뇨|내과|의료기관|진료|원장|전문의|"
                  r"환자|검진|내원|비급여|의료광고|임플란트|라식|라섹|교정|수술|시술",
    "restricted_sales": r"안경|콘택트렌즈|렌즈|도수|의약품|약국|약사|일반의약품|상비약|인공눈물|영양제 판매|라이브커머스|"
                        r"스마트스토어|쿠팡|온라인 판매|택배 배송|배달",
    "unlicensed_practice": r"세무|세금|종합소득세|부가세|부가가치세|기장|신고 대행|법률|소송|고소장|변호사|합의금|이혼|회생|파산|"
                           r"공인중개사|중개|매물|분양|부동산|에스테틱|피부관리실|네일|미용실|헤어|문신|타투|반영구|필러|보톡스|"
                           r"수액|주사|PRP",
}


def route_domain(request, domains):
    """Candidate domains for a request: one when the keywords decide, the tied ones otherwise."""
    score = {d: len(re.findall(p, request or "")) for d, p in DOMAIN_KEYWORDS.items() if d in domains}
    best = max(score.values(), default=0)
    return [d for d in domains if score.get(d) == best] if best else list(domains)


class RegGuard:
    def __init__(self, lora, model="Qwen/Qwen3-4B-Instruct-2507", backend="vllm", rules_path=None,
                 hazard="S6/KR-reggen", max_tokens=384, max_model_len=16384, gpu_memory_utilization=0.90):
        self.rules = load_rules(rules_path) if rules_path else load_rules()
        self.domains = list(self.rules["domains"])
        self.hazard = hazard
        self.backend = backend
        self.max_tokens = max_tokens
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model)
        if backend == "vllm":
            from vllm import LLM, SamplingParams
            from vllm.lora.request import LoRARequest
            self.llm = LLM(model=model, enable_lora=True, max_lora_rank=64, max_model_len=max_model_len,
                           gpu_memory_utilization=gpu_memory_utilization, tensor_parallel_size=1)
            self.sp = SamplingParams(temperature=0.0, max_tokens=max_tokens)
            self.lora = LoRARequest("guard", 1, lora)
        elif backend == "hf":
            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM
            base = AutoModelForCausalLM.from_pretrained(model, dtype=torch.bfloat16, device_map="auto")
            self.model = PeftModel.from_pretrained(base, lora).eval()
        else:
            raise ValueError(f"unknown backend {backend}")

    # ------------------------------------------------------------------ prompting
    def _prompt(self, domain, request, response):
        msgs = guard_messages(self.rules, domain, request, response or "(빈 응답)")
        kw = dict(tokenize=False, add_generation_prompt=True, enable_thinking=False)
        try:
            return self.tok.apply_chat_template(msgs, **kw)
        except TypeError:
            kw.pop("enable_thinking")
            return self.tok.apply_chat_template(msgs, **kw)

    def _generate(self, prompts):
        if self.backend == "vllm":
            outs = self.llm.generate(prompts, self.sp, lora_request=self.lora, use_tqdm=len(prompts) > 50)
            return [o.outputs[0].text for o in outs]
        import torch
        texts = []
        for p in prompts:
            ids = self.tok(p, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                out = self.model.generate(**ids, max_new_tokens=self.max_tokens, do_sample=False)
            texts.append(self.tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True))
        return texts

    # ------------------------------------------------------------------ results
    def _result(self, domain, text):
        label, ids, spans = parse_guard(clean(re.sub(r"<\|[a-z_]+\|>", "", text)))
        ids = [i for i in ids if i in self.rules["by_id"] and self.rules["by_id"][i]["domain"] == domain]
        flagged = label in ("위반", "확인필요")
        # 규칙 ID 없이 플래그한 출력도 판정은 그대로 두고(보수적으로) 오류만 표시한다
        error = "unparsed" if not label else "no_rule" if flagged and not ids else None
        rows = [self.rules["by_id"][i] for i in ids] if flagged else []
        return {
            "label": label,
            "domain": domain,
            "rules": [{"id": r["id"], "title": r["title"], "articles": r["articles"],
                       "reason": r["condition"] if label == "확인필요" and r.get("condition") else r["prohibited"]}
                      for r in rows],
            "spans": spans if flagged else [],
            "notice": notice(self.rules, label, ids, spans) if flagged else "",
            "alternatives": [r["allowed"] for r in rows],
            "hazard": self.hazard if flagged else None,
            "llamaguard": f"unsafe\n{self.hazard.split('/')[0]}" if flagged else "safe",
            "error": error,
            "raw": text,
        }

    def check_batch(self, items):
        """items: [{request, response, domain?}] -> list of results (same order)."""
        jobs = []
        for n, it in enumerate(items):
            doms = [it["domain"]] if it.get("domain") else route_domain(it["request"], self.domains)
            for d in doms:
                if d not in self.rules["domains"]:
                    raise ValueError(f"unknown domain {d}; one of {self.domains}")
                jobs.append((n, d, self._prompt(d, it["request"], it["response"])))
        texts = self._generate([p for _, _, p in jobs])
        per_item = [[] for _ in items]
        for (n, d, _), t in zip(jobs, texts):
            per_item[n].append(self._result(d, t))
        out = []
        for n, rs in enumerate(per_item):
            ok = [r for r in rs if r["label"]]
            best = max(ok, key=lambda r: SEVERITY[r["label"]]) if ok else rs[0]
            if len(rs) > 1:
                best = {**best, "checked_domains": [r["domain"] for r in rs]}
            out.append(best)
        return out

    def check(self, request, response, domain=None):
        return self.check_batch([{"request": request, "response": response, "domain": domain}])[0]

    @staticmethod
    def annotate(response, result):
        """The response as shown to the user: unchanged if lawful, notice appended if flagged."""
        return (response or "") + (result.get("notice") or "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--backend", default="vllm", choices=["vllm", "hf"])
    ap.add_argument("--input", default="", help="jsonl with request, response, optional domain/id")
    ap.add_argument("--out", default="")
    ap.add_argument("--request", default="")
    ap.add_argument("--response", default="")
    ap.add_argument("--domain", default="")
    ap.add_argument("--annotate", action="store_true", help="also write the response with the notice appended")
    ap.add_argument("--keep-raw", action="store_true")
    a = ap.parse_args()

    guard = RegGuard(lora=a.lora, model=a.model, backend=a.backend)
    items = read_jsonl(a.input) if a.input else [{"request": a.request, "response": a.response, "domain": a.domain or None}]
    results = guard.check_batch(items)
    rows = []
    for it, r in zip(items, results):
        if not a.keep_raw:
            r = {k: v for k, v in r.items() if k != "raw"}
        row = {"id": it.get("id"), **r}
        if a.annotate:
            row["annotated"] = guard.annotate(it["response"], r)
        rows.append(row)
    if a.out:
        write_jsonl(a.out, rows)
        print(f"[guard] {len(rows)} checked -> {a.out}", flush=True)
    else:
        print(json.dumps(rows[0] if len(rows) == 1 else rows, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
