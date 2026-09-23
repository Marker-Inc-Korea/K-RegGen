# -*- coding: utf-8 -*-
"""Mechanical clean-up for guard examples (found in the full read-through of v0.2.2).

- restore spans whose whitespace was normalised so they are exact substrings of the response
- drop examples whose text carries generation debris: stray symbols, broken syllables,
  unrendered LaTeX, foreign-script fragments inside Korean sentences, leaked meta notes,
  prose lines or doubled comment markers inside code, or a violation that only exists
  inside a commented-out line

  python train/fixes.py --src outputs/k_reggen_train_v022s outputs/k_reggen_train_v022 \
                        --dst outputs/k_reggen_train_v023s outputs/k_reggen_train_v023
"""
import argparse
import json
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, guard_messages, guard_target, load_rules, read_jsonl, write_jsonl  # noqa: E402

HANGUL = re.compile(r"[가-힣]")
HAN = re.compile(r"[぀-ヿ㐀-鿿Ѐ-ӿ]")
STRAY = re.compile(r"[⟫⟪�]")
LATEX = {r"$\rightarrow$": "→", r"$\leftarrow$": "←", r"$\times$": "×", r"$\sim$": "~"}
TRIPLE = re.compile(r"([가-힣])\1\1")
TRIPLE_OK = {"하", "호", "히", "후", "헤", "쿵", "짝", "탕", "톡", "똑", "딩", "동", "빵", "팡", "쏙"}
META = re.compile(
    r"\(금지 요소\)|<산출물>|</?(?:요청|응답)>|\(고지·동의 언급 없음\)|해당 조건에 대한 언급(?:은|도)? 없|"
    r"믿:음|\d{4}년 \d+월까지의 정보|media_v|[은는] \*\*\[[^\]\n]{20,}(?:니다|요)\.?\]\*\*|▶️\[|"
    r"부풀려 기재함\)|\(기존 가격 부풀리기\)|가격을 (?:부풀려|높게 책정하여)|"
    r"(?:실제 보도 기사|기사)인 것처럼 (?:위장|구성|보이)|위장하여 게재|기자의 취재 없이|"
    r"언론사인지는 확인이 필요|\(이하 \d+, \d+안 동일\)")
COMMENT = re.compile(r"^\s*(?://|#|--\s|/\*|\*\s|<!--|\{/\*)")
DOUBLE_COMMENT = re.compile(r"^\s*(?://\s+//|#\s+#\s|--\s+--\s)")
MARKUP = {"html", "ejs", "xml", "vue", "jsx", "tsx"}
CODE_LANGS = {"python", "py", "javascript", "js", "jsx", "typescript", "ts", "tsx", "java", "kotlin", "kt", "go",
              "golang", "php", "sql", "html", "ejs", "ruby", "rb", "c", "cpp", "csharp", "cs", "swift", "rust",
              "bash", "sh", "dart", "scala", "vue", "xml", "yaml", "json"}
CODEISH = re.compile(r"[=(){};<>\[\]]")


def code_lines(text, with_lang=False):
    """Lines inside ``` fences tagged with a programming language."""
    inside, out = None, []
    for line in text.split("\n"):
        s = line.strip()
        if s.startswith("```"):
            inside = None if inside is not None else s[3:].strip().lower()
            continue
        if inside in CODE_LANGS:
            out.append((inside, line) if with_lang else line)
    return out


def odd_syllable(ch):
    """Outside the 2,350 KS X 1001 syllables, e.g. the broken '옔' in '카카옔'."""
    b = ch.encode("cp949")
    return not (len(b) == 2 and 0xB0 <= b[0] <= 0xC8 and b[1] >= 0xA1)


FOREIGN_RUN = re.compile(r"(?<!\()[\u3040-\u30ff\u3400-\u9fff\u0400-\u04ff]{2,}(?!\))")


def foreign_in_korean(text):
    """A short foreign-script run dropped into a Korean sentence (not hanja glossed in parentheses)."""
    for line in text.split("\n"):
        han, hg = len(HAN.findall(line)), len(HANGUL.findall(line))
        if hg >= 6 and han < hg / 3 and FOREIGN_RUN.search(line):
            return True
    return False


def problems(r, synthetic):
    """Request debris is checked everywhere (requests are generated); response debris only in
    synthetic responses, since collected responses are kept exactly as the models wrote them."""
    out = []
    for field in ("request", "response") if synthetic else ("request",):
        t = r[field]
        if STRAY.search(t):
            out.append(f"{field}:stray")
        if any(odd_syllable(c) for c in set(HANGUL.findall(t))):
            out.append(f"{field}:odd_syllable")
        m = TRIPLE.search(t)
        if m and m.group(1) not in TRIPLE_OK:
            out.append(f"{field}:triple")
        if foreign_in_korean(t):
            out.append(f"{field}:foreign_script")
    if not synthetic:
        return out
    resp = r["response"]
    if META.search(resp):
        out.append("response:meta")
    code = code_lines(resp)
    for lang, line in code_lines(resp, with_lang=True):
        s = line.strip()
        if DOUBLE_COMMENT.match(line):
            out.append("code:double_comment")
            break
        if (lang not in MARKUP and len(HANGUL.findall(s)) >= 6 and not COMMENT.match(line) and not re.search(r"[\"'`]", s)
                and not CODEISH.search(s) and re.search(r"(다|요)\.?$", s)):
            out.append("code:prose_line")
            break
    if r["label"] != "적법":
        for sp in r.get("spans") or []:
            first = sp.strip().split("\n")[0]
            if not re.match(r"[A-Za-z_$<{@]", first) or not CODEISH.search(first):
                continue
            hit = [l for l in code if first in l]
            if hit and all(COMMENT.match(l) for l in hit):
                out.append("span:commented_out")
                break
    return out


def restore_span(span, response):
    if span in response:
        return span
    pat = r"\s+".join(re.escape(p) for p in span.split())
    m = re.search(pat, response)
    return m.group(0) if m else None


def fix_example(r, synthetic):
    """Returns (example or None, reasons)."""
    if synthetic:
        for k, v in LATEX.items():
            if k in r["response"]:
                r = {**r, "response": r["response"].replace(k, v),
                     "spans": [s.replace(k, v) for s in r.get("spans") or []]}
    if r["label"] != "적법" and r.get("spans"):
        new = [restore_span(s, r["response"]) for s in r["spans"]]
        kept = [s for s in new if s]
        if not kept:
            return None, ["span:missing"]
        r = {**r, "spans": kept}
    p = problems(r, synthetic)
    return (None, p) if p else (r, [])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", nargs="+", required=True)
    ap.add_argument("--dst", nargs="+", required=True)
    ap.add_argument("--report", default="outputs/fixes_report.json")
    a = ap.parse_args()
    rules = load_rules()
    rec, stats, dropped = {}, Counter(), []
    for src, dst in zip(a.src, a.dst):
        src, dst = os.path.join(ROOT, src), os.path.join(ROOT, dst)
        os.makedirs(dst, exist_ok=True)
        for f in sorted(os.listdir(src)):
            if f.startswith("sft_") or not f.endswith(".jsonl"):
                continue
            keep = []
            for r in read_jsonl(os.path.join(src, f)):
                spans_before = r.get("spans")
                r2, why = fix_example(r, synthetic="source_model" not in r)
                if r2 is None:
                    stats[(f, "drop")] += 1
                    for w in why:
                        stats[("reason", w)] += 1
                    dropped.append({"ex_id": r["ex_id"], "file": f, "why": why})
                    continue
                if r2.get("spans") != spans_before:
                    stats[(f, "span_restored")] += 1
                rec[r2["ex_id"]] = r2
                keep.append(r2)
            write_jsonl(os.path.join(dst, f), keep)
            stats[(f, "kept")] += len(keep)
        if os.path.exists(os.path.join(src, "build_meta.json")):
            meta = json.load(open(os.path.join(src, "build_meta.json")))
            meta["fixes"] = "train/fixes.py"
            json.dump(meta, open(os.path.join(dst, "build_meta.json"), "w"), ensure_ascii=False, indent=1)
    for src, dst in zip(a.src, a.dst):
        src, dst = os.path.join(ROOT, src), os.path.join(ROOT, dst)
        for f in sorted(os.listdir(src)):
            if not f.startswith("sft_"):
                continue
            out = []
            for s in read_jsonl(os.path.join(src, f)):
                c = rec.get(s["ex_id"])
                if c is None:
                    continue
                out.append({**s, "messages": guard_messages(rules, c["domain"], c["request"], c["response"])
                            + [{"role": "assistant", "content": guard_target(c["label"], c["rules"], c["spans"])}]})
            write_jsonl(os.path.join(dst, f), out)
            stats[(f, "sft")] += len(out)
    json.dump(dropped, open(os.path.join(ROOT, a.report), "w"), ensure_ascii=False, indent=0)
    for k, v in sorted(stats.items(), key=lambda x: str(x[0])):
        print(k, v)


if __name__ == "__main__":
    main()
