# -*- coding: utf-8 -*-
"""택소노미의 validity 블록을 모아 유효성 검사 보고서를 만든다."""
import json, glob, os, collections

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORDER = ["precedent", "precedent+inference", "text+precedent+interpretation",
         "text+precedent", "text+interpretation", "text", "text+inference"]

rows, beyond, counts = [], [], collections.Counter()
for f in sorted(glob.glob(os.path.join(ROOT, "dataset", "taxonomy", "*.json"))):
    d = json.load(open(f, encoding="utf-8"))
    for kind, es in (("위반", d["violations"]), ("대조군", d["controls"])):
        for e in es:
            v = e.get("validity", {})
            b = v.get("basis", "?")
            counts[b] += 1
            auth = []
            for pr in e.get("precedent", []):
                auth.append(f"{pr['case_no']}({pr.get('target')}/{pr.get('case_id')})")
            for it in e.get("interpretation", []):
                auth.append(f"해석례 {it['case_id']}")
            rows.append((d["domain_ko"], kind, e["code"], b, ", ".join(auth) or "—", v.get("note", "")))
            for t in v.get("beyond_text", []):
                beyond.append((e["code"], t, v.get("note", "")))

out = [f"# K-RegGen 유효성 검사 보고서", "",
       "검사일 **2026-09-19** · 국가법령정보 OPEN API 대조", "",
       "> **이 검사는 법률 전문가 검수를 대체하지 않는다.** 확인한 것은 (1) 인용 조문이 실재하고 원문과 일치하는가, "
       "(2) 인용 판례·해석례가 실재하고 사건번호·일자가 맞는가, (3) gold의 주장이 조문 문언에서 도출되는가 세 가지다. "
       "**그 조문이 해당 사안에 적용되는지에 대한 법률 판단은 검증 범위 밖이다.**", "",
       "## 근거 유형 분포", "", "| 근거 유형 | 유형 수 |", "|---|--:|"]
for b in ORDER:
    if counts[b]:
        out.append(f"| `{b}` | {counts[b]} |")
for b, c in counts.items():
    if b not in ORDER:
        out.append(f"| `{b}` | {c} |")

out += ["", "## 주의가 필요한 항목 — 조문 문언을 넘어선 주장", ""]
if beyond:
    out.append("| 유형 | 문언을 넘어선 주장 |")
    out.append("|---|---|")
    for code, t, _ in beyond:
        out.append(f"| `{code}` | {t} |")
    out += ["", f"**{len(beyond)}건.** 이 주장들은 조문 문언만으로는 확정되지 않는다. "
            "전문가 검수 우선순위이며, 그때까지 해당 문항의 채점 결과는 보수적으로 해석해야 한다."]
else:
    out.append("없음.")

out += ["", "## 전체 유형별 판정", "", "| 도메인 | 구분 | 유형 | 근거 | 확인된 권위 |", "|---|---|---|---|---|"]
for dom, kind, code, b, auth, _ in rows:
    out.append(f"| {dom} | {kind} | `{code}` | `{b}` | {auth} |")

out += ["", "## 검사 중 발견해 수정한 오류", "",
        "| 대상 | 오류 | 수정 |", "|---|---|---|",
        "| 법령ID 5건 | 약사법=001760(실제 산업재해보상보험법), 변호사법=001729(실제 교육세법), 나머지 3건 미존재 | API 조회로 교체 |",
        "| 의료기사법 §12⑤ | 각 호 열거 방식 개정을 놓침 | 제5항 제1호(전자상거래)·제2호(구매·배송대행)로 분리 |",
        "| 개인정보보호법 §24조의2 | ①처리제한과 ②암호화를 뭉뚱그림 | 항 단위로 분리 |",
        "| 헌재 2020헌가10 | 종국일자를 2024-04-25로 기재(뉴스 기사) | 실제 **2024-03-28** |",
        "| 전합 판례 대응 | 2021도15611이 두 판례를 모두 변경한 것으로 기재 | **2021도15611→91도3219**, **2022도13370→2004도673** |",
        "| 의료법 §56②13호 | 비급여 할인 광고 근거 조문 누락 | 제13호로 확정(gold 3차 수정) |",
        "| RS-OPTICAL-ONLINE | 전면 금지로 서술 | 해석례 315379로 **외국 소비자 대상은 비금지** 반영 |",
        "| MA-INDUCEMENT-GIFT | 교통편의 제공을 전면 금지로 서술 | 해석례 326841로 **환자 전용 직영 순환버스는 비해당** 반영 |",
        "", "## 남은 공백", "",
        "- **보건복지부 고시 미확보.** 의료법 시행령 제23조 제3항이 인터넷 홈페이지 의료광고의 세부기준을 고시에 위임하나, 행정규칙 API 검색으로 해당 고시를 찾지 못했다. 전후사진 기준이 여기 있을 수 있다.",
        "- **판례 DB 수록 범위가 좁다.** 국가법령정보 판례 API는 주로 판례공보 수록분이라, 개인정보·의료광고 하급심 사례는 거의 조회되지 않았다.",
        "- **개인정보위 처분 의결서 미수집.** 학습셋 출처로 계획했으나 이번 검사에는 포함하지 않았다.",
        "- **`gloss`와 `gold_points`는 `lawsync --check`의 대조 대상이 아니다.** 틀린 해석이 남아 있어도 CI는 통과한다.", ""]
open(os.path.join(ROOT, "dataset", "validity_report.md"), "w", encoding="utf-8").write("\n".join(out))
print("\n".join(out[:34]))
