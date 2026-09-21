# -*- coding: utf-8 -*-
"""gold 전수 감사. 조문 근거·채점 기준·verdict의 내적 정합성을 기계적으로 검사한다.

lawsync가 '인용이 원문과 같은가'를 보는 반면, 이 감사는
'gold가 주장하는 바가 인용한 조문에서 나오는가'와 '채점 기준이 그 주장과 맞는가'를 본다.
"""
import json, glob, os, re, sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = re.compile(r"제\s*(\d+)\s*조(?:의\s*(\d+))?(?:\s*제\s*(\d+)\s*항)?(?:\s*제\s*(\d+)\s*호)?")
LAWNAME = re.compile(r"(의료법 시행령|의료법|개인정보\s*보호법|의료기사[^,\s]*법?|약사법|변호사법|세무사법|공인중개사법)")


def arts_in(text):
    out = set()
    for m in ART.finditer(text or ""):
        no, ga, hang, ho = m.groups()
        out.add((f"{no}의{ga}" if ga else no, hang, ho))
    return out


def basis_arts(lbs):
    """legal_basis가 실제로 덮는 (조, 항, 호) 집합. 상위 단위는 하위를 덮는다."""
    out = set()
    for lb in lbs:
        for a in arts_in(lb["article"]):
            out.add(a)
    return out


def covers(basis, claim):
    """basis의 어떤 항목이 claim을 덮는가. 상위(항/호 미지정)가 하위를 덮는다."""
    k, h, o = claim
    for bk, bh, bo in basis:
        if bk != k:
            continue
        if bh is not None and h is not None and bh != h:
            continue
        if bo is not None and o is not None and bo != o:
            continue
        return True
    return False


def entries(d):
    out = []
    for kind, es in (("violation", d["violations"]), ("control", d["controls"])):
        for e in es:
            out.append((kind, e["code"], e))
    for mp in d.get("minimal_pairs", []):
        for sd in ("a", "b"):
            e = dict(mp[sd])
            e["code"] = f"{mp['code']}/{sd}"
            k = "control" if e["verdict"] == "allowed" else "violation"
            out.append((k, e["code"], e))
    return out


def main():
    issues = defaultdict(list)
    for f in sorted(glob.glob(os.path.join(ROOT, "dataset", "taxonomy", "*.json"))):
        d = json.load(open(f, encoding="utf-8"))
        for kind, code, e in entries(d):
            lbs = e.get("legal_basis", [])
            basis = basis_arts(lbs)
            mm = e.get("must_mention") or []

            # [1] 위반인데 채점 기준이 없으면 구조적으로 통과 불가
            if kind == "violation" and not mm:
                issues["A. 위반인데 must_mention 없음(구조적 통과 불가)"].append(code)

            # [2] gold_points가 인용한 조문이 legal_basis에 없는 경우
            for gp in e.get("gold_points", []):
                for claim in arts_in(gp):
                    if not covers(basis, claim):
                        k, h, o = claim
                        lbl = f"제{k}조" + (f"제{h}항" if h else "") + (f"제{o}호" if o else "")
                        issues["B. gold가 근거 밖 조문을 인용"].append(f"{code}: {lbl}")

            # [3] must_mention 그룹0의 조문 앵커가 legal_basis에 없는 경우
            if mm:
                g0 = mm[0]
                nums = re.findall(r"(\d+)\\s\*조", g0) + re.findall(r"(\d+)\s*조", g0)
                for n in set(nums):
                    if not any(bk == n or bk.startswith(n + "의") for bk, _, _ in basis):
                        issues["C. 채점 기준이 근거에 없는 조문을 요구"].append(f"{code}: 제{n}조")
                hos = re.findall(r"(\d+)\s*\\s\*호|(\d+)\s*호", g0)
                hos = [a or b for a, b in hos]
                for n in set(hos):
                    if not any(bo == n for _, _, bo in basis):
                        issues["C. 채점 기준이 근거에 없는 조문을 요구"].append(f"{code}: 제{n}호")

            # [4] 대조군인데 인용을 요구
            if kind == "control" and mm:
                issues["D. 대조군인데 조문 인용을 요구"].append(code)

            # [5] 시행령이 근거인데 채점은 법률 조문만 요구
            has_decree = any("시행령" in lb["law"] for lb in lbs)
            if has_decree and mm and "시행령" not in mm[0]:
                issues["E. 시행령이 근거인데 채점이 법률 조문만 인정"].append(code)

            # [7] ★ gold 자체가 채점 기준을 통과하는가 (자기 정합성)
            if kind == "violation" and mm:
                gtext = " ".join(e.get("gold_points", []))
                miss = [g for g in mm if not re.search(g, gtext)]
                if miss:
                    issues["G. gold가 자기 채점 기준을 통과하지 못함"].append(
                        f"{code}: 미충족 그룹 {miss}")

            # [8] 근거인데 gold·채점 어디서도 언급되지 않는 조문
            gall = " ".join(e.get("gold_points", [])) + " " + " ".join(mm)
            for lb in (lbs if kind == "violation" else []):
                if lb.get("gloss"):
                    continue
                a = lb["article"]
                m0 = ART.match(a)
                if not m0:
                    continue
                no = m0.group(1)
                if not re.search(rf"제\s*{no}\s*조|{no}\s*\\s\*조", gall):
                    issues["H. 근거로 달았지만 gold가 쓰지 않는 조문"].append(f"{code}: {a}")

            # [9] verdict와 gold 논조
            gt = " ".join(e.get("gold_points", []))
            if e["verdict"] == "allowed" and not re.search(r"과잉거부|차단하면|해당하지\s*않|아니다|가능|오답|올바른\s*대응", gt):
                issues["I. 적법 항목인데 gold가 '차단하면 과잉거부'를 명시하지 않음"].append(code)
            if e["verdict"] == "prohibited" and not re.search(r"금지|위반|아니?\s*된다|처벌", gt):
                issues["J. 금지 항목인데 gold가 금지를 명시하지 않음"].append(code)

        # [6] 최소쌍 b면은 적법이어야 경계를 잰다
        for mp in d.get("minimal_pairs", []):
            if mp["b"]["verdict"] != "allowed":
                issues["F. 최소쌍 b면이 적법이 아님(경계 측정 불가)"].append(
                    f"{mp['code']} (b verdict={mp['b']['verdict']})")

    total = sum(len(v) for v in issues.values())
    print(f"# gold 전수 감사 — 지적 {total}건\n")
    for k in sorted(issues):
        v = issues[k]
        print(f"## {k} — {len(v)}건")
        for x in v:
            print(f"  - {x}")
        print()
    return issues


if __name__ == "__main__":
    main()
