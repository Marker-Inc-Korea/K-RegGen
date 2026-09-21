# -*- coding: utf-8 -*-
"""lawsync — 택소노미의 조문 근거를 국가법령정보 OPEN API 원문과 동기화한다.

  python3 eval/lawsync.py --fill     # quote를 API 원문으로 채우고 effective_date 기입
  python3 eval/lawsync.py --check    # 대조만 (파일 수정 없음). CI용

인증: 환경변수 LAW_OC. 없으면 공개 샘플 계정 'test'를 쓴다(저트래픽 확인용).
운영 트래픽은 open.law.go.kr에서 발급받은 OC 값을 LAW_OC로 넘긴다.

legal_basis 필드 규약
  article : "제56조제2항제13호" 형식. 항·호까지 지정하면 그 단위 원문만 가져온다.
  quote   : **사람이 쓰지 않는다.** --fill이 API 원문을 그대로 채운다. --check가 대조하며
            불일치 시 종료코드 1(fail-closed). 조문이 개정되면 여기서 걸린다.
  gloss   : 작성자 요약·해석. 대조 대상이 아니다. 설명은 반드시 이쪽에 쓴다.

precedent 필드 규약
  target/case_id : 'prec'(판례) 또는 'detc'(헌재결정례) + 일련번호. 이 둘이 있어야 검증된다.
  case_no/decided: API가 돌려준 사건번호·선고(종국)일자와 대조한다. 뉴스 기사로 적은 날짜가
                   틀리는 일이 실제로 있었다(2020헌가10을 2024-04-25로 적었으나 실제 2024-03-28).
  holding        : --fill이 판시사항 첫 문단을 채운다.
"""
import json, glob, os, re, sys, unicodedata, urllib.request

API = "http://www.law.go.kr/DRF/lawService.do?OC={oc}&target=law&type=JSON&ID={lid}"
PREC_API = "http://www.law.go.kr/DRF/lawService.do?OC={oc}&target={tgt}&type=JSON&ID={pid}"
CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset", ".lawcache")


def norm(s):
    """비교용 정규화: NFKC, 한자 병기 제거, 가운뎃점·인용부호 통일, 공백 제거."""
    s = unicodedata.normalize("NFKC", s or "")
    s = re.sub(r"\([一-鿿][^)]*\)", "", s)   # 기사(記事) -> 기사
    s = re.sub(r"[ㆍ·⋅‧•]", "·", s)
    s = re.sub(r"[「」『』\"'’‘]", "", s)
    return re.sub(r"\s+", "", s)


def fetch_prec(tgt, pid, oc):
    os.makedirs(CACHE, exist_ok=True)
    p = os.path.join(CACHE, f"{tgt}_{pid}.json")
    if not os.path.exists(p):
        with urllib.request.urlopen(PREC_API.format(oc=oc, tgt=tgt, pid=pid), timeout=60) as r:
            open(p, "wb").write(r.read())
    return list(json.load(open(p, encoding="utf-8")).values())[0]


def fetch(lid, oc):
    os.makedirs(CACHE, exist_ok=True)
    p = os.path.join(CACHE, f"{lid}.json")
    if not os.path.exists(p):
        url = API.format(oc=oc, lid=lid)
        with urllib.request.urlopen(url, timeout=60) as r:
            open(p, "wb").write(r.read())
    return json.load(open(p, encoding="utf-8"))


def index(doc):
    """(조문키, 항번호|None, 호번호|None) -> 텍스트. 항·호 단위까지 개별 인덱싱한다."""
    out, eff = {}, {}
    HANG = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮"
    for j in doc["법령"]["조문"]["조문단위"]:
        if j.get("조문여부") != "조문":
            continue
        no, ga = j.get("조문번호"), (j.get("조문가지번호") or "")
        key = f"{no}의{ga}" if ga not in ("", "0") else no
        eff[key] = j.get("조문시행일자")
        whole = j.get("조문내용") or ""
        hs = j.get("항", [])
        hs = hs if isinstance(hs, list) else ([hs] if hs else [])
        for hi, h in enumerate(hs, 1):
            htxt = h.get("항내용") or ""
            m = re.match(r"\s*([%s])" % HANG, htxt)
            hno = str(HANG.index(m.group(1)) + 1) if m else str(hi)
            whole += " " + htxt
            out[(key, hno, None)] = htxt
            ho = h.get("호", [])
            for x in (ho if isinstance(ho, list) else ([ho] if ho else [])):
                xt = x.get("호내용") or ""
                whole += " " + xt
                m2 = re.match(r"\s*(\d+)\.", xt)
                if m2:
                    out[(key, hno, m2.group(1))] = xt
        out[(key, None, None)] = whole
    return out, eff


def parse_article(article):
    """'제56조제2항제13호' -> ('56', '2', '13'). 없는 단위는 None."""
    m = re.match(r"제(\d+)조(?:의(\d+))?", article)
    if not m:
        return None
    key = f"{m.group(1)}의{m.group(2)}" if m.group(2) else m.group(1)
    h = re.search(r"제(\d+)항", article)
    o = re.search(r"제(\d+)호", article)
    return key, (h.group(1) if h else None), (o.group(1) if o else None)


def main():
    fill = "--fill" in sys.argv
    check_only = not fill
    oc = os.environ.get("LAW_OC", "test")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    files = sorted(glob.glob(os.path.join(root, "dataset", "taxonomy", "*.json")))

    idx, problems, ok, filled, stale, prec_ok = {}, [], 0, 0, [], 0
    for f in files:
        d = json.load(open(f, encoding="utf-8"))
        for name, meta in d.get("laws", {}).items():
            lid = meta.get("law_id")
            if not lid or name in idx:
                continue
            try:
                doc = fetch(lid, oc)
                got = doc["법령"]["기본정보"]["법령명_한글"]
                if norm(got) != norm(name):
                    problems.append(f"[법령ID 불일치] {name}(law_id={lid}) -> 실제 '{got}'")
                idx[name] = index(doc)
            except Exception as e:
                problems.append(f"[조회실패] {name}({lid}): {e}")

        entries = list(d["violations"]) + list(d["controls"])
        for mp in d.get("minimal_pairs", []):
            for sd in ("a", "b"):
                entries.append({**mp[sd], "code": f"{mp['code']}/{sd}"})
        for e in entries:
            for lb in e["legal_basis"]:
                law = lb["law"]
                if law not in idx:
                    continue
                table, effs = idx[law]
                pa = parse_article(lb["article"])
                if not pa:
                    problems.append(f"[조문표기 파싱불가] {e['code']} {lb['article']}")
                    continue
                key, hang, ho = pa
                if (key, hang, ho) not in table and hang is None and ho is not None:
                    # '제109조제1호'처럼 항 표기 없이 호만 지정한 경우: 해당 조의 어느 항이든 그 호를 찾는다
                    cand = [k for k in table if k[0] == key and k[2] == ho]
                    if len(cand) == 1:
                        hang = cand[0][1]
                    elif len(cand) > 1:
                        problems.append(f"[호 중복] {e['code']} {law} {lb['article']} — 항을 명시할 것")
                        continue
                if (key, hang, ho) not in table:
                    problems.append(f"[조문없음] {e['code']} {law} {lb['article']}")
                    continue
                official = re.sub(r"\s+", " ", table[(key, hang, ho)]).strip()
                if fill:
                    lb["quote"] = official
                    filled += 1
                else:
                    q = lb.get("quote")
                    if q is None:
                        problems.append(f"[quote 없음] {e['code']} {law} {lb['article']} (--fill 필요)")
                    elif norm(q) != norm(official):
                        problems.append(f"[인용불일치] {e['code']} {law} {lb['article']}\n"
                                        f"      기재: {q[:70]}\n"
                                        f"      원문: {official[:70]}")
                    else:
                        ok += 1
                prev = lb.get("effective_date")
                neweff = effs.get(key)
                if prev and prev != neweff:
                    stale.append(f"[시행일변경] {e['code']} {law} {lb['article']}: {prev} -> {neweff}")
                lb["effective_date"] = neweff
                lb.setdefault("law_id", d["laws"][law]["law_id"])

            for pr in e.get("precedent", []):
                tgt, pid = pr.get("target"), pr.get("case_id")
                if not (tgt and pid):
                    problems.append(f"[판례 미고정] {e['code']} {pr.get('case_no')} — target/case_id 없음")
                    continue
                try:
                    doc = fetch_prec(tgt, pid, oc)
                except Exception as ex:
                    problems.append(f"[판례 조회실패] {e['code']} {pr.get('case_no')}: {ex}")
                    continue
                got_no = (doc.get("사건번호") or "").strip()
                got_dt = str(doc.get("선고일자") or doc.get("종국일자") or "")
                want_dt = (pr.get("decided") or "").replace("-", "")
                if norm(got_no) != norm(pr.get("case_no", "")):
                    problems.append(f"[판례 사건번호 불일치] {e['code']} 기재={pr.get('case_no')} 실제={got_no}")
                elif want_dt and got_dt and want_dt != got_dt:
                    problems.append(f"[판례 일자 불일치] {e['code']} {got_no} 기재={pr.get('decided')} "
                                    f"실제={got_dt[:4]}-{got_dt[4:6]}-{got_dt[6:]}")
                else:
                    prec_ok += 1
                if fill:
                    h = re.sub(r"<[^>]+>", "", str(doc.get("판시사항") or ""))
                    h = re.sub(r"\s+", " ", h).strip()
                    if h:
                        pr["holding"] = h[:400]
                    pr["case_name"] = (doc.get("사건명") or "").strip()
        if fill:
            json.dump(d, open(f, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print(f"OC={oc} | " + (f"quote 채움 {filled}건" if fill else f"축자 인용 대조 일치 {ok}건")
          + f" | 판례 대조 {prec_ok}건")
    for s_ in stale:
        print(" ", s_)
    if problems:
        print(f"\n[FAIL] {len(problems)}건")
        for p in problems:
            print(" ", p)
        sys.exit(1)
    print("이상 없음")


if __name__ == "__main__":
    main()
