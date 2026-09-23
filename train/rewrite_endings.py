# -*- coding: utf-8 -*-
"""Rebalance the closing phrase of inquiry-style requests.

Teachers copied the example closings from the synthesis prompt, so a third of inquiry
requests ended with the same words. Only the formulaic closing phrase is replaced; the
plan described before it is left exactly as written, so stored responses and labels stay
valid. Replacements keep the yes-means-go-ahead direction and the original register, and
each pick is the variant used least so far. Families above --cap are reduced to the cap.

(An earlier version let a model rewrite the whole last sentence; it dropped plan details
that the label depends on, e.g. 'the camera shows the inside of the changing room'.)

  python train/rewrite_endings.py --cand <candidate files> --out outputs/request_map.json
"""
import argparse
import glob
import json
import os
import random
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, read_jsonl  # noqa: E402

# 문장 끝의 상투 구절(앞의 계획 서술은 건드리지 않는다)
TAILS = [
    ("legal_ok", re.compile(r"법적으로\s*(?:도\s*)?(?:아무\s*)?문제\s*(?:가|는)?\s*(?:없|안\s*되|되지\s*않)[^.?!,，]{0,12}[?？.!]?\s*$")),
    ("okay", re.compile(r"괜찮(?:을까요|겠죠|은가요|나요|을지요|죠|을까|겠지|아|은지|습니까)[^.?!,，]{0,12}[?？.!]?\s*$")),
    ("no_problem", re.compile(r"(?:아무\s*)?문제\s*(?:가|는)?\s*(?:없|안\s*되|되지\s*않)[^.?!,，]{0,12}[?？.!]?\s*$")),
]
# 앞말이 '-도/-면'으로 끝날 때 이어 붙이는 말(답이 '네'면 진행해도 된다는 뜻)
CONT = {"polite": ["될까요?", "되나요?", "되겠죠?", "되는 거죠?", "무방할까요?", "되겠습니까?", "될지 여쭤봅니다.",
                   "되는 건가요?", "상관없을까요?", "허용되나요?", "되는지 확인 부탁드립니다.", "되는 거 맞죠?",
                   "될지 궁금합니다.", "되는지 알려주실 수 있나요?", "되는 걸까요?"],
        "casual": ["될까?", "되지?", "되겠지?", "되는 거지?", "상관없지?", "되는 거 맞지?"]}
# 앞말이 연결어미로 끝나지 않을 때 쓰는 완결형
FULL = {"polite": ["이대로 진행해도 될까요?", "그대로 해도 되나요?", "이 방향으로 가도 될까요?", "이대로 올려도 되겠죠?",
                   "진행해도 무방할까요?", "이렇게 해도 되는 거죠?", "이대로 시작해도 될지 여쭤봅니다.",
                   "이 계획대로 해도 되는지 확인 부탁드립니다."],
        "casual": ["이대로 해도 될까?", "그냥 진행해도 되지?", "이렇게 해도 되는 거지?", "이대로 올려도 되겠지?"]}
# 앞말이 '이 방식이'처럼 주어로 끝날 때
SUBJ = {"polite": ["허용될까요?", "가능한가요?", "용인되는 범위인가요?", "통할까요?", "받아들여질까요?"],
        "casual": ["허용되지?", "통하겠지?", "가능하지?"]}
# 앞말이 '…하는 거/것'처럼 명사로 끝날 때
NOUN = {"polite": ["가능할까요?", "허용되나요?", "상관없을까요?", "무방한가요?", "해도 되는지 여쭤봅니다."],
        "casual": ["가능하지?", "상관없지?", "해도 되지?"]}


def last_split(q):
    q = q.strip()
    m = list(re.finditer(r"(?<=[.?!])\s+", q))
    return (q[:m[-1].end()], q[m[-1].end():]) if m else ("", q)


def find_tail(last):
    for name, pat in TAILS:
        m = pat.search(last)
        if m:
            return name, m.start()
    return "other", None


def ending_key(s):
    return re.sub(r"\s+", "", s)[-8:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", nargs="+", required=True)
    ap.add_argument("--out", default="outputs/request_map.json")
    ap.add_argument("--cap", type=float, default=0.05, help="max share per closing family after rewriting")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    req = {}
    for pat in a.cand:
        for f in sorted(glob.glob(os.path.join(ROOT, pat))):
            for r in read_jsonl(f):
                if r.get("style") in ("inquiry", "neg_inquiry"):
                    req.setdefault(r["scenario"], r["request"].strip())
    ids = sorted(req)
    info = {s: find_tail(last_split(req[s])[1]) for s in ids}
    rng = random.Random(a.seed)
    todo = []
    for name, _ in TAILS:
        members = [s for s in ids if info[s][0] == name]
        keep = int(a.cap * len(ids))
        if len(members) > keep:
            rng.shuffle(members)
            todo += members[keep:]
    before = Counter(v[0] for v in info.values())
    used = Counter(ending_key(req[s]) for s in ids if s not in set(todo))
    out = {}
    for s in sorted(todo):
        head, last = last_split(req[s])
        _, pos = info[s]
        stem, tail = last[:pos], last[pos:]
        reg = "polite" if re.search(r"요|죠|니다|니까|세요", tail) else "casual"
        core = stem.rstrip(" ,，")
        if re.search(r"(도|면)$", core):
            opts = CONT[reg]
        elif re.search(r"(?:[가-힣][이가]|게)$", core):
            opts = SUBJ[reg]
        elif re.search(r"(거|것)$", core):
            opts = NOUN[reg]
        else:
            opts = FULL[reg]
        pick = min(opts, key=lambda o: (used[ending_key(o)], rng.random()))
        new_last = (stem.rstrip() + " " + pick).strip()
        used[ending_key(new_last)] += 1
        out[s] = (head + new_last).strip()
    os.makedirs(os.path.dirname(os.path.join(ROOT, a.out)), exist_ok=True)
    json.dump(out, open(os.path.join(ROOT, a.out), "w"), ensure_ascii=False, indent=0)
    after = Counter(find_tail(last_split(out.get(s, req[s]))[1])[0] for s in ids)
    top = Counter(ending_key(out.get(s, req[s])) for s in ids).most_common(5)
    print(f"[rewrite] inquiry {len(ids)} | before {dict(before)} | rewritten {len(out)} | after {dict(after)}")
    print("  top closings:", [(k, round(v / len(ids), 3)) for k, v in top])


if __name__ == "__main__":
    main()
