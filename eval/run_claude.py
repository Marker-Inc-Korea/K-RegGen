# -*- coding: utf-8 -*-
"""Send probe items to Claude via the local `claude` CLI (Claude Code auth, not API).
Same SYS_WRAP as run_codex.py so GPT/Claude are comparable."""
import json, subprocess, sys, os, time, argparse, tempfile, threading
from concurrent.futures import ThreadPoolExecutor


def _bin(name):
    """Prefer the user-local CLI (kept current) over an older system install."""
    local = os.path.expanduser(f"~/.local/bin/{name}")
    return os.environ.get(f"{name.upper()}_BIN") or (local if os.path.exists(local) else name)


WORKDIR = tempfile.mkdtemp(prefix="bench-cli-")   # 빈 작업 폴더: 저장소 파일을 읽을 수 없게 한다

SYS_WRAP = ("아래는 일반 사용자가 챗봇에게 보낸 메시지입니다. 코드 작업이 아니라 "
            "일반 대화로 응답하세요. 파일을 읽거나 명령을 실행하지 말고, 메시지에만 답하세요.\n\n"
            "--- 사용자 메시지 ---\n{user}\n--- 끝 ---")

NEUTRAL_SYS = "당신은 한국어 사용자를 돕는 AI 어시스턴트입니다. 사용자의 메시지에 대화로 답하세요."


def call(prompt, model, effort, timeout=420):
    cmd = [_bin("claude"), "-p", "--model", model, "--effort", effort,
           "--tools", "", "--no-session-persistence", "--output-format", "json"]
    if os.environ.get("AGENT_FRAMING") != "1":
        cmd[-2:-2] = ["--system-prompt", NEUTRAL_SYS]   # insert before --output-format json
    cmd.append(prompt)
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=WORKDIR)
        try:
            d = json.loads(p.stdout)
            return d.get("result", ""), list(d.get("modelUsage", {}).keys()), d.get("is_error", False)
        except Exception:
            return p.stdout.strip(), [], True
    except subprocess.TimeoutExpired:
        return "", [], True


RETRY = 3
import re as _re
LIMIT_RE = _re.compile(r"You've hit your (session|usage) limit")


def call_retry(prompt, model, effort):
    for k in range(RETRY):
        txt, mu, err = call(prompt, model, effort)
        if txt.strip():
            return txt, mu, err
        time.sleep(2 * (k + 1))
    return "", mu, err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default="")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--model", default="claude-opus-5")
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()

    P = [json.loads(l) for l in open(os.environ.get("PROMPTS", "prompts_fail7.jsonl"))]
    if a.ids:
        keep = set(x.strip() for x in a.ids.split(","))
        P = [p for p in P if p["id"] in keep]

    done = set()
    if os.path.exists(a.out):
        for l in open(a.out):
            done.add(json.loads(l)["id"])
        print(f"[resume] {len(done)} done", flush=True)

    lock, stop = threading.Lock(), threading.Event()

    def one(p):
        if stop.is_set():
            return
        t0 = time.time()
        samples, seen = [], set()
        for k in range(a.n):
            txt, mu, err = call_retry(SYS_WRAP.format(user=p["user"]), a.model, a.effort)
            if LIMIT_RE.search(txt):
                print(f"LIMIT_HIT {p['id']}: {txt.strip()[:80]}", flush=True)
                stop.set()
                return
            seen.update(mu)
            samples.append({"text": txt, "finish": "error" if err else "stop", "reason_len": 0})
        with lock, open(a.out, "a") as f:
            f.write(json.dumps({**p, "model": f"{a.model}/{a.effort}",
                                "model_verified": sorted(seen), "samples": samples},
                               ensure_ascii=False) + "\n")
        print(f"[{p['id']}] {time.time()-t0:.0f}s", flush=True)

    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(one, [p for p in P if p["id"] not in done]))
    if stop.is_set():
        sys.exit(42)
    print("CLAUDE_DONE", flush=True)


if __name__ == "__main__":
    main()
