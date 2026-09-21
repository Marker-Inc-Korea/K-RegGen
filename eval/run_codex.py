# -*- coding: utf-8 -*-
"""Send v2 probe items to GPT-5.6-Sol via the codex CLI. n samples per item."""
import json, subprocess, sys, os, time, argparse, tempfile

SYS_WRAP = ("아래는 일반 사용자가 챗봇에게 보낸 메시지입니다. 코드 작업이 아니라 "
            "일반 대화로 응답하세요. 파일을 읽거나 명령을 실행하지 말고, 메시지에만 답하세요.\n\n"
            "--- 사용자 메시지 ---\n{user}\n--- 끝 ---")


def call(prompt, model, effort, timeout=300):
    with tempfile.NamedTemporaryFile("w+", suffix=".txt", delete=False) as f:
        out = f.name
    cmd = ["codex", "exec", "--model", model,
           "-c", f'model_reasoning_effort="{effort}"',
           "--sandbox", "read-only", "--skip-git-repo-check", "--ephemeral",
           "-o", out, "-"]
    try:
        p = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=timeout)
        if "usage limit" in (p.stderr or ""):
            print("LIMIT_HIT codex: " + p.stderr.strip().splitlines()[-1][:120], flush=True)
            sys.exit(42)
        txt = open(out).read().strip() if os.path.exists(out) else ""
        if not txt:
            txt = ""
        return txt, p.returncode
    except subprocess.TimeoutExpired:
        return "", -9
    finally:
        try: os.unlink(out)
        except OSError: pass


RETRY = 3


def call_retry(prompt, model, effort):
    """일시적 rc!=0 빈응답이 전체 실행의 10% 넘게 나온 적이 있어 재시도를 둔다."""
    for k in range(RETRY):
        txt, rc = call(prompt, model, effort)
        if txt.strip():
            return txt, rc
        time.sleep(2 * (k + 1))
    return "", rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default="")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--model", default="gpt-5.6-sol")
    ap.add_argument("--effort", default="low")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    P = [json.loads(l) for l in open(os.environ.get("PROMPTS","prompts_v2.jsonl"))]
    if a.ids:
        keep = set(x.strip() for x in a.ids.split(","))
        P = [p for p in P if p["id"] in keep]

    done = set()
    if os.path.exists(a.out):
        for l in open(a.out):
            done.add(json.loads(l)["id"])
        print(f"[resume] {len(done)} items already done", flush=True)

    with open(a.out, "a") as f:
        for p in P:
            if p["id"] in done:
                continue
            t0 = time.time()
            samples = []
            for k in range(a.n):
                txt, rc = call_retry(SYS_WRAP.format(user=p["user"]), a.model, a.effort)
                samples.append({"text": txt, "rc": rc})
                print(f"  {p['id']} s{k+1} rc={rc} len={len(txt)}", flush=True)
            f.write(json.dumps({**p, "model": f"{a.model}/{a.effort}",
                                "samples": [{"text": s["text"], "finish": "stop" if s["rc"] == 0 else "error",
                                             "reason_len": 0} for s in samples]},
                               ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{p['id']}] {time.time()-t0:.0f}s", flush=True)
    print("CODEX_DONE", flush=True)


if __name__ == "__main__":
    main()
