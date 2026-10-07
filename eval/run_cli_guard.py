# -*- coding: utf-8 -*-
"""Prompted guard baseline through the local `claude` / `codex` CLIs (no API keys).

Each item gets exactly the guard prompt (train/common.py: system + rule table + request +
response); the answer is parsed like every other guard. Calls run from an empty temporary
directory with tools disabled, so the model only answers. Resumable; stops on a usage limit.

  python3 eval/run_cli_guard.py --cli claude --model claude-opus-5-5 --effort low \
      --input outputs/eval_frozen/real_test.jsonl --out outputs/guard/test_claude-opus-5-5.jsonl
  python3 eval/run_cli_guard.py --cli codex --model gpt-6.1-sol --effort low --gold-sample ... --out ...
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "train"))
from common import GUARD_SYS, clean, guard_user, load_rules, parse_guard, read_jsonl  # noqa: E402

LIMIT = re.compile(r"hit your (session|usage) limit|usage limit", re.I)
WORKDIR = tempfile.mkdtemp(prefix="cliguard-")


def _bin(name):
    """Prefer the user-local CLI (kept current) over an older system install."""
    local = os.path.expanduser(f"~/.local/bin/{name}")
    return os.environ.get(f"{name.upper()}_BIN") or (local if os.path.exists(local) else name)


def call_claude(user, model, effort, timeout=300):
    cmd = [_bin("claude"), "-p", "--model", model, "--effort", effort, "--tools", "", "--no-session-persistence",
           "--system-prompt", GUARD_SYS, "--output-format", "json", user]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=WORKDIR)
    try:
        d = json.loads(p.stdout)
        return d.get("result", ""), d.get("is_error", False)
    except json.JSONDecodeError:
        return (p.stdout or p.stderr).strip(), True


def call_codex(user, model, effort, timeout=300):
    out = os.path.join(WORKDIR, f"o{threading.get_ident()}.txt")
    cmd = [_bin("codex"), "exec", "--model", model, "-c", f'model_reasoning_effort="{effort}"',
           "--sandbox", "read-only", "--skip-git-repo-check", "--ephemeral", "-o", out, "-"]
    p = subprocess.run(cmd, input=f"{GUARD_SYS}\n\n{user}", capture_output=True, text=True,
                       timeout=timeout, cwd=WORKDIR)
    txt = open(out).read().strip() if os.path.exists(out) else ""
    if os.path.exists(out):
        os.unlink(out)
    if not txt and LIMIT.search(p.stderr or ""):
        return p.stderr.strip()[-200:], True
    return txt, p.returncode != 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cli", choices=["claude", "codex"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--effort", default="low")
    ap.add_argument("--input", nargs="*", default=[], help="labelled jsonl (ex_id, domain, request, response)")
    ap.add_argument("--gold-sample", default="", help="bench sample json (key, domain, request, response)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    rules = load_rules()
    items = [dict(key=r["ex_id"], domain=r["domain"], request=r["request"], response=r["response"])
             for p in a.input for r in read_jsonl(p)]
    if a.gold_sample:
        items += [dict(key=x["key"], domain=x["domain"], request=x["request"], response=x["response"])
                  for x in json.load(open(a.gold_sample))]
    done = {r["key"] for r in read_jsonl(a.out)} if os.path.exists(a.out) else set()
    todo = [it for it in items if it["key"] not in done]
    if a.limit:
        todo = todo[:a.limit]
    print(f"[cli-guard] {a.cli}:{a.model}/{a.effort} todo {len(todo)} (done {len(done)})", flush=True)
    call = call_claude if a.cli == "claude" else call_codex
    lock, stop = threading.Lock(), threading.Event()

    def work(it):
        if stop.is_set():
            return
        text, err = "", True
        for _ in range(3):
            try:
                text, err = call(guard_user(rules, it["domain"], it["request"], it["response"] or "(빈 응답)"),
                                 a.model, a.effort)
            except subprocess.TimeoutExpired:
                text, err = "", True
            if LIMIT.search(text or ""):
                print(f"LIMIT_HIT {it['key']}: {text[:120]}", flush=True)
                stop.set()
                return
            if text.strip() and not err:
                break
        label, ids, spans = parse_guard(clean(text))
        row = {"key": it["key"], "guard": f"{a.model}-{a.effort}", "label": label, "rules": ids, "spans": spans,
               "finish": "error" if err else "stop", "answer": text[-400:]}
        with lock:
            with open(a.out, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(work, todo))
    n = len(read_jsonl(a.out)) if os.path.exists(a.out) else 0
    print(f"[cli-guard] written {n}{' (stopped on limit)' if stop.is_set() else ''}", flush=True)
    sys.exit(42 if stop.is_set() else 0)


if __name__ == "__main__":
    main()
