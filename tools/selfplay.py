"""
Self-play A/B test: agent.py with settings A vs agent.py with settings B,
over several seeds, pooling every deal into one confidence interval.

    python tools/selfplay.py --a "" --b "SAMPLES_POSTFLOP=1200" --seeds 1-6
    python tools/selfplay.py --a "SAMPLES_POSTFLOP=400" --b "SAMPLES_POSTFLOP=2500" --deals 500
    python tools/selfplay.py --a-file opponents/equitybot_v1.py --b ""   # new agent.py vs frozen v1

--a / --b are space-separated KEY=VALUE environment overrides read by agent.py.
Positive numbers mean B is better.
"""

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_env(text):
    return dict(kv.split("=", 1) for kv in text.split()) if text else {}


def parse_seeds(text):
    if "-" in text:
        lo, hi = text.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(s) for s in text.split(",")]


def start_bot(port, overrides, path="agent.py"):
    env = {**os.environ, **overrides, "PORT": str(port)}
    proc = subprocess.Popen([sys.executable, path], cwd=ROOT, env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(150):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=0.5)
            return proc
        except Exception:
            time.sleep(0.1)
    proc.kill()
    raise RuntimeError(f"bot on port {port} did not start")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="", help="env overrides for bot A (agent1)")
    ap.add_argument("--b", default="", help="env overrides for bot B (agent2)")
    ap.add_argument("--a-file", default="agent.py", help="bot file for A (default agent.py)")
    ap.add_argument("--b-file", default="agent.py", help="bot file for B (default agent.py)")
    ap.add_argument("--seeds", default="1-4")
    ap.add_argument("--deals", type=int, default=1000)
    ap.add_argument("--ports", default="5308,5309")
    args = ap.parse_args()
    pa, pb = (int(p) for p in args.ports.split(","))

    pooled = []   # B's result per deal, in bb
    times = {"A": [], "B": []}
    print(f"A: {args.a_file} {args.a or '(defaults)'}   B: {args.b_file} {args.b or '(defaults)'}")
    for seed in parse_seeds(args.seeds):
        a = start_bot(pa, parse_env(args.a), args.a_file)
        b = start_bot(pb, parse_env(args.b), args.b_file)
        try:
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
                out = f.name
            subprocess.run([sys.executable, "judge_engine.py", "--agent1", str(pa), "--agent2", str(pb),
                            "--deals", str(args.deals), "--seed", str(seed), "--json", out],
                           cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
            res = json.load(open(out))
            os.unlink(out)
        finally:
            a.kill(); b.kill(); a.wait(); b.wait()
        per_deal = [-x for x in res["paired_bb"]]          # paired_bb is agent1's view
        pooled += per_deal
        st = res["agent_stats"]
        ta = max(h["time_seconds"] for h in st["agent1"])
        tb = max(h["time_seconds"] for h in st["agent2"])
        errs = sum(h["errors"] for s in st.values() for h in s)
        times["A"].append(ta); times["B"].append(tb)
        print(f"seed {seed}: B {sum(per_deal)/len(per_deal):+.3f} bb/deal  "
              f"(match verdict: {res['verdict']})  time/half A {ta:.2f}s B {tb:.2f}s  errors {errs}")

    n = len(pooled)
    mean = sum(pooled) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in pooled) / (n - 1))
    se = sd / math.sqrt(n)
    print(f"POOLED over {n} deals: B {mean:+.3f} bb/deal, 95% CI [{mean-1.96*se:+.3f}, {mean+1.96*se:+.3f}]"
          f"  | max time/half A {max(times['A']):.2f}s B {max(times['B']):.2f}s")


if __name__ == "__main__":
    main()
