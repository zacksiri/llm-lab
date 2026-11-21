#!/usr/bin/env python3
"""Sum GPU kernel time by kernel name from a torch profiler trace (vLLM --profiler-config, *.pt.trace.json[.gz]).
Written to find where an Inkling 2 Mini decode step spends its time on a 170HX.

  python3 trace_kernels.py TRACE [--steps N] [--top 40]

Prints each kernel name (shortened) with total ms, calls, mean us and share of all kernel time; with --steps N
(decode steps in the trace) also ms per step. GPU busy time vs wall span shows idle gaps between kernels.
"""
import argparse
import collections
import gzip
import json
import re


def short(name):
    name = re.sub(r"<.*", "", name)
    name = re.sub(r"\(.*", "", name)
    return name[:90]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--top", type=int, default=40)
    a = ap.parse_args()
    op = gzip.open if a.trace.endswith(".gz") else open
    ev = json.load(op(a.trace, "rt"))["traceEvents"]
    k = [e for e in ev if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and "dur" in e]
    if not k:
        print("no GPU kernel events")
        return
    tot = collections.defaultdict(float)
    n = collections.Counter()
    for e in k:
        s = short(e["name"])
        tot[s] += e["dur"]
        n[s] += 1
    busy = sum(tot.values())
    t0 = min(e["ts"] for e in k)
    t1 = max(e["ts"] + e["dur"] for e in k)
    print(f"{len(k)} GPU events; busy {busy / 1000:.1f} ms over a {(t1 - t0) / 1000:.1f} ms span "
          f"({busy / (t1 - t0):.0%} busy)" + (f"; {a.steps} steps: {(t1 - t0) / 1000 / a.steps:.2f} ms/step" if a.steps else ""))
    print(f"{'kernel':90s} {'ms':>8s} {'calls':>7s} {'mean us':>8s} {'share':>6s}" + ("  ms/step" if a.steps else ""))
    for s, v in sorted(tot.items(), key=lambda x: -x[1])[:a.top]:
        print(f"{s:90s} {v / 1000:8.2f} {n[s]:7d} {v / n[s]:8.1f} {v / busy:6.1%}" +
              (f"  {v / 1000 / a.steps:7.3f}" if a.steps else ""))


if __name__ == "__main__":
    main()
