"""Compare eager and captured DSpark with identical fixed-length output probes."""

import json
import statistics
import time
import urllib.request

from probe import BASE, MODEL


def run(prompt):
    body = json.dumps({
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": 384,
        "ignore_eos": True,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()
    request = urllib.request.Request(
        f"{BASE}/v1/completions", data=body,
        headers={"Content-Type": "application/json"},
    )
    start = time.monotonic()
    first = last = tokens = None
    with urllib.request.urlopen(request, timeout=90) as response:
        for line in response:
            if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                continue
            event = json.loads(line[6:])
            for choice in event.get("choices", []):
                if choice.get("text") or choice.get("reasoning"):
                    last = time.monotonic()
                    if first is None:
                        first = last
            if usage := event.get("usage"):
                tokens = usage["completion_tokens"]
    if tokens != 384 or first is None or last <= first:
        raise RuntimeError(f"incomplete timing: {tokens=} {first=} {last=}")
    return {"ttft_s": first - start, "decode_s": last - first,
            "completion_tokens": tokens, "tok_s": (tokens - 1) / (last - first)}


if __name__ == "__main__":
    prompts = [
        "Continue this Python module with code only. Implement a small parser and tests for a log of colored objects.\n\nfrom dataclasses import dataclass\n\n",
        "Write a detailed guide to starting a small garden, with concrete examples.\n\n1. Choose a site\n",
    ]
    for prompt in prompts:
        run(prompt)
        samples = [run(prompt) for _ in range(2)]
        print(json.dumps({"prompt": prompt.splitlines()[0], "runs": samples,
                          "median_tok_s": statistics.median(s["tok_s"] for s in samples)}), flush=True)
