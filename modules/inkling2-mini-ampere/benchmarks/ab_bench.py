"""One A/B pass on a running server: short decode, two-session decode, and a
cold ~64K prefill followed by long-context decode. Prints one JSON object."""

import json
import statistics
import sys
import threading
import time
import urllib.request
import uuid

from probe import BASE, MODEL, make_prompt

CODE = ("Continue this Python module with code only. Implement a small parser and tests "
        "for a log of colored objects.\n\nfrom dataclasses import dataclass\n\n")
PROSE = "Write a detailed guide to starting a small garden, with concrete examples.\n\n1. Choose a site\n"


def stream(path, body, timeout=600):
    req = urllib.request.Request(BASE + path, data=json.dumps({
        "model": MODEL, "temperature": 0, "ignore_eos": True, "stream": True,
        "stream_options": {"include_usage": True}, **body}).encode(),
        headers={"Content-Type": "application/json"})
    start = time.monotonic()
    first = last = None
    tokens = 0
    text = []
    with urllib.request.urlopen(req, timeout=timeout) as response:
        for line in response:
            if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                continue
            event = json.loads(line[6:])
            for choice in event.get("choices", []):
                piece = choice.get("text") or ""
                delta = choice.get("delta") or {}
                piece = piece or delta.get("content") or delta.get("reasoning") or ""
                if piece:
                    text.append(piece)
                    last = time.monotonic()
                    first = first or last
            if usage := event.get("usage"):
                tokens = usage["completion_tokens"]
    return {"ttft_s": first - start, "decode_s": last - first, "tokens": tokens,
            "tok_s": (tokens - 1) / (last - first), "text": "".join(text)}


def complete(prompt, max_tokens=384):
    return stream("/v1/completions", {"prompt": prompt, "max_tokens": max_tokens})


def main():
    result = {}
    for name, prompt in (("code", CODE), ("prose", PROSE)):
        complete(prompt)  # warm-up
        runs = [complete(prompt) for _ in range(2)]
        result[name] = {"tok_s": statistics.median(r["tok_s"] for r in runs),
                        "text_head": runs[0]["text"][:400]}

    barrier = threading.Barrier(2)
    pair = [None, None]

    def worker(i, prompt):
        barrier.wait()
        pair[i] = complete(prompt)

    threads = [threading.Thread(target=worker, args=(i, p)) for i, p in enumerate((CODE, PROSE))]
    start = time.monotonic()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.monotonic() - start
    result["pair"] = {"per_session_tok_s": [r["tok_s"] for r in pair],
                      "aggregate_tok_s": sum(r["tokens"] for r in pair) / wall}

    prompt, count = make_prompt(64000, f"cobalt lantern {uuid.uuid4().hex[:8]}")
    long = stream("/v1/chat/completions", {
        "messages": [{"role": "user", "content": prompt}], "max_tokens": 256,
        "chat_template_kwargs": {"reasoning_effort": "none"}})
    result["long64k"] = {"prompt_tokens": count, "ttft_s": long["ttft_s"], "decode_tok_s": long["tok_s"]}
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    sys.exit(main())
