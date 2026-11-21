"""Probe the running vLLM server with cold synthetic archive retrieval prompts."""

import json
import os
import sys
import threading
import time
import urllib.request


BASE = os.environ.get("INKLING_BASE_URL", "http://127.0.0.1:6002").rstrip("/")
MODEL = "inkling-2-mini"
_OUTPUT_LOCK = threading.Lock()


def emit(event):
    with _OUTPUT_LOCK:
        print(json.dumps(event), flush=True)


def request(path, body=None, timeout=600):
    data = None if body is None else json.dumps(body).encode()
    return urllib.request.urlopen(
        urllib.request.Request(BASE + path, data=data, headers={"Content-Type": "application/json"}),
        timeout=timeout,
    )


def tokenize(prompt):
    with request("/tokenize", {"model": MODEL, "prompt": prompt}, 30) as response:
        return json.load(response)["count"]


def make_prompt(target, phrase):
    header = f"Archive {target} verification record. The exact verification phrase is: {phrase}.\n"
    query = "\nWhat was the exact archive verification phrase? Reply with only that phrase."
    rows = target // 15
    for _ in range(8):
        filler = "\n".join(f"Log {i:06}: equipment inspection complete; no repairs required." for i in range(rows))
        prompt = header + filler + query
        count = tokenize(prompt)
        if target - 280 <= count <= target:
            return prompt, count
        rows = max(1, round(rows * (target - 140) / count))
    raise RuntimeError(f"Could not size prompt for {target}; last count {count}")


def probe(target, phrase, *, prepared=None, max_tokens=512):
    prompt, count = prepared if prepared is not None else make_prompt(target, phrase)
    emit({"event": "start", "epoch": time.time(), "target": target, "raw_tokens": count, "phrase": phrase})
    start = time.monotonic()
    first = None
    content = []
    usage = None
    finish = None
    with request("/v1/chat/completions", {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "chat_template_kwargs": {"reasoning_effort": "none"},
        "cache_salt": f"bf16-probe-{target}-{time.time_ns()}",
        "stream": True,
        "stream_options": {"include_usage": True},
    }) as response:
        for line in response:
            if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            usage = chunk.get("usage") or usage
            for choice in chunk.get("choices", []):
                delta = choice["delta"]
                finish = choice.get("finish_reason") or finish
                if first is None and any(delta.get(k) for k in ("content", "reasoning", "reasoning_content")):
                    first = time.monotonic() - start
                content.append(delta.get("content") or "")
    text = "".join(content).strip()
    ok = text.lower().rstrip(".") == phrase
    emit({
        "event": "result", "epoch": time.time(), "target": target, "ttft_s": first, "phrase": phrase,
        "wall_s": time.monotonic() - start, "usage": usage, "finish_reason": finish,
        "result": text, "ok": ok,
    })
    return bool(ok and usage and abs(usage["prompt_tokens"] - target) <= 300)


if __name__ == "__main__":
    passed = True
    for i, arg in enumerate(sys.argv[1:]):
        target = int(arg)
        phrase = ("cobalt lantern 394", "silver bridge 731", "amber compass 582")[i]
        passed = probe(target, phrase) and passed
    raise SystemExit(0 if passed else 1)
