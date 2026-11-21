"""Check text, one image, tools, and (optionally) DSpark on the local API."""

import argparse
import base64
import json
import os
from pathlib import Path
from urllib.request import Request, urlopen


BASE = os.environ.get("INKLING_BASE_URL", "http://127.0.0.1:6002").rstrip("/")


def chat(messages, **kwargs):
    payload = json.dumps({
        "model": "inkling-2-mini", "messages": messages, "max_tokens": 512,
        "temperature": 0, "chat_template_kwargs": {"reasoning_effort": "none"},
        **kwargs,
    }).encode()
    req = Request(f"{BASE}/v1/chat/completions", data=payload,
                  headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=120) as response:
        return json.load(response)["choices"][0]["message"]


def accepted_tokens():
    with urlopen(f"{BASE}/metrics", timeout=5) as response:
        return sum(float(line.split()[-1]) for line in response
                   if line.startswith(b"vllm:spec_decode_num_accepted_tokens_total{"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dspark", action="store_true", help="expect accepted draft tokens")
    options = parser.parse_args()
    before = accepted_tokens() if options.dspark else 0

    reply = chat([{"role": "user", "content": "Reply with exactly the result of 9 times 7."}])
    assert reply["content"].strip().rstrip(".") == "63", reply
    print("PASS text")

    path = Path(__file__).resolve().parent.parent / "fixtures" / "red.png"
    uri = "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")
    reply = chat([{"role": "user", "content": [
        {"type": "text", "text": "What color is this image? Answer in one word."},
        {"type": "image_url", "image_url": {"url": uri}},
    ]}])
    assert reply["content"].strip().lower().rstrip(".") == "red", reply
    print("PASS image")

    messages = [{"role": "user", "content": "Use get_temperature for Bangkok. Do not guess."}]
    tools = [{"type": "function", "function": {
        "name": "get_temperature", "description": "Look up a city's temperature",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                       "required": ["city"]},
    }}]
    reply = chat(messages, tools=tools)
    call = reply["tool_calls"][0]
    assert call["function"]["name"] == "get_temperature", call
    assert "bangkok" in json.loads(call["function"]["arguments"])["city"].lower(), call
    follow_up = chat(messages + [reply, {"role": "tool", "tool_call_id": call["id"],
                                          "content": "31 degrees Celsius"}], tools=tools)
    assert "31" in follow_up["content"], follow_up
    print("PASS tools")

    if options.dspark:
        count = accepted_tokens() - before
        assert count > 0, "no draft tokens accepted"
        print(f"PASS DSpark ({int(count)} accepted draft tokens)")


if __name__ == "__main__":
    main()
