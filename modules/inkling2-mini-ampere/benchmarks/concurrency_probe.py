"""Send two independent cold requests and monitor the shared vLLM KV pool."""

from concurrent.futures import ThreadPoolExecutor
import json
import sys
import threading
import time

from probe import make_prompt, probe, request


METRICS = (
    "num_requests_running", "num_requests_waiting", "num_preemptions_total",
    "kv_cache_usage_perc", "prompt_tokens_total", "generation_tokens_total",
)


def sample():
    result = {"epoch": time.time()}
    for key in METRICS:
        result[key] = 0.0
    with request("/metrics", timeout=3) as response:
        for line in response:
            for key in METRICS:
                if line.startswith(f"vllm:{key}{{".encode()):
                    result[key] += float(line.split()[-1])
    return result


def main():
    target = int(sys.argv[1])
    phrases = ("silver bridge 731", "amber compass 582")
    prepared = [make_prompt(target, phrase) for phrase in phrases]
    barrier = threading.Barrier(len(phrases) + 1, timeout=10)

    def worker(phrase, prompt):
        barrier.wait()
        return probe(target, phrase, prepared=prompt)

    with open("concurrency-metrics.jsonl", "w", buffering=1) as log:
        initial = sample()
        print(json.dumps(initial), file=log)
        latest = initial
        simultaneous = 0
        peak = latest["kv_cache_usage_perc"]
        with ThreadPoolExecutor(max_workers=2) as pool:
            jobs = [pool.submit(worker, phrase, prompt) for phrase, prompt in zip(phrases, prepared)]
            barrier.wait()
            while not all(job.done() for job in jobs):
                latest = sample()
                print(json.dumps(latest), file=log)
                simultaneous += latest["num_requests_running"] >= 2
                peak = max(peak, latest["kv_cache_usage_perc"])
                time.sleep(0.5)
            final = sample()
            print(json.dumps(final), file=log)
            preemptions = final["num_preemptions_total"] - initial["num_preemptions_total"]
            print(json.dumps({"event": "summary", "two_running_samples": simultaneous,
                              "peak_kv_fraction": peak, "preemptions": preemptions}), flush=True)
            passed = [job.result() for job in jobs]
    return 0 if all(passed) and simultaneous and not preemptions else 1


if __name__ == "__main__":
    raise SystemExit(main())
