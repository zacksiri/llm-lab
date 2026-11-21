# Inkling 2 Mini on Ampere

A pinned vLLM container for **two NVLink-connected RTX A4500s (SM86, 20 GiB each)**. It serves text, tool calls and one image per request. Full-attention KV uses INT8 with one FP32 scale per token, KV head and K/V stream; the sliding-window KV and short-convolution state stay BF16. The model weights are unchanged. DSpark is optional and uses its own BF16 KV cache.

## Build and run

Prerequisites: Linux, Docker Compose v2 or later, an NVIDIA driver compatible with the pinned vLLM image, and NVIDIA Container Toolkit with CDI enabled. The tested host used driver `580.178.04`. Run `nvidia-ctk cdi list` to confirm the GPU selectors. The TP2 and NCCL configuration is intended for the two NVLink cards; other GPU/topology combinations are untested.

From this module directory:

```bash
hf download thinkingmachines/Inkling-2-Mini \
  --revision ed40e7da625f594fc9c4990f5f66be24873c22d7 \
  --local-dir "$HOME/models/Inkling-2-Mini"
cp .env.example .env
# Edit .env: set MODEL_DIR to the absolute checkpoint path and select the GPUs.

docker compose -f compose.yaml -f compose.dspark.yaml build
docker compose -f compose.yaml -f compose.dspark.yaml up -d
docker compose -f compose.yaml -f compose.dspark.yaml logs -f inkling
```

The server listens on `127.0.0.1:6002`, without an API key. Model loading and first-time Triton compilation may take a few minutes. Speculation is opt-in through `compose.dspark.yaml`; omit it from both commands to run the same INT8 + vision profile without DSpark. Do not start both profiles on the same GPUs. `docker compose down` stops the server but keeps the kernel cache volume.

The DSpark overlay uses four draft tokens and probabilistic sampling. Its decode graphs capture **one and two requests**, with a **10-token** verify budget (`2 × (1 + 4)`). The old one-token capture size left DSpark running without graphs. If you change the draft width or slot count, update the decode capture budget accordingly. Cold prefill remains eager.

## Verify

```bash
curl -fsS http://127.0.0.1:6002/v1/models
python3 tools/smoke.py --dspark       # omit --dspark if the overlay is off
python3 benchmarks/decode_probe.py  # two warmed-up, fixed-length decode probes
python3 benchmarks/probe.py 199500 # cold archive retrieval, just below 200K
```

Set `INKLING_BASE_URL` for the Python probes if you changed `HOST_PORT`. To test the INT8 kernels without serving the model or loading weights:

```bash
docker build -f Dockerfile.int8-kv-test -t local/inkling2-int8-kv-test:sm86 .
docker run --rm --network none --device nvidia.com/gpu=0 local/inkling2-int8-kv-test:sm86
```

## Pi configuration (optional)

Add this entry under `providers` in `~/.pi/agent/models.json` and select it via `/model`. Adjust the URL if you changed the port. Merge into your existing file; don't overwrite other providers.

```json
{
  "llm-lab-inkling": {
  "baseUrl": "http://127.0.0.1:6002/v1",
  "api": "openai-completions",
  "apiKey": "local",
  "models": [{
    "id": "inkling-2-mini",
    "name": "Inkling 2 Mini (Ampere, INT8 KV)",
    "input": ["text", "image"],
    "reasoning": true,
    "contextWindow": 200000,
    "maxTokens": 4096,
    "compat": {
      "supportsDeveloperRole": false,
      "supportsReasoningEffort": false,
      "thinkingFormat": "chat-template",
      "chatTemplateKwargs": {
        "reasoning_effort": {"$var": "thinking.effort", "omitWhenOff": true}
      }
    }
  }]
  }
}
```

## Results and limits

On the tested two-A4500 host, the warmed-up 384-token coding probe improved from **23.2 to 210.4 tok/s** when decode graph capture was fixed (same DSpark profile, median of two timed runs each). The prose probe measured **111.9 tok/s** with graphs; its before benchmark timed out. [Raw timings](benchmarks/). These are short-context decode measurements, not general throughput claims.

The INT8 + vision + DSpark profile retrieved the correct answer from a cold **199,368-token** prompt before the graph change (83.2 s to first token). The graph-enabled server still reports **200,000 tokens per request**, including generation. Its KV pool was **4.96 GiB per GPU**, and the pool is shared across both active slots. Two short concurrent completions passed; two simultaneous 200K requests have **not** been validated with vision and DSpark on. Two ~199K requests overlapped in the earlier INT8 text-only configuration. More VRAM reserved by the allocator is not necessarily more usable KV capacity.

For SM86-specific results, see [INT8 prefill tuning](benchmarks/SM86_PREFILL.md) and the [DSpark verification sweep](benchmarks/SM86_VERIFY.md). Prefill tuning is available as an opt-in; the verification sweep is a microbenchmark only.

## Provenance

Based on [`snailbrainx/inkling-2-mini-sm80` at `23dbf478`](https://github.com/snailbrainx/inkling-2-mini-sm80/tree/23dbf478a5b9fc4dd314b26c9d58c94692c06395). [Upstream notes](UPSTREAM.md) describe that original project, not this module's run procedure. LLM Lab changes include the SM86/TP2 overrides, INT8 KV attention/short-conv integration, FA2 vision fallback, DSpark opt-in, and Docker/Compose/tests. The code overlays the pinned `vllm/vllm-openai` preview; it is not an upstream vLLM build. See [LICENSE](LICENSE) and [NOTICE](NOTICE) for attribution and terms. No model weights or container images are distributed in this repository.
