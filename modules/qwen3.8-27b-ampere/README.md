# Qwen3.8-27B on two RTX A4500s

A separate, offline vLLM profile for **2× RTX A4500 (SM86, 20 GiB each), joined by NVLink**. It uses the
already cached [palmfuture GPTQ INT4 checkpoint](https://huggingface.co/palmfuture/Qwen3.8-27B-GPTQ-Int4/tree/d85a556c59b2959d083d685a8d1db100f4aeb88d)
and pins stock [vLLM v0.34.0](https://github.com/vllm-project/vllm/releases/tag/v0.34.0)
(CUDA 13.0, amd64). Baseline: **64 Ki tokens per request**, two slots, BF16 KV, text only, and a
loopback API on port **6003**. This is an evaluation profile, not a replacement for `llm.service`.

## What carries over from Inkling?

| Setting | Qwen profile |
| --- | --- |
| Two NVLink GPUs, TP2, NCCL restricted to NVLink, 1,024-token prefill chunks, two slots | Kept as starting points. On stock vLLM use its ordinary NCCL path first; if there is a collective failure, test `QWEN_DISABLE_PYNCCL=1` separately. |
| Decode-only CUDA graphs for one or two requests | Kept. Set `QWEN_GRAPH_MODE=NONE` if graph capture fails. |
| Inkling SM86 kernels, Humming tiles, vocab/TP patch, and BLHNC cache layout | Not loaded. The Inkling preview has limited compiled kernel coverage. The stock release [includes SM86 in its build targets](https://github.com/vllm-project/vllm/blob/v0.34.0/docker/Dockerfile) and supports GPTQ/Marlin on Ampere. |
| Inkling's INT8 attention cache and DSpark | Use stock Triton INT8 KV and Qwen's own MTP head as separate experiments; no Inkling patches needed. |
| Inkling parsers | Replaced with `qwen3` and `qwen3_coder` from the [vLLM Qwen recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-27B). |

The cached GPTQ checkpoint is ~19.5 GiB on disk including vision and the MTP head. The official BF16
checkpoint is ~51.7 GiB on disk and cannot fit fully on the pair. Official FP8 (~28.7 GiB)
is a possible comparison, but Ampere lacks native FP8 tensor math and the bigger weights leave less
context headroom. The existing `~/llm/compose.qwen.yaml` is a separate llama.cpp GGUF setup;
its split, quantization and throughput do not transfer to vLLM.

## Start during a maintenance window

From this directory, stage the stock image while Inkling is still serving (the amd64 manifest has
about 10 GB of compressed layers). The default cache path and revision already exist on this host.
If needed, populate them with `hf download palmfuture/Qwen3.8-27B-GPTQ-Int4 --revision d85a556c59b2959d083d685a8d1db100f4aeb88d`.
On another host, check `nvidia-ctk cdi list` and `nvidia-smi topo -m`, and set `QWEN_GPU_0`
and `QWEN_GPU_1` to the NVLink pair's CDI indices or UUIDs. The RTX A1000 is not used.

```sh
cd ~/Projects/llm-lab/modules/qwen3.8-27b-ampere
docker compose config -q
docker compose pull
# Different port, same GPUs: do not start Qwen until Inkling has stopped.
sudo systemctl stop llm.service
docker compose up -d
docker compose logs -f qwen
```

Once startup finishes, check the model, a non-thinking chat, and the reported KV pool:

```sh
curl -fsS http://127.0.0.1:6003/v1/models
curl -fsS http://127.0.0.1:6003/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"Reply with the word READY."}],"chat_template_kwargs":{"enable_thinking":false},"temperature":0,"max_tokens":32}' \
  | jq -er '.choices[0].message.content | select(type == "string" and length > 0)'
docker compose logs qwen | grep -E 'Marlin|GPU KV cache size|Maximum concurrency'
```

Return the GPUs to Inkling with `docker compose down` and `sudo systemctl start llm.service`.
The lab Compose project, port and kernel cache are separate from the live deployment.

## Test one change at a time

1. **Context.** Try `QWEN_MAX_MODEL_LEN=131072 docker compose up -d` after the baseline. The model
   has 16 full-attention and 48 gated-delta layers. At TP2, full-attention BF16 KV alone costs
   **32 KiB/token/GPU**, or **8 GiB/GPU** for one native-max (262,144 token) request. This excludes
   recurrent state, block padding, graph memory and workspace. Read actual KV capacity from the
   startup logs and test cold retrieval at the target length; the two slots share that pool.
2. **INT8 KV.** Use upstream `int8_per_token_head` with `TRITON_ATTN` as a separate A/B:
   `QWEN_ATTENTION_BACKEND=TRITON_ATTN QWEN_KV_CACHE_DTYPE=int8_per_token_head docker compose up -d`.
   This changes full-attention KV (packed INT8 with dynamic FP32 scales), not the recurrent-state
   precision. The raw full-attention footprint is ~4.06 GiB/GPU at 262,144 tokens, but that is not a
   capacity or speed guarantee. To isolate quantization from the backend change, compare with
   `QWEN_ATTENTION_BACKEND=TRITON_ATTN` and the default BF16 KV. Check retrieval quality, time to
   first token and decode throughput before increasing context or switching from FA2/BF16.
3. **MTP.** `docker compose -f compose.yaml -f compose.mtp.yaml up -d` enables one draft token.
   Only after it passes, compare `QWEN_MTP_K=2` and `3`. Check acceptance in
   `/metrics` (`spec_decode_num_accepted_tokens` and `spec_decode_num_draft_tokens`) and compare
   decode speed at your actual sampling settings. Use plain `docker compose up -d` to revert.
4. **Vision.** `QWEN_VISION_FLAG=--no-language-model-only docker compose up -d` allows one image
   per request. Test separately since the vision encoder changes the memory budget.

Compose and cached checkpoint layout were checked without stopping Inkling. Qwen model load,
kernel selection, tool calls and throughput on this host still need a GPU test.
