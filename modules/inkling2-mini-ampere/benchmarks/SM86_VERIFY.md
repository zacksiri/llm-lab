# SM86: INT8 decode verification sweep

On an RTX A4500, replayed CUDA graphs with five query tokens per request (four DSpark drafts plus the target token), one or two requests, and 3:1 GQA with four local KV heads. This times one full-attention INT8 layer **including split-KV merge and gate/sink**, not the drafter or whole model. Candidates were checked against the original kernel, including after graph replay. The table uses medians of five interleaved samples, 20 replays each. Pages are shuffled, independent per request, with compact single-layer backing rather than the production BLHNC inter-layer stride.

| Context per request | Requests | Baseline | Fastest candidate | Speedup |
| ---: | ---: | ---: | ---: | ---: |
| 8,192 | 1 | 56.9 µs | 43.6 µs (24 splits) | 1.30× |
| 8,192 | 2 | 100.5 µs | 74.5 µs (32 splits) | 1.35× |
| 65,536 | 1 | 293.4 µs | 231.1 µs (24 splits) | 1.27× |
| 65,536 | 2 | 593.9 µs | 416.3 µs (48 splits) | 1.43× |
| 249,856 | 1 | 1,079.5 µs | 857.1 µs (48 splits) | 1.26× |
| 249,856 | 2 | 2,166.6 µs | 1,576.7 µs (48 splits) | 1.37× |

The original tile was **128 keys, 4 warps, 2 pipeline stages** and used 14 splits for one request or 7 for two. All six winners used **128 keys, 4 warps, 1 stage**. A fixed **48 splits** gained 1.24–1.43× across these shapes and is a good first A/B candidate because it does not require context-dependent CUDA graphs. Two near-250K requests here are a synthetic kernel shape, not a claim that the server admits them concurrently. No serving defaults were changed; a TP2 end-to-end benchmark is needed to establish a generation speedup.

From the module directory, after building its serving image:

```bash
docker run --rm --network none --device nvidia.com/gpu=0 \
  -e INKLING_SM80=1 --entrypoint python3 \
  --mount type=bind,src="$PWD/benchmarks/tune_int8_verify.py",dst=/tune.py,readonly \
  local/inkling2-mini-ampere:sm86 /tune.py
```

Use an idle SM86 GPU and substitute its CDI selector. No checkpoint is needed. [Raw samples](sm86-verify-results/sweep.jsonl).
