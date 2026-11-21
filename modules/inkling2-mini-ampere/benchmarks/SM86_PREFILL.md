# SM86 full-attention INT8 prefill tuning

The opt-in `INKLING_SM86_INT8_PREFILL=1` applies only when running full-attention INT8 prefill on SM86 with a 128-row query tile and one KV split. It changes the key tile from 32 keys / 4 warps / 2 stages to 128 keys / 8 warps / 2 stages. It leaves BF16 KV, sliding-window attention, and decode/verification tile selection unchanged.

After building the serving image, run the sweep without loading a checkpoint. Use one idle SM86 GPU with at least 3 GiB available for the largest shape:

```bash
docker run --rm --network none --device nvidia.com/gpu=0 \
  --entrypoint python3 \
  --mount type=bind,src="$PWD/benchmarks/tune_int8_prefill.py",dst=/tune.py,readonly \
  local/inkling2-mini-ampere:sm86 /tune.py
```

Five interleaved CUDA-event samples per tile, 2,048 query tokens, 64-token pages, head dimension 128 and 3:1 GQA. Pages are shuffled and offset to match the live BLHNC stride. The candidate must agree with the baseline attention and log-sum-exp before being timed. Larger tiles change online softmax partitioning, so BF16 rounding is not bit-identical. The synthetic kernel tests compare against dense FP32 reference attention as well.

| Context | Baseline / tuned kernel speed ratio |
| ---: | ---: |
| 8,192 | 1.54× |
| 32,768 | 1.62× |
| 131,072 | 1.84× |
| 249,856 | 1.87× (147.28 → 78.76 ms) |

These are single-kernel timings, not total model latency. On two A4500s with the same INT8 + vision + four-token DSpark profile, one cold matched 249,273-token retrieval per build measured **118.7 s → 103.0 s** to first generated token (13.2% less time). Both retrieved the exact phrase; text, image, tools and DSpark smoke tests passed with tuning enabled. The configured 250,000-token limit and 4.93 GiB KV pool per GPU were unchanged. Two simultaneous 250K requests were not tested.

The module defaults are unchanged. Set `MAX_MODEL_LEN=250000` and `INKLING_SM86_INT8_PREFILL=1` in the module's `.env` to try the tested configuration on comparable hardware.
