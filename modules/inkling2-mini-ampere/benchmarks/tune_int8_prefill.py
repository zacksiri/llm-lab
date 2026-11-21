"""Sweep SM86 full-attention INT8 prefill tiles against the serial baseline."""

import argparse
import json
import random
import statistics

import torch
from triton.runtime.errors import OutOfResources

from vllm.models.inkling2.nvidia.ops.attention_sm90 import MODE_OUT_LSE, rel_attention


BASELINE = (32, 4, 2)
TILES = (BASELINE, (16, 4, 2), (32, 4, 1), (32, 8, 2),
         (64, 4, 1), (64, 4, 2), (64, 8, 2), (64, 8, 3), (128, 8, 2))


def sweep(context, query_len, repeat):
    page, heads, dim = 64, 4, 128
    pages = (context + page - 1) // page
    width = 2 * (dim + 4)
    # Consecutive pages of this layer are separated by other layers in BLHNC.
    # Only fill this layer's data; the gap mimics the live DSpark pool's stride.
    payload = heads * page * width
    stride = 655360
    storage = torch.empty((pages - 1) * stride + payload, device="cuda", dtype=torch.int8)
    cache = storage.as_strided((pages, heads, page, width),
                               (stride, page * width, width, 1))
    key, value = cache.transpose(1, 2).split(dim + 4, dim=-1)
    for tensor in (key, value):
        tensor[..., :dim].random_(-127, 128)
        tensor[..., dim:].view(torch.float32).fill_(0.01)
    table = torch.randperm(pages, device="cuda", dtype=torch.int32).unsqueeze(0)
    q = torch.randn(query_len, 3 * heads, dim, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(query_len, 3 * heads, 1280, device="cuda", dtype=torch.bfloat16) * 0.25
    out = torch.empty_like(q, dtype=torch.float32)
    lse = torch.empty((3 * heads, query_len), device="cuda", dtype=torch.float32)
    seq = table.new_tensor([context])
    cu = table.new_tensor([0, query_len])

    def call(tile):
        rel_attention(q, key, value, out, lse, bias, block_table=table, cache_seqlens=seq,
                      cu_seqlens_q=cu, max_query_len=query_len, softmax_scale=dim ** -0.5,
                      window=0, num_splits=1, mode=MODE_OUT_LSE, prefill_tile=tile)

    call(BASELINE)
    expected_out, expected_lse = out.clone(), lse.clone()
    assert torch.isfinite(expected_out).all().item() and torch.isfinite(expected_lse).all().item()
    timings = {}
    for tile in TILES:
        try:
            call(tile)
            torch.cuda.synchronize()
        except OutOfResources as exc:
            print(json.dumps({"context": context, "tile": tile, "skip": str(exc)}), flush=True)
            continue
        # A different online-softmax partition changes BF16 probability rounding.
        torch.testing.assert_close(out, expected_out, atol=3e-3, rtol=5e-3)
        torch.testing.assert_close(lse, expected_lse, atol=2e-4, rtol=2e-5)
        timings[tile] = []

    rng = random.Random(42)
    for _ in range(repeat):
        order = list(timings)
        rng.shuffle(order)
        for tile in order:
            start = torch.cuda.Event(enable_timing=True)
            stop = torch.cuda.Event(enable_timing=True)
            start.record()
            call(tile)
            stop.record()
            stop.synchronize()
            timings[tile].append(start.elapsed_time(stop))
    base_ms = statistics.median(timings[BASELINE])
    for tile, samples in timings.items():
        latency = statistics.median(samples)
        print(json.dumps({"context": context, "query": query_len, "tile": tile,
                          "median_ms": latency, "speedup": base_ms / latency,
                          "samples_ms": samples}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contexts", type=int, nargs="+", default=[8192, 32768, 131072, 249856])
    parser.add_argument("--query", type=int, default=2048)
    parser.add_argument("--repeat", type=int, default=5)
    args = parser.parse_args()
    assert torch.cuda.get_device_capability() == (8, 6), "benchmark is for SM86"
    torch.manual_seed(42)
    for context in args.contexts:
        sweep(context, args.query, args.repeat)
        torch.cuda.empty_cache()
