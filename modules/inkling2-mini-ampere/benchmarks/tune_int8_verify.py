"""Time graphed Inkling INT8 verification attention and its split-KV merge."""

import argparse
import json
import random
import statistics

import torch
import triton
from triton.runtime.errors import OutOfResources
from vllm.models.inkling2.nvidia.ops.attention import (
    _rel_decode_combine_kernel,
    plan_decode_attention,
)
from vllm.models.inkling2.nvidia.ops.attention_sm90 import MODE_PARTIAL, rel_attention


PAGE, D, KV_HEADS, GROUP = 64, 128, 4, 3
HEADS, QUERY = KV_HEADS * GROUP, 5
BASE_TILE = (128, 4, 2)
TILES = (BASE_TILE, (64, 4, 2), (64, 4, 1), (128, 4, 1), (128, 8, 2), (256, 8, 2))


def sweep(context, requests, repeat, replays):
    pages = triton.cdiv(context, PAGE)
    width = 2 * (D + 4)
    # Distinct backing pages for each request, compact within this one layer.
    cache = torch.empty((pages * requests, KV_HEADS, PAGE, width), device="cuda", dtype=torch.int8)
    key, value = cache.transpose(1, 2).split(D + 4, dim=-1)
    for stream in (key, value):
        stream[..., :D].random_(-127, 128)
        stream[..., D:].view(torch.float32).fill_(0.01)
    table = torch.randperm(pages * requests, device="cuda", dtype=torch.int32).view(requests, pages)
    lens = table.new_full((requests,), context)
    cu = table.new_tensor([i * QUERY for i in range(requests + 1)])
    tokens = requests * QUERY
    opts = {"device": "cuda", "dtype": torch.bfloat16}
    q = torch.randn((tokens, HEADS, D), **opts)
    bias = torch.randn((tokens, HEADS, 1280), **opts) * 0.25
    gate = torch.randn((tokens, HEADS * D), **opts)
    sinks = torch.randn((tokens, HEADS), **opts)
    out = torch.empty_like(gate)
    baseline_splits = plan_decode_attention(False, requests, QUERY, HEADS, KV_HEADS, 1024)
    configs = [(tile, n) for tile in TILES for n in sorted({8, 16, 24, 32, 48, 64, baseline_splits})]
    baseline = (BASE_TILE, baseline_splits)
    configs.remove(baseline)
    configs.insert(0, baseline)
    partial = torch.empty((max(n for _, n in configs), tokens, HEADS, D), device="cuda")
    lse = torch.empty((partial.shape[0], tokens, HEADS), device="cuda")

    def launch(tile, splits):
        rel_attention(q, key, value, partial[:splits], lse[:splits], bias,
                      block_table=table, cache_seqlens=lens, cu_seqlens_q=cu,
                      max_query_len=QUERY, softmax_scale=D ** -0.5, window=0,
                      num_splits=splits, mode=MODE_PARTIAL, decode_tile=tile)
        _rel_decode_combine_kernel[(tokens, HEADS, D // 32)](
            partial, lse, gate, sinks, out, tokens, tokens, gate.stride(0), sinks.stride(0),
            H=HEADS, D=D, NUM_SPLITS=splits, BLOCK_S=triton.next_power_of_2(splits),
            BLOCK_D=32, GATED=True, num_warps=4, launch_pdl=False,
        )

    graphs = {}
    expected = None
    for config in configs:
        try:
            launch(*config)
            torch.cuda.synchronize()
        except OutOfResources as exc:
            if config == baseline:
                raise
            print(json.dumps({"context": context, "requests": requests, "config": config,
                              "skip": str(exc)}), flush=True)
            continue
        if expected is None:
            assert torch.isfinite(out).all().item()
            expected = out.clone()
        else:
            # Different split boundaries change online softmax and BF16 rounding.
            torch.testing.assert_close(out, expected, atol=2e-3, rtol=5e-3)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            launch(*config)
        graph.replay()
        torch.testing.assert_close(out, expected, atol=2e-3, rtol=5e-3)
        graphs[config] = graph

    samples = {config: [] for config in graphs}
    order = list(graphs)
    rng = random.Random(42)
    for _ in range(repeat):
        rng.shuffle(order)
        for config in order:
            start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            start.record()
            for _ in range(replays):
                graphs[config].replay()
            end.record()
            end.synchronize()
            samples[config].append(start.elapsed_time(end) * 1000 / replays)
    base_us = statistics.median(samples[baseline])
    for (tile, splits), measurements in samples.items():
        latency = statistics.median(measurements)
        print(json.dumps({"context": context, "requests": requests, "query": QUERY,
                          "tile": tile, "splits": splits, "baseline": (tile, splits) == baseline,
                          "median_us": latency, "speedup": base_us / latency,
                          "samples_us": measurements}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contexts", type=int, nargs="+", default=[8192, 65536, 249856])
    parser.add_argument("--requests", type=int, nargs="+", default=[1, 2], choices=[1, 2])
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--replays", type=int, default=20)
    args = parser.parse_args()
    assert torch.cuda.get_device_capability() == (8, 6), "benchmark is for SM86"
    torch.manual_seed(42)
    for context in args.contexts:
        for requests in args.requests:
            sweep(context, requests, args.repeat, args.replays)
            torch.cuda.empty_cache()
