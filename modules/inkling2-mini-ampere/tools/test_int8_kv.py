"""Single-GPU test for the opt-in Inkling INT8 KV kernels; no model needed."""

import math

import torch

from vllm.models.inkling2.nvidia.ops.attention_sm90 import (
    MODE_OUT_LSE,
    MODE_PARTIAL,
    rel_attention,
    write_int8_kv,
)


D, H, G, E = 128, 4, 3, 1024


def unpack(packed):
    scales = packed[..., D:].view(torch.float32)
    return (packed[..., :D].float() * scales).bfloat16()


def check_dense_reference(q, k, v, rel, out, lse):
    seq_len, query_len = k.shape[0], q.shape[0]
    token = torch.arange(query_len, device=q.device)
    key = torch.arange(seq_len, device=q.device)
    past = seq_len - query_len
    tile_end = ((token // 128 + 1) * 128).clamp(max=query_len)
    rel_base = ((past + tile_end + 127) // 128) * 128 - (E + 256)
    col = (key[None, :] - rel_base[:, None]).clamp(0, E + 255)
    dist = past + token[:, None] - key[None, :]
    bias = rel.gather(2, col[:, None, :].expand(-1, H * G, -1)).float()
    bias.masked_fill_((dist >= E)[:, None, :], 0)
    k = k.float().repeat_interleave(G, dim=1)
    v = v.float().repeat_interleave(G, dim=1)
    scores = torch.einsum("qhd,khd->qhk", q.float(), k) / math.sqrt(D) + bias
    scores.masked_fill_((dist < 0)[:, None, :], -torch.inf)
    ref = torch.einsum("qhk,khd->qhd", scores.softmax(-1), v)
    torch.testing.assert_close(lse, scores.logsumexp(-1).T, atol=3e-5, rtol=1e-5)
    # Triton's fast path casts probabilities to BF16 before the value MMA.
    torch.testing.assert_close(out, ref, atol=3e-3, rtol=5e-3)


def check_case(query_len, seq_len, page, splits, graph=False):
    dtype = torch.bfloat16
    opts = {"device": "cuda", "dtype": dtype}
    pages = math.ceil(seq_len / page)
    table = torch.randperm(pages, device="cuda", dtype=torch.int32).unsqueeze(0)
    positions = torch.arange(seq_len, device="cuda")
    slots = table[0, positions // page].long() * page + positions % page
    slots = torch.cat((slots, slots.new_tensor([-1])))
    # Use the backend's head-major layout and adjacent K/V rows. Reserve one
    # extra physical page to assert masked writes leave the backing untouched.
    backing = torch.full((pages + 1, H, page, 2 * (D + 4)), 42, device="cuda", dtype=torch.int8)
    kc, vc = backing.transpose(1, 2).split(D + 4, dim=-1)
    k, v = torch.randn((seq_len + 1, H, D * 2), **opts).split(D, dim=-1)
    k[0].zero_()
    v[1].zero_()
    write_int8_kv(k, v, kc, vc, slots)
    assert torch.all(backing[-1] == 42).item()
    physical = slots[:-1]
    for src, packed in ((k, kc), (v, vc)):
        rows = packed[physical // page, physical % page]
        maxabs = src[:-1].float().abs().amax(dim=-1, keepdim=True)
        scale = torch.where(maxabs > 0, maxabs / 127, 1.0)
        torch.testing.assert_close(rows[..., D:].view(torch.float32), scale)
        err = (unpack(rows).float() - src[:-1].float()).abs()
        assert (err <= scale * 0.5 + src[:-1].float().abs() / 256 + 1e-6).all().item()

    q = torch.randn((query_len, H * G, D), **opts)
    rel = torch.randn((query_len, H * G, E + 256), **opts) * 0.25
    seq = table.new_tensor([seq_len])
    cu = table.new_tensor([0, query_len])
    fp32 = {"device": "cuda", "dtype": torch.float32}
    if splits == 1:
        out_shape, lse_shape = q.shape, (H * G, query_len)
        mode = MODE_OUT_LSE
    else:
        out_shape, lse_shape = (splits, *q.shape), (splits, query_len, H * G)
        mode = MODE_PARTIAL

    def run(key, value, out, lse):
        rel_attention(
            q, key, value, out, lse, rel, block_table=table,
            cache_seqlens=seq, cu_seqlens_q=cu, max_query_len=query_len,
            softmax_scale=D ** -0.5, window=0, num_splits=splits, mode=mode,
        )

    out, lse = torch.empty(out_shape, **fp32), torch.empty(lse_shape, **fp32)
    ref, ref_lse = torch.empty_like(out), torch.empty_like(lse)
    run(kc, vc, out, lse)
    dense_k, dense_v = unpack(kc), unpack(vc)
    run(dense_k, dense_v, ref, ref_lse)
    if splits == 1:
        index = (physical // page, physical % page)
        check_dense_reference(q, dense_k[index], dense_v[index], rel, out, lse)
    assert torch.isfinite(ref).all().item() and torch.isfinite(ref_lse).all().item()
    # Triton's packed and dense readers can pick different MMA layouts; the
    # BF16 probability cast makes a tiny rounding delta visible in the output.
    torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(lse, ref_lse, atol=5e-5, rtol=1e-4)
    print(f"max output delta vs unpacked BF16: {(out - ref).abs().max().item():.3g}", flush=True)
    if graph:
        expected_out, expected_lse = out.clone(), lse.clone()
        torch.cuda.synchronize()
        captured = torch.cuda.CUDAGraph()
        with torch.cuda.graph(captured):
            write_int8_kv(k, v, kc, vc, slots)
            run(kc, vc, out, lse)
        out.zero_()
        lse.zero_()
        captured.replay()
        torch.testing.assert_close(out, expected_out, atol=0, rtol=0)
        torch.testing.assert_close(lse, expected_lse, atol=0, rtol=0)
    print(f"PASS q={query_len} kv={seq_len} page={page} splits={splits} graph={graph}", flush=True)


if __name__ == "__main__":
    print(torch.cuda.get_device_name(), torch.cuda.get_device_capability(), flush=True)
    torch.manual_seed(42)
    check_case(query_len=139, seq_len=139, page=64, splits=1)
    check_case(query_len=137, seq_len=1407, page=128, splits=1)
    check_case(query_len=1, seq_len=4099, page=128, splits=4, graph=True)
