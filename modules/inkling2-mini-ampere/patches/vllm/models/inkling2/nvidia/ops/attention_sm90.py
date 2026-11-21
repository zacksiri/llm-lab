# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Modified for the Inkling 2 Mini Ampere module in LLM Lab.
"""Inkling2 relative-bias paged attention for Hopper and SM12x (Triton).

Stands in for tml_fa4 (prefill) and the tcgen05 decode/verify kernel of
`ops/attention.py`, reading the same sheared relative logits of
`attention_prologue`: column `c` of token `t` holds key
`c + cdiv(past + tile_end, 128) * 128 - (E + 256)`, where `tile_end` ends
the token's 128-row query tile (see `qr_clip_rel_tokens`).

A program owns `BLOCK_M` rows (tokens x GQA group) of one request and KV
head, and one KV split: split `s` of `S` takes key blocks `s, s + S, ...`
of the rows' visible range, so a split count that depends only on the batch
holds for every replay of a CUDA graph.
"""

import os

import torch
from triton.language.extra.cuda import libdevice

from vllm.triton_utils import tl, triton

# The query tile of tml_fa4 that shears the relative logits.
_SHEAR_TILE = tl.constexpr(128)

# Output of `_rel_attention_kernel`: fp32 output and natural-log LSE (the
# tml_fa4 replacement), the gated bf16 output, or fp32 split partials with
# log2 LSEs (merged by `_rel_decode_combine_kernel`).
MODE_OUT_LSE = 0
MODE_GATED = 1
MODE_PARTIAL = 2

_TUNE_SM86_INT8_PREFILL = os.environ.get("INKLING_SM86_INT8_PREFILL") == "1"
_TUNE_SM86_INT8_VERIFY = os.environ.get("INKLING_SM86_INT8_VERIFY_TILE") == "1"


@triton.jit
def _sigmoid(x):
    return 1.0 / (1.0 + tl.exp2(-x * 1.4426950408889634))


@triton.jit
def _load_fp8_kv(ptr, offsets, mask, dtype: tl.constexpr, D: tl.constexpr):
    # Load each E4M3 pair once, retaining 16-bit packing for BF16 MMA order.
    half = tl.arange(0, D // 2)
    pair = tl.load(
        ptr.to(tl.pointer_type(tl.uint16))
        + tl.multiple_of(offsets // 2, D // 2)[:, None]
        + half[None, :],
        mask=mask[:, None],
        other=0,
    )
    lo, hi = tl.inline_asm_elementwise(
        "{ .reg .b32 pair; cvt.rn.f16x2.e4m3x2 pair, $2; mov.b32 {$0, $1}, pair; }",
        "=h,=h,h",
        [pair],
        dtype=(tl.float16, tl.float16),
        is_pure=True,
        pack=1,
    )
    return tl.interleave(lo, hi).to(dtype)


@triton.jit
def quantize_int8_kv(x, axis: tl.constexpr):
    # Round to the same BF16 inputs the unquantized attention path sees.
    x = x.to(tl.bfloat16).to(tl.float32)
    abs_max = tl.max(tl.abs(x), axis=axis)
    scale = tl.where(abs_max > 0, abs_max / 127.0, 1.0)
    codes = tl.clamp(
        libdevice.rint(x / tl.expand_dims(scale, axis)), -127.0, 127.0
    ).to(tl.int8)
    return codes, scale


@triton.jit
def _store_int8_kv_row(src, dst, src_offset, dst_offset, D: tl.constexpr):
    cols = tl.arange(0, D)
    x = tl.load(src + src_offset + cols)
    codes, scale = quantize_int8_kv(x, 0)
    tl.store(dst + dst_offset + cols, codes)
    tl.store((dst + dst_offset + D).to(tl.pointer_type(tl.float32)), scale)


@triton.jit
def _write_int8_kv_kernel(
    K, V, KCache, VCache, Slots,
    stride_kt, stride_kh, stride_vt, stride_vh,
    stride_page, stride_slot, stride_head,
    PAGE: tl.constexpr, D: tl.constexpr,
):
    token, head = tl.program_id(0), tl.program_id(1)
    slot = tl.load(Slots + token).to(tl.int64)
    if slot >= 0:
        offset = (slot // PAGE) * stride_page + (slot % PAGE) * stride_slot + head * stride_head
        _store_int8_kv_row(K, KCache, token * stride_kt + head * stride_kh, offset, D)
        _store_int8_kv_row(V, VCache, token * stride_vt + head * stride_vh, offset, D)


def _check_int8_cache(cache: torch.Tensor, head_dim: int) -> None:
    # Each row holds signed codes followed by one aligned float32 scale.
    assert cache.ndim == 4 and cache.dtype == torch.int8
    assert cache.shape[-1] == head_dim + 4 and cache.stride(-1) == 1
    assert head_dim >= 4 and head_dim & (head_dim - 1) == 0
    assert cache.data_ptr() % 4 == 0 and all(s % 4 == 0 for s in cache.stride()[:-1])


def write_int8_kv(
    k: torch.Tensor,
    v: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """Pack prepared BF16 K/V into paged int8 rows; negative slots are skipped."""
    tokens, heads, head_dim = k.shape
    assert k.dtype == v.dtype == torch.bfloat16 and k.shape == v.shape
    assert k.stride(-1) == v.stride(-1) == 1
    assert slot_mapping.dtype == torch.int64 and slot_mapping.is_contiguous()
    assert slot_mapping.shape == (tokens,)
    _check_int8_cache(key_cache, head_dim)
    _check_int8_cache(value_cache, head_dim)
    assert key_cache.shape == value_cache.shape and key_cache.shape[2] == heads
    assert key_cache.stride() == value_cache.stride()
    if tokens:
        _write_int8_kv_kernel[(tokens, heads)](
            k, v, key_cache, value_cache, slot_mapping,
            k.stride(0), k.stride(1), v.stride(0), v.stride(1),
            *key_cache.stride()[:3], key_cache.shape[1], head_dim,
        )


@triton.jit
def _load_int8_kv(ptr, offsets, mask, dtype: tl.constexpr, D: tl.constexpr):
    cols = tl.arange(0, D)
    codes = tl.load(ptr + offsets[:, None] + cols[None, :], mask=mask[:, None], other=0)
    scales = tl.load((ptr + offsets + D).to(tl.pointer_type(tl.float32)), mask=mask, other=0)
    return (codes.to(tl.float32) * scales[:, None]).to(dtype)


@triton.jit
def _rel_attention_kernel(
    Q,
    K,
    V,
    Rel,
    BlockTable,
    SeqLens,
    CuSeqlensQ,
    Out,
    Lse,
    Gate,
    Sink,
    softmax_scale,
    num_tokens,
    num_splits,
    stride_q_t,
    stride_q_h,
    stride_kv_blk,
    stride_kv_slot,
    stride_kv_h,
    stride_rel_t,
    stride_bt,
    stride_out_t,
    stride_out_h,
    stride_lse_h,
    stride_lse_t,
    stride_gate,
    stride_sink,
    H: tl.constexpr,
    G: tl.constexpr,
    D: tl.constexpr,
    E: tl.constexpr,
    PAGE: tl.constexpr,
    WINDOW: tl.constexpr,
    DRAFT: tl.constexpr,
    MODE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    TOKENS: tl.constexpr = BLOCK_M // G
    E_PAD: tl.constexpr = E + 256
    LOG2E: tl.constexpr = 1.4426950408889634
    m_tile = tl.program_id(0)
    kv_head = tl.program_id(1) // num_splits
    split = tl.program_id(1) % num_splits
    req = tl.program_id(2)
    q_start = tl.load(CuSeqlensQ + req)
    q_len = tl.load(CuSeqlensQ + req + 1) - q_start
    first_tok = m_tile * TOKENS
    if first_tok >= q_len:
        return
    seqlen_k = tl.load(SeqLens + req)
    past = seqlen_k - q_len

    rows = tl.arange(0, BLOCK_M)
    tok = first_tok + rows // G
    row_valid = (rows < TOKENS * G) & (tok < q_len)
    head = kv_head * G + rows % G
    token = (q_start + tok).to(tl.int64)
    if DRAFT:
        pos = tl.zeros([BLOCK_M], tl.int32) + seqlen_k - 1
        pos_lo = seqlen_k - 1
        pos_hi = seqlen_k - 1
    else:
        pos = past + tok
        pos_lo = past + first_tok
        pos_hi = past + tl.minimum(first_tok + TOKENS, q_len) - 1
    # Visible key blocks of the tile, split `split` taking every
    # `num_splits`-th of them.
    n_lo = 0
    if WINDOW > 0:
        n_lo = tl.maximum(pos_lo - (WINDOW - 1), 0) // BLOCK_N
    n_hi = tl.minimum(pos_hi + 1, seqlen_k)
    n_hi = (n_hi + BLOCK_N - 1) // BLOCK_N
    num_iters = tl.maximum(n_hi - n_lo - split + num_splits - 1, 0) // num_splits

    offs_d = tl.arange(0, D)
    q = tl.load(
        Q + token[:, None] * stride_q_t + head[:, None] * stride_q_h + offs_d[None, :],
        mask=row_valid[:, None],
        other=0.0,
    )
    if not DRAFT:
        tile_end = tl.minimum((tok // _SHEAR_TILE + 1) * _SHEAR_TILE, q_len)
        rel_base = (past + tile_end + 127) // 128 * 128 - E_PAD
        rel_row = Rel + token * stride_rel_t + head * E_PAD - rel_base

    qk_scale = softmax_scale * LOG2E
    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, D], tl.float32)
    block_table = BlockTable + req.to(tl.int64) * stride_bt
    kv_off = kv_head * stride_kv_h
    # sm_80 (Ampere): key blocks wholly at distance >= E from every row of the tile (full layers, long prompts:
    # nearly all of them) are fully visible and get no relative bias, so they skip the distance, mask and bias work.
    # Same values as the general loop below (its bias is 0 and its mask all-true there).
    num_far = 0
    if WINDOW == 0 and not DRAFT:
        far_hi = (pos_lo - E + 1) // BLOCK_N
        num_far = tl.minimum(
            tl.maximum(far_hi - n_lo - split + num_splits - 1, 0) // num_splits, num_iters
        )
    for i in range(num_far):
        n = n_lo + split + i * num_splits
        keys = n * BLOCK_N + tl.arange(0, BLOCK_N)
        page = tl.load(block_table + keys // PAGE)
        kv_ptrs = (
            page.to(tl.int64) * stride_kv_blk + (keys % PAGE) * stride_kv_slot + kv_off
        )
        if K.dtype.element_ty == tl.int8:
            k = _load_int8_kv(K, kv_ptrs, keys >= 0, q.dtype, D).T
        elif K.dtype.element_ty.primitive_bitwidth == 8:
            k = _load_fp8_kv(K, kv_ptrs, keys >= 0, q.dtype, D).T
        else:
            k = tl.load(K + kv_ptrs[None, :] + offs_d[:, None]).to(q.dtype)
        s = tl.dot(q, k) * qk_scale
        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        if V.dtype.element_ty == tl.int8:
            v = _load_int8_kv(V, kv_ptrs, keys >= 0, q.dtype, D)
        elif V.dtype.element_ty.primitive_bitwidth == 8:
            v = _load_fp8_kv(V, kv_ptrs, keys >= 0, q.dtype, D)
        else:
            v = tl.load(V + kv_ptrs[:, None] + offs_d[None, :]).to(q.dtype)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new
    for i in range(num_far, num_iters):
        n = n_lo + split + i * num_splits
        keys = n * BLOCK_N + tl.arange(0, BLOCK_N)
        key_valid = keys < seqlen_k
        page = tl.load(block_table + keys // PAGE, mask=key_valid, other=0)
        kv_ptrs = (
            page.to(tl.int64) * stride_kv_blk + (keys % PAGE) * stride_kv_slot + kv_off
        )
        if K.dtype.element_ty == tl.int8:
            k = _load_int8_kv(K, kv_ptrs, key_valid, q.dtype, D).T
        elif K.dtype.element_ty.primitive_bitwidth == 8:
            k = _load_fp8_kv(K, kv_ptrs, key_valid, q.dtype, D).T
        else:
            k = tl.load(
                K + kv_ptrs[None, :] + offs_d[:, None],
                mask=key_valid[None, :],
                other=0.0,
            ).to(q.dtype)
        s = tl.dot(q, k) * qk_scale
        dist = pos[:, None] - keys[None, :]
        visible = (dist >= 0) & key_valid[None, :]
        if WINDOW > 0:
            visible &= dist < WINDOW
        if not DRAFT:
            in_extent = visible & (dist < E) & row_valid[:, None]
            bias = tl.load(rel_row[:, None] + keys[None, :], mask=in_extent, other=0.0)
            s += bias.to(tl.float32) * LOG2E
        s = tl.where(visible, s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp2(m_i - m_safe)
        p = tl.exp2(s - m_safe[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        if V.dtype.element_ty == tl.int8:
            v = _load_int8_kv(V, kv_ptrs, key_valid, q.dtype, D)
        elif V.dtype.element_ty.primitive_bitwidth == 8:
            v = _load_fp8_kv(V, kv_ptrs, key_valid, q.dtype, D)
        else:
            v = tl.load(
                V + kv_ptrs[:, None] + offs_d[None, :],
                mask=key_valid[:, None],
                other=0.0,
            ).to(q.dtype)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    empty = l_i == 0.0
    o = acc / tl.where(empty, 1.0, l_i)[:, None]
    lse2 = tl.where(empty, float("-inf"), m_i + tl.log2(tl.where(empty, 1.0, l_i)))
    if MODE == 0:
        tl.store(
            Out
            + token[:, None] * stride_out_t
            + head[:, None] * stride_out_h
            + offs_d[None, :],
            o,
            mask=row_valid[:, None],
        )
        tl.store(
            Lse + head * stride_lse_h + token * stride_lse_t,
            lse2 * 0.6931471805599453,
            mask=row_valid,
        )
    elif MODE == 1:
        cols = head[:, None] * D + offs_d[None, :]
        if not DRAFT:
            gate = tl.load(
                Gate + token[:, None] * stride_gate + cols,
                mask=row_valid[:, None],
                other=0.0,
            )
            sink = tl.load(Sink + token * stride_sink + head, mask=row_valid, other=0.0)
            o = o * _sigmoid(gate.to(tl.float32))
            o = o * _sigmoid(lse2 * 0.6931471805599453 - sink.to(tl.float32))[:, None]
        tl.store(
            Out + token[:, None] * stride_out_t + cols,
            o.to(Out.dtype.element_ty),
            mask=row_valid[:, None],
        )
    else:
        part = (split * num_tokens + token) * H + head
        tl.store(Lse + part, lse2, mask=row_valid)
        tl.store(Out + part[:, None] * D + offs_d[None, :], o, mask=row_valid[:, None])


def _block_m(max_query_len: int, group: int) -> int:
    rows = max_query_len * group
    return 16 if rows <= 16 else 32 if rows <= 32 else 64 if rows <= 64 else 128


def rel_attention(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor | None,
    rel_logits: torch.Tensor | None,
    gate: torch.Tensor | None = None,
    sinks: torch.Tensor | None = None,
    *,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    max_query_len: int,
    softmax_scale: float,
    window: int,
    num_splits: int,
    mode: int,
    prefill_tile: tuple[int, int, int] | None = None,
    decode_tile: tuple[int, int, int] | None = None,
) -> None:
    """Launch `_rel_attention_kernel` over the requests of `cu_seqlens_q`.

    `rel_logits` (sheared, `(T + 128, H, extent + 256)`) is None for the
    DSpark draft: plain block attention with all queries of a request at
    its last position. `window` (keys up to a query's position, inclusive) is
    0 for full layers. `mode` selects the output: `MODE_OUT_LSE` writes the
    fp32 `out` `(T, H, D)` and natural-log `lse` `(H, T)`; `MODE_GATED`
    the bf16 `(T, H * D)` `out` (gated by `gate` and `sinks` unless
    drafting; a single split); `MODE_PARTIAL` the fp32 `out`
    `(S, T, H, D)` and log2 `lse` `(S, T, H)` partials.
    """
    num_tokens, num_heads, head_dim = q.shape
    page, num_kv_heads = key_cache.shape[1:3]
    group = num_heads // num_kv_heads
    assert key_cache.dtype == value_cache.dtype
    if key_cache.dtype == torch.int8:
        _check_int8_cache(key_cache, head_dim)
        _check_int8_cache(value_cache, head_dim)
        assert window == 0 and rel_logits is not None
    assert key_cache.stride() == value_cache.stride() and key_cache.stride(3) == 1
    assert q.stride(2) == 1
    assert mode != MODE_GATED or num_splits == 1
    draft = rel_logits is None
    extent = 0
    if rel_logits is not None:
        extent = rel_logits.shape[2] - 256
        assert rel_logits.stride(1) == rel_logits.shape[2]
        assert rel_logits.stride(2) == 1
    if mode == MODE_GATED and not draft:
        assert gate is not None and sinks is not None
        assert gate.stride(1) == sinks.stride(1) == 1
    # Unused pointers keep the launches typed alike.
    rel = q if rel_logits is None else rel_logits
    gate = q if gate is None else gate
    sinks = q if sinks is None else sinks
    if lse is None:
        lse = out
    block_m = _block_m(max_query_len, group)
    tokens_per_tile = block_m // group
    num_reqs = cache_seqlens.shape[0]
    grid = (
        triton.cdiv(max_query_len, tokens_per_tile),
        num_kv_heads * num_splits,
        num_reqs,
    )
    if mode == MODE_OUT_LSE:
        out_strides = (out.stride(0), out.stride(1), lse.stride(0), lse.stride(1))
    elif mode == MODE_GATED:
        assert out.is_contiguous()
        out_strides = (out.stride(0), 0, 0, 0)
    else:
        assert out.is_contiguous() and lse.is_contiguous()
        out_strides = (0, 0, 0, 0)
    block_n = (64 if window else 32) if block_m >= 128 else 128 if block_m <= 32 else 64
    num_warps = 8 if window and block_m >= 128 else 4
    num_stages = 3 if window and block_m >= 128 else 2
    full_int8_prefill = (
        key_cache.dtype == torch.int8 and not window and block_m == 128
        and mode == MODE_OUT_LSE and num_splits == 1
    )
    if decode_tile is not None:
        # Explicit benchmark override; serving still uses the established defaults.
        assert prefill_tile is None and key_cache.dtype == torch.int8 and not window
        assert mode in (MODE_GATED, MODE_PARTIAL) and block_m <= 32
        block_n, num_warps, num_stages = decode_tile
    elif prefill_tile is not None:
        assert full_int8_prefill
        block_n, num_warps, num_stages = prefill_tile
    elif (
        _TUNE_SM86_INT8_VERIFY and key_cache.dtype == torch.int8 and not window
        and block_m <= 32 and mode in (MODE_GATED, MODE_PARTIAL)
        and torch.cuda.get_device_capability(q.device) == (8, 6)
    ):
        # SM86 verify sweep: one pipeline stage won for every tested shape.
        block_n, num_warps, num_stages = (128, 4, 1)
    elif (
        _TUNE_SM86_INT8_PREFILL and full_int8_prefill
        and torch.cuda.get_device_capability(q.device) == (8, 6)
    ):
        # SM86 sweep: larger key tiles amortize INT8 unpacking and softmax.
        block_n, num_warps, num_stages = (128, 8, 2)
    _rel_attention_kernel[grid](
        q,
        key_cache,
        value_cache,
        rel,
        block_table,
        cache_seqlens,
        cu_seqlens_q,
        out,
        lse,
        gate,
        sinks,
        softmax_scale,
        num_tokens,
        num_splits,
        q.stride(0),
        q.stride(1),
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        rel.stride(0),
        block_table.stride(0),
        *out_strides,
        gate.stride(0),
        sinks.stride(0),
        H=num_heads,
        G=group,
        D=head_dim,
        E=extent,
        PAGE=page,
        WINDOW=window,
        DRAFT=draft,
        MODE=mode,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=num_warps,
        num_stages=num_stages,
    )
