# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Low-M BF16 tensor-core projections for SM120/SM121."""

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

MAX_M = 16
_BN, _BK = 32, 256


@triton.jit
def _gemv_kernel(
    X,
    W,
    B,
    Y,
    M,
    N,
    K,
    SX,
    SW,
    SY,
    HAS_BIAS: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    MP: tl.constexpr,
):
    pid = tl.program_id(0)
    cols = pid * BN + tl.arange(0, BN)
    rows = tl.arange(0, MP)
    acc = tl.zeros((MP, BN), tl.float32)
    for k0 in range(0, K, BK):
        ks = k0 + tl.arange(0, BK)
        w = tl.load(
            W + cols[None, :].to(tl.int64) * SW + ks[:, None],
            (cols[None, :] < N) & (ks[:, None] < K),
            other=0,
        )
        x = tl.load(
            X + rows[:, None] * SX + ks[None, :],
            (rows[:, None] < M) & (ks[None, :] < K),
            other=0,
        )
        acc += tl.dot(x, w)
    if HAS_BIAS:
        acc += tl.load(B + cols, cols < N, other=0).to(tl.float32)[None, :]
    tl.store(
        Y + rows[:, None] * SY + cols[None, :],
        acc.to(Y.dtype.element_ty),
        (rows[:, None] < M) & (cols[None, :] < N),
    )


# sm_80 (Ampere): the same kernel beats cuBLAS at 5-16 rows on the GA100 too (cuBLAS picks split-K tiles
# plus a reduce kernel there). Tiles measured per shape with the Inkling2 weights: (BN, BK, warps, stages).
# Measured with cold weights (cycled past the 40 MB L2, as in the model).
def _sm80_tiles(n: int, k: int) -> tuple[int, int, int, int]:
    if n >= 65536:  # vocab heads; the draft's Markov fold has K=256
        return (64, 64, 2, 3) if k <= 512 else (64, 256, 2, 4)
    if n <= 1024:
        return 16, 256, 2, 3
    if n >= 16384:
        return 64, 128, 8, 2
    if n >= 8192:
        return 64, 128, 4, 3
    if k >= 3072:
        return 32, 256, 4, 4
    return 32, 128, 4, 4


@triton.jit
def _seg3_gemv_kernel(X0, X1, X2, W, Y, M, K, SX0, SX1, SX2, SW, SY, N0, N1, N2, NB0, NB1,
                      BN: tl.constexpr, BK: tl.constexpr, MP: tl.constexpr):
    # Y[:, seg] = X_seg @ W[seg].T for three consecutive column segments (= weight rows) of widths N0, N1, N2.
    pid = tl.program_id(0)
    if pid < NB0:
        X = X0
        SX = SX0
        lo = 0
        hi = N0
        local = pid
    elif pid < NB0 + NB1:
        X = X1
        SX = SX1
        lo = N0
        hi = N0 + N1
        local = pid - NB0
    else:
        X = X2
        SX = SX2
        lo = N0 + N1
        hi = N0 + N1 + N2
        local = pid - NB0 - NB1
    cols = lo + local * BN + tl.arange(0, BN)
    cmask = cols < hi
    rows = tl.arange(0, MP)
    acc = tl.zeros((MP, BN), tl.float32)
    for k0 in range(0, K, BK):
        ks = k0 + tl.arange(0, BK)
        w = tl.load(W + cols[None, :].to(tl.int64) * SW + ks[:, None], cmask[None, :] & (ks[:, None] < K), other=0)
        x = tl.load(X + rows[:, None] * SX + ks[None, :], (rows[:, None] < M) & (ks[None, :] < K), other=0)
        acc += tl.dot(x, w)
    tl.store(Y + rows[:, None] * SY + cols[None, :], acc.to(Y.dtype.element_ty), (rows[:, None] < M) & cmask[None, :])


SEG3_TILES = (64, 128, 4, 3)


def sm80_seg3_gemv(xs: tuple[torch.Tensor, torch.Tensor, torch.Tensor], weight: torch.Tensor,
                   widths: tuple[int, int, int]) -> torch.Tensor | None:
    """sm_80 (Ampere): one launch for `cat([xs[i] @ weight[seg_i].T])` over three consecutive weight-row segments
    of `widths`, each with its own input (a full layer's or loop pass's q-side / k / v projections, three launches
    otherwise). Returns bf16 (M, sum(widths)), or None where it does not apply."""
    if not current_platform.is_device_capability_family(80) or weight.dtype != torch.bfloat16:
        return None
    m, k = xs[0].shape[0], weight.shape[1]
    for x in xs:
        if (x.ndim != 2 or x.dtype != torch.bfloat16 or x.shape != (m, k) or x.stride(-1) != 1
                or x.device != weight.device):
            return None
    if not 0 < m <= MAX_M or weight.stride(-1) != 1 or weight.shape[0] < sum(widths):
        return None
    n = sum(widths)
    out = torch.empty((m, n), device=weight.device, dtype=torch.bfloat16)
    bn, bk, warps, stages = SEG3_TILES
    nb = [triton.cdiv(w, bn) for w in widths]
    _seg3_gemv_kernel[(sum(nb),)](xs[0], xs[1], xs[2], weight, out, m, k, xs[0].stride(0), xs[1].stride(0),
        xs[2].stride(0), weight.stride(0), out.stride(0), *widths, nb[0], nb[1], BN=bn, BK=bk, MP=16,
        num_warps=warps, num_stages=stages)
    return out


@triton.jit
def _ctrl_down_kernel(X, W, P, M, N, K, SXS, SXM, SWS, SWN, K_PER: tl.constexpr,
                      BN: tl.constexpr, BK: tl.constexpr, MP: tl.constexpr):
    # fp32 partial [split, s, m, n] of X[s] @ W[s].T over one K slice.
    pid = tl.program_id(0)
    split = tl.program_id(1)
    s = tl.program_id(2)
    S = tl.num_programs(2)
    cols = pid * BN + tl.arange(0, BN)
    rows = tl.arange(0, MP)
    acc = tl.zeros((MP, BN), tl.float32)
    for k0 in range(split * K_PER, tl.minimum(split * K_PER + K_PER, K), BK):
        ks = k0 + tl.arange(0, BK)
        w = tl.load(W + s.to(tl.int64) * SWS + cols[None, :].to(tl.int64) * SWN + ks[:, None],
                    (cols[None, :] < N) & (ks[:, None] < K), other=0)
        x = tl.load(X + s.to(tl.int64) * SXS + rows[:, None] * SXM + ks[None, :],
                    (rows[:, None] < M) & (ks[None, :] < K), other=0)
        acc += tl.dot(x, w)
    tl.store(P + ((split * S + s) * MP + rows[:, None]) * N + cols[None, :], acc, cols[None, :] < N)


@triton.jit
def _ctrl_gf_kernel(P, W, Y, M, N, SWS, SWN, SYS, SYM, SPLITS: tl.constexpr, R: tl.constexpr,
                    BN: tl.constexpr, MP: tl.constexpr):
    # Y[s] = bf16(sum of the partials)[s] @ W[s].T in fp32: hidden rounded to bf16 as torch.bmm's bf16 output.
    pid = tl.program_id(0)
    s = tl.program_id(1)
    S = tl.num_programs(1)
    rows = tl.arange(0, MP)
    rk = tl.arange(0, R)
    h = tl.zeros((MP, R), tl.float32)
    for sp in tl.static_range(SPLITS):
        h += tl.load(P + ((sp * S + s) * MP + rows[:, None]) * R + rk[None, :])
    h = h.to(tl.bfloat16)
    cols = pid * BN + tl.arange(0, BN)
    w = tl.load(W + s.to(tl.int64) * SWS + cols[None, :].to(tl.int64) * SWN + rk[:, None], cols[None, :] < N, other=0)
    acc = tl.dot(h, w)
    tl.store(Y + s.to(tl.int64) * SYS + rows[:, None] * SYM + cols[None, :], acc, (rows[:, None] < M) & (cols[None, :] < N))


def sm80_controller_gain_forget(
    pair: torch.Tensor, down: torch.Tensor, gain_forget: torch.Tensor
) -> torch.Tensor | None:
    """sm_80 (Ampere): fp32 `bmm(bmm(pair, down.T), gain_forget.T)` of the loop message controllers
    (`Inkling2LoopDecoderLayer.blend`) for bf16 pair (S, M <= MAX_M, K), down (S, R, K), gain_forget (S, N, R), the
    hidden rounded to bf16 as there. cuBLAS gives the first product (N=64, K=4608) one block per stream: 18 us of
    25 for both at S=3; here the K slices run in parallel and the second kernel sums them (~10 us). None where it
    does not apply (the caller keeps torch.bmm)."""
    if (
        current_platform.has_device_capability(90)
        or pair.dtype != torch.bfloat16
        or down.dtype != torch.bfloat16
        or gain_forget.dtype != torch.bfloat16
        or pair.ndim != 3
        or not 0 < pair.shape[1] <= MAX_M
        or pair.stride(-1) != 1
        or down.stride(-1) != 1
        or gain_forget.stride(-1) != 1
        or down.shape[1] not in (16, 32, 64, 128)
        or gain_forget.shape[2] != down.shape[1]
        or pair.shape[2] != down.shape[2]
        or not pair.shape[0] == down.shape[0] == gain_forget.shape[0]
    ):
        return None
    splits, bn_gf, warps = (6, 128, 4) if pair.shape[0] > 1 else (8, 128, 4)
    s, m, k = pair.shape
    r = down.shape[1]
    n = gain_forget.shape[1]
    part = torch.empty((splits, s, 16, r), device=pair.device, dtype=torch.float32)
    k_per = triton.cdiv(triton.cdiv(k, splits), 64) * 64
    _ctrl_down_kernel[(triton.cdiv(r, 16), splits, s)](pair, down, part, m, r, k, pair.stride(0), pair.stride(1),
        down.stride(0), down.stride(1), K_PER=k_per, BN=16, BK=64, MP=16, num_warps=warps, num_stages=3)
    out = torch.empty((s, m, n), device=pair.device, dtype=torch.float32)
    _ctrl_gf_kernel[(triton.cdiv(n, bn_gf), s)](part, gain_forget, out, m, n, gain_forget.stride(0), gain_forget.stride(1),
        out.stride(0), out.stride(1), SPLITS=splits, R=r, BN=bn_gf, MP=16, num_warps=4)
    return out


def sm12x_bf16_gemv(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    lead = x.shape[:-1]
    x2 = x.reshape(-1, x.shape[-1])
    m, k = x2.shape
    assert m <= MAX_M
    n = weight.shape[0]
    if out is None:
        out = torch.empty((m, n), device=x.device, dtype=out_dtype or x.dtype)
    bn, bk, warps, stages = _BN, _BK, 4, 3
    if not current_platform.is_device_capability_family(120):
        bn, bk, warps, stages = _sm80_tiles(n, k)
    if m:
        _gemv_kernel[(triton.cdiv(n, bn),)](
            x2,
            weight,
            bias,
            out,
            m,
            n,
            k,
            x2.stride(0),
            weight.stride(0),
            out.stride(0),
            HAS_BIAS=bias is not None,
            BN=bn,
            BK=bk,
            MP=16,
            num_warps=warps,
            num_stages=stages,
        )
    return out.view(*lead, n)


def use_sm12x_bf16_gemv(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None
) -> bool:
    return (
        x.is_cuda
        and weight.device == x.device
        and x.dtype == weight.dtype == torch.bfloat16
        and x.ndim >= 2
        and weight.ndim == 2
        and x.shape[-1] == weight.shape[1]
        and x.stride(-1) == weight.stride(-1) == 1
        and 0 < x.numel() // x.shape[-1] <= MAX_M
        and (
            bias is None
            or (
                bias.dtype == torch.bfloat16
                and bias.device == x.device
                and bias.stride(0) == 1
            )
        )
        and (
            current_platform.is_device_capability_family(120)
            # sm_80 (Ampere): faster than cuBLAS here too (_sm80_tiles).
            or current_platform.is_device_capability_family(80)
        )
    )
