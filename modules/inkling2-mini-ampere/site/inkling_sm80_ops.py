"""Stand-ins for compiled vLLM ops that vllm/vllm-openai:inkling-2-mini ships without sm_80 kernels.

The image's _C / _C_cache_ops / _moe_C libraries carry sm_100 code for nearly everything (on sm_80 only Marlin and
some MoE kernels exist), so such an op launches nothing on sm_80 and leaves cudaErrorNoKernelImageForDevice behind.
The image's vLLM commit is not public, so the libraries cannot be rebuilt; instead each op the Inkling 2 Mini path
reaches is re-registered here for the CUDA dispatch key, so every caller (eager, torch.compile, CUDA graphs) gets it.
Found with INKLING_SM80_TRACE=1 (sitecustomize): on start-up and decode only reshape_and_cache_flash (the DSpark
drafter's KV write) was hit; moe_align_block_size is replaced in the patched moe_sm90.py instead.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _reshape_and_cache_flash_kernel(K, V, KC, VC, S, k_t, k_h, v_t, v_h, kc_b, kc_p, kc_h, vc_b, vc_p, vc_h,
                                    BLOCK_SIZE: tl.constexpr, D: tl.constexpr, DP: tl.constexpr):
    t = tl.program_id(0)
    h = tl.program_id(1)
    slot = tl.load(S + t).to(tl.int64)
    b = slot // BLOCK_SIZE
    p = slot % BLOCK_SIZE
    d = tl.arange(0, DP)
    m = (d < D) & (slot >= 0)
    k = tl.load(K + t * k_t + h * k_h + d, mask=m)
    v = tl.load(V + t * v_t + h * v_h + d, mask=m)
    tl.store(KC + b * kc_b + p * kc_p + h * kc_h + d, k, mask=m)
    tl.store(VC + b * vc_b + p * vc_p + h * vc_h + d, v, mask=m)


def reshape_and_cache_flash(key, value, key_cache, value_cache, slot_mapping, kv_cache_dtype, k_scale, v_scale):
    # key/value [T, H, D] (T may be padded past slot_mapping), caches [blocks, block_size, H, D] (strided views).
    if kv_cache_dtype not in ("auto", "bfloat16", "float16") or key_cache.dtype != key.dtype:
        raise NotImplementedError(f"sm_80 reshape_and_cache_flash: kv cache dtype {kv_cache_dtype} / {key_cache.dtype}")
    n = slot_mapping.shape[0]
    if n == 0:
        return
    heads, dim = key.shape[1], key.shape[2]
    assert key.stride(2) == 1 and value.stride(2) == 1 and key_cache.stride(3) == 1 and value_cache.stride(3) == 1
    _reshape_and_cache_flash_kernel[(n, heads)](
        key, value, key_cache, value_cache, slot_mapping,
        key.stride(0), key.stride(1), value.stride(0), value.stride(1),
        key_cache.stride(0), key_cache.stride(1), key_cache.stride(2),
        value_cache.stride(0), value_cache.stride(1), value_cache.stride(2),
        BLOCK_SIZE=key_cache.shape[1], D=dim, DP=triton.next_power_of_2(dim), num_warps=1)


_LIBS = []


def install():
    lib = torch.library.Library("_C_cache_ops", "IMPL")
    lib.impl("reshape_and_cache_flash", reshape_and_cache_flash, "CUDA")
    _LIBS.append(lib)  # keep the registration alive
    print("[sm80] reshape_and_cache_flash -> Triton stand-in", flush=True)
