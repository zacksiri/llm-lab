"""Pre-Hopper (sm_80, Ampere) shims for the vllm/vllm-openai:inkling-2-mini image.

Loaded through PYTHONPATH only when INKLING_SM80=1 (run.sh sets both). The Inkling2 Triton kernels call Triton's
grid-dependency-control helpers (griddepcontrol.wait / launch_dependents) and are launched with launch_pdl=True:
programmatic dependent launch, a Hopper feature. The PTX instructions do not exist before sm_90, so the kernels would
not compile. Triton documents both helpers as safe no-ops when dependent launch is off, so here they become an empty
instruction and every launch runs without dependent launch: each kernel then waits for the one before it on the
stream, which is what the helpers guarantee anyway. Nothing else is changed; no network, files or processes.
"""
import os

if os.environ.get("INKLING_SM80") == "1":
    try:
        import triton.language.extra.cuda as _tlc
        import triton.language.extra.cuda.gdc as _gdc
        from triton.language import core as _core

        @_core.extern
        def gdc_wait(_semantic=None):
            return _core.inline_asm_elementwise("mov.u32 $0, 0; // gdc_wait off (sm_80)", "=r", [],
                                                dtype=_core.int32, is_pure=False, pack=1, _semantic=_semantic)

        @_core.extern
        def gdc_launch_dependents(_semantic=None):
            return _core.inline_asm_elementwise("mov.u32 $0, 0; // gdc_launch_dependents off (sm_80)", "=r", [],
                                                dtype=_core.int32, is_pure=False, pack=1, _semantic=_semantic)

        for _m in (_tlc, _gdc):
            _m.gdc_wait = gdc_wait
            _m.gdc_launch_dependents = gdc_launch_dependents

        import triton.backends.nvidia.driver as _drv

        _orig_init = _drv.CudaLauncher.__init__

        def _init(self, src, metadata):
            _orig_init(self, src, metadata)
            self.launch_pdl = False

        _drv.CudaLauncher.__init__ = _init
    except Exception as e:  # never stop the server from starting; the log shows it
        print(f"[sitecustomize] INKLING_SM80 shims not applied: {e!r}", flush=True)

if os.environ.get("INKLING_SM80") == "1":
    # After vllm._custom_ops has loaded the compiled op libraries: register the Triton stand-ins for ops that have no
    # sm_80 kernel in this image (inkling_sm80_ops.py) and, with INKLING_SM80_TRACE=1 (debug), wrap every compiled op
    # in _C / _C_cache_ops / _moe_C so a call that leaves cudaErrorNoKernelImageForDevice (209) is logged once.
    import importlib.abc
    import importlib.util
    import sys

    def _install_trace():
        import ctypes
        import glob
        import torch

        rt = None
        for c in glob.glob("/usr/local/lib/python3.14/dist-packages/nvidia/cu13/lib/libcudart.so*"):
            try:
                rt = ctypes.CDLL(c)
                break
            except OSError:
                pass
        seen = set()
        names = [n for n in torch._C._dispatch_get_all_op_names() if n.split("::")[0] in ("_C", "_C_cache_ops", "_moe_C", "_C_custom_ar")]
        for full in names:
            ns, op = full.split("::", 1)
            op = op.split(".")[0]
            space = getattr(torch.ops, ns)
            try:
                packet = getattr(space, op)
            except Exception:
                continue
            if getattr(packet, "_sm80_traced", False):
                continue

            def make(packet, full):
                def check(name):
                    e = rt.cudaGetLastError()
                    if e and name not in seen:
                        seen.add(name)
                        print(f"[sm80-trace] {name} left CUDA error {e}", flush=True)

                class Traced:
                    _sm80_traced = True

                    def __init__(self, target, name):
                        self._t, self._n = target, name

                    def __call__(self, *a, **k):
                        out = self._t(*a, **k)
                        check(self._n)
                        return out

                    def __getattr__(self, attr):
                        v = getattr(self._t, attr)
                        if isinstance(v, torch._ops.OpOverload):
                            return Traced(v, f"{self._n}.{attr}")
                        return v

                return Traced(packet, full)

            setattr(space, op, make(packet, full))
        print(f"[sm80-trace] wrapped {len(names)} compiled ops", flush=True)

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name != "vllm._custom_ops":
                return None
            sys.meta_path.remove(self)
            spec = importlib.util.find_spec(name)
            orig = spec.loader.exec_module

            def exec_module(module):
                orig(module)
                try:
                    import inkling_sm80_ops

                    inkling_sm80_ops.install()
                except Exception as e:
                    print(f"[sm80] op stand-ins not installed: {e!r}", flush=True)
                if os.environ.get("INKLING_SM80_TRACE") == "1":
                    try:
                        _install_trace()
                    except Exception as e:
                        print(f"[sm80-trace] not installed: {e!r}", flush=True)

            spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _Finder())
