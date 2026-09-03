#!/usr/bin/env python3
"""
gpu_probe.py -- confirm the GPU is ACTUALLY being used.

THE BUG THIS EXISTS FOR
    onnxruntime was installed with CUDA 13 wheels against a CUDA 12.2 driver. It listed
    CUDAExecutionProvider as available, accepted it without complaint, and then SILENTLY
    RAN EVERYTHING ON THE CPU.

    No error. No warning. Every "GPU" timing recorded for days was a CPU timing.

    The fix was torch cu126 + onnxruntime-gpu 1.22.0. The lesson is that "the provider is
    listed" and "the provider is being used" are different claims, and only the second one
    matters. Measured difference once it was real: 16.5x (0.094 -> 1.552 Mpx/s).
"""

from __future__ import annotations

import time

from training.common.safe_imports import HAVE_TORCH, torch


def probe_torch() -> dict:
    """Is torch on the GPU, and does it actually compute there?"""
    if not HAVE_TORCH:
        return {"available": False, "reason": "torch not installed"}
    info = {"available": torch.cuda.is_available(), "torch": torch.__version__,
            "built_cuda": torch.version.cuda}
    if not info["available"]:
        return info
    info["device"] = torch.cuda.get_device_name(0)
    info["capability"] = ".".join(map(str, torch.cuda.get_device_capability(0)))

    # Do real work and time it. Listing a device proves nothing; running on it does.
    a = torch.randn(2048, 2048, device="cuda")
    torch.cuda.synchronize()          # CUDA is asynchronous -- without this you time the
    t = time.time()                   # QUEUEING of the work, not the work itself.
    for _ in range(10):
        a = a @ a.T
        a = a / a.norm()
    torch.cuda.synchronize()
    dt = time.time() - t
    info["matmul_2048_x10_s"] = round(dt, 4)
    # A 3090 does this in well under a second. Anything near a second means we are on CPU
    # in all but name.
    info["plausibly_gpu"] = dt < 1.0
    return info


def probe_onnxruntime() -> dict:
    """Is onnxruntime's CUDA provider AVAILABLE, and is it actually SELECTED?

    These are two different questions -- conflating them is the bug above.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        return {"available": False, "reason": "onnxruntime not installed"}
    avail = ort.get_available_providers()
    return {"version": ort.__version__, "providers": avail,
            "cuda_listed": "CUDAExecutionProvider" in avail}


def assert_gpu(what: str = "this run"):
    """Fail loudly BEFORE a long job rather than producing CPU timings labelled GPU."""
    t = probe_torch()
    if not t.get("available"):
        raise RuntimeError(f"{what} expects a GPU; torch reports none available.")
    if not t.get("plausibly_gpu", False):
        raise RuntimeError(
            f"{what}: CUDA is reported available but a 2048x2048 matmul x10 took "
            f"{t['matmul_2048_x10_s']}s, which is CPU-like. This is the silent-fallback "
            f"failure mode -- check the CUDA wheel matches the driver.")
    return t


if __name__ == "__main__":
    for k, v in probe_torch().items():
        print(f"  torch  {k:<22} {v}")
    for k, v in probe_onnxruntime().items():
        print(f"  ort    {k:<22} {v}")
