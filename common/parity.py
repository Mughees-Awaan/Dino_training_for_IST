#!/usr/bin/env python3
"""
parity.py -- prove the exported model matches the one that was trained.

THE PROBLEM
    A model is trained in PyTorch, exported to ONNX, then quantised to INT8 for the CPU
    laptop. Each conversion can change the numbers. If it changes them enough, the shipped
    product is not the model that was measured -- and nothing in the export pipeline says so.

WHY COSINE SIMILARITY IS NOT ENOUGH ON ITS OWN
    Cosine compares the DIRECTION of two descriptor vectors. It is the right first check,
    but a model can score cos 0.99 and still detect noticeably worse, because detection
    depends on fine distinctions that a 1% angular difference can erase.

    So parity is TWO gates: descriptors agree, AND detection F1 is re-run and agrees.
    The second is the one that matters; the first tells you where it broke.

MEASURED ON THIS PROJECT
    Dynamic INT8 quantisation scored cosine 0.56 on this codebase -- catastrophic, and it
    would have shipped. Static QDQ quantisation is the correct method here.
"""

from __future__ import annotations

import numpy as np


def cosine_agreement(a, b) -> dict:
    """How closely do two sets of descriptors point the same way?

    Compares row by row. Returns the mean and, more importantly, the WORST row -- an
    average can hide a handful of catastrophically wrong descriptors.
    """
    a = np.asarray(a, np.float64).reshape(len(a), -1)
    b = np.asarray(b, np.float64).reshape(len(b), -1)
    an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    cos = (an * bn).sum(1)
    return {"mean_cos": float(cos.mean()), "min_cos": float(cos.min()),
            "frac_below_0.99": float((cos < 0.99).mean())}


def assert_parity(a, b, name: str, min_mean: float = 0.99, min_worst: float = 0.95):
    """Fail the export rather than shipping a model that is not the one measured."""
    r = cosine_agreement(a, b)
    if r["mean_cos"] < min_mean or r["min_cos"] < min_worst:
        raise RuntimeError(
            f"{name} parity FAILED: mean cos {r['mean_cos']:.4f} (need >= {min_mean}), "
            f"worst {r['min_cos']:.4f} (need >= {min_worst}), "
            f"{100*r['frac_below_0.99']:.1f}% of descriptors below 0.99.\n"
            f"Do not ship this export.")
    return r
