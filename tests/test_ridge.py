#!/usr/bin/env python3
"""
Tests for the click-conditioned ridge adapter -- the heart of the product.

WHY THESE FIVE
    The ridge solve is the one piece that runs on every single user click, and the one the
    programme plan flagged as "never prototyped". Each test below guards a way it could be
    silently wrong rather than loudly broken.

    python training/tests/test_ridge.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from training.common.ridge import (add_bias, fit_predict, l2_normalise,  # noqa: E402
                                   solve_ridge, stability_report)
from training.common.safe_imports import torch  # noqa: E402


def _separable(d=32, k=8, n=40, seed=0):
    """A toy problem with a genuine answer: two clouds either side of a random direction."""
    g = torch.Generator().manual_seed(seed)
    proto = torch.randn(d, generator=g)
    sup = torch.cat([proto + 0.15 * torch.randn(k, d, generator=g),
                     -proto + 0.15 * torch.randn(k, d, generator=g)])
    lab = torch.tensor([1.0] * k + [-1.0] * k)
    qry = torch.cat([proto + 0.15 * torch.randn(n, d, generator=g),
                     -proto + 0.15 * torch.randn(n, d, generator=g)])
    qlab = torch.tensor([1.0] * n + [-1.0] * n)
    return sup, lab, qry, qlab


def test_separates_a_solvable_problem():
    """The floor: if it cannot solve an obviously separable problem, nothing else matters."""
    sup, lab, qry, qlab = _separable()
    acc = ((fit_predict(sup, lab, qry) > 0).float() * 2 - 1).eq(qlab).float().mean()
    assert acc > 0.95, f"only {acc:.2%} correct on a linearly separable problem"


def test_normalisation_and_bias():
    """L2-normalise must give unit length; add_bias must append exactly one constant 1."""
    x = torch.randn(5, 16) * 100          # deliberately huge, to catch a missing normalise
    assert torch.allclose(l2_normalise(x).norm(dim=-1), torch.ones(5), atol=1e-5)
    b = add_bias(x)
    assert b.shape == (5, 17) and torch.allclose(b[:, -1], torch.ones(5))


def test_gradient_reaches_the_descriptors():
    """The whole point of T2: query loss must flow BACK through the solve into the backbone.

    If this breaks, training runs and learns nothing -- with no error anywhere.
    """
    sup, lab, qry, qlab = _separable()
    sup = sup.clone().requires_grad_(True)
    fit_predict(sup, lab, qry).pow(2).mean().backward()
    assert sup.grad is not None, "no gradient path through the ridge solve"
    assert torch.isfinite(sup.grad).all(), "gradient contains NaN or Inf"
    assert sup.grad.abs().sum() > 0, "gradient is all zeros -- the solve is detached"


def test_backward_is_stable_across_click_counts():
    """The scaffold-stage EXIT CRITERION from the programme plan.

    Backward-through-Cholesky had never been prototyped. If any configuration produces a
    non-finite gradient, episodic training will diverge -- and it is far cheaper to learn
    that here than six hours into a GPU run.
    """
    bad = [r for r in stability_report() if not r["finite"]]
    assert not bad, f"unstable configurations: {bad}"


def test_ridge_refuses_a_bad_solve():
    """check=True must RAISE on non-finite descriptors rather than returning quiet garbage."""
    sup, lab, _, _ = _separable()
    sup = sup.clone()
    sup[0, 0] = float("nan")
    try:
        solve_ridge(sup, lab, check=True)
    except Exception:
        return                                     # correct: it refused
    raise AssertionError("solve_ridge accepted NaN descriptors without complaint")


if __name__ == "__main__":
    for fn in (test_separates_a_solvable_problem, test_normalisation_and_bias,
               test_gradient_reaches_the_descriptors,
               test_backward_is_stable_across_click_counts,
               test_ridge_refuses_a_bad_solve):
        fn(); print(f"  PASS  {fn.__name__}")
    print("ridge tests pass")
