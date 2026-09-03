#!/usr/bin/env python3
"""
Tests for the CPU student model.

    python training/tests/test_student_model.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from training.common.safe_imports import nn, torch  # noqa: E402
from training.student.model import StudentBackbone  # noqa: E402

_M = None


def _model():
    global _M
    if _M is None:
        _M = StudentBackbone(out_dim=384, pretrained=False).eval()
    return _M


def test_output_strides_are_exactly_4_8_16():
    """A stride that is off by a factor of two silently misaligns every detection."""
    out = _model()(torch.randn(1, 3, 512, 512))
    assert out["desc_s16"].shape[-1] == 512 // 16
    assert out["desc_s8"].shape[-1] == 512 // 8
    assert out["centre"].shape[-1] == 512 // 4


def test_descriptors_are_unit_length():
    """The ridge adapter compares by DIRECTION, so the model must emit normalised vectors."""
    out = _model()(torch.randn(1, 3, 256, 256))
    n = out["desc_s16"].norm(dim=1)
    assert torch.allclose(n, torch.ones_like(n), atol=1e-4)


def test_batchnorm_stays_frozen_even_after_train():
    """The product runs ONE tile at a time, so BatchNorm batch statistics would be wrong.

    Calling .train() must NOT silently un-freeze them.
    """
    m = _model()
    m.train()
    bns = [x for x in m.modules() if isinstance(x, nn.BatchNorm2d)]
    assert bns, "expected BatchNorm layers inherited from MobileNet"
    assert all(not b.training for b in bns), "BatchNorm un-froze on .train()"
    m.eval()


def test_batch_size_does_not_change_the_answer():
    """Consequence of the above. Batch 1 and batch 4 must give identical descriptors."""
    m = _model()
    x = torch.randn(4, 3, 128, 128)
    with torch.no_grad():
        one = torch.cat([m(x[i:i+1])["desc_s16"] for i in range(4)])
        four = m(x)["desc_s16"]
    assert torch.allclose(one, four, atol=1e-5), "output depends on batch size"


def test_small_enough_for_a_cpu_laptop():
    n = sum(p.numel() for p in _model().parameters())
    assert n < 5e6, f"{n/1e6:.1f}M parameters is too large for the CPU target"


if __name__ == "__main__":
    for fn in (test_output_strides_are_exactly_4_8_16, test_descriptors_are_unit_length,
               test_batchnorm_stays_frozen_even_after_train,
               test_batch_size_does_not_change_the_answer,
               test_small_enough_for_a_cpu_laptop):
        fn(); print(f"  PASS  {fn.__name__}")
    print("student model tests pass")
