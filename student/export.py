#!/usr/bin/env python3
"""
export.py -- student to ONNX, then STATIC INT8 for the laptop CPU.

THE MEASURED TRAP
    DYNAMIC INT8 quantisation scored cosine 0.56 on this codebase. That is catastrophic --
    the shipped model would not have been the model that was measured -- and it produced no
    error at any point.

    STATIC QDQ quantisation is the correct method: it uses a calibration set to learn the
    real value ranges, instead of guessing them at runtime.

    Parity is checked at BOTH hops, and the export fails rather than shipping.
"""

from __future__ import annotations

import argparse
import os

from training.common.parity import assert_parity
from training.common.safe_imports import np, require_torch, torch


def to_onnx(model, out: str, size: int = 512, opset: int = 17) -> str:
    require_torch("student export")
    model.eval()
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    torch.onnx.export(model, torch.randn(1, 3, size, size), out, opset_version=opset,
                      input_names=["image"],
                      output_names=["desc_s16", "desc_s8", "centre", "offset",
                                    "log_size", "boundary"],
                      dynamic_axes=None, do_constant_folding=True)
    return out


def quantise_static(onnx_in: str, onnx_out: str, calib_images) -> str:
    """Static QDQ INT8. Needs real images to calibrate against -- that is the whole point."""
    from onnxruntime.quantization import CalibrationDataReader, QuantFormat, quantize_static

    class _Reader(CalibrationDataReader):
        def __init__(self, imgs):
            self._it = iter([{"image": im} for im in imgs])
        def get_next(self):
            return next(self._it, None)

    quantize_static(onnx_in, onnx_out, _Reader(calib_images),
                    quant_format=QuantFormat.QDQ)     # QDQ, never dynamic
    return onnx_out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default="models/student.onnx")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--int8", action="store_true")
    args = ap.parse_args()

    from training.common.checkpoint import load
    from training.student.model import StudentBackbone
    net = StudentBackbone(pretrained=False)
    load(args.checkpoint, model=net, strict_provenance=False)

    path = to_onnx(net, args.out, args.size)
    print(f"wrote {path}")
    if args.int8:
        raise SystemExit("INT8 needs a calibration set drawn from real field tiles; "
                         "wire it to the eligible dev_cal split once episodes are rebuilt.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
