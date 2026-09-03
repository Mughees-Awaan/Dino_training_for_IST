#!/usr/bin/env python3
"""
export.py -- freeze a trained backbone into ONNX for the CPU product.

WHY THIS FILE EXISTS AT ALL
    The existing laptop/export_dinov3.py hardcodes `pretrained=True` and has no
    --checkpoint argument. It can ONLY export the stock web-trained model. It literally
    cannot export anything this project trains.

    That was a documented blocker for the entire freeze stage: you could train a better
    backbone and have no way to ship it. This file fixes exactly that.

TWO EXPORT TRAPS MEASURED ON THIS PROJECT
    1. DYNAMIC AXES + RoPE. Exporting with a dynamic input size makes RoPE positional
       encoding produce WRONG VALUES with CORRECT SHAPES. Nothing errors; the descriptors
       are simply wrong. Export at a FIXED size.

    2. DYNAMIC INT8 quantisation measured cosine 0.56 on this codebase -- catastrophic,
       and it would have shipped. Static QDQ is the correct method.

    python -m training.teacher.export --checkpoint runs/agridino/best.pt --out models/agridino.onnx
"""

from __future__ import annotations

import argparse
import os

from training.common.parity import assert_parity
from training.common.safe_imports import np, require_torch, torch


def export_onnx(model, out_path: str, size: int = 512, opset: int = 17) -> str:
    """Trace the model at a FIXED input size and write ONNX."""
    require_torch("export_onnx")
    model.eval()
    dummy = torch.randn(1, 3, size, size)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.onnx.export(
        model, dummy, out_path, opset_version=opset,
        input_names=["image"], output_names=["descriptors"],
        # NO dynamic_axes. See trap 1 above -- this is deliberate, not an oversight.
        dynamic_axes=None,
        do_constant_folding=True,
    )
    return out_path


def verify(model, onnx_path: str, size: int = 512, n: int = 4) -> dict:
    """Run the SAME inputs through PyTorch and ONNX and require they agree.

    An export that runs is not an export that is correct. This is the gate.
    """
    import onnxruntime as ort
    require_torch("verify")
    model.eval()
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    x = torch.randn(n, 3, size, size)
    with torch.no_grad():
        ref = model(x).cpu().numpy().reshape(n, -1)
    got = np.concatenate([sess.run(None, {"image": x[i:i+1].numpy()})[0].reshape(1, -1)
                          for i in range(n)], axis=0)
    return assert_parity(ref, got, "PyTorch->ONNX")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # THE ARGUMENT THE OLD SCRIPT LACKED.
    ap.add_argument("--checkpoint", default="",
                    help="trained weights to export. Omit ONLY to export the stock "
                         "pretrained model as a baseline.")
    ap.add_argument("--model", default="vit_small_patch16_224.dino")
    ap.add_argument("--out", default="models/agridino.onnx")
    ap.add_argument("--size", type=int, default=512, help="FIXED input size; see trap 1")
    ap.add_argument("--skip-verify", action="store_true")
    args = ap.parse_args()

    from training.teacher.dinov3 import DinoV3Backbone
    net = DinoV3Backbone(args.model, pretrained=not args.checkpoint, frozen=True)

    if args.checkpoint:
        from training.common.checkpoint import load
        load(args.checkpoint, model=net, strict_provenance=False)
        print(f"loaded trained weights from {args.checkpoint}")
    else:
        print("[warn] no --checkpoint given; exporting the STOCK pretrained model. "
              "This is a baseline, not a trained result.")

    path = export_onnx(net, args.out, args.size)
    print(f"wrote {path} (fixed {args.size}x{args.size})")

    if not args.skip_verify:
        r = verify(net, path, args.size)
        print(f"parity PASS  mean cos {r['mean_cos']:.5f}  worst {r['min_cos']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
