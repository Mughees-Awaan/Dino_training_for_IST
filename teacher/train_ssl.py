#!/usr/bin/env python3
"""
train_ssl.py -- T1: self-supervised continuation on agricultural imagery.

LABEL-FREE, which is why it can start NOW. It never looks at an annotation, so it does not
wait on the label review or the coverage audit that block everything else.

THE IDEA
    Show the model two different crops of the same photograph and ask it to produce the
    same answer for both. No labels needed -- the photograph is its own supervision.

THE FAILURE TO WATCH FOR: COLLAPSE
    The model can satisfy "both crops agree" by outputting the SAME THING FOR EVERY IMAGE.
    Loss goes to zero, training looks perfect, the model is worthless. Guards:
      * sharp teacher / soft student temperatures
      * a centring term
      * and the drift check below -- features should move 0.7-0.9 cosine from where they
        started. Near 1.0 means nothing was learned; near 0 means the model was destroyed.
"""

from __future__ import annotations

import argparse

from training.common.safe_imports import require_torch, torch


def feature_drift(before, after) -> float:
    """How far have the descriptors moved? The collapse alarm."""
    a = torch.nn.functional.normalize(before.reshape(len(before), -1), dim=-1)
    b = torch.nn.functional.normalize(after.reshape(len(after), -1), dim=-1)
    return float((a * b).sum(-1).mean())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="tables/manifest.parquet")
    ap.add_argument("--out", default="runs/t1")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--lora", action="store_true",
                    help="train a small adapter instead of the whole backbone. Cheaper and "
                         "far harder to collapse -- the plan's recommended first attempt.")
    ap.add_argument("--drift-min", type=float, default=0.70)
    ap.add_argument("--drift-max", type=float, default=0.90)
    args = ap.parse_args()

    require_torch("train_ssl")
    raise SystemExit(
        "T1 is scaffolded but not wired to a data loader yet.\n"
        "It is label-free so it does NOT block on the label review -- but it does need the\n"
        "corpus reader, and the current manifest is still pending the geometry rebuild.\n"
        "Gate when it runs: feature drift must land in "
        f"[{args.drift_min}, {args.drift_max}] -- outside that is collapse.")


if __name__ == "__main__":
    raise SystemExit(main())
