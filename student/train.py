#!/usr/bin/env python3
"""
train.py -- S-E (episodic+KD), S-B (baseline, no KD), S-U (KD from an UNTRAINED teacher). Without S-B and S-U the KD claim is unfalsifiable -- you cannot tell whether distillation helped or the architecture would have got there anyway.

--arm {S-E,S-B,S-U} and --kd-weight {0.2,0.5,1.0}

STATUS: scaffolded. Blocked on the same dependency as everything downstream --
episodes must carry positive AND negative clicks before this can run for real.
"""

from __future__ import annotations

import argparse


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", default="S-E", choices=["S-E","S-B","S-U"])
    ap.add_argument("--kd-weight", type=float, default=0.5, choices=[0.2,0.5,1.0])
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--episodes", default="tables/v2-staging/episodes/dev_select.parquet")
    ap.add_argument("--manifest", default="tables/manifest.parquet")
    ap.add_argument("--out", default="runs/eval.json")
    args = ap.parse_args()
    raise SystemExit(
        "Blocked: episodes carry positive clicks only. The click-conditioned adapter "
        "needs negatives to fit a decision boundary.\n"
        "Redesign build_episodes first (Required Order step 4).")


if __name__ == "__main__":
    raise SystemExit(main())
