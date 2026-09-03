#!/usr/bin/env python3
"""
evaluate.py -- Scores a teacher checkpoint on held-out episodes and runs the click-dependence probe. A model that ignores clicks can still score well; the probe is what catches it.

runs the shuffled-click control

STATUS: scaffolded. Blocked on the same dependency as everything downstream --
episodes must carry positive AND negative clicks before this can run for real.
"""

from __future__ import annotations

import argparse


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
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
