#!/usr/bin/env python3
"""
build_splits.py -- STEP 6: DIVIDE INTO PRACTICE AND EXAM SETS
=============================================================

THE FIVE PILES
    train        60%   the model learns from these
    dev_cal      10%   used to tune the sensitivity dial (how confident before we say "yes")
    dev_select   10%   used to CHOOSE between different trained versions
    dev_confirm  10%   used ONCE, to confirm a choice already made
    sealed_test  10%   opened ONCE, at the very end. Never touched before that.

WHY FIVE PILES AND NOT THE USUAL TWO?
    Because every time you LOOK at a set of data to make a decision, you use it up a little.
    Pick the best of twenty models on the same set, and the winner is partly just the one that
    happened to suit that particular set. Its score is optimistic and you cannot tell by how
    much.

    So each pile has exactly one job, and jobs are not shared:
      - tune the dial on dev_cal
      - pick the winner on dev_select
      - confirm the winner ONCE on dev_confirm
      - and sealed_test is untouched until the very end, so its number is the honest one.

    sealed_test is excluded from learning, from statistics, from caching, from threshold
    selection, and from label cleanup. If it is ever used for anything before the end, it stops
    being sealed and there is no way to un-see it.

THE ONE RULE THAT MATTERS
    ASSIGNMENT IS BY FAMILY, NEVER BY PHOTOGRAPH.

    Every photograph in a family goes into the same pile. This is the entire payoff of
    build_lineage_groups.py -- if we split photograph-by-photograph, two copies of the same
    field end up on opposite sides and every score afterwards is inflated.

THE PROBLEM WITH SPLITTING BY FAMILY
    Families are wildly different sizes -- from 1 photograph up to 1,876. Deal them out
    randomly and the shares come out lopsided: one unlucky draw puts three enormous families
    in `sealed_test` and it ends up holding 22% instead of 10%.

THE FIX -- LARGEST FIRST, INTO THE EMPTIEST PILE
    Sort the families biggest-first. Take each one in turn and put it in whichever pile is
    currently FURTHEST BELOW its target.

    This is the same intuition as packing a suitcase: put the big awkward items in first, while
    you still have room to place them well, and let the small ones fill the gaps at the end.
    Deal the big ones last and there is nowhere left to put them without spoiling the balance.

    Measured result on this corpus: exactly 60/10/10/10/10.

    python -m training.data.build_splits --seed 20260817 --out tables/splits.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from training.data import schema  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="tables/manifest.parquet")
    ap.add_argument("--lineage", default="tables/lineage_groups.parquet")
    ap.add_argument("--out", default="tables/splits.yaml")
    ap.add_argument("--seed", type=int, default=20260817,
                    help="fixes the tie-breaking, so the same seed always gives the same split")
    ap.add_argument("--write-manifest", action="store_true")
    args = ap.parse_args()

    man = pd.read_parquet(args.manifest)
    lin = pd.read_parquet(args.lineage)
    schema.validate(man, schema.MANIFEST_COLUMNS, "manifest")
    schema.validate(lin, schema.LINEAGE_COLUMNS, "lineage_groups")

    sizes = Counter(lin["leakage_group_id"])   # {family_name: how many photographs}
    print(f"{len(man):,} photographs in {len(sizes):,} families")

    # How many photographs each pile SHOULD end up with.
    targets = {s: schema.SPLIT_SHARES[s] * len(man) for s in schema.SPLITS}
    got = {s: 0 for s in schema.SPLITS}        # how many it has so far
    assign: dict[str, str] = {}                # {family_name: which pile}

    # A seeded random generator. "Seeded" means: given the same seed, it produces the same
    # sequence of numbers every time. So this whole script is REPRODUCIBLE -- run it next year
    # with the same seed and you get byte-identical splits, which is what makes two experiments
    # months apart genuinely comparable.
    rng = random.Random(args.seed)

    # Sort families largest-first. The `kv[0]` is a tie-breaker: two families of the same size
    # are ordered by NAME, so the order never depends on dictionary iteration order (which can
    # vary between runs and would quietly make the output non-reproducible).
    order = sorted(sizes.items(), key=lambda kv: (-kv[1], kv[0]))

    for gid, n in order:
        # Choose the pile that is furthest below its target.
        #   got[s] - targets[s]  is negative when a pile is under-filled; the MOST negative
        #   pile is the emptiest, and min() picks it.
        # The tiny random term (up to 0.000001) only breaks EXACT ties -- at the very start,
        # all five piles are equally empty, and without it the first five families would always
        # be dealt in the same fixed order.
        pick = min(schema.SPLITS, key=lambda s: (got[s] - targets[s] + rng.random() * 1e-6))
        assign[gid] = pick
        got[pick] += n

    # Translate family -> pile into photograph -> pile.
    lin = lin.assign(split=lin["leakage_group_id"].map(assign))
    per_photo = dict(zip(lin["image_uid"], lin["split"]))
    man = man.assign(split=man["image_uid"].map(per_photo).fillna(""))

    # ---- THE CHECK THAT JUSTIFIES THE WHOLE SCRIPT -----------------------------------------
    # For each family, count how many DISTINCT piles its photographs landed in. The answer must
    # be 1 for every single family. Anything else means leakage, and the script exits non-zero
    # rather than writing a split it does not believe in.
    bad = lin.groupby("leakage_group_id")["split"].nunique()
    leaked = int((bad > 1).sum())

    # ---- Write the result -------------------------------------------------------------------
    out = {"seed": args.seed, "photographs": int(len(man)), "families": len(sizes),
           "splits": {}, "families_by_split": defaultdict(list)}
    for s in schema.SPLITS:
        n = int((man["split"] == s).sum())
        out["splits"][s] = {"photographs": n, "share": round(n / max(len(man), 1), 4),
                            "target": schema.SPLIT_SHARES[s],
                            "families": int(sum(1 for g, v in assign.items() if v == s))}
    # The full family lists are written out too, so any later step can verify a photograph's
    # pile without re-running this script (and possibly getting a different answer).
    for g, s in assign.items():
        out["families_by_split"][s].append(g)
    out["families_by_split"] = dict(out["families_by_split"])

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        # We write JSON into a .yaml file on purpose: every JSON document is also valid YAML,
        # so this reads with either parser and needs no extra dependency.
        json.dump(out, fh, indent=1)

    print(f"{'split':<12}{'photographs':>13}{'share':>9}{'target':>9}{'families':>10}")
    for s in schema.SPLITS:
        d = out["splits"][s]
        print(f"{s:<12}{d['photographs']:>13,}{d['share']:>9.3f}{d['target']:>9.2f}{d['families']:>10,}")
    print(f"\nfamilies appearing in more than one split: {leaked}  "
          f"{'(correct)' if leaked == 0 else '(LEAKAGE)'}")
    print(f"wrote {args.out}")

    if leaked:
        return 1     # non-zero exit: do NOT let a leaking split be written into the manifest
    if args.write_manifest:
        man.to_parquet(args.manifest, index=False)
        print("manifest updated with split")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
