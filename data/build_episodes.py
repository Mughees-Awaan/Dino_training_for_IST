#!/usr/bin/env python3
"""
build_episodes.py -- STEP 7: BUILD THE PRACTICE SESSIONS
========================================================

WHAT AN "EPISODE" IS, AND WHY THE PRODUCT NEEDS THEM
    Remember what this tool actually does for a user. They open a big field image, click on
    five or ten examples of a plant, and the tool finds the rest of them. No training, no
    waiting -- clicks in, results out.

    So the model must be good at ONE specific skill: "here are a few examples of something you
    have never seen named before. Go find more of it." That skill is what we train and measure,
    and an EPISODE is one exercise in it:

        support  = the clicks.  A handful of marked examples of one plant.
        query    = the targets. More of that same plant, which the model must find WITHOUT
                   having been shown them.

    36,000 of these are built, and the same episodes get replayed across every experiment --
    which is what makes two experiments a fair comparison rather than two different tests.

THIS SCRIPT IS WHERE LEAKAGE IS EITHER PREVENTED OR CREATED
    Two rules. Both are ENFORCED here, then VERIFIED at the end of the run -- not trusted.

    RULE 1: support and query must come from DIFFERENT photographs.

        This is the subtle one. Why not just take clicks and targets from the same photograph,
        far apart from each other?

        Because the model (a Vision Transformer) reads an entire image AT ONCE. Every part of
        the picture can see every other part. So a target sitting in the same photograph as its
        clicks can be found by CONTEXT -- "the thing that looks like the other things in this
        image" -- rather than by recognising the plant. Distance inside one photograph does not
        help; the model is not looking locally.

        An episode like that teaches nothing and inflates every score built on it.

    RULE 2: every photograph in an episode comes from the same split.

        An episode with its clicks in the practice pile and its targets in the exam pile is
        leakage by definition -- the exam photograph has just been used for practice.

WHAT WE RECORD RATHER THAN FORBID
    Whether support and query come from the same FAMILY is written into a `same_family` column
    instead of being banned.

    Why: in the real product, the user clicks and searches within ONE field. Same-family
    episodes are exactly the realistic case. But they are also EASIER -- same lighting, same
    soil, same camera, same day. So we keep them, tag them, and evaluation reports them as a
    separate slice. That way the easy case cannot silently flatter the average.

    python -m training.data.build_episodes --out tables/episodes
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from collections import defaultdict

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from training.data import hashing, schema  # noqa: E402


def build_for_split(man, inst, split, n_episodes, k_support, k_query, rng, dedup=True):
    """Build `n_episodes` episodes using only photographs from one split."""
    sub = man[man["split"] == split]

    # ---- COLLAPSE BYTE-IDENTICAL COPIES ---------------------------------------------------
    # This is a DIFFERENT problem from leakage, and it needs a different fix.
    #
    # Splitting by family already guarantees copies never cross into the exam pile. Good. But
    # within the practice pile, if one photograph exists 40 times, random sampling picks it 40
    # times as often as a unique photograph -- while carrying exactly the same information.
    # The model gets a distorted view of what the world looks like.
    #
    # Measured: 82,099 photographs contain only 45,953 distinct images. The largest group is
    # 40 identical copies.
    #
    # drop_duplicates(subset="content_key") keeps one representative per identical group.
    # Nothing is deleted from disk -- this only affects sampling.
    if dedup:
        before = len(sub)
        # Group on the real content hash, not the old CRC32+size key.
        sub = sub.assign(_dedup=hashing.dedup_key(sub)).drop_duplicates(
            subset="_dedup", keep="first").drop(columns="_dedup")
        if before != len(sub):
            print(f"  {split}: {before:,} -> {len(sub):,} photographs after collapsing "
                  f"byte-identical copies")
    if sub.empty:
        return []

    uid_family = dict(zip(sub["image_uid"], sub["leakage_group_id"]))
    keep = inst[inst["image_uid"].isin(set(sub["image_uid"]))]

    # Use the approved names if step 4 has been done, otherwise fall back to the raw ones, so
    # this script still runs while the vocabulary review is outstanding.
    label_col = "label_canon" if (keep["label_canon"].fillna("") != "").any() else "label_raw"

    # ---- Build the index we sample from ----------------------------------------------------
    # Shape: {plant name: {photograph: [(x, y), (x, y), ...]}}
    #
    # defaultdict(lambda: defaultdict(list)) is a two-level auto-creating dictionary: ask for
    # any label and any photograph and you get an empty list rather than a KeyError. Without
    # it, every insertion needs two existence checks.
    by_label: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for uid, lab, x, y in zip(keep["image_uid"], keep[label_col], keep["x"], keep["y"]):
        if str(lab):
            by_label[str(lab)][uid].append((float(x), float(y)))

    episodes = []
    # A plant needs to appear in AT LEAST TWO photographs, otherwise we cannot obey Rule 1 --
    # there is nowhere to put the query.
    labels = [l for l, d in by_label.items() if len(d) >= 2]
    if not labels:
        return []

    for i in range(n_episodes):
        label = rng.choice(labels)                # pick a plant
        photos = list(by_label[label].keys())     # every photograph containing it
        rng.shuffle(photos)
        sup_uid, qry_uid = photos[0], photos[1]   # <-- RULE 1, structurally: two different
                                                  #     photographs, taken from a shuffled list

        sup_pts = by_label[label][sup_uid]
        qry_pts = by_label[label][qry_uid]
        if not sup_pts or not qry_pts:
            continue

        # rng.sample picks WITHOUT replacement, so the same plant is never clicked twice in one
        # episode. min(k, len(...)) handles a photograph with fewer marks than we asked for.
        sup = rng.sample(sup_pts, min(k_support, len(sup_pts)))
        qry = rng.sample(qry_pts, min(k_query, len(qry_pts)))

        episodes.append({
            "episode_id": f"{split}-{i:06d}",
            "split": split,
            "label": label,
            "support_uid": sup_uid,          # the photograph the user "clicks" in
            "query_uid": qry_uid,            # the photograph the model must search
            # Coordinates are stored as lists inside a single cell. Parquet handles nested
            # lists natively, so one episode stays one row and nothing has to be re-joined.
            "support_x": [p[0] for p in sup],
            "support_y": [p[1] for p in sup],
            "query_x": [p[0] for p in qry],
            "query_y": [p[1] for p in qry],
            "n_support": len(sup),
            "n_query": len(qry),
            # Recorded, not forbidden. Evaluation slices on this -- see the header.
            "same_family": int(uid_family.get(sup_uid) == uid_family.get(qry_uid)),
            "sampler_version": 1,            # bump this if the sampling logic ever changes,
                                             # so old and new episodes are distinguishable
            "seed": rng.randint(0, 2**31 - 1),
        })
    return episodes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="tables/manifest.parquet")
    ap.add_argument("--instances", default="tables/instances.parquet")
    ap.add_argument("--out", default="tables/episodes")
    ap.add_argument("--per-split", type=int, default=20000)
    ap.add_argument("--k-support", type=int, default=10, help="how many clicks per episode")
    ap.add_argument("--k-query", type=int, default=20, help="how many targets to find")
    ap.add_argument("--seed", type=int, default=20260817)
    ap.add_argument("--keep-duplicates", action="store_true",
                    help="sample from every copy. Off by default: see build_for_split")
    args = ap.parse_args()

    man = pd.read_parquet(args.manifest)
    inst = pd.read_parquet(args.instances)

    # Refuse to run without splits. Building episodes from an unsplit manifest would silently
    # produce episodes that straddle the practice/exam divide -- exactly the thing this script
    # exists to prevent.
    if "split" not in man.columns or (man["split"].fillna("") == "").all():
        raise SystemExit("manifest has no split column; run build_splits --write-manifest first")

    os.makedirs(args.out, exist_ok=True)
    rng = random.Random(args.seed)
    total = 0
    print(f"{'split':<13}{'episodes':>10}{'labels':>9}{'same-family':>13}")
    for split in schema.SPLITS:
        # The training pile gets the full count; the evaluation piles need far fewer, since
        # they only have to measure, not teach.
        n = args.per_split if split == "train" else max(2000, args.per_split // 5)
        eps = build_for_split(man, inst, split, n, args.k_support, args.k_query, rng,
                              dedup=not args.keep_duplicates)
        if not eps:
            print(f"{split:<13}{'0':>10}   (no label has two photographs in this split)")
            continue
        df = pd.DataFrame(eps)
        path = os.path.join(args.out, f"{split}.parquet")
        df.to_parquet(path, index=False)
        total += len(df)
        print(f"{split:<13}{len(df):>10,}{df['label'].nunique():>9}"
              f"{100 * df['same_family'].mean():>12.1f}%")

    # =====================================================================================
    # VERIFY THE GUARANTEE -- do not merely assume it
    # =====================================================================================
    # The sampling logic above SHOULD make both of these impossible. That is exactly why they
    # are worth checking: a future edit could break the guarantee without breaking anything
    # visible. These two lines are the difference between "we believe it is safe" and "we
    # checked, on this run, on this data".
    print("\nleakage check")
    bad_photo = bad_split = 0
    for split in schema.SPLITS:
        p = os.path.join(args.out, f"{split}.parquet")
        if not os.path.exists(p):
            continue
        df = pd.read_parquet(p)
        # Rule 1: clicks and targets must never be the same photograph.
        bad_photo += int((df["support_uid"] == df["query_uid"]).sum())
        # Rule 2: both photographs must belong to the split this file claims to be.
        sp = dict(zip(man["image_uid"], man["split"]))
        bad_split += int(sum(1 for a, b in zip(df["support_uid"], df["query_uid"])
                             if sp.get(a) != split or sp.get(b) != split))
    print(f"  support and query in the same photograph : {bad_photo}  "
          f"{'(correct)' if bad_photo == 0 else '(LEAKAGE)'}")
    print(f"  session straddling two splits            : {bad_split}  "
          f"{'(correct)' if bad_split == 0 else '(LEAKAGE)'}")
    print(f"\nwrote {total:,} sessions to {args.out}/")
    # Non-zero exit if either check failed, so a pipeline stops here.
    return 1 if (bad_photo or bad_split) else 0


if __name__ == "__main__":
    raise SystemExit(main())
