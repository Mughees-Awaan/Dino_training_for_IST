#!/usr/bin/env python3
"""
audit_duplicates.py -- STEP 2: FIND THE REPEATED PHOTOGRAPHS
============================================================

THE PROBLEM THIS SOLVES
    Imagine you are teaching a student, and you give them 100 practice questions and 10 exam
    questions. If three of the exam questions are copies of practice questions, the student
    scores well without having learned anything. You cannot see this in the score. The score
    just looks good.

    That is exactly what happens with duplicate photographs. Our corpus is full of them:
    36,146 of 82,099 photographs (44%) share their content with an earlier row. Those copies
    exist for ordinary reasons -- the same field was exported twice, one dataset was folded
    into another, a video was re-cut. But if a random split puts one copy in the practice pile
    and its twin in the exam pile, every number we report afterwards is inflated, and nothing
    in the numbers reveals it.

    So before we split anything, we must know what is a copy of what.

THE MOST IMPORTANT RULE HERE: NOTHING IS DELETED
    The obvious reaction to a duplicate is "delete it". We do not, for two reasons:

      1. Deleting destroys the evidence that the duplicate ever existed. Six weeks later,
         nobody can check whether the de-duplication was correct.
      2. We do not actually need to delete. We only need the copies to end up on the SAME
         side of the practice/exam divide. Grouping achieves that and keeps everything.

    This script therefore only ever *reports*. build_lineage_groups.py (step 3) reads the
    report and welds each listed set into one family.

THREE KINDS OF DUPLICATE, CHEAPEST FIRST
    exact        Byte-for-byte the same picture. Detected from the zip's table of contents,
                 with zero pixel reads. Free and certain.
    overlapping  Two tiles cut from the same big survey map, whose windows overlap on the
                 ground -- so the same plant appears in both. Detected from arithmetic on the
                 tile positions.
    near         Same dataset, same dimensions, nearly the same byte size. These are only
                 CANDIDATES. Confirming them would need to actually look at the pixels, which
                 this stage deliberately never does, so they are reported for a person to
                 sample by eye.

HOW TO RUN IT
    python -m training.data.audit_duplicates --manifest tables/manifest.parquet \
                                             --out tables/duplicate_report.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict   # a dict that creates missing entries automatically

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from training.data import hashing, schema  # noqa: E402


def exact_groups(df: pd.DataFrame) -> dict[str, list[str]]:
    """Find sets of photographs that are byte-for-byte identical.

    Keys on `content_hash` -- a real 128-bit xxh3 hash of the actual file bytes, computed by
    build_manifest. Two photographs with the same content_hash ARE the same file.

    (It used to key on `content_key`, a CRC32 checksum plus the byte size, both free from the
    zip index. That was verified correct on THIS corpus but is only 32 bits of checksum, which
    is not a safe basis for a decision that silently merges photographs. hashing.dedup_key
    prefers the real hash and falls back per-row for anything unhashable. See data/hashing.py.)

    Returns {hash: [image_uid, image_uid, ...]} -- only for hashes more than one photograph
    shares. A hash with a single photograph is not a duplicate, it is just a photo.
    """
    # defaultdict(list) means "if I ask for a key that isn't there, invent an empty list".
    # Without it we would have to write `if key not in g: g[key] = []` on every iteration.
    g = defaultdict(list)

    # zip() here is Python's built-in "walk two lists side by side" -- nothing to do with
    # zip ARCHIVES. Confusing name, same word, completely different thing.
    for uid, key in zip(df["image_uid"], hashing.dedup_key(df)):
        g[key].append(uid)

    # Keep only the keys with 2 or more photographs.
    return {k: v for k, v in g.items() if len(v) > 1}


def overlap_groups(df: pd.DataFrame) -> dict[str, list[str]]:
    """Find tiles cut from the same parent map whose windows overlap on the ground.

    WHY TILES OVERLAP IN THE FIRST PLACE
        A drone survey makes one enormous image, far too big to annotate. It gets chopped into
        squares. But if you chop with no overlap, a plant sitting exactly on a cut line ends up
        as two half-plants and gets annotated as neither. So the squares are cut with a margin
        of overlap, guaranteeing every object appears whole in at least one tile.

        Good for annotation. Bad for splitting -- because the same plant now genuinely appears
        in two different tiles, and those two tiles must not be separated.

    HOW WE DETECT IT
        Each tile knows the pixel position it was cut from (tile_x, tile_y) and how big it is
        (width, height). Two squares of size w x h at positions A and B overlap if their
        centres are closer than a full tile in BOTH directions:

            |x_A - x_B| < w   AND   |y_A - y_B| < h

        That's it -- no image data involved, just arithmetic on four numbers.
    """
    out = {}

    # Only tiles can overlap. A photograph with no parent map, or no recorded position, is
    # skipped. (-1 is our "unknown" marker from build_manifest, hence the >= 0 test.)
    sub = df[(df["source_mosaic"] != "") & (df["tile_x"] >= 0) & (df["tile_y"] >= 0)]

    # .groupby("source_mosaic") splits the table into one chunk per parent map. Tiles from
    # DIFFERENT maps can never overlap, so we never compare across maps.
    for mosaic, part in sub.groupby("source_mosaic"):
        if len(part) < 2:
            continue    # one tile cannot overlap itself

        # Use the median tile size for this map rather than each tile's own size: edge tiles
        # are often cut short, and a short edge tile would otherwise under-report its overlap.
        w = int(part["width"].median()) if (part["width"] > 0).any() else 0
        h = int(part["height"].median()) if (part["height"] > 0).any() else 0
        if w <= 0 or h <= 0:
            continue    # we never learned the sizes for this map; cannot judge

        boxes = list(zip(part["image_uid"], part["tile_x"], part["tile_y"]))
        members = []
        # Compare every tile with every LATER tile. `boxes[i + 1:]` means "everything after
        # position i", which stops us comparing a pair twice or a tile with itself.
        for i, (uid_a, xa, ya) in enumerate(boxes):
            for uid_b, xb, yb in boxes[i + 1:]:
                if abs(int(xa) - int(xb)) < w and abs(int(ya) - int(yb)) < h:
                    members.extend([uid_a, uid_b])

        if members:
            # set() removes the repeats (a tile overlapping five neighbours got added five
            # times), sorted() makes the output stable between runs so the report can be
            # diffed against a previous one.
            out[str(mosaic)] = sorted(set(members))
    return out


def perceptual_groups(df: pd.DataFrame, max_bits: int = 6) -> dict[str, list[str]]:
    """Find photographs that LOOK the same, even when their bytes are completely different.

    This is the question content_hash cannot answer. Re-save one photograph at a different
    JPEG quality and every byte changes -- but it is still the same picture, and for leakage
    purposes the two must not be split across the practice/exam divide.

    `max_bits` is how many of the 64 perceptual-hash bits may differ. Measured on this corpus:
        0        identical picture
        1-6      same picture, re-saved or re-compressed          <- what we group
        14+      unrelated (the closest unrelated pair, in a 4,000-pair sample, was 14 apart)
    Nothing unrelated came within 6, so 6 is a safe threshold with room to spare.

    THE SPEED PROBLEM
        Comparing every photograph with every other is 82,099 x 82,098 / 2 = 3.4 BILLION
        comparisons. Done as a Python loop that does not finish in any useful time -- the
        first attempt at this ran for three minutes without emitting a single group.

    THE TWO FIXES
        1. COLLAPSE TO UNIQUE HASHES FIRST. 82,099 photographs carry only 45,907 distinct
           perceptual hashes, so most of the work is comparing a hash with itself. Group the
           identical ones for free in a dict, compare only the distinct values, then expand
           back at the end. That alone removes about half the comparisons.

        2. DO THE COMPARISON IN NUMPY, NOT IN PYTHON. Pack the hashes into 64-bit integers,
           XOR a block of them against the rest in one array operation, and count the
           differing bits with np.bitwise_count. Python does the loop 180 times over big
           blocks instead of a billion times over single pairs.

        Chunked to keep memory bounded, and only the upper triangle is compared, since
        distance(a, b) is the same as distance(b, a).
    """
    if "phash" not in df.columns:
        return {}
    sub = df[df["phash"].fillna("") != ""]
    if sub.empty:
        return {}

    # ---- 1. collapse identical hashes ---------------------------------------------------
    by_hash: dict[str, list[str]] = defaultdict(list)
    for uid, h in zip(sub["image_uid"], sub["phash"]):
        by_hash[h].append(uid)
    uniq = list(by_hash.keys())
    n = len(uniq)

    # np.uint64 holds a 64-bit hash exactly. `int(h, 16)` reads the hex string as a number.
    H = np.array([int(h, 16) for h in uniq], dtype=np.uint64)

    # ---- 2. vectorised near-pair search --------------------------------------------------
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    CHUNK = 256          # rows per block: 256 x 45,907 x 8 bytes is about 94 MB
    pairs = 0
    for i0 in range(0, n, CHUNK):
        i1 = min(i0 + CHUNK, n)
        # XOR every hash in this block against every hash from i0 onward. `[:, None]` turns
        # the block into a column so numpy broadcasts it against the row -- producing the
        # whole block-vs-rest difference matrix in one operation.
        diff = np.bitwise_xor(H[i0:i1, None], H[None, i0:])
        # bitwise_count counts the 1-bits in each entry: exactly the Hamming distance.
        dist = np.bitwise_count(diff)
        # np.nonzero gives the positions that pass. `+ i0` puts the column index back into
        # the full array's coordinates.
        rows, cols = np.nonzero(dist <= max_bits)
        for r, c in zip(rows, cols + i0):
            if r + i0 < c:                     # strictly upper triangle: skips self-pairs
                union(int(r + i0), int(c))
                pairs += 1

    # ---- 3. expand back to photographs ---------------------------------------------------
    groups: dict[int, list[str]] = defaultdict(list)
    for i, h in enumerate(uniq):
        groups[find(i)].extend(by_hash[h])
    return {f"p{r:06d}": v for r, v in groups.items() if len(v) > 1}


def near_candidates(df: pd.DataFrame, tol: float = 0.02) -> list[dict]:
    """FALLBACK for when no perceptual hash has been computed yet.

    Same dataset, same dimensions, byte size within `tol`. This is a crude proxy: two
    genuinely different photographs of the same field, seconds apart, also match it. It is
    reported for a person to sample, never acted on.

    Run build_phash.py and use perceptual_groups() instead -- it answers the question properly
    rather than guessing at it.
    """
    cands = []
    sub = df[(df["width"] > 0) & (df["height"] > 0)]   # skip rows with unknown dimensions

    # Group by (dataset, width, height) -- copies live in the same dataset at the same size.
    for (ds, w, h), part in sub.groupby(["dataset", "width", "height"]):
        if len(part) < 2 or len(part) > 4000:
            # A block of thousands of same-sized files in one dataset is a TILE GRID, not a
            # pile of copies -- every tile is 512x512 by construction.
            continue
        rows = part.sort_values("size_bytes")[["image_uid", "size_bytes", "content_key"]].values
        for i in range(len(rows) - 1):
            a_uid, a_sz, a_key = rows[i]
            b_uid, b_sz, b_key = rows[i + 1]
            if a_key == b_key or a_sz <= 0:
                continue
            if abs(int(b_sz) - int(a_sz)) / int(a_sz) <= tol:
                cands.append({"dataset": str(ds), "a": str(a_uid), "b": str(b_uid),
                              "size_a": int(a_sz), "size_b": int(b_sz)})
    return cands


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="tables/manifest.parquet")
    ap.add_argument("--out", default="tables/duplicate_report.json")
    ap.add_argument("--near-tolerance", type=float, default=0.02,
                    help="byte-size tolerance for the FALLBACK near-candidate proxy. 0.02 = 2%%")
    ap.add_argument("--phash-bits", type=int, default=6,
                    help="how many of the 64 perceptual bits may differ and still count as "
                         "the same picture. Must be < 8. Measured: unrelated pairs are 14+ "
                         "apart, so 6 is safe.")
    args = ap.parse_args()

    df = pd.read_parquet(args.manifest)
    schema.validate(df, schema.MANIFEST_COLUMNS, "manifest")   # check what we just read
    print(f"{len(df):,} photographs")

    ex = exact_groups(df)
    ov = overlap_groups(df)

    # Prefer the real perceptual hash if build_phash has been run. Fall back to the crude
    # byte-size proxy only when it has not, so this step still works on a fresh manifest.
    pg = perceptual_groups(df, args.phash_bits)
    nr = [] if pg else near_candidates(df, args.near_tolerance)

    ex_imgs = sum(len(v) for v in ex.values())
    # A photograph can appear in several overlap groups, so count DISTINCT ones.
    # The double-for inside {...} is a set comprehension: "for each group v, for each uid u
    # in that group, collect u" -- and the set removes repeats.
    ov_imgs = len({u for v in ov.values() for u in v})

    # ---- The dangerous kind of duplication ------------------------------------------------
    # A duplicate INSIDE one dataset is annoying. A duplicate that spans TWO datasets is
    # dangerous, because splits are largely organised by dataset -- so those two copies are
    # actively likely to land on opposite sides of the divide.
    by_ds = {k: df.set_index("image_uid").loc[v, "dataset"].nunique() for k, v in ex.items()}
    cross = sum(1 for n in by_ds.values() if n > 1)

    report = {
        "photographs": int(len(df)),
        "exact": {"groups": len(ex), "photographs": ex_imgs,
                  "cross_dataset_groups": cross,
                  # default=0 stops max() crashing if there are no groups at all
                  "largest_group": max((len(v) for v in ex.values()), default=0)},
        "overlapping": {"mosaics": len(ov), "photographs": ov_imgs},
        "perceptual": {"groups": len(pg),
                       "photographs": len({u for v in pg.values() for u in v}),
                       "max_bits": args.phash_bits,
                       "largest_group": max((len(v) for v in pg.values()), default=0)},
        # Only the first 25 candidate pairs go in the summary -- the point is for a human to
        # spot-check a sample, not to read thousands of rows.
        "near_candidates": {"pairs": len(nr), "sample": nr[:25]},
        "note": "Nothing is deleted. build_lineage_groups joins every listed set into one family.",
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        # {**report, ...} means "everything in report, plus these extra keys". The full group
        # lists are written to the file (step 3 needs them) even though only the summary is
        # printed to the screen.
        json.dump({**report, "exact_groups": {k: v for k, v in list(ex.items())},
                   "overlap_groups": ov, "perceptual_groups": pg}, fh, indent=1)

    print(f"  exact        {len(ex):,} groups covering {ex_imgs:,} photographs "
          f"({100 * ex_imgs / max(len(df), 1):.1f}%)")
    print(f"               {cross:,} of those groups span more than one dataset")
    print(f"               largest single group: {report['exact']['largest_group']:,} copies")
    print(f"  overlapping  {len(ov):,} parent maps, {ov_imgs:,} photographs")
    if pg:
        pgi = report["perceptual"]["photographs"]
        print(f"  perceptual   {len(pg):,} groups covering {pgi:,} photographs "
              f"(phash within {args.phash_bits} bits)")
        print(f"               largest: {report['perceptual']['largest_group']:,} photographs")
    else:
        print(f"  perceptual   -- no phash column; run build_phash.py")
        print(f"  near         {len(nr):,} candidate pairs -- REVIEW REQUIRED, not acted on")
    print(f"\nwrote {args.out}")
    print("Review the report before continuing. Spot-check ten pairs by eye.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
