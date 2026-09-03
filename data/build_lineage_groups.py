#!/usr/bin/env python3
"""
build_lineage_groups.py -- STEP 3: DECIDE WHICH PHOTOGRAPHS ARE FAMILY
======================================================================

THIS IS THE MOST CONSEQUENTIAL SCRIPT IN THE PIPELINE.

WHAT IT DOES
    It sorts all 82,099 photographs into "families". A family is a set of photographs that are
    so closely related that they must never be separated -- if one is in the practice pile, all
    of them must be in the practice pile.

WHY IT MATTERS SO MUCH
    Every score we ever report depends on this being right, and a mistake here is INVISIBLE in
    the numbers. If a photograph of a garlic field ends up in the practice pile and the tile
    right next to it -- showing the same plants from a slightly different angle -- ends up in
    the exam pile, the model scores brilliantly on that exam. It has effectively already seen
    the answer. Nothing in the score says so. It just looks like a very good model.

    The whole point of this script is to make that impossible before it can happen.

WHAT IS "UNION-FIND"?
    A very old, very simple technique for answering "are these two things in the same group?"
    when groups keep merging.

    Picture 82,099 people, each alone in their own room. You are told facts like "A and B are
    related". Each time, you knock down the wall between A's group and B's group. At the end,
    whoever shares a room is family.

    The trick is that you never store the full membership of a group. Each person just
    remembers ONE other person -- a pointer "upwards". Follow the pointers and you arrive at
    the group's representative, called the ROOT. Two people are in the same group if they
    arrive at the same root. Merging two groups is a single operation: point one root at the
    other. See the Union class below.

THE EVIDENCE LADDER -- STRONGEST FIRST
    We are not equally sure about every reason two photographs might be related, so the
    reasons are ranked, and the rank that joined each photograph is recorded:

      1 duplicate   identical bytes                          CERTAIN
      2 mosaic      cut from the same parent map             very strong (61,723 of 82,099 rows)
      3 flight      same drone flight                        strong
      4 spacetime   same timestamp                           strong
      5 perceptual  overlapping tiles (from the step-2 report) moderate
      6 site        the spreadsheet says it is the same place  FALLBACK ONLY

    Recording the rank means that months later, when a family looks wrong, you can ask "which
    rule put these together?" and get an actual answer.

WHICH DIRECTION DO WE ERR IN?
    Deliberately towards grouping TOO MUCH.

    Group too much  -> some genuinely independent photographs get locked together, so we lose
                       a little usable training data. Costs us data. Cannot inflate a score.
    Group too little -> related photographs land on opposite sides of the divide, and every
                       number we publish is wrong in our own favour. Nobody can tell.

    The first mistake is cheap and visible. The second is expensive and invisible. So when in
    doubt, we group.

HOW TO RUN IT
    python -m training.data.build_lineage_groups --manifest tables/manifest.parquet \
        --site-map ~/workspace/manifest/site_mapping.csv \
        --duplicates tables/duplicate_report.json --out tables/lineage_groups.parquet

NOTE: this script also absorbs the old manifest/apply_site_mapping.py -- the join with the
human-filled spreadsheet happens here, because that is where its output is first needed.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from training.data import hashing, schema  # noqa: E402


class Union:
    """Union-find: the "knock down the wall between two rooms" structure described above.

    Two methods:
        find(a)          -> which group is `a` in? (returns the group's root)
        union(a, b, rank) -> merge a's group and b's group
    """

    def __init__(self, keys):
        # Everyone starts as their own root -- 82,099 groups of one.
        self.parent = {k: k for k in keys}
        # The strongest evidence (LOWEST rank number) that has ever touched this photograph.
        # 99 means "nothing yet". Used only for reporting, never for the grouping itself.
        self.rank_of = {k: 99 for k in keys}

    def find(self, a):
        """Follow the pointers upward until we reach the root of a's group."""
        while self.parent[a] != a:      # a root is the only node that points at itself
            # PATH COMPRESSION: as we walk up, re-point each node at its GRANDparent. This
            # flattens the chain a little on every lookup, so the structure never degenerates
            # into a long line. Without it, 82,099 merges could build a chain 82,099 deep and
            # every lookup would walk the whole thing.
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a

    def union(self, a, b, rank):
        """Merge the group containing `a` with the group containing `b`."""
        ra, rb = self.find(a), self.find(b)
        # Record the strongest reason each of these two was ever joined by. min() because
        # rank 1 is stronger than rank 6 -- a smaller number wins.
        self.rank_of[a] = min(self.rank_of[a], rank)
        self.rank_of[b] = min(self.rank_of[b], rank)
        if ra != rb:                 # already the same group? nothing to do
            self.parent[rb] = ra     # point b's root at a's root. One line, groups merged.


def apply_site_map(df: pd.DataFrame, path: str) -> pd.DataFrame:
    """Join the human-filled spreadsheet onto every photograph row.

    THE SPREADSHEET
        A person goes through the 212 datasets and writes down, for each one: which farm it
        is, which field, which season, a stable site_id, and whether the annotation is
        exhaustive. This is the one part of the pipeline a computer cannot do -- the
        information simply is not in the files.

    WHY ONLY NON-EMPTY VALUES OVERWRITE
        The spreadsheet takes hours to fill in. This function is written so it can be run
        repeatedly on a HALF-FILLED sheet: a blank cell leaves whatever was already there,
        rather than wiping it. So you can fill in fifty rows, run the pipeline, see how it
        looks, fill in fifty more. (This is inherited from the original apply_site_mapping.py.)

    IF THERE IS NO SPREADSHEET YET
        We fall back to using the dataset name as the site_id. That is coarser -- it treats
        every dataset as one place -- but it is the SAFE direction to be wrong in (see the
        header: grouping too much only costs data).
    """
    if not path or not os.path.exists(path):
        print(f"  [warn] no site map at {path}; site fallback will use `dataset` instead")
        # .where(condition, other) keeps the value where the condition is True, and swaps in
        # `other` where it is False. So: keep site_id if it is non-empty, else use dataset.
        df["site_id"] = df["site_id"].where(df["site_id"] != "", df["dataset"])
        return df

    # Read the CSV into a lookup: {dataset_name: that_row_of_the_spreadsheet}
    by_ds = {}
    with open(path, newline="") as fh:
        # DictReader gives each row as a dict keyed by the column headers, so we can write
        # row["farm"] instead of row[6].
        for row in csv.DictReader(fh):
            by_ds[row.get("dataset", "").strip()] = row

    for col in ("farm", "field", "season", "site_id", "exhaustive"):
        if col not in df.columns:
            df[col] = ""
        vals = []
        for ds, cur in zip(df["dataset"], df[col]):
            # `(x or "")` guards against the cell being missing OR being None.
            v = (by_ds.get(ds, {}).get(col) or "").strip()
            vals.append(v if v else cur)   # blank in the sheet -> keep what we had
        df[col] = vals

    filled = int((df["site_id"].astype(str).str.len() > 0).sum())
    print(f"  site map applied: {filled:,}/{len(df):,} rows carry a site_id")
    # Any row the sheet did not cover still needs a coarse grouping key.
    df["site_id"] = df["site_id"].where(df["site_id"] != "", df["dataset"])
    return df


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="tables/manifest.parquet")
    ap.add_argument("--site-map", default=os.path.expanduser("~/workspace/manifest/site_mapping.csv"))
    ap.add_argument("--duplicates", default="tables/duplicate_report.json")
    ap.add_argument("--out", default="tables/lineage_groups.parquet")
    ap.add_argument("--write-manifest", action="store_true",
                    help="also write the group id and site columns back into the manifest")
    args = ap.parse_args()

    df = pd.read_parquet(args.manifest)
    schema.validate(df, schema.MANIFEST_COLUMNS, "manifest")
    print(f"{len(df):,} photographs")

    df = apply_site_map(df, args.site_map)

    # Everyone alone in their own room, to begin with.
    uf = Union(df["image_uid"].tolist())

    def join_by(series, rank, name, skip_empty=True):
        """Merge every photograph that shares a value in `series`.

        `series` is one column of the table. Photographs sharing a non-empty value in that
        column get welded into one family, with the given evidence rank.
        """
        buckets = defaultdict(list)
        for uid, key in zip(df["image_uid"], series):
            k = str(key)
            # An EMPTY value is not evidence of anything. If we did not skip these, all
            # 20,376 photographs with no known parent map would be joined into one family on
            # the grounds that they all equally have no parent map -- which is nonsense.
            # "-1" and "nan" are the two other ways "unknown" shows up in this data.
            if skip_empty and (k == "" or k == "-1" or k == "nan"):
                continue
            buckets[k].append(uid)

        merged = 0
        for members in buckets.values():
            # Chain everyone in the bucket to the FIRST member. n members needs only n-1
            # merges, not n*(n-1)/2 -- union-find makes them all one family either way.
            for other in members[1:]:
                uf.union(members[0], other, rank)
                merged += 1
        print(f"  {rank} {name:<11} {len(buckets):,} keys, {merged:,} merges")

    # ---- Ranks 1 to 4: run straight off manifest columns ----------------------------------
    # Rank 1 keys on content_hash (real 128-bit xxh3), not the old CRC32+size key.
    join_by(hashing.dedup_key(df), 1, "duplicate")    # identical bytes
    join_by(df["source_mosaic"], 2, "mosaic")         # same parent survey map
    join_by(df["flight_id"], 3, "flight")             # same drone flight
    join_by(df["capture_datetime"], 4, "spacetime")   # same instant in time

    # ---- Rank 5: overlapping tiles, from step 2's report -----------------------------------
    # Rank 5 is TRUE perceptual evidence: photographs that LOOK identical even though their
    # bytes differ -- a re-saved or re-compressed copy. This slot was always named
    # "perceptual" in the schema but used to be filled by tile-overlap arithmetic, because
    # nothing perceptual existed. build_phash.py now fills it properly. Overlapping tiles are
    # still joined here too: they genuinely show the same ground.
    if os.path.exists(args.duplicates):
        rep = json.load(open(args.duplicates))
        merged = ov_merged = 0
        for members in (rep.get("overlap_groups") or {}).values():
            for other in members[1:]:
                # Guard against a stale report naming photographs that are no longer in the
                # manifest -- an unknown key would crash union-find.
                if other in uf.parent and members[0] in uf.parent:
                    uf.union(members[0], other, 5)
                    ov_merged += 1
        for members in (rep.get("perceptual_groups") or {}).values():
            for other in members[1:]:
                if other in uf.parent and members[0] in uf.parent:
                    uf.union(members[0], other, 5)
                    merged += 1
        print(f"  5 perceptual  {merged:,} merges from phash, "
              f"{ov_merged:,} from tile overlap")
    else:
        print(f"  [warn] no duplicate report at {args.duplicates}; overlap evidence skipped")

    # =========================================================================================
    # RANK 6: THE SITE FALLBACK -- AND THE BUG THAT LIVED HERE
    # =========================================================================================
    # The obvious implementation is to run join_by(df["site_id"], 6, "site") like the others.
    # That is WRONG, and it destroyed the grouping the first time we ran it.
    #
    # Why: union-find is TRANSITIVE. Photo A and photo B are duplicates, so they are family.
    # But A lives in dataset X and B lives in dataset Y -- and there are 14,653 such
    # cross-dataset duplicate groups. Now join everything in dataset X together, and everything
    # in dataset Y together. A's link to B has just welded ALL of X to ALL of Y. Do that
    # 14,653 times and the chain runs dataset -> duplicate -> dataset -> duplicate across the
    # whole corpus.
    #
    # MEASURED RESULT: one single family containing 76% of all 82,099 photographs. Which means
    # 76% of the data has to go into one split, and the whole splitting exercise is destroyed.
    #
    # THE FIX: rank 6 is a FALLBACK, not another merge pass. It applies ONLY to photographs
    # that are still completely alone after ranks 1-5 -- the ones for which we have no better
    # evidence at all. Those cannot chain anything, because they are not attached to anything.
    #
    # Result after the fix: 9,242 families, largest holds 2.3%.
    # =========================================================================================
    sizes_now = Counter(uf.find(u) for u in df["image_uid"])   # how big is each family now?
    lonely = [u for u in df["image_uid"] if sizes_now[uf.find(u)] == 1]
    lonely_set = set(lonely)   # a set, because `in` on a set is instant, on a list it is slow

    buckets = defaultdict(list)
    for uid, key in zip(df["image_uid"], df["site_id"]):
        if uid in lonely_set:       # <-- the entire fix is this one line
            buckets[str(key)].append(uid)
    merged = 0
    for members in buckets.values():
        for other in members[1:]:
            uf.union(members[0], other, 6)
            merged += 1
    print(f"  6 site        {len(lonely):,} unattached photographs, {merged:,} merges "
          f"(fallback only)")

    # ---- Turn the union-find structure into a table ----------------------------------------
    roots = [uf.find(u) for u in df["image_uid"]]

    # Roots are currently arbitrary image_uids. Rename them to tidy sequential names --
    # g000000, g000001, ... -- so the output is readable and stable between runs.
    # ":06d" means "pad with zeros to 6 digits".
    names = {r: f"g{i:06d}" for i, r in enumerate(sorted(set(roots)))}

    # For each photograph, translate its numeric rank back into the evidence NAME.
    # next((...), "site") means: take the first match, or "site" if nothing matched.
    # Rank 99 (never merged with anyone) falls through to "site" as well -- a family of one.
    rank_to_name = {rk: n for n, rk in schema.LINEAGE_EVIDENCE}
    out = pd.DataFrame({
        "image_uid": df["image_uid"],
        "leakage_group_id": [names[r] for r in roots],
        "evidence": [rank_to_name.get(uf.rank_of[u], "site") for u in df["image_uid"]],
        "evidence_rank": [int(uf.rank_of[u]) if uf.rank_of[u] != 99 else 6
                          for u in df["image_uid"]],
    })
    schema.validate(out, schema.LINEAGE_COLUMNS, "lineage_groups")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out.to_parquet(args.out, index=False)

    # ---- Report ---------------------------------------------------------------------------
    sizes = Counter(out["leakage_group_id"])
    big = sizes.most_common(1)[0]     # (name, size) of the biggest family
    n_groups = len(sizes)
    print(f"\nwrote {args.out}")
    print(f"  families            {n_groups:,}")
    print(f"  median size         {sorted(sizes.values())[len(sizes) // 2]:,}")
    print(f"  largest family      {big[1]:,} photographs ({big[0]})")
    print(f"  evidence mix        {dict(Counter(out['evidence']).most_common())}")

    # ---- The two alarms --------------------------------------------------------------------
    # These exist precisely because of the rank-6 bug above. They are the automatic version of
    # "does this result look sane?", so the same mistake cannot pass silently a second time.
    share = big[1] / max(len(df), 1)
    if share > 0.5:
        print(f"  [WARN] the largest family holds {100 * share:.0f}% of the corpus. "
              f"A join is too aggressive; inspect before continuing.")
    if n_groups < 30:
        # Fewer than 30 families means the statistical comparison used to decide whether one
        # model beats another has too few independent units to say anything.
        print(f"  [WARN] only {n_groups} families. The paired comparison needs at least 30.")

    # ---- Optionally write the answer back into the manifest ---------------------------------
    # Off by default: the manifest is the input to this script, and overwriting your input is
    # how you end up unable to re-run a step. Pass --write-manifest when you are satisfied.
    if args.write_manifest:
        df = df.drop(columns=["leakage_group_id"]).merge(
            out[["image_uid", "leakage_group_id"]], on="image_uid", how="left")
        df.to_parquet(args.manifest, index=False)
        print(f"  manifest updated with leakage_group_id and site columns")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
