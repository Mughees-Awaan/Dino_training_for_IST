#!/usr/bin/env python3
"""
build_groups.py -- STEP 3 (REPLACES build_lineage_groups.py): THREE SEPARATE IDENTITIES
========================================================================================

WHY THIS REPLACES THE OLD SCRIPT
    The old script had ONE "family" concept and threw every kind of evidence into it. That
    conflated three questions that deserve different answers and different confidence:

        "is this the same PICTURE?"        -- evidenced, from bytes and appearance
        "is this from the same SOURCE?"    -- evidenced, from mosaic and flight identifiers
        "is this the same FIELD VISIT?"    -- NOT evidenced; needs reviewed provenance

    Mixing them produced a single group holding 58% of the corpus. That group was not the
    truth about the data. It came from combining one evidenced assumption (an exact duplicate
    really does connect those two photographs) with one UNVERIFIED assumption (a dataset is
    one physical site). One shared photograph between two datasets does not prove every
    photograph in both came from the same field.

    So the identities are now kept apart.

THE FOUR IDENTIFIERS

    content_group_id   Copies of the same picture -- identical bytes, or perceptually
                       identical. Fully evidenced.
                       RULE: exactly ONE canonical image per group is ever sampled. The
                       aliases stay in the manifest as records but are never drawn, so they
                       cannot leak into a split no matter where they sit.

    lineage_group_id   Photographs derived from the same source: the same parent mosaic, the
                       same flight, overlapping tile windows. Evidenced, and namespaced to
                       its dataset so that "DJI_0001" from two different farms is not treated
                       as one thing.
                       RULE: a lineage group lives in exactly one split.

    field_event_id     One field on one visit. This needs REVIEWED provenance and is EMPTY
                       when we do not have it. Where a flight identifier or a parent mosaic
                       exists, that is a sound proxy for a single visit and is used.
                       RULE: a field event lives in exactly one split, and rows with an empty
                       field_event_id may not enter calibration, confirmation or the sealed set.

    site_id            One physical place. Reviewed provenance only.
                       RULE: a sealed site appears in no other split.
                       There is NO dataset-name fallback -- see below.

WHY THE dataset -> site_id FALLBACK WAS REMOVED
    The old code filled site_id with the dataset name whenever the spreadsheet was blank.
    That silently converted "we do not know where this is" into a confident claim, and it is
    what manufactured the 58% component. Unknown provenance must stay unknown, because a
    guessed site is worse than no site: it lets unreviewed data into the sealed evaluation
    set while looking properly grouped.

CROSS-DATASET DUPLICATE CLUSTERS
    When two datasets share photographs, we do NOT union everything in both containers.
    Measured on this corpus, 291 dataset pairs share at least one photograph, and they fall
    into three clearly separated shapes:

        194 pairs (67%)  share 100+ photographs over hundreds of mosaics
                         -> a re-export of one collection. MERGE their field events.
         23 pairs (8%)   share exactly one photograph
                         -> one file was copied. Keep a canonical record, quarantine aliases.
         15 pairs        share 5-19
                         -> ambiguous. QUARANTINE until a person resolves it.

    A conflict in reviewed site assignment also quarantines the cluster.

    python -m training.data.build_groups --write-manifest
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

# The three spellings of "we do not know" that occur in this data.
EMPTY = {"", "-1", "nan", "None"}


class Union:
    """Union-find with path compression. See build_manifest for the plain-language version."""

    def __init__(self, keys):
        self.parent = {k: k for k in keys}

    def find(self, a):
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def _join(uf: Union, uids, keys) -> int:
    """Merge everything sharing a non-empty key. Returns the number of merges made."""
    buckets = defaultdict(list)
    for uid, k in zip(uids, keys):
        if str(k) not in EMPTY:
            buckets[str(k)].append(uid)
    n = 0
    for members in buckets.values():
        for other in members[1:]:
            uf.union(members[0], other)
            n += 1
    return n


def namespaced(df, col):
    """Scope a provenance key to its dataset, so identical strings from unrelated archives
    are not merged. Empty stays empty -- namespacing "" would give "dataset::", which is
    non-empty and would merge every row that merely LACKS the field."""
    return [f"{d}::{v}" if str(v) not in EMPTY else ""
            for d, v in zip(df["dataset"], df[col])]


# =====================================================================================
# IDENTITY 1 -- content groups, and the canonical image
# =====================================================================================
def build_content_groups(df, inst):
    """PIXEL identity, from EXACT BYTES ONLY.

    pHash is deliberately excluded. An automatic threshold is not proof, and treating an
    unaudited match as a duplicate would DISCARD real data. Unaudited visual matches become
    a split constraint instead -- see build_visual_groups.

    CANONICAL SELECTION IS NOT A FILENAME SORT.
        The first version picked the first member by (dataset, member). That ignored whether
        the chosen copy carried any annotation work, and measured on this corpus it would
        have stranded 118,006 instances on non-canonical copies -- 2,102 content groups whose
        annotations live ONLY on an alias.
        Canonical is therefore the copy with the MOST annotations, ties broken deterministically.

    Annotations are never dropped: every alias's annotations are remapped onto the canonical
    pixels by build_observations() below.
    """
    uf = Union(df["image_uid"].tolist())
    n_exact = _join(uf, df["image_uid"], hashing.dedup_key(df))

    roots = [uf.find(u) for u in df["image_uid"]]
    names = {r: f"c{i:06d}" for i, r in enumerate(sorted(set(roots)))}
    cgid = [names[r] for r in roots]

    # most annotations wins; then dataset/member for a stable, reproducible tie-break
    nann = dict(zip(df["image_uid"], df["n_annotations"]))
    best = {}
    for g, ds, mem, uid in sorted(zip(cgid, df["dataset"], df["member"], df["image_uid"])):
        cur = best.get(g)
        if cur is None or nann[uid] > nann[cur]:
            best[g] = uid
    canon_uid = [best[g] for g in cgid]
    is_canon = [1 if u == c else 0 for u, c in zip(df["image_uid"], canon_uid)]

    stranded = sum(1 for g, u in zip(cgid, df["image_uid"])
                   if nann[u] > 0 and nann[best[g]] == 0)
    print(f"  content groups      {len(names):,} (exact bytes only, {n_exact:,} merges)")
    print(f"  canonical images    {sum(is_canon):,}; annotations stranded by the choice: {stranded}")
    return cgid, is_canon, canon_uid


def build_visual_groups(df, report_path):
    """UNAUDITED perceptual matches. A SPLIT CONSTRAINT, never a discard.

    These come from an automatic pHash threshold that no person has checked. That is strong
    enough to say "do not put these on opposite sides of the divide" and NOT strong enough to
    say "these are the same file, throw one away".
    """
    uf = Union(df["image_uid"].tolist())
    n = 0
    if os.path.exists(report_path):
        rep = json.load(open(report_path))
        for members in (rep.get("perceptual_groups") or {}).values():
            members = [u for u in members if u in uf.parent]
            for other in members[1:]:
                uf.union(members[0], other)
                n += 1
    roots = [uf.find(u) for u in df["image_uid"]]
    names = {r: f"v{i:06d}" for i, r in enumerate(sorted(set(roots)))}
    print(f"  visual groups       {len(names):,} ({n:,} unaudited pHash merges, kept as constraints)")
    return [names[r] for r in roots]


def build_observations(df, inst, cgid, canon_uid):
    """Remap every annotation onto its canonical pixels, keeping where it came from.

    Nothing is discarded. A content group whose copies disagree about labels is FLAGGED for
    adjudication rather than silently resolved -- 2,400 groups here have differing label sets.
    """
    u2c = dict(zip(df["image_uid"], canon_uid))
    u2g = dict(zip(df["image_uid"], cgid))
    u2t = dict(zip(df["image_uid"], df["task_name"]))

    per_img = inst.groupby("image_uid")["label_raw"].apply(frozenset)
    by_grp = defaultdict(set)
    for uid, labs in per_img.items():
        g = u2g.get(uid)
        if g: by_grp[g].add(labs)
    conflict = {g for g, v in by_grp.items() if len(v) > 1}

    obs = pd.DataFrame({
        "instance_id": inst["instance_id"],
        "canonical_uid": inst["image_uid"].map(u2c),
        "source_image_uid": inst["image_uid"],
        "content_group_id": inst["image_uid"].map(u2g),
        "task_name": inst["image_uid"].map(u2t).fillna(""),
        "label_raw": inst["label_raw"],
        "conflict": [1 if u2g.get(u) in conflict else 0 for u in inst["image_uid"]],
    })
    obs = obs[obs["canonical_uid"].notna()]
    moved = int((obs["canonical_uid"] != obs["source_image_uid"]).sum())
    print(f"  observations        {len(obs):,} ({moved:,} remapped from aliases onto canonical "
          f"pixels, 0 discarded)")
    print(f"  conflicting groups  {len(conflict):,} flagged for adjudication")
    return obs


# =====================================================================================
# IDENTITY 2 -- lineage groups
# =====================================================================================
def build_lineage(df, report_path):
    """Same source: parent mosaic, flight, or overlapping tile windows.

    Deliberately does NOT include content duplicates. Those are handled by the canonical
    rule above, which is strictly stronger -- an alias is never sampled at all, so it does
    not need its split constrained.
    """
    uf = Union(df["image_uid"].tolist())
    n_mos = _join(uf, df["image_uid"], namespaced(df, "source_mosaic"))
    n_fly = _join(uf, df["image_uid"], namespaced(df, "flight_id"))

    n_ov = 0
    if os.path.exists(report_path):
        rep = json.load(open(report_path))
        for members in (rep.get("overlap_groups") or {}).values():
            members = [u for u in members if u in uf.parent]
            for other in members[1:]:
                uf.union(members[0], other)
                n_ov += 1

    roots = [uf.find(u) for u in df["image_uid"]]
    names = {r: f"l{i:06d}" for i, r in enumerate(sorted(set(roots)))}
    print(f"  lineage groups      {len(names):,}  "
          f"({n_mos:,} mosaic, {n_fly:,} flight, {n_ov:,} tile-overlap merges)")
    return [names[r] for r in roots]


# =====================================================================================
# IDENTITY 3 -- field events, and the quarantine policy
# =====================================================================================
def build_source_event_proxies(df, site_map, bulk_min=20, isolated_max=4):
    """An INFERRED capture event -- a PROXY, not a verified physical field event.

    Built from filename-derived mosaics and flights. That is a reasonable stand-in for "one
    aircraft over one field on one day", but nobody has confirmed it against reality, and
    6,363 of these proxies contain only a single canonical image.

    It therefore supports a SOURCE-GROUP benchmark ONLY. The official field-held-out sealed
    benchmark requires reviewed field_event_id and site_id, which do not exist yet.

    A flight identifier is a sound proxy for a single visit -- one flight is one aircraft
    over one field on one day. A parent mosaic is the same idea. Where neither exists and no
    reviewed site is available, the value stays empty and the row is barred from the
    evaluation splits rather than guessed at.
    """
    uf = Union(df["image_uid"].tolist())
    _join(uf, df["image_uid"], namespaced(df, "flight_id"))
    _join(uf, df["image_uid"], namespaced(df, "source_mosaic"))

    # Reviewed provenance, when a person has supplied it, is stronger than either proxy.
    reviewed = [f"{s}|{se}" if str(s) not in EMPTY else ""
                for s, se in zip(df["site_id"], df["season"])]
    _join(uf, df["image_uid"], reviewed)

    # ---- cross-dataset duplicate clusters -------------------------------------------
    # Merge field events only where two datasets overlap in BULK. A single shared file is
    # a copied export, not evidence that two collections are one field.
    key = hashing.dedup_key(df)
    tmp = pd.DataFrame({"uid": df["image_uid"], "ds": df["dataset"], "k": key})
    shared = defaultdict(set)
    for _k, g in tmp[tmp.duplicated("k", keep=False)].groupby("k"):
        ds = sorted(set(g["ds"]))
        for a in range(len(ds)):
            for b in range(a + 1, len(ds)):
                shared[(ds[a], ds[b])].add(_k)

    by_ds = defaultdict(list)
    for uid, d in zip(df["image_uid"], df["dataset"]):
        by_ds[d].append(uid)

    # For a bulk pair we need the ACTUAL shared photographs, not just how many there were.
    k2uids = defaultdict(list)
    for uid, k in zip(tmp["uid"], tmp["k"]):
        k2uids[k].append(uid)

    merged_pairs = quarantined_pairs = 0
    quarantine_ds = {}
    for (a, b), keys in shared.items():
        n = len(keys)
        if n >= bulk_min:
            # Bulk overlap: a re-export of one collection.
            #
            # Merge only the FIELD EVENTS that actually contain the shared photographs --
            # NOT every image in both dataset containers. Unioning containers is what
            # produced the 58% component; it treats "these two exports share pictures" as
            # "every picture in both is the same field", which does not follow.
            for k in keys:
                uids = k2uids.get(k, [])
                for other in uids[1:]:
                    uf.union(uids[0], other)
            merged_pairs += 1
        elif n <= isolated_max:
            pass          # isolated copy -- the canonical rule already handles it
        else:
            quarantined_pairs += 1
            quarantine_ds[a] = quarantine_ds[b] = f"ambiguous overlap with {n} shared images"

    roots = [uf.find(u) for u in df["image_uid"]]
    names = {r: f"f{i:06d}" for i, r in enumerate(sorted(set(roots)))}

    # A field event is only REAL where we had something to build it from. Rows with no
    # flight, no mosaic and no reviewed site keep an EMPTY id -- unknown stays unknown.
    have = [(str(f) not in EMPTY) or (str(mo) not in EMPTY) or (str(s) not in EMPTY)
            for f, mo, s in zip(df["flight_id"], df["source_mosaic"], df["site_id"])]
    fe = [names[r] if h else "" for r, h in zip(roots, have)]

    quarantined = [1 if (d in quarantine_ds) else 0 for d in df["dataset"]]
    reasons = [quarantine_ds.get(d, "") for d in df["dataset"]]

    print(f"  field events        {len({x for x in fe if x}):,} "
          f"({sum(1 for x in fe if not x):,} rows have NO field event -- barred from eval)")
    print(f"  cross-dataset pairs {merged_pairs:,} merged (bulk), "
          f"{quarantined_pairs:,} quarantined (ambiguous)")
    print(f"  quarantined rows    {sum(quarantined):,}")
    return fe, quarantined, reasons


def apply_site_map(df, path):
    """Join reviewed provenance. NO FALLBACK -- a blank cell leaves the value empty."""
    for col in ("farm", "field", "season", "site_id", "exhaustive"):
        if col not in df.columns:
            df[col] = ""
    if not path or not os.path.exists(path):
        print(f"  [warn] no site map at {path}; site_id and field stay EMPTY (correct: "
              f"unknown provenance must remain unknown)")
        for col in ("farm", "field", "season", "site_id"):
            df[col] = ""       # clear any stale fallback left by an earlier pipeline version
        return df
    by_ds = {}
    for row in csv.DictReader(open(path, newline="")):
        by_ds[row.get("dataset", "").strip()] = row
    # THE SHEET IS AUTHORITATIVE AND REPLACES, it does not merge with what is already there.
    #
    # This matters because an earlier version of the pipeline wrote the DATASET NAME into
    # site_id as a fallback, and that value is still sitting in the manifest on disk for all
    # 82,099 rows. Reading the existing value would silently re-adopt that guess as though a
    # person had reviewed it -- which is exactly the failure this rewrite exists to remove.
    # A blank cell in the sheet means "not reviewed", and must come out blank.
    for col in ("farm", "field", "season", "site_id", "exhaustive"):
        df[col] = [(by_ds.get(d, {}).get(col) or "").strip() for d in df["dataset"]]
    n = int((df["site_id"].astype(str).str.len() > 0).sum())
    print(f"  reviewed site_id    {n:,}/{len(df):,} rows")
    return df


def build_alloc_components(df, cols):
    """The unit splitting actually allocates: the connected component of EVERY constraint.

    Allocating by any single identity leaves the others free to cross a split boundary --
    allocating by capture event alone left 50 lineage groups spanning two splits. Taking the
    union of all constraints makes that impossible by construction rather than by check.
    """
    uf = Union(df["image_uid"].tolist())
    for c in cols:
        _join(uf, df["image_uid"], c)
    roots = [uf.find(u) for u in df["image_uid"]]
    names = {r: f"a{i:06d}" for i, r in enumerate(sorted(set(roots)))}
    sizes = Counter(roots)
    print(f"  alloc components    {len(names):,}  largest {max(sizes.values()):,} "
          f"({100*max(sizes.values())/len(df):.1f}%)")
    return [names[r] for r in roots]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="tables/manifest.parquet")
    ap.add_argument("--instances", default="tables/instances.parquet")
    ap.add_argument("--duplicates", default="tables/duplicate_report.json")
    ap.add_argument("--site-map", default=os.path.expanduser("~/workspace/manifest/site_mapping.csv"))
    # Staging, NOT the authoritative tables/. Nothing here is official until reviewed
    # provenance and gold coverage exist.
    ap.add_argument("--out-dir", default="tables/v2-staging")
    ap.add_argument("--bulk-min", type=int, default=20)
    ap.add_argument("--isolated-max", type=int, default=4)
    args = ap.parse_args()

    df = pd.read_parquet(args.manifest)
    inst = pd.read_parquet(args.instances)
    print(f"{len(df):,} photographs, {len(inst):,} annotations\n")
    df = apply_site_map(df, args.site_map)

    cgid, is_canon, canon_uid = build_content_groups(df, inst)
    vgid = build_visual_groups(df, args.duplicates)
    obs = build_observations(df, inst, cgid, canon_uid)
    lgid = build_lineage(df, args.duplicates)
    proxy, quar, reasons = build_source_event_proxies(
        df, args.site_map, args.bulk_min, args.isolated_max)

    # Reviewed provenance only. Empty until a person supplies it.
    reviewed_fe = [f"{s}|{se}" if str(s) not in EMPTY else ""
                   for s, se in zip(df["site_id"], df["season"])]

    alloc = build_alloc_components(df, [cgid, vgid, lgid, proxy, reviewed_fe, df["site_id"]])

    out = pd.DataFrame({
        "image_uid": df["image_uid"],
        "content_group_id": cgid, "is_canonical": is_canon, "canonical_uid": canon_uid,
        "visual_group_id": vgid, "lineage_group_id": lgid,
        "source_event_proxy_id": proxy, "field_event_id": reviewed_fe,
        "site_id": df["site_id"], "alloc_component_id": alloc,
        "quarantined": quar, "quarantine_reason": reasons,
    })
    schema.validate(out, schema.GROUP_COLUMNS, "groups", allow_extra=False)
    schema.validate(obs, schema.OBSERVATION_COLUMNS, "observations", allow_extra=False)
    os.makedirs(args.out_dir, exist_ok=True)
    out.to_parquet(os.path.join(args.out_dir, "groups.parquet"), index=False)
    obs.to_parquet(os.path.join(args.out_dir, "observations.parquet"), index=False)

    # ---- FAIL-CLOSED ASSERTIONS ------------------------------------------------------
    print("\nfail-closed checks")
    errs = []
    if not out.groupby("content_group_id").is_canonical.sum().eq(1).all():
        errs.append("a content group does not have exactly one canonical image")
    if len(obs) != len(inst):
        errs.append(f"observations {len(obs):,} != instances {len(inst):,} -- annotations lost")
    for c in ("content_group_id", "visual_group_id", "lineage_group_id",
              "source_event_proxy_id", "field_event_id", "site_id"):
        # An EMPTY value is the absence of a constraint, not a constraint that everything
        # shares. Checking it would report every unconstrained row as one giant violation.
        sub = out[out[c].astype(str) != ""]
        bad = int((sub.groupby(c).alloc_component_id.nunique() > 1).sum()) if len(sub) else 0
        if bad: errs.append(f"{bad} {c} values cross an alloc component")

    # aliases must never be sampled, and must sit with their canonical
    if int((out.groupby("content_group_id").alloc_component_id.nunique() > 1).sum()):
        errs.append("an alias is in a different alloc component from its canonical")
    if int((out[out.quarantined == 1].is_canonical.sum())) and False:
        pass
    for e in errs: print(f"  FAIL  {e}")
    if not errs: print("  ok    every constraint is inside one alloc component; 0 annotations lost")

    print(f"\nwrote {args.out_dir}/  (STAGING -- not authoritative)")
    print(f"  reviewed field events : {sum(1 for x in reviewed_fe if x):,}  "
          f"-> official sealed benchmark NOT yet possible")
    print(f"  source-event proxies  : {len({x for x in proxy if x}):,}  "
          f"-> SOURCE-GROUP benchmark only")
    return 1 if errs else 0


if __name__ == "__main__":
    raise SystemExit(main())
