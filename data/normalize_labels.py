#!/usr/bin/env python3
"""
normalize_labels.py -- STEP 4: ONE NAME PER PLANT
=================================================

THE PROBLEM
    212 datasets were annotated by different people at different times. Nobody agreed on
    spelling. The same plant appears in the data as:

        Garlic    garlic    GARLIC    garlic_1    garlic_2    garlics    Garlic Plant

    To a computer those are seven completely unrelated categories. Left alone, the model
    learns seven weak concepts instead of one strong one.

    There are 155 distinct label strings in this corpus. Some of them are the same plant.
    Some are genuinely different plants. Some are not plants at all -- "Ripe" is a state of
    ripeness, not a species, and "z11" is almost certainly a plot code.

WHY THIS SCRIPT DOES NOT DECIDE
    It PROPOSES groupings and prints them for a human to check. It merges nothing on its own.

    Here is the asymmetry that forces this. Suppose the script wrongly merges two different
    plants into one name. What happens? The model quietly learns a blurred, muddled category.
    It scores a bit worse. There is NO error message, no failed check, no anomaly in the data
    -- just a slightly worse number, six weeks later, with nothing pointing at the cause.

    An automatic merge that is wrong is invisible. That is why a person signs off on it.

    The script goes further and REFUSES to run the apply step unless every row has been filled
    in AND has a name in `approved_by`. It will not guess, and it will not accept an anonymous
    decision.

THE TWO-PART WORKFLOW
    # 1. The script proposes -- writes a CSV of every label with a suggested grouping
    python -m training.data.normalize_labels --instances tables/instances.parquet \
                                             --propose tables/label_review.csv

    # 2. A PERSON opens label_review.csv in a spreadsheet and fills in three columns:
    #       label_canon  -- what this should be called
    #       decision     -- keep / merge / drop
    #       approved_by  -- their name
    #    The script has already written a `suggested_canon` and a `note` next to each row to
    #    make this quick, but the person can overrule any of it.

    # 3. The script applies the approved decisions
    python -m training.data.normalize_labels --instances tables/instances.parquet \
                                             --approved tables/label_review.csv \
                                             --out tables/label_ontology.parquet

CURRENT STATUS: step 2 is where the pipeline is parked. 155 of 155 rows await a human.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from collections import Counter

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from training.data import schema  # noqa: E402


# =====================================================================================
# TWO GUARDS THAT STOP THE PROPOSER FROM SUGGESTING NONSENSE
# =====================================================================================

# GUARD 1 -- the short-code guard.
# The proposer's basic trick is to strip trailing digits, so "garlic_1", "garlic_2" and
# "garlic_10" all collapse to "garlic". Correct, and it handles most of the mess.
#
# But apply the same trick to "z11", "z12", "z13" and they ALL collapse to "z" -- three
# separate field plots merged into one imaginary plant. Same for "CNG1", "CNG2", "CNG3".
#
# The fix: only strip trailing digits when what remains is at least 4 characters long. A real
# plant name is longer than 3 letters; a plot code usually isn't. Below the threshold the
# digits are kept and the label is FLAGGED for a person instead of being merged silently.
MIN_STEM = 4

# GUARD 2 -- the modifier guard.
# These words describe a STATE or a CONDITION, not a species. If an annotator labelled
# something "Ripe", we have no idea what plant it is -- only how ripe it was. Merging those
# into a species is meaningless. The proposer flags them for a person to resolve.
MODIFIERS = {"ripe", "unripe", "green", "dry", "dried", "fresh", "old", "young", "new",
             "large", "small", "big", "cut", "uncut", "damaged", "healthy", "dead"}


def norm(name: str) -> str:
    """Reduce a label to a spelling-insensitive KEY, used only for grouping candidates.

    Walk through with "Garlic Plant-2" as the example:

        "Garlic Plant-2"
        -> .strip().lower()               "garlic plant-2"   drop spaces, force lowercase
        -> [\\s\\-]+ becomes "_"           "garlic_plant_2"   spaces and hyphens both -> _
        -> drop anything not a-z0-9_      "garlic_plant_2"   removes stray punctuation
        -> drop trailing "_2"             "garlic_plant"     IF at least MIN_STEM long
        -> drop a trailing "s"            "garlic_plant"     IF longer than 3 chars

    Two labels with the same key are CANDIDATES to be the same thing. Only candidates -- the
    decision is still a person's.

    The final `or "unknown"` catches the case where a label was pure punctuation and everything
    got stripped away, leaving an empty string. An empty key would silently swallow every other
    empty one into a single bogus group.
    """
    s = str(name).strip().lower()
    s = re.sub(r"[\s\-]+", "_", s)      # spaces and hyphens both become underscores
    s = re.sub(r"[^a-z0-9_]", "", s)    # remove anything that isn't a letter, digit or _

    stripped = re.sub(r"_?\d+$", "", s) # what it would look like without trailing digits
    if len(stripped) >= MIN_STEM:       # GUARD 1: only accept if enough is left
        s = stripped

    # Singular/plural: "garlics" -> "garlic". The length test stops "gas" becoming "ga".
    s = re.sub(r"s$", "", s) if len(s) > 3 else s
    return s or "unknown"


def propose(inst: pd.DataFrame, out_path: str) -> int:
    """Write the review CSV: every label, its count, a suggestion, and a warning note."""
    # How many times each raw label was actually used. Counts matter: a label used 18,195
    # times deserves more of the reviewer's attention than one used twice.
    counts = Counter(inst["label_raw"].astype(str))

    # Bucket the labels by their normalised key -- these are the candidate groups.
    groups: dict[str, list[tuple[str, int]]] = {}
    for raw, n in counts.items():
        groups.setdefault(norm(raw), []).append((raw, n))

    rows = []
    # Sort groups by total usage, biggest first, so the reviewer sees the important ones at
    # the top of the spreadsheet. `key=lambda kv: -sum(...)` sorts DESCENDING (the minus sign).
    for key, members in sorted(groups.items(), key=lambda kv: -sum(n for _, n in kv[1])):
        members.sort(key=lambda t: -t[1])     # within a group, most-used spelling first
        suggested = members[0][0]             # ...and that becomes the suggested name

        for raw, n in members:
            rows.append({
                "label_raw": raw,
                "count": n,
                "suggested_canon": suggested,
                # ---- the three columns a PERSON must fill in -----------------------------
                "label_canon": "",     # what it should be called
                "decision": "",        # keep | merge | drop
                "domain": "",          # optional grouping, e.g. "cereal"
                "approved_by": "",     # who decided -- required, no anonymous decisions
                # ---- an automatic warning to speed the review up ---------------------------
                "note": (
                    "modifier, not a species?" if norm(raw) in MODIFIERS else
                    # fullmatch(r"[a-z]{1,3}\d+") = 1-3 letters then digits, e.g. "z11", "cng1"
                    ("SHORT CODE -- separate plots, not spellings?"
                     if re.fullmatch(r"[a-z]{1,3}\d+", norm(raw) or "") else
                     (f"{len(members)} spellings share this key" if len(members) > 1
                      else ""))
                ),
            })

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline="") as fh:
        # newline="" is required by Python's csv module on all platforms -- without it you get
        # a blank line between every row on Windows.
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    multi = sum(1 for m in groups.values() if len(m) > 1)
    mods = [k for k in groups if k in MODIFIERS]
    print(f"  {len(counts):,} distinct raw labels -> {len(groups):,} spelling groups")
    print(f"  {multi:,} groups contain more than one spelling")
    if mods:
        print(f"  {len(mods)} look like modifiers rather than species: {sorted(mods)[:8]}")
    print(f"\nwrote {out_path}")
    print("  A PERSON now fills label_canon, decision and approved_by, then re-runs with")
    print("  --approved. Nothing is merged until then.")
    return 0


def apply(inst: pd.DataFrame, approved_path: str, out_path: str, inst_out: str) -> int:
    """Apply a REVIEWED CSV: write the official vocabulary and update every object's label."""
    rows = list(csv.DictReader(open(approved_path)))

    # ---- REFUSAL 1: every row must be filled in -------------------------------------------
    # If even one label is undecided, we stop. Half-applying a vocabulary would leave some
    # objects with a canonical name and some without, and nothing downstream could tell which
    # was which. Better to stop and say exactly which row is missing.
    unfilled = [r for r in rows if not (r.get("label_canon") or "").strip()
                or not (r.get("decision") or "").strip()]
    if unfilled:
        # SystemExit stops the script with a message and a non-zero exit code, which means an
        # automated pipeline running this will halt too, instead of carrying on.
        raise SystemExit(
            f"{len(unfilled)} of {len(rows)} rows are not filled in "
            # !r prints the value with quotes around it, so a trailing space is visible
            f"(label_canon and decision are required). First: {unfilled[0]['label_raw']!r}. "
            f"This step refuses to guess.")

    # ---- REFUSAL 2: somebody must own each decision ----------------------------------------
    # A vocabulary with no author cannot be questioned later. If in three months a merge looks
    # wrong, you need to know who to ask.
    missing_approver = [r for r in rows if not (r.get("approved_by") or "").strip()]
    if missing_approver:
        raise SystemExit(f"{len(missing_approver)} rows have no approved_by. "
                         f"The mapping must record who decided it.")

    # ---- Write the official vocabulary -----------------------------------------------------
    now = pd.Timestamp.utcnow().isoformat()   # UTC, so timestamps are comparable across machines
    ont = pd.DataFrame([{
        "label_raw": r["label_raw"],
        "label_canon": r["label_canon"].strip(),
        "domain": (r.get("domain") or "").strip(),
        "decision": r["decision"].strip(),
        "approved_by": r["approved_by"].strip(),
        "approved_at": now,
    } for r in rows])
    schema.validate(ont, schema.ONTOLOGY_COLUMNS, "label_ontology")
    ont.to_parquet(out_path, index=False)

    # ---- Apply it to all 879,253 objects ---------------------------------------------------
    mapping = dict(zip(ont["label_raw"], ont["label_canon"]))
    dropped = {r["label_raw"] for r in rows if r["decision"].strip() == "drop"}

    inst = inst.copy()   # work on a copy, so the caller's table is never modified underneath it
    # .map() replaces each raw label with its canonical name in one pass over the column.
    # .fillna("") handles a label in the data that somehow isn't in the CSV -- it gets an empty
    # canonical name, which validate_annotations.py will then catch and report.
    inst["label_canon"] = inst["label_raw"].map(mapping).fillna("")

    before = len(inst)
    # `~` means NOT. So: keep the rows whose label is NOT in the dropped set.
    inst = inst[~inst["label_raw"].isin(dropped)]
    inst.to_parquet(inst_out, index=False)

    print(f"wrote {out_path}  {len(ont):,} label decisions")
    print(f"  canonical names     {ont['label_canon'].nunique():,}")
    print(f"  dropped labels      {len(dropped)} ({before - len(inst):,} objects removed)")
    print(f"  instances updated   {inst_out}  {len(inst):,} objects")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--instances", default="tables/instances.parquet")
    ap.add_argument("--propose", default="", help="write the review CSV here")
    ap.add_argument("--approved", default="", help="read a filled-in review CSV from here")
    ap.add_argument("--out", default="tables/label_ontology.parquet")
    args = ap.parse_args()

    inst = pd.read_parquet(args.instances)
    print(f"{len(inst):,} marked objects")
    if args.propose:
        return propose(inst, args.propose)
    if args.approved:
        return apply(inst, args.approved, args.out, args.instances)
    # Neither mode chosen -- print the usage message and exit. Doing nothing silently would
    # look like success.
    ap.error("give either --propose or --approved")


if __name__ == "__main__":
    raise SystemExit(main())
