#!/usr/bin/env python3
"""
validate_annotations.py -- STEP 5: THE GATE
===========================================

WHAT A "GATE" IS
    Every check in this script either PASSES or STOPS THE PIPELINE. There is no third option
    where something is wrong but we carry on anyway.

WHY IT IS BUILT THAT WAY
    A warning that is printed and ignored is worse than no warning at all. It scrolls off the
    screen, everybody forgets it, and six weeks later there is a wrong number in a report with
    nothing pointing back at the cause. This is not hypothetical -- it is the failure mode this
    entire pipeline has been bitten by, more than once.

    So: --fail-on-error makes the script exit with code 1. Any automated pipeline running it
    stops dead. Nothing downstream gets to touch bad data.

WHAT IT CHECKS, IN FOUR GROUPS
    structural  Are the tables even self-consistent? Does every marked object belong to a
                photograph that actually exists?
    geometry    Is every mark physically inside the photograph it claims to be in? Are boxes
                a sensible size?
    labels      Does every object carry an approved name?
    lineage     Does every photograph have a family, and do the families look sane?

THE GEOMETRY CHECK EARNED ITS PLACE
    "object centre inside its photograph" once reported 2,548 failures, and it was briefly
    written up as a defect in the source data. It was not. It was OUR bug -- the frame-ordering
    mistake in build_manifest.py, attaching annotations to the wrong photographs. After the fix
    the count is 0. The check did exactly its job: it noticed that something was impossible.

FATAL vs WARNING
    A few checks are marked fatal=False. Those describe things we cannot control -- for
    example, some photographs have no recorded width or height, so we simply cannot bounds-
    check their marks. That is a limitation to report, not a defect to halt on.

    python -m training.data.validate_annotations --fail-on-error
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from training.data import schema  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="tables/manifest.parquet")
    ap.add_argument("--instances", default="tables/instances.parquet")
    ap.add_argument("--lineage", default="tables/lineage_groups.parquet")
    ap.add_argument("--ontology", default="tables/label_ontology.parquet")
    ap.add_argument("--fail-on-error", action="store_true",
                    help="exit 1 if any fatal check fails, so a pipeline halts")
    ap.add_argument("--max-report", type=int, default=5,
                    help="how many failures to print in detail")
    args = ap.parse_args()

    man = pd.read_parquet(args.manifest)
    inst = pd.read_parquet(args.instances)
    schema.validate(man, schema.MANIFEST_COLUMNS, "manifest")
    schema.validate(inst, schema.INSTANCE_COLUMNS, "instances")
    print(f"manifest {len(man):,} photographs   instances {len(inst):,} objects\n")

    errors: list[str] = []
    warns: list[str] = []

    def check(name, bad_count, total, detail="", fatal=True):
        """Run one check, print one aligned line, remember it if it failed.

        Every check is phrased so that ZERO IS GOOD -- `bad_count` is always the number of
        things that are WRONG. That way a passing pipeline is a column of zeros, and anything
        non-zero jumps out. It also means a new check can be added without inventing a new
        convention for what "good" looks like.
        """
        ok = bad_count == 0
        mark = "ok  " if ok else ("FAIL" if fatal else "warn")
        pct = f"{100 * bad_count / max(total, 1):.2f}%"   # max(...,1) avoids divide-by-zero
        print(f"  {mark}  {name:<44} {bad_count:>8,} / {total:,}  {pct}  {detail}")
        if not ok:
            (errors if fatal else warns).append(f"{name}: {bad_count:,} ({detail})")

    # =====================================================================================
    # STRUCTURAL -- are the tables internally consistent?
    # =====================================================================================
    # Every photograph must appear exactly once. rows minus distinct ids = number of repeats.
    check("manifest image_uid unique", len(man) - man["image_uid"].nunique(), len(man))

    # Every marked object must point at a photograph we actually have. An "orphan" instance
    # would silently disappear from every later step -- counted in the totals, never used.
    known = set(man["image_uid"])
    check("every instance belongs to a photograph",
          int((~inst["image_uid"].isin(known)).sum()), len(inst))

    # =====================================================================================
    # GEOMETRY -- is every mark physically possible?
    # =====================================================================================
    # Attach each object's photograph dimensions to it, so we can compare. .join(on=...) is a
    # lookup: for each instance row, fetch the width/height of its image_uid.
    dims = man.set_index("image_uid")[["width", "height"]]
    j = inst.join(dims, on="image_uid")

    # Only photographs whose size we actually know can be bounds-checked. (-1 means unknown.)
    measured = j[(j["width"] > 0) & (j["height"] > 0)]

    # A mark outside its own photograph is impossible. If this fires, either the source data
    # is broken or -- far more likely, as we learned -- WE attached it to the wrong photograph.
    outside = ((measured["x"] < 0) | (measured["y"] < 0)
               | (measured["x"] > measured["width"]) | (measured["y"] > measured["height"]))
    check("object centre inside its photograph", int(outside.sum()), len(measured))

    # Boxes and polygons must enclose actual area. Dots legitimately have w=h=0, so exclude them.
    boxes = measured[measured["shape_type"] != "points"]
    degenerate = (boxes["w"] <= 0) | (boxes["h"] <= 0)
    check("box has positive size", int(degenerate.sum()), len(boxes))

    # A box cannot be bigger than the photograph containing it. The 1.01 gives 1% slack for
    # rounding in the polygon area formula -- without it, a polygon tracing the exact border
    # of the image would fail on a floating-point rounding error.
    too_big = boxes["area_px"] > (boxes["width"] * boxes["height"] * 1.01)
    check("box no larger than its photograph", int(too_big.sum()), len(boxes))

    # Not fatal: we cannot bounds-check what we cannot measure. Worth reporting, not halting.
    unmeasured = int(((man["width"] <= 0) | (man["height"] <= 0)).sum())
    check("photograph dimensions known", unmeasured, len(man),
          "cannot bounds-check these", fatal=False)

    # =====================================================================================
    # LABELS -- only checkable once a person has approved the vocabulary
    # =====================================================================================
    if os.path.exists(args.ontology):
        ont = pd.read_parquet(args.ontology)
        schema.validate(ont, schema.ONTOLOGY_COLUMNS, "label_ontology")
        approved = set(ont["label_raw"])
        # A label in the data that nobody reviewed means the review was done against an older
        # version of the data.
        check("every label appears in the approved list",
              int((~inst["label_raw"].isin(approved)).sum()), len(inst))
        check("every instance carries a canonical label",
              int((inst["label_canon"].fillna("") == "").sum()), len(inst))
    else:
        # Not an error -- just not done yet. This is exactly where the pipeline is parked today.
        warns.append("no label ontology yet; label checks skipped")
        print(f"  warn  {'label ontology not built yet':<44} "
              f"{'--':>8}          run normalize_labels first")

    # =====================================================================================
    # LINEAGE -- do the families look sane?
    # =====================================================================================
    if os.path.exists(args.lineage):
        lin = pd.read_parquet(args.lineage)
        schema.validate(lin, schema.LINEAGE_COLUMNS, "lineage_groups")

        # A photograph with no family cannot be assigned to a split at all.
        check("every photograph has a family",
              int((~man["image_uid"].isin(set(lin["image_uid"]))).sum()), len(man))

        n_groups = lin["leakage_group_id"].nunique()
        biggest = lin["leakage_group_id"].value_counts().iloc[0]

        # This check exists BECAUSE of the rank-6 over-merging bug in build_lineage_groups.
        # That bug welded 76% of the corpus into one family. `int(True)` is 1 and `int(False)`
        # is 0, so passing the boolean as bad_count out of a total of 1 gives a clean pass/fail.
        check("no family swallows the corpus", int(biggest > 0.5 * len(man)), 1,
              f"largest holds {100 * biggest / len(man):.1f}%")

        # Below ~30 independent families, the statistical test used to decide whether one model
        # genuinely beats another has too few independent units to say anything.
        check("at least 30 families for the paired test", int(n_groups < 30), 1,
              f"{n_groups:,} families")
    else:
        warns.append("no lineage table yet")

    # =====================================================================================
    # COVERAGE -- a NOTE, deliberately not a check
    # =====================================================================================
    # Only 3,038 of 82,099 photographs are marked "completed" in CVAT. It is very tempting to
    # treat that as "only 3.6% of the data is good" -- but task_status describes the annotation
    # TEAM'S WORKFLOW, not the data's quality. A task can be perfectly annotated and sit in
    # "annotation" forever because nobody clicked the button.
    #
    # So this prints a note and, at most, a warning. Turning it into a fatal check would throw
    # away 78,483 photographs on the strength of a button nobody pressed.
    reviewed = int((man["task_status"] == "completed").sum())
    print(f"\n  note  photographs marked 'completed': {reviewed:,} of {len(man):,} "
          f"({100 * reviewed / len(man):.1f}%)")
    if reviewed < 0.05 * len(man):
        warns.append(f"only {reviewed:,} photographs are marked completed; the scoring set "
                     f"may be too small to gate on")

    # =====================================================================================
    # VERDICT
    # =====================================================================================
    print()
    for w in warns:
        print(f"  WARNING  {w}")
    if errors:
        print(f"\n{len(errors)} CHECK(S) FAILED:")
        for e in errors[: args.max_report]:
            print(f"    {e}")
        if args.fail_on_error:
            # Exit code 1 = failure. A shell script or CI job running this will stop here.
            print("\nPipeline halted. Nothing downstream may run against this data.")
            return 1
    else:
        print("\nAll structural checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
