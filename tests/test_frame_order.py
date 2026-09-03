#!/usr/bin/env python3
"""
test_frame_order.py -- REGRESSION TEST FOR THE WORST BUG IN THIS PIPELINE
=========================================================================

WHAT A "REGRESSION TEST" IS
    A test written AFTER a bug has been found and fixed. Its job is not to find new problems.
    Its job is to make sure this SPECIFIC problem can never come back unnoticed.

THE BUG IT GUARDS
    Inside a CVAT archive, annotations are attached to photographs BY NUMBER: "frame 7 has a
    garlic plant at (412, 88)". So the code must know which photograph is frame 7.

    CVAT numbers frames by their order in `data/manifest.jsonl`.
    The zip file stores its members in a completely different order.

    An early version of build_manifest.py used the zip's order. Measured on one sampled task,
    46 of 48 positions differed -- so effectively every annotation in that task was attached
    to the wrong photograph.

WHY IT SURVIVED SO LONG
    Nothing errored. Every table built successfully. Every count was plausible. Every
    structural check passed. The data was perfectly self-consistent and completely wrong.

    It was found by render_episodes.py -- by DRAWING the clicks onto the photographs and
    noticing they were sitting on bare soil.

THE THREE TESTS, AND WHY THERE ARE THREE

    1. test_orders_really_differ
       Proves the bug is still POSSIBLE. If the two orders ever agree, this test fails on
       purpose, to tell you the premise is gone and the complexity is no longer needed.

    2. test_build_manifest_uses_manifest_order
       Proves the FIX is in place. Checks frame_index directly against manifest.jsonl.

    3. test_annotations_land_inside_their_photograph
       Proves the SYMPTOM is gone. Wrong frame numbers push coordinates outside their image.

    Testing the cause and the symptom separately matters: a future bug could produce the same
    symptom by a different route, and this way the two are told apart.

    python training/tests/test_frame_order.py
"""

from __future__ import annotations

import json
import os
import zipfile

ROOT = os.path.expanduser("~/workspace/gdrive_datasets")
IMG = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp")


def _orders(zpath: str, task: str):
    """Return the SAME task's photographs in both orders, for comparison.

    Returns (order_the_zip_stores_them_in, order_manifest.jsonl_declares).
    """
    with zipfile.ZipFile(zpath) as z:
        # Order 1: the zip's own table of contents -- the tempting, wrong answer.
        zip_order = [os.path.basename(i.filename) for i in z.infolist()
                     if i.filename.startswith(f"{task}/") and i.filename.lower().endswith(IMG)]

        # Order 2: manifest.jsonl -- the one CVAT actually numbers by.
        man_order = []
        try:
            raw = z.read(f"{task}/data/manifest.jsonl").decode("utf-8", "replace")
        except KeyError:
            return zip_order, []     # no manifest.jsonl in this task
        for line in raw.splitlines():
            line = line.strip()
            if line.startswith("{"):     # skip the header lines at the top of the file
                d = json.loads(line)
                if "name" in d:
                    man_order.append(os.path.basename(f"{d['name']}{d.get('extension', '')}"))
    return zip_order, man_order


def test_orders_really_differ():
    """The bug's PREMISE: the two orders genuinely disagree on real data.

    This is an unusual test -- it asserts that a problem still EXISTS. If CVAT ever changes so
    the two orders match, this fails and tells you the careful handling in build_manifest.py
    is no longer earning its keep. A test that quietly keeps passing after its reason has gone
    is how dead complexity accumulates.
    """
    z = os.path.join(ROOT, "Apple Gopro.zip")
    if not os.path.exists(z):
        return      # archives not present on this machine; skip rather than fail
    zip_order, man_order = _orders(z, "task_0")
    assert man_order, "task has no manifest.jsonl"
    n = min(len(zip_order), len(man_order))
    differing = sum(1 for a, b in zip(zip_order[:n], man_order[:n]) if a != b)
    assert differing > 0, ("zip order now matches manifest order; this test's premise is "
                           "gone and build_manifest could be simplified")


def test_build_manifest_uses_manifest_order():
    """The FIX: frame_index must equal the position in manifest.jsonl, not in the zip."""
    from training.data.build_manifest import scan_archive
    z = os.path.join(ROOT, "Apple Gopro.zip")
    if not os.path.exists(z):
        return
    rows, _ = scan_archive(z)          # run the real function on a real archive
    _, man_order = _orders(z, "task_0")
    expected = {name: i for i, name in enumerate(man_order)}   # the correct answer

    checked = mismatched = 0
    for r in rows:
        if not r["member"].startswith("task_0/"):
            continue
        want = expected.get(r["image_name"])
        if want is None:
            continue
        checked += 1
        if int(r["frame_index"]) != want:
            mismatched += 1

    # This second assertion matters as much as the first. If a future refactor renamed a
    # column, `checked` would be 0 and the mismatch test would pass VACUOUSLY -- a green tick
    # for having tested nothing at all.
    assert checked > 0, "no task_0 rows were checked"
    assert mismatched == 0, (
        f"{mismatched}/{checked} photographs carry the wrong frame number. Annotations will "
        f"attach to the wrong images and nothing will error.")


def test_annotations_land_inside_their_photograph():
    """The SYMPTOM: wrong frame numbers push coordinates outside their photograph.

    A photograph is 3840x2160 and its neighbour is 1024x1024. Attach the big one's
    annotations to the small one and many coordinates land beyond the edge -- physically
    impossible, and easy to detect.

    This is the check that once reported 2,548 failures and was briefly written up as a defect
    in the source data. It was not. It was this bug. After the fix the count is 0.
    """
    import pandas as pd
    m, i = "tables/manifest.parquet", "tables/instances.parquet"
    if not (os.path.exists(m) and os.path.exists(i)):
        return      # tables not built yet; nothing to check
    man = pd.read_parquet(m)[["image_uid", "width", "height"]]
    inst = pd.read_parquet(i)[["image_uid", "x", "y"]]
    j = inst.join(man.set_index("image_uid"), on="image_uid")
    j = j[(j["width"] > 0) & (j["height"] > 0)]     # only where we know the size
    outside = ((j["x"] < 0) | (j["y"] < 0) | (j["x"] > j["width"]) | (j["y"] > j["height"])).sum()
    assert outside == 0, (f"{outside:,} objects fall outside their photograph -- the signature "
                          f"of a frame-numbering mismatch")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    for fn in (test_orders_really_differ, test_build_manifest_uses_manifest_order,
               test_annotations_land_inside_their_photograph):
        fn()
        print(f"  PASS  {fn.__name__}")
    print("frame-order regression tests pass")
