#!/usr/bin/env python3
"""
render_episodes.py -- STEP 8: DRAW THE SESSIONS SO A PERSON CAN LOOK AT THEM
============================================================================

THIS SCRIPT IS THE CHEAPEST DEFECT DETECTOR IN THE WHOLE PROGRAMME.

    Everything up to here produces numbers. Numbers can be perfectly self-consistent and
    completely wrong. This script produces PICTURES, and the human eye catches in two seconds
    what a table of statistics cannot show at all:

        - x and y swapped                    -> circles form a mirrored pattern
        - an off-by-one in the frame mapping -> circles land on bare soil
        - a label attached to the wrong plant -> the title says "garlic", the circle is on maize
        - empty sessions                     -> a picture with no circles at all

IT HAS ALREADY PAID FOR ITSELF, ONCE, DECISIVELY
    The frame-ordering bug in build_manifest.py -- every annotation attached to the wrong
    photograph -- was found HERE. Not by a test, not by a check, not by a statistic. By
    rendering episodes, looking at them, and noticing the amber circles were sitting on empty
    dirt. Every table in the pipeline was internally consistent at the time.

WHAT IT DRAWS (default mode)
    A side-by-side pair for each episode:
        LEFT   the photograph the user "clicked" in, with amber circles on the clicks
        RIGHT  a DIFFERENT photograph, with green circles on the targets to be found

    Confirming a session is correct takes about a second: are the amber circles on the plant
    named in the title, and are the green circles on the same kind of plant, elsewhere?

    python -m training.data.render_episodes --n 100 --out review/episodes

WHAT IT DRAWS (--audit-exhaustive mode)
    A completely different question. See audit_exhaustive() below.
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2               # OpenCV -- image reading, drawing, resizing
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from training.runtime import preprocessing  # noqa: E402

# OpenCV stores colours as (Blue, Green, Red) -- NOT the usual RGB order. This trips up
# everyone once. (0, 200, 255) is no blue, lots of green, full red = amber.
SUPPORT = (0, 200, 255)     # amber -- the clicks
QUERY = (0, 255, 0)         # green -- the targets to find


def panel(root, where, uid, xs, ys, colour, title, side=760):
    """Draw ONE photograph with circles on the given points, and a title bar.

    `where` maps image_uid -> (archive, member), so we can find the photograph inside its zip.
    `side` is the size we shrink the long edge to -- these are drone photographs, often
    3840 pixels wide, and a review sheet needs them small enough to see at a glance.
    """
    archive, member = where[uid]
    # Reads the photograph straight out of the zip. Nothing was ever extracted to disk.
    rgb = preprocessing.read_image(root, archive, member)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)   # OpenCV draws in BGR order

    h, w = bgr.shape[:2]
    # One scale factor for BOTH axes -- computed from the LONGER edge so the whole image fits.
    # Using separate x and y factors would stretch the picture and move the circles off their
    # plants, which would make this tool report a bug that does not exist.
    s = side / max(h, w)
    bgr = cv2.resize(bgr, (int(w * s), int(h * s)))

    # np.atleast_1d wraps a lone number into a list, so a single-point episode does not crash
    # the loop. Coordinates are scaled by the SAME s used for the image.
    for x, y in zip(np.atleast_1d(xs), np.atleast_1d(ys)):
        cv2.circle(bgr, (int(float(x) * s), int(float(y) * s)),
                   9,          # radius, in pixels
                   colour,
                   2)          # line thickness. NOT filled -- a filled dot would hide the
                               # very plant the reviewer is trying to check.

    # A solid black strip along the top, then the title on it. The -1 thickness means "fill".
    # Without the strip, white text vanishes over pale soil and dark text over shadow.
    cv2.rectangle(bgr, (0, 0), (bgr.shape[1] - 1, 26), (0, 0, 0), -1)
    cv2.putText(bgr, title, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 1, cv2.LINE_AA)
    return bgr


def audit_exhaustive(args) -> int:
    """A DIFFERENT job: are ALL the plants in a photograph marked, or only some?

    THE QUESTION NO METADATA CAN ANSWER
        There is no field anywhere in the data that says "every object in this photograph is
        marked". You can only find out by looking.

    WHY IT MATTERS ASYMMETRICALLY
        For TRAINING, partial marking is survivable. The model learns from the examples that
        are there and simply sees fewer of them.

        For SCORING it is fatal, and in a way that gets the answer BACKWARDS. Suppose a
        photograph holds ten plants and only six are marked. A model that correctly finds all
        ten is charged with FOUR FALSE POSITIVES -- punished for being right. Meanwhile a worse
        model that finds only the six marked ones scores perfectly.

        Score on partially-marked data and you systematically select the worse model.

    WHY WE CANNOT USE task_status FOR THIS
        We measured it. Photographs marked `annotation` carry MORE marked objects than ones
        marked `completed` (median 5 against 4). The field records the annotation team's
        workflow, not the data's completeness. It answers a different question entirely.

    So: sample photographs, circle every marked object, and put them in front of a person with
    one question -- "is anything NOT circled?" -- plus a CSV to record the answers in.
    """
    ep = pd.read_parquet(args.manifest)
    man = ep
    if args.dataset:
        # na=False stops a missing dataset name crashing the string match
        man = man[man["dataset"].str.contains(args.dataset, case=False, na=False)]
    man = man[man["n_annotations"] > 0]     # a photograph with nothing marked tells us nothing
    if man.empty:
        print("no annotated photographs matched")
        return 1

    inst = pd.read_parquet(args.instances)
    where = {u: (a, m) for u, a, m in zip(ep["image_uid"], ep["archive"], ep["member"])}
    os.makedirs(args.out, exist_ok=True)

    # ---- Spread the sample across datasets -------------------------------------------------
    # A plain random sample of 100 would be dominated by whichever dataset is biggest, and the
    # verdict would really only describe that one dataset. Instead take a few from each.
    #
    # Written as an explicit loop rather than groupby().apply() on purpose: pandas 3 drops the
    # grouping column inside apply(), which silently removes "dataset" from the result.
    per = max(1, args.n // max(man["dataset"].nunique(), 1))
    parts = []
    for _, d in man.groupby("dataset"):
        parts.append(d.sample(min(per, len(d)), random_state=args.seed))
    take = pd.concat(parts, ignore_index=True)
    take = take.sample(min(args.n, len(take)), random_state=args.seed)

    rows = []
    for _, r in take.iterrows():
        pts = inst[inst["image_uid"] == r["image_uid"]]
        try:
            img = panel(args.archives, where, r["image_uid"], pts["x"], pts["y"], QUERY,
                        # The title shows the status too, so the reviewer can see for
                        # themselves whether "completed" correlates with completeness.
                        f"{r['dataset'][:40]}  status={r['task_status']}  marked={len(pts)}")
        except Exception as exc:
            # One unreadable photograph must not abandon a review batch of 100.
            print(f"  [skip] {str(exc)[:60]}")
            continue
        name = f"{r['image_uid'][:12]}.jpg"
        cv2.imwrite(os.path.join(args.out, name), img)
        rows.append({"image": name, "dataset": r["dataset"], "task_status": r["task_status"],
                     "marked": len(pts),
                     # blank columns for the reviewer to fill in
                     "missed_count": "", "exhaustive": "", "reviewer": ""})

    import csv as _csv
    sheet = os.path.join(args.out, "exhaustiveness_review.csv")
    with open(sheet, "w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    print(f"wrote {len(rows)} images and {sheet}")
    print("For each image the reviewer counts plants that are NOT circled, then fills")
    print("missed_count and exhaustive (yes/no). Datasets where anything is missed cannot")
    print("be used for scoring -- only for training.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", default="tables/episodes/train.parquet")
    ap.add_argument("--manifest", default="tables/manifest.parquet")
    ap.add_argument("--archives", default=os.path.expanduser("~/workspace/gdrive_datasets"))
    ap.add_argument("--out", default="review/episodes")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--audit-exhaustive", action="store_true",
                    help="sample annotated photographs for a completeness review")
    ap.add_argument("--instances", default="tables/instances.parquet")
    ap.add_argument("--dataset", default="", help="restrict the audit to matching datasets")
    args = ap.parse_args()

    if args.audit_exhaustive:
        return audit_exhaustive(args)

    # ---- Default mode: draw episodes side by side -------------------------------------------
    ep = pd.read_parquet(args.episodes)
    man = pd.read_parquet(args.manifest)
    where = {u: (a, m) for u, a, m in zip(man["image_uid"], man["archive"], man["member"])}
    os.makedirs(args.out, exist_ok=True)

    # random_state=seed makes the SAME 100 episodes come out every run, so a reviewer can
    # re-run after a fix and compare like with like.
    take = ep.sample(min(args.n, len(ep)), random_state=args.seed)
    written = 0
    for _, r in take.iterrows():
        try:
            left = panel(args.archives, where, r["support_uid"], r["support_x"], r["support_y"],
                         SUPPORT, f"CLICKS  {r['label']}  n={r['n_support']}")
            right = panel(args.archives, where, r["query_uid"], r["query_x"], r["query_y"],
                          QUERY, f"FIND    {r['label']}  n={r['n_query']}"
                                 # Flag same-family episodes in the title, so the reviewer
                                 # knows when the two photographs are expected to look alike.
                                 + ("  [same family]" if r["same_family"] else ""))
        except Exception as exc:
            print(f"  [skip] {r['episode_id']}: {str(exc)[:70]}")
            continue

        # The two photographs may have different shapes. To place them side by side, pad the
        # shorter one with dark grey down to the taller one's height.
        h = max(left.shape[0], right.shape[0])
        pad = lambda im: cv2.copyMakeBorder(im, 0, h - im.shape[0], 0, 0,  # noqa: E731
                                            cv2.BORDER_CONSTANT, value=(20, 20, 20))
        # np.hstack glues arrays left-to-right. The middle piece is a 6-pixel grey divider, so
        # it is obvious where one photograph ends and the other begins.
        cv2.imwrite(os.path.join(args.out, f"{r['episode_id']}.jpg"),
                    np.hstack([pad(left), np.full((h, 6, 3), 40, np.uint8), pad(right)]))
        written += 1

    print(f"wrote {written} session images to {args.out}/")
    print("A PERSON now looks at all of them. Confirm the amber circles sit on the plant")
    print("named in the title, and that the green circles do too, in a different photograph.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
