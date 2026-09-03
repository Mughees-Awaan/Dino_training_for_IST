#!/usr/bin/env python3
"""
matching.py -- pair up predictions with ground truth, one to one.

WHY THIS IS NOT OBVIOUS
    The model outputs a list of points it thinks are plants. Ground truth is a list of
    points that really are plants. To score the model you must decide WHICH prediction
    corresponds to WHICH real plant.

    Get this wrong and every metric downstream is wrong. Two traps:

    1. ONE-TO-MANY. If a prediction is allowed to match several plants, a single lucky
       detection dropped in a dense patch can "find" ten plants at once. Scores soar and
       the model has done nothing.

    2. GREEDY BY THE WRONG ORDER. Matching in arbitrary order lets a mediocre prediction
       claim a plant that a better one was closer to, pushing the better prediction to
       count as a false positive.

    So: sort candidate pairs by distance, take the closest first, and once a prediction
    or a plant is used it is OFF THE TABLE.

WHY DISTANCE AND NOT OVERLAP
    This product outputs POINTS -- the user clicks dots, the tool returns dots. There are
    no boxes to overlap. A prediction counts as correct if it lands within `radius` pixels
    of a real plant.
"""

from __future__ import annotations

import numpy as np


def greedy_match(pred_xy, gt_xy, radius: float):
    """Match predictions to ground truth, closest pair first, one to one.

    Returns (matched_pairs, unmatched_pred_idx, unmatched_gt_idx).
        matched_pairs        list of (prediction index, ground-truth index)
        unmatched_pred_idx   predictions that matched nothing -> FALSE POSITIVES
        unmatched_gt_idx     real plants nothing found        -> FALSE NEGATIVES
    """
    pred = np.asarray(pred_xy, dtype=np.float64).reshape(-1, 2)
    gt = np.asarray(gt_xy, dtype=np.float64).reshape(-1, 2)
    if len(pred) == 0 or len(gt) == 0:
        return [], list(range(len(pred))), list(range(len(gt)))

    # Distance from every prediction to every plant.
    # pred[:, None, :] makes a column, gt[None, :, :] makes a row; numpy broadcasting
    # then produces the full (n_pred x n_gt) grid in one operation.
    d = np.linalg.norm(pred[:, None, :] - gt[None, :, :], axis=2)

    # Consider only pairs close enough to possibly be the same plant.
    pi, gi = np.nonzero(d <= radius)
    if len(pi) == 0:
        return [], list(range(len(pred))), list(range(len(gt)))

    # THE ORDER IS THE ALGORITHM: closest pair first.
    order = np.argsort(d[pi, gi], kind="stable")

    used_p, used_g, pairs = set(), set(), []
    for k in order:
        p, g = int(pi[k]), int(gi[k])
        if p in used_p or g in used_g:
            continue                      # one-to-one: both sides are consumed
        used_p.add(p); used_g.add(g)
        pairs.append((p, g))

    return (pairs,
            [i for i in range(len(pred)) if i not in used_p],
            [j for j in range(len(gt)) if j not in used_g])
