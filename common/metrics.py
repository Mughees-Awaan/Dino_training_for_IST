#!/usr/bin/env python3
"""
metrics.py -- how we score a model. Detection F1, and nothing else gates.

THE PROJECT'S HARDEST-WON LESSON
    Point-AUC INVERTED THE RANKING FOUR SEPARATE TIMES on this project's own data. The
    worst case, measured on the poa field:

        shufflenet dense    AUC 0.9063    detection F1 0.6154
        patch @0.5x         AUC 0.9032    detection F1 0.0904

    Two configurations with essentially identical AUC, differing by SEVEN TIMES in F1.

    Why: AUC scores POINTS -- can the model rank a plant pixel above a soil pixel? The
    product must separate INSTANCES -- how many distinct plants did it find, and how many
    things did it flag that were not plants? A model can rank beautifully and still merge
    every plant in a dense patch into one blob.

    RULE: AUC may be logged. It may NEVER gate a decision.
"""

from __future__ import annotations

import numpy as np

from training.common.matching import greedy_match


def detection_prf(pred_xy, gt_xy, radius: float) -> dict:
    """Precision, recall and F1 from one-to-one matched points.

        precision  of everything I flagged, what fraction was real?
        recall     of everything real, what fraction did I find?
        F1         their harmonic mean -- punishes a model that wins one by sacrificing
                   the other, which a plain average would not.
    """
    pairs, fp, fn = greedy_match(pred_xy, gt_xy, radius)
    tp = len(pairs)
    precision = tp / (tp + len(fp)) if (tp + len(fp)) else 0.0
    recall = tp / (tp + len(fn)) if (tp + len(fn)) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"tp": tp, "fp": len(fp), "fn": len(fn),
            "precision": precision, "recall": recall, "f1": f1}


def sweep_threshold(scores, xy, gt_xy, radius: float, n: int = 24) -> dict:
    """Find the best achievable F1 by trying many confidence cut-offs.

    A model outputs confidences, not decisions. Reporting F1 at one arbitrary cut-off
    measures the cut-off as much as the model, so we sweep and report the best -- and
    also return the threshold that achieved it, because the product has to ship one.

    Quantiles rather than evenly spaced values: score distributions are wildly skewed, so
    fixed steps would waste most of the sweep in an empty range.
    """
    scores = np.asarray(scores, dtype=np.float64)
    if scores.size == 0:
        return {"f1": 0.0, "threshold": 0.0, "precision": 0.0, "recall": 0.0}
    best = {"f1": -1.0}
    for t in np.quantile(scores, np.linspace(0.50, 0.999, n)):
        keep = scores >= t
        r = detection_prf(np.asarray(xy)[keep], gt_xy, radius)
        if r["f1"] > best["f1"]:
            best = {**r, "threshold": float(t)}
    return best


def macro_by_group(per_group: dict[str, dict]) -> dict:
    """Average F1 ACROSS FIELDS, not across pooled detections.

    Pooling every detection into one number lets the biggest field dominate: one site with
    40,000 plants would drown out twenty sites with 200 each, and the headline would
    describe that one site.

    Averaging per field gives every field one vote. This is the project's primary
    reporting statistic.
    """
    if not per_group:
        return {"macro_f1": 0.0, "n_groups": 0}
    f1s = [v["f1"] for v in per_group.values()]
    return {"macro_f1": float(np.mean(f1s)), "n_groups": len(f1s),
            "min_f1": float(np.min(f1s)), "max_f1": float(np.max(f1s)),
            "std_f1": float(np.std(f1s))}
