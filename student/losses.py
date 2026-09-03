#!/usr/bin/env python3
"""
losses.py -- what the student is asked to minimise.

THREE PARTS, AND THE ABLATION THAT MATTERS
    kd        copy the teacher's descriptors      (knowledge distillation)
    episodic  do well on the click task directly
    geometry  predict centres, offsets, sizes

    The programme spec moved the KD weight 0.20 -> 1.00 and only ever ablated the KD
    SOURCE, never the WEIGHT. So the weight is a plain argument here and the sweep
    {0.2, 0.5, 1.0} is part of the plan, not an afterthought.
"""

from __future__ import annotations

from training.common.safe_imports import F, require_torch, torch


def kd_loss(student_desc, teacher_desc):
    """Make the student's descriptors point the same way as the teacher's.

    COSINE, not mean-squared-error. The ridge adapter compares descriptors by DIRECTION,
    so matching magnitude is wasted capacity -- and MSE spends most of its effort there.
    """
    require_torch("kd_loss")
    s = F.normalize(student_desc, dim=1)
    t = F.normalize(teacher_desc, dim=1).detach()
    return (1.0 - (s * t).sum(dim=1)).mean()


def centre_loss(pred, target, alpha=2.0, beta=4.0):
    """Focal loss for the centre heatmap.

    THE IMBALANCE IS EXTREME: a 128x128 grid has 16,384 cells and maybe 40 plant centres.
    Plain cross-entropy lets the model answer "no" everywhere and score 99.8%.

    Focal loss down-weights the easy negatives so the rare positives dominate the gradient.
    `beta` additionally softens cells NEAR a centre -- being one cell off is nearly right
    and should not be punished like a wild miss.
    """
    require_torch("centre_loss")
    p = torch.sigmoid(pred).clamp(1e-4, 1 - 1e-4)
    pos = (target >= 1.0).float()
    pos_loss = -((1 - p) ** alpha) * torch.log(p) * pos
    neg_loss = -((1 - target) ** beta) * (p ** alpha) * torch.log(1 - p) * (1 - pos)
    n = pos.sum().clamp(min=1.0)
    return (pos_loss.sum() + neg_loss.sum()) / n


def offset_loss(pred, target, mask):
    """Sub-cell position, scored ONLY where a plant actually is.

    Everywhere else there is no correct offset, so including those cells would train the
    head on meaningless targets. L1 rather than L2 because it is far less swayed by a
    single badly-labelled outlier.
    """
    require_torch("offset_loss")
    if mask.sum() == 0:
        return pred.sum() * 0.0
    return F.l1_loss(pred[mask], target[mask])


def total(outputs, targets, w_kd=0.5, w_ep=1.0, w_geo=1.0) -> dict:
    """Combine the parts and return every component, not just the total.

    Logging only the sum hides the failure where one term collapses to zero and the others
    silently take over.
    """
    parts = {}
    if "teacher_desc" in targets:
        parts["kd"] = kd_loss(outputs["desc_s16"], targets["teacher_desc"]) * w_kd
    if "centre" in targets:
        parts["centre"] = centre_loss(outputs["centre"], targets["centre"]) * w_geo
    if "offset" in targets and "offset_mask" in targets:
        parts["offset"] = offset_loss(outputs["offset"], targets["offset"],
                                      targets["offset_mask"]) * w_geo
    parts["total"] = sum(parts.values())
    return parts
