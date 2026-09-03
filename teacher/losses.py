#!/usr/bin/env python3
"""
losses.py -- what the teacher is actually being asked to minimise.

Two families:
  SSL losses       label-free. Can start before the label review is finished.
  Episodic loss    the click task, trained THROUGH the ridge solve.
"""

from __future__ import annotations

from training.common.safe_imports import F, require_torch, torch


def dino_loss(student_out, teacher_out, student_temp=0.1, teacher_temp=0.04, center=None):
    """Make the student's answer for one crop match the teacher's for another crop.

    THE TWO TEMPERATURES ARE NOT SYMMETRIC, ON PURPOSE.
        The teacher's distribution is made SHARP (temp 0.04) and the student's SOFT
        (temp 0.1). The student is chasing a confident target. Equal temperatures let both
        drift to the same vague answer -- which is collapse.

    THE CENTRE TERM
        Subtracting a running mean stops one prototype swallowing everything, the other
        collapse mode. Sharpening pushes toward confidence; centring pushes toward using
        the whole space. They oppose each other deliberately.
    """
    require_torch("dino_loss")
    t = teacher_out.detach()                      # no gradient into the teacher
    if center is not None:
        t = t - center
    t = F.softmax(t / teacher_temp, dim=-1)
    s = F.log_softmax(student_out / student_temp, dim=-1)
    return -(t * s).sum(dim=-1).mean()


def ibot_loss(student_masked, teacher_full, mask, student_temp=0.1, teacher_temp=0.04):
    """Predict what was hidden. Loss is computed ONLY on the masked positions.

    Including unmasked positions would let the model score well by copying what it can
    already see, which teaches it nothing.
    """
    require_torch("ibot_loss")
    if mask.sum() == 0:
        return student_masked.sum() * 0.0         # keeps the graph valid, contributes nothing
    t = F.softmax(teacher_full.detach()[mask] / teacher_temp, dim=-1)
    s = F.log_softmax(student_masked[mask] / student_temp, dim=-1)
    return -(t * s).sum(dim=-1).mean()


def episodic_loss(support, support_labels, query, query_labels, lam=1e-2, tau=0.07):
    """THE ONE THAT MATTERS. Train the backbone through the click adapter.

    Fit the ridge head on the support clicks, score the query points with it, and take the
    loss there. Because the ridge solve is differentiable, the gradient flows back through
    it into the BACKBONE -- so the backbone learns to produce descriptors that are easy to
    separate from a handful of clicks.

    That is the actual product objective, expressed directly.

    pos_weight handles the imbalance: a field has far more non-plant than plant, and
    without it the model learns to answer "no" everywhere.
    """
    from training.common.ridge import fit_predict
    require_torch("episodic_loss")
    logits = fit_predict(support, support_labels, query, lam=lam, tau=tau)
    y = (query_labels > 0).float()
    n_pos = y.sum().clamp(min=1.0)
    pos_weight = ((y.numel() - n_pos) / n_pos).clamp(1.0, 50.0)
    return F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)


def click_dependence_probe(support, support_labels, query, query_labels, **kw):
    """Does the model ACTUALLY use the clicks, or is it ignoring them?

    THE FAILURE THIS CATCHES
        A model can score well on the click task by learning "green blobby things are
        plants" and ignoring the support entirely. It would look excellent and would fail
        the moment a user clicked something unusual -- which is the whole product.

    So: run the episode normally, then run it again with the support LABELS SHUFFLED. If
    performance barely drops, the clicks are decorative.

    The plan requires correct clicks to beat shuffled by >= 0.10 F1.
    """
    require_torch("click_dependence_probe")
    real = episodic_loss(support, support_labels, query, query_labels, **kw)
    perm = torch.randperm(len(support_labels), device=support_labels.device)
    shuf = episodic_loss(support, support_labels[perm], query, query_labels, **kw)
    return {"loss_real": float(real), "loss_shuffled": float(shuf),
            "gap": float(shuf - real)}
