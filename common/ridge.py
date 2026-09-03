#!/usr/bin/env python3
"""
ridge.py -- THE CLICK-CONDITIONED ADAPTER. The heart of the whole architecture.

WHAT PROBLEM THIS SOLVES
    The product promises: the user clicks 5-20 examples, and the tool finds the rest --
    instantly, with no training and no waiting.

    "No training" is the hard part. You cannot run gradient descent while a user waits.
    So the model that adapts to the user's clicks must be solvable in CLOSED FORM: one
    direct calculation, no iteration, no learning rate, no epochs.

    Ridge regression is exactly that.

THE IDEA IN PLAIN TERMS
    The frozen backbone turns every patch of the image into a list of numbers -- a
    "descriptor" -- that describes what that patch looks like. Similar-looking patches get
    similar descriptors.

    The user's clicks give us a handful of labelled descriptors:
        positive clicks -> "this is the plant"      label +1
        negative clicks -> "this is not the plant"  label -1

    We need a rule that separates them, and that we can apply to every other patch in the
    field. The simplest useful rule is a WEIGHTED SUM: give each number in the descriptor a
    weight, add them up, and if the total is high it is the plant.

    Ridge regression finds those weights in one shot.

THE MATHS, AND WHY EACH PIECE IS THERE
        X       the support descriptors, L2-normalised, with a bias column   (k x d)
        y       their labels, +1 or -1                                       (k,)
        A       X @ X.T  +  lambda * I                                       (k x k)
        alpha   solve(A, y)                                via Cholesky      (k,)
        w       X.T @ alpha                                                  (d,)
        logits  X_query @ w / tau

    * L2-NORMALISE first so a patch that happens to produce large numbers does not
      dominate one that produces small ones. We care about DIRECTION, not magnitude.

    * The BIAS COLUMN is a constant 1 appended to every descriptor. Without it the
      decision boundary is forced through the origin, which is an arbitrary restriction.

    * lambda * I is the "ridge". With only 5-20 clicks and 384 numbers per descriptor,
      there are far more unknowns than equations -- infinitely many perfect answers exist,
      most of them nonsense that memorises the clicks. Adding lambda to the diagonal picks
      the smallest, smoothest answer among them. It also GUARANTEES the matrix can be
      inverted, which is what makes this safe to run unattended on a user's click.

    * WHY k x k, NOT d x d. The obvious formulation inverts a 384x384 matrix. Because there
      are only k clicks (k is 5-20), the "kernel trick" form inverts a k x k matrix instead
      -- a 20x20 solve rather than 384x384. That is the difference between instant and not.

    * tau (temperature) just scales the output into a sensible range for a sigmoid.

WHY IT MUST BE DIFFERENTIABLE
    During teacher training we want the loss on the QUERY points to improve the BACKBONE.
    The gradient has to flow backwards through the ridge solve itself. torch.linalg
    supports that, so the whole thing is one differentiable expression -- there are no
    learned parameters in this file at all.

    THIS IS THE RISKIEST PIECE OF THE TRAINING PLAN. Backward-through-Cholesky has never
    been prototyped on this project, and its numerical stability is the stated exit
    criterion for the scaffold stage. Hence solve_ridge(..., check=True) and the
    stability_report() helper below.

CURRENT BLOCKER
    This needs POSITIVE AND NEGATIVE support points. The episodes in tables/ carry
    positives only, so this cannot yet be exercised end to end. The interface below is the
    one the episode redesign must produce.
"""

from __future__ import annotations

from training.common.safe_imports import HAVE_TORCH, np, require_torch


def l2_normalise(x, eps: float = 1e-8):
    """Scale every descriptor to length 1, so only its DIRECTION matters.

    Works on numpy arrays or torch tensors -- the operations used exist in both.
    `keepdims` keeps the result shaped (n, 1) so it divides row-wise rather than
    collapsing to a single number.
    """
    if HAVE_TORCH and hasattr(x, "norm"):
        return x / (x.norm(dim=-1, keepdim=True) + eps)
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + eps)


def add_bias(x):
    """Append a constant 1 to every descriptor.

    Without it the decision boundary must pass through the origin -- an arbitrary and
    usually wrong constraint. One extra column buys the freedom to shift it.
    """
    if HAVE_TORCH and hasattr(x, "new_ones"):
        return __import__("torch").cat([x, x.new_ones(*x.shape[:-1], 1)], dim=-1)
    return np.concatenate([x, np.ones((*x.shape[:-1], 1), x.dtype)], axis=-1)


def solve_ridge(support, labels, lam: float = 1e-2, check: bool = False):
    """Fit the click-conditioned head. Returns the weight vector `w`.

    support : (k, d) descriptors at the user's clicks
    labels  : (k,)   +1 for a positive click, -1 for a negative one
    lam     : the ridge. Larger = smoother, less willing to contort itself to fit clicks.

    Everything is forced to FLOAT32 (or better). Cholesky on float16 is numerically
    hopeless -- the whole point of the decomposition is that it is well conditioned, and
    half precision throws that away.
    """
    torch = require_torch("solve_ridge")

    X = add_bias(l2_normalise(support.float()))          # (k, d+1)
    y = labels.float().reshape(-1, 1)                    # (k, 1)
    k = X.shape[0]

    # A is the k x k Gram matrix plus the ridge on its diagonal.
    # torch.eye builds the identity matrix; adding lam to the diagonal is what makes A
    # positive-definite, which is precisely the condition Cholesky requires.
    A = X @ X.transpose(-1, -2) + lam * torch.eye(k, device=X.device, dtype=X.dtype)

    # Cholesky factorises A into L @ L.T with L triangular, then solves by substitution.
    # Roughly twice as fast as a general solve AND better conditioned -- and it FAILS LOUDLY
    # if A is not positive-definite, which is a useful alarm rather than silent nonsense.
    L = torch.linalg.cholesky(A)
    alpha = torch.cholesky_solve(y, L)                   # (k, 1)
    w = X.transpose(-1, -2) @ alpha                      # (d+1, 1)

    if check:
        # Did we actually solve it? Residual should be ~0. This is the guard against the
        # silent-garbage failure mode: a solve can "succeed" and be wrong.
        resid = (A @ alpha - y).abs().max().item()
        if not np.isfinite(resid) or resid > 1e-3:
            raise RuntimeError(
                f"ridge solve did not converge: max residual {resid:.3e}. "
                f"Usually means lam is too small for {k} clicks, or the descriptors "
                f"contain NaN/Inf.")
    return w


def apply_ridge(query, w, tau: float = 0.07):
    """Score every query descriptor with the fitted weights.

    Returns raw logits. Push them through a sigmoid to get 0-1 confidences.
    `tau` scales them into a usable range -- without it the numbers sit so close to zero
    that a sigmoid returns ~0.5 everywhere.
    """
    Xq = add_bias(l2_normalise(query.float()))
    return (Xq @ w).squeeze(-1) / tau


def fit_predict(support, labels, query, lam: float = 1e-2, tau: float = 0.07,
                check: bool = False):
    """The whole click-conditioned adapter in one call: clicks in, scores out."""
    return apply_ridge(query, solve_ridge(support, labels, lam, check), tau)


def stability_report(d: int = 384, ks=(5, 10, 20, 40), lams=(1e-4, 1e-2, 1e-1),
                     seed: int = 0) -> list[dict]:
    """Probe backward-through-Cholesky before trusting it in a training loop.

    This is the scaffold-stage exit criterion. For each click count and ridge value it
    fits on random descriptors, backpropagates a query loss, and records whether the
    gradient is finite and how large it is.

    An exploding or NaN gradient here means training WILL diverge -- and it is far cheaper
    to learn that from this function than from a GPU run that dies at hour six.
    """
    torch = require_torch("stability_report")
    out = []
    for k in ks:
        for lam in lams:
            g = torch.Generator().manual_seed(seed)
            sup = torch.randn(k, d, generator=g, requires_grad=True)
            qry = torch.randn(64, d, generator=g)
            # deliberately balanced: alternating +1 / -1
            lab = torch.tensor([1.0 if i % 2 == 0 else -1.0 for i in range(k)])
            try:
                logits = fit_predict(sup, lab, qry, lam=lam, check=True)
                logits.pow(2).mean().backward()
                gn = sup.grad.norm().item()
                out.append({"k": k, "lam": lam, "grad_norm": gn,
                            "finite": bool(np.isfinite(gn)), "error": ""})
            except Exception as exc:
                out.append({"k": k, "lam": lam, "grad_norm": float("nan"),
                            "finite": False, "error": str(exc)[:80]})
    return out


if __name__ == "__main__":
    print(f"{'clicks':>7}{'lambda':>10}{'grad norm':>14}  status")
    for r in stability_report():
        status = "ok" if r["finite"] else f"UNSTABLE {r['error']}"
        print(f"{r['k']:>7}{r['lam']:>10.4f}{r['grad_norm']:>14.4e}  {status}")
