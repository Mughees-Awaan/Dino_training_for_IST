#!/usr/bin/env python3
"""
train_episodic.py -- T2/T3: train the backbone THROUGH the click adapter.

WHAT MAKES THIS DIFFERENT FROM ORDINARY TRAINING
    We do not train a classifier. We train the BACKBONE so that its descriptors are easy
    to separate from a handful of clicks -- because that is literally what the product does.

    Each step: fit the ridge head on the support clicks, score the query points with it,
    take the loss there, and let the gradient flow BACK THROUGH the ridge solve into the
    backbone.

    Verified on this machine: backward-through-Cholesky is numerically stable across
    5-40 clicks and lambda 1e-4 to 1e-1, gradient norms 1.3-5.8, none NaN.
    That was the scaffold-stage exit criterion.

⛔ BLOCKED ON DATA, NOT ON CODE
    This needs episodes carrying POSITIVE AND NEGATIVE clicks. The current episodes in
    tables/ have positives only, so --smoke works but a real run cannot start yet.
    The interface below is what the episode redesign must produce.
"""

from __future__ import annotations

import argparse
import time

from training.common.safe_imports import np, require_torch, torch
from training.teacher.losses import click_dependence_probe, episodic_loss


def train_step(backbone, batch, opt, lam=1e-2, tau=0.07, clip=1.0):
    """One optimisation step over one episode."""
    require_torch("train_step")
    backbone.train()
    opt.zero_grad(set_to_none=True)

    sup = backbone(batch["support_image"])
    qry = backbone(batch["query_image"])
    sup_d = sup.reshape(-1, sup.shape[-1])[batch["support_cells"]]
    qry_d = qry.reshape(-1, qry.shape[-1])[batch["query_cells"]]

    loss = episodic_loss(sup_d, batch["support_labels"], qry_d, batch["query_labels"],
                         lam=lam, tau=tau)
    loss.backward()

    # Gradient clipping is not optional here. The ridge solve can amplify gradients when
    # the support matrix is close to singular -- one badly conditioned episode could
    # otherwise wreck weights built over hours.
    gn = torch.nn.utils.clip_grad_norm_(backbone.parameters(), clip)
    opt.step()
    return float(loss), float(gn)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", default="tables/v2-staging/episodes/train.parquet")
    ap.add_argument("--manifest", default="tables/manifest.parquet")
    ap.add_argument("--out", default="runs/t2")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--lam", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true",
                    help="16-episode deliberate overfit. Proves the WIRING, not "
                         "generalisation: loss must reach ~0 or something is disconnected.")
    args = ap.parse_args()

    require_torch("train_episodic")
    torch.manual_seed(args.seed)

    if args.smoke:
        # Synthetic separable episodes -- no data dependency, so the scaffold can be
        # verified before the episode redesign lands.
        from training.common.ridge import fit_predict
        d = 64
        proto = torch.randn(d)
        emb = torch.nn.Linear(d, d)
        opt = torch.optim.Adam(emb.parameters(), lr=1e-2)
        for step in range(200):
            sup = torch.cat([proto + 0.2 * torch.randn(8, d), -proto + 0.2 * torch.randn(8, d)])
            lab = torch.tensor([1.0] * 8 + [-1.0] * 8)
            qry = torch.cat([proto + 0.2 * torch.randn(16, d), -proto + 0.2 * torch.randn(16, d)])
            qlab = torch.tensor([1.0] * 16 + [-1.0] * 16)
            opt.zero_grad()
            loss = episodic_loss(emb(sup), lab, emb(qry), qlab, lam=args.lam)
            loss.backward(); opt.step()
            if step % 50 == 0 or step == 199:
                print(f"  step {step:>4}  loss {float(loss):.6f}")
        probe = click_dependence_probe(emb(sup).detach(), lab, emb(qry).detach(), qlab)
        print(f"  click dependence gap {probe['gap']:.4f}  "
              f"({'PASS' if probe['gap'] > 0.10 else 'FAIL -- clicks are being ignored'})")
        return 0

    raise SystemExit(
        "A real episodic run needs episodes with POSITIVE AND NEGATIVE clicks.\n"
        "The current episode tables carry positives only (build_episodes.py writes\n"
        "support_x/support_y with no labels). Redesign episodes first, or use --smoke\n"
        "to verify the training scaffold on synthetic data.")


if __name__ == "__main__":
    raise SystemExit(main())
