#!/usr/bin/env python3
"""
heads.py -- the small networks bolted onto the backbone during self-supervised training.

These exist ONLY during training and are thrown away afterwards. The product ships the
backbone alone.

WHY A HEAD AT ALL
    Self-supervised learning works by showing the model two different crops of the same
    photograph and asking it to produce the same answer for both. But comparing raw
    descriptors directly makes a degenerate shortcut too easy: output the same constant
    for everything and the loss is zero. That is "collapse", and it is the classic failure.

    Projecting through a head into a wider space, then comparing there, makes collapse
    much harder while leaving the backbone free to keep useful detail.
"""

from __future__ import annotations

from training.common.safe_imports import HAVE_TORCH, F, nn, require_torch


class DINOHead(nn.Module if HAVE_TORCH else object):
    """Projects a descriptor to a probability distribution over K abstract 'prototypes'.

    The prototypes are not classes anybody named -- they are just K directions the model
    invents. Two crops of the same photograph should land on the same prototypes.
    """

    def __init__(self, in_dim: int = 384, hidden: int = 2048, bottleneck: int = 256,
                 out_dim: int = 65536):
        require_torch("DINOHead")
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, bottleneck),
        )
        # weight_norm splits the last layer into a direction and a length, and the length is
        # frozen at 1. Without it this layer's weights grow without bound during training --
        # a documented DINO instability.
        self.last = nn.utils.parametrizations.weight_norm(
            nn.Linear(bottleneck, out_dim, bias=False))
        self.last.parametrizations.weight.original0.data.fill_(1)
        self.last.parametrizations.weight.original0.requires_grad = False

    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)     # compare by direction only
        return self.last(x)


class IBOTHead(DINOHead if HAVE_TORCH else object):
    """Same architecture, different job: predicting MASKED patches.

    DINO asks "do two crops agree about the whole image?". iBOT hides some patches and asks
    "what belonged here?". That forces the model to learn local structure rather than only
    a global gist -- which is what this product needs, since it must find small plants at
    specific locations, not describe a field.
    """
    pass
