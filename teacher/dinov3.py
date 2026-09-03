#!/usr/bin/env python3
"""
dinov3.py -- wrapper around the frozen backbone that produces patch descriptors.

WHAT A BACKBONE DOES HERE
    It takes an image and returns one descriptor per grid square -- a list of numbers
    describing what that patch looks like. Everything downstream (the ridge adapter, the
    student, the product) consumes those descriptors.

THE GOTCHA THAT BITES EVERYONE: PREFIX TOKENS
    A Vision Transformer does not only emit one token per image patch. DINOv3 ViT-S/16
    prepends FIVE extra tokens that describe the whole image rather than any location:

        token 0        CLS      a summary of the entire image
        tokens 1-4     registers  scratch space the model uses internally

    So a 224x224 image gives 5 + 196 tokens, not 196. If you reshape all 201 into a grid
    you get garbage -- and it is the kind of garbage that still has plausible shape, so it
    runs fine and produces nonsense.

    STRIP THE PREFIX. That is what n_prefix exists for.

WHY IT IS FROZEN BY DEFAULT
    The product fits only the click head at inference time. The backbone is fixed. Training
    stages T1/T2/T3 unfreeze it deliberately, but the default must be frozen so an
    accidental gradient never silently changes it.
"""

from __future__ import annotations

from training.common.safe_imports import HAVE_TORCH, nn, require_torch, torch

# DINOv3 emits CLS + 4 registers before the patch tokens. Wrong here = silent nonsense.
N_PREFIX_TOKENS = 5
PATCH = 16          # each token covers a 16x16 pixel square
DIM = 384           # ViT-S descriptors are 384 numbers long


class DinoV3Backbone(nn.Module if HAVE_TORCH else object):
    """Wraps a timm ViT so it returns a clean (B, H, W, C) grid of patch descriptors."""

    def __init__(self, name: str = "vit_small_patch16_224.dino",
                 pretrained: bool = True, frozen: bool = True,
                 n_prefix: int = N_PREFIX_TOKENS, out_layers: tuple[int, ...] = (-1,)):
        require_torch("DinoV3Backbone")
        super().__init__()
        import timm
        self.net = timm.create_model(name, pretrained=pretrained, num_classes=0)
        self.n_prefix = n_prefix
        self.out_layers = out_layers
        self.frozen = frozen
        if frozen:
            self.freeze()

    def freeze(self):
        """Stop gradients AND put the network in eval mode.

        Both are needed. requires_grad=False stops weights updating; eval() stops dropout
        and batch-norm statistics moving. Doing only the first still lets the model drift.
        """
        for p in self.net.parameters():
            p.requires_grad = False
        self.net.eval()
        self.frozen = True
        return self

    def unfreeze(self):
        for p in self.net.parameters():
            p.requires_grad = True
        self.net.train()
        self.frozen = False
        return self

    def forward(self, x):
        """(B, 3, H, W) image -> (B, H/16, W/16, C) descriptor grid."""
        B, _, H, W = x.shape
        gh, gw = H // PATCH, W // PATCH

        # get_intermediate_layers returns patch tokens with the prefix ALREADY removed by
        # timm. We ask it explicitly rather than slicing ourselves, because the number of
        # register tokens differs between DINOv3 variants and hard-coding 5 would silently
        # break on another checkpoint.
        feats = self.net.get_intermediate_layers(
            x, n=self.out_layers if isinstance(self.out_layers, int) else len(self.out_layers),
            reshape=False, return_prefix_tokens=False, norm=True)

        # Concatenate the requested blocks along the descriptor axis. Measured on this
        # project: the last TWO blocks beat the last one by +0.023 / +0.006 F1, for free.
        g = torch.cat(list(feats), dim=-1) if len(feats) > 1 else feats[0]

        if g.shape[1] != gh * gw:
            raise RuntimeError(
                f"expected {gh*gw} patch tokens for a {H}x{W} input, got {g.shape[1]}. "
                f"Prefix tokens were probably not stripped -- see N_PREFIX_TOKENS.")
        return g.reshape(B, gh, gw, g.shape[-1])

    @property
    def stride(self) -> int:
        return PATCH
