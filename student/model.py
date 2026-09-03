#!/usr/bin/env python3
"""
model.py -- the small CPU model that ships to the laptop.

WHY A SECOND MODEL AT ALL
    The teacher is a Vision Transformer. Accurate, and far too slow for a laptop CPU. The
    student is a small convolutional network trained to IMITATE the teacher's descriptors
    -- most of the quality, a fraction of the cost.

ARCHITECTURE (from the programme spec)
    MobileNetV3-Small, truncated after features[8]. Three taps at different scales:

        features[1]  -> stride 4    fine detail, small plants
        features[3]  -> stride 8    medium
        features[8]  -> stride 16   coarse, matches the teacher's grid

    Each tap is squeezed to 96 channels by a 1x1 convolution, then fused TOP-DOWN: the
    coarse map is upsampled and added to the finer one. Coarse layers know *what* a thing
    is; fine layers know *where* it is. Fusing gives both.

WHY GROUPNORM AND NOT BATCHNORM
    BatchNorm's statistics depend on the whole batch. At inference the product runs ONE
    tile at a time, so those statistics are wrong, and results change with batch size.
    GroupNorm normalises within a single sample -- identical answer for batch 1 or 32.
    The borrowed MobileNet BatchNorm layers stay FROZEN for the same reason.
"""

from __future__ import annotations

from training.common.safe_imports import HAVE_TORCH, F, nn, require_torch, torch

TAPS = {1: 4, 3: 8, 8: 16}      # feature index -> stride it represents
WIDTH = 96                      # channels every tap is squeezed to


def dw_sep(cin: int, cout: int, groups: int = 12):
    """Depthwise-separable 3x3 -> GroupNorm -> SiLU.

    A normal 3x3 conv costs cin*cout*9 multiplications. Depthwise-separable splits it into
    a 3x3 that mixes only SPACE (per channel) and a 1x1 that mixes only CHANNELS -- roughly
    8-9x cheaper for nearly the same capability. That ratio is the entire reason a model
    like this runs on a CPU.
    """
    return nn.Sequential(
        nn.Conv2d(cin, cin, 3, padding=1, groups=cin, bias=False),   # groups=cin = depthwise
        nn.Conv2d(cin, cout, 1, bias=False),                         # pointwise
        nn.GroupNorm(groups, cout),
        nn.SiLU(inplace=True),
    )


class StudentBackbone(nn.Module if HAVE_TORCH else object):
    """MobileNetV3-Small trunk + top-down fusion + the product's output heads."""

    def __init__(self, out_dim: int = 384, pretrained: bool = True):
        require_torch("StudentBackbone")
        super().__init__()
        from torchvision.models import mobilenet_v3_small
        m = mobilenet_v3_small(weights="DEFAULT" if pretrained else None)
        self.features = nn.ModuleList(list(m.features)[:9])   # truncate after features[8]

        chans = {1: 16, 3: 24, 8: 48}                         # channels at each tap
        self.lateral = nn.ModuleDict({str(k): nn.Conv2d(c, WIDTH, 1, bias=False)
                                      for k, c in chans.items()})
        self.fuse = nn.ModuleDict({str(k): dw_sep(WIDTH, WIDTH) for k in TAPS})

        # Descriptor heads -- what the ridge adapter consumes. Two scales, because plant
        # size varies enormously between crops and the right scale is field-dependent.
        self.desc_s8 = nn.Conv2d(WIDTH, out_dim, 1)
        self.desc_s16 = nn.Conv2d(WIDTH, out_dim, 1)

        # Geometry heads at the finest scale, for turning a heatmap into instances.
        self.centre = nn.Conv2d(WIDTH, 1, 1)      # is a plant centred here?
        self.offset = nn.Conv2d(WIDTH, 2, 1)      # sub-cell nudge, so points are not
                                                  # quantised to the grid
        self.log_size = nn.Conv2d(WIDTH, 1, 1)    # log so one head covers tiny and large
        self.boundary = nn.Conv2d(WIDTH, 1, 1)    # helps split touching plants

        self.freeze_bn()

    def freeze_bn(self):
        """Stop BatchNorm statistics moving. See the header for why."""
        for mod in self.modules():
            if isinstance(mod, nn.BatchNorm2d):
                mod.eval()
                for p in mod.parameters():
                    p.requires_grad = False
        return self

    def train(self, mode: bool = True):
        """Override so .train() can never silently un-freeze BatchNorm."""
        super().train(mode)
        self.freeze_bn()
        return self

    def forward(self, x) -> dict:
        feats, h = {}, x
        for i, layer in enumerate(self.features):
            h = layer(h)
            if i in TAPS:
                feats[i] = self.lateral[str(i)](h)

        # TOP-DOWN FUSION: start coarse, upsample, add into the next finer level.
        p = feats[8]
        p8 = self.fuse["8"](p)
        p = feats[3] + F.interpolate(p8, size=feats[3].shape[-2:], mode="nearest")
        p3 = self.fuse["3"](p)
        p = feats[1] + F.interpolate(p3, size=feats[1].shape[-2:], mode="nearest")
        p1 = self.fuse["1"](p)

        return {
            # L2-normalised because the ridge adapter compares by DIRECTION. Normalising
            # here rather than in the adapter means the exported ONNX already does it.
            "desc_s16": F.normalize(self.desc_s16(p8), dim=1),
            "desc_s8": F.normalize(self.desc_s8(p3), dim=1),
            "centre": self.centre(p1),
            "offset": self.offset(p1),
            "log_size": self.log_size(p1),
            "boundary": self.boundary(p1),
        }
