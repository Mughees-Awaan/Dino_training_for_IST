#!/usr/bin/env python3
"""
transforms.py -- THE DELIBERATE WOBBLES
=======================================

WHAT "AUGMENTATION" MEANS
    Before showing a photograph to the model, we mess it up slightly on purpose -- flip it,
    rotate it, nudge the brightness. Every time the same photograph comes round, it looks a
    little different.

WHY ON EARTH WOULD WE DO THAT
    Because a model that sees one photograph a hundred times identically will memorise THAT
    PHOTOGRAPH -- the exact shadows, the exact angle, the colour cast of that one afternoon.
    Show it the same field on a cloudy day and it fails.

    Wobbling the input forces it to learn what stays the SAME across the wobbles. That is the
    actual plant.

THE DECISION THIS FILE MAKES, AND WHY IT IS NOT A DEFAULT
    Whatever you wobble, you are teaching the model to IGNORE. So the choice of wobbles is a
    real decision about the task, and copying a standard recipe is how you get it wrong.

    GEOMETRY -- aggressive. Flip freely, rotate by any quarter turn.
        Aerial photographs have no natural "up". A drone flying north-to-south and one flying
        south-to-north produce the same field upside down. Orientation carries no information
        here, so scrambling it is pure gain.

    COLOUR -- gentle. Very small nudges only.
        This is where standard recipes would ruin us. The usual computer-vision recipe jitters
        colour hard and randomly converts images to greyscale, deliberately teaching colour
        invariance. That is right for photographs of cars and dogs.

        It is WRONG here. Our task is very often a yellow-green weed against a green crop.
        Colour is not a nuisance to be ignored -- it is frequently the ONLY signal there is.
        Teach the model to ignore colour and you have taught it to fail at its job.

        Hence brightness 0.15, contrast 0.15, saturation 0.10, hue 4 degrees. Small on purpose.

THE THING THAT MAKES THIS FILE DANGEROUS
    If you flip the image, YOU MUST FLIP THE CLICK COORDINATES TOO.

    Move the pixels but not the clicks, and every click now points at the wrong place. The
    model is being taught, patiently and consistently, to find plants where there are none.

    Nothing errors. No shape mismatch, no exception, no warning. Training runs beautifully to
    completion and the result is quietly worthless.

    So every transform here RETURNS a record of what it did (the `Applied` object), and the
    caller must push the coordinates through `Applied.map_points`. The pixels and the points
    are moved by the same object, from the same recorded decision.

    (This exact bug was found in this file's own rotation code and fixed -- see map_points.)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Applied:
    """A record of what was done to an image, so its points can follow.

    @dataclass is a Python shortcut: it turns a list of named fields into a full class with a
    constructor, so `Applied(flip_x=True)` just works without writing __init__ by hand.
    """
    flip_x: bool = False    # mirrored left-right
    flip_y: bool = False    # mirrored top-bottom
    rot90: int = 0          # number of quarter turns, 0 to 3

    def map_points(self, x, y, w: int, h: int):
        """Move point coordinates the same way the pixels were moved.

        `w` and `h` are the ORIGINAL image's width and height, before any of this was applied.

        --- FLIPS -----------------------------------------------------------------------
        Mirroring left-right sends a point at x to (w - 1 - x). The "- 1" is there because
        pixels are numbered from 0: in an 8-pixel-wide image the columns are 0..7, so column
        0 must map to column 7, not column 8. Forgetting the -1 pushes every point one pixel
        off and, at the edge, one pixel outside the image entirely.

        --- ROTATION -- and the bug that lived here --------------------------------------
        np.rot90 rotates COUNTER-clockwise. The original code here rotated the points
        CLOCKWISE. For 180 degrees that makes no difference, so half the cases looked fine;
        for 90 and 270 degrees every click was moved to the wrong corner of the image.

        Verified failing before the fix, on a deliberately non-square 5x8 test image: 8 of 16
        flip/rotation combinations put the click on the wrong pixel. It survived this long
        precisely because it raises no error -- it just trains on nonsense.

        The correct counter-clockwise mapping, matching np.rot90:

            new_x = y
            new_y = w - 1 - x

        and the image's width and height swap, which is why w and h are exchanged on the same
        line. Python evaluates the whole right-hand side BEFORE assigning, so the old w is
        still the old w when it is used in `w - 1 - x`.
        """
        # np.asarray lets this work on a single number or on a whole array of them.
        x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
        if self.flip_x:
            x = w - 1 - x
        if self.flip_y:
            y = h - 1 - y
        # `% 4` because four quarter turns is a full circle, back where we started.
        for _ in range(self.rot90 % 4):
            x, y, w, h = y, (w - 1 - x), h, w
        return x, y


def geometric(rgb: np.ndarray, rng) -> tuple[np.ndarray, Applied]:
    """Flips and quarter turns. Free information for aerial imagery -- there is no 'up'.

    Returns the changed image AND the record of what was changed. Both, always, together --
    it must be impossible to get the new pixels without also getting the mapping for the
    points.
    """
    a = Applied(flip_x=rng.random() < 0.5,     # a coin flip: rng.random() gives 0.0 to 1.0
                flip_y=rng.random() < 0.5,
                rot90=rng.randint(0, 3))       # 0, 1, 2 or 3 quarter turns
    out = rgb
    # `[:, ::-1]` means "all rows, columns in reverse order" -- a left-right mirror.
    if a.flip_x:
        out = out[:, ::-1]
    # `[::-1]` means "rows in reverse order" -- a top-bottom mirror.
    if a.flip_y:
        out = out[::-1]
    if a.rot90:
        out = np.rot90(out, a.rot90)           # counter-clockwise; map_points matches this
    # All of the above produce a "view" -- a window onto the original memory, not a real copy,
    # with the rows or columns walked backwards. Some libraries refuse such arrays. This makes
    # a proper, normally-laid-out copy.
    return np.ascontiguousarray(out), a


def photometric(rgb: np.ndarray, rng, brightness=0.15, contrast=0.15,
                saturation=0.10, hue_deg=4.0) -> np.ndarray:
    """Nudge the colours. THE DEFAULTS ARE SMALL ON PURPOSE -- see the header.

    No `Applied` is returned, because none is needed: changing colours does not move anything.
    A click stays on the same pixel whatever colour that pixel becomes.
    """
    # Work in 0.0-1.0 floats. Doing this arithmetic on 0-255 whole numbers would clip and
    # round at every step and lose precision.
    x = rgb.astype(np.float32) / 255.0

    if brightness:
        # Multiply everything by something near 1.0 -- makes the whole image slightly lighter
        # or slightly darker.
        x = x * (1.0 + rng.uniform(-brightness, brightness))

    if contrast:
        # Contrast means "how far from the average". Push every pixel slightly further from
        # the image's mean brightness (more contrast) or slightly closer to it (less).
        m = x.mean()
        x = (x - m) * (1.0 + rng.uniform(-contrast, contrast)) + m

    if saturation or hue_deg:
        import cv2
        # HSV is a different way of describing colour that separates the three ideas:
        #   H (hue)        WHICH colour it is        0-360 degrees around a colour wheel
        #   S (saturation) how VIVID it is           0 = grey, 1 = pure colour
        #   V (value)      how BRIGHT it is
        # We convert to HSV precisely so we can nudge hue and saturation independently --
        # in RGB you cannot touch one without disturbing the others.
        hsv = cv2.cvtColor(np.clip(x, 0, 1).astype(np.float32), cv2.COLOR_RGB2HSV)
        if hue_deg:
            # `% 360` wraps around the colour wheel: 358 degrees + 4 = 2 degrees, not 362.
            hsv[..., 0] = (hsv[..., 0] + rng.uniform(-hue_deg, hue_deg)) % 360.0
        if saturation:
            hsv[..., 1] = np.clip(hsv[..., 1] * (1.0 + rng.uniform(-saturation, saturation)), 0, 1)
        x = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    # np.clip forces everything back into the valid 0-1 range -- brightness may have pushed
    # some pixels past 1.0 -- then convert back to whole 0-255 values.
    return (np.clip(x, 0, 1) * 255).astype(np.uint8)


def apply(rgb: np.ndarray, rng, geo: bool = True, photo: bool = True):
    """Do both, and return (image, Applied).

    CALLERS MUST PASS THEIR POINTS THROUGH `Applied.map_points`. Returning the two together
    from one function is the whole design: it makes forgetting deliberate rather than easy.
    """
    a = Applied()
    if geo:
        rgb, a = geometric(rgb, rng)
    if photo:
        rgb = photometric(rgb, rng)
    return rgb, a
