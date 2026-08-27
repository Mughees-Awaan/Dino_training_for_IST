#!/usr/bin/env python3
"""
test_transform_points.py -- REGRESSION TEST FOR THE AUGMENTATION COORDINATE BUG
===============================================================================

THE BUG THIS GUARDS AGAINST
    transforms.geometric() flips and rotates the PIXELS. Applied.map_points() moves the CLICK
    COORDINATES to match. If the two ever disagree, the model is trained to look for plants
    where there are none -- and NOTHING ERRORS. Training runs to completion and produces a
    quietly worthless model.

WHAT WAS ACTUALLY WRONG
    np.rot90 turns COUNTER-clockwise. map_points turned CLOCKWISE. For 180 degrees that makes
    no difference, so half the cases looked correct; for 90 and 270 degrees every click landed
    in the wrong corner. Measured: 8 of 16 flip/rotation combinations were wrong.

HOW THIS TEST WORKS
    Build a small image where EVERY PIXEL HAS A DIFFERENT VALUE -- pixel (x, y) holds the
    number y*width + x. Now the value at a pixel is a name for that exact pixel.

    Then, for every combination of flip and rotation, and for every pixel:
        - transform the image
        - ask map_points where that pixel went
        - check the value sitting at the new location is the value we started with

    If the pixels and the points move together, the value matches. If they disagree by even
    one step, it does not.

WHY THE TEST IMAGE IS 5 x 8 AND NOT SQUARE
    A square image hides rotation bugs completely -- with equal width and height, several
    wrong formulas produce coordinates that are still in range and still plausible. The
    original bug survived precisely because nobody tested it on a rectangle.

    python training/tests/test_transform_points.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from training.data.transforms import geometric  # noqa: E402


class _FixedRng:
    """A fake random generator that always returns the choices we want to test.

    geometric() asks for two coin flips and one number 0-3. This hands back exactly the
    combination under test, so we can walk through all 16 deliberately rather than hoping
    random chance covers them.
    """

    def __init__(self, flip_x: bool, flip_y: bool, k: int):
        # geometric() tests `rng.random() < 0.5`, so 0.0 means "yes, flip" and 1.0 means "no".
        self._vals = [0.0 if flip_x else 1.0, 0.0 if flip_y else 1.0]
        self._k = k
        self._i = 0

    def random(self) -> float:
        v = self._vals[self._i]
        self._i += 1
        return v

    def randint(self, a: int, b: int) -> int:
        return self._k


def test_points_follow_pixels():
    """Every pixel, every flip, every rotation: the click must land on the same pixel."""
    h, w = 5, 8                                     # NOT square -- see the header
    img = np.arange(h * w, dtype=np.int32).reshape(h, w)
    rgb = np.dstack([img, img, img]).astype(np.uint8)   # value identifies the pixel

    failures = []
    for flip_x in (False, True):
        for flip_y in (False, True):
            for k in range(4):                       # 0, 90, 180, 270 degrees
                out, applied = geometric(rgb, _FixedRng(flip_x, flip_y, k))
                new_h, new_w = out.shape[:2]
                for y in range(h):
                    for x in range(w):
                        nx, ny = applied.map_points([x], [y], w, h)
                        nx, ny = int(nx[0]), int(ny[0])
                        if not (0 <= nx < new_w and 0 <= ny < new_h):
                            failures.append(
                                f"flip_x={flip_x} flip_y={flip_y} rot90={k}: pixel ({x},{y}) "
                                f"mapped to ({nx},{ny}), outside the {new_w}x{new_h} result")
                        elif out[ny, nx, 0] != rgb[y, x, 0]:
                            failures.append(
                                f"flip_x={flip_x} flip_y={flip_y} rot90={k}: pixel ({x},{y}) "
                                f"holds {rgb[y, x, 0]} but ({nx},{ny}) holds {out[ny, nx, 0]}")

    assert not failures, (
        f"{len(failures)} of {4 * 4 * h * w} checks failed -- the image and the click "
        f"coordinates are being moved differently, so training would learn wrong locations "
        f"and nothing would error.\nFirst three:\n  " + "\n  ".join(failures[:3]))


def test_180_degrees_is_direction_agnostic():
    """Documents WHY the bug hid for so long: at 180 degrees both directions agree.

    This test is expected to pass both before and after the fix. It exists so that a future
    reader who tries to reproduce the bug with a 180-degree case, sees it pass, and concludes
    the code was fine, has this note in front of them.
    """
    h, w = 5, 8
    img = np.arange(h * w, dtype=np.int32).reshape(h, w)
    rgb = np.dstack([img, img, img]).astype(np.uint8)
    out, applied = geometric(rgb, _FixedRng(False, False, 2))
    nx, ny = applied.map_points([0], [0], w, h)
    assert out[int(ny[0]), int(nx[0]), 0] == rgb[0, 0, 0]


if __name__ == "__main__":
    for fn in (test_points_follow_pixels, test_180_degrees_is_direction_agnostic):
        fn()
        print(f"  PASS  {fn.__name__}")
    print("augmentation coordinate tests pass")
