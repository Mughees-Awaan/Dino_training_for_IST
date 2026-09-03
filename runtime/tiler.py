#!/usr/bin/env python3
"""
tiler.py -- CUT A HUGE PHOTOGRAPH UP, AND PUT IT BACK TOGETHER WITHOUT SEAMS
=============================================================================

THE PROBLEM
    A drone survey map can be 30,000 pixels across. No model can read that in one go -- it
    would need hundreds of gigabytes of memory. So the image is cut into tiles, each tile is
    processed, and the answers are assembled back into one map.

    Two things go wrong when you do this naively.

PROBLEM 1: SEAMS
    The model decides what a square of pixels contains by ALSO looking at what surrounds it.
    A weed is recognisable partly because of the crop rows around it.

    Now cut the image. A square that used to sit comfortably in the middle of the picture is
    suddenly at the very edge of a tile, with nothing beyond it. The model sees less context
    and produces a different description for the exact same ground.

    The result is a visible grid of errors across the whole map, following the tile
    boundaries. Detections appear and disappear along straight lines.

THE FIX: A HALO
    Read each tile BIGGER than you need -- with a margin of real pixels on every side. Run the
    model on the whole enlarged tile. Then THROW THE MARGIN AWAY and keep only the middle.

    Every square you keep had full context when it was computed. The margin squares, the ones
    that suffered from being at an edge, are exactly the ones discarded.

        +---------------------------+
        |   halo (read, discarded)  |
        |   +-------------------+   |
        |   |                   |   |
        |   |   core (kept)     |   |
        |   |                   |   |
        |   +-------------------+   |
        |                           |
        +---------------------------+

    The cost is reading some pixels more than once. That is cheap. Seams are not.

PROBLEM 2: HOLES AND DOUBLE-WRITES
    Trimming the halo means discarding whole squares. If the halo is not an exact number of
    squares, you discard slightly too much (leaving a hole in the output) or slightly too
    little (leaving neighbouring tiles to overwrite each other).

    lattice.check_halo refuses that outright, before anything runs.

    And this module found a real instance of the second kind in its own code: an early version
    kept the trailing halo as well as the core, so adjacent tiles overwrote each other's edges.
    MEASURED: 416 cells written twice on a single 1024x1536 image. It was caught by the
    write-once assertion at the bottom of stitch() -- which is why that assertion is there
    instead of a comment saying "this should be fine".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from training.runtime import lattice


@dataclass(frozen=True)
class Tile:
    """One tile's complete instructions.

    `frozen=True` makes it read-only after creation. A plan is a set of decisions already
    made; something quietly adjusting a tile's boundaries mid-run is a bug, not a feature, so
    the language is told to forbid it.
    """
    # WHAT TO READ from the source image -- the core PLUS its halo, in pixels.
    read_x0: int; read_y0: int; read_x1: int; read_y1: int
    # WHERE THE KEPT SQUARES GO in the final assembled grid -- in squares, not pixels.
    keep_col0: int; keep_row0: int
    # HOW MANY SQUARES TO DISCARD off the top and left after the model runs.
    # This is not always the full halo: a tile at the image's edge has no room for a halo on
    # that side, so nothing was read there and nothing needs discarding.
    trim_left: int; trim_top: int


def plan(height: int, width: int, core: int, halo: int, stride: int) -> list[Tile]:
    """Work out the full set of tiles BEFORE reading anything.

    Separating the plan from the work is deliberate. The plan is pure arithmetic -- no images,
    no model, no memory -- so it can be printed, inspected and tested on its own. Every tiling
    bug this file has had was an arithmetic bug, and arithmetic you can look at is arithmetic
    you can fix.
    """
    lattice.check_halo(halo, stride)     # refuse a misaligned halo before doing any work
    lattice.check_halo(core, stride)     # the core must be whole squares too

    tiles = []
    # Step across the image in CORE-sized jumps. The halo is extra reading around each step,
    # not part of the step -- otherwise the cores would not tile the image exactly.
    for y0 in range(0, height, core):
        for x0 in range(0, width, core):
            # Expand by the halo, but never past the edge of the actual image.
            # max(0, ...) and min(width, ...) clamp it -- a tile in the top-left corner simply
            # gets no halo above or to its left, because there is nothing there to read.
            rx0, ry0 = max(0, x0 - halo), max(0, y0 - halo)
            rx1, ry1 = min(width, x0 + core + halo), min(height, y0 + core + halo)

            tiles.append(Tile(
                rx0, ry0, rx1, ry1,
                x0 // stride, y0 // stride,     # where this core starts, in squares
                # How much halo we ACTUALLY got on each side, converted to squares. For an
                # interior tile this is halo/stride; for an edge tile it is 0, because the
                # clamp above meant no halo was read there.
                (x0 - rx0) // stride, (y0 - ry0) // stride))
    return tiles


def stitch(height: int, width: int, core: int, halo: int, stride: int, channels: int,
           run) -> np.ndarray:
    """Run the model over every tile and assemble one seamless map.

    `run` is a FUNCTION you pass in: run(x0, y0, x1, y1) -> a (h, w, C) block of descriptions.

    Passing the model in as a function keeps this file completely ignorant of what the model
    is -- PyTorch, ONNX, a test stub returning constants. The tiling arithmetic can therefore
    be tested with no model at all, which is exactly how the double-write bug was reproduced.
    """
    rows, cols = lattice.grid_shape(height, width, stride)
    out = np.zeros((rows, cols, channels), np.float32)
    # A parallel grid counting how many times each cell has been written. This exists purely
    # for the check at the bottom.
    seen = np.zeros((rows, cols), np.int32)

    for t in plan(height, width, core, halo, stride):
        got = run(t.read_x0, t.read_y0, t.read_x1, t.read_y1)

        # ============ THE LINE THAT HAD THE BUG ============
        # Trim the LEADING halo, then take EXACTLY the core -- not "whatever is left".
        #
        # "Whatever is left" was the original code, and it is subtly wrong: after removing the
        # leading halo, what remains is the core AND the TRAILING halo. Keeping both means each
        # tile writes into its neighbour's territory, and whichever tile runs last wins.
        #
        # Measured on one 1024x1536 image: 416 cells written twice.
        #
        # min(..., rows - t.keep_row0) additionally stops the LAST tile in a row or column
        # from writing past the end of the output grid.
        keep_r = min(core // stride, rows - t.keep_row0)
        keep_c = min(core // stride, cols - t.keep_col0)
        sub = got[t.trim_top: t.trim_top + keep_r, t.trim_left: t.trim_left + keep_c]

        r1 = min(rows, t.keep_row0 + sub.shape[0])
        c1 = min(cols, t.keep_col0 + sub.shape[1])
        if r1 <= t.keep_row0 or c1 <= t.keep_col0:
            continue     # this tile contributes no whole squares (a sliver at the edge)

        out[t.keep_row0:r1, t.keep_col0:c1] = sub[: r1 - t.keep_row0, : c1 - t.keep_col0]
        seen[t.keep_row0:r1, t.keep_col0:c1] += 1

    # ============ THE WRITE-ONCE CHECK ============
    # Every cell in the output must have been written EXACTLY ONCE.
    #   written zero times -> a hole; that part of the field has no answer at all
    #   written twice      -> tiles overlapped; one silently overwrote the other
    #
    # Both are invisible in the result -- a hole full of zeros and a double-write both look
    # like perfectly ordinary output. This is the entire reason the `seen` grid exists, and
    # it is what caught the 416-cell bug above. It is an error, not a warning, because a map
    # with holes in it must never reach a user.
    holes = int((seen == 0).sum())
    doubles = int((seen > 1).sum())
    if holes or doubles:
        raise RuntimeError(f"tiling wrote {holes} cells never and {doubles} cells twice; "
                           f"core={core} halo={halo} stride={stride}")
    return out
