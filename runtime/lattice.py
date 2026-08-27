#!/usr/bin/env python3
"""
lattice.py -- WHICH PIXEL BELONGS TO WHICH OUTPUT SQUARE
=========================================================

THE IDEA IN PLAIN TERMS
    The model does not give an answer for every single pixel. That would be enormously
    wasteful -- neighbouring pixels are nearly identical.

    Instead it divides the photograph into a grid of small squares -- 16 pixels across, say --
    and produces ONE description for each square. A 1600x1600 photograph becomes a 100x100
    grid of descriptions.

    That number, 16, is called the STRIDE: how far apart the squares are.

WHY THIS DESERVES ITS OWN FILE
    Everything downstream has to agree, exactly, on which pixels a square covers.

    The user clicks at pixel (412, 88). Which square is that? The model finds something
    interesting in square (37, 12). Which pixel do we report to the user?

    Get this wrong by ONE SQUARE and the user's click is matched with the description of the
    patch of soil next to their plant. The model then learns "this plant looks like soil".
    And -- the recurring theme of this pipeline -- NOTHING ERRORS. The shapes all match. The
    code runs. The results are just quietly wrong.

    So the convention is written down ONCE, here, and every other file imports these functions
    rather than doing the arithmetic itself.

THE CONVENTION, STATED ONCE

    square (col, row) covers pixels  [col*stride, (col+1)*stride)  x  [row*stride, (row+1)*stride)
    its CENTRE is at                 ((col + 0.5) * stride, (row + 0.5) * stride)
    pixel (x, y) belongs to square   (floor(x / stride), floor(y / stride))

    The square bracket and round bracket are standard mathematical notation: [a, b) means
    "from a up to but NOT INCLUDING b". So with stride 16, square 0 covers pixels 0 to 15, and
    pixel 16 is the first pixel of square 1. No pixel belongs to two squares, and none belongs
    to none.
"""

from __future__ import annotations

import numpy as np


def grid_shape(height: int, width: int, stride: int) -> tuple[int, int]:
    """How many WHOLE squares fit in a photograph of this size.

    `//` is integer division -- it divides and throws away the remainder.
    1000 // 16 = 62, because 62 whole squares of 16 fit, with 8 pixels left over.

    Those 8 leftover pixels along the edge produce NO square. That is deliberate: a partial
    square would describe less ground than a full one, so its description would not be
    comparable with the others. Better to have no answer at the very edge than a subtly
    different one that looks the same.
    """
    return height // stride, width // stride


def pixel_to_cell(x, y, stride: int):
    """Pixel -> the square containing it. Works on one number or a whole array of them.

    np.floor rounds DOWN. Not round-to-nearest -- down. Pixel 31 with stride 16 gives
    31/16 = 1.94, floor 1, so it is in square 1. Round-to-nearest would say square 2, which
    would mean the boundary between squares sat in the middle of a square. Consistently wrong
    by half a square across the entire image.
    """
    return np.floor(np.asarray(x) / stride).astype(np.int64), \
           np.floor(np.asarray(y) / stride).astype(np.int64)


def cell_to_pixel(col, row, stride: int):
    """Square -> the pixel at its CENTRE. This is the pixel a detection reports.

    The + 0.5 is the whole point. Square 3 with stride 16 covers pixels 48 to 63, and its
    centre is 56 -- not 48. Returning the corner instead would shift every single detection
    up and to the left by half a square, systematically, in every result the product ever
    produces.
    """
    return (np.asarray(col) + 0.5) * stride, (np.asarray(row) + 0.5) * stride


def check_halo(halo: int, stride: int) -> None:
    """Refuse a halo that is not a whole number of squares.

    WHAT A HALO IS (the full story is in tiler.py)
        When a huge photograph is cut into tiles, each tile is read with a margin of extra
        pixels around it, so squares near the tile's edge still have context to look at. That
        margin is the halo. It is thrown away after the model runs.

    WHY IT MUST DIVIDE EXACTLY BY THE STRIDE
        You throw the halo away by discarding whole squares. If the halo is 144 pixels and the
        stride is 32, then 144 / 32 = 4.5 squares. You cannot discard half a square.

        Discard 4 and you have kept 16 pixels of halo, shifting everything after it by 16
        pixels. Discard 5 and you have thrown away 16 pixels of real content, leaving a hole.

        Either way the grid is silently misaligned for the whole rest of the image.

    THIS IS A REAL DEFECT IN THE CURRENT CODEBASE, which is why the error message names the
    two nearest values that would work, instead of just saying "invalid".
    """
    if halo % stride != 0:      # `%` gives the remainder: 144 % 32 = 16
        raise ValueError(
            f"halo {halo} is not a multiple of stride {stride} "
            f"({halo} % {stride} = {halo % stride}). Every interior tile would shift by "
            f"{halo % stride} px and the edges would be left unwritten. "
            f"Use {halo - halo % stride} or {halo + stride - halo % stride}.")


def round_halo(halo: int, stride: int) -> int:
    """Round a halo UP to the next whole number of squares.

    Always up, never down: the halo exists to give edge squares enough context, so erring
    towards more context is harmless while erring towards less reintroduces the seam the halo
    was there to remove.
    """
    return halo if halo % stride == 0 else halo + stride - halo % stride
