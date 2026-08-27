#!/usr/bin/env python3
"""
preprocessing.py -- ONE SINGLE WAY TO PREPARE AN IMAGE
======================================================

WHY THIS TINY FILE IS SHARED WITH THE PRODUCT
    Before a model sees a photograph, the numbers have to be put in the exact form it expects.
    That is a handful of lines of arithmetic. It is very tempting for the training code and
    the shipped application to each have their own copy.

    Do that and, sooner or later, the two copies disagree -- by a constant, by a channel order,
    by a divide-by-255. Now the model was TRAINED on one kind of input and RUNS on another.

    The result is the worst kind of failure: both sides look completely correct in isolation,
    the application produces plausible-looking output, and the model is simply worse than it
    should be for a reason nobody can see. There is no error to grep for.

    So there is exactly ONE implementation, here, and both sides import it.

WHY IMAGES ARE READ STRAIGHT OUT OF THE ZIPS
    Nothing was ever extracted to disk (see build_manifest.py). So this file opens zip members
    directly. That is affordable because the members are already-compressed JPEGs: reading one
    costs a JPEG decode, which ANY image loader has to pay anyway, from a folder or a zip
    alike. The zip adds essentially nothing.

    What it saves is 41 GB of duplicated disk, against 12 GB free.
"""

from __future__ import annotations

import io
import os
import zipfile
from functools import lru_cache

import numpy as np

# The average brightness and spread of each colour channel across ImageNet, the enormous
# photograph collection the backbone was originally trained on.
#
# WHY SUBTRACT AND DIVIDE BY THESE
#   Neural networks work best when their inputs are roughly centred on zero and roughly the
#   same size in every channel. Raw pixels are 0-255 and lopsided -- outdoor photographs are
#   greener than they are blue.
#
#   Subtracting the mean centres it; dividing by the standard deviation makes each channel
#   about the same width. These SPECIFIC numbers are not a preference: they are what the model
#   was trained with. Change them and every learned feature is being fed slightly the wrong
#   thing.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)   # red, green, blue
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


@lru_cache(maxsize=64)
def _archive(root: str, name: str) -> zipfile.ZipFile:
    """Open a zip once and keep it open.

    @lru_cache is a Python decorator that REMEMBERS results. Call this with the same archive
    name twice and the second call returns the already-open handle instead of re-opening it.

    Why it matters: opening a zip means reading and parsing its table of contents, which for
    a large archive is thousands of entries. A training run reads hundreds of thousands of
    photographs; re-opening the archive each time would dominate the cost entirely.

    maxsize=64 keeps the 64 most recently used archives open and quietly closes the rest --
    enough to cover a normal working set without holding all 213 open at once. ("LRU" =
    Least Recently Used: when it is full, the one untouched longest is dropped.)
    """
    return zipfile.ZipFile(os.path.join(root, name))


def read_image(root: str, archive: str, member: str) -> np.ndarray:
    """Read ONE photograph straight out of its archive. Returns RGB, values 0-255.

    `root`    the folder holding the .zip files
    `archive` which zip, e.g. "Apple Gopro.zip"
    `member`  the path inside it, e.g. "task_0/data/GX010004_frame_6645.jpg"

    Those last two are precisely the `archive` and `member` columns from the manifest.
    """
    from PIL import Image     # imported here, not at the top, so a caller that only needs
                              # normalise() does not pay for loading the imaging library
    with _archive(root, archive).open(member) as fh:
        # io.BytesIO wraps the raw bytes so PIL can treat them like a file. We read the whole
        # member first because PIL needs to seek backwards and forwards while decoding, which
        # a zip member stream does not support.
        img = Image.open(io.BytesIO(fh.read()))
        # .convert("RGB") forces a consistent 3-channel result. Without it, a greyscale
        # photograph comes back with 1 channel and a PNG with transparency comes back with 4,
        # and the shape mismatch surfaces much later, deep inside the model.
        return np.asarray(img.convert("RGB"))


def normalise(rgb: np.ndarray, mean=IMAGENET_MEAN, std=IMAGENET_STD) -> np.ndarray:
    """Turn a plain photograph into exactly the numbers the model expects.

    Three changes, in order:

      1. 0-255 whole numbers  ->  0.0-1.0 decimals
      2. centre and scale each colour channel using the constants above
      3. reorder the dimensions from (height, width, channel) to (channel, height, width)

    Step 3 catches people out. Image libraries store a picture as "a grid of pixels, each
    holding 3 numbers". Neural networks want "3 separate grids, one per colour". Same data,
    different arrangement -- and getting it wrong does not crash, it just feeds the model
    garbage that happens to be the right size.
    """
    x = rgb.astype(np.float32) / 255.0
    x = (x - mean) / std
    # .transpose(2, 0, 1) means "the new order of dimensions is: old #2, old #0, old #1"
    # -- so (height, width, channel) becomes (channel, height, width).
    #
    # ascontiguousarray then makes a real, normally-laid-out copy. transpose alone only
    # changes how the SAME memory is interpreted; some libraries reject such arrays, and
    # others silently take a slow path.
    return np.ascontiguousarray(x.transpose(2, 0, 1))


def prepare(root: str, archive: str, member: str, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """Read and normalise in one call -- the shortcut most callers actually want."""
    return normalise(read_image(root, archive, member), mean, std)
