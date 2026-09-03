#!/usr/bin/env python3
"""
dataset.py -- HAND EPISODES TO THE TRAINER, ONE AT A TIME
==========================================================

WHAT A "DATASET" CLASS IS FOR
    Training reads examples one after another, hundreds of thousands of times. It does not
    want to know about zip archives, parquet tables, or augmentation. It wants to say
    "give me example number 4,712" and get back numbers.

    That is all this class is: a lookup from a number to a ready-to-use training example.

WHAT ONE EXAMPLE CONTAINS
    One episode, from build_episodes.py:

        support_image + support_x/y   the photograph the user "clicked" in, and the clicks
        query_image   + query_x/y     a DIFFERENT photograph, and the targets to be found

    Those two photographs are always different, always from the same split. That guarantee was
    established in build_episodes.py and verified there; this file simply reads the result.

THE ONE THING THIS FILE MUST NOT GET WRONG
    Augmentation moves the pixels around (see transforms.py). If the pixels move and the
    CLICKS do not, the model is being taught to find plants at the wrong coordinates -- and
    nothing errors.

    So `_load` below never touches the image without pushing the coordinates through the same
    `Applied` record in the same breath. The two lines sit together on purpose; splitting them
    up is how the bug gets reintroduced.

WHY IT READS STRAIGHT FROM THE ZIPS
    Nothing was ever extracted to disk (see build_manifest.py). Reading a zip member costs a
    JPEG decode, which any loader pays regardless. It saves duplicating 41 GB against 12 GB
    of free space.

WHY THERE IS NO PyTorch HERE
    This class deliberately imports no deep-learning framework. It returns plain numpy arrays
    in a plain dictionary. Whatever trainer is used later wraps it in about five lines. That
    keeps the data logic testable without a GPU, a framework, or a model.
"""

from __future__ import annotations

import os
import random

import numpy as np
import pandas as pd

from training.data import transforms
from training.runtime import preprocessing


class EpisodeDataset:
    """One practice session per item. Framework-agnostic; wrap for the trainer in use."""

    def __init__(self, episodes_path: str, manifest_path: str,
                 archive_root: str = os.path.expanduser("~/workspace/gdrive_datasets"),
                 augment: bool = True, seed: int = 0):
        self.ep = pd.read_parquet(episodes_path)
        man = pd.read_parquet(manifest_path)

        # A lookup from photograph id -> (which zip, which path inside it). Built ONCE here,
        # not per example: the manifest has 82,099 rows, and searching it for every one of
        # hundreds of thousands of reads would dominate the entire training run.
        self.where = {u: (a, m) for u, a, m in
                      zip(man["image_uid"], man["archive"], man["member"])}

        self.root = archive_root
        self.augment = augment
        # A seeded generator, so a training run is reproducible: same seed, same wobbles, in
        # the same order. Without it two runs of "the same" experiment are not comparable.
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        """How many examples exist. Python calls this when you write len(dataset)."""
        return len(self.ep)

    def _load(self, uid: str, xs, ys):
        """Read one photograph and its points, wobble both together, return model-ready numbers."""
        archive, member = self.where[uid]
        rgb = preprocessing.read_image(self.root, archive, member)

        # Capture the ORIGINAL size before anything changes it. map_points needs the size the
        # points were measured against -- a rotation swaps width and height, so asking
        # afterwards would give the wrong numbers.
        h, w = rgb.shape[:2]

        if self.augment:
            # THESE TWO LINES BELONG TOGETHER. The first moves the pixels; the second moves
            # the clicks by the same recorded decision. Separating them is the bug.
            rgb, applied = transforms.apply(rgb, self.rng)
            xs, ys = applied.map_points(xs, ys, w, h)

        return preprocessing.normalise(rgb), np.asarray(xs, np.float32), np.asarray(ys, np.float32)

    def __getitem__(self, i: int) -> dict:
        """Return example number `i`. Python calls this when you write dataset[i]."""
        r = self.ep.iloc[i]     # .iloc[i] = row number i, regardless of any index labels

        # The support and query photographs are loaded SEPARATELY, so each gets its own
        # independent wobble. That is correct and deliberate: in the real product the clicked
        # photograph and the searched photograph are genuinely different images with different
        # lighting and orientation. Applying one shared wobble would make them artificially
        # more alike than they will ever be at run time.
        sup_img, sup_x, sup_y = self._load(r["support_uid"], r["support_x"], r["support_y"])
        qry_img, qry_x, qry_y = self._load(r["query_uid"], r["query_x"], r["query_y"])

        return {
            "episode_id": r["episode_id"],
            "label": r["label"],
            "support_image": sup_img, "support_x": sup_x, "support_y": sup_y,
            "query_image": qry_img, "query_x": qry_x, "query_y": qry_y,
            # Carried through so evaluation can report same-family episodes as their own
            # slice -- they are realistic but easier. See build_episodes.py.
            "same_family": int(r["same_family"]),
        }
