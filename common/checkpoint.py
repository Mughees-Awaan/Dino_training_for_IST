#!/usr/bin/env python3
"""
checkpoint.py -- save and restore a run, with its provenance attached.

A checkpoint without provenance is a liability: you have weights and no idea what made
them. Every save here embeds the config hash, data digest and git SHA, and every load
checks them.
"""

from __future__ import annotations

import os

from training.common.provenance import check_resume, stamp
from training.common.safe_imports import require_torch


def save(path: str, model, optimizer=None, step: int = 0, cfg: dict | None = None,
         data_paths: list[str] | None = None, extra: dict | None = None):
    """Write a checkpoint atomically.

    ATOMIC MATTERS. Writing directly to the final path means a crash mid-write leaves a
    truncated file that looks like a checkpoint and fails hours later. Writing to a
    temporary name and renaming is atomic on POSIX: the file is either the old one or the
    complete new one, never a half.
    """
    torch = require_torch("checkpoint.save")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    blob = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "step": step,
        "provenance": stamp(cfg or {}, data_paths or []),
        "extra": extra or {},
    }
    tmp = path + ".tmp"
    torch.save(blob, tmp)
    os.replace(tmp, path)
    return path


def load(path: str, model=None, optimizer=None, cfg: dict | None = None,
         data_paths: list[str] | None = None, strict_provenance: bool = True):
    """Restore a checkpoint, refusing if the inputs have changed since it was written."""
    torch = require_torch("checkpoint.load")
    blob = torch.load(path, map_location="cpu", weights_only=False)

    if strict_provenance and cfg is not None:
        check_resume(blob.get("provenance", {}), stamp(cfg, data_paths or []), strict=True)

    if model is not None:
        model.load_state_dict(blob["model"])
    if optimizer is not None and blob.get("optimizer"):
        optimizer.load_state_dict(blob["optimizer"])
    return blob
