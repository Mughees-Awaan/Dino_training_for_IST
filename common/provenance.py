#!/usr/bin/env python3
"""
provenance.py -- record exactly what produced every artifact.

THE FAILURE THIS PREVENTS
    Six weeks from now there will be a checkpoint on disk and a number in a report, and
    somebody will ask "which data was this trained on?". Without provenance the honest
    answer is "we think the August tables, probably".

    Worse is the silent case: a run is interrupted, the tables are rebuilt in the
    meantime, and the run RESUMES against different data than it started on. Half the
    training saw one dataset and half saw another. Nothing errors. The checkpoint is
    quietly meaningless.

    So every artifact records three things, and resume REFUSES if any has changed.

WHAT GETS RECORDED
    config_hash   the exact settings, hashed
    data_digest   a hash of the input tables (their bytes, not their names)
    git_sha       the code commit
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:16]


def config_hash(cfg: dict) -> str:
    """Hash a settings dictionary.

    sort_keys=True is essential: {"a":1,"b":2} and {"b":2,"a":1} are the same settings and
    must give the same hash. Without it, re-ordering a config file would look like a
    different experiment.
    """
    return _sha(json.dumps(cfg, sort_keys=True, default=str).encode())


def file_digest(path: str, chunk: int = 1 << 20) -> str:
    """Hash a file's CONTENTS. Not its name, size or timestamp -- all of which can match
    while the contents differ."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while b := fh.read(chunk):
            h.update(b)
    return h.hexdigest()[:16]


def data_digest(paths: list[str]) -> str:
    """One digest for a whole set of input tables.

    Sorted so the order the caller happens to list them in cannot change the answer.
    A missing file becomes "MISSING" rather than an exception -- the mismatch should be
    reported by the resume check, with context, not raised here.
    """
    parts = []
    for p in sorted(paths):
        parts.append(f"{os.path.basename(p)}:"
                     f"{file_digest(p) if os.path.exists(p) else 'MISSING'}")
    return _sha("|".join(parts).encode())


def git_sha(default: str = "unknown") -> str:
    """The current commit, so a result can be traced to the exact code that made it."""
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        if out.returncode == 0:
            sha = out.stdout.strip()
            dirty = subprocess.run(["git", "status", "--porcelain"],
                                   capture_output=True, text=True, timeout=5)
            # "-dirty" matters: a commit id alone is a lie if the tree was modified.
            return sha + ("-dirty" if dirty.stdout.strip() else "")
    except Exception:
        pass
    return default


def stamp(cfg: dict, data_paths: list[str]) -> dict:
    """The provenance block written into every checkpoint and every result file."""
    return {"config_hash": config_hash(cfg), "data_digest": data_digest(data_paths),
            "git_sha": git_sha(), "schema_version": 2}


def check_resume(previous: dict, current: dict, strict: bool = True) -> list[str]:
    """Compare the stamp a run STARTED with against the one it is resuming into.

    Returns the list of things that changed. `strict` raises instead, which is the right
    default for training: continuing across a data change silently corrupts the run in a
    way no later check can detect.
    """
    changed = [k for k in ("config_hash", "data_digest", "git_sha", "schema_version")
               if previous.get(k) != current.get(k)]
    if changed and strict:
        detail = "\n".join(f"    {k}: {previous.get(k)} -> {current.get(k)}" for k in changed)
        raise RuntimeError(
            f"refusing to resume: the run's inputs changed since it started.\n{detail}\n"
            f"Half of this run would have seen different data from the other half.\n"
            f"Start a fresh run, or pass strict=False if you are certain.")
    return changed
