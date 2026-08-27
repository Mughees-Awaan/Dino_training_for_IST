#!/usr/bin/env python3
"""
build_phash.py -- STEP 1b: THE PERCEPTUAL HASH PASS
====================================================

WHY THIS IS A SEPARATE STEP AND NOT PART OF build_manifest
    Everything in step 1 reads either the zip's index or the raw bytes of a member. Neither
    requires understanding what a JPEG means.

    This step is different: it has to actually DECODE the pictures. That is a different cost
    profile entirely -- about 7 minutes rather than about 1 -- so it gets its own script that
    can be run separately, re-run, interrupted, and resumed.

WHAT IT ANSWERS
    build_manifest's content_hash answers "are these the same FILE?".
    This answers "are these the same PICTURE?"

    Those are genuinely different questions. Take one photograph, open it, save it again at
    95% JPEG quality instead of 92%. Every single byte changes -- content_hash says
    "completely unrelated". A human looking at the two cannot tell them apart, and for
    leakage purposes they ARE the same photograph and must not be split across a divide.

    A perceptual hash catches exactly that case. See data/hashing.py for how it works.

WHY IT IS WORTH 7 MINUTES
    Before this existed, audit_duplicates guessed at near-duplicates using "same dataset, same
    dimensions, byte size within 2%" -- a crude proxy that its own comments admitted was only
    a candidate list needing human review. And lineage evidence rank 5 is literally named
    "perceptual", but was being filled by tile-overlap arithmetic instead, because nothing
    perceptual existed. This fills the slot the design always reserved for it.

RESUMABLE ON PURPOSE
    Results are written to their own table keyed by image_uid. Re-running skips anything
    already done, so an interrupted run costs only the work not yet finished. Given the
    machine this runs on loses power, that is not a nicety.

    python -m training.data.build_phash --workers 8
    python -m training.data.build_phash --write-manifest      # merge into manifest.parquet
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from training.data import hashing, schema  # noqa: E402


class _Archives:
    """Keeps zip handles open and shares them safely between worker threads.

    Opening a zip means parsing its index, which is slow for a big archive. Threads all read
    from the same handful of archives, so we open each one once and reuse it.

    THE LOCK: `zipfile` objects are NOT safe to use from several threads at once -- two
    threads seeking in the same underlying file will read each other's bytes and return
    garbage, without raising anything. The lock guards only the OPENING of a new handle and
    the per-read seek; the expensive part (decoding the JPEG) happens outside it, which is
    where the parallelism actually comes from.
    """

    def __init__(self, root: str):
        self.root = root
        self._z: dict[str, zipfile.ZipFile] = {}
        self._lock = Lock()

    def read(self, archive: str, member: str) -> bytes:
        with self._lock:
            z = self._z.get(archive)
            if z is None:
                z = self._z[archive] = zipfile.ZipFile(os.path.join(self.root, archive))
            return z.read(member)      # returns bytes; decoding happens outside the lock


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="tables/manifest.parquet")
    ap.add_argument("--archives", default=os.path.expanduser("~/workspace/gdrive_datasets"))
    ap.add_argument("--out", default="tables/phash.parquet")
    ap.add_argument("--workers", type=int, default=8,
                    help="decoding threads. PIL releases the interpreter lock while decoding, "
                         "so threads genuinely run in parallel here.")
    ap.add_argument("--limit", type=int, default=0, help="only do N photographs. 0 = all")
    ap.add_argument("--write-manifest", action="store_true",
                    help="merge the results back into the manifest's phash column")
    args = ap.parse_args()

    man = pd.read_parquet(args.manifest)
    schema.validate(man, schema.MANIFEST_COLUMNS, "manifest")

    # ---- resume: skip anything already hashed ------------------------------------------
    done: dict[str, str] = {}
    if os.path.exists(args.out):
        prev = pd.read_parquet(args.out)
        done = dict(zip(prev["image_uid"], prev["phash"]))
        print(f"resuming: {len(done):,} photographs already hashed")

    todo = man[~man["image_uid"].isin(done.keys())]
    if args.limit:
        todo = todo.head(args.limit)
    print(f"{len(man):,} photographs, {len(todo):,} to do, {args.workers} threads")
    if todo.empty:
        print("nothing to do")
        return 0

    arc = _Archives(args.archives)
    results: dict[str, str] = {}
    failed: list[tuple[str, str]] = []
    lock = Lock()
    t0 = time.time()

    def work(row):
        uid, archive, member = row
        try:
            data = arc.read(archive, member)
            return uid, hashing.phash64(data), None
        except Exception as exc:
            # A picture that will not decode gets an empty hash rather than stopping the run.
            # Downstream, an empty phash simply means "no perceptual evidence for this row",
            # which is a weaker claim -- never a wrong one.
            return uid, "", str(exc)[:80]

    rows = list(zip(todo["image_uid"], todo["archive"], todo["member"]))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for n, (uid, ph, err) in enumerate(pool.map(work, rows), 1):
            with lock:
                results[uid] = ph
                if err:
                    failed.append((uid, err))
            if n % 5000 == 0 or n == len(rows):
                rate = n / max(time.time() - t0, 1e-9)
                left = (len(rows) - n) / max(rate, 1e-9)
                print(f"  {n:,}/{len(rows):,}  {rate:.0f} img/s  "
                      f"~{left / 60:.1f} min remaining", flush=True)

    # ---- write -------------------------------------------------------------------------
    done.update(results)
    out = pd.DataFrame({"image_uid": list(done.keys()), "phash": list(done.values())})
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out.to_parquet(args.out, index=False)

    ok = out[out["phash"] != ""]
    print(f"\nwrote {args.out}  {len(out):,} rows")
    print(f"  hashed successfully {len(ok):,}")
    print(f"  distinct phashes    {ok['phash'].nunique():,}  "
          f"({len(ok) - ok['phash'].nunique():,} rows share a phash exactly)")
    if failed:
        print(f"  [warn] {len(failed):,} could not be decoded, e.g. {failed[0][1]}")

    if args.write_manifest:
        man = man.drop(columns=["phash"]).merge(out, on="image_uid", how="left")
        man["phash"] = man["phash"].fillna("")
        schema.validate(man, schema.MANIFEST_COLUMNS, "manifest")
        man.to_parquet(args.manifest, index=False)
        print(f"  manifest updated with phash")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
