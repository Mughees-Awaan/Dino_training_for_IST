#!/usr/bin/env python3
"""
hashing.py -- HOW WE DECIDE TWO PHOTOGRAPHS ARE THE SAME
=========================================================

There are TWO completely different questions here, and they need two different tools.

    QUESTION 1: "are these the same FILE?"     -> content_hash()  (xxHash)
    QUESTION 2: "are these the same PICTURE?"  -> phash64()       (perceptual hash)

Question 1 is about bytes. A file that has been copied is byte-for-byte identical.
Question 2 is about appearance. The same photograph re-saved by a different tool, or
re-compressed at a different quality, has COMPLETELY different bytes but looks identical to a
human. Question 1 will say "different"; question 2 will say "same".

Both matter. Both feed build_lineage_groups.py, at different evidence ranks.


---------------------------------------------------------------------------------------
WHY WE MOVED OFF CRC32
---------------------------------------------------------------------------------------
The first version of this pipeline used CRC32 -- a small checksum that the ZIP format already
stores for every file, so it was completely free to read. Paired with the file's byte size it
gave a usable duplicate key with zero disk reads.

It was the wrong tool, for one reason: CRC32 IS NOT A HASH. It is an ERROR-DETECTION CODE. It
was designed to notice a few flipped bits on a noisy wire, not to be hard to collide. It is
only 32 bits wide, which means:

    with 82,099 items, the chance of at least one CRC32 collision is 54%.

Pairing it with the file size hides most of that -- a collision then needs BOTH the same
checksum AND the same byte length -- and we verified on this corpus that it happened to be
correct: 1,500 sampled groups covering 4,978 real files, checked against a real hash, gave
ZERO wrong merges and ZERO missed duplicates.

So it was not broken. But "we got away with it" is not a guarantee, and it does not scale --
add another 50,000 photographs and there is no bound at all on how wrong it can be. Two
things it would break, silently:

    - build_lineage_groups would weld two UNRELATED photographs into one family
    - build_episodes would DROP a genuinely distinct photograph, thinking it was a copy

The cost of doing it properly turned out to be almost nothing (see below), so we do it
properly.


---------------------------------------------------------------------------------------
WHY THE PROPER FIX IS CHEAP HERE -- THE ARCHIVES ARE "STORED"
---------------------------------------------------------------------------------------
The objection to a real hash is obvious: CRC32 was free because the ZIP index already had it.
A real hash means READING ALL 41 GB.

Measured on these archives:

    compression method:            STORED  (i.e. no compression at all)
    compressed / original size:    1.0000
    read + hash throughput:        599 MB/s
    whole corpus:                  ~1.1 minutes

The zips are pure containers -- they hold already-compressed JPEGs, so the ZIP layer does not
compress anything. Reading a member is therefore plain disk I/O with NO decompression step.
About a minute, once, is not a reason to keep a 32-bit checksum.


---------------------------------------------------------------------------------------
WHY xxHash AND NOT SHA-256
---------------------------------------------------------------------------------------
We are asking "are these bytes the same?", not "could an attacker have forged these bytes?".
Nobody is attacking this corpus.

    SHA-256   cryptographic. Very strong, deliberately slow.       ~500 MB/s
    xxh3_128  non-cryptographic. 128 bits wide, extremely fast.    several GB/s

128 bits is far more than enough: the chance of an accidental collision among 82,099 items is
about 1 in 10^28. And xxh3 is fast enough that the disk, not the hash, is the bottleneck --
which is exactly what you want.
"""

from __future__ import annotations

import io

import numpy as np

# xxhash is a small C extension. If it is missing we fall back to blake2b, which ships with
# Python, is also 128-bit at this digest size, and is plenty fast. The pipeline should never
# fail just because an optional package is absent -- but it records WHICH was used, because a
# hash is only comparable with itself.
try:
    import xxhash
    _HAVE_XXHASH = True
except ImportError:                                   # pragma: no cover
    _HAVE_XXHASH = False

HASH_NAME = "xxh3_128" if _HAVE_XXHASH else "blake2b_128"


def content_hash(fh, chunk: int = 1 << 20) -> str:
    """QUESTION 1: exact bytes. Returns a 32-character hex string.

    `fh` is any open file-like object -- here, a member opened out of a zip.

    Read in 1 MB chunks rather than all at once. Some of these photographs are 40+ MB, and
    loading 82,099 of them whole would be pointless memory churn. A hash is designed to be fed
    incrementally, so chunking costs nothing.

    The walrus operator `:=` assigns and tests in one step:
        while c := fh.read(chunk):   ->   read a chunk, stop when it comes back empty
    """
    h = xxhash.xxh3_128() if _HAVE_XXHASH else __import__("hashlib").blake2b(digest_size=16)
    while c := fh.read(chunk):
        h.update(c)
    return h.hexdigest()


def phash64(data: bytes) -> str:
    """QUESTION 2: appearance. Returns a 16-character hex string standing for 64 bits.

    A "perceptual hash" is built so that two images that LOOK alike get similar codes, even
    when their bytes are completely different. Compare two of them by counting how many bits
    differ (see hamming below).

    THE FIVE STEPS, AND WHY EACH ONE IS THERE
    -----------------------------------------
    1. `im.draft("L", (64, 64))`
       A JPEG can be decoded at 1/2, 1/4 or 1/8 size almost for free, because of how JPEG
       stores data internally. `draft` asks for that. We only need a 32x32 thumbnail, so
       decoding a 4000x3000 photograph at full size would be pure waste.
       MEASURED: this is what takes the corpus from hours to ~7 minutes (187 images/second).

    2. Convert to greyscale, resize to 32x32.
       Colour is discarded deliberately -- we are asking "is this the same picture?", and a
       re-saved copy can shift colour slightly. Shrinking to 32x32 throws away fine detail and
       keeps overall structure, which is exactly the part that survives re-compression.

    3. DCT (Discrete Cosine Transform).
       This re-describes the image as "how much of each pattern, from very coarse to very
       fine, is present". It is the same mathematics JPEG itself uses. The coarse patterns are
       the ones a human recognises; the fine ones are mostly noise.

    4. Keep the top-left 8x8 -- the 64 coarsest patterns -- and throw away the rest.

    5. Compare each value to the MEDIAN and record 1 or 0.
       Using the median, rather than a fixed number, is what makes this survive brightness and
       contrast changes: brighten the whole image and every value shifts, but which ones are
       ABOVE THE MIDDLE does not.

       The very first value (`d[0]`, the "DC term") is the image's overall brightness. It is
       excluded from the median and forced to 0, because on its own it says nothing about
       content -- keeping it would just record how bright the photo was.

    MEASURED ON THIS CORPUS: unrelated pairs differ by 31.5 bits on average (about half of 64,
    which is what you expect from two unrelated codes), and the closest unrelated pair in a
    4,000-pair sample was 14 bits apart. Nothing unrelated came within 6 bits, which is why 6
    is a safe "these are the same picture" threshold.
    """
    import cv2
    from PIL import Image

    im = Image.open(io.BytesIO(data))
    try:
        im.draft("L", (64, 64))     # JPEG fast path; silently does nothing for PNG etc.
    except Exception:
        pass
    im = im.convert("L").resize((32, 32), Image.BILINEAR)

    a = np.asarray(im, np.float32)
    d = cv2.dct(a)[:8, :8].flatten()     # 64 coarsest patterns
    med = float(np.median(d[1:]))        # median EXCLUDING the brightness term
    bits = (d > med).astype(np.uint8)
    bits[0] = 0                          # and force the brightness term itself to 0

    v = 0
    for b in bits:                       # pack 64 ones and zeros into a single number
        v = (v << 1) | int(b)
    return f"{v:016x}"                   # as 16 hex characters


def hamming(a: str, b: str) -> int:
    """How many of the 64 bits differ between two perceptual hashes. 0 = identical.

    `int(a, 16)` reads a hex string as a number. `^` is XOR: it gives a 1 in every position
    where the two numbers DISAGREE. `bin(...).count("1")` then just counts those positions.

    Rough guide, from the measurements above:
        0        identical picture
        1-6      the same picture, re-saved / re-compressed / lightly edited
        7-13     possibly related -- worth a human look
        14+      unrelated (the closest unrelated pair we measured was 14)
    """
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def dedup_key(df):
    """Which column should be trusted to answer "is this the same file?".

    Prefer the real hash. Fall back to the old CRC32-and-size key only for a table built
    before content_hash existed, so an old manifest still works instead of crashing.

    Returns a pandas Series, ready to group by.
    """
    if "content_hash" in df.columns:
        s = df["content_hash"].fillna("")
        if (s != "").any():
            # Where a hash is missing (an unreadable member), fall back per-row rather than
            # letting every unhashed row collapse into one giant fake "duplicate" group.
            return s.where(s != "", df["content_key"])
    return df["content_key"]
