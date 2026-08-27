#!/usr/bin/env python3
"""
build_manifest.py -- STEP 1: MAKE THE MASTER LIST
=================================================

WHAT THIS SCRIPT DOES, IN ONE SENTENCE
    It opens all 213 zip archives, and writes down one line for every photograph inside them
    (82,099 lines) plus one line for every object an annotator marked (879,253 lines).

WHAT IT DOES *NOT* DO
    It never unzips a photograph. Not one. This is the single most important thing to
    understand about this script, so here is why:

        - The archives are 41 GB. Unzipping them would need about another 41 GB.
        - We only have 12 GB of free disk.
        - The photographs are JPEGs. JPEG is ALREADY compressed, so putting them in a zip
          barely shrinks them (measured: 1.00x -- zero saving). The zip here is just a
          container, like a cardboard box.

    So instead of taking the photographs out of the box, we write down WHICH BOX and WHICH
    SHELF each one is on, and later scripts reach into the box for one photograph at a time.
    That is what the `archive` and `member` columns are.

HOW CAN WE KNOW A PHOTOGRAPH'S SIZE WITHOUT OPENING IT?
    Every zip file has a table of contents at the end, called the "central directory". It
    lists every file inside, with its name, its byte size, and a CRC32 checksum -- a short
    number computed from the file's contents. Python reads that table instantly. We also read
    a handful of tiny JSON files that the annotation tool (CVAT) puts inside each archive.
    That is all. 41 GB catalogued in minutes.

WHAT IS A "CVAT ARCHIVE"?
    CVAT is the web tool the annotation team used to draw on the photographs. When you export
    a project, you get a zip laid out like this:

        project.json                       <- description of the whole project
        task_0/
            task.json                      <- the job's name and its progress status
            annotations.json               <- EVERY mark drawn in this job
            data/manifest.jsonl            <- the list of photographs, IN ORDER
            data/GX010004_frame_6645.jpg   <- the photographs themselves
            data/GX010004_frame_6720.jpg
        task_1/
            ...

    A "task" is one batch of work handed to one annotator.

THE TRAP THAT COST US A DAY
    Inside annotations.json, marks are attached to photographs by NUMBER, not by filename:
    "frame 7 has a garlic plant at (412, 88)". So we must know which photograph is frame 7.

    The obvious guess -- "the 8th file listed in the zip" -- is WRONG. CVAT numbers frames by
    their order in data/manifest.jsonl, and the zip stores files in a completely different
    order. In one task we checked, 46 of 48 positions differed.

    Nothing crashes when you get this wrong. The tables build fine. The counts look right.
    The marks are simply attached to the wrong photographs, silently. We only caught it by
    DRAWING the clicks onto the photographs and noticing they landed on bare soil.
    See read_frame_sizes() below, and tests/test_frame_order.py.

HOW TO RUN IT
    python -m training.data.build_manifest --archives ~/workspace/gdrive_datasets \
                                           --out tables/manifest.parquet

WHAT COMES OUT
    tables/manifest.parquet   -- 82,099 rows, one per photograph
    tables/instances.parquet  -- 879,253 rows, one per marked object

    Parquet is a file format for tables. Think "a spreadsheet, but built for large data":
    it stores each column separately, which makes it small on disk and fast to load.
"""

from __future__ import annotations

import argparse      # reads the options you type after the script name
import hashlib       # makes short fixed-length codes ("hashes") out of text
import json          # reads .json files
import os            # file paths
import re            # "regular expressions" -- pattern matching in text
import sys
import zipfile       # opens .zip archives WITHOUT unpacking them

import pandas as pd  # the spreadsheet-in-memory library

# Make `from training.data import schema` work no matter what folder you run this from.
# It walks three folders up from this file (data -> training -> updated_scripts) and adds
# that to the list of places Python looks for code.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from training.data import hashing, schema  # noqa: E402


# =====================================================================================
# PATTERNS FOR RECOGNISING THINGS IN FILENAMES
# =====================================================================================
# A "regular expression" is a small pattern language for finding shapes in text.
# Quick key:  \d = any digit    + = one or more    ( ) = remember this bit
#             ^ = start of text  $ = end of text   re.I = ignore upper/lower case

IMG_RE = re.compile(r"\.(jpg|jpeg|png|tif|tiff|bmp|webp)$", re.I)   # is this file a picture?
TASK_RE = re.compile(r"^(task_\d+)/")                               # which task folder?

# ---- Patterns that pull provenance out of filenames --------------------------------------
# Drone software bakes useful facts into filenames. These patterns dig them back out.
# Example: "DJI_20250602104520_0007_D.JPG"
#            |         |        |    |
#            brand  when taken  flight seq  camera band

DJI_TS_RE = re.compile(r"DJI_(\d{14})_(\d{4})")   # 14 digits = YYYYMMDDHHMMSS, then a sequence
DJI_SEQ_RE = re.compile(r"DJI_(\d{4})(?!\d)")     # older naming: just a 4-digit sequence
                                                  # (?!\d) means "not followed by another digit"
TILE_XY_RE = re.compile(r"\((\d+)\s*,\s*(\d+)\)") # "(1024, 2048)" -- position in a big map
TILE_RC_RE = re.compile(r"tile_r(\d+)_c(\d+)", re.I)          # "tile_r3_c7" -- row/column
GEO_RE = re.compile(r"splitted_\d+_(-?\d+\.\d+)_(-?\d+\.\d+)") # real-world coordinates
CROP_XY_RE = re.compile(r"crop_(\d+)_(\d+)_(\d+)")             # "crop_5_1024_2048"
BAND_RE = re.compile(r"_(D|W|V)(?=[._]|$)")       # camera band letter at the end of the name


def parse_name(stem: str) -> dict:
    """Read whatever facts the filename is willing to tell us.

    `stem` is the filename with the extension removed, e.g. "DJI_20250602104520_0007_D".

    RULE: if a fact is not in the name, the value stays as an empty string "". We never
    guess. A wrong guess about which flight a photo came from would silently corrupt the
    train/test split later on, and nobody would notice.
    """
    # Start with everything blank, then fill in whatever we can find.
    out = dict(sensor="", capture_datetime="", capture_year="", flight_id="",
               source_mosaic="", band="", tile_x="", tile_y="", tile_row="", tile_col="",
               geo_x="", geo_y="")

    # --- Case 1: a modern DJI name with a full timestamp ---------------------------------
    m = DJI_TS_RE.search(stem)   # .search() returns None if the pattern is not found
    if m:
        ts, seq = m.group(1), m.group(2)   # group(1) is the 14 digits, group(2) the sequence
        out.update(
            sensor="DJI",
            # Chop the 14 digits into a readable date. ts[0:4] means "characters 0,1,2,3".
            #   20250602104520  ->  2025-06-02 10:45:20
            capture_datetime=f"{ts[0:4]}-{ts[4:6]}-{ts[6:8]} {ts[8:10]}:{ts[10:12]}:{ts[12:14]}",
            capture_year=ts[0:4],
            flight_id=f"DJI_{ts}",           # photos from one flight share this
            source_mosaic=f"DJI_{ts}_{seq}", # tiles cut from one map share this
        )
    else:
        # --- Case 2: an older DJI name with only a sequence number ------------------------
        m = DJI_SEQ_RE.search(stem)
        if m:
            out.update(sensor="DJI", source_mosaic=f"DJI_{m.group(1)}")

    # A different drone model announces itself in plain words.
    if "phantom" in stem.lower():
        out["sensor"] = "DJI Phantom"

    # Which camera on a multi-camera drone.
    m = BAND_RE.search(stem)
    if m:
        out["band"] = m.group(1)

    # --- Position of this tile inside a bigger map, written as "(x, y)" -------------------
    m = TILE_XY_RE.search(stem)
    if m:
        out["tile_x"], out["tile_y"] = m.group(1), m.group(2)
        if not out["source_mosaic"]:
            # The text BEFORE the "(x, y)" part names the big map this tile came from.
            # m.start() is where the "(" was found, so stem[:m.start()] is everything before.
            # Then strip trailing spaces/underscores and any trailing "_123".
            head = re.sub(r"_\d+$", "", stem[: m.start()].rstrip("_ "))
            out["source_mosaic"] = head

    # Some datasets number tiles by row and column instead of by pixel position.
    m = TILE_RC_RE.search(stem)
    if m:
        out["tile_row"], out["tile_col"] = m.group(1), m.group(2)

    # Real-world map coordinates, when the exporter embedded them.
    m = GEO_RE.search(stem)
    if m:
        out["geo_x"], out["geo_y"] = m.group(1), m.group(2)

    # A third naming convention for the same idea.
    m = CROP_XY_RE.search(stem)
    if m and not out["source_mosaic"]:
        out["source_mosaic"] = f"crop_{m.group(1)}"
        out["tile_x"], out["tile_y"] = m.group(2), m.group(3)

    return out


def _num(v, cast=int, default=-1):
    """Turn text into a number, and return `default` instead of crashing if it isn't one.

    Example: _num("2025") -> 2025      _num("") -> -1      _num("abc") -> -1

    We use -1 rather than 0 as "unknown", because 0 is a legitimate tile position and we must
    be able to tell "this tile is at x=0" apart from "we have no idea where this tile is".
    """
    try:
        return cast(v)
    except (TypeError, ValueError):
        return default


def read_task_json(z: zipfile.ZipFile, task: str) -> dict:
    """Read task_N/task.json -- the job's name and its progress status.

    `z` is an OPEN zip file. `z.read("path/inside")` pulls out one small member's bytes; it
    does not unpack the archive.

    If the file is missing or malformed we return {} rather than crashing: one broken task
    out of 948 should not stop the other 947 from being catalogued.
    """
    try:
        d = json.loads(z.read(f"{task}/task.json"))
    except (KeyError, json.JSONDecodeError):
        # KeyError       = there is no such member in the zip
        # JSONDecodeError = the member exists but isn't valid JSON
        return {}
    return {"task_name": str(d.get("name", "")),
            "task_status": str(d.get("status", ""))}


def read_frame_sizes(z: zipfile.ZipFile, task: str):
    """Read task_N/data/manifest.jsonl -- gives us BOTH the picture sizes AND the frame order.

    ================================ THIS IS THE IMPORTANT ONE ================================

    A ".jsonl" file is "JSON Lines": one complete JSON object per line, e.g.

        {"name": "GX010004_frame_6645", "extension": ".jpg", "width": 3840, "height": 2160}
        {"name": "GX010004_frame_6720", "extension": ".jpg", "width": 3840, "height": 2160}

    Two things come out of it.

    1. WIDTH AND HEIGHT, without decoding a single JPEG. That alone saves hours.

    2. THE ORDER. The first line is frame 0, the second is frame 1, and so on. THIS is what
       annotations.json means when it says "frame 7". The order the files happen to sit in
       inside the zip is unrelated -- in a task we sampled, 46 of 48 positions differed.

       Get this wrong and every annotation attaches to the wrong photograph. Nothing errors.
       The tables look perfect. The clicks just land on bare soil. This is exactly what
       happened to us, and it was found by rendering the episodes as pictures and looking at
       them. tests/test_frame_order.py now guards it.

    Returns two lookup tables:
        sizes  = {"GX010004_frame_6645.jpg": (3840, 2160), ...}
        order  = {"GX010004_frame_6645.jpg": 0, "GX010004_frame_6720.jpg": 1, ...}
    """
    sizes, order = {}, {}
    try:
        # .decode() turns raw bytes into text. "replace" means "if a byte isn't valid text,
        # put a placeholder character rather than crashing" -- a stray byte in one line must
        # not lose us an entire task.
        raw = z.read(f"{task}/data/manifest.jsonl").decode("utf-8", "replace")
    except KeyError:
        return sizes, order   # no manifest.jsonl in this task -- caller has a fallback

    for line in raw.splitlines():
        line = line.strip()
        # The first lines of the file are headers, not picture entries. Real entries start
        # with "{", so skip anything that doesn't.
        if not line or not line.startswith("{"):
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue          # one bad line skipped; the rest of the file still counts
        name = d.get("name")
        if name is None:
            continue

        # Some exports put the extension in a separate field, some include it in "name".
        # Glue them together only if it isn't already there.
        ext = d.get("extension", "")
        key = f"{name}{ext}" if ext and not str(name).endswith(str(ext)) else str(name)
        base = os.path.basename(key)   # drop any folders, keep just "picture.jpg"

        sizes[base] = (_num(d.get("width")), _num(d.get("height")))
        # setdefault(base, len(order)) means: "if we haven't seen this name yet, give it the
        # next number". len(order) is 0 for the first name, 1 for the second, and so on --
        # which is exactly the frame numbering CVAT uses. If a name repeats, the FIRST
        # position wins, which is also what CVAT does.
        order.setdefault(base, len(order))

    return sizes, order


def read_shapes(z: zipfile.ZipFile, task: str) -> dict:
    """Read task_N/annotations.json -- every mark an annotator drew in this job.

    Returns {frame_number: [mark, mark, ...]}.

    WHY WE READ IT ONCE AND KEEP EVERYTHING
        annotations.json is by far the biggest JSON in an archive. The manifest table needs
        the COUNT of marks per photograph; the instances table needs the actual COORDINATES.
        Reading the file twice would double the only slow part of the whole scan, so we parse
        it once and hand the full contents back.

    TWO KINDS OF MARK IN CVAT
        "shapes"  -- a single mark on a single frame. Straightforward.
        "tracks"  -- one object followed across several frames (e.g. the same plant seen in
                     consecutive video frames). Each keyframe of a track counts as its own
                     mark for us, because the model sees each frame separately.
    """
    per_frame: dict[int, list] = {}
    try:
        raw = json.loads(z.read(f"{task}/annotations.json"))
    except (KeyError, json.JSONDecodeError):
        return per_frame

    # Some exports wrap everything in a list, others give a single object. Handle both by
    # always treating it as a list.
    for block in (raw if isinstance(raw, list) else [raw]):
        if not isinstance(block, dict):
            continue

        # --- plain marks -----------------------------------------------------------------
        # `block.get("shapes") or []` means "the shapes list, or an empty list if there is
        # none or it is null". Without the `or []` this would crash on a task with no marks.
        for item in (block.get("shapes") or []):
            fr = item.get("frame")
            if fr is not None:
                # setdefault(key, []) means "get the list for this frame, creating an empty
                # one first if this is the first mark on that frame".
                per_frame.setdefault(int(fr), []).append(item)

        # --- tracked objects --------------------------------------------------------------
        for track in (block.get("tracks") or []):
            label = track.get("label", "")   # the label lives on the track, not each keyframe
            for item in (track.get("shapes") or []):
                fr = item.get("frame")
                if fr is None:
                    continue
                item = dict(item)            # copy it, so we don't modify the parsed JSON
                item.setdefault("label", label)   # give the keyframe the track's label
                per_frame.setdefault(int(fr), []).append(item)

    return per_frame


def shape_to_instances(shape: dict, image_uid: str, n: int) -> list[dict]:
    """Turn ONE mark from CVAT into one or more rows of our instances table.

    CVAT stores coordinates as one flat list, alternating x and y:
        points = [x1, y1, x2, y2, x3, y3, ...]

    WHY "one or more"
        A `points` mark can hold MANY dots at once -- an annotator clicking twenty plants in
        a row may produce a single points mark with forty numbers in it. For this product,
        one dot means one plant, so we split it into twenty separate instance rows.
        Rectangles and polygons describe a single object each, so they stay whole.
    """
    pts = shape.get("points") or []
    kind = str(shape.get("type", ""))    # "points" / "rectangle" / "polygon" / "polyline"
    label = str(shape.get("label", ""))
    out = []

    if kind == "points":
        # Step through the flat list two at a time: (x1,y1), then (x2,y2), ...
        # range(0, len(pts) - 1, 2) means "start at 0, step by 2, stop before the last item"
        # -- the -1 protects against a malformed odd-length list.
        for k in range(0, len(pts) - 1, 2):
            out.append(dict(x=float(pts[k]), y=float(pts[k + 1]),
                            w=0.0, h=0.0,       # a dot has no width or height
                            area_px=0.0, points_wkt=""))

    elif kind == "rectangle" and len(pts) >= 4:
        # CVAT gives two opposite corners, but not necessarily top-left then bottom-right --
        # if the annotator dragged upwards or leftwards they arrive swapped. min/max fixes it
        # so that (x1,y1) is always the top-left. Without this, w and h come out negative.
        x1, y1, x2, y2 = (float(v) for v in pts[:4])
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        out.append(dict(x=(x1 + x2) / 2,        # we store the CENTRE, not a corner,
                        y=(y1 + y2) / 2,        # so dots and boxes are comparable
                        w=x2 - x1, h=y2 - y1,
                        area_px=(x2 - x1) * (y2 - y1), points_wkt=""))

    elif kind in ("polygon", "polyline") and len(pts) >= 6:
        # 6 numbers = 3 corners = the fewest that can enclose an area.
        # pts[0::2] means "every other item starting at 0" -> all the x values.
        # pts[1::2] means "every other item starting at 1" -> all the y values.
        xs = [float(v) for v in pts[0::2]]
        ys = [float(v) for v in pts[1::2]]
        w, h = max(xs) - min(xs), max(ys) - min(ys)   # the box that just contains the shape

        # The "shoelace formula" computes the area of any polygon from its corners.
        # You multiply each corner's x by the NEXT corner's y, subtract the mirror product,
        # add them all up, and halve the absolute value. `% len(xs)` makes the last corner
        # wrap around to the first, closing the loop.
        # A polyline is an open line, not a closed shape, so it has no enclosed area --
        # for those we fall back to the area of the bounding box.
        area = abs(sum(xs[i] * ys[(i + 1) % len(xs)] - xs[(i + 1) % len(xs)] * ys[i]
                       for i in range(len(xs)))) / 2 if kind == "polygon" else w * h

        # Keep the FULL outline as text, so nothing is thrown away. WKT ("Well-Known Text")
        # is the standard way to write a shape as a string:
        #     POLYGON((10 20, 30 20, 30 40, 10 40))
        wkt = "POLYGON((" + ", ".join(f"{a} {b}" for a, b in zip(xs, ys)) + "))"

        out.append(dict(x=sum(xs) / len(xs),    # the average corner -- good enough as a centre
                        y=sum(ys) / len(ys),
                        w=w, h=h, area_px=area, points_wkt=wkt))

    # Wrap each piece of geometry into a full table row.
    rows = []
    for i, g in enumerate(out):   # enumerate gives us (0, first), (1, second), ...
        rows.append(dict(
            image_uid=image_uid,
            # A name that can never collide: photo id + which mark + which dot within it.
            instance_id=f"{image_uid}-{n}-{i}",
            label_raw=label,      # exactly what the annotator typed
            label_canon="",       # the tidied version -- filled in by normalize_labels.py
            shape_type=kind,
            is_crowd=0,
            **g,                  # ** unpacks the geometry dict into these keyword arguments
        ))
    return rows


def scan_archive(path: str, hash_mode: str = "xxh3"):
    """Catalogue ONE zip archive.

    Returns (photograph_rows, instance_rows).

    hash_mode="xxh3"   (default) also READ each member and compute a real 128-bit content
                       hash. Costs about a minute for the whole 41 GB corpus, because these
                       archives are STORED -- the ZIP layer does not compress, so reading a
                       member is plain disk I/O with no decompression. See data/hashing.py.
    hash_mode="crc32"  metadata only, exactly as before: the zip index and the small JSON
                       members, no photograph bytes read at all. Faster, but the duplicate
                       key is then only a 32-bit checksum. Use for a quick catalogue pass.
    """
    rows, instances = [], []
    dataset = os.path.splitext(os.path.basename(path))[0]   # "Apple Gopro.zip" -> "Apple Gopro"

    # `with` makes sure the zip is closed properly even if something goes wrong inside.
    with zipfile.ZipFile(path) as z:
        infos = z.infolist()   # the table of contents. Instant -- nothing is decompressed.

        # --- Which task folders exist in this archive? ------------------------------------
        # The walrus operator `:=` both runs the match AND stores the result, so we can test
        # it and use it in the same line. The braces {...} make a set, which removes
        # duplicates -- every file in task_0/ contributes "task_0", we want it once.
        tasks = sorted({m.group(1) for i in infos
                        if (m := TASK_RE.match(i.filename))})

        # --- Read all the small JSONs ONCE, up front --------------------------------------
        # Doing this now, rather than inside the per-photograph loop below, means each JSON
        # is parsed once instead of once per photograph.
        meta = {t: read_task_json(z, t) for t in tasks}          # names and statuses
        frames = {t: read_frame_sizes(z, t) for t in tasks}      # (sizes, order) per task
        sizes = {t: v[0] for t, v in frames.items()}             # pull the sizes half out
        order = {t: v[1] for t, v in frames.items()}             # pull the order half out
        shapes = {t: read_shapes(z, t) for t in tasks}           # all the marks

        # Fallback counter, used only for tasks that have no manifest.jsonl at all.
        per_task_index: dict[str, int] = {}

        # --- Now walk every entry in the archive ------------------------------------------
        for info in infos:
            # Skip folder entries and anything that isn't a picture (the JSONs, etc.)
            if info.is_dir() or not IMG_RE.search(info.filename):
                continue

            m = TASK_RE.match(info.filename)
            task = m.group(1) if m else ""
            base = os.path.basename(info.filename)     # "GX010004_frame_6645.jpg"
            stem, ext = os.path.splitext(base)         # ("GX010004_frame_6645", ".jpg")

            # ===== THE FRAME NUMBER -- see read_frame_sizes() for why this matters =========
            # Prefer the position recorded in manifest.jsonl. Only if this task has no
            # manifest.jsonl at all do we fall back to the order files arrive in, and even
            # then we start counting AFTER the known ones so the numbers cannot collide.
            known = order.get(task, {})
            if base in known:
                frame = known[base]
            else:
                frame = len(known) + per_task_index.get(task, 0)
                per_task_index[task] = per_task_index.get(task, 0) + 1

            # Look up this photograph's size and its marks.
            # .get(x, {}).get(y, default) is a safe two-step lookup: if either level is
            # missing we get the default instead of a crash.
            w, h = sizes.get(task, {}).get(base, (-1, -1))
            frame_shapes = shapes.get(task, {}).get(frame, [])

            # Read the member and hash it, unless we were asked for the metadata-only pass.
            # A member that will not open (a truncated archive, a bad entry) leaves the hash
            # empty rather than aborting the whole scan -- one bad file out of 82,099 must
            # not cost the other 82,098.
            chash = ""
            if hash_mode == "xxh3":
                try:
                    with z.open(info) as fh:
                        chash = hashing.content_hash(fh)
                except Exception:
                    chash = ""

            row = {
                # ---- identity ------------------------------------------------------------
                # sha1 turns any text into a fixed 40-character code. We hash the LOCATION
                # ("dataset::path/inside/zip"), not the picture's bytes -- hashing bytes would
                # mean decompressing 41 GB. Location is unique by construction: no two rows
                # can share it, because no zip can hold two files with the same path.
                "image_uid": hashlib.sha1(f"{dataset}::{info.filename}".encode()).hexdigest(),
                "archive": os.path.basename(path),
                "member": info.filename,

                # ---- content fingerprint, free from the table of contents -----------------
                # info.CRC is a checksum of the file's contents that zip already stores.
                # ":08x" formats it as 8 hexadecimal digits. Pairing it with the byte size
                # makes an accidental collision vanishingly unlikely. Two rows sharing this
                # value are the SAME PICTURE -- which is how step 2 finds duplicates without
                # reading a single pixel.
                "content_key": f"{info.CRC:08x}-{info.file_size}",

                # ---- the authoritative duplicate key -------------------------------------
                # A real 128-bit hash of the actual bytes. This is what downstream steps
                # group on. Empty in crc32 mode, and empty for a member we could not read --
                # in which case hashing.dedup_key() falls back to content_key FOR THAT ROW
                # ONLY, so one unreadable file cannot collapse into a fake duplicate group.
                "content_hash": chash,

                # Filled later by build_phash.py, which has to decode pixels.
                "phash": "",

                # ---- provenance -----------------------------------------------------------
                "dataset": dataset,
                "task_name": meta.get(task, {}).get("task_name", ""),
                "task_status": meta.get(task, {}).get("task_status", ""),
                "frame_index": frame,
                "image_name": base,
                "extension": ext.lstrip(".").lower(),   # ".JPG" -> "jpg"

                # ---- the picture ----------------------------------------------------------
                "width": w,
                "height": h,
                "megapixels": round(w * h / 1e6, 3) if w > 0 and h > 0 else 0.0,
                "size_bytes": info.file_size,

                # ---- annotations (filled in a few lines below) -----------------------------
                "has_annotation": 0,
                "n_annotations": 0,

                # ---- placeholders for later steps ------------------------------------------
                # Written now so every row has every column from the start. A column that
                # appears halfway through a build makes the file unreadable.
                "farm": "", "field": "", "season": "", "site_id": "",
                "leakage_group_id": "", "split": "", "exhaustive": "unknown",
            }

            # Add whatever the filename told us.
            p = parse_name(stem)
            row.update(
                sensor=p["sensor"], band=p["band"],
                capture_datetime=p["capture_datetime"],
                capture_year=_num(p["capture_year"]),
                flight_id=p["flight_id"], source_mosaic=p["source_mosaic"],
                tile_x=_num(p["tile_x"]), tile_y=_num(p["tile_y"]),
                tile_row=_num(p["tile_row"]), tile_col=_num(p["tile_col"]),
                geo_x=_num(p["geo_x"], float, -1.0), geo_y=_num(p["geo_y"], float, -1.0),
            )

            # Expand this photograph's marks into instance rows, and count them.
            inst = []
            for n, sh in enumerate(frame_shapes):
                inst.extend(shape_to_instances(sh, row["image_uid"], n))
            row["n_annotations"] = len(inst)
            row["has_annotation"] = 1 if inst else 0

            instances.extend(inst)
            rows.append(row)

    return rows, instances


def main() -> int:
    """Run the whole scan: every archive, then write the two tables."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--archives", default=os.path.expanduser("~/workspace/gdrive_datasets"),
                    help="folder containing the .zip files")
    ap.add_argument("--out", default="tables/manifest.parquet",
                    help="where to write the manifest (instances.parquet goes beside it)")
    ap.add_argument("--limit", type=int, default=0,
                    help="only scan the first N archives -- handy for a quick test. 0 = all")
    ap.add_argument("--hash", default="xxh3", choices=["xxh3", "crc32"],
                    help="xxh3 (default) reads every member and computes a real 128-bit "
                         "content hash, ~1 min for the whole corpus. crc32 is metadata-only "
                         "and never touches photograph bytes, but is only a 32-bit checksum.")
    args = ap.parse_args()

    zips = sorted(f for f in os.listdir(args.archives) if f.lower().endswith(".zip"))
    if args.limit:
        zips = zips[: args.limit]
    print(f"{len(zips)} archives in {args.archives}")

    rows, insts, failed = [], [], []
    # enumerate(zips, 1) numbers them starting from 1 instead of 0, for nicer progress output.
    for n, name in enumerate(zips, 1):
        try:
            got, got_inst = scan_archive(os.path.join(args.archives, name), args.hash)
            rows.extend(got)
            insts.extend(got_inst)
        except Exception as exc:
            # One corrupt archive must not throw away the work done on the other 212.
            # We record what failed and carry on; the warning is printed at the end.
            failed.append({"archive": name, "error": str(exc)[:200]})
        if n % 25 == 0 or n == len(zips):
            # flush=True forces the line out immediately instead of sitting in a buffer, so
            # you can actually watch progress on a long run.
            print(f"  {n}/{len(zips)}  {len(rows):,} photographs", flush=True)

    # ---- write the manifest --------------------------------------------------------------
    df = pd.DataFrame(rows)                                    # list of dicts -> table
    schema.validate(df, schema.MANIFEST_COLUMNS, "manifest")   # check before writing
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_parquet(args.out, index=False)   # index=False: don't save pandas' row numbers

    # ---- write the instances --------------------------------------------------------------
    # If nothing at all was annotated we still write a correctly shaped EMPTY table, so the
    # next script can read it normally rather than crashing on a missing file.
    idf = pd.DataFrame(insts) if insts else schema.empty_frame(schema.INSTANCE_COLUMNS)
    schema.validate(idf, schema.INSTANCE_COLUMNS, "instances")
    inst_out = os.path.join(os.path.dirname(args.out) or ".", "instances.parquet")
    idf.to_parquet(inst_out, index=False)

    # ---- summary -----------------------------------------------------------------------
    # .nunique() counts distinct values, so this is "rows minus distinct pictures" =
    # how many rows are a repeat of a picture we have already seen.
    key = hashing.dedup_key(df)
    dup = len(df) - key.nunique()
    unhashed = int((df["content_hash"] == "").sum()) if args.hash == "xxh3" else 0
    print(f"\nwrote {args.out}  {len(df):,} rows x {len(df.columns)} columns")
    print(f"  datasets            {df['dataset'].nunique()}")
    print(f"  with annotations    {int(df['has_annotation'].sum()):,}")
    print(f"  source_mosaic known {int((df['source_mosaic'] != '').sum()):,}")
    print(f"  hash                {hashing.HASH_NAME if args.hash == 'xxh3' else 'crc32+size'}")
    print(f"  exact duplicates    {dup:,} rows share their content with an earlier row")
    if unhashed:
        print(f"  [warn] {unhashed:,} member(s) could not be read; those rows fall back to "
              f"crc32+size")
    print(f"wrote {inst_out}  {len(idf):,} marked objects")
    if len(idf):
        top = idf["label_raw"].value_counts().head(5)
        print("  top labels          " + ", ".join(f"{k} {v:,}" for k, v in top.items()))
    if failed:
        print(f"  [warn] {len(failed)} archive(s) failed: {[f['archive'] for f in failed][:3]}")
    return 0


# This block runs only when you execute the file directly (python -m ...), not when another
# script imports it. That is what lets tests import shape_to_instances() without kicking off
# a full 213-archive scan.
if __name__ == "__main__":
    raise SystemExit(main())
