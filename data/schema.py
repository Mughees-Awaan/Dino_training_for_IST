#!/usr/bin/env python3
"""
schema.py -- THE RULEBOOK
=========================

This file contains NO logic and NO data. It only describes what our tables must look like.

WHY DOES THAT DESERVE ITS OWN FILE?
    Six different scripts build six different tables, written over about two weeks. If each
    script decides for itself what columns to write, they slowly drift apart -- one script
    writes "img_id", another expects "image_uid", and the mismatch only shows up much later
    in a script that did nothing wrong. Putting the definitions in ONE place and checking
    against them means a mistake fails immediately, in the script that caused it.

WHAT IS A "TABLE" HERE?
    Think of a spreadsheet. Rows are things, columns are facts about those things. We store
    them as Parquet files (a spreadsheet format built for large data -- see build_manifest).

WHAT IS A "dict" BELOW?
    A dict is Python's lookup table: {key: value, key: value}. We use it as
    {column_name: type_of_data}. "string" means text, "int64" a whole number, "float64" a
    decimal number.
"""

from __future__ import annotations   # lets us write modern type hints on older Pythons


# =====================================================================================
# TABLE 1: the manifest -- ONE ROW PER PHOTOGRAPH (82,099 rows)
# =====================================================================================
# This is the master list of every photograph we have. Everything else refers back to it.
MANIFEST_COLUMNS: dict[str, str] = {

    # ---- WHERE THE PHOTOGRAPH LIVES -------------------------------------------------
    # We do NOT copy photographs out of the zip archives. They stay inside, and these two
    # columns say which zip and which file inside it. (Why: the photos are already
    # compressed JPEGs, so copying them out would duplicate 41 GB of disk for no benefit,
    # and we only have 12 GB free.)
    "image_uid": "string",        # a unique name for this row, made by scrambling
                                  # "archive::member" into a fixed-length code (a "hash").
                                  # Two different photographs can never collide.
    "archive": "string",          # e.g. "Apple Gopro.zip"
    "member": "string",           # e.g. "task_0/data/GX010004_frame_6645.jpg"

    # ---- FINGERPRINTS OF THE CONTENT ------------------------------------------------
    # Two different questions, two different columns. See data/hashing.py for the full
    # explanation of why one is not enough.
    #
    #   content_hash  "are these the same FILE?"     xxh3_128 of the real bytes, 128 bits.
    #                 THIS IS THE AUTHORITATIVE DUPLICATE KEY. Computed by reading the
    #                 member out of the zip (~1 min for the whole corpus: the archives are
    #                 STORED, so there is no decompression step).
    #
    #   phash         "are these the same PICTURE?"  64-bit perceptual hash. Two photographs
    #                 that LOOK the same get similar codes even when their bytes differ
    #                 completely -- a re-saved or re-compressed copy. Compare with
    #                 hashing.hamming(); <= 6 bits apart means the same picture.
    #                 Filled by build_phash.py, which must decode pixels (~7 min).
    #
    #   content_key   the OLD key: CRC32 + byte size, both free from the zip index.
    #                 Kept as a cheap pre-filter and for backwards compatibility, but no
    #                 longer authoritative -- CRC32 is a 32-bit error-detection code, not a
    #                 hash, and with 82,099 items it has a 54% chance of a collision on its
    #                 own. (Verified correct on THIS corpus; not guaranteed on the next one.)
    "content_hash": "string",     # 32 hex chars, e.g. "55a20adac904f03ef72d4b31b9d2a7bb"
    "phash": "string",            # 16 hex chars, e.g. "7a85c456cebe0c8c"
    "content_key": "string",      # looks like "0efb69ed-67499"

    # ---- WHERE IT CAME FROM ---------------------------------------------------------
    "dataset": "string",          # the archive name without ".zip"
    "task_name": "string",        # the annotation job it belonged to
    "task_status": "string",      # "annotation" / "completed" / "validation".
                                  # CAREFUL: this describes the team's WORKFLOW, not whether
                                  # the data is good. See validate_annotations.py.
    "frame_index": "int64",       # this photograph's position within its task.
                                  # This number MUST match the order in data/manifest.jsonl,
                                  # not the order files happen to sit in the zip. Getting it
                                  # wrong attaches every annotation to the wrong photograph.
    "image_name": "string",       # just the filename
    "extension": "string",        # "jpg", "png", ...

    # ---- THE PICTURE ITSELF ---------------------------------------------------------
    "width": "int64",             # in pixels. -1 means "we could not find out"
    "height": "int64",
    "megapixels": "float64",      # width x height / 1,000,000. Just for convenience.
    "size_bytes": "int64",        # file size

    # ---- WHEN AND HOW IT WAS TAKEN --------------------------------------------------
    "sensor": "string",           # e.g. "DJI" (a drone brand)
    "band": "string",             # some drones carry two cameras; this says which one
    "capture_datetime": "string", # "2025-06-02 10:45:20", read out of the filename
    "capture_year": "int64",
    "flight_id": "string",        # photographs from one drone flight share this

    # ---- POSITION INSIDE A BIGGER MAP -----------------------------------------------
    # A drone survey produces one huge map, which is then chopped into small square tiles.
    # These columns say which big map a tile came from and where in it. This is the SECOND
    # strongest clue for deciding which photographs belong together (see
    # build_lineage_groups.py) -- 61,723 of our 82,099 rows have it.
    "source_mosaic": "string",    # name of the big map
    "tile_x": "int64",            # position of this tile inside that map, in pixels
    "tile_y": "int64",
    "tile_row": "int64",          # some datasets number tiles by row/column instead
    "tile_col": "int64",
    "geo_x": "float64",           # real-world coordinates, where available
    "geo_y": "float64",

    # ---- WHAT IS MARKED IN IT -------------------------------------------------------
    "has_annotation": "int64",    # 1 if anything is marked, 0 if not
    "n_annotations": "int64",     # how many things are marked.
                                  # The marks themselves live in the instances table below.

    # ---- FILLED IN BY LATER STEPS ---------------------------------------------------
    # These start empty. Each later script fills one in.
    # ---- THE THREE SEPARATE IDENTITIES ---------------------------------------------
    # One overloaded "family" column conflated three different questions and produced a
    # single component holding 58% of the corpus. They are now kept apart, because they are
    # evidenced to different degrees and enforce different rules.
    #
    #   content_group_id  "these are literally the same picture"   EVIDENCED (bytes/pHash)
    #   lineage_group_id  "these are derived from the same source" EVIDENCED (mosaic/flight)
    #   field_event_id    "these are one field on one visit"       REVIEWED provenance only
    #   site_id           "this is one physical place"             REVIEWED provenance only
    #
    # Enforcement: one canonical image per content group (aliases are never sampled, so they
    # cannot leak); every lineage group in exactly one split; every field event in exactly
    # one split; a sealed site appears nowhere else.

    "farm": "string",             # step 3, from the spreadsheet a person fills
    "field": "string",            # step 3
    "season": "string",           # step 3
    "site_id": "string",          # step 3 -- a stable name for a physical place.
                                  # STAYS EMPTY when unknown. There is deliberately no
                                  # dataset-name fallback: unknown provenance must remain
                                  # unknown, and a guessed site cannot enter calibration,
                                  # confirmation or the sealed set.
    "split": "string",            # step 6 -- practice set or exam set
    "exhaustive": "string",       # "yes"/"no"/"unknown" -- is EVERY object in this photo
                                  # marked? Only "yes" photos can be used for scoring.
}


# =====================================================================================
# TABLE 2: instances -- ONE ROW PER MARKED OBJECT (879,253 rows)
# =====================================================================================
# One photograph can contain hundreds of marked plants, so this table is much longer than
# the manifest. Each row points back to its photograph via image_uid.
INSTANCE_COLUMNS: dict[str, str] = {
    "image_uid": "string",        # which photograph this mark is in (matches the manifest)
    "instance_id": "string",      # a unique name for this individual mark
    "label_raw": "string",        # the name exactly as the annotator typed it, e.g. "garlic_3"
    "label_canon": "string",      # the tidied name after a person approves it, e.g. "garlic".
                                  # Empty until normalize_labels.py runs.
    "shape_type": "string",       # how it was drawn: "points", "rectangle", "polygon", "polyline"

    # Position, in pixels, measured from the top-left corner of the photograph.
    "x": "float64",               # centre of the object, left-to-right
    "y": "float64",               # centre of the object, top-to-bottom
    "w": "float64",               # width of the box around it. 0 if it was just a dot.
    "h": "float64",               # height
    "area_px": "float64",         # how many pixels it covers
    "points_wkt": "string",       # for polygons: the full outline, saved as text so nothing
                                  # is lost. Empty for simple dots and boxes.
    "is_crowd": "int64",          # 1 would mean "a blob of many objects, not one" (unused yet)
}


# =====================================================================================
# TABLE 2b: groups -- ONE ROW PER PHOTOGRAPH, ALL GROUPING IDENTITIES
# =====================================================================================
# Grouping used to be a single overloaded "family" column on the manifest. That conflated
# identities evidenced to very different degrees and produced one component holding 58% of
# the corpus. They are now separate columns in their own table, and the manifest carries none
# of them -- so there is exactly one place that answers "what is grouped with what".
#
# EVIDENCE STRENGTH, strongest first. This ordering is the point of the table:
#
#   content_group_id        identical BYTES.                        PROVEN
#   visual_group_id         pHash within threshold, NOT audited.    AUTOMATIC, unverified
#   lineage_group_id        same mosaic / flight / tile overlap.    EVIDENCED (filename)
#   source_event_proxy_id   inferred "one capture event".           PROXY, filename-derived
#   field_event_id          one field on one visit.                 REVIEWED PROVENANCE ONLY
#   site_id                 one physical place.                     REVIEWED PROVENANCE ONLY
#
# Only the last two may back a production field-held-out claim. Everything above them
# supports a SOURCE-GROUP benchmark and must be labelled as such.
GROUP_COLUMNS: dict[str, str] = {
    "image_uid": "string",

    # ---- pixel identity -------------------------------------------------------------
    # Exact bytes only. pHash is deliberately NOT folded in here: an automatic threshold is
    # not proof, and discarding an unaudited match as a duplicate would throw away real data.
    "content_group_id": "string",
    "is_canonical": "int64",        # 1 = the representative whose pixels are sampled
    "canonical_uid": "string",      # which image that is (self, when is_canonical = 1)

    # ---- unaudited visual similarity -- a SPLIT CONSTRAINT, never a discard -----------
    "visual_group_id": "string",

    # ---- derivation -----------------------------------------------------------------
    "lineage_group_id": "string",

    # ---- capture event: inferred vs reviewed. NEVER conflate these two. ---------------
    "source_event_proxy_id": "string",   # from filename mosaics/flights. A PROXY.
    "field_event_id": "string",          # reviewed provenance ONLY. Empty until then.
    "site_id": "string",                 # reviewed provenance ONLY. Empty until then.

    # ---- what splitting actually allocates -------------------------------------------
    # The connected component of every constraint above. One split per component.
    "alloc_component_id": "string",

    "quarantined": "int64",
    "quarantine_reason": "string",
}


# =====================================================================================
# TABLE 2c: observations -- ANNOTATIONS REMAPPED ONTO CANONICAL PIXELS
# =====================================================================================
# Canonicalisation is about PIXELS, not about annotations. Two byte-identical files may
# carry completely different annotation work: measured on this corpus, 2,102 content groups
# have annotations ONLY on a non-canonical copy (118,006 instances), and 343,254 annotation
# rows sit on aliases overall. Selecting a canonical image and dropping its aliases would
# silently discard all of that.
#
# So annotations are remapped onto the canonical image rather than discarded, and the
# original context is preserved so a disagreement can be adjudicated rather than guessed.
OBSERVATION_COLUMNS: dict[str, str] = {
    "instance_id": "string",
    "canonical_uid": "string",      # the pixels this observation now refers to
    "source_image_uid": "string",   # the file it was actually drawn on
    "content_group_id": "string",
    "task_name": "string",          # WHICH annotation job -- needed for task-scoped ontology
    "label_raw": "string",
    "conflict": "int64",            # 1 = this content group's copies disagree on labels
}


# =====================================================================================
# TABLE 3: coverage -- IS THIS PHOTOGRAPH FULLY MARKED?
# =====================================================================================
# This is the difference between "someone worked on this" and "everything in it is marked".
# It matters enormously for scoring -- see validate_annotations.py and render_episodes.py.
COVERAGE_COLUMNS: dict[str, str] = {
    "image_uid": "string",
    "reviewed": "int64",          # 1 = a person looked at it
    "exhaustive": "int64",        # 1 = EVERY object is marked. Required for scoring.
    "reviewer": "string",         # who said so
    "reviewed_at": "string",      # when
    "source": "string",           # how we know: "task_status" / "spreadsheet" / "manual"
}


# =====================================================================================
# TABLE 4: lineage -- WHICH PHOTOGRAPHS BELONG TOGETHER
# =====================================================================================
# Photographs of the same field must never be split between the practice set and the exam
# set, or the model gets tested on pictures it has effectively already seen.
LINEAGE_COLUMNS: dict[str, str] = {
    "image_uid": "string",
    "leakage_group_id": "string", # the "family" name, e.g. "g000123"
    "evidence": "string",         # WHY it was put in that family: "duplicate", "mosaic", ...
    "evidence_rank": "int64",     # 1 = strongest reason, 6 = weakest. Recorded so that a
                                  # suspicious family can be traced back to the rule that
                                  # created it.
}


# =====================================================================================
# TABLE 5: the label vocabulary -- THE APPROVED LIST OF NAMES
# =====================================================================================
# Annotators typed 155 different label strings. Many are the same plant spelled differently.
# A PERSON decides which are the same; this table records their decisions.
ONTOLOGY_COLUMNS: dict[str, str] = {
    "label_raw": "string",        # what was typed
    "label_canon": "string",      # what we will call it from now on
    "domain": "string",           # optional grouping, e.g. "cereal", "fruit"
    "decision": "string",         # "keep" / "merge" / "drop"
    "approved_by": "string",      # WHO decided. Required -- the script refuses without it.
    "approved_at": "string",      # when
}


# =====================================================================================
# CONSTANTS -- values several scripts need to agree on
# =====================================================================================

# The five piles we divide the data into.
SPLITS = ("train", "dev_cal", "dev_select", "dev_confirm", "sealed_test")

# What fraction of the photographs goes into each pile.
#   train        the model learns from these
#   dev_cal      used to tune the sensitivity dial
#   dev_select   used to choose between different trained versions
#   dev_confirm  used ONCE to confirm a choice already made
#   sealed_test  opened ONCE, at the very end. Never touched before that.
SPLIT_SHARES = {"train": 0.60, "dev_cal": 0.10, "dev_select": 0.10,
                "dev_confirm": 0.10, "sealed_test": 0.10}

# The three values CVAT (the annotation tool) uses for a job's progress.
TASK_STATUSES = ("annotation", "completed", "validation")

EXHAUSTIVE_VALUES = ("yes", "no", "unknown")

# The reasons we might decide two photographs belong to the same family, STRONGEST FIRST.
# build_lineage_groups.py works down this list. The number is stored with each row so we can
# later ask "which rule put these together?"
LINEAGE_EVIDENCE = (
    ("duplicate", 1),      # byte-for-byte the same picture. Certain.
    ("mosaic", 2),         # cut from the same big survey map. Very strong.
    ("flight", 3),         # same drone flight. Strong.
    ("spacetime", 4),      # same moment and same place. Strong.
    ("perceptual", 5),     # looks nearly the same. Moderate.
    ("site", 6),           # the spreadsheet says it is the same place. LAST RESORT.
)


def validate(df, columns: dict[str, str], name: str, allow_extra: bool = True):
    """Check a table has the columns it is supposed to have. Raise an error if not.

    WHAT IS `df`?
        A pandas DataFrame -- Python's version of a spreadsheet held in memory.

    WHY CALL THIS TWICE?
        Every script calls it before writing a table AND after reading one. That way a
        missing column is reported by the script that caused the problem, instead of
        surfacing three steps later somewhere innocent.

    `allow_extra=True` means extra columns are fine -- scripts add columns as they go.
    """
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: missing columns {missing}")
    if not allow_extra:
        extra = [c for c in df.columns if c not in columns]
        if extra:
            raise ValueError(f"{name}: unexpected columns {extra}")
    return df


def empty_frame(columns: dict[str, str]):
    """Build an empty table with the right columns and types.

    Used when a step legitimately finds nothing -- we still want a correctly shaped, empty
    table rather than a crash or a file that is missing entirely.
    """
    import pandas as pd
    return pd.DataFrame({c: pd.Series(dtype=t) for c, t in columns.items()})
