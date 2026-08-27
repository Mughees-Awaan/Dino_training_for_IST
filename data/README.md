# Preprocessing — how it works, in order

Turns 213 archives of photographs into training data that will not flatter the model.

Fourteen files. Eight are run in sequence; six are libraries the others import. This document
explains every one: what it reads, what it writes, why it exists, and how it can fail.

---

## Contents

- [The three failures this prevents](#the-three-failures-this-prevents)
- [Run order at a glance](#run-order-at-a-glance)
- [Step 0 — the spreadsheet](#step-0--the-spreadsheet-a-person)
- [Step 1 — build_manifest](#step-1--build_manifestpy)
- [Step 2 — audit_duplicates](#step-2--audit_duplicatespy)
- [Step 3 — build_lineage_groups](#step-3--build_lineage_groupspy)
- [Step 4 — normalize_labels](#step-4--normalize_labelspy)
- [Step 5 — validate_annotations](#step-5--validate_annotationspy)
- [Step 6 — build_splits](#step-6--build_splitspy)
- [Step 7 — build_episodes](#step-7--build_episodespy)
- [Step 8 — render_episodes](#step-8--render_episodespy)
- [Libraries](#libraries-not-run-directly)
- [What ends up on disk](#what-ends-up-on-disk)
- [Where a person is required](#where-a-person-is-required)

---

## The three failures this prevents

All three are silent. The pipeline runs, the model trains, and the reported score is wrong.

| Failure | What you would see | What actually happened |
|---|---|---|
| Related photographs split across practice and exam | A great score | The model was tested on pictures it had already seen |
| Clicks and targets in the same photograph | A great score | The model found targets by looking next to the clicks |
| One plant recorded under five names | A poor score, no explanation | The model was marked wrong for correct answers |

Every design decision below traces back to one of these.

---

## Run order at a glance

```
  STEP 0   a person fills the spreadsheet                       ~1 hour
  STEP 1   python -m training.data.build_manifest              ~2 min
  STEP 2   python -m training.data.audit_duplicates            ~3 min
              -> a person reads the report
  STEP 3   python -m training.data.build_lineage_groups        ~1 min
  STEP 4   python -m training.data.normalize_labels --propose  seconds
              -> a person fills label_review.csv               ~1 hour
           python -m training.data.normalize_labels --approved
  STEP 5   python -m training.data.validate_annotations        ~1 min
              STOPS EVERYTHING if anything is malformed
  STEP 6   python -m training.data.build_splits                seconds
  STEP 7   python -m training.data.build_episodes              ~1 min
  STEP 8   python -m training.data.render_episodes             ~1 min
              -> a person looks at 100 pictures
```

Each step reads what the previous one wrote. Running them out of order fails loudly.

---

## Step 0 — the spreadsheet (a person)

Not a script. Someone records which farm and field each of the 212 datasets came from.

**Why:** photographs of the same field must stay together when the data is split. Automatic
evidence catches most of this — 84% of photographs are grouped by duplicate detection or by
their parent map — but it cannot see that `comberton_2` and `comberton_9` are one farm
photographed twice. **23 dataset pairs have names suggesting the same place with no evidence
linking them.** Each is a route for the exam to leak into the practice set.

The 48 datasets not already linked are where the value is, not all 212.

---

## Step 1 — `build_manifest.py`

**Reads** 213 zip archives · **Writes** `manifest.parquet`, `instances.parquet`

Produces one row per photograph (82,099) and one row per marked object (879,253).

### How it works

It never decompresses a photograph. It reads only the zip's table of contents and three small
JSON files inside each task. That is why 41 GB is catalogued in about two minutes.

| Function | Job |
|---|---|
| `parse_name` | Pulls date, flight and tile position out of the filename. `DJI_20250602104520_0963` yields a timestamp and a parent map |
| `read_task_json` | The task's name and workflow status |
| `read_frame_sizes` | Width, height **and frame order** |
| `read_shapes` | All geometry from `annotations.json` |
| `shape_to_instances` | One CVAT shape becomes one or more object rows |
| `scan_archive` | One archive becomes its rows |

### Two things that matter

**Frame order comes from `manifest.jsonl`, never the zip directory.** CVAT numbers frames by
their position in that file. The zip's directory is in a different order entirely — measured on
one task, **46 of 48 positions differed**. An earlier version used directory order, so every
annotation attached to the wrong photograph. Nothing errored; the clicks simply landed on bare
soil. It was found by looking at pictures, and is now guarded by `tests/test_frame_order.py`.

**A `points` shape can carry hundreds of marks, and each becomes its own row.** One mark is one
plant. That is why 82,099 photographs yield 879,253 objects.

### Two identifiers

| Column | Value | Purpose |
|---|---|---|
| `image_uid` | `sha1(archive::member)` | Unique per row. A hash of the location, not the bytes, so it costs nothing |
| `content_key` | `"<crc32>-<size>"` | Both sit in the zip index. Exact-duplicate detection with **zero pixel reads** |

Photographs are never extracted. They stay in the archives and are read on demand — they are
already-compressed JPEGs (measured 1.00x ratio), so a copy would duplicate 41 GB against 12 GB
of free space and gain nothing.

---

## Step 1b — `build_phash.py`

**Answers a question step 1 cannot.** `content_hash` says "are these the same *file*?".
This says "are these the same *picture*?"

Re-save one photograph at a different JPEG quality and every byte changes — the content hash
calls them unrelated, a human calls them identical. For leakage purposes they *are* the same
photograph and must not be split across the divide.

```
python -m training.data.build_phash --workers 8 --write-manifest
```

- ~1 minute for all 82,099 photographs at 8 threads (1,400 img/s).
- **Resumable** — results are keyed by `image_uid`, so an interrupted run only redoes what it
  had not finished. That matters on a machine that loses power.
- It is a separate step because it is the only one that must *decode* pixels rather than just
  read bytes, so it has a different cost profile.

Measured: unrelated pairs differ by 31.5 of 64 bits on average, and the closest unrelated pair
in a 4,000-pair sample was 14 apart. A threshold of ≤6 bits is safe with room to spare.

---

## Step 2 — `audit_duplicates.py`

**Reads** `manifest.parquet` · **Writes** `duplicate_report.json`

Finds photographs that exist more than once. Three kinds:

| Kind | Method | Cost |
|---|---|---|
| exact | identical `content_key` | free, from the index |
| overlapping | tiles from one parent map whose windows intersect | cheap |
| near | same dataset and size, byte length within 2% | **candidates only** |

### What it found

**52,602 photographs (64.1%) have an exact copy elsewhere**, in 16,456 groups. **14,653 of
those groups (89%) span more than one dataset.** Largest group: 40 copies.

Verified by opening five random cross-dataset groups and hashing the real bytes — 5/5 identical.
The filenames explain it: `Best Tassel Model's dataset (Copy by Zeeshan).zip`,
`..._backup_with_new_data.zip` and `Combined_detector_dataset.zip` all hold the same
`DJI_0141_4.jpg`.

### Nothing is deleted

A deleted duplicate destroys the evidence that it existed. Grouping keeps it, so the next step
can put the whole family on one side of the split.

Near-duplicates are reported, never acted on — confirming them needs pixel reads, which this
stage deliberately avoids. **A person reviews the report and spot-checks ten pairs.**

---

## Step 3 — `build_lineage_groups.py`

**Reads** `manifest.parquet`, `duplicate_report.json`, the spreadsheet · **Writes** `lineage_groups.parquet`

The most consequential script here. Everything downstream inherits its correctness.

Joins photographs into families using evidence, strongest first:

| Rank | Evidence | Strength | Share of corpus |
|---|---|---|---|
| 1 | identical bytes | certain | 64.1% |
| 2 | same parent map | very strong | 20.3% |
| 3 | same flight | strong | ~0% |
| 4 | same timestamp and place | strong | — |
| 5 | near-identical appearance | moderate | — |
| 6 | same site from the spreadsheet | **fallback** | 15.7% |

Result: **9,242 families**, median size 2, largest 1,876 photographs.

### The fallback is a fallback, not another merge

Rank 6 applies **only to photographs still alone after ranks 1–5**.

This was learned the hard way. Applying it to everything chains
`duplicate → dataset → duplicate → dataset` transitively, because 89% of duplicate groups span
datasets. Measured result: **one family holding 76% of the corpus**. Restricting it to
unattached photographs gives 9,242 families with the largest at 2.3%.

The script warns if any family exceeds half the corpus, or if fewer than 30 families exist.

---

## Step 4 — `normalize_labels.py`

**Reads** `instances.parquet` · **Writes** `label_review.csv`, then `label_ontology.parquet`

The only script that refuses to finish on its own.

### `--propose`

Groups the **155 raw label strings** by a spelling-insensitive key, giving **139 groups**, and
writes a CSV with three columns left **blank**: `label_canon`, `decision`, `approved_by`.

Genuine merges it finds:

```
garlic       <- garlic, garlic_1 ... garlic_10
tree         <- tree, trees, Tree
black_grass  <- black_grass, black grass
female       <- female, Female
```

### `--approved`

Reads the filled file back. **Refuses to run** if any row is unfilled, or if `approved_by` is
empty. It will not guess.

### Two guards, both from real mistakes

**`MIN_STEM = 4`** — dropping trailing digits is right for `garlic_1` and wrong for `z11`.
Stems under four characters keep their digits, so `z11/z12/z13`, `V2/V5` and `CNG1/CNG2/CNG3`
stay separate and are **flagged as probable plot codes** instead of merged into one "plant".

**`MODIFIERS`** — `Ripe`, `Dead`, `cut`, `healthy`, `large`, `small`, `Damaged`, `Green` are
states, not species. Flagged, never merged.

### Why manual

A wrong merge is invisible afterwards. The model learns a blurred category and scores worse,
with nothing pointing at the cause. A missed merge is visible: two labels that behave
identically. The asymmetry is the whole argument for a person.

---

## Step 5 — `validate_annotations.py`

**Reads** everything so far · **Writes** nothing · **Stops the pipeline on any failure**

| Check | Fatal |
|---|---|
| photograph identifiers unique | yes |
| every object belongs to a known photograph | yes |
| object centre inside its photograph | yes |
| box has positive size, no larger than the image | yes |
| every label is in the approved list | yes |
| every photograph has a family | yes |
| no family holds more than half the corpus | yes |
| at least 30 families | yes |
| photograph dimensions known | warning |

Failing closed is the point. A warning printed here and ignored becomes a wrong number six
weeks later with nothing pointing back at the cause.

It also reports how many photographs are marked `completed` — currently **3,038 of 82,099
(3.7%)** — because that decides whether the scoring set is large enough to gate on.

---

## Step 6 — `build_splits.py`

**Reads** `manifest.parquet`, `lineage_groups.parquet` · **Writes** `splits.yaml`

Divides everything five ways, **by family, never by photograph**:

| Set | Share | Photographs | Purpose |
|---|---|---|---|
| `train` | 60% | 49,259 | learning |
| `dev_cal` | 10% | 8,210 | setting the operating point |
| `dev_select` | 10% | 8,210 | choosing between checkpoints |
| `dev_confirm` | 10% | 8,210 | confirming a decision already made |
| `sealed_test` | 10% | 8,210 | opened once, at the very end |

Families are placed largest-first into whichever set is furthest below its target. That keeps
shares tight despite family sizes running from 1 to 1,876. Achieved shares are exact to three
decimals, and **zero families appear in two sets**.

The sealed set is excluded from learning, statistics, caching, threshold selection and label
cleanup.

---

## Step 7 — `build_episodes.py`

**Reads** `manifest.parquet`, `instances.parquet` · **Writes** `episodes/{split}.parquet`

Builds the practice questions — **36,000** of them across the five sets.

One question is: *here are 10 clicked examples in photograph A; find 20 more in photograph B.*

```
episode_id, split, label
support_uid, support_x[], support_y[]     the clicks
query_uid,   query_x[],   query_y[]       scored, never clicked
same_family, n_support, n_query, seed
```

### Two rules, enforced then verified

**Support and query come from different photographs.** The model reads a whole image at once,
so a target in the same photograph as its clicks can be found by proximity rather than
recognition. Both counts are checked after generation and must be zero.

**Both photographs come from the same set.** A question straddling practice and exam is leakage
by construction.

### Recorded, not forbidden

`same_family` marks whether the two photographs are related. Same-family questions are
realistic — in the product a user clicks and searches within one field — but easier, so
evaluation reports them separately rather than excluding them.

### Duplicates collapsed first

Byte-identical copies are reduced to one representative before sampling. Splitting by family
already stops duplicates leaking; this stops them being **trained on repeatedly**. The corpus
holds 82,099 photographs but only **45,953 distinct images**, and the largest group is 40
copies — without this, that photograph is seen 40 times as often as a unique one, for no extra
information.

---

## Step 8 — `render_episodes.py`

**Writes** 100 side-by-side images, or an exhaustiveness audit

### Default mode

Clicks circled amber on the left, targets green on the right. **A person looks at all 100.**

This is the cheapest defect detection in the programme, and it has already earned its place:
it caught the frame-ordering bug. The circles were sitting on bare soil. Every table looked
perfectly healthy.

### `--audit-exhaustive`

Samples annotated photographs across datasets, circles **every** marked object, and writes
`exhaustiveness_review.csv` for a reviewer to fill in.

It answers the one question no metadata can: **are all the plants marked, or only some?**

This matters asymmetrically:

| Use | Partial marking |
|---|---|
| Training | survivable — the model learns from what is there |
| Scoring | **fatal** — a photograph with 10 plants and 6 marked charges a model that finds all 10 with 4 false positives *for being right* |

Task status cannot answer it. Measured on this corpus, photographs marked `annotation` carry
**more** objects than those marked `completed` (median 5 against 4), so the field records
workflow state, not data quality. Datasets that fail the audit remain usable for training and
are barred from scoring.

---

## Libraries (not run directly)

### `hashing.py` — how we decide two photographs are the same

Two questions, two tools:

| question | function | notes |
|---|---|---|
| same **file**? | `content_hash()` | xxh3_128 over the real bytes, 128 bits |
| same **picture**? | `phash64()` | 64-bit DCT perceptual hash |
| how close? | `hamming()` | counts differing bits; 0 = identical, 14+ = unrelated |
| which key to group on? | `dedup_key()` | prefers the real hash, falls back per-row |

**Why not CRC32 (what this replaced).** CRC32 is an *error-detection code*, not a hash — 32
bits, with a 54% chance of at least one collision across 82,099 items on its own. Pairing it
with the byte size hid most of that, and rebuilding with xxh3 proved it had in fact been
**exactly right on this corpus** (identical group count *and* identical memberships). The
change removes an unbounded risk on future data, and it costs about a minute — the archives
are STORED, so reading a member is plain I/O at 599 MB/s with no decompression.

**Why not SHA-256.** We are asking "are these bytes the same?", not "could someone have forged
these bytes?". xxh3 is several GB/s against SHA-256's ~500 MB/s, and 128 bits gives an
accidental-collision chance of about 1 in 10²⁸ here.



| File | Purpose |
|---|---|
| `schema.py` | The rulebook: required columns for all six tables, the split shares, the evidence ranking. `validate()` is called by producers before writing and consumers after reading, so a missing column fails in the script that caused it |
| `dataset.py` | Feeds sessions to the trainer, reading photographs straight from the archives. Carries click coordinates through augmentation so pixels and marks move together |
| `transforms.py` | The deliberate variation. **Gentle on colour, aggressive on geometry** — aerial photographs have no natural orientation, so flips and rotations are free, while hue often carries the class. Standard recipes jitter colour hard and add greyscale; that teaches colour invariance, which is wrong when the target is a yellow-green plant in green grass |
| `runtime/lattice.py` | Which pixel belongs to which output square. Refuses a halo that is not a whole number of squares — the current defect, where 144 % 32 = 16 shifts every interior tile by 16 px |
| `runtime/tiler.py` | Cuts large photographs into tiles with a halo and reassembles them. **Asserts every cell was written exactly once** — this caught a double-write of 416 cells during development |
| `runtime/preprocessing.py` | One single way to prepare an image. Two copies that disagree by a normalisation constant give two different answers for the same field, and both look plausible |

The three `runtime/` files are shared with the product. Today the application and the
command-line tool keep separate copies of this logic and report different results for the same
field. One copy removes that permanently.

---

## What ends up on disk

```
tables/
  manifest.parquet          7 MB     82,099 photographs
  instances.parquet       144 MB    879,253 marked objects
  lineage_groups.parquet    3 MB      9,242 families
  duplicate_report.json     4 MB     52,602 duplicates found
  label_review.csv          6 KB        155 names awaiting a decision
  label_ontology.parquet             the approved vocabulary
  splits.yaml             130 KB    five sets, no overlap
  episodes/*.parquet                 36,000 practice questions
review/
  episodes/                          100 images for inspection
  exhaustive/                        audit images + review sheet
```

None of this is committed. It is rebuilt from the archives by running the eight steps.

---

## Where a person is required

| Step | Task | Time |
|---|---|---|
| 0 | Fill the spreadsheet | ~1 hour for the 48 unlinked datasets |
| 2 | Read the duplicate report, spot-check ten pairs | 15 min |
| 4 | Decide 155 label names | ~1 hour |
| 8 | Look at 100 rendered sessions | 20 min |
| 8 | Count missed plants in 30 audit images | 1 hour |

None can be automated. Step 8 has already proved its value by catching a defect that every
automated check passed.

---

## Current state

- Steps 1, 1b, 2, 3, 5, 6, 7 have all run against the real corpus.
- 82,099 photographs · 879,253 objects · 9,201 families (largest 2.22%)
- Splits exactly 60/10/10/10/10, zero families in two splits.
- 36,000 episodes; both leakage checks clean.
- `validate_annotations --fail-on-error` passes every structural check.
- **Step 4 is the blocker, and it is human:** 155 of 155 rows in `tables/label_review.csv`
  await a decision.
