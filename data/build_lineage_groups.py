#!/usr/bin/env python3
"""
DEPRECATED -- replaced by build_groups.py. This module refuses to run.

WHY IT WAS REPLACED
    It produced a single overloaded `leakage_group_id` that mixed together identities
    evidenced to very different degrees:

      * proven pixel identity (identical bytes)
      * an unaudited automatic pHash threshold
      * filename-derived mosaic and flight identifiers
      * an inferred capture event
      * and, fatally, a fallback that wrote the DATASET NAME into site_id whenever the
        review spreadsheet was blank

    That last step converted "we do not know where this is" into a confident claim, and
    combined with transitive merging it produced one component holding 58% of the corpus.

    It also selected a canonical image by filename order, which would have stranded 118,006
    annotations on non-canonical copies.

WHAT TO USE
    python -m training.data.build_groups

    which emits content_group_id, visual_group_id, lineage_group_id, source_event_proxy_id,
    field_event_id, site_id and alloc_component_id as SEPARATE columns, remaps every
    annotation onto canonical pixels without discarding any, and fails closed.

This file is kept only so that an old command line fails loudly instead of silently
producing a table that looks plausible.
"""

import sys

MESSAGE = (
    "build_lineage_groups is DEPRECATED and will not run.\n"
    "It emitted a single overloaded leakage_group_id and guessed site_id from the dataset "
    "name, which produced a 58%-of-corpus component and stranded 118,006 annotations.\n"
    "Use:  python -m training.data.build_groups\n"
)

if __name__ == "__main__":
    sys.stderr.write(MESSAGE)
    raise SystemExit(2)
