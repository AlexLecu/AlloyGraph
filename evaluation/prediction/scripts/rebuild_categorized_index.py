#!/usr/bin/env python3
"""Rebuild data/all_categorized.jsonl and data/manifest.json from the category files.

There were two data paths into the evaluation set and they disagreed. The four
category files -- SSS, precip, sc_ds, other -- are what every committed result
was computed from. Beside them sat ``all_categorized.jsonl`` and
``manifest.json``, snapshots of an earlier run that nothing regenerated:

    all_categorized.jsonl   106 records, 8 of which no category file holds,
                            and missing RGT* 13, which precip.jsonl does hold
    manifest.json           counts 111 records against a file holding 106

``generate_predictions.py --dataset all`` reads that stale snapshot, so the
second path was live, not merely lying around. This script makes the snapshot a
derived view of the category files instead of an independent copy: it is
concatenation, nothing else, and ``--check`` fails if the two ever diverge
again.

The category files stay authoritative. Nothing here regenerates them, and no
shipped metric changes: the four evaluation files are untouched.

Usage:
    python rebuild_categorized_index.py            # rewrite both derived files
    python rebuild_categorized_index.py --check    # verify, exit 1 on drift
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(BASE_DIR, "data")

#: Category file -> the manifest entry it backs. Order fixes the record order in
#: the combined file, so a rebuild is byte-reproducible.
CATEGORIES = (
    ("precip", "precip_poly", "Precipitation-hardened Polycrystalline",
     "Ni > 40%, Al+Ti >= 2%, not SC/DS"),
    ("sc_ds", "precip_sc_ds", "Precipitation-hardened SC/DS",
     "Ni > 40%, Al+Ti >= 2%, SC/DS processing"),
    ("SSS", "solid_solution", "Solid-Solution Strengthened",
     "Ni > 40%, Al+Ti < 2%"),
    ("other", "other", "Other Nickel Alloys",
     "Ni < 40% or Cu > 20% or Ni > 95%"),
)

#: The three that make up the 88-alloy evaluation set. `other` is carried in the
#: index but is not evaluated.
EVALUATED = ("precip", "sc_ds", "SSS")


def load(stem):
    """Records of one category file. Missing is tolerated only for `other`.

    `*.jsonl` is globally git-ignored and the three evaluated category files are
    force-added exceptions; `other.jsonl` is not tracked. On a fresh clone it is
    therefore absent, and the index has to build without it rather than fail --
    `other` is carried for completeness and is evaluated by nothing.
    """
    path = os.path.join(DATA_DIR, f"{stem}.jsonl")
    if not os.path.exists(path):
        if stem in EVALUATED:
            raise SystemExit(f"missing authoritative category file: {path}")
        print(f"  {stem+'.jsonl':22s}    -  absent (untracked, not evaluated)")
        return []
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def build():
    records, counts = [], {}
    for stem, cat, _, _ in CATEGORIES:
        rows = load(stem)
        counts[cat] = len(rows)
        records.extend(rows)
    return records, counts


def manifest_for(records, counts):
    return {
        "description": "Categorized evaluation datasets for AlloyGraph",
        "generated_by": "evaluation/prediction/scripts/rebuild_categorized_index.py",
        "note": ("Derived view. The four category files are authoritative; this "
                 "index is their concatenation and must never be edited directly."),
        "categories": {
            cat: {"name": name, "criteria": crit, "count": counts[cat],
                  "file": f"{stem}.jsonl"}
            for stem, cat, name, crit in CATEGORIES
        },
        "evaluated_alloys": sum(counts[cat] for stem, cat, _, _ in CATEGORIES
                                if stem in EVALUATED),
        "total_records": len(records),
    }


def serialise(records):
    return "".join(json.dumps(r) + "\n" for r in records)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="verify the derived files match, exit 1 if not")
    args = ap.parse_args()

    records, counts = build()
    manifest = manifest_for(records, counts)
    combined_path = os.path.join(DATA_DIR, "all_categorized.jsonl")
    manifest_path = os.path.join(DATA_DIR, "manifest.json")
    text = serialise(records)

    for stem, cat, _, _ in CATEGORIES:
        mark = "evaluated" if stem in EVALUATED else "not evaluated"
        print(f"  {stem+'.jsonl':22s} {counts[cat]:4d}  ({mark})")
    print(f"  {'combined':22s} {len(records):4d}")
    print(f"  {'of them evaluated':22s} {manifest['evaluated_alloys']:4d}")

    if args.check:
        drift = []
        if not os.path.exists(combined_path) or open(combined_path, encoding="utf-8").read() != text:
            drift.append("all_categorized.jsonl")
        if not os.path.exists(manifest_path):
            drift.append("manifest.json")
        else:
            have = json.load(open(manifest_path, encoding="utf-8"))
            if {k: v for k, v in have.items() if k != "generated"} != manifest:
                drift.append("manifest.json")
        if drift:
            print(f"\nFAIL: {', '.join(drift)} does not match the category files. "
                  f"Run without --check to rebuild.")
            return 1
        print("\nPASS: derived files match the category files.")
        return 0

    with open(combined_path, "w", encoding="utf-8") as fh:
        fh.write(text)
    manifest_out = dict(manifest)
    manifest_out["generated"] = datetime.now(timezone.utc).isoformat()
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest_out, fh, indent=2)
        fh.write("\n")
    print(f"\nRewrote {os.path.basename(combined_path)} and "
          f"{os.path.basename(manifest_path)} from the category files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
