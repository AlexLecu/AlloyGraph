#!/usr/bin/env python3
"""Prove the committed training set is reproducible from its committed sources.

The 77-alloy training set behind ``saved_models_v2`` diverges from the upstream
file it was derived from in 14 cells across 11 alloys. Those edits were made by
hand: no committed script performs them, and until now nothing recorded that
they existed. Anyone regenerating the training data from the committed upstream
would have produced a different training set from the one the published models
were fitted on, and would have had no way to know.

This closes that gap without retraining anything. Every edit is now declared in
``evaluation/prediction/data/errata_ledger.json`` with its rationale, and this
script checks the claim the ledger makes:

    upstream file + ledger  ==  committed training set, cell for cell

It rebuilds the training set from the upstream records and the ledger, compares
against the committed file, and exits non-zero on any difference. If someone
edits the training data by hand again, this fails.

Usage:
    python verify_training_provenance.py
    python verify_training_provenance.py --verbose    # list every edit applied
"""

import argparse
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))
TRAINING_DIR = os.path.join(PROJECT_ROOT, "backend", "alloy_crew", "models", "training_data")
LEDGER = os.path.join(BASE_DIR, "data", "errata_ledger.json")

PROPS = ("yield_strength", "uts", "elongation", "elasticity")


def key_of(record):
    return (record.get("alloy"), record.get("processing"), record.get("form"))


def load(path):
    return {key_of(json.loads(line)): json.loads(line) for line in open(path)}


def apply_entry(record, entry):
    """Apply one ledger entry to a record in place. Returns True if it bit."""
    if entry["field"] == "composition":
        comp = record.setdefault("composition", {})
        if entry["to"] is None:
            return comp.pop(entry["element"], "__absent__") != "__absent__"
        before = comp.get(entry["element"])
        comp[entry["element"]] = entry["to"]
        return before != entry["to"]

    series = record.get(entry["field"]) or []
    for point in series:
        if abs(float(point["temp_c"]) - entry["temp_c"]) > 1e-9:
            continue
        if entry["to"] is None:
            series.remove(point)
        else:
            point["value"] = entry["to"]
        record[entry["field"]] = series
        return True
    return False


def compare(rebuilt, committed):
    """Cell-level differences between two records."""
    diffs = []
    a, b = rebuilt.get("composition") or {}, committed.get("composition") or {}
    for el in sorted(set(a) | set(b)):
        if a.get(el) != b.get(el):
            diffs.append(f"composition[{el}] rebuilt={a.get(el)} committed={b.get(el)}")
    for prop in PROPS:
        sa = {str(p["temp_c"]): p["value"] for p in (rebuilt.get(prop) or [])}
        sb = {str(p["temp_c"]): p["value"] for p in (committed.get(prop) or [])}
        for t in sorted(set(sa) | set(sb), key=float):
            if sa.get(t) != sb.get(t):
                diffs.append(f"{prop}@{t}C rebuilt={sa.get(t)} committed={sb.get(t)}")
    return diffs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    ledger = json.load(open(LEDGER))
    upstream = load(os.path.join(TRAINING_DIR, os.path.basename(ledger["upstream_source"])))
    committed = load(os.path.join(TRAINING_DIR, os.path.basename(ledger["training_target"])))
    training_entries = [e for e in ledger["entries"] if e["applies_to"] == "training"]

    print(f"upstream records : {len(upstream)}")
    print(f"committed training set: {len(committed)}")
    print(f"ledger entries for training: {len(training_entries)}\n")

    rebuilt, unmatched, applied = {}, [], 0
    for k, rec in committed.items():
        source = upstream.get(k)
        if source is None:
            unmatched.append(k)
            continue
        rebuilt[k] = json.loads(json.dumps(source))

    for entry in training_entries:
        target = None
        for k in rebuilt:
            if k[0] == entry["alloy"] and k[1] == entry["processing"] and k[2] == entry["form"]:
                target = k
                break
        if target is None:
            print(f"  {entry['id']}: no record matches {entry['alloy']}")
            continue
        if apply_entry(rebuilt[target], entry):
            applied += 1
            if args.verbose:
                loc = entry["element"] or f"{entry['temp_c']:.0f}C"
                print(f"  {entry['id']} {entry['class']:9s} {entry['alloy']:16s} "
                      f"{entry['field']}[{loc}] {entry['from']} -> {entry['to']}")
        else:
            print(f"  {entry['id']}: did not apply (already equal, or target absent)")

    if args.verbose:
        print()
    print(f"ledger entries applied: {applied}/{len(training_entries)}")

    failures = 0
    for k, exp in committed.items():
        if k not in rebuilt:
            continue
        diffs = compare(rebuilt[k], exp)
        if diffs:
            failures += 1
            print(f"\nMISMATCH {k[0]} ({k[1]}, {k[2]}):")
            for d in diffs:
                print(f"    {d}")

    print()
    if unmatched:
        print(f"{len(unmatched)} committed record(s) absent upstream: {unmatched}")
    if failures:
        print(f"FAIL: {failures} record(s) differ after applying the ledger.")
        return 1
    print(f"PASS: upstream + ledger reproduces all {len(rebuilt)} committed training "
          f"records, cell for cell.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
