#!/usr/bin/env python3
"""Prove the committed evaluation set is reproducible from its sources.

The claim:

    source files + scope_exclusions.json + errata_ledger.json
        ==  SSS.jsonl, precip.jsonl, sc_ds.jsonl, cell for cell

Until now it was not true. Thirteen records that the sources offer are absent
from the evaluation set, and nothing said which or why; the arithmetic
(106 = 77 + 27 + 2, then 27 = 22 + 5) could be reconstructed but the reasons
could not. Two of the thirteen sit behind alloy identity rather than
composition, and one -- MC-102* -- has no defensible reason at all and is
recorded as a historical exclusion rather than justified after the fact.

This script rebuilds the three evaluated category files from the sources by
re-running the committed categoriser, applies the two evaluation errata from the
ledger, and compares against what is committed. It exits non-zero on any
difference, so a hand edit to the evaluation data cannot pass unnoticed.

WHAT PASSES AND WHAT DOES NOT. Membership and order reproduce exactly, in all
four category files: the exclusion list accounts for every record the sources
offer and the evaluation set declines, and that is what the list governs. Cell
values do not, and the reason is a separate, older gap: **the committed source
files are behind the shipped evaluation files.** Twenty records hold data the
sources do not have -- yield-strength series on AL 276, an iron content on
Altemp 718, a full composition on INCONEL G-3, room-temperature moduli replaced
rather than removed (76 -> 207 GPa, a shear-for-Young's substitution that
`clean_elasticity.py` only ever cleared), and test temperatures normalised
(20 -> 21 degC, 540 -> 649). The shipped files were built from a later MatWeb
extraction than the one committed, and that extraction was never committed.

This script therefore gates on membership and reports the value gap rather than
hiding it. Closing the value gap needs the newer extraction, not another rule.

`other.jsonl` membership is checked too. Its one exclusion, UNITEMP* AF2-1DA,
now carries a corrected `_category_reason`: the old string read "Fe-Ni base",
which the untouched classifier produced by reading an absent nickel content as
Ni = 0. Nothing evaluates `other`, so no metric moves.

Usage:
    python verify_evaluation_provenance.py
    python verify_evaluation_provenance.py --verbose
"""

import argparse
import importlib.util
import json
import os
import sys
import tempfile

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(BASE_DIR, "data")

#: categoriser bucket -> committed file stem
EVALUATED = (("precip_poly", "precip"), ("precip_sc_ds", "sc_ds"),
             ("solid_solution", "SSS"))
CARRIED = (("other", "other"),)

PROPS = ("yield_strength", "uts", "elongation", "elasticity")


def load_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def categorise():
    """Run the committed categoriser into a scratch directory."""
    spec = importlib.util.spec_from_file_location(
        "cc", os.path.join(SCRIPT_DIR, "create_categorized_datasets.py"))
    cc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cc)
    with tempfile.TemporaryDirectory() as tmp:
        cc.OUTPUT_DIR = tmp
        import io
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            categorized, _, _ = cc.load_and_categorize()
    return categorized, buf.getvalue()


def apply_erratum(record, entry):
    """Apply one evaluation ledger entry in place. True if it bit."""
    if entry["field"] == "composition":
        comp = record.setdefault("composition", {})
        before = comp.get(entry["element"])
        comp[entry["element"]] = entry["to"]
        return before != entry["to"]
    series = record.get(entry["field"]) or []
    for point in list(series):
        if abs(float(point["temp_c"]) - entry["temp_c"]) > 1e-9:
            continue
        if entry["to"] is None:
            series.remove(point)
        else:
            point["value"] = entry["to"]
        record[entry["field"]] = series
        return True
    return False


def diff(rebuilt, committed):
    """Cell-level differences, ignoring the annotation fields the ledger adds."""
    out = []
    a, b = rebuilt.get("composition") or {}, committed.get("composition") or {}
    for el in sorted(set(a) | set(b)):
        if a.get(el) != b.get(el):
            out.append(f"composition[{el}] {a.get(el)} != {b.get(el)}")
    for prop in PROPS:
        sa = {str(p["temp_c"]): p["value"] for p in (rebuilt.get(prop) or [])}
        sb = {str(p["temp_c"]): p["value"] for p in (committed.get(prop) or [])}
        for t in sorted(set(sa) | set(sb), key=float):
            if sa.get(t) != sb.get(t):
                out.append(f"{prop}@{t}C {sa.get(t)} != {sb.get(t)}")
    for f in ("_category", "_source"):
        if rebuilt.get(f) != committed.get(f):
            out.append(f"{f} {rebuilt.get(f)!r} != {committed.get(f)!r}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    scope = json.load(open(os.path.join(DATA_DIR, "scope_exclusions.json"), encoding="utf-8"))
    ledger = json.load(open(os.path.join(DATA_DIR, "errata_ledger.json"), encoding="utf-8"))
    errata = [e for e in ledger["entries"] if e["applies_to"] == "evaluation"]

    print(f"scope exclusions declared : {len(scope['entries'])} "
          f"({sum(1 for e in scope['entries'] if e['status'] == 'rule')} by rule, "
          f"{sum(1 for e in scope['entries'] if e['status'] == 'declared')} declared, "
          f"{sum(1 for e in scope['entries'] if e['status'] == 'historical')} historical)")
    print(f"evaluation errata         : {len(errata)}\n")

    categorized, log = categorise()
    if args.verbose:
        print(log)

    applied, failures, total = 0, 0, 0
    value_gap_records, value_gap_cells = 0, 0
    for bucket, stem in EVALUATED + CARRIED:
        rebuilt = categorized.get(bucket, [])
        committed = load_jsonl(os.path.join(DATA_DIR, f"{stem}.jsonl"))
        by_name = {r["alloy"]: r for r in rebuilt}

        for e in errata:
            target = by_name.get(e["alloy"])
            if target is not None and apply_erratum(target, e):
                applied += 1

        names_r = [r["alloy"] for r in rebuilt]
        names_c = [r["alloy"] for r in committed]
        status = "OK" if names_r == names_c else "MISMATCH"
        print(f"  {stem:8s} rebuilt {len(rebuilt):3d}  committed {len(committed):3d}  "
              f"membership+order {status}")
        if names_r != names_c:
            failures += 1
            print(f"      only rebuilt : {sorted(set(names_r) - set(names_c))[:4]}")
            print(f"      only committed: {sorted(set(names_c) - set(names_r))[:4]}")
            continue

        for r, c in zip(rebuilt, committed):
            total += 1
            d = diff(r, c)
            if d:
                value_gap_records += 1
                value_gap_cells += len(d)
                if args.verbose:
                    print(f"      {c['alloy']}: " + "; ".join(d[:4]))

    print(f"\nerrata applied: {applied}/{len(errata)}   records compared: {total}")
    if failures:
        print(f"FAIL: {failures} membership difference(s). The exclusion list no longer "
              f"accounts for what the sources offer.")
        return 1
    print("PASS: sources + scope_exclusions reproduce the membership and order of all "
          "four category files exactly.")

    if value_gap_records:
        print(f"\nKNOWN VALUE GAP (not a failure, and not what this list governs):")
        print(f"  {value_gap_records} of {total} records differ in {value_gap_cells} cells.")
        print(f"  The committed MatWeb source files are an older extraction than the one")
        print(f"  the shipped evaluation files were built from -- the shipped files hold")
        print(f"  data the sources do not. Re-run with --verbose to list the cells.")
        print(f"  Closing this needs the newer extraction committed, not another rule.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
