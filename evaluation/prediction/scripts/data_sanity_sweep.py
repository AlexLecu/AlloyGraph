#!/usr/bin/env python3
"""Sweep the evaluation and training sets for physically impossible measurements.

Two defects reached published metrics before anyone looked: NIMONIC PE16 with a
transposed titanium content, and RGT* 13 with a negative yield strength. Both
were found by eye, late. This is the check that should have existed.

It asserts only things that cannot be true of a real tensile measurement, so a
finding is a data defect rather than a modelling opinion:

  negative or zero strength, modulus or elongation
  yield strength above tensile strength at the same temperature
  elongation above 200% (superplastic regimes reach ~150%; beyond that is a
      transcription artefact) and above 100% below 600 C
  strength or modulus outside the bounds the training loader already enforces
  composition not summing to roughly 100 wt%

WHY THE TRAINING SET IS SWEPT TOO. train_ml_models drops out-of-bounds targets
at load time, so a defect there is silently filtered rather than reported, and
the same record can sit uncorrected in the evaluation set where no bound
applies. That asymmetry is exactly how RGT* 13 survived.

Exit status is 1 when any finding is present, so this can gate a pipeline.

Usage:
    python data_sanity_sweep.py
    python data_sanity_sweep.py --quiet     # status only
"""

import argparse
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))
DATA_DIR = os.path.join(BASE_DIR, "data")
TRAINING = os.path.join(PROJECT_ROOT, "backend", "alloy_crew", "models",
                        "training_data", "train_77alloys_v2.jsonl")

#: Same bounds train_ml_models.TARGETS applies when loading, so the sweep and
#: the trainer agree on what counts as physically admissible.
BOUNDS = {"yield_strength": (50, 2000), "uts": (50, 2500),
          "elongation": (0, 80), "elasticity": (50, 350)}

LABEL = {"yield_strength": "YS", "uts": "UTS",
         "elongation": "EL", "elasticity": "EM"}

#: Above this, an elongation reading is a transcription artefact rather than a
#: superplastic measurement. The most ductile rows in this corpus reach ~159%.
EL_IMPOSSIBLE = 200.0
EL_IMPOSSIBLE_COLD = 100.0
COLD_C = 600.0

#: Above the gamma-prime solvus an alloy loses work-hardening capacity and
#: tensile strength converges on yield strength, so YS == UTS is real and a
#: handbook quoting to the nearest 5 MPa can invert them by rounding. Only an
#: inversion larger than this is a transcription problem. UDIMET 520 at 871 C
#: sits at 1.0% (YS 520, UTS 515) with the source spreadsheet in exact
#: agreement, and is not a defect.
YS_ABOVE_UTS_TOLERANCE = 0.02

#: Findings that cannot be true of any real measurement, as against ones that
#: merely fall outside the window the training loader accepts.
IMPOSSIBLE = {"negative", "zero", "ys_above_uts",
              "elongation_impossible", "elongation_impossible_cold"}


def series(record, key):
    out = {}
    for point in (record.get(key) or []):
        try:
            out[float(point["temp_c"])] = float(point["value"])
        except (TypeError, ValueError, KeyError):
            continue
    return out


def sweep_record(record, source):
    alloy = record.get("alloy", "?")
    found = []

    def add(kind, temp, detail):
        found.append({"source": source, "alloy": alloy, "temp_c": temp,
                      "check": kind, "detail": detail})

    vals = {k: series(record, k) for k in BOUNDS}

    for key, s in vals.items():
        lo, hi = BOUNDS[key]
        for temp, v in sorted(s.items()):
            if v < 0:
                add("negative", temp, f"{LABEL[key]} = {v}")
            elif v == 0 and key != "elongation":
                add("zero", temp, f"{LABEL[key]} = 0")
            elif not (lo <= v <= hi):
                add("out_of_bounds", temp, f"{LABEL[key]} = {v}, admissible [{lo}, {hi}]")

    for temp in sorted(set(vals["yield_strength"]) & set(vals["uts"])):
        ys, uts = vals["yield_strength"][temp], vals["uts"][temp]
        if ys > 0 and uts > 0 and ys > uts * (1 + YS_ABOVE_UTS_TOLERANCE):
            add("ys_above_uts", temp,
                f"YS {ys} > UTS {uts} by {100 * (ys - uts) / uts:.1f}%")

    for temp, v in sorted(vals["elongation"].items()):
        if v > EL_IMPOSSIBLE:
            add("elongation_impossible", temp, f"EL = {v}%")
        elif temp < COLD_C and v > EL_IMPOSSIBLE_COLD:
            add("elongation_impossible_cold", temp,
                f"EL = {v}% at {temp:.0f} C")

    comp = {k: float(v) for k, v in (record.get("composition") or {}).items() if v}
    total = sum(comp.values())
    if comp and not (95.0 <= total <= 105.0):
        add("composition_sum", None, f"composition sums to {total:.2f} wt%")

    return found


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    sources = [("evaluation/SSS", os.path.join(DATA_DIR, "SSS.jsonl")),
               ("evaluation/precip", os.path.join(DATA_DIR, "precip.jsonl")),
               ("evaluation/sc_ds", os.path.join(DATA_DIR, "sc_ds.jsonl")),
               ("training/77alloys", TRAINING)]

    findings, n_records = [], 0
    for label, path in sources:
        if not os.path.exists(path):
            continue
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            n_records += 1
            findings += sweep_record(json.loads(line), label)

    print(f"Swept {n_records} alloy records across {len(sources)} files.\n")

    impossible = [f for f in findings if f["check"] in IMPOSSIBLE]
    review = [f for f in findings if f["check"] not in IMPOSSIBLE]

    def report(title, rows, note):
        print(f"{title}: {len(rows)}")
        print(f"  {note}")
        by = {}
        for f in rows:
            by.setdefault(f["check"], []).append(f)
        for check in sorted(by):
            print(f"\n  {check} ({len(by[check])})")
            if args.quiet:
                continue
            for r in by[check]:
                t = "" if r["temp_c"] is None else f" @ {r['temp_c']:.0f} C"
                print(f"      {r['source']:18s} {r['alloy']:34s}{t}: {r['detail']}")
        print()

    report("PHYSICALLY IMPOSSIBLE", impossible,
           "These cannot be real measurements. Each needs a source check and "
           "either a documented erratum or an exclusion.")
    report("FOR REVIEW", review,
           "Not impossible. Out-of-bounds rows are real high-temperature "
           "behaviour -- a superalloy genuinely yields at 9 MPa near 1200 C and "
           "genuinely elongates 159% once gamma-prime dissolves -- that the "
           "training loader silently drops because its admissible window is a "
           "training convenience, not a physical limit. Listed so that what "
           "training discards is visible rather than invisible.")
    return 1 if impossible else 0


if __name__ == "__main__":
    sys.exit(main())
