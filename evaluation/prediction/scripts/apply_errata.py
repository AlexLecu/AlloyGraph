#!/usr/bin/env python3
"""Propagate a declared data erratum into the committed prediction CSVs.

Every prediction CSV carries its own copy of the ground truth in the
``actual_*`` columns, snapshotted when the run happened. Correcting or
withdrawing a measurement afterwards leaves those copies stale, and every metric
built from them keeps scoring against a value the data no longer holds.

Re-running the arms would also fix it, but the agent arms cost real money and
their predictions do not depend on the measured value at all: a model sees
composition, processing and temperature. Only the ground truth changed.

WHY THIS IS A DECLARED LIST AND NOT A RE-SYNC. The obvious implementation --
rebuild every ``actual_*`` cell from the evaluation data files -- was written
first and rejected. The evaluation data records each property at its own
measured temperature (elongation at 650 C where yield strength is at 649 C, a
room-temperature modulus at 20 C where strength is at 21 C), and the prediction
harness matches those to a row with a tolerance. An exact re-match therefore
reported 902 "stale" values across 26 files and would have deleted hundreds of
perfectly good measurements. Naming the exact cells to change is the only safe
version.

Errata are listed in ERRATA below with their justification. Adding one means
adding an entry, not writing new logic.

Usage:
    python apply_errata.py            # dry run, prints what would change
    python apply_errata.py --apply
"""

import argparse
import glob
import os

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

#: (alloy, temperature_c, column, new_value, why). new_value None withdraws the
#: measurement, leaving the row scored on its other properties.
ERRATA = [
    ("RGT* 13", 871.0, "actual_ys", None,
     "Erratum 2. The annotated source records this cell verbatim as '-435', a "
     "negative yield strength, which cannot be a measurement. The source does "
     "not settle what it should be -- a stray dash on 435 MPa, or a 'no data' "
     "dash run together with the empty 982 C column -- so the measurement is "
     "withdrawn rather than guessed. UTS and elongation at 871 C are retained."),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--outdir", default=OUTPUT_DIR)
    args = ap.parse_args()

    for alloy, temp, col, new, why in ERRATA:
        print(f"{alloy} @ {temp:.0f} C  {col} -> "
              f"{'withdrawn' if new is None else new}")
        print(f"  {why}\n")

    total, touched = 0, []
    for path in sorted(glob.glob(os.path.join(args.outdir, "*.csv"))):
        if path.endswith("_errors.csv"):
            continue
        df = pd.read_csv(path)
        if not {"alloy", "temperature"} <= set(df.columns):
            continue
        dirty = False
        for alloy, temp, col, new, _ in ERRATA:
            if col not in df.columns:
                continue
            hit = (df.alloy == alloy) & (df.temperature.astype(float).sub(temp).abs() < 0.5)
            for i in df.index[hit]:
                have = df.at[i, col]
                if pd.isna(have) if new is None else (
                        not pd.isna(have) and abs(float(have) - new) < 1e-9):
                    continue
                df.at[i, col] = pd.NA if new is None else new
                total += 1
                dirty = True
        if dirty:
            touched.append(os.path.basename(path))
            if args.apply:
                df.to_csv(path, index=False)

    print(f"{total} cell(s) in {len(touched)} file(s): {', '.join(touched) or 'none'}")
    print("APPLIED" if args.apply else "DRY RUN -- rerun with --apply to write")
    print("No pred_* column was read or modified.")


if __name__ == "__main__":
    main()
