#!/usr/bin/env python3
"""Wilcoxon signed-rank tests: full system against ML+physics+KG.

The headline comparison in the paper is a difference of two MAEs. A difference
of means says nothing about whether the improvement is consistent across cases
or driven by a few large ones, and with 285 yield-strength rows over 88 alloys
that distinction matters. This pairs the two arms case by case and asks whether
the absolute errors differ.

The test is Wilcoxon signed-rank on paired absolute errors, two-sided, with
zero differences dropped by the default ``wilcoxon`` behaviour. It is the right
test here rather than a paired t-test: absolute errors are bounded below at
zero and heavily right-skewed, so their differences are not normal.

The full system is stochastic. Predictions are the row-wise mean over the five
campaign seeds, matching how the parity figure and stratified tables treat it,
so the test asks about the averaged system rather than about one lucky run.

WHAT THIS TEST DOES NOT ESTABLISH. Cases are not independent: 285 rows come
from 88 alloys, several rows per alloy, and rows from one alloy share its
composition and its ML prediction. The p-values are therefore optimistic, and
the honest reading is descriptive -- how consistently one arm beats the other
across cases -- rather than a population inference. An alloy-level test is
reported alongside for that reason: it aggregates each alloy to its mean
absolute error first, so the units being compared are independent alloys.

Usage:
    python wilcoxon_tests.py
    python wilcoxon_tests.py --csv ../results/wilcoxon_tests.csv
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
OUTPUT = os.path.join(BASE_DIR, "output")
RESULTS = os.path.join(BASE_DIR, "results")

DATASETS = ("sss", "precip", "sc_ds")
SEEDS = (42, 43, 44, 45, 46)
PROPS = (("ys", "YS", "MPa"), ("uts", "UTS", "MPa"),
         ("el", "EL", "%"), ("em", "EM", "GPa"))

KEY = ["alloy", "temperature"]


def load_arm(tag):
    return pd.concat([pd.read_csv(os.path.join(OUTPUT, f"{tag}_{ds}.csv"))
                      for ds in DATASETS], ignore_index=True)


def full_system_mean():
    """Row-wise mean prediction over the five campaign seeds."""
    frames = [load_arm(f"stageb_seed{s}_full_system") for s in SEEDS]
    cols = [f"pred_{k}" for k, _, _ in PROPS]
    base = frames[0][KEY + cols + [f"actual_{k}" for k, _, _ in PROPS]].copy()
    for c in cols:
        stack = np.vstack([f.set_index(KEY)[c].reindex(
            pd.MultiIndex.from_frame(base[KEY])).values for f in frames])
        base[c] = np.nanmean(stack, axis=0)
    return base


def paired(a, b, key):
    """Absolute errors for both arms on the cases both scored."""
    m = a[KEY + [f"actual_{key}", f"pred_{key}"]].merge(
        b[KEY + [f"pred_{key}"]], on=KEY, suffixes=("_a", "_b"))
    m = m.dropna(subset=[f"actual_{key}", f"pred_{key}_a", f"pred_{key}_b"])
    m = m[m[f"actual_{key}"] != 0]
    return (m.assign(err_a=(m[f"pred_{key}_a"] - m[f"actual_{key}"]).abs(),
                     err_b=(m[f"pred_{key}_b"] - m[f"actual_{key}"]).abs()))


def test(err_a, err_b, label, scope, prop, unit, unit_of):
    """One signed-rank test. err_a is the full system, err_b the KG arm."""
    d = np.asarray(err_a) - np.asarray(err_b)
    nz = int((d != 0).sum())
    if nz < 6:
        return {"comparison": label, "scope": scope, "property": prop,
                "n": len(d), "n_nonzero": nz, "W": None, "p": None,
                "median_delta": None, "wins": None, "losses": None,
                "direction": "too few non-zero differences", "unit": unit,
                "units_compared": unit_of}
    stat, p = wilcoxon(err_a, err_b, alternative="two-sided")
    better = int((d < 0).sum())
    worse = int((d > 0).sum())
    return {"comparison": label, "scope": scope, "property": prop,
            "n": len(d), "n_nonzero": nz, "W": float(stat), "p": float(p),
            "median_delta": float(np.median(d)),
            "wins": better, "losses": worse,
            "direction": ("full system lower error" if np.median(d) < 0
                          else "KG arm lower error"),
            "unit": unit, "units_compared": unit_of}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=os.path.join(RESULTS, "wilcoxon_tests.csv"))
    args = ap.parse_args()

    dist = pd.read_csv(os.path.join(RESULTS, "nn_distance.csv"))
    far = set(dist[dist.stratum == "FAR"].alloy)

    full = full_system_mean()
    kg = load_arm("seed42_v2prod_ml_physics_kg")

    rows = []
    for key, prop, unit in PROPS:
        m = paired(full, kg, key)
        for scope, sub in (("ALL", m), ("FAR", m[m.alloy.isin(far)])):
            rows.append(test(sub.err_a.values, sub.err_b.values,
                             "Full system vs ML+physics+KG", scope, prop, unit,
                             "cases (alloy x temperature)"))
            # Alloy-level: one independent unit per alloy, mean absolute error.
            g = sub.groupby("alloy")[["err_a", "err_b"]].mean()
            rows.append(test(g.err_a.values, g.err_b.values,
                             "Full system vs ML+physics+KG", scope + " (by alloy)",
                             prop, unit, "alloys"))

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.csv), exist_ok=True)
    df.to_csv(args.csv, index=False)

    print(f"{'scope':18s} {'prop':4s} {'n':>4s} {'nz':>4s} {'W':>10s} "
          f"{'p':>10s} {'med delta':>10s} {'win/loss':>10s}")
    print("-" * 78)
    for r in rows:
        if r["W"] is None:
            print(f"{r['scope']:18s} {r['property']:4s} {r['n']:4d} "
                  f"{r['n_nonzero']:4d}  {r['direction']}")
            continue
        print(f"{r['scope']:18s} {r['property']:4s} {r['n']:4d} {r['n_nonzero']:4d} "
              f"{r['W']:10.1f} {r['p']:10.3g} {r['median_delta']:+10.2f} "
              f"{r['wins']:5d}/{r['losses']:<4d}")
    print(f"\nWritten to {args.csv}")


if __name__ == "__main__":
    main()
