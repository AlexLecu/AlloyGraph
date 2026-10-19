#!/usr/bin/env python3
"""Conformal coverage at the delivered predictions, and the platform's calibration file.

conformal_intervals.py builds CV+ intervals for the ML ensemble: each interval is
centred, in effect, on the ensemble's own prediction, and only there does the CV+
marginal guarantee hold. The platform, however, delivers the full system's value
after agent reasoning and deterministic post-processing, which is a different
number. Expert Systems R2.3 asks which of the two the intervals describe.

This script answers it in two parts.

1. Coverage at the delivered point. Each row's CV+ offsets (lower - centre,
   upper - centre) are transported onto the full system's prediction -- the
   row-wise mean over the five repeat campaigns -- and scored against the
   measurement. This carries NO formal guarantee: the agent-adjusted value is not
   the quantity CV+ was calibrated around. It is an empirical check only.

2. The platform's calibration. CV+ widths vary by under 5% across all 471 rows
   (an absolute-residual score gives one quantile per property), so the platform
   applies one pair of offsets per property: the median lower and upper offset of
   the committed CV+ intervals. Coverage of that constant-offset interval at the
   delivered points is reported beside the per-row version, so the approximation
   is measured rather than assumed. The offsets are written to
   backend/alloy_crew/models/saved_models_v2/conformal_offsets.json, the file the
   platform reads.

Nothing here is tuned on the evaluation rows: the offsets are read from intervals
calibrated on the 77 training alloys, and the evaluation set only scores them.

Outputs:
    ../results/conformal_delivered.md
    ../results/conformal_delivered.csv
    backend/alloy_crew/models/saved_models_v2/conformal_offsets.json

Usage:
    python conformal_delivered.py
"""

import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import revision_robustness as rr  # noqa: E402

CONFORMAL_CSV = os.path.join(rr.RESULTS_DIR, "conformal_intervals.csv")
OFFSETS_JSON = os.path.join(rr.PROJECT_ROOT, "backend", "alloy_crew", "models",
                            "saved_models_v2", "conformal_offsets.json")

#: Platform property names, as the evaluator's output dictionary carries them.
PLATFORM_NAME = {"ys": "Yield Strength", "uts": "Tensile Strength",
                 "el": "Elongation", "em": "Elastic Modulus"}
#: The physical domain conformal_intervals.py clips to.
PHYSICAL_BOUNDS = {"ys": (0.0, None), "uts": (0.0, None), "el": (0.0, 100.0), "em": (0.0, None)}
NOMINAL = 0.90


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        h.update(fh.read())
    return h.hexdigest()


def main():
    conf = pd.read_csv(CONFORMAL_CSV)
    frame = rr.build_frame().set_index("k")
    conf["k"] = list(zip(conf.alloy, conf.temperature.round(1)))

    offsets, rows = {}, []
    for p, label, unit in rr.PROPS:
        c = conf[conf.property == p].copy()
        c["off_lo"] = c.lower - c.pred_cv_plus
        c["off_hi"] = c.upper - c.pred_cv_plus
        med_lo, med_hi = float(c.off_lo.median()), float(c.off_hi.median())
        offsets[PLATFORM_NAME[p]] = {
            "lower_offset": round(med_lo, 3), "upper_offset": round(med_hi, 3), "unit": unit,
            "physical_min": PHYSICAL_BOUNDS[p][0], "physical_max": PHYSICAL_BOUNDS[p][1],
            "row_width_range": [round(float((c.off_hi - c.off_lo).min()), 3),
                                round(float((c.off_hi - c.off_lo).max()), 3)],
        }

        s = c.dropna(subset=["actual"]).copy()
        delivered = s.k.map(frame[[f"full{seed}_{p}" for seed in rr.SEEDS]].mean(axis=1))
        s = s[delivered.notna()]
        delivered = delivered[delivered.notna()]
        a = s.actual
        cov_ml = (a >= s.lower) & (a <= s.upper)
        cov_row = (a >= delivered + s.off_lo) & (a <= delivered + s.off_hi)
        cov_const = (a >= delivered + med_lo) & (a <= delivered + med_hi)
        for scope in rr.STRATA + ("ALL",):
            m = np.ones(len(s), bool) if scope == "ALL" else (s.stratum == scope).to_numpy()
            rows.append({
                "property": label, "stratum": scope, "n": int(m.sum()),
                "coverage_at_ml_centre": round(100 * float(cov_ml[m].mean()), 1),
                "coverage_delivered_row_offsets": round(100 * float(cov_row[m].mean()), 1),
                "coverage_delivered_constant_offsets": round(100 * float(cov_const[m].mean()), 1),
                "half_width": round((med_hi - med_lo) / 2, 1),
            })

    cov = pd.DataFrame(rows)
    cov.to_csv(os.path.join(rr.RESULTS_DIR, "conformal_delivered.csv"), index=False)

    payload = {
        "description": ("Per-property offsets of the 90% cross-conformal (CV+) interval of the "
                        "ML ensemble, applied by the platform around the delivered prediction. "
                        "The CV+ marginal guarantee holds at the ensemble's own prediction; at the "
                        "agent-adjusted value it is an empirical approximation, measured in "
                        "evaluation/prediction/results/conformal_delivered.md."),
        "nominal_coverage": NOMINAL,
        "method": "CV+ (MAPIE CrossConformalRegressor, method='plus'), GroupKFold(5) on alloy",
        "calibration_data": "backend/alloy_crew/models/training_data/train_77alloys_v2.jsonl",
        "models": "saved_models_v2",
        "source": "evaluation/prediction/results/conformal_intervals.csv",
        "source_sha256": sha256(CONFORMAL_CSV),
        "generated_by": "evaluation/prediction/scripts/conformal_delivered.py",
        "properties": offsets,
    }
    with open(OFFSETS_JSON, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")

    L = ["# Conformal coverage at the delivered predictions", "",
         "Generated by `evaluation/prediction/scripts/conformal_delivered.py`. Nominal coverage 90%. "
         "The CV+ intervals of `conformal_intervals.py` are calibrated on the ML ensemble; here their "
         "offsets are moved onto the full system's delivered value (row-wise mean of the five repeat "
         "campaigns). That transport has **no formal guarantee** -- the agent-adjusted value is not "
         "the quantity CV+ was calibrated around -- so these columns are an empirical check.", "",
         "*Constant offsets* is what the platform applies: one median offset pair per property, "
         "written to `backend/alloy_crew/models/saved_models_v2/conformal_offsets.json`.", "",
         "| property | stratum | n | at ML centre (guaranteed marginally) | delivered, per-row offsets | delivered, constant offsets (platform) | half-width |",
         "|---|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        L.append(f"| {r['property']} | {r['stratum']} | {r['n']} | {r['coverage_at_ml_centre']}% | "
                 f"{r['coverage_delivered_row_offsets']}% | {r['coverage_delivered_constant_offsets']}% | "
                 f"{r['half_width']} |")
    L += ["", "Platform offsets (lower, upper relative to the delivered value):", ""]
    for name, o in offsets.items():
        L.append(f"- {name}: {o['lower_offset']:+.1f} / {o['upper_offset']:+.1f} {o['unit']} "
                 f"(per-row width range {o['row_width_range'][0]:.1f}-{o['row_width_range'][1]:.1f})")
    L.append("")
    with open(os.path.join(rr.RESULTS_DIR, "conformal_delivered.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
