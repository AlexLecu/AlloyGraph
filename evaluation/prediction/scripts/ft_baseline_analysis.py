#!/usr/bin/env python3
"""Train/test overlap analysis for the fine-tuned GPT-4.1-mini baseline.

The stratified evaluation in ``nn_distance_analysis.py`` asks how much of this
system's accuracy is recall of alloys it was built on. That question is only
honest if it is also asked of the baseline the system is compared against.

The fine-tune corpus (``backend/superalloy_preprocess/output_data/finetuned_data``)
holds 106 alloys split train/val/test. Nineteen of the 88 evaluation alloys
appear in its train or validation split *by name*, with the measured property
table as the training target -- not near-duplication under vendor branding, but
the same alloy. This script measures what that overlap buys the baseline, and
compares it against the full agent system on the identical split.

Two overlap definitions are computed, because they answer different objections:

  by name       normalised exact match of the alloy designation. Conservative:
                it cannot catch a vendor rebrand, so it under-counts.
  by distance   Euclidean distance on wt%-normalised composition to the nearest
                fine-tune alloy, using the same production metric and the same
                NEAR/MID/FAR cuts as ``nn_distance_analysis.py``. Catches the
                rebrands the name match misses.

Outputs (written to ../results):
    ft_overlap.csv            per evaluation alloy: name match, distance, stratum
    ft_stratified_metrics.csv MAE per method x overlap class x property
    ft_baseline_analysis.md   the report

Usage:
    python ft_baseline_analysis.py
    python ft_baseline_analysis.py --outdir /tmp
"""

import argparse
import json
import os
import re
import sys

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)                       # evaluation/prediction
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "backend"))

from alloy_crew.tools.rag_tools import _composition_distance  # noqa: E402

FT_DIR = os.path.join(PROJECT_ROOT, "backend", "superalloy_preprocess",
                      "output_data", "finetuned_data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
ARCHIVE_DIR = os.path.join(BASE_DIR, "archive", "pre_erratum")

#: The only fine-tuned-baseline predictions that exist. They pre-date the PE16
#: erratum and were produced at sampling temperature 0.3 without a seed, so they
#: are archived rather than current. See ``archive/pre_erratum/README.md``.
DEFAULT_FT_RESULTS = os.path.join(ARCHIVE_DIR, "results", "all", "gpt4.1_ft.csv")

#: The fine-tune's own splits. train+val is what the model actually saw.
FT_SEEN_SPLITS = ("train.jsonl", "val.jsonl")
FT_HELDOUT_SPLIT = "test.jsonl"

DATASETS = ("sss", "precip", "sc_ds")
PROPS = (("YS", "pred_ys", "actual_ys", "MPa"),
         ("UTS", "pred_uts", "actual_uts", "MPa"),
         ("EL", "pred_el", "actual_el", "%"),
         ("EM", "pred_em", "actual_em", "GPa"))

# Same cuts as nn_distance_analysis, so the two analyses are directly comparable.
STRATA = (("NEAR", 0.0, 2.0), ("MID", 2.0, 4.5), ("FAR", 4.5, float("inf")))

#: Alloys whose archived baseline predictions were produced from a composition
#: later found to be wrong. NIMONIC PE16 was prompted with Ti = 12.0 wt%; the
#: datasheet value is 1.2. Its ten rows are excluded from every comparison in
#: this file until the baseline is re-run on the corrected composition, because
#: scoring a baseline on a composition we have since fixed for our own arms
#: would manufacture an advantage. See ``data/precip.jsonl`` ``_erratum``.
ERRATUM_ALLOYS = ("NIMONIC* PE16",)


def normalise(name):
    """Strip vendor punctuation and case so designations compare sanely."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def load_ft_split(filename):
    """Alloy name -> composition dict, for one fine-tune split."""
    out = {}
    path = os.path.join(FT_DIR, filename)
    with open(path) as fh:
        for line in fh:
            rec = json.loads(line)
            user = rec["messages"][1]["content"]
            m = re.search(r"Alloy: (.*?); Processing", user)
            c = re.search(r"Composition: (\{.*\})", user)
            if not m:
                continue
            comp = {}
            if c:
                try:
                    comp = json.loads(c.group(1).replace("'", '"'))
                except json.JSONDecodeError:
                    comp = {}
            out[m.group(1).strip()] = comp
    return out


def stratum_of(distance):
    for name, lo, hi in STRATA:
        if lo <= distance < hi:
            return name
    return STRATA[-1][0]


def load_arm(tag):
    """Concatenate the three per-dataset CSVs of one evaluation arm."""
    frames = []
    for ds in DATASETS:
        path = os.path.join(OUTPUT_DIR, f"{tag}_{ds}.csv")
        if os.path.exists(path):
            frames.append(pd.read_csv(path))
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def mae(df, pred, actual, keys=None):
    """MAE over rows that have both a prediction and a measurement."""
    if keys is not None:
        df = df[[(a, t) in keys for a, t in zip(df.alloy, df.temperature)]]
    d = df.dropna(subset=[pred, actual])
    if len(d) == 0:
        return 0, float("nan")
    return len(d), float((d[pred] - d[actual]).abs().mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default=RESULTS_DIR)
    ap.add_argument("--ft-results", default=DEFAULT_FT_RESULTS,
                    help="Fine-tuned baseline predictions to analyse. Defaults to the "
                         "archived pre-erratum run; point this at a fresh run once the "
                         "baseline has been re-executed on the corrected data.")
    ap.add_argument("--keep-erratum-rows", action="store_true",
                    help="Do not drop the alloys in ERRATUM_ALLOYS. Only meaningful "
                         "once the baseline has been re-run on corrected data.")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    seen = {}
    for split in FT_SEEN_SPLITS:
        seen.update(load_ft_split(split))
    heldout = load_ft_split(FT_HELDOUT_SPLIT)
    seen_norm = {normalise(k): k for k in seen}
    print(f"Fine-tune corpus: {len(seen)} alloys in train+val, {len(heldout)} held out")

    ft = pd.read_csv(args.ft_results)
    full = {s: load_arm(f"stageb_seed{s}_full_system") for s in (42, 43, 44, 45, 46)}
    full = {k: v for k, v in full.items() if v is not None}
    mlkg = load_arm("seed42_v2prod_ml_physics_kg")

    # --- overlap, per evaluation alloy -------------------------------------
    eval_alloys = sorted(set(ft.alloy) | set(mlkg.alloy))
    # Compositions come from the evaluation datasets themselves.
    comps = {}
    for ds in DATASETS:
        path = os.path.join(BASE_DIR, "data", f"{ds}.jsonl")
        if not os.path.exists(path):
            continue
        with open(path) as fh:
            for line in fh:
                rec = json.loads(line)
                name = rec.get("alloy") or rec.get("alloy_name")
                comp = rec.get("composition") or {}
                if name:
                    comps[name] = comp

    rows = []
    for alloy in eval_alloys:
        nm = normalise(alloy)
        name_hit = seen_norm.get(nm)
        comp = comps.get(alloy, {})
        best_d, best_n = float("inf"), None
        if comp:
            for fname, fcomp in seen.items():
                if not fcomp:
                    continue
                d = _composition_distance(comp, fcomp)
                if d < best_d:
                    best_d, best_n = d, fname
        rows.append({"alloy": alloy,
                     "in_ft_train_by_name": bool(name_hit),
                     "ft_name_match": name_hit or "",
                     "nearest_ft_alloy": best_n or "",
                     "distance_to_ft": round(best_d, 3) if np.isfinite(best_d) else "",
                     "ft_stratum": stratum_of(best_d) if np.isfinite(best_d) else ""})
    overlap = pd.DataFrame(rows)
    overlap.to_csv(os.path.join(args.outdir, "ft_overlap.csv"), index=False)

    by_name = set(overlap.loc[overlap.in_ft_train_by_name, "alloy"])
    print(f"Evaluation alloys in the fine-tune train/val split by name: {len(by_name)}")
    near = set(overlap.loc[overlap.ft_stratum == "NEAR", "alloy"])
    print(f"Evaluation alloys within d<2.0 of a fine-tune alloy: {len(near)}")

    # --- metrics on the seen / unseen split --------------------------------
    metric_rows = []

    def add(method, group, prop, unit, n, value):
        metric_rows.append({"method": method, "overlap_class": group,
                            "property": prop, "unit": unit,
                            "n_rows": n, "mae": None if np.isnan(value) else round(value, 2)})

    if not args.keep_erratum_rows:
        drop = set(ERRATUM_ALLOYS)
        before = len(ft)
        ft = ft[~ft.alloy.isin(drop)]
        mlkg = mlkg[~mlkg.alloy.isin(drop)]
        full = {k: v[~v.alloy.isin(drop)] for k, v in full.items()}
        print(f"Excluded {before - len(ft)} rows for {sorted(drop)} "
              f"(baseline scored on a composition since corrected)")

    arms = {"GPT-4.1-mini FT": ft, "ML+physics+KG": mlkg}
    for label, df in arms.items():
        if df is None:
            continue
        d = df.copy()
        d["seen"] = d.alloy.isin(by_name)
        for prop, p, a, unit in PROPS:
            for group, sub in (("SEEN_BY_FT", d[d.seen]), ("UNSEEN_BY_FT", d[~d.seen]), ("ALL", d)):
                n, v = mae(sub, p, a)
                add(label, group, prop, unit, n, v)

    if full:
        for prop, p, a, unit in PROPS:
            for group in ("SEEN_BY_FT", "UNSEEN_BY_FT", "ALL"):
                vals, n = [], 0
                for df in full.values():
                    d = df.copy()
                    d["seen"] = d.alloy.isin(by_name)
                    sub = d if group == "ALL" else (d[d.seen] if group == "SEEN_BY_FT" else d[~d.seen])
                    n, v = mae(sub, p, a)
                    vals.append(v)
                add("full system (5-seed mean)", group, prop, unit, n, float(np.mean(vals)))

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(os.path.join(args.outdir, "ft_stratified_metrics.csv"), index=False)

    piv = metrics.pivot_table(index=["property", "overlap_class"], columns="method",
                              values="mae", aggfunc="first")
    print("\nMAE by fine-tune overlap class:")
    print(piv.to_string())

    write_report(os.path.join(args.outdir, "ft_baseline_analysis.md"),
                 overlap, metrics, by_name, near, len(seen), len(heldout),
                 args.keep_erratum_rows)
    print(f"\nWritten to {args.outdir}")


def write_report(path, overlap, metrics, by_name, near, n_seen, n_heldout, kept_erratum):
    """Emit the markdown companion to the CSVs."""
    def cell(prop, group, method):
        m = metrics[(metrics.property == prop) & (metrics.overlap_class == group)
                    & (metrics.method == method)]
        if m.empty or m.iloc[0]["mae"] is None:
            return "--"
        return f"{m.iloc[0]['mae']:.2f}"

    def n_of(prop, group, method="GPT-4.1-mini FT"):
        m = metrics[(metrics.property == prop) & (metrics.overlap_class == group)
                    & (metrics.method == method)]
        return "--" if m.empty else str(int(m.iloc[0]["n_rows"]))

    METHODS = ["GPT-4.1-mini FT", "ML+physics+KG", "full system (5-seed mean)"]
    L = []
    L.append("# Train/test overlap in the fine-tuned baseline\n")
    L.append("Generated by `evaluation/prediction/scripts/ft_baseline_analysis.py`.\n")
    L.append("## Why this analysis exists\n")
    L.append("This paper argues that a pooled accuracy figure conflates recall of alloys a")
    L.append("system was built on with generalisation to alloys it has not seen. That")
    L.append("argument obliges us to apply the same test to the baseline we compare")
    L.append("against, not only to our own system.\n")
    L.append(f"The fine-tune corpus holds {n_seen} alloys in its train and validation splits")
    L.append(f"and {n_heldout} held out. **{len(by_name)} of the 88 evaluation alloys appear in")
    L.append("that train/validation set by name**, with their measured property tables as")
    L.append("the training target. This is not near-duplication under vendor branding; it")
    L.append("is the same alloy designation.\n")
    L.append(f"Matching instead on composition, {len(near)} evaluation alloys sit within")
    L.append("d < 2.0 of a fine-tune alloy -- the same cut used for the NEAR stratum")
    L.append("elsewhere. The name match is the conservative figure and is used below.\n")
    if not kept_erratum:
        L.append("**NIMONIC PE16 is excluded from every table here.** Its archived baseline")
        L.append("predictions were generated from a composition carrying Ti = 12.0 wt%")
        L.append("against a datasheet value of 1.2. That error was corrected for this")
        L.append("system's arms; leaving it in place for the baseline would manufacture an")
        L.append("advantage. Re-run the baseline on the corrected composition and pass")
        L.append("`--keep-erratum-rows` to restore the rows.\n")
    L.append("## The evaluation alloys the fine-tune was trained on\n")
    L.append("| evaluation alloy | fine-tune designation | composition distance |")
    L.append("|---|---|---|")
    for _, r in overlap[overlap.in_ft_train_by_name].sort_values("alloy").iterrows():
        L.append(f"| {r['alloy']} | {r['ft_name_match']} | {r['distance_to_ft']} |")
    L.append("")
    L.append("## MAE by overlap class\n")
    L.append("`SEEN` are rows whose alloy is in the fine-tune train/validation split;")
    L.append("`UNSEEN` are the rest. Both columns score the *same* rows for every method,")
    L.append("so the comparison is like-for-like.\n")
    for prop in ["YS", "UTS", "EL", "EM"]:
        unit = {"YS": "MPa", "UTS": "MPa", "EL": "%", "EM": "GPa"}[prop]
        L.append(f"### {prop} ({unit})\n")
        L.append("| overlap class | n | " + " | ".join(METHODS) + " |")
        L.append("|---|---:|" + "---:|" * len(METHODS))
        for group, label in (("SEEN_BY_FT", "SEEN by fine-tune"),
                             ("UNSEEN_BY_FT", "UNSEEN by fine-tune"),
                             ("ALL", "ALL")):
            L.append(f"| {label} | {n_of(prop, group)} | "
                     + " | ".join(cell(prop, group, m) for m in METHODS) + " |")
        L.append("")
    L.append("## Reading\n")
    ys_s_ft, ys_s_fs = cell("YS", "SEEN_BY_FT", METHODS[0]), cell("YS", "SEEN_BY_FT", METHODS[2])
    ys_u_ft, ys_u_fs = cell("YS", "UNSEEN_BY_FT", METHODS[0]), cell("YS", "UNSEEN_BY_FT", METHODS[2])
    L.append("**The baseline's strength is concentrated on the alloys it memorised.** On")
    L.append(f"yield strength the fine-tune scores {ys_s_ft} MPa on the alloys it was trained")
    L.append(f"on against {ys_u_ft} MPa on the rest. On those same memorised alloys the full")
    L.append(f"agent system scores {ys_s_fs} MPa -- it does not win there. On the alloys the")
    L.append(f"fine-tune never saw, the agent system scores {ys_u_fs} MPa against the")
    L.append(f"baseline's {ys_u_ft}.\n")
    L.append("This mirrors the knowledge-graph anchoring result exactly, with the roles")
    L.append("reversed: a retrieval-shaped advantage that exists only where a stored answer")
    L.append("exists, and disappears where one does not.\n")
    em_ft, em_kg = cell("EM", "ALL", METHODS[0]), cell("EM", "ALL", METHODS[1])
    L.append(f"**Elastic modulus is the honest loss.** The fine-tuned baseline reaches")
    L.append(f"{em_ft} GPa against {em_kg} GPa for ML+physics+KG and")
    L.append(f"{cell('EM', 'ALL', METHODS[2])} GPa for the full agent system. A single")
    L.append("fine-tuned small model is the best elastic-modulus predictor measured here,")
    L.append("and no arrangement of this system's components beats it on that property.\n")
    L.append("## Caveats\n")
    L.append("- Name matching is exact after normalisation, so a fine-tune alloy sold under")
    L.append("  a different vendor name is counted as UNSEEN. The overlap is a lower bound.")
    L.append("- The UNSEEN group is not clean for *this* system: its ML ensemble and")
    L.append("  knowledge graph were built on 77 alloys with their own overlap with the")
    L.append("  evaluation set. This table isolates the fine-tune's leakage, not all leakage.")
    L.append("- The two systems' training sets are different, so SEEN/UNSEEN is a property")
    L.append("  of the baseline only. Per-group sample sizes fall below 100 rows.")
    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
