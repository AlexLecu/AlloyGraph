#!/usr/bin/env python3
"""Nearest-neighbour distance analysis and stratified evaluation.

Every evaluation alloy is scored by how far it sits, compositionally, from the
closest alloy the system was built on -- the same 77 alloys that train the ML
ensemble and populate the knowledge graph. That distance decides what a result
means: an alloy the system has effectively already seen tests recall, while a
genuinely unseen one tests generalisation. Reporting a single pooled number
averages the two together.

Distances are Euclidean on wt%-normalised compositions, using the production
metric (``rag_tools._composition_distance``) so the strata line up exactly with
the thresholds the KG anchoring gate applies at run time.

Strata
    NEAR   d < 2.0    near-duplicates; the same alloy under another vendor name
    MID    2.0 <= d < 4.5   anchorable: inside the KG gate, real weight possible
    FAR    d >= 4.5   beyond the anchoring cutoff; anchoring can never fire

Outputs (written to ../results and ../figures):
    nn_distance.csv                 per-alloy distances and stratum
    nn_distance_strata.csv          stratum x alloy-class counts
    stratified_metrics.csv          MAE/R2 per method x stratum x property
    nn_distance_hist.{png,pdf}      distance distribution figure

Usage:
    python nn_distance_analysis.py                  # uses default seed-42 CSVs
    python nn_distance_analysis.py --outdir /tmp    # write elsewhere
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)                       # evaluation/prediction
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))    # repo root
sys.path.insert(0, os.path.join(PROJECT_ROOT, "backend"))

from alloy_crew.tools.rag_tools import _composition_distance  # noqa: E402

TRAINING_JSONL = os.path.join(
    PROJECT_ROOT, "backend", "alloy_crew", "models", "training_data", "train_77alloys.jsonl"
)
DATASETS = {"sss": "SSS", "precip": "precip", "sc_ds": "sc_ds"}
CLASS_LABEL = {"sss": "SSS", "precip": "Precip", "sc_ds": "SC/DS"}

# (stratum, lower bound inclusive, upper bound exclusive)
STRATA = [("NEAR", 0.0, 2.0), ("MID", 2.0, 4.5), ("FAR", 4.5, float("inf"))]

METHODS = [
    ("ML-only", "seed42_v2_ml_only_{ds}.csv"),
    ("ML+physics", "seed42_v2prod_ml_deterministic_{ds}.csv"),
    ("ML+physics+KG", "seed42_v2prod_ml_physics_kg_{ds}.csv"),
    # Commercial LLM baseline, same corrected data and same seed. Included so
    # the strata answer the obvious question: does a general-purpose model also
    # do better on the alloys that are near-duplicates of our training set?
    ("GPT-4.1-mini", "seed42_gpt41mini_{ds}.csv"),
    ("Llama-3.3-70B", "seed42_llama70b_{ds}.csv"),
    ("GBM raw", "seed42_gbm_raw_{ds}.csv"),
    ("RF raw", "seed42_rf_raw_{ds}.csv"),
    ("GPR raw", "seed42_gpr_raw_{ds}.csv"),
]

#: Arms whose value is a mean over seeds rather than a single deterministic run.
#: The full agent system is stochastic, so its stratified numbers are a 5-seed
#: mean with the seed standard deviation reported alongside. Kept separate from
#: METHODS because the loader has to average across files rather than read one.
#:
#: This exists because the paper's headline agent result -- FAR yield strength --
#: was previously computed ad hoc and appeared in no generated table.
SEEDED_METHODS = [
    ("Full system (5 seeds)", "stageb_seed{seed}_full_system_{ds}.csv",
     (42, 43, 44, 45, 46)),
]

PROPERTIES = [("ys", "YS", "MPa"), ("uts", "UTS", "MPa"),
              ("el", "EL", "%"), ("em", "EM", "GPa")]


def stratum_of(distance):
    for name, lo, hi in STRATA:
        if lo <= distance < hi:
            return name
    return STRATA[-1][0]


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def compute_distances():
    """Nearest training alloy for every evaluation alloy."""
    training = load_jsonl(TRAINING_JSONL)
    rows = []
    for key, stem in DATASETS.items():
        for alloy in load_jsonl(os.path.join(BASE_DIR, "data", f"{stem}.jsonl")):
            comp = alloy.get("composition", {}) or {}
            if not comp:
                continue
            best_d, best_name = min(
                ((_composition_distance(comp, t.get("composition", {}) or {}), t.get("alloy", "?"))
                 for t in training),
                key=lambda pair: pair[0],
            )
            rows.append({
                "alloy": alloy.get("alloy", "?"),
                "alloy_class": CLASS_LABEL[key],
                "processing": alloy.get("processing", ""),
                "nearest_training_alloy": best_name,
                "distance": round(best_d, 3),
                "stratum": stratum_of(best_d),
            })
    df = pd.DataFrame(rows).sort_values("distance").reset_index(drop=True)
    return df


def metrics(actual, predicted):
    """MAE and R2 over the pairs where both values exist and actual is non-zero.

    Mirrors analyze_results.calculate_metrics so numbers are comparable.
    R2 needs at least two points and non-zero variance in the target.
    """
    a = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(p) | (a == 0))
    a, p = a[mask], p[mask]
    if len(a) == 0:
        return 0, np.nan, np.nan
    mae = float(np.mean(np.abs(a - p)))
    if len(a) < 2 or np.allclose(a, a[0]):
        return len(a), mae, np.nan
    ss_res = float(np.sum((a - p) ** 2))
    ss_tot = float(np.sum((a - np.mean(a)) ** 2))
    return len(a), mae, 1.0 - ss_res / ss_tot


def load_method(pattern, outdir_note):
    frames = []
    for ds in DATASETS:
        path = os.path.join(BASE_DIR, "output", pattern.format(ds=ds))
        if not os.path.exists(path):
            raise FileNotFoundError(f"missing prediction CSV: {path}\n{outdir_note}")
        frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True)


def load_seeded(pattern, seed, note):
    frames = []
    for ds in DATASETS:
        path = os.path.join(BASE_DIR, "output", pattern.format(ds=ds, seed=seed))
        if not os.path.exists(path):
            raise FileNotFoundError(f"missing prediction CSV: {path}\n{note}")
        frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True)


def stratified_metrics(dist_df):
    strat_by_alloy = dist_df.set_index("alloy")["stratum"]
    note = "Run generate_predictions.py for the missing mode/dataset first."
    records = []
    for method, pattern in METHODS:
        df = load_method(pattern, note)
        df = df.copy()
        df["stratum"] = df["alloy"].map(strat_by_alloy)
        unmapped = int(df["stratum"].isna().sum())
        if unmapped:
            print(f"  WARNING: {method}: {unmapped} rows had no distance entry", file=sys.stderr)
        for stratum in [s[0] for s in STRATA] + ["ALL"]:
            sub = df if stratum == "ALL" else df[df["stratum"] == stratum]
            for key, label, unit in PROPERTIES:
                n, mae, r2 = metrics(sub.get(f"actual_{key}"), sub.get(f"pred_{key}"))
                records.append({
                    "method": method, "stratum": stratum, "property": label, "unit": unit,
                    "n_rows": n,
                    "mae": None if np.isnan(mae) else round(mae, 2),
                    "mae_sd": None,
                    "r2": None if np.isnan(r2) else round(r2, 3),
                    "n_seeds": 1,
                })

    for method, pattern, seeds in SEEDED_METHODS:
        per_seed = []
        for seed in seeds:
            df = load_seeded(pattern, seed, note).copy()
            df["stratum"] = df["alloy"].map(strat_by_alloy)
            per_seed.append(df)
        for stratum in [s[0] for s in STRATA] + ["ALL"]:
            for key, label, unit in PROPERTIES:
                maes, r2s, n = [], [], 0
                for df in per_seed:
                    sub = df if stratum == "ALL" else df[df["stratum"] == stratum]
                    n, mae, r2 = metrics(sub.get(f"actual_{key}"), sub.get(f"pred_{key}"))
                    maes.append(mae)
                    r2s.append(r2)
                records.append({
                    "method": method, "stratum": stratum, "property": label, "unit": unit,
                    "n_rows": n,
                    "mae": None if np.isnan(np.mean(maes)) else round(float(np.mean(maes)), 2),
                    "mae_sd": None if np.isnan(np.mean(maes)) else round(float(np.std(maes, ddof=1)), 2),
                    "r2": None if np.isnan(np.mean(r2s)) else round(float(np.mean(r2s)), 3),
                    "n_seeds": len(seeds),
                })
    return pd.DataFrame(records)


def memorisation_check(dist_df):
    """How much of the ML-only result is recall rather than generalisation.

    Compares ML-only accuracy on NEAR against FAR. Pooled across all alloys
    this is confounded: the strata have different class mixes and different
    property ranges, so the comparison is repeated within each alloy class,
    which is the part that actually supports a claim.
    """
    strat = dist_df.set_index("alloy")["stratum"]
    klass = dist_df.set_index("alloy")["alloy_class"]
    df = load_method(METHODS[0][1], "").copy()
    df["stratum"] = df["alloy"].map(strat)
    df["alloy_class"] = df["alloy"].map(klass)

    records = []
    for scope, sub in [("ALL CLASSES", df)] + [(c, df[df.alloy_class == c])
                                               for c in ["SSS", "Precip", "SC/DS"]]:
        for key, plabel, _u in PROPERTIES:
            n_near, mae_near, _ = metrics(sub[sub.stratum == "NEAR"].get(f"actual_{key}"),
                                          sub[sub.stratum == "NEAR"].get(f"pred_{key}"))
            n_far, mae_far, _ = metrics(sub[sub.stratum == "FAR"].get(f"actual_{key}"),
                                        sub[sub.stratum == "FAR"].get(f"pred_{key}"))
            if n_near == 0 or n_far == 0:
                continue
            records.append({
                "scope": scope, "property": plabel,
                "n_near": n_near, "mae_near": round(mae_near, 2),
                "n_far": n_far, "mae_far": round(mae_far, 2),
                "near_minus_far_pct": round(100.0 * (mae_near - mae_far) / mae_far, 1),
            })
    return pd.DataFrame(records)


def make_histogram(dist_df, fig_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = {"NEAR": "#c0392b", "MID": "#e67e22", "FAR": "#2980b9"}
    fig, ax = plt.subplots(figsize=(7.2, 4.2))

    edges = np.arange(0, max(12.0, float(dist_df["distance"].max()) + 1.0) + 0.5, 0.5)
    for name, lo, hi in STRATA:
        vals = dist_df.loc[dist_df["stratum"] == name, "distance"]
        ax.hist(vals, bins=edges, color=colours[name], alpha=0.85,
                label=f"{name} (n={len(vals)})", edgecolor="white", linewidth=0.5)

    for bound in (2.0, 4.5):
        ax.axvline(bound, color="#2c3e50", linestyle="--", linewidth=1.1, zorder=5)
    ax.annotate("NEAR / MID\nd = 2.0", xy=(2.0, ax.get_ylim()[1] * 0.92),
                xytext=(2.35, ax.get_ylim()[1] * 0.92), fontsize=8, va="top", color="#2c3e50")
    ax.annotate("MID / FAR\nd = 4.5 (KG anchoring cutoff)", xy=(4.5, ax.get_ylim()[1] * 0.92),
                xytext=(4.85, ax.get_ylim()[1] * 0.92), fontsize=8, va="top", color="#2c3e50")

    ax.set_xlabel("Euclidean distance to nearest training alloy (wt%)")
    ax.set_ylabel("Evaluation alloys")
    ax.set_title("Compositional distance from the 88 evaluation alloys\n"
                 "to the 77 training / knowledge-graph alloys", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    os.makedirs(fig_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(fig_dir, f"nn_distance_hist.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default=os.path.join(BASE_DIR, "results"))
    ap.add_argument("--figdir", default=os.path.join(BASE_DIR, "figures"))
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    dist = compute_distances()
    dist.to_csv(os.path.join(args.outdir, "nn_distance.csv"), index=False)

    print(f"Evaluation alloys: {len(dist)}   training alloys: {len(load_jsonl(TRAINING_JSONL))}")
    print(f"distance: min {dist.distance.min():.2f}  median {dist.distance.median():.2f}  "
          f"max {dist.distance.max():.2f}\n")

    counts = (dist.pivot_table(index="stratum", columns="alloy_class",
                               values="alloy", aggfunc="count", fill_value=0)
              .reindex([s[0] for s in STRATA]).fillna(0).astype(int))
    counts["TOTAL"] = counts.sum(axis=1)
    counts.to_csv(os.path.join(args.outdir, "nn_distance_strata.csv"))
    print("Alloys per stratum and class:")
    print(counts.to_string(), "\n")

    metrics_df = stratified_metrics(dist)
    metrics_df.to_csv(os.path.join(args.outdir, "stratified_metrics.csv"), index=False)

    for label in [s[0] for s in STRATA] + ["ALL"]:
        block = metrics_df[metrics_df.stratum == label]
        n_alloys = "88" if label == "ALL" else str(int((dist.stratum == label).sum()))
        print(f"--- {label} ({n_alloys} alloys) " + "-" * 44)
        print(f"  {'prop':5s} {'n':>4s} " + "".join(f"{m:>22s}" for m, _ in METHODS))
        for _, plabel, _u in PROPERTIES:
            rows = block[block.property == plabel]
            n = int(rows.n_rows.max()) if len(rows) else 0
            cells = ""
            for method, _ in METHODS:
                r = rows[rows.method == method]
                if len(r) and r.iloc[0].mae is not None:
                    r2 = r.iloc[0].r2
                    cells += f"{r.iloc[0].mae:>13.2f}/{'  n/a' if r2 is None else f'{r2:5.3f}'}"
                else:
                    cells += f"{'--':>22s}"
            print(f"  {plabel:5s} {n:4d} {cells}")
        print()

    memo = memorisation_check(dist)
    memo.to_csv(os.path.join(args.outdir, "memorisation_check.csv"), index=False)
    print("ML-only: NEAR vs FAR (negative = better on alloys it has effectively seen)")
    print(f"  {'scope':12s} {'prop':5s} {'n_near':>7s} {'MAE_near':>9s} {'n_far':>6s} "
          f"{'MAE_far':>9s} {'near-far':>9s}")
    for _, r in memo.iterrows():
        print(f"  {r.scope:12s} {r['property']:5s} {r.n_near:7d} {r.mae_near:9.2f} "
              f"{r.n_far:6d} {r.mae_far:9.2f} {r.near_minus_far_pct:+8.1f}%")
    print()

    if not args.no_figure:
        make_histogram(dist, args.figdir)
        print(f"Figure written to {args.figdir}/nn_distance_hist.{{png,pdf}}")
    print(f"CSVs written to {args.outdir}")


if __name__ == "__main__":
    main()
