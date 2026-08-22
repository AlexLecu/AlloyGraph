#!/usr/bin/env python3
"""Publication figures for the manuscript, built from the committed results.

Every figure is vector PDF with editable text (TrueType, not outlines), sized
for a single journal column and set in a colour-blind-safe palette. Nothing is
recomputed from raw predictions that a committed results table already holds;
where a figure needs per-row data it reads the same prediction CSVs the tables
were built from, so a figure and its table cannot disagree.

Figures
    nn_distance_hist.pdf        distance to nearest training alloy, strata marked
    parity_stratified.pdf       predicted vs measured, full system, by stratum
    accuracy_vs_distance.pdf    per-alloy YS error against distance, three arms
    coverage_by_temperature.pdf conformal coverage by temperature bin

Usage:
    python make_figures.py                    # all four, into ../../paper_assets/figures
    python make_figures.py --outdir /tmp
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
PRED_DIR = os.path.join(PROJECT_ROOT, "evaluation", "prediction")
RESULTS = os.path.join(PRED_DIR, "results")
OUTPUT = os.path.join(PRED_DIR, "output")
DEFAULT_OUT = os.path.join(PROJECT_ROOT, "paper_assets", "figures")

DATASETS = ("sss", "precip", "sc_ds")
SEEDS = (42, 43, 44, 45, 46)

#: Okabe-Ito, safe under deuteranopia, protanopia and tritanopia, and it
#: survives greyscale printing because the luminances differ.
BLUE, ORANGE, GREEN = "#0072B2", "#E69F00", "#009E73"
VERMILLION, PURPLE, GREY = "#D55E00", "#CC79A7", "#4D4D4D"

STRATUM_COLOUR = {"NEAR": BLUE, "MID": ORANGE, "FAR": GREEN}
STRATUM_MARKER = {"NEAR": "o", "MID": "s", "FAR": "^"}
STRATA = ("NEAR", "MID", "FAR")

PROPS = (("ys", "Yield strength", "MPa"),
         ("uts", "Tensile strength", "MPa"),
         ("el", "Elongation", "%"),
         ("em", "Elastic modulus", "GPa"))

#: Single journal column. Figures that need two rows of panels get more height,
#: never more width, so nothing is scaled down at typesetting.
COL_W = 3.5


def style():
    plt.rcParams.update({
        "pdf.fonttype": 42,          # embed TrueType: text stays selectable
        "ps.fonttype": 42,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8.5,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "lines.linewidth": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def load_arm(tag):
    frames = [pd.read_csv(os.path.join(OUTPUT, f"{tag}_{ds}.csv")) for ds in DATASETS]
    return pd.concat(frames, ignore_index=True)


def full_system_mean():
    """Row-wise mean prediction across the five campaign seeds.

    The agent system is stochastic, so a parity plot of one seed would show
    that seed's noise as if it were the method's behaviour. Averaging per
    (alloy, temperature) plots the quantity the tables report.
    """
    frames = [load_arm(f"stageb_seed{s}_full_system") for s in SEEDS]
    keys = ["alloy", "temperature"]
    base = frames[0][keys + [f"actual_{p}" for p, _, _ in PROPS]].copy()
    for p, _, _ in PROPS:
        stacked = pd.concat([f.set_index(keys)[f"pred_{p}"] for f in frames], axis=1)
        base[f"pred_{p}"] = stacked.mean(axis=1).reindex(
            pd.MultiIndex.from_frame(base[keys])).values
    return base


def with_stratum(df, dist):
    m = dist.set_index("alloy")
    out = df.copy()
    out["stratum"] = out["alloy"].map(m["stratum"])
    out["distance"] = out["alloy"].map(m["distance"])
    return out


# --------------------------------------------------------------------------
def fig_distance_hist(dist, path):
    """Distance to the nearest training alloy, with the stratum cuts marked."""
    fig, ax = plt.subplots(figsize=(COL_W, 2.3))
    bins = np.arange(0, np.ceil(dist.distance.max()) + 0.5, 0.5)
    for s in STRATA:
        sub = dist[dist.stratum == s]
        ax.hist(sub.distance, bins=bins, color=STRATUM_COLOUR[s],
                edgecolor="white", linewidth=0.3, label=f"{s} (n={len(sub)})")
    for cut in (2.0, 4.5):
        ax.axvline(cut, color=GREY, linestyle="--", linewidth=0.8, zorder=5)
        ax.text(cut, ax.get_ylim()[1] * 0.97, f" d={cut}", color=GREY,
                fontsize=6.5, va="top", ha="left")
    ax.set_xlabel("Euclidean distance to nearest training alloy (wt%)")
    ax.set_ylabel("Evaluation alloys")
    ax.legend(frameon=False, loc="upper right", handlelength=1.1)
    fig.savefig(path)
    plt.close(fig)


def fig_parity(dist, path):
    """Predicted vs measured for the full system, coloured by stratum.

    Points are the row-wise mean over the five seeds. The MAE annotated on each
    panel is read from stratified_metrics.csv rather than recomputed here, so
    the figure cannot disagree with the tables. The two differ slightly by
    construction: averaging predictions first cancels independent seed noise,
    so the plotted points give a marginally lower error (78.6 against 80.9 MPa
    on yield strength) than the mean of the per-seed errors the tables report.
    The tables' definition is the one quoted.
    """
    strat = pd.read_csv(os.path.join(RESULTS, "stratified_metrics.csv"))
    table_mae = {r["property"]: r["mae"] for _, r in
                 strat[(strat.method == "Full system (5 seeds)")
                       & (strat.stratum == "ALL")].iterrows()}
    df = with_stratum(full_system_mean(), dist)
    fig, axes = plt.subplots(2, 2, figsize=(COL_W, COL_W * 1.02))
    for ax, (key, label, unit) in zip(axes.ravel(), PROPS):
        d = df.dropna(subset=[f"pred_{key}", f"actual_{key}"])
        # Robust limits. One measured yield strength in the evaluation set is
        # -435 MPa, which is not physical; letting it set the axis wastes half
        # the panel and makes the model look worse than it is. Points outside
        # the drawn range are counted in the panel rather than hidden.
        both = np.concatenate([d[f"actual_{key}"].values, d[f"pred_{key}"].values])
        lo, hi = np.percentile(both, [0.5, 99.5])
        pad = (hi - lo) * 0.06
        lo, hi = max(0.0, lo - pad), hi + pad
        ax.plot([lo, hi], [lo, hi], color=GREY, linewidth=0.7, zorder=1)
        for s in STRATA:
            ss = d[d.stratum == s]
            ax.scatter(ss[f"actual_{key}"], ss[f"pred_{key}"], s=6,
                       facecolor=STRATUM_COLOUR[s], edgecolor="none",
                       alpha=0.75, marker=STRATUM_MARKER[s], zorder=2, label=s)
        mae = table_mae.get(key.upper())
        outside = int(((d[f"actual_{key}"] < lo) | (d[f"actual_{key}"] > hi)
                       | (d[f"pred_{key}"] < lo) | (d[f"pred_{key}"] > hi)).sum())
        note = f"{label}\nMAE {mae:.1f} {unit}"
        if outside:
            note += f"\n({outside} outside axes)"
        ax.text(0.04, 0.95, note, transform=ax.transAxes,
                fontsize=6.8, va="top", ha="left")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        ax.tick_params(length=2)
    for ax in axes[-1]:
        ax.set_xlabel("Measured")
    for ax in axes[:, 0]:
        ax.set_ylabel("Predicted")
    handles = [Line2D([], [], marker=STRATUM_MARKER[s], color="none",
                      markerfacecolor=STRATUM_COLOUR[s], markersize=4, label=s)
               for s in STRATA]
    fig.legend(handles=handles, frameon=False, ncol=3, loc="lower center",
               bbox_to_anchor=(0.5, -0.035), handletextpad=0.3, columnspacing=1.2)
    fig.tight_layout(pad=0.4)
    fig.savefig(path)
    plt.close(fig)


def fig_accuracy_vs_distance(dist, path):
    """Per-alloy yield-strength error against distance, for three arms.

    The claim this figure carries: knowledge-graph anchoring separates from
    ML-only only among near-duplicates, while the agent layer sits below both
    across the whole distance range.

    ML-only and ML+physics+KG are identical past d = 4.5 -- anchoring cannot
    fire beyond the gate, and no physics rule touches yield strength -- so their
    trend lines coincide exactly over most of the axis. Drawn naively the upper
    line simply hides the lower one and the reader sees two arms where there are
    three. ML-only is therefore drawn thick and solid underneath, with the KG
    arm dashed on top, so coincidence reads as coincidence.
    """
    arms = [("ML-only", "seed42_v2_ml_only", BLUE, "o", "-", 2.2, 0.9),
            ("ML+physics+KG", "seed42_v2prod_ml_physics_kg", ORANGE, "s", (0, (3, 2)), 1.3, 1.0),
            ("Full system", None, VERMILLION, "^", "-", 1.5, 1.0)]
    fig, ax = plt.subplots(figsize=(COL_W, 2.7))

    # Stratum bands, drawn behind everything and labelled along the bottom.
    xmax = float(dist.distance.max()) * 1.02
    for lo, hi, name in ((0, 2.0, "NEAR"), (2.0, 4.5, "MID"), (4.5, xmax, "FAR")):
        ax.axvspan(lo, hi, color=STRATUM_COLOUR[name], alpha=0.055, lw=0, zorder=0)
        ax.text((lo + min(hi, xmax)) / 2, 0.015, name, transform=ax.get_xaxis_transform(),
                fontsize=6.3, color=GREY, ha="center", va="bottom")

    ymax = 0
    for label, tag, colour, marker, dash, lw, alpha in arms:
        df = full_system_mean() if tag is None else load_arm(tag)
        d = df.dropna(subset=["pred_ys", "actual_ys"]).copy()
        d["abs_err"] = (d.pred_ys - d.actual_ys).abs()
        per_alloy = (d.groupby("alloy")["abs_err"].mean().rename("mae").reset_index()
                     .merge(dist[["alloy", "distance"]], on="alloy")
                     .sort_values("distance"))
        ax.scatter(per_alloy.distance, per_alloy.mae, s=5, marker=marker,
                   facecolor=colour, edgecolor="none", alpha=0.3, zorder=2)
        w = max(7, len(per_alloy) // 5)
        trend = per_alloy.mae.rolling(w, center=True, min_periods=3).median()
        ax.plot(per_alloy.distance, trend, color=colour, linewidth=lw,
                linestyle=dash, alpha=alpha, label=label, zorder=3,
                solid_capstyle="round")
        ymax = max(ymax, float(np.nanpercentile(per_alloy.mae, 95)))

    ax.set_xlabel("Distance to nearest training alloy (wt%)")
    ax.set_ylabel("Per-alloy yield-strength MAE (MPa)")
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax * 1.25)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.16),
              ncol=3, handlelength=1.8, columnspacing=1.0, handletextpad=0.4)
    fig.savefig(path)
    plt.close(fig)


def fig_coverage_by_temperature(path):
    """Empirical conformal coverage per property, binned by temperature."""
    iv = pd.read_csv(os.path.join(RESULTS, "conformal_intervals.csv"))
    iv = iv.dropna(subset=["actual", "covered"])
    # The sub-zero bin holds two cryogenic rows for two properties; a bin that
    # thin cannot carry a coverage estimate, so it is excluded rather than
    # drawn as an empty column. Stated in the manifest.
    edges = [0, 400, 700, 900, 1300]
    labels = ["0–400", "400–700", "700–900", ">900"]
    iv["bin"] = pd.cut(iv.temperature, bins=edges, labels=labels, right=False)
    fig, ax = plt.subplots(figsize=(COL_W, 2.4))
    # conformal_intervals.csv keys properties in lower case (ys, uts, el, em).
    colours = {"ys": BLUE, "uts": ORANGE, "el": GREEN, "em": PURPLE}
    markers = {"ys": "o", "uts": "s", "el": "^", "em": "D"}
    display = {"ys": "YS", "uts": "UTS", "el": "EL", "em": "EM"}
    x = np.arange(len(labels))
    for prop in ("ys", "uts", "el", "em"):
        sub = iv[iv.property.str.lower() == prop]
        g = sub.groupby("bin", observed=False)["covered"]
        cov, n = g.mean() * 100, g.size()
        vals = [cov.get(l, np.nan) if n.get(l, 0) >= 5 else np.nan for l in labels]
        ax.plot(x, vals, marker=markers[prop], markersize=3.4, color=colours[prop],
                label=display[prop], linewidth=1.1)
    ax.axhline(90, color=GREY, linestyle="--", linewidth=0.8)
    ax.text(len(labels) - 0.5, 90.8, "nominal 90%", fontsize=6.5, color=GREY, ha="right")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_xlabel("Test temperature (°C)")
    ax.set_ylabel("Empirical coverage (%)")
    ax.set_ylim(35, 105)
    ax.legend(frameon=False, ncol=4, loc="lower left", handlelength=1.2,
              columnspacing=0.9, handletextpad=0.3)
    fig.savefig(path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default=DEFAULT_OUT)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    style()

    dist = pd.read_csv(os.path.join(RESULTS, "nn_distance.csv"))
    jobs = [
        ("nn_distance_hist.pdf", lambda p: fig_distance_hist(dist, p)),
        ("parity_stratified.pdf", lambda p: fig_parity(dist, p)),
        ("accuracy_vs_distance.pdf", lambda p: fig_accuracy_vs_distance(dist, p)),
        ("coverage_by_temperature.pdf", lambda p: fig_coverage_by_temperature(p)),
    ]
    for name, fn in jobs:
        path = os.path.join(args.outdir, name)
        fn(path)
        print(f"  {name:30s} {os.path.getsize(path) / 1024:6.1f} kB")
    print(f"\nWritten to {args.outdir}")


if __name__ == "__main__":
    main()
