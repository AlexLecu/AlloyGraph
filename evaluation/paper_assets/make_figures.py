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
    mcq_accuracy.pdf            chatbot MCQ accuracy by question type

Usage:
    python make_figures.py                    # all four, into ../../paper_assets/figures
    python make_figures.py --outdir /tmp
"""

import argparse
import json
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
CHATBOT_RESULTS = os.path.join(PROJECT_ROOT, "evaluation", "chatbot", "results")
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

#: Full text width, for the one figure that cannot fit a column. Twelve grouped
#: bars with a value label on each need the width; squeezed to COL_W the labels
#: collide. The manuscript already sets this figure across both columns.
FULL_W = 7.16


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
        # Tolerance bands around the diagonal, nested so the +/-10% region
        # carries both fills and reads darker than the +/-20% one. zorder 0
        # keeps them under the diagonal and the points; no edge, and an alpha
        # low enough that the stratum colours stay dominant.
        edge = np.array([lo, hi])
        for frac in (0.20, 0.10):
            ax.fill_between(edge, edge * (1 - frac), edge * (1 + frac),
                            color=GREY, alpha=0.07, linewidth=0, zorder=0)
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


#: Distance bands for the KG step-function figure, finer than NEAR/MID/FAR so
#: the shape of the effect is visible rather than averaged into three numbers.
BANDS = ((0.0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0),
         (2.0, 3.0), (3.0, 4.5), (4.5, float("inf")))
BAND_LABEL = ("0–0.5", "0.5–1", "1–1.5", "1.5–2", "2–3", "3–4.5", "≥4.5")

#: Recomputed band values, declared so a silent change is caught. Order matches
#: BANDS. (n rows, ML+physics MAE, +KG MAE, full-system 5-seed mean MAE).
#: These do NOT satisfy the "every band below 2.0 gains >=25%, none above gains
#: >0.2%" reading: the 0-0.5 band gets 5.7% WORSE with anchoring, and 3.0-4.5
#: gains 0.24%. See the manifest.
BAND_EXPECTED = ((12, 36.71, 38.80, 72.77), (24, 137.70, 96.22, 59.78),
                 (12, 63.95, 33.25, 35.17), (8, 188.83, 131.25, 100.81),
                 (25, 90.51, 90.51, 92.53), (66, 70.57, 70.41, 63.96),
                 (138, 110.10, 110.10, 89.83))


def band_stats(dist):
    """Per-band yield-strength MAE for the three arms the figure compares."""
    at = dist.set_index("alloy")["distance"]

    def mae(df, alloys):
        d = df[df.alloy.isin(alloys)].dropna(subset=["pred_ys", "actual_ys"])
        d = d[d.actual_ys != 0]
        return len(d), float((d.pred_ys - d.actual_ys).abs().mean())

    phys, kg = load_arm("seed42_v2prod_ml_deterministic"), load_arm("seed42_v2prod_ml_physics_kg")
    seeds = [load_arm(f"stageb_seed{s}_full_system") for s in SEEDS]

    out = []
    for lo, hi in BANDS:
        alloys = at[(at >= lo) & (at < hi)].index
        n, m_phys = mae(phys, alloys)
        _, m_kg = mae(kg, alloys)
        per_seed = [mae(f, alloys)[1] for f in seeds]
        out.append({"n": n, "alloys": len(alloys), "phys": m_phys, "kg": m_kg,
                    "full": float(np.mean(per_seed)), "full_sd": float(np.std(per_seed)),
                    "gain": 100.0 * (m_phys - m_kg) / m_phys})

    for i, (n, ph, k, fu) in enumerate(BAND_EXPECTED):
        got = (out[i]["n"], round(out[i]["phys"], 2), round(out[i]["kg"], 2),
               round(out[i]["full"], 2))
        if got != (n, ph, k, fu):
            raise SystemExit(f"band {BAND_LABEL[i]} is {got}, declared {(n, ph, k, fu)}. "
                             f"The results moved; update BAND_EXPECTED deliberately.")
    return out


def fig_accuracy_vs_distance(dist, path):
    """Where knowledge-graph anchoring pays, and where the agent layer does.

    (a) Relative yield-strength error reduction from adding KG anchoring to
    ML+physics, per distance band. Positive is an improvement.

    (b) The same bands carrying absolute MAE for three arms, which is what
    separates the two mechanisms: anchoring is confined to short distances,
    the agent layer is not.

    Read (a) with the row counts in mind. They are printed on the panel because
    four of the seven bands rest on 12 rows or fewer, and because the effect is
    carried by six alloys in total -- anchoring changes no prediction at all for
    the other 80. A band is not a sample of a population here.
    """
    st = band_stats(dist)
    x = np.arange(len(BANDS))
    fig, (axa, axb) = plt.subplots(2, 1, figsize=(COL_W, 4.5))

    # (a) relative gain -------------------------------------------------
    gains = [b["gain"] for b in st]
    colours = [GREEN if g > 0 else VERMILLION for g in gains]
    axa.bar(x, gains, 0.68, color=colours, edgecolor="none", zorder=3)
    axa.axhline(0, color=GREY, linewidth=0.6, zorder=4)
    for xi, (g, b) in enumerate(zip(gains, st)):
        off = 1.6 if g >= 0 else -1.6
        axa.text(xi, g + off, f"{g:+.1f}", ha="center", fontsize=6,
                 va="bottom" if g >= 0 else "top", color=GREY)
        axa.text(xi, -49, f"n={b['n']}", ha="center", va="bottom",
                 fontsize=5.6, color=GREY)
    # d = 2.0, the NEAR/MID cut, falls between the fourth and fifth band.
    axa.axvline(3.5, color=GREY, linestyle=(0, (2, 2)), linewidth=0.7, zorder=2)
    axa.text(3.42, 44, "d = 2.0", fontsize=6, color=GREY, ha="right", va="top")
    axa.set_ylim(-52, 56)
    axa.set_ylabel("YS error reduction\nfrom KG anchoring (%)")
    axa.set_title("(a)", loc="left", fontweight="bold")

    # (b) absolute MAE ---------------------------------------------------
    series = (("ML+physics", "phys", BLUE, "o", "-"),
              ("+ KG", "kg", ORANGE, "s", (0, (3, 2))),
              ("Full system", "full", VERMILLION, "^", "-"))
    for label, key, colour, marker, dash in series:
        axb.plot(x, [b[key] for b in st], color=colour, marker=marker,
                 markersize=3.2, linewidth=1.2, linestyle=dash, label=label, zorder=3)
    lo = np.array([b["full"] - b["full_sd"] for b in st])
    hi = np.array([b["full"] + b["full_sd"] for b in st])
    axb.fill_between(x, lo, hi, color=VERMILLION, alpha=0.16, lw=0, zorder=2)
    axb.axvline(3.5, color=GREY, linestyle=(0, (2, 2)), linewidth=0.7, zorder=1)
    axb.set_ylabel("Yield-strength MAE (MPa)")
    axb.set_ylim(0, 205)
    # Inside the axes: the upper-left corner is empty (all three arms are at
    # their lowest in the nearest band) and a legend above the panel crowds
    # panel (a)'s tick labels.
    axb.legend(frameon=False, loc="upper left", handlelength=1.8,
               labelspacing=0.25, handletextpad=0.4, borderaxespad=0.2)
    axb.set_title("(b)", loc="left", fontweight="bold")

    for ax in (axa, axb):
        ax.set_xticks(x)
        ax.set_xticklabels(BAND_LABEL, fontsize=6.2)
        ax.set_xlim(-0.6, len(BANDS) - 0.4)
        ax.tick_params(length=2)
    axb.set_xlabel("Distance to nearest training alloy (wt%)")
    fig.tight_layout(pad=0.4)
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


def fig_mcq_accuracy(path):
    """Multiple-choice accuracy by question type, three systems.

    Read from the committed mcq_report.json rather than restated. The notebook
    that first drew this figure (evaluation/chatbot/notebooks/mcq_analysis.ipynb,
    cell 7) hard-codes its twelve percentages, so the figure could drift from
    the run behind it with nothing to catch the drift. The values here are
    recomputed from the report and asserted against the published figure below.

    "Overall" is an aggregate of the other three bars, not a fourth question
    type, so it is drawn in grey with a hatch instead of being given a fourth
    hue of equal weight.
    """
    report = json.load(open(os.path.join(CHATBOT_RESULTS, "mcq_report.json")))

    #: report key -> label in the paper. Labels match the published figure.
    systems = (("chatbot", "Chatbot + KG"),
               ("llama", "Llama 3.3 70B"),
               ("gpt", "GPT-4o"))
    hops = (("1hop", "1-Hop"), ("2hop", "2-Hop"),
            ("general", "General"), ("overall", "Overall"))

    pct = {}
    for hop_key, hop_label in hops:
        pct[hop_label] = [100.0 * report["systems"][s][hop_key]["correct"]
                          / report["systems"][s][hop_key]["total"]
                          for s, _ in systems]

    # The figure in the manuscript. Regenerating must not silently change it.
    published = {"1-Hop": [100, 35, 33], "2-Hop": [79, 43, 42],
                 "General": [98, 92, 100], "Overall": [91, 50, 50]}
    for label, expected in published.items():
        got = [round(v) for v in pct[label]]
        if got != expected:
            raise SystemExit(f"mcq_accuracy: {label} is {got}, published {expected}. "
                             f"The results moved; update `published` deliberately.")

    # Light grey, not GREY: at full strength the aggregate bar out-weighs the
    # three measured ones it summarises, which is backwards.
    AGGREGATE = "#B0B0B0"
    colour = {"1-Hop": BLUE, "2-Hop": ORANGE, "General": GREEN, "Overall": AGGREGATE}
    hatch = {"1-Hop": "", "2-Hop": "", "General": "", "Overall": "//"}

    fig, ax = plt.subplots(figsize=(FULL_W, 2.9))
    bar_w = 0.17
    group_w = len(hops) * bar_w
    left = np.arange(len(systems)) * (group_w + 0.4)

    for i, (_, label) in enumerate(hops):
        bars = ax.bar(left + i * bar_w, pct[label], bar_w, label=label,
                      color=colour[label], hatch=hatch[label],
                      edgecolor="white", linewidth=0.6, zorder=3)
        for bar, v in zip(bars, pct[label]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.2,
                    f"{v:.0f}%", ha="center", va="bottom", fontsize=6.5, color=GREY)

    for y in range(20, 101, 20):
        ax.axhline(y, color="#E8E8E8", linewidth=0.5, zorder=1)
    ax.set_ylim(0, 108)
    ax.set_yticks(range(0, 101, 20))
    ax.set_yticklabels([f"{y}%" for y in range(0, 101, 20)])
    ax.set_ylabel("Accuracy")
    ax.set_xticks(left + group_w / 2 - bar_w / 2)
    ax.set_xticklabels([label for _, label in systems])
    ax.tick_params(axis="x", length=0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.14), ncol=len(hops),
              frameon=False, columnspacing=1.5, handlelength=1.2,
              handletextpad=0.4)
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
        ("mcq_accuracy.pdf", lambda p: fig_mcq_accuracy(p)),
    ]
    for name, fn in jobs:
        path = os.path.join(args.outdir, name)
        fn(path)
        print(f"  {name:30s} {os.path.getsize(path) / 1024:6.1f} kB")
    print(f"\nWritten to {args.outdir}")


if __name__ == "__main__":
    main()
