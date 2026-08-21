#!/usr/bin/env python3
"""The authoritative headline comparison across evaluation arms.

Replaces the per-arm tables under ``archive/pre_erratum/results/``, which were
computed by averaging each arm's own CSV independently. Those files do not
share a row set -- ``ml_deterministic`` holds 471 rows, the LLM baselines 466,
and the March ``full_system`` 461, missing every NIMONIC PE16 row -- so the
proposed system was scored on a strictly easier subset than the baselines it
was compared against.

This script fixes that by construction: **every metric is computed on the set
of (alloy, temperature) rows for which every included arm produced a
prediction.** The common set is reported alongside the numbers, and an arm that
would shrink it is excluded rather than silently narrowing the comparison for
everyone else.

The four current arms are complete at 471 rows each, so the common set is 471
and the correction costs nothing today. It exists so that a future arm with
gaps cannot reintroduce the defect.

The three LLM baselines are **not** included. They exist only in
``archive/pre_erratum/`` and were scored on PE16's uncorrected composition;
including them would either drag the common row set down to 456 for everyone or
compare current arms against stale ones. Pass ``--with-archived-baselines`` to
add them for inspection -- the output then carries a warning column and must
not be published as-is. See ``archive/pre_erratum/README.md``.

Outputs (written to ../results):
    headline_metrics.csv    MAE and R2 per arm x property, plus seed SD
    headline_table.md       the report

Usage:
    python headline_table.py
    python headline_table.py --with-archived-baselines
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)                       # evaluation/prediction
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
ARCHIVE_RESULTS = os.path.join(BASE_DIR, "archive", "pre_erratum", "results", "all")

DATASETS = ("sss", "precip", "sc_ds")
SEEDS = (42, 43, 44, 45, 46)

PROPS = (("YS", "pred_ys", "actual_ys", "MPa"),
         ("UTS", "pred_uts", "actual_uts", "MPa"),
         ("EL", "pred_el", "actual_el", "%"),
         ("EM", "pred_em", "actual_em", "GPa"))

#: Current arms, in the order they appear in the paper.
CURRENT_ARMS = (
    ("ML-only", ("seed42_v2_ml_only",)),
    ("ML+physics", ("seed42_v2prod_ml_deterministic",)),
    ("ML+physics+KG", ("seed42_v2prod_ml_physics_kg",)),
    ("Full system (5 seeds)", tuple(f"stageb_seed{s}_full_system" for s in SEEDS)),
)

#: Pre-erratum baselines, off by default. See module docstring.
ARCHIVED_BASELINES = (
    ("GPT-4.1 zero-shot [stale]", "gpt4.1.csv"),
    ("GPT-4.1-mini fine-tuned [stale]", "gpt4.1_ft.csv"),
    ("LLM-only [stale]", "llm_only.csv"),
)


def load_arm(tag):
    frames = []
    for ds in DATASETS:
        path = os.path.join(OUTPUT_DIR, f"{tag}_{ds}.csv")
        if os.path.exists(path):
            frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True) if frames else None


def keyset(df):
    return set(zip(df.alloy, df.temperature))


def restrict(df, keys):
    return df[[k in keys for k in zip(df.alloy, df.temperature)]]


def score(df, pred, actual):
    """MAE and R2 over rows carrying both a prediction and a measurement."""
    d = df.dropna(subset=[pred, actual])
    if len(d) < 2:
        return 0, float("nan"), float("nan")
    err = d[pred] - d[actual]
    ss_res = float((err ** 2).sum())
    ss_tot = float(((d[actual] - d[actual].mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return len(d), float(err.abs().mean()), r2


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default=RESULTS_DIR)
    ap.add_argument("--with-archived-baselines", action="store_true",
                    help="Include the pre-erratum LLM baselines. Output is then "
                         "for inspection only and must not be published.")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    arms = []
    for label, tags in CURRENT_ARMS:
        frames = [load_arm(t) for t in tags]
        frames = [f for f in frames if f is not None]
        if frames:
            arms.append((label, frames, False))

    if args.with_archived_baselines:
        for label, fname in ARCHIVED_BASELINES:
            path = os.path.join(ARCHIVE_RESULTS, fname)
            if os.path.exists(path):
                arms.append((label, [pd.read_csv(path)], True))

    common = set.intersection(*[keyset(f[0]) for _, f, _ in arms])
    total = max(len(keyset(f[0])) for _, f, _ in arms)
    print(f"Arms: {len(arms)}   common row set: {len(common)} of {total}")
    for label, frames, stale in arms:
        dropped = len(keyset(frames[0]) - common)
        flag = "  [PRE-ERRATUM]" if stale else ""
        print(f"  {label:34s} {len(keyset(frames[0])):4d} rows, {dropped:3d} outside common{flag}")

    rows = []
    for label, frames, stale in arms:
        for prop, p, a, unit in PROPS:
            ns, maes, r2s = [], [], []
            for f in frames:
                n, m, r2 = score(restrict(f, common), p, a)
                ns.append(n)
                maes.append(m)
                r2s.append(r2)
            rows.append({
                "arm": label, "property": prop, "unit": unit,
                "n_seeds": len(frames), "n_rows": ns[0],
                "mae": round(float(np.mean(maes)), 2),
                "mae_sd": round(float(np.std(maes, ddof=1)), 2) if len(maes) > 1 else None,
                "r2": round(float(np.mean(r2s)), 3),
                "pre_erratum": stale,
            })
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(args.outdir, "headline_metrics.csv"), index=False)

    piv = out.pivot_table(index="arm", columns="property", values="mae",
                          aggfunc="first", sort=False)
    print("\nMAE on the common row set:")
    print(piv.to_string())

    write_report(os.path.join(args.outdir, "headline_table.md"), out, arms,
                 len(common), total, args.with_archived_baselines)
    print(f"\nWritten to {args.outdir}")


def write_report(path, out, arms, n_common, n_total, with_stale):
    L = []
    L.append("# Headline comparison\n")
    L.append("Generated by `evaluation/prediction/scripts/headline_table.py`.")
    L.append("Seed 42 for the deterministic arms, seeds 42-46 for the agent system;")
    L.append("`saved_models_v2`, `PRODUCTION` correction profile, PE16-corrected data.\n")
    L.append("## Row set\n")
    L.append(f"All metrics are computed on the **{n_common} (alloy, temperature) rows")
    L.append("for which every arm below produced a prediction.** The superseded tables")
    L.append("in `archive/pre_erratum/results/` averaged each arm over its own rows,")
    L.append("which scored the March `full_system` on 461 rows against baselines on 466")
    L.append("and 471 -- and the ten rows it was missing were NIMONIC PE16, the single")
    L.append("worst alloy for every ML arm.\n")
    L.append("| arm | rows produced | outside common set |")
    L.append("|---|---:|---:|")
    for label, frames, stale in arms:
        ks = set(zip(frames[0].alloy, frames[0].temperature))
        L.append(f"| {label} | {len(ks)} | {len(ks) - n_common} |")
    L.append("")
    if with_stale:
        L.append("> **This run includes pre-erratum baselines.** Rows marked `[stale]`")
        L.append("> were scored on NIMONIC PE16's uncorrected composition and come from")
        L.append("> the March model generation. They are shown for inspection and must")
        L.append("> not be published. See `archive/pre_erratum/README.md`.\n")
    L.append("## MAE (lower is better)\n")
    props = [(p, u) for p, _, _, u in PROPS]
    L.append("| arm | " + " | ".join(f"{p} ({u})" for p, u in props) + " |")
    L.append("|---|" + "---:|" * len(props))
    for label, _, _ in arms:
        cells = []
        for p, _ in props:
            r = out[(out.arm == label) & (out.property == p)].iloc[0]
            cells.append(f"{r['mae']:.2f}" + (f" ± {r['mae_sd']:.2f}"
                                              if r["mae_sd"] is not None else ""))
        L.append(f"| {label} | " + " | ".join(cells) + " |")
    L.append("")
    L.append("## R²\n")
    L.append("| arm | " + " | ".join(p for p, _ in props) + " |")
    L.append("|---|" + "---:|" * len(props))
    for label, _, _ in arms:
        cells = []
        for p, _ in props:
            r = out[(out.arm == label) & (out.property == p)].iloc[0]
            cells.append(f"{r['r2']:.3f}")
        L.append(f"| {label} | " + " | ".join(cells) + " |")
    L.append("")
    L.append("## Reading\n")
    cur = out[~out.pre_erratum]
    for p, u in props:
        sub = cur[cur.property == p].sort_values("mae")
        best, second = sub.iloc[0], sub.iloc[1]
        L.append(f"- **{p}**: best is {best['arm']} at {best['mae']:.2f} {u}, "
                 f"ahead of {second['arm']} at {second['mae']:.2f}.")
    L.append("")
    ml = cur[(cur.arm == "ML-only")].set_index("property")
    mp = cur[(cur.arm == "ML+physics")].set_index("property")
    L.append("ML-only and ML+physics are identical on YS by construction -- no")
    L.append("enforcement rule touches yield strength -- and the table shows that:")
    L.append(f"{ml.loc['YS', 'mae']:.2f} against {mp.loc['YS', 'mae']:.2f}. In the")
    L.append("superseded tables the same two arms differed by 1.5 MPa, purely because")
    L.append("they were averaged over different rows. That discrepancy was the")
    L.append("most visible symptom of the row-set defect.\n")
    L.append("The three LLM baselines are absent because they have not been re-run on")
    L.append("the corrected data. Any comparison against them is currently a comparison")
    L.append("against a baseline scored on a composition we have since fixed for")
    L.append("ourselves, which is not a comparison worth publishing.")
    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
