#!/usr/bin/env python3
"""Robustness of the stratified claims (Expert Systems revision R1.3, R1.4, R2.2).

Three questions the reviewers asked of results already in the manuscript. No
model is run and no prediction changes: every number here is re-derived from
the committed prediction CSVs.

1. How fragile is the NEAR knowledge-graph result? (R1.4)
   The 30% NEAR yield-strength gain rests on few anchoring events. This reports
   an alloy-level cluster bootstrap interval for it, the leave-one-alloy-out
   range, and the per-alloy accounting of which rows anchoring actually moved.
   The same bootstrap is applied to the agent-layer FAR gain for comparison.

2. Is the MID non-monotonicity a property of the anchoring gate or of the
   stratum definition? (R1.3)
   Four checks: (a) sweep both stratum boundaries and watch the MID cells move;
   (b) compare the alloy-class mix of the strata, since MID is not drawn from
   the same population as FAR; (c) attribute each MID cell to the alloys that
   carry it; (d) count how often anchoring fires in MID at all.
   The 100% MID elastic-modulus conformal coverage is read against the error
   the intervals are asked to cover.

3. How much does the stratification depend on the distance metric? (R2.2)
   The production metric is Euclidean on wt%-normalised composition. Three
   alternatives are scored -- atomic percent, per-element z-scored wt%, and a
   standardised space of the engineered metallurgical features -- and each is
   cut into strata of the same sizes (13/31/44) so membership can be compared
   directly. The two headline effects are recomputed under each.

Strata and row sets are the ones the manuscript uses: the strata come from
nn_distance.csv, and every metric is computed on the (alloy, temperature) rows
shared by all compared arms. The full system enters as the row-wise mean of the
absolute error over its five campaigns, whose mean over rows equals the mean of
the per-campaign MAEs reported in the paper.

Outputs (../results):
    revision_robustness.md     the report
    revision_robustness.json   every number in it, for the manuscript checker

Usage:
    python revision_robustness.py
"""

import json
import os
import sys

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)                       # evaluation/prediction
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))    # repo root
sys.path.insert(0, os.path.join(PROJECT_ROOT, "backend"))

from alloy_crew.models.feature_engineering import (  # noqa: E402
    compute_alloy_features, wt_to_at_percent,
)
from alloy_crew.tools.rag_tools import _composition_distance  # noqa: E402

OUTPUT_DIR = os.path.join(BASE_DIR, "output")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
DATA_DIR = os.path.join(BASE_DIR, "data")
TRAINING_JSONL = os.path.join(
    PROJECT_ROOT, "backend", "alloy_crew", "models", "training_data", "train_77alloys.jsonl"
)

DATASETS = {"sss": "SSS", "precip": "precip", "sc_ds": "sc_ds"}
SEEDS = (42, 43, 44, 45, 46)
STRATA = ("NEAR", "MID", "FAR")
PROPS = (("ys", "YS", "MPa"), ("uts", "UTS", "MPa"), ("el", "EL", "%"), ("em", "EM", "GPa"))

ARMS = {
    "ml": "seed42_v2_ml_only_{ds}.csv",
    "phys": "seed42_v2prod_ml_deterministic_{ds}.csv",
    "kg": "seed42_v2prod_ml_physics_kg_{ds}.csv",
}
FULL = "stageb_seed{seed}_full_system_{ds}.csv"

N_BOOT = 10_000
RNG_SEED = 0

#: Engineered features used for the feature-space distance. Chosen as the
#: physically interpretable descriptors that define the alloy classes and the
#: physics models, not as the full 50-80 column ML input.
FEATURE_KEYS = ("gamma_prime_estimated_vol_pct", "Md_avg", "Md_gamma", "lattice_mismatch_pct",
                "density_calculated_gcm3", "SSS_total_wt_pct", "refractory_total_wt_pct",
                "GP_formers_wt_pct", "VEC_avg")


# --------------------------------------------------------------------------- data

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_arm(pattern, **fmt):
    return pd.concat([pd.read_csv(os.path.join(OUTPUT_DIR, pattern.format(ds=ds, **fmt)))
                      for ds in DATASETS], ignore_index=True)


def key(df):
    return list(zip(df.alloy, df.temperature.round(1)))


def build_frame():
    """One row per (alloy, temperature) with every arm's prediction side by side."""
    base = load_arm(ARMS["ml"])
    base["k"] = key(base)
    frame = base[["k", "alloy", "temperature"] + [f"actual_{p}" for p, _, _ in PROPS]].copy()
    for p, _, _ in PROPS:
        frame[f"ml_{p}"] = base[f"pred_{p}"].to_numpy()
    for arm in ("phys", "kg"):
        df = load_arm(ARMS[arm])
        df["k"] = key(df)
        df = df.set_index("k")
        for p, _, _ in PROPS:
            frame[f"{arm}_{p}"] = frame.k.map(df[f"pred_{p}"])
    for s in SEEDS:
        df = load_arm(FULL, seed=s)
        df["k"] = key(df)
        df = df.set_index("k")
        for p, _, _ in PROPS:
            frame[f"full{s}_{p}"] = frame.k.map(df[f"pred_{p}"])

    dist = pd.read_csv(os.path.join(RESULTS_DIR, "nn_distance.csv")).set_index("alloy")
    frame["distance"] = frame.alloy.map(dist["distance"])
    frame["stratum"] = frame.alloy.map(dist["stratum"])
    frame["alloy_class"] = frame.alloy.map(dist["alloy_class"])
    if frame.stratum.isna().any():
        raise RuntimeError("rows without a distance entry: "
                           f"{sorted(frame.loc[frame.stratum.isna(), 'alloy'].unique())}")
    return frame


def abs_errors(frame, prop, arm):
    """Per-row absolute error of an arm; NaN where the row is not scorable.

    'full' is the mean over the five campaigns of each campaign's absolute error,
    so averaging it over rows reproduces the mean of per-campaign MAEs.
    """
    a = frame[f"actual_{prop}"].to_numpy(float)
    valid = ~np.isnan(a) & (a != 0)
    if arm == "full":
        errs = np.column_stack([np.abs(frame[f"full{s}_{prop}"].to_numpy(float) - a)
                                for s in SEEDS])
        e = errs.mean(axis=1)
        valid &= ~np.isnan(errs).any(axis=1)
    else:
        e = np.abs(frame[f"{arm}_{prop}"].to_numpy(float) - a)
        valid &= ~np.isnan(e)
    return np.where(valid, e, np.nan)


def scorable(frame, prop, arms):
    """Rows scorable for every listed arm: the common row set of the manuscript."""
    m = np.ones(len(frame), bool)
    for arm in arms:
        m &= ~np.isnan(abs_errors(frame, prop, arm))
    return m


# --------------------------------------------------------------- 1. bootstrap

def relative_change(e_new, e_ref):
    return 100.0 * (np.nanmean(e_new) - np.nanmean(e_ref)) / np.nanmean(e_ref)


def cluster_bootstrap(frame, prop, new, ref, mask, rng):
    """Alloy-level bootstrap of the relative MAE change new vs ref.

    Rows from one alloy share a composition and an ML prediction, so they are
    resampled together: the unit of independence is the alloy, not the row.
    """
    sub = frame[mask]
    e_new, e_ref = abs_errors(sub, prop, new), abs_errors(sub, prop, ref)
    alloys = sub.alloy.to_numpy()
    uniq = np.unique(alloys)
    idx_by_alloy = {a: np.flatnonzero(alloys == a) for a in uniq}
    sums_new = np.array([e_new[idx_by_alloy[a]].sum() for a in uniq])
    sums_ref = np.array([e_ref[idx_by_alloy[a]].sum() for a in uniq])
    draws = rng.integers(0, len(uniq), size=(N_BOOT, len(uniq)))
    boot = 100.0 * (sums_new[draws].sum(1) - sums_ref[draws].sum(1)) / sums_ref[draws].sum(1)
    point = relative_change(e_new, e_ref)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    loao = []
    for a in uniq:
        keep = alloys != a
        loao.append((a, relative_change(e_new[keep], e_ref[keep])))
    return {
        "n_rows": int(len(sub)), "n_alloys": int(len(uniq)),
        "mae_new": float(np.nanmean(e_new)), "mae_ref": float(np.nanmean(e_ref)),
        "rel_change_pct": float(point), "ci95": [float(lo), float(hi)],
        "p_boot_ge0": float(np.mean(boot >= 0)),
        "loao_min": float(min(v for _, v in loao)), "loao_max": float(max(v for _, v in loao)),
        "loao_most_influential": max(loao, key=lambda t: abs(t[1] - point))[0],
        "loao_most_influential_value": float(max(loao, key=lambda t: abs(t[1] - point))[1]),
    }


def anchoring_events(frame):
    """Which yield-strength rows anchoring moved, and by how much, per alloy."""
    m = scorable(frame, "ys", ("phys", "kg"))
    sub = frame[m].copy()
    sub["moved"] = (sub.kg_ys - sub.phys_ys).abs() > 1e-6
    sub["d_err"] = abs_errors(sub, "ys", "kg") - abs_errors(sub, "ys", "phys")
    moved = sub[sub.moved]
    per_alloy = (moved.groupby(["alloy", "stratum", "distance"])
                 .agg(rows=("moved", "size"), err_change_mpa=("d_err", "sum"))
                 .reset_index().sort_values("distance"))
    near = sub[sub.stratum == "NEAR"]
    near_moved = near[near.moved]
    total_gain = -near.d_err.sum()
    return {
        "rows_scored": int(len(sub)), "rows_moved": int(len(moved)),
        "alloys_moved": int(moved.alloy.nunique()),
        "near_rows": int(len(near)), "near_rows_moved": int(len(near_moved)),
        "near_alloys_moved": int(near_moved.alloy.nunique()),
        "near_gain_share_top_alloy": float(
            (-near_moved.groupby("alloy").d_err.sum()).max() / total_gain) if total_gain else None,
        "per_alloy": per_alloy.to_dict("records"),
    }


# --------------------------------------------------------------- 2. MID analysis

def boundary_sweep(frame):
    """MID cells as both stratum boundaries move.

    A genuine property of the anchoring gate would sit at the 4.5 cutoff and
    move with it; an artefact of where the boundaries happen to fall would make
    the cells swing as alloys enter and leave the stratum.
    """
    rows = []
    for lo in (1.5, 1.75, 2.0, 2.25, 2.5):
        for hi in (4.0, 4.25, 4.5, 4.75, 5.0):
            band = (frame.distance >= lo) & (frame.distance < hi)
            rec = {"lower": lo, "upper": hi, "n_alloys": int(frame[band].alloy.nunique())}
            m = band & scorable(frame, "ys", ("kg", "full"))
            rec["ys_full_vs_kg_pct"] = relative_change(abs_errors(frame[m], "ys", "full"),
                                                       abs_errors(frame[m], "ys", "kg"))
            m = band & scorable(frame, "em", ("ml", "phys", "full"))
            rec["em_ml"] = float(np.nanmean(abs_errors(frame[m], "em", "ml")))
            rec["em_phys"] = float(np.nanmean(abs_errors(frame[m], "em", "phys")))
            rec["em_full"] = float(np.nanmean(abs_errors(frame[m], "em", "full")))
            rec["em_full_vs_ml_pct"] = relative_change(abs_errors(frame[m], "em", "full"),
                                                       abs_errors(frame[m], "em", "ml"))
            rows.append(rec)
    return rows


def class_mix(frame):
    alloys = frame.drop_duplicates("alloy")
    tab = pd.crosstab(alloys.stratum, alloys.alloy_class).reindex(list(STRATA))
    return {s: {c: int(v) for c, v in tab.loc[s].items()} for s in STRATA}


def within_class(frame):
    """The agent YS effect and the EM cells per stratum x class.

    If the MID reversal follows alloy class rather than distance, the per-class
    cells should look alike across strata and only the mix should differ.
    """
    out = []
    for s in STRATA:
        for c in ("Precip", "SSS", "SC/DS"):
            sel = (frame.stratum == s) & (frame.alloy_class == c)
            rec = {"stratum": s, "class": c, "n_alloys": int(frame[sel].alloy.nunique())}
            m = sel & scorable(frame, "ys", ("kg", "full"))
            rec["ys_n"] = int(m.sum())
            if m.sum():
                rec["ys_kg"] = float(np.nanmean(abs_errors(frame[m], "ys", "kg")))
                rec["ys_full"] = float(np.nanmean(abs_errors(frame[m], "ys", "full")))
            m = sel & scorable(frame, "em", ("ml", "phys", "full"))
            rec["em_n"] = int(m.sum())
            if m.sum():
                for arm in ("ml", "phys", "full"):
                    rec[f"em_{arm}"] = float(np.nanmean(abs_errors(frame[m], "em", arm)))
            out.append(rec)
    return out


def per_alloy_contribution(frame, stratum, prop, new, ref, arms, top=6):
    """Each alloy's share of a stratum-level MAE difference new - ref."""
    m = (frame.stratum == stratum) & scorable(frame, prop, arms)
    sub = frame[m].copy()
    sub["d"] = abs_errors(sub, prop, new) - abs_errors(sub, prop, ref)
    total = sub.d.sum()
    by = sub.groupby("alloy").agg(rows=("d", "size"), sum_d=("d", "sum")).reset_index()
    by["share_of_net"] = by.sum_d / total if total else np.nan
    by = by.reindex(by.sum_d.abs().sort_values(ascending=False).index)
    keep_mask_top3 = ~sub.alloy.isin(by.alloy.head(3))
    without_top3 = sub[keep_mask_top3]
    return {
        "stratum": stratum, "property": prop, "new": new, "ref": ref,
        "n_rows": int(len(sub)), "n_alloys": int(sub.alloy.nunique()),
        "net_mae_diff": float(total / len(sub)),
        "net_mae_diff_without_top3": float(without_top3.d.mean()) if len(without_top3) else None,
        "top": by.head(top).to_dict("records"),
    }


def conformal_vs_error(frame):
    """Interval half-width against the error it must cover, per stratum.

    Coverage is the fraction of residuals inside the half-width, so a stratum
    whose 90th-percentile residual sits far below the half-width reaches 100%
    trivially. That is uninformative everywhere the width is constant, not a
    MID-specific property.
    """
    conf = pd.read_csv(os.path.join(RESULTS_DIR, "conformal_intervals.csv"))
    out = []
    for prop in ("ys", "em"):
        c = conf[conf.property == prop].dropna(subset=["actual"])
        for s in STRATA:
            cs = c[c.stratum == s]
            resid = (cs.actual - cs.pred_cv_plus).abs()
            out.append({
                "property": prop, "stratum": s, "n": int(len(cs)),
                "median_half_width": float(cs.width.median() / 2),
                "resid_median": float(resid.median()),
                "resid_p90": float(resid.quantile(0.9)),
                "resid_max": float(resid.max()),
                "coverage": float(cs.covered.mean()),
            })
    return out


# --------------------------------------------------------------- 3. distance metrics

def _features(comp):
    feats = compute_alloy_features({"composition": comp})
    return [float(feats.get(k, np.nan)) if feats.get(k) is not None else np.nan
            for k in FEATURE_KEYS]


def distance_metrics(frame):
    training = load_jsonl(TRAINING_JSONL)
    evaluation = []
    for stem in DATASETS.values():
        for rec in load_jsonl(os.path.join(DATA_DIR, f"{stem}.jsonl")):
            if rec.get("composition"):
                evaluation.append(rec)
    eval_names = set(frame.alloy)
    evaluation = [r for r in evaluation if r["alloy"] in eval_names]

    elements = sorted({e for r in training + evaluation for e in r["composition"]})

    def vec_wt(comp):
        tot = sum(comp.values())
        return np.array([100.0 * comp.get(e, 0.0) / tot for e in elements])

    def vec_at(comp):
        at = wt_to_at_percent(comp)
        return np.array([at.get(e, 0.0) for e in elements])

    tr_wt = np.array([vec_wt(t["composition"]) for t in training])
    sd = tr_wt.std(axis=0)
    sd[sd < 1e-9] = 1.0                       # elements absent from training carry raw units
    tr_at = np.array([vec_at(t["composition"]) for t in training])
    tr_ft = np.array([_features(t["composition"]) for t in training])
    ft_mu, ft_sd = np.nanmean(tr_ft, axis=0), np.nanstd(tr_ft, axis=0)
    ft_sd[ft_sd < 1e-9] = 1.0
    tr_ft = np.nan_to_num((tr_ft - ft_mu) / ft_sd)

    rows = []
    for r in evaluation:
        comp = r["composition"]
        w, a = vec_wt(comp), vec_at(comp)
        f = np.nan_to_num((np.array(_features(comp)) - ft_mu) / ft_sd)
        rows.append({
            "alloy": r["alloy"],
            "wt_production": min(_composition_distance(comp, t["composition"]) for t in training),
            "at_pct": float(np.min(np.linalg.norm(tr_at - a, axis=1))),
            "z_wt_pct": float(np.min(np.linalg.norm((tr_wt - w) / sd, axis=1))),
            "feature_space": float(np.min(np.linalg.norm(tr_ft - f, axis=1))),
        })
    d = pd.DataFrame(rows).drop_duplicates("alloy").set_index("alloy")

    prod = pd.read_csv(os.path.join(RESULTS_DIR, "nn_distance.csv")).set_index("alloy")
    sizes = prod.stratum.value_counts().reindex(list(STRATA)).to_dict()
    if not np.allclose(d.loc[prod.index, "wt_production"], prod.distance, atol=1e-3):
        raise RuntimeError("production distance does not reproduce nn_distance.csv")

    out = {"stratum_sizes": sizes, "metrics": []}
    for metric in ("wt_production", "at_pct", "z_wt_pct", "feature_space"):
        ranks = d[metric].rank(method="first")
        strata = pd.Series("FAR", index=d.index)
        strata[ranks <= sizes["NEAR"]] = "NEAR"
        strata[(ranks > sizes["NEAR"]) & (ranks <= sizes["NEAR"] + sizes["MID"])] = "MID"
        agree = float((strata.loc[prod.index] == prod.stratum).mean())
        rho = float(d[metric].corr(d["wt_production"], method="spearman"))
        f2 = frame.copy()
        f2["stratum"] = f2.alloy.map(strata)
        rec = {"metric": metric, "spearman_vs_production": rho, "stratum_agreement": agree,
               "near_overlap": int(((strata == "NEAR") & (prod.stratum.reindex(strata.index) == "NEAR")).sum()),
               "far_overlap": int(((strata == "FAR") & (prod.stratum.reindex(strata.index) == "FAR")).sum())}
        for s, new, ref, label in (("NEAR", "kg", "phys", "near_kg_vs_phys_pct"),
                                   ("FAR", "full", "kg", "far_full_vs_kg_pct")):
            m = (f2.stratum == s) & scorable(f2, "ys", (new, ref))
            rec[label] = relative_change(abs_errors(f2[m], "ys", new), abs_errors(f2[m], "ys", ref))
        out["metrics"].append(rec)
    return out


# --------------------------------------------------------------- report

def fmt(x, nd=1):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def write_report(res, path):
    L = ["# Robustness of the stratified claims", "",
         "Generated by `evaluation/prediction/scripts/revision_robustness.py` from the committed "
         "prediction CSVs. Nothing is re-run. Full system = mean over its five repeat campaigns. "
         f"Bootstrap: {N_BOOT:,} alloy-level resamples, 95% percentile intervals.", ""]

    L += ["## 1. Effect sizes with alloy-level uncertainty (R1.4)", "",
          "| effect | stratum | rows | alloys | MAE ref → new | change | 95% CI | leave-one-alloy-out range | most influential alloy |",
          "|---|---|---:|---:|---|---:|---|---|---|"]
    for name, b in res["bootstrap"].items():
        L.append(f"| {name} | {b['stratum']} | {b['n_rows']} | {b['n_alloys']} | "
                 f"{fmt(b['mae_ref'])} → {fmt(b['mae_new'])} | {fmt(b['rel_change_pct'])}% | "
                 f"[{fmt(b['ci95'][0])}, {fmt(b['ci95'][1])}]% | "
                 f"{fmt(b['loao_min'])} to {fmt(b['loao_max'])}% | "
                 f"{b['loao_most_influential']} ({fmt(b['loao_most_influential_value'])}%) |")
    ev = res["anchoring_events"]
    L += ["", f"Anchoring moved **{ev['rows_moved']} of {ev['rows_scored']}** scored yield-strength rows "
          f"across **{ev['alloys_moved']} alloys**; on NEAR, {ev['near_rows_moved']} of {ev['near_rows']} rows "
          f"across {ev['near_alloys_moved']} alloys. The single largest NEAR alloy carries "
          f"{fmt(100 * ev['near_gain_share_top_alloy'], 0)}% of the net NEAR gain.", "",
          "| alloy | stratum | d | rows moved | Σ error change (MPa, negative = better) |",
          "|---|---|---:|---:|---:|"]
    for r in ev["per_alloy"]:
        L.append(f"| {r['alloy']} | {r['stratum']} | {r['distance']:.2f} | {r['rows']} | {r['err_change_mpa']:+.1f} |")

    L += ["", "## 2. The MID stratum (R1.3)", "", "### Class mix", "",
          "| stratum | Precip | SSS | SC/DS |", "|---|---:|---:|---:|"]
    for s, m in res["class_mix"].items():
        L.append(f"| {s} | {m.get('Precip', 0)} | {m.get('SSS', 0)} | {m.get('SC/DS', 0)} |")
    L += ["", "### Agent yield-strength effect by class (full system vs ML+physics+KG)", "",
          "| class | stratum | alloys | rows | KG arm | full | change | 95% CI |",
          "|---|---|---:|---:|---:|---:|---:|---|"]
    for b in res["agent_by_class"]:
        L.append(f"| {b['class']} | {b['stratum']} | {b['n_alloys']} | {b['n_rows']} | {fmt(b['mae_ref'])} | "
                 f"{fmt(b['mae_new'])} | {fmt(b['rel_change_pct'])}% | [{fmt(b['ci95'][0])}, {fmt(b['ci95'][1])}]% |")
    L += ["", "### Within-class cells", "",
          "| stratum | class | alloys | YS rows | YS KG | YS full | EM rows | EM ML | EM ML+phys | EM full |",
          "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in res["within_class"]:
        L.append(f"| {r['stratum']} | {r['class']} | {r['n_alloys']} | {r['ys_n']} | {fmt(r.get('ys_kg'))} | "
                 f"{fmt(r.get('ys_full'))} | {r['em_n']} | {fmt(r.get('em_ml'), 2)} | "
                 f"{fmt(r.get('em_phys'), 2)} | {fmt(r.get('em_full'), 2)} |")
    L += ["", "### Boundary sweep (MID = lower ≤ d < upper)", "",
          "| lower | upper | alloys | YS full vs KG | EM ML | EM ML+phys | EM full | EM full vs ML |",
          "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in res["boundary_sweep"]:
        L.append(f"| {r['lower']} | {r['upper']} | {r['n_alloys']} | {fmt(r['ys_full_vs_kg_pct'])}% | "
                 f"{fmt(r['em_ml'], 2)} | {fmt(r['em_phys'], 2)} | {fmt(r['em_full'], 2)} | "
                 f"{fmt(r['em_full_vs_ml_pct'])}% |")
    L += ["", "### Which alloys carry the MID cells", ""]
    for c in res["mid_contributions"]:
        L.append(f"**{c['property'].upper()} {c['new']} − {c['ref']}** on {c['stratum']} "
                 f"({c['n_rows']} rows, {c['n_alloys']} alloys): net {c['net_mae_diff']:+.2f}; "
                 f"without the three largest contributors {fmt(c['net_mae_diff_without_top3'], 2)}.")
        L.append("")
        L.append("| alloy | rows | Σ difference | share of net |")
        L.append("|---|---:|---:|---:|")
        for t in c["top"]:
            L.append(f"| {t['alloy']} | {t['rows']} | {t['sum_d']:+.1f} | {fmt(100 * t['share_of_net'], 0)}% |")
        L.append("")
    L += ["### Conformal coverage against the error to be covered", "",
          "| property | stratum | n | median half-width | median |resid| | 90th pct |resid| | max |resid| | coverage |",
          "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in res["conformal_vs_error"]:
        L.append(f"| {r['property'].upper()} | {r['stratum']} | {r['n']} | {fmt(r['median_half_width'])} | "
                 f"{fmt(r['resid_median'])} | {fmt(r['resid_p90'])} | {fmt(r['resid_max'])} | "
                 f"{fmt(100 * r['coverage'])}% |")

    dm = res["distance_metrics"]
    L += ["", "## 3. Sensitivity to the distance metric (R2.2)", "",
          f"Each metric is cut into strata of the production sizes {dm['stratum_sizes']}.", "",
          "| metric | Spearman vs production | same stratum | NEAR overlap | FAR overlap | NEAR KG vs ML+phys | FAR full vs KG |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    for r in dm["metrics"]:
        L.append(f"| {r['metric']} | {r['spearman_vs_production']:.3f} | {fmt(100 * r['stratum_agreement'], 0)}% | "
                 f"{r['near_overlap']}/{dm['stratum_sizes']['NEAR']} | {r['far_overlap']}/{dm['stratum_sizes']['FAR']} | "
                 f"{fmt(r['near_kg_vs_phys_pct'])}% | {fmt(r['far_full_vs_kg_pct'])}% |")
    L.append("")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))


def main():
    rng = np.random.default_rng(RNG_SEED)
    frame = build_frame()
    res = {"bootstrap": {}}
    for name, s, prop, new, ref in (
        ("YS, KG anchoring vs ML+physics", "NEAR", "ys", "kg", "phys"),
        ("YS, KG anchoring vs ML+physics ", "MID", "ys", "kg", "phys"),
        ("YS, full system vs ML+physics+KG", "NEAR", "ys", "full", "kg"),
        ("YS, full system vs ML+physics+KG ", "MID", "ys", "full", "kg"),
        ("YS, full system vs ML+physics+KG  ", "FAR", "ys", "full", "kg"),
        ("EM, full system vs ML-only", "MID", "em", "full", "ml"),
        ("EM, ML+physics vs ML-only", "MID", "em", "phys", "ml"),
    ):
        m = (frame.stratum == s) & scorable(frame, prop, ("ml", "phys", "kg", "full"))
        b = cluster_bootstrap(frame, prop, new, ref, m, rng)
        b["stratum"] = s
        res["bootstrap"][name.strip() + f" [{s}]"] = b
    # The agent YS effect per alloy class x stratum. This is the table that
    # resolves the MID question: within a class the effect does not grow with
    # distance, and the strata differ in class mix.
    res["agent_by_class"] = []
    for cls in ("SSS", "Precip", "SC/DS"):
        for s in STRATA + ("ALL",):
            m = (frame.alloy_class == cls) & scorable(frame, "ys", ("ml", "phys", "kg", "full"))
            if s != "ALL":
                m &= frame.stratum == s
            if frame[m].alloy.nunique() < 3:
                continue
            b = cluster_bootstrap(frame, "ys", "full", "kg", m, rng)
            res["agent_by_class"].append({"class": cls, "stratum": s, **b})
    res["anchoring_events"] = anchoring_events(frame)
    res["class_mix"] = class_mix(frame)
    res["within_class"] = within_class(frame)
    res["boundary_sweep"] = boundary_sweep(frame)
    res["mid_contributions"] = [
        per_alloy_contribution(frame, "MID", "ys", "full", "kg", ("kg", "full")),
        per_alloy_contribution(frame, "MID", "em", "full", "ml", ("ml", "full")),
        per_alloy_contribution(frame, "MID", "em", "phys", "ml", ("ml", "phys")),
    ]
    res["conformal_vs_error"] = conformal_vs_error(frame)
    res["distance_metrics"] = distance_metrics(frame)

    with open(os.path.join(RESULTS_DIR, "revision_robustness.json"), "w") as fh:
        json.dump(res, fh, indent=2, default=float)
    write_report(res, os.path.join(RESULTS_DIR, "revision_robustness.md"))
    print(open(os.path.join(RESULTS_DIR, "revision_robustness.md")).read())


if __name__ == "__main__":
    main()
