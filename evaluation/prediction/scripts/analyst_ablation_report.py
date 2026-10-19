#!/usr/bin/env python3
"""What the Reviewer agent adds: the Analyst-only ablation, scored against the full system.

The pipeline runs Analyst then Reviewer then a deterministic layer, and a
difference between the full system and ML+physics+KG cannot say which of the
three produced it. This ablation removes exactly one thing: the Reviewer task
leaves the crew, the Analyst's output goes straight to deterministic
post-processing, and every guard -- the evidence envelope, the trust system,
the correction reconciliation, the density/gamma-prime override -- still runs.
Prompts, tools, anchors and agents are untouched.

Both arms are five campaigns (seed labels 42-46, temperature 0.0, same provider,
model and prompts). The provider exposes no seed control, so the campaigns of
each arm are independent repeats and each arm's run-to-run noise is estimated
from its own five campaigns -- not borrowed from the other arm, which is what the
earlier single-campaign version had to do (Expert Systems R1.2).

The comparison per cell is the difference of the two five-campaign mean MAEs,
judged against the standard error sqrt(sd_a^2/5 + sd_f^2/5) and a Welch t-test
over the five per-campaign MAEs of each arm. Campaign seeds are labels, not
shared randomness, so the arms are not paired by seed.

Usage:
    python analyst_ablation_report.py
    python analyst_ablation_report.py --md ../results/analyst_ablation.md
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
OUTPUT = os.path.join(BASE_DIR, "output")
RESULTS = os.path.join(BASE_DIR, "results")

DATASETS = ("sss", "precip", "sc_ds")
SEEDS = (42, 43, 44, 45, 46)
PROPS = (("ys", "YS", "MPa"), ("uts", "UTS", "MPa"),
         ("el", "EL", "%"), ("em", "EM", "GPa"))
STRATA = ("NEAR", "MID", "FAR")

ARM_FILES = {"analyst": "seed{s}_analyst_only", "full": "stageb_seed{s}_full_system"}

#: DeepInfra list price for Llama-3.3-70B at the time of the campaign,
#: USD per million tokens. Prompt and completion are billed at the same rate.
USD_PER_MTOK = 0.23


def load(tag):
    return pd.concat([pd.read_csv(os.path.join(OUTPUT, f"{tag}_{d}.csv"))
                      for d in DATASETS], ignore_index=True)


def mae(df, key):
    a = pd.to_numeric(df.get(f"actual_{key}"), errors="coerce").to_numpy(float)
    p = pd.to_numeric(df.get(f"pred_{key}"), errors="coerce").to_numpy(float)
    m = ~(np.isnan(a) | np.isnan(p) | (a == 0))
    return int(m.sum()), (float(np.mean(np.abs(a[m] - p[m]))) if m.sum() else np.nan)


def per_campaign(dist):
    """MAE per (arm, property, scope) for each of the five campaigns."""
    frames = {arm: [load(pat.format(s=s)) for s in SEEDS] for arm, pat in ARM_FILES.items()}
    out = {}
    for arm, dfs in frames.items():
        for df in dfs:
            df["stratum"] = df.alloy.map(dist)
        for key, _, _ in PROPS:
            for scope in ("ALL",) + STRATA:
                vals, ns = [], []
                for df in dfs:
                    sub = df if scope == "ALL" else df[df.stratum == scope]
                    n, m = mae(sub, key)
                    vals.append(m)
                    ns.append(n)
                out[(arm, key, scope)] = (np.array(vals), ns)
    return frames, out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--md", default=os.path.join(RESULTS, "analyst_ablation.md"))
    args = ap.parse_args()

    nn = pd.read_csv(os.path.join(RESULTS, "nn_distance.csv"))
    dist = nn.set_index("alloy")["stratum"]
    kg = load("seed42_v2prod_ml_physics_kg")
    kg["stratum"] = kg.alloy.map(dist)

    frames, pc = per_campaign(dist)
    for arm, dfs in frames.items():
        keys = [set(zip(d.alloy, d.temperature)) for d in dfs]
        if any(k != keys[0] for k in keys):
            raise RuntimeError(f"{arm}: campaigns do not cover the same cases")
    if set(zip(frames["analyst"][0].alloy, frames["analyst"][0].temperature)) != \
            set(zip(frames["full"][0].alloy, frames["full"][0].temperature)):
        raise RuntimeError("the two arms do not cover the same cases")

    rows = []
    for key, prop, unit in PROPS:
        for scope in ("ALL",) + STRATA:
            a, ns_a = pc[("analyst", key, scope)]
            f, ns_f = pc[("full", key, scope)]
            if len(set(ns_a + ns_f)) != 1:
                raise RuntimeError(f"{prop} {scope}: scored case counts differ {ns_a} {ns_f}")
            k = kg if scope == "ALL" else kg[kg.stratum == scope]
            sd_a, sd_f = a.std(ddof=1), f.std(ddof=1)
            se = np.sqrt(sd_a ** 2 / len(a) + sd_f ** 2 / len(f))
            delta = a.mean() - f.mean()
            p = stats.ttest_ind(a, f, equal_var=False).pvalue
            rows.append({
                "property": prop, "unit": unit, "scope": scope, "n": ns_a[0],
                "kg": mae(k, key)[1],
                "analyst_mean": a.mean(), "analyst_sd": sd_a,
                "analyst_min": a.min(), "analyst_max": a.max(),
                "full_mean": f.mean(), "full_sd": sd_f,
                "full_min": f.min(), "full_max": f.max(),
                "reviewer_delta": delta, "delta_se": se,
                "delta_in_se": delta / se if se else np.nan,
                "welch_p": p,
                "ranges_disjoint": bool(a.min() > f.max() or a.max() < f.min()),
            })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "analyst_ablation.csv"), index=False)

    ab_all = pd.concat(frames["analyst"], ignore_index=True)
    fs_all = pd.concat(frames["full"], ignore_index=True)

    def num(d, col):
        return pd.to_numeric(d.get(col), errors="coerce").fillna(0)

    stages = ab_all.pipeline_stage.value_counts().to_dict() if "pipeline_stage" in ab_all else {}
    env = int(num(ab_all, "envelope_overrides").gt(0).sum())
    env_fs = int(num(fs_all, "envelope_overrides").gt(0).sum())
    tok_ab, tok_fs = num(ab_all, "total_tokens"), num(fs_all, "total_tokens")
    req_ab, req_fs = num(ab_all, "llm_requests"), num(fs_all, "llm_requests")
    secs = num(ab_all, "eval_time_sec")

    L = ["# Analyst-only ablation, five campaigns per arm", "",
         "Generated by `evaluation/prediction/scripts/analyst_ablation_report.py`.", "",
         "The full pipeline with the Reviewer agent removed: the Analyst's output goes "
         "straight to deterministic post-processing, and every guard still runs. Both arms "
         "are five campaigns (seed labels 42-46), 471 cases each, sampling temperature 0.0, "
         "the same provider, model and prompts. The provider exposes no seed control, so "
         "the campaigns are independent repeats and each arm's standard deviation is "
         "estimated from its own five campaigns.", "",
         "*Reviewer adds* = Analyst-only MAE minus full-system MAE (positive: the Reviewer "
         "helps). *SE* = sqrt(sd_a²/5 + sd_f²/5). *p* = Welch t-test over the five "
         "per-campaign MAEs of each arm. *Disjoint* = the two arms' five-campaign ranges do "
         "not overlap.", "",
         "| property | scope | n | ML+physics+KG | Analyst only (5) | Full system (5) | Reviewer adds | in SE | p | disjoint |",
         "|---|---|---:|---:|---:|---:|---:|---:|---:|:---:|"]
    for _, r in df.iterrows():
        L.append(f"| {r.property} ({r.unit}) | {r.scope} | {r.n} | {r.kg:.2f} | "
                 f"{r.analyst_mean:.2f} ± {r.analyst_sd:.2f} | {r.full_mean:.2f} ± {r.full_sd:.2f} | "
                 f"{r.reviewer_delta:+.2f} | {r.delta_in_se:+.1f} | {r.welch_p:.3g} | "
                 f"{'yes' if r.ranges_disjoint else 'no'} |")
    # The same comparison resolved by alloy class (yield strength only): the
    # stratum cells mix classes in different proportions, and the class
    # analysis (revision_robustness.py) shows the agent effect follows class.
    cls = nn.set_index("alloy")["alloy_class"]
    L += ["", "## Yield strength by stratum and alloy class", "",
          "| stratum | class | n | Analyst only (5) | Full system (5) | Reviewer adds |",
          "|---|---|---:|---:|---:|---:|"]
    class_rows = []
    for scope in STRATA:
        for c in ("SSS", "Precip", "SC/DS"):
            vals = {}
            for arm, dfs in frames.items():
                ms = []
                for d in dfs:
                    sub = d[(d.stratum == scope) & (d.alloy.map(cls) == c)]
                    n, m = mae(sub, "ys")
                    ms.append(m)
                vals[arm] = np.array(ms)
            if n == 0:
                continue
            delta = vals["analyst"].mean() - vals["full"].mean()
            class_rows.append({"stratum": scope, "class": c, "n": n,
                               "analyst_mean": vals["analyst"].mean(),
                               "full_mean": vals["full"].mean(), "reviewer_delta": delta})
            L.append(f"| {scope} | {c} | {n} | {vals['analyst'].mean():.1f} | "
                     f"{vals['full'].mean():.1f} | {delta:+.1f} |")
    pd.DataFrame(class_rows).to_csv(os.path.join(RESULTS, "analyst_ablation_by_class.csv"), index=False)

    L += ["", "## Pipeline stages and guards", "",
          f"- pipeline_stage over the five Analyst-only campaigns: {stages}",
          f"- envelope guard fired on {env} of {len(ab_all)} Analyst-only rows; "
          f"{env_fs} of {len(fs_all)} full-system rows",
          "", "## Cost, five campaigns per arm", "",
          "| | Analyst only | Full system |", "|---|---:|---:|",
          f"| LLM requests, median per case | {req_ab.median():.0f} | {req_fs.median():.0f} |",
          f"| tokens, total | {tok_ab.sum():,.0f} | {tok_fs.sum():,.0f} |",
          f"| tokens, median per case | {tok_ab.median():,.0f} | {tok_fs.median():,.0f} |",
          f"| token ratio, full / Analyst only | | {tok_fs.sum() / max(tok_ab.sum(), 1):.1f}x |",
          f"| cost at ${USD_PER_MTOK}/Mtok | ${tok_ab.sum() / 1e6 * USD_PER_MTOK:.2f} | "
          f"${tok_fs.sum() / 1e6 * USD_PER_MTOK:.2f} |",
          f"| wall clock, median per case | {secs.median():.0f} s | — |", ""]
    with open(args.md, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
