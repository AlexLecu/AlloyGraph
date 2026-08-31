#!/usr/bin/env python3
"""What the Reviewer agent adds: the Analyst-only ablation, scored against the full system.

Reviewer 2.4 asked what separates the two agents. The pipeline runs Analyst then
Reviewer then a deterministic layer, and a difference between the full system
and ML+physics+KG cannot say which of the three produced it. This ablation
removes exactly one thing: the Reviewer task leaves the crew, the Analyst's
output goes straight to deterministic post-processing, and every guard -- the
evidence envelope, the trust system, the correction reconciliation, the
density/gamma-prime override -- still runs. Prompts, tools, anchors and agents
are untouched.

One seed only. The full system is a five-seed mean with a seed standard
deviation of 1.1-3.7 MPa depending on stratum, so a single-seed ablation carries
that much noise and differences smaller than roughly two standard deviations
should not be read as real. This is stated in the output rather than left for
the reader to work out.

Usage:
    python analyst_ablation_report.py
    python analyst_ablation_report.py --md ../results/analyst_ablation.md
"""

import argparse
import os

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
OUTPUT = os.path.join(BASE_DIR, "output")
RESULTS = os.path.join(BASE_DIR, "results")

DATASETS = ("sss", "precip", "sc_ds")
SEEDS = (42, 43, 44, 45, 46)
PROPS = (("ys", "YS", "MPa"), ("uts", "UTS", "MPa"),
         ("el", "EL", "%"), ("em", "EM", "GPa"))
STRATA = ("NEAR", "MID", "FAR")

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


def full_system_stats(dist):
    """Per-seed MAE for the full system, so the ablation can be read against its noise."""
    frames = [load(f"stageb_seed{s}_full_system") for s in SEEDS]
    out = {}
    for key, _, _ in PROPS:
        for scope in ("ALL",) + STRATA:
            vals = []
            for f in frames:
                f = f.copy()
                f["stratum"] = f.alloy.map(dist)
                sub = f if scope == "ALL" else f[f.stratum == scope]
                vals.append(mae(sub, key)[1])
            out[(key, scope)] = (float(np.mean(vals)), float(np.std(vals)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--md", default=os.path.join(RESULTS, "analyst_ablation.md"))
    args = ap.parse_args()

    nn = pd.read_csv(os.path.join(RESULTS, "nn_distance.csv"))
    dist = nn.set_index("alloy")["stratum"]

    ab = load("seed42_analyst_only")
    kg = load("seed42_v2prod_ml_physics_kg")
    for d in (ab, kg):
        d["stratum"] = d.alloy.map(dist)
    fs = full_system_stats(dist)

    rows = []
    for key, prop, unit in PROPS:
        for scope in ("ALL",) + STRATA:
            a = ab if scope == "ALL" else ab[ab.stratum == scope]
            k = kg if scope == "ALL" else kg[kg.stratum == scope]
            n, m_ab = mae(a, key)
            _, m_kg = mae(k, key)
            m_fs, sd_fs = fs[(key, scope)]
            rows.append({"property": prop, "unit": unit, "scope": scope, "n": n,
                         "kg": m_kg, "analyst_only": m_ab,
                         "full": m_fs, "full_sd": sd_fs,
                         "reviewer_delta": m_ab - m_fs,
                         # The ablation is ONE seed and the full system a
                         # five-seed mean, so the difference carries
                         # sqrt(1 + 1/5) = 1.095 times the single-seed sigma.
                         # Dividing by sigma alone overstates significance by
                         # about 10%.
                         "reviewer_delta_sd": ((m_ab - m_fs) / (sd_fs * np.sqrt(1 + 1 / len(SEEDS)))
                                               if sd_fs else np.nan)})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "analyst_ablation.csv"), index=False)

    stages = ab.pipeline_stage.value_counts().to_dict() if "pipeline_stage" in ab else {}
    env = int(pd.to_numeric(ab.get("envelope_overrides"), errors="coerce").fillna(0).gt(0).sum())
    env_n = int(pd.to_numeric(ab.get("envelope_overrides"), errors="coerce").fillna(0).sum())
    tok = pd.to_numeric(ab.get("total_tokens"), errors="coerce").fillna(0)
    req = pd.to_numeric(ab.get("llm_requests"), errors="coerce").fillna(0)
    secs = pd.to_numeric(ab.get("eval_time_sec"), errors="coerce").fillna(0)

    fs42 = load("stageb_seed42_full_system")
    tok_fs = pd.to_numeric(fs42.get("total_tokens"), errors="coerce").fillna(0)
    req_fs = pd.to_numeric(fs42.get("llm_requests"), errors="coerce").fillna(0)
    env_fs = int(pd.to_numeric(fs42.get("envelope_overrides"), errors="coerce").fillna(0).gt(0).sum())

    L = ["# Analyst-only ablation", "",
         "Generated by `evaluation/prediction/scripts/analyst_ablation_report.py`.", "",
         "The full pipeline with the Reviewer agent removed: the Analyst's output goes "
         "straight to deterministic post-processing, and every guard still runs. One "
         "seed (42), 471 cases, sampling temperature 0.0, the same provider and prompts "
         "as the reported campaigns.", "",
         "**Read the differences against seed noise.** The full system is a five-seed "
         "mean; its seed standard deviation is given so that a difference can be judged "
         "against it. The ablation is a single seed and carries the same noise, so it "
         "is only worth calling a difference real when it clears roughly two standard "
         "deviations. The \"in SE\" column divides by "
         "sigma*sqrt(1 + 1/5) = 1.095*sigma, the standard error of a "
         "single-seed value minus a five-seed mean, not by sigma alone.", "",
         "| property | scope | n | ML+physics+KG | Analyst only | Full system (5 seeds) | Reviewer adds | in SE |",
         "|---|---|---:|---:|---:|---:|---:|---:|"]
    for _, r in df.iterrows():
        L.append(f"| {r.property} ({r.unit}) | {r.scope} | {r.n} | {r.kg:.2f} | "
                 f"{r.analyst_only:.2f} | {r.full:.2f} ± {r.full_sd:.2f} | "
                 f"{r.reviewer_delta:+.2f} | {r.reviewer_delta_sd:+.1f} |")
    L += ["",
          "\"Reviewer adds\" is Analyst-only MAE minus full-system MAE: positive means "
          "the Reviewer improved the result, negative means the Analyst alone was better.",
          "", "## Pipeline stages and guards", "",
          f"- pipeline_stage: {stages}",
          f"- envelope guard fired on **{env} of {len(ab)} cases** "
          f"({100 * env / max(len(ab), 1):.2f}%), {env_n} property overrides in total; "
          f"full system seed 42 fired on {env_fs} of {len(fs42)}",
          "", "## Cost", "",
          f"| | Analyst only | Full system (seed 42) |",
          f"|---|---:|---:|",
          f"| LLM requests, median per case | {req.median():.0f} | {req_fs.median():.0f} |",
          f"| tokens, total | {tok.sum():,.0f} | {tok_fs.sum():,.0f} |",
          f"| tokens, median per case | {tok.median():,.0f} | {tok_fs.median():,.0f} |",
          f"| cost at ${USD_PER_MTOK}/Mtok | ${tok.sum() / 1e6 * USD_PER_MTOK:.2f} | "
          f"${tok_fs.sum() / 1e6 * USD_PER_MTOK:.2f} |",
          f"| wall clock, median per case | {secs.median():.0f} s | — |",
          ""]
    with open(args.md, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))

    print(df.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print(f"\npipeline_stage: {stages}")
    print(f"envelope guard: {env}/{len(ab)} cases, {env_n} overrides "
          f"(full system seed 42: {env_fs}/{len(fs42)})")
    print(f"tokens {tok.sum():,.0f} vs {tok_fs.sum():,.0f}; "
          f"cost ${tok.sum() / 1e6 * USD_PER_MTOK:.2f} vs "
          f"${tok_fs.sum() / 1e6 * USD_PER_MTOK:.2f}")
    print(f"\nWritten to {args.md}")


if __name__ == "__main__":
    main()
