#!/usr/bin/env python3
"""Where the agent layer's yield-strength gain comes from (Expert Systems revision R1.3).

The stratified tables show the agent gain is carried by solid-solution (SSS)
alloys at every distance and is absent for gamma-prime alloys outside NEAR.
The agents see two deterministic estimates before they decide: the ML ensemble
and the class-specific physics model (Labusch-Nabarro for SSS, gamma-prime
fraction scaling for precipitation-hardened alloys). This asks whether the
class pattern is explained by how good the physics estimate is in each class
and by which of the two estimates the agents end up nearer to.

Per row, with ML = the raw ensemble and PHYS = AlloyAnalysisTool's physics
estimate, both recomputed deterministically from the committed code:

    position t = (FULL - ML) / (PHYS - ML)
        t = 0 the agents kept the ML value, t = 1 they adopted the physics value.
        Only rows where ML and PHYS differ by more than 5% are used, since t is
        undefined when the two agree.
    oracle     = the better of ML and PHYS on each row, an upper bound for any
        selector that only chooses between the two.

Full system = row-wise mean prediction over the five repeat campaigns.

Outputs (../results):
    revision_agent_mechanism.md
    revision_agent_mechanism.json

Usage:
    python revision_agent_mechanism.py
"""

import json
import logging
import os
import sys

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "backend"))
sys.path.insert(0, SCRIPT_DIR)

logging.disable(logging.WARNING)

from alloy_crew.tools.analysis_tool import AlloyAnalysisTool  # noqa: E402
import revision_robustness as rr  # noqa: E402

RESULTS_DIR = rr.RESULTS_DIR
MIN_SPREAD = 0.05


def physics_ys(frame):
    comps = {}
    for stem in rr.DATASETS.values():
        for rec in rr.load_jsonl(os.path.join(rr.DATA_DIR, f"{stem}.jsonl")):
            if rec.get("composition"):
                comps[rec["alloy"]] = (rec["composition"], rec.get("processing", "wrought"))
    tool = AlloyAnalysisTool()
    out = []
    for alloy, t in zip(frame.alloy, frame.temperature):
        comp, proc = comps[alloy]
        res, _ = tool._get_physics_predictions(comp, int(round(t)), proc)
        out.append(float(res.get("Yield Strength", np.nan)))
    return np.array(out)


def summarise(sub):
    a = sub.actual_ys.to_numpy(float)
    ml, ph, full, kg = (sub[c].to_numpy(float) for c in ("ml_ys", "phys_model_ys", "full_mean_ys", "kg_ys"))
    spread = np.abs(ph - ml) / np.maximum(np.abs(ml), 1e-9)
    use = spread > MIN_SPREAD
    t = (full[use] - ml[use]) / (ph[use] - ml[use])
    phys_better = np.abs(ph - a) < np.abs(ml - a)
    closer_to_phys = np.abs(full - ph) < np.abs(full - ml)
    return {
        "rows": int(len(sub)), "alloys": int(sub.alloy.nunique()),
        "mae_ml": float(np.mean(np.abs(ml - a))),
        "mae_physics": float(np.mean(np.abs(ph - a))),
        "mae_kg_arm": float(np.mean(np.abs(kg - a))),
        "mae_full": float(np.mean(np.abs(full - a))),
        "mae_oracle": float(np.mean(np.minimum(np.abs(ml - a), np.abs(ph - a)))),
        "physics_better_share": float(phys_better.mean()),
        "rows_with_spread": int(use.sum()),
        "median_position_t": float(np.median(t)) if len(t) else None,
        "full_closer_to_physics_share": float(closer_to_phys[use].mean()) if use.any() else None,
        "agreement_with_better_source": float(
            (closer_to_phys[use] == phys_better[use]).mean()) if use.any() else None,
    }


def main():
    frame = rr.build_frame()
    m = rr.scorable(frame, "ys", ("ml", "kg", "full"))
    frame = frame[m].copy()
    frame["phys_model_ys"] = physics_ys(frame)
    frame["full_mean_ys"] = np.column_stack(
        [frame[f"full{s}_ys"].to_numpy(float) for s in rr.SEEDS]).mean(axis=1)

    res = []
    for cls in ("SSS", "Precip", "SC/DS"):
        for s in rr.STRATA + ("ALL",):
            sel = frame.alloy_class == cls
            if s != "ALL":
                sel &= frame.stratum == s
            if sel.sum() < 5:
                continue
            res.append({"class": cls, "stratum": s, **summarise(frame[sel])})

    # Post-hoc diagnostic, NOT a competitor: a deterministic routing rule written
    # after seeing the class pattern above. It asks how much of the agent gain a
    # rule with no LLM reproduces, the same question the manuscript already asks
    # of elastic modulus. Fitted on the evaluation set, so it cannot be reported
    # as a better system.
    a = frame.actual_ys.to_numpy(float)
    rule = np.where(frame.alloy_class == "SSS", frame.phys_model_ys, frame.kg_ys)
    routing = []
    for s in rr.STRATA + ("ALL",):
        sel = np.ones(len(frame), bool) if s == "ALL" else (frame.stratum == s).to_numpy()
        routing.append({
            "stratum": s, "rows": int(sel.sum()),
            "mae_kg_arm": float(rr.abs_errors(frame[sel], "ys", "kg").mean()),
            "mae_rule_sss_physics": float(np.abs(rule[sel] - a[sel]).mean()),
            "mae_full": float(rr.abs_errors(frame[sel], "ys", "full").mean()),
        })

    with open(os.path.join(RESULTS_DIR, "revision_agent_mechanism.json"), "w") as fh:
        json.dump({"by_class": res, "posthoc_routing_rule": routing}, fh, indent=2)

    L = ["# Agent yield-strength gain: mechanism by alloy class", "",
         "Generated by `evaluation/prediction/scripts/revision_agent_mechanism.py`. "
         "MAE in MPa on the rows scorable for every arm. *t* is the median position of the "
         "full-system value between ML (0) and physics (1), over rows where the two differ by "
         f"more than {MIN_SPREAD:.0%}. *Oracle* picks the better of ML and physics per row.", "",
         "| class | stratum | rows | alloys | ML | physics | ML+phys+KG | full | oracle | physics better | median t | agents nearer the better source |",
         "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    f = rr.fmt
    for r in res:
        L.append(f"| {r['class']} | {r['stratum']} | {r['rows']} | {r['alloys']} | {f(r['mae_ml'])} | "
                 f"{f(r['mae_physics'])} | {f(r['mae_kg_arm'])} | {f(r['mae_full'])} | {f(r['mae_oracle'])} | "
                 f"{f(100 * r['physics_better_share'], 0)}% | {f(r['median_position_t'], 2)} | "
                 f"{f(100 * r['agreement_with_better_source'], 0) if r['agreement_with_better_source'] is not None else '—'}% |")
    L += ["", "## Post-hoc routing rule (diagnostic only)", "",
          "SSS rows take the physics (Labusch–Nabarro) estimate, every other row the ML+physics+KG "
          "value. Written after seeing the table above and scored on the same rows, so it bounds how "
          "much of the agent gain a deterministic rule reproduces; it is not a fair competitor.", "",
          "| stratum | rows | ML+phys+KG | rule | full system |", "|---|---:|---:|---:|---:|"]
    for r in routing:
        L.append(f"| {r['stratum']} | {r['rows']} | {f(r['mae_kg_arm'])} | {f(r['mae_rule_sss_physics'])} | "
                 f"{f(r['mae_full'])} |")
    L.append("")
    with open(os.path.join(RESULTS_DIR, "revision_agent_mechanism.md"), "w") as fh:
        fh.write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
