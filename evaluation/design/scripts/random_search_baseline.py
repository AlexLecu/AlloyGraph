#!/usr/bin/env python3
"""Random-search baselines for inverse design (KBS R1.5).

Asks what an LLM Designer buys over sampling compositions at random. Three
arms are scored against the same 20 target specifications and the same
criterion the design evaluation uses -- a property counts as met when the
predicted value reaches 90% of its target:

  random          N random compositions per target, best-of-N by hit count
  random+opt      the same random seeds put through the deterministic
                  Guard + Tuner, isolating the optimizer's contribution
                  from the Designer LLM's
  designer        the 20 compositions the LLM pipeline actually produced,
                  re-scored here so all three arms share one scorer

That last arm matters. The paper's 72% comes from the full agent pipeline,
whose Phase-3 evaluation is a different (and more generous) estimator than
QuickCheck. Comparing a QuickCheck-scored random baseline against an
agent-scored 72% would flatter the baseline or the Designer depending on
which way the estimators disagree, so the Designer's own compositions are
re-scored on QuickCheck for a like-for-like number.

Everything is deterministic given --seed. No API calls: scoring uses
QuickCheckTool (physics) and the deterministic optimizer, never the agents.

Usage:
    python random_search_baseline.py                 # N = 5, the pipeline budget
    python random_search_baseline.py --n 100 1000    # scaling sweep
"""

import argparse
import importlib.util
import json
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)                       # evaluation/design
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "backend"))
sys.path.insert(0, SCRIPT_DIR)

from alloy_crew.deterministic_optimizer import ELEMENT_BOUNDS, optimize  # noqa: E402
from alloy_crew.tools.quick_check_tool import QuickCheckTool  # noqa: E402
from plausibility_filter import check_plausibility, composition_stats  # noqa: E402

#: Property key in the targets -> QuickCheck output key.
PROP_KEYS = {
    "Yield Strength": "estimated_ys_mpa",
    "Tensile Strength": "estimated_uts_mpa",
    "Elongation": "estimated_el_pct",
    "Elastic Modulus": "estimated_em_gpa",
}

#: A property is met at 90% of target, matching the design evaluation.
HIT_FRACTION = 0.90

#: Ni is the balance element, as in every real superalloy and in the Guard's
#: rebalancing, so it is solved for rather than sampled.
BALANCE = "Ni"

_QC = QuickCheckTool()


def load_targets():
    """Import the 20 target specs from the design evaluation script."""
    path = os.path.join(SCRIPT_DIR, "run_design_evaluation.py")
    spec = importlib.util.spec_from_file_location("_rde", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_rde"] = mod
    spec.loader.exec_module(mod)
    return mod.TARGETS


def sample_composition(rng):
    """One random composition inside ELEMENT_BOUNDS, Ni as balance.

    Returns None when the draw leaves Ni outside its own bounds, which the
    caller retries. Sampling the minor elements and solving for Ni keeps the
    draws in the region where superalloys actually live; sampling Ni too and
    renormalising would mostly produce compositions no bound could accept.
    """
    comp = {}
    for el, (lo, hi) in ELEMENT_BOUNDS.items():
        if el == BALANCE:
            continue
        comp[el] = float(rng.uniform(lo, hi))
    ni = 100.0 - sum(comp.values())
    lo, hi = ELEMENT_BOUNDS[BALANCE]
    if not (lo <= ni <= hi):
        return None
    comp[BALANCE] = ni
    return {k: round(v, 3) for k, v in comp.items() if v > 1e-9}


def score(composition, target, processing, temperature_c):
    """Hits, total goals and TCP risk for one composition under QuickCheck."""
    try:
        r = json.loads(_QC._run(composition=composition, processing=processing,
                                temperature_c=temperature_c))
    except Exception:
        return 0, len(target), "Error", {}
    hits, preds = 0, {}
    for prop, tval in target.items():
        key = PROP_KEYS.get(prop)
        pred = r.get(key)
        if not isinstance(pred, (int, float)) or not tval:
            continue
        preds[prop] = pred
        if pred / tval >= HIT_FRACTION:
            hits += 1
    return hits, len(target), r.get("tcp_risk", "Unknown"), preds


def run_random(targets, n, rng, with_optimizer=False, plausible_aware=False):
    """Best-of-N random search over every target spec."""
    per_target, total_hits, total_goals, tcp = [], 0, 0, []
    for t in targets:
        tgt, proc, temp = t["target_props"], t["processing"], t["temperature"]
        best = (-1, None, "Critical", (-1, -1, -1))
        draws = 0
        while draws < n:
            comp = sample_composition(rng)
            if comp is None:
                continue                       # rejected draw, does not use budget
            draws += 1
            if with_optimizer:
                try:
                    comp = optimize(comp, tgt, temperature_c=temp,
                                    processing=proc)["composition"]
                except Exception:
                    pass
            h, g, risk, _ = score(comp, tgt, proc, temp)
            # Rank on (not-Critical, hits): any practitioner running a random
            # search would discard a Critical-TCP candidate before counting its
            # property hits, so selecting on hits alone would understate the
            # baseline. This makes it the harder comparison.
            ok, _, _ = check_plausibility(comp, proc)
            cand = ((1 if ok else 0) if plausible_aware else 1,
                    0 if str(risk).lower().startswith("crit") else 1, h)
            if best[1] is None or cand > best[3]:
                best = (h, comp, risk, cand)
        total_hits += best[0]
        total_goals += len(tgt)
        tcp.append(best[2])
        ok, failed, _ = check_plausibility(best[1], proc)
        per_target.append({"id": t["id"], "hits": best[0], "goals": len(tgt),
                           "tcp_risk": best[2], "plausible": ok,
                           "failed_rules": failed, "processing": proc,
                           **composition_stats(best[1])})
    return per_target, total_hits, total_goals, tcp


def run_designer(targets):
    """Re-score the compositions the LLM pipeline produced, same scorer."""
    import pandas as pd
    path = os.path.join(BASE_DIR, "results", "design_evaluation_results.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    # The results CSV also carries aggregate columns -- "Refractory (wt%)" and
    # "Al+Ti (wt%)" -- alongside the real elements. Treating those as elements
    # adds roughly 11 wt% of phantom material and wrecks the composition, so
    # accept only genuine element symbols.
    valid_elements = set(ELEMENT_BOUNDS)
    el_cols = [c for c in df.columns
               if c.endswith("(wt%)") and c.replace(" (wt%)", "") in valid_elements]
    per_target, total_hits, total_goals, tcp = [], 0, 0, []
    for t, (_, row) in zip(targets, df.iterrows()):
        comp = {c.replace(" (wt%)", ""): float(row[c])
                for c in el_cols if pd.notna(row[c]) and float(row[c]) > 0}
        if not comp:
            continue
        h, g, risk, _ = score(comp, t["target_props"], t["processing"], t["temperature"])
        total_hits += h
        total_goals += len(t["target_props"])
        tcp.append(risk)
        ok, failed, _ = check_plausibility(comp, t["processing"])
        per_target.append({"id": t["id"], "hits": h, "goals": len(t["target_props"]),
                           "tcp_risk": risk, "plausible": ok,
                           "failed_rules": failed, "processing": t["processing"],
                           **composition_stats(comp)})
    return per_target, total_hits, total_goals, tcp


def summarise(label, res):
    """Report property, TCP and plausibility criteria side by side.

    "usable" = every property target met AND no Critical TCP.
    "credible" = usable AND inside the metallurgical plausibility envelope.
    The last column is the one an inverse-design method should be judged on:
    a composition that scores well but could not be manufactured is not a
    design result.
    """
    import collections
    per, hits, goals, tcp = res
    crit = sum(1 for r in tcp if str(r).lower().startswith("crit"))
    full = sum(1 for p in per if p["hits"] == p["goals"])
    usable = sum(1 for p in per if p["hits"] == p["goals"]
                 and not str(p["tcp_risk"]).lower().startswith("crit"))
    credible = sum(1 for p in per if p["hits"] == p["goals"]
                   and not str(p["tcp_risk"]).lower().startswith("crit")
                   and p.get("plausible"))
    lost = sum(1 for p in per if not p.get("plausible"))
    rules = collections.Counter(r for p in per for r in p.get("failed_rules", []))
    n = len(per)
    print(f"  {label:32s} {100*hits/goals:5.1f}%  usable {usable:2d}/{n:2d}={100*usable/n:5.1f}%  "
          f"CREDIBLE {credible:2d}/{n:2d}={100*credible/n:5.1f}%  filtered-out {lost:2d}")
    if rules:
        print(f"      rules tripped: {dict(rules)}")
    med_maj = sorted(p["n_major"] for p in per)[n // 2]
    med_ref = sorted(p["refractory"] for p in per)[n // 2]
    frac_tr = sum(1 for p in per if p["has_cu_mn_si"]) / n
    print(f"      median elements>1wt%: {med_maj}   median refractory: {med_ref:.1f} wt%   "
          f"Cu/Mn/Si above trace: {100*frac_tr:.0f}%")
    return {"label": label, "hit_rate": round(100 * hits / goals, 1),
            "usable": usable, "credible": credible, "n_designs": n,
            "usable_rate": round(100 * usable / n, 1),
            "credible_rate": round(100 * credible / n, 1),
            "filtered_out": lost, "rules_tripped": dict(rules),
            "median_major_elements": med_maj, "median_refractory": med_ref,
            "frac_trace_violation": round(frac_tr, 3),
            "per_target": per}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, nargs="+", default=[5],
                    help="candidates per target (default 5 = Phase-1 attempts x iterations)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=os.path.join(BASE_DIR, "results",
                                                  "random_search_baseline.json"))
    ap.add_argument("--skip-optimizer", action="store_true",
                    help="skip the random+Guard+Tuner arm (it is the slow one)")
    args = ap.parse_args()

    targets = load_targets()
    print(f"Target specifications: {len(targets)}")
    print(f"Criterion: predicted >= {HIT_FRACTION:.0%} of target, QuickCheck (physics) scorer")
    print(f"Seed: {args.seed}\n")

    out = {"seed": args.seed, "n_targets": len(targets),
           "hit_fraction": HIT_FRACTION, "arms": []}

    d = run_designer(targets)
    if d:
        print("REFERENCE — LLM Designer compositions, re-scored on this scorer:")
        out["arms"].append(summarise("designer (LLM pipeline)", d))
        print()

    for n in args.n:
        print(f"N = {n} candidates per target:")
        rng = np.random.default_rng(args.seed)
        out["arms"].append(summarise(f"random best-of-{n}", run_random(targets, n, rng)))
        rng = np.random.default_rng(args.seed)
        out["arms"].append(summarise(f"random best-of-{n} (plaus-aware)",
                                     run_random(targets, n, rng, plausible_aware=True)))
        if not args.skip_optimizer:
            rng = np.random.default_rng(args.seed)
            out["arms"].append(summarise(
                f"random+Guard+Tuner best-of-{n}",
                run_random(targets, n, rng, with_optimizer=True)))
        print()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"Written to {args.out}")


if __name__ == "__main__":
    main()
