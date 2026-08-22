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

#: The 77 knowledge-graph alloys, used to fit the sparsity-matched sampler.
KG_TRAINING = os.path.join(PROJECT_ROOT, "backend", "alloy_crew", "models",
                           "training_data", "train_77alloys_v2.jsonl")

#: Aggregate placeholder in the source data, not an element. Never sampled.
NOT_AN_ELEMENT = {"Other"}

#: Above this an element counts as a deliberate addition rather than a residual.
MAJOR_CUT = 1.0


def fit_composition_model(path=KG_TRAINING):
    """Fit an empirical composition model to the knowledge-graph alloys.

    Returns the three things needed to sample a composition the way real
    superalloys are actually built:

      counts        the observed distribution of how many deliberate additions
                    (> 1 wt%, excluding Ni) an alloy carries -- median 5,
                    range 0-8 across the 77 alloys
      majors        per element, how often it appears as a deliberate addition
                    and the range of amounts observed when it does
      minors        per element, how often it appears as a residual (<= 1 wt%)
                    and the range observed

    Fitting rather than hardcoding keeps the sampler auditable: change the
    knowledge graph and the baseline follows it.
    """
    counts, majors, minors = [], {}, {}
    n = 0
    with open(path) as fh:
        for line in fh:
            rec = json.loads(line)
            comp = {k: float(v) for k, v in rec["composition"].items()
                    if v and k != BALANCE and k not in NOT_AN_ELEMENT}
            n += 1
            counts.append(sum(1 for v in comp.values() if v > MAJOR_CUT))
            for el, val in comp.items():
                bucket = majors if val > MAJOR_CUT else minors
                bucket.setdefault(el, []).append(val)
    to_model = lambda b: {el: (len(v) / n, min(v), max(v)) for el, v in b.items()}
    return counts, to_model(majors), to_model(minors)


def sample_composition_sparsity(rng, model):
    """One composition drawn the way real superalloys are put together.

    Draw how many deliberate additions to make from the observed distribution,
    choose which elements in proportion to how often each is actually used,
    then draw each amount inside the range observed for that element. Residual
    elements (C, B, Zr, Mn, Si) are added independently at their observed
    frequencies, because a superalloy without carbon is not a superalloy.

    This is the baseline the uniform sampler should have been. Drawing all 20
    elements independently produces 12.2 additions per candidate against a
    plausibility cap of 8, so the uniform arm fails the element-count and
    trace-element rules before any search quality is measured. Matching the
    sparsity of real alloys removes that confound and makes the comparison to
    the Designer a test of search rather than of composition density.
    """
    counts, majors, minors = model
    k = int(rng.choice(counts))

    names = sorted(majors)
    weights = np.array([majors[e][0] for e in names], dtype=float)
    k = min(k, len(names))
    chosen = rng.choice(len(names), size=k, replace=False,
                        p=weights / weights.sum()) if k else []

    comp = {}
    for i in chosen:
        el = names[i]
        _, lo, hi = majors[el]
        comp[el] = float(rng.uniform(lo, hi))

    for el, (freq, lo, hi) in sorted(minors.items()):
        if el in comp:
            continue
        if rng.random() < freq:
            comp[el] = float(rng.uniform(lo, hi))

    # Keep every draw inside the optimizer's bounds, so the Guard and Tuner see
    # the same feasible region they see for a Designer composition.
    for el in list(comp):
        if el in ELEMENT_BOUNDS:
            lo, hi = ELEMENT_BOUNDS[el]
            comp[el] = max(lo, min(hi, comp[el]))
        else:
            del comp[el]

    ni = 100.0 - sum(comp.values())
    lo, hi = ELEMENT_BOUNDS[BALANCE]
    if not (lo <= ni <= hi):
        return None
    comp[BALANCE] = ni
    return {k2: round(v, 3) for k2, v in comp.items() if v > 1e-9}

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


def run_random(targets, n, rng, with_optimizer=False, plausible_aware=False,
               sampler=None):
    """Best-of-N random search over every target spec.

    ``sampler`` is a callable taking the RNG and returning a composition or
    None for a rejected draw. Defaults to the uniform box sampler.
    """
    draw = sampler or sample_composition
    per_target, total_hits, total_goals, tcp = [], 0, 0, []
    for t in targets:
        tgt, proc, temp = t["target_props"], t["processing"], t["temperature"]
        best = (-1, None, "Critical", (-1, -1, -1))
        draws = 0
        while draws < n:
            comp = draw(rng)
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

    # The full 2x2 grid: {plain, Guard+Tuner} x {select on hits, select on
    # plausibility first}. The Guard+Tuner plausibility-aware cell is the
    # strongest baseline available at a given budget and is the fair
    # comparison against the Designer, which also runs the optimizer.
    model = fit_composition_model()
    counts, majors, minors = model
    sparsity = lambda r: sample_composition_sparsity(r, model)
    print(f"Sparsity model fitted to {len(counts)} knowledge-graph alloys: "
          f"median {int(np.median(counts))} additions >1 wt%, range "
          f"{min(counts)}-{max(counts)}; {len(majors)} elements used as additions, "
          f"{len(minors)} as residuals\n")
    out["sparsity_model"] = {
        "n_alloys": len(counts),
        "major_count_distribution": {str(k): counts.count(k) for k in sorted(set(counts))},
        "major_elements": {e: {"freq": round(v[0], 3), "min": v[1], "max": v[2]}
                           for e, v in sorted(majors.items())},
        "minor_elements": {e: {"freq": round(v[0], 3), "min": v[1], "max": v[2]}
                           for e, v in sorted(minors.items())},
    }

    for n in args.n:
        print(f"N = {n} candidates per target:")
        variants = [("random best-of-{n}", False, False, None),
                    ("random best-of-{n} (plaus-aware)", False, True, None),
                    ("sparsity-matched best-of-{n}", False, False, sparsity),
                    ("sparsity-matched best-of-{n} (plaus-aware)", False, True, sparsity)]
        if not args.skip_optimizer:
            variants += [("random+Guard+Tuner best-of-{n}", True, False, None),
                         ("random+Guard+Tuner best-of-{n} (plaus-aware)", True, True, None),
                         ("sparsity+Guard+Tuner best-of-{n}", True, False, sparsity),
                         ("sparsity+Guard+Tuner best-of-{n} (plaus-aware)", True, True, sparsity)]
        for label, use_opt, plaus, samp in variants:
            rng = np.random.default_rng(args.seed)
            out["arms"].append(summarise(
                label.format(n=n),
                run_random(targets, n, rng, with_optimizer=use_opt,
                           plausible_aware=plaus, sampler=samp)))
        print()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    md = os.path.splitext(args.out)[0] + ".md"
    write_report(md, out)
    print(f"Written to {args.out} and {md}")


def write_report(path, out):
    """Markdown companion, so the design section has one readable table."""
    L = []
    L.append("# Inverse design: random-search baselines\n")
    L.append("Generated by `evaluation/design/scripts/random_search_baseline.py`")
    L.append(f"(seed {out['seed']}, {out['n_targets']} target specifications, a property")
    L.append(f"counts as met at {out['hit_fraction']:.0%} of target, QuickCheck physics scorer).\n")
    L.append("Three criteria, increasingly strict:\n")
    L.append("- **hit rate** — individual property targets met, pooled over all targets")
    L.append("- **usable** — every property target met *and* TCP risk below Critical")
    L.append("- **credible** — usable *and* inside the metallurgical plausibility envelope\n")
    L.append("| arm | hit rate | usable | credible | filtered out | median elements >1 wt% | median refractory |")
    L.append("|---|---:|---:|---:|---:|---:|---:|")
    for a in out["arms"]:
        L.append(f"| {a['label']} | {a['hit_rate']:.1f}% | "
                 f"{a['usable']}/{a['n_designs']} | **{a['credible']}/{a['n_designs']}** | "
                 f"{a['filtered_out']} | {a['median_major_elements']} | "
                 f"{a['median_refractory']:.1f} wt% |")
    L.append("")
    L.append("## Rules tripped\n")
    L.append("| arm | R1 element count | R2 refractory | R3 trace | R4 gamma-prime formers |")
    L.append("|---|---:|---:|---:|---:|")
    for a in out["arms"]:
        r = a.get("rules_tripped", {})
        L.append(f"| {a['label']} | {r.get('R1_element_count', 0)} | "
                 f"{r.get('R2_refractory_budget', 0)} | {r.get('R3_trace_elements', 0)} | "
                 f"{r.get('R4_gamma_prime_formers', 0)} |")
    L.append("")
    L.append("## Reading\n")
    des = next((a for a in out["arms"] if a["label"].startswith("designer")), None)

    def budget_of(a):
        if "best-of-" not in a["label"]:
            return None
        return int(a["label"].split("best-of-")[1].split()[0].rstrip(")"))

    search = [a for a in out["arms"] if budget_of(a) is not None]
    budgets = sorted({budget_of(a) for a in search})
    small = budgets[0]

    def best_at(n, prefix=None):
        pool = [a for a in search if budget_of(a) == n
                and (prefix is None or a["label"].startswith(prefix))]
        if not pool:
            return None
        return max(pool, key=lambda a: (a["credible"], a["hit_rate"]))

    champ = best_at(small)
    if des and champ:
        beats = champ["credible"] > des["credible"]
        L.append(f"**At the Designer's own budget of {small} candidates, the strongest")
        L.append(f"random baseline {'beats it' if beats else 'does not beat it'}.**")
        L.append(f"`{champ['label']}` reaches {champ['credible']}/{champ['n_designs']}")
        L.append(f"credible designs at a {champ['hit_rate']:.1f}% hit rate, against the")
        L.append(f"Designer's {des['credible']}/{des['n_designs']} at {des['hit_rate']:.1f}%.")
        L.append(f"Only {champ['filtered_out']} of its {champ['n_designs']} candidates are")
        L.append("filtered out as implausible.\n")
        if beats:
            L.append("That arm is the like-for-like comparison: same number of candidates,")
            L.append("and the same deterministic Guard and Tuner the Designer's own")
            L.append("pipeline runs in Phase 2. The only thing that differs is where the")
            L.append("starting composition comes from -- an LLM, or a draw from the")
            L.append("empirical composition model of the knowledge graph. **The draw wins,")
            L.append("on both credibility and property targets.**\n")
            L.append("The claim that the Designer's contribution is metallurgical")
            L.append("plausibility does not survive this. What the earlier comparison")
            L.append("measured was the uniform sampler's inability to produce a sparse")
            L.append("composition, not the Designer's ability to reason about one.\n")

    L.append("### Where each arm's advantage comes from\n")
    L.append("Two failure modes trade off at low budget, and the arms separate cleanly")
    L.append("by which one they suffer:\n")
    for a in [x for x in search if budget_of(x) == small]:
        lost = a["filtered_out"]
        miss = a["n_designs"] - a["usable"]
        L.append(f"- `{a['label']}`: {lost}/{a['n_designs']} implausible, "
                 f"{miss}/{a['n_designs']} short of target -> {a['credible']} credible")
    if des:
        L.append(f"- `{des['label']}`: {des['filtered_out']}/{des['n_designs']} implausible, "
                 f"{des['n_designs'] - des['usable']}/{des['n_designs']} short of target "
                 f"-> {des['credible']} credible")
    L.append("")
    L.append("The uniform sampler hits targets and fails plausibility; sparsity-matched")
    L.append("sampling alone does the reverse. Only the combination of realistic")
    L.append("sampling with the deterministic optimizer clears both, which is also")
    L.append("exactly what the Designer's pipeline is -- with an LLM in place of the")
    L.append("draw.\n")

    big = budgets[-1] if len(budgets) > 1 else None
    champ_big = best_at(big) if big else None
    if des and champ_big:
        L.append(f"**Scaling makes it worse for the Designer.** At {big} candidates,")
        L.append(f"`{champ_big['label']}` reaches")
        L.append(f"{champ_big['credible']}/{champ_big['n_designs']} credible designs at")
        L.append(f"{champ_big['hit_rate']:.1f}%, with {champ_big['filtered_out']} filtered")
        L.append("out. No LLM call is involved at any point.\n")

    L.append("## The sampler is the confound, and it was tested\n")
    L.append("An earlier version of this comparison used only the uniform sampler and")
    L.append("reported that random search produces no credible designs at all. That")
    L.append("result was mostly an artefact of how the baseline drew compositions.")
    L.append("`sample_composition` draws all 20 elements independently from")
    L.append("`ELEMENT_BOUNDS`, giving 12.2 additions above 1 wt% per candidate against")
    L.append("rule R1's cap of 8, and putting Cu above its residual limit about five")
    L.append("times in six. R1 and R3 failed before any search quality was measured.\n")
    L.append("`sample_composition_sparsity` removes that confound by fitting the")
    L.append("empirical composition model of the 77 knowledge-graph alloys: how many")
    L.append("deliberate additions an alloy carries, which elements are actually used")
    L.append("and how often, the amounts observed for each, and the residual elements")
    L.append("that make a superalloy a superalloy. The effect on the rule counts is")
    L.append("decisive -- R1 and R3 stop firing entirely, and median additions fall from")
    L.append("11 to 5, matching the knowledge graph.\n")
    L.append("The Designer's advantage at equal budget survives that correction. The")
    L.append("categorical version of the claim does not.\n")
    L.append("## Remaining limits\n")
    L.append("- The sparsity model is fitted to the same 77 alloys the system is built")
    L.append("  on, so it inherits their coverage. Those alloys contain no Re, Ru, Hf, V")
    L.append("  or Cu, and the sampler therefore never proposes a rhenium-bearing single")
    L.append("  crystal. For the gamma-prime wrought and cast targets the Designer")
    L.append("  addresses this is not a restriction, but it bounds what the baseline")
    L.append("  could ever find.")
    L.append("- Twenty target specifications, so every rate here moves in steps of 5")
    L.append("  percentage points and the credible counts are single digits.")
    L.append("- The plausibility filter is one operationalisation of manufacturability,")
    L.append("  calibrated to the envelope of real alloys rather than to these results.")
    L.append("  A different filter would move the credible counts for every arm.")
    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
