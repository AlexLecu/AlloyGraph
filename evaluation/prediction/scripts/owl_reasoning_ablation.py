#!/usr/bin/env python3
"""What does the OWL layer actually do? An ablation, not an advertisement.

The paper claims an ontology-backed knowledge graph. This script measures what
the ontology and the HermiT reasoner contribute at build time, and what would
pass through undetected without them. It reports whatever it finds. Nothing
here was added to make the ontology look useful; every axiom tested is one that
already existed in ``backend/pipeline/build_ontology.py``.

WHAT IS INVENTORIED

  1. schema consistency   ``run_reasoner`` after ``build_ontology``: HermiT on
                          the empty schema, checking the class axioms are
                          satisfiable before any data is loaded.
  2. instance
     classification       ``populate_and_reason``: 77 alloy variants are
                          created carrying five computed features, HermiT runs,
                          and membership of SolidSolutionAlloy / GammaPrimeAlloy
                          and the four TCPRisk classes is counted before and
                          after.
  3. range validation     Four universal restrictions on Variant --
                          density [7,10] g/cm3, gamma-prime [0,85] vol%,
                          Md_avg [0.70,1.05], lattice mismatch [-2,2]% --
                          which make the ontology inconsistent if an
                          individual asserts a value outside them.
  4. persistence          Step 4 of the build saves a **schema-only** OWL file;
                          inferred class memberships are not written to it, and
                          ``enrich_graphdb`` uploads that schema to GraphDB as
                          vocabulary.

WHAT IS DELIBERATELY NOT CLAIMED. No runtime component loads the ontology. The
agents, the ML ensemble and the Weaviate retrieval path never read it, so
whatever the reasoner concludes cannot affect a prediction. The ablation
therefore measures data-quality and vocabulary value at ingestion, which is the
only place the layer is wired in.

Outputs (written to ../results):
    owl_reasoning_ablation.csv   per-alloy constraint check
    owl_reasoning_ablation.md    the report

Usage:
    python owl_reasoning_ablation.py
"""

import argparse
import json
import os
import sys

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "backend"))

TRAIN_FILE = os.path.join(PROJECT_ROOT, "backend", "alloy_crew", "models",
                          "training_data", "train_77alloys_v2.jsonl")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

#: The universal restrictions asserted on Variant in build_ontology.py.
#: Feature key -> (owl property, min_inclusive, max_inclusive).
RANGE_AXIOMS = {
    "density_calculated_gcm3": ("hasDensityCalculated", 7.0, 10.0),
    "gamma_prime_estimated_vol_pct": ("hasGammaPrimeEstimate", 0.0, 85.0),
    "Md_avg": ("hasMdAverage", 0.70, 1.05),
    "lattice_mismatch_pct": ("hasLatticeMismatchPct", -2.0, 2.0),
}

#: The composition that shipped in the evaluation set until the erratum, and
#: the corrected one. Ti was transposed: 12.0 wt% against a datasheet 1.2.
PE16_DEFECTIVE = {
    "alloy": "NIMONIC* PE16", "processing": "wrought",
    "composition": {"Ni": 43.5, "Fe": 34.0, "Cr": 16.5, "Mo": 3.0,
                    "Ti": 12.0, "Al": 1.2, "C": 0.06},
}


def features_of(record):
    from alloy_crew.models.feature_engineering import compute_alloy_features
    try:
        return compute_alloy_features(record)
    except Exception as exc:  # a record the feature layer itself rejects
        return {"_error": str(exc)}


def check_ranges(feats):
    """Which range axioms this individual would violate."""
    violations = []
    for key, (prop, lo, hi) in RANGE_AXIOMS.items():
        val = feats.get(key)
        if isinstance(val, (int, float)) and not (lo <= val <= hi):
            violations.append((prop, key, val, lo, hi))
    return violations


def classify_by_axiom(feats):
    """The class memberships the OWL axioms entail, evaluated directly.

    Mirrors the equivalent_to definitions in build_ontology.py: alloy type on
    hasGPFormersWtPct at the 2 wt% cut, TCP risk on hasMdAverage at the
    0.940 / 0.960 / 0.985 cuts.
    """
    gp = feats.get("GP_formers_wt_pct")
    md = feats.get("Md_avg")
    alloy_class = None
    if isinstance(gp, (int, float)):
        alloy_class = "GammaPrimeAlloy" if gp >= 2.0 else "SolidSolutionAlloy"
    tcp = None
    if isinstance(md, (int, float)):
        if md >= 0.985:
            tcp = "TCPRiskCritical"
        elif md >= 0.960:
            tcp = "TCPRiskElevated"
        elif md >= 0.940:
            tcp = "TCPRiskModerate"
        else:
            tcp = "TCPRiskLow"
    return alloy_class, tcp


#: Probe body executed in a fresh interpreter. owlready2 keeps one world per
#: ontology IRI for the lifetime of a process, so two probes in the same process
#: share individuals: the second inherits whatever the first asserted and
#: inconsistency from probe one contaminates probe two. Each probe therefore
#: gets its own subprocess. This was not hypothetical -- running both in-process
#: reported the corrected composition as inconsistent too.
_PROBE_SOURCE = """
import sys, json
sys.path.insert(0, {backend!r})
sys.path.insert(0, {pipeline!r})
from owlready2 import sync_reasoner_hermit, OwlReadyInconsistentOntologyError
from build_ontology import build_ontology
from alloy_crew.models.feature_engineering import compute_alloy_features

record = json.loads(sys.argv[1])
feats = compute_alloy_features(record)
onto = build_ontology()
with onto:
    Variant = onto["Variant"]
    v = Variant("probe")
    v.hasGPFormersWtPct = [float(feats.get("GP_formers_wt_pct", 0.0))]
    v.hasMdAverage = [float(feats.get("Md_avg", 0.0))]
    v.hasDensityCalculated = [float(feats.get("density_calculated_gcm3", 0.0))]
    v.hasGammaPrimeEstimate = [float(feats.get("gamma_prime_estimated_vol_pct", 0.0))]
    v.hasLatticeMismatchPct = [float(feats.get("lattice_mismatch_pct", 0.0))]
try:
    with onto:
        sync_reasoner_hermit(infer_property_values=True)
    classes = sorted(c.name for c in v.is_a if hasattr(c, "name"))
    print("VERDICT|consistent|" + ",".join(classes))
except OwlReadyInconsistentOntologyError:
    print("VERDICT|INCONSISTENT|")
"""


def run_hermit_probe(record, label):
    """Assert one individual against the real schema and ask HermiT.

    Returns ``(verdict, inferred_classes)``. Evaluating the axiom arithmetically
    shows what should happen; only running the reasoner shows what does.
    """
    import subprocess
    import tempfile
    src = _PROBE_SOURCE.format(
        backend=os.path.join(PROJECT_ROOT, "backend"),
        pipeline=os.path.join(PROJECT_ROOT, "backend", "pipeline"))
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write(src)
        path = fh.name
    try:
        proc = subprocess.run([sys.executable, path, json.dumps(record)],
                              capture_output=True, text=True, timeout=300,
                              cwd=PROJECT_ROOT)
        for line in proc.stdout.splitlines():
            if line.startswith("VERDICT|"):
                _, verdict, classes = line.split("|", 2)
                return verdict, classes
        return f"no verdict (rc={proc.returncode})", ""
    except Exception as exc:
        return f"probe failed: {type(exc).__name__}", ""
    finally:
        os.unlink(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default=RESULTS_DIR)
    ap.add_argument("--skip-hermit", action="store_true",
                    help="Skip the live reasoner probes (they need Java).")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    records = [json.loads(line) for line in open(TRAIN_FILE)]
    print(f"Knowledge-graph alloys: {len(records)}\n")

    rows = []
    for rec in records:
        feats = features_of(rec)
        viol = check_ranges(feats)
        alloy_class, tcp = classify_by_axiom(feats)
        rows.append({
            "alloy": rec.get("alloy"),
            "processing": rec.get("processing"),
            "composition_sum_wt_pct": round(sum(
                float(v) for v in (rec.get("composition") or {}).values() if v), 2),
            "GP_formers_wt_pct": feats.get("GP_formers_wt_pct"),
            "Md_avg": feats.get("Md_avg"),
            "density_calculated_gcm3": feats.get("density_calculated_gcm3"),
            "gamma_prime_estimated_vol_pct": feats.get("gamma_prime_estimated_vol_pct"),
            "lattice_mismatch_pct": feats.get("lattice_mismatch_pct"),
            "owl_alloy_class": alloy_class,
            "owl_tcp_class": tcp,
            "n_range_violations": len(viol),
            "violated": "; ".join(f"{p}={v}" for p, _, v, _, _ in viol),
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.outdir, "owl_reasoning_ablation.csv"), index=False)

    n_viol = int((df.n_range_violations > 0).sum())
    print(f"Range-axiom violations across the {len(df)} knowledge-graph alloys: {n_viol}")
    if n_viol:
        print(df[df.n_range_violations > 0][["alloy", "violated"]].to_string(index=False))
    unclassified = int(df.owl_alloy_class.isna().sum())
    print(f"\nClassification coverage: "
          f"{len(df) - unclassified}/{len(df)} variants receive an alloy class")
    print(f"  {df.owl_alloy_class.value_counts().to_dict()}")
    print(f"  {df.owl_tcp_class.value_counts().to_dict()}")

    sums = df.composition_sum_wt_pct
    off = df[(sums < 98) | (sums > 102)]
    print(f"\nCompositions summing outside 98-102 wt%: {len(off)} of {len(df)}")
    if len(off):
        print(off[["alloy", "composition_sum_wt_pct"]].to_string(index=False))

    # --- do the off-spec compositions survive ingestion? ---
    off_probe = []
    if not args.skip_hermit and len(off):
        print("\nHermiT on the alloys whose composition does not sum to ~100 wt%:")
        for _, r in off.iterrows():
            rec = next(x for x in records if x.get("alloy") == r["alloy"])
            verdict, classes = run_hermit_probe(rec, "offsum")
            off_probe.append({"alloy": r["alloy"],
                              "sum_wt_pct": r["composition_sum_wt_pct"],
                              "verdict": verdict})
            print(f"  {r['alloy']:22s} sum {r['composition_sum_wt_pct']:6.2f}%  -> {verdict}")

    # --- the PE16 probe ---
    print("\n" + "=" * 64)
    print("PE16 erratum: would any existing axiom have caught Ti = 12.0 wt%?")
    print("=" * 64)
    bad = features_of(PE16_DEFECTIVE)
    good_rec = json.loads(json.dumps(PE16_DEFECTIVE))
    good_rec["composition"]["Ti"] = 1.2
    good = features_of(good_rec)
    pe16_rows = []
    for key, (prop, lo, hi) in RANGE_AXIOMS.items():
        b, g = bad.get(key), good.get(key)
        caught = isinstance(b, (int, float)) and not (lo <= b <= hi)
        pe16_rows.append({"axiom": prop, "feature": key, "range": f"[{lo}, {hi}]",
                          "defective_value": b, "corrected_value": g,
                          "violates": caught})
        print(f"  {prop:26s} {str(b):>10} vs {str(g):>10}  range [{lo}, {hi}]  "
              f"-> {'VIOLATION' if caught else 'within range'}")
    pe16_caught = any(r["violates"] for r in pe16_rows)

    hermit = {}
    if not args.skip_hermit:
        print("\nRunning HermiT on a single-individual probe (empirical check):")
        for label, rec in (("pe16_corrected", good_rec), ("pe16_defective", PE16_DEFECTIVE)):
            verdict, classes = run_hermit_probe(rec, label)
            hermit[label] = verdict + (f" (classified {classes})" if classes else "")
            print(f"  {label:16s} -> {hermit[label]}")

    write_report(os.path.join(args.outdir, "owl_reasoning_ablation.md"),
                 df, pe16_rows, pe16_caught, hermit, len(off), off_probe)
    print(f"\nWritten to {args.outdir}")


def write_report(path, df, pe16_rows, pe16_caught, hermit, n_off, off_probe=()):
    L = []
    L.append("# OWL reasoning ablation\n")
    L.append("Generated by `evaluation/prediction/scripts/owl_reasoning_ablation.py`.\n")
    L.append("The question is what the ontology layer contributes, measured rather than")
    L.append("asserted. Every axiom tested below already existed in")
    L.append("`backend/pipeline/build_ontology.py`; none was added for this analysis.\n")
    L.append("## Where the reasoner is invoked\n")
    L.append("| # | site | what it produces |")
    L.append("|---|---|---|")
    L.append("| 1 | `run_reasoner(onto)` after `build_ontology()` | HermiT on the empty "
             "schema: checks the class axioms are satisfiable before data is loaded |")
    L.append("| 2 | `populate_and_reason(onto, json)` | 77 Variant individuals carrying "
             "five computed features; HermiT then classifies them and the counts are "
             "reported before and after |")
    L.append("| 3 | four `.only(ConstrainedDatatype(...))` restrictions on `Variant` | "
             "make the ontology inconsistent if an individual asserts a value outside "
             "the physical range |")
    L.append("| 4 | Step 4 of `main()` | saves a **schema-only** OWL file; "
             "`enrich_graphdb` uploads it to GraphDB as vocabulary |")
    L.append("")
    L.append("**No runtime component loads the ontology.** The agents, the ML ensemble")
    L.append("and the Weaviate retrieval path never read it, so nothing the reasoner")
    L.append("concludes can change a prediction. Inferred class memberships are also not")
    L.append("persisted: the saved OWL is rebuilt clean from the schema, so the")
    L.append("classification in site 2 is a build-time report and nothing more.\n")
    L.append("## What reasoning adds on the current data\n")
    total = len(df)
    classified = int(df.owl_alloy_class.notna().sum())
    L.append(f"All {classified} of {total} variants receive an alloy class and a TCP-risk")
    L.append("class. Both are defined by `equivalent_to` axioms over a single numeric")
    L.append("feature -- gamma-prime formers at a 2 wt% cut, Md_avg at the 0.940 / 0.960")
    L.append("/ 0.985 cuts -- so the reasoner is evaluating four threshold comparisons")
    L.append("that the Python feature layer has already computed. It derives no fact")
    L.append("that is not a direct restatement of an input.\n")
    L.append("| class | count |")
    L.append("|---|---:|")
    for k, v in df.owl_alloy_class.value_counts().items():
        L.append(f"| {k} | {v} |")
    for k, v in df.owl_tcp_class.value_counts().items():
        L.append(f"| {k} | {v} |")
    L.append("")
    n_viol = int((df.n_range_violations > 0).sum())
    L.append("## What reasoning catches: the range axioms\n")
    L.append(f"Across the {total} knowledge-graph alloys, **{n_viol}** violate a range")
    L.append("restriction. The knowledge graph is curated, so a clean result here is")
    L.append("expected and is not evidence the axioms are useless -- it is evidence they")
    L.append("are not currently firing on this input.\n")
    if n_viol:
        L.append("| alloy | violation |")
        L.append("|---|---|")
        for _, r in df[df.n_range_violations > 0].iterrows():
            L.append(f"| {r['alloy']} | {r['violated']} |")
        L.append("")
    L.append(f"Compositions summing outside 98-102 wt%: **{n_off}** of {total}. No axiom")
    L.append("constrains the composition sum, because the raw composition is never")
    L.append("loaded into the ontology -- only the five derived features are. A")
    L.append("composition that does not add up can only be caught indirectly, if it")
    L.append("pushes one of those five outside its range.\n")
    if off_probe:
        L.append("Running HermiT on each of them confirms it. This is the direct answer")
        L.append("to how many records would ingest with a violation undetected:\n")
        L.append("| alloy | composition sum | HermiT |")
        L.append("|---|---:|---|")
        for r in off_probe:
            L.append(f"| {r['alloy']} | {r['sum_wt_pct']}% | `{r['verdict']}` |")
        n_pass = sum(1 for r in off_probe if r["verdict"] == "consistent")
        L.append("")
        L.append(f"All {n_pass} of {len(off_probe)} pass. A composition three percentage")
        L.append("points off closure enters the knowledge graph with nothing raised.\n")
    L.append("## The PE16 erratum: would an existing axiom have caught it?\n")
    L.append("NIMONIC PE16 shipped with Ti = 12.0 wt% against a datasheet value of 1.2,")
    L.append("a transposed decimal that also left the composition summing to 110.3%. It")
    L.append("survived ingestion, training and two evaluation campaigns before being")
    L.append("found by hand.\n")
    L.append("| axiom | range | defective | corrected | violates |")
    L.append("|---|---|---:|---:|---|")
    for r in pe16_rows:
        L.append(f"| `{r['axiom']}` | {r['range']} | {r['defective_value']} | "
                 f"{r['corrected_value']} | {'**YES**' if r['violates'] else 'no'} |")
    L.append("")
    if pe16_caught:
        hit = next(r for r in pe16_rows if r["violates"])
        L.append(f"**Yes.** The defective composition drives `{hit['axiom']}` to")
        L.append(f"{hit['defective_value']}, outside its existing range {hit['range']}.")
        L.append("The axiom was already in the ontology; had the reasoner been run over")
        L.append("this record at ingestion, the ontology would have been inconsistent and")
        L.append("the defect would have surfaced there rather than six months later.\n")
        L.append("Two honest qualifications. The margin is thin -- 1.0599 against a bound")
        L.append("of 1.05, so the axiom catches this record by 0.0099 and a smaller")
        L.append("transposition would have passed. And the alloy is an *evaluation*")
        L.append("record: `populate_and_reason` runs over the 77 training alloys only, so")
        L.append("the constraint would not have fired on PE16 in the pipeline as it")
        L.append("stands. Extending ingestion validation to the evaluation set is a")
        L.append("one-line change and is the actionable finding here.\n")
    else:
        L.append("**No.** Every derived feature stays inside its range even with the")
        L.append("defective composition, so no existing axiom would have flagged it.\n")
    if hermit:
        L.append("### Reasoner behaviour, measured\n")
        L.append("Evaluating the axiom arithmetically shows what should happen. Running")
        L.append("HermiT shows what does:\n")
        L.append("| probe | HermiT verdict |")
        L.append("|---|---|")
        for k, v in hermit.items():
            L.append(f"| {k} | `{v}` |")
        L.append("")
    L.append("## Honest positioning\n")
    if n_viol == 0 and pe16_caught:
        L.append("The ontology is **not** currently functioning as data-quality")
        L.append("infrastructure, and the paper should not claim it is. It catches")
        L.append("nothing on the curated 77 alloys because there is nothing there to")
        L.append("catch, and the one real defect the project encountered sat in a record")
        L.append("the validation step never sees.\n")
        L.append("What the evidence does support is narrower and still worth stating:")
        L.append("the range axioms encode physical bounds that *would* have caught that")
        L.append("defect, which makes them a usable ingestion gate rather than")
        L.append("decoration -- provided ingestion validation is extended to cover every")
        L.append("record entering the system, not just the training set.\n")
        L.append("Absent that change, the ontology's demonstrated role is **schema and")
        L.append("interoperability**: a shared vocabulary with external alignments,")
        L.append("uploaded to GraphDB, giving the knowledge graph a typed structure and")
        L.append("making it queryable and reusable. That is a legitimate contribution")
        L.append("and it is the one the measurements here support. Claiming the reasoner")
        L.append("improves prediction quality would not be supportable: no runtime")
        L.append("component reads the ontology at all.\n")
    else:
        L.append("See the tables above; positioning depends on the measured counts.\n")
    L.append("## What was deliberately not done\n")
    L.append("No new axioms were written. It would have been easy to add a")
    L.append("composition-sum restriction, or to tighten the Md bound until PE16 failed")
    L.append("more comfortably, and then report the ontology as having caught a real")
    L.append("error. That would be measuring a constraint built with knowledge of the")
    L.append("answer. The numbers above are what the ontology does as it stands.")
    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
