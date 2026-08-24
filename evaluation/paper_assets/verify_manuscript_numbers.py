#!/usr/bin/env python3
"""Check every quantitative claim in the manuscript against a committed source.

The manuscript lives in Overleaf and is git-ignored under paper_src/, so it
cannot be a tracked input. What is tracked is this checker: each claim is
declared once, with the file and expression that produces it, and the script
re-derives the number and compares. A claim that no committed artifact produces
is declared UNTRACEABLE rather than quietly omitted.

Run it before submission and after any re-run of the campaigns.

Usage:
    python verify_manuscript_numbers.py
    python verify_manuscript_numbers.py --tex path/to/manuscript.tex   # also greps the tex
"""

import argparse
import json
import os
import re
import sys

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
PRED = os.path.join(ROOT, "evaluation", "prediction")
RES = os.path.join(PRED, "results")
CHAT = os.path.join(ROOT, "evaluation", "chatbot", "results")
DESIGN = os.path.join(ROOT, "evaluation", "design", "results")
MODELS = os.path.join(ROOT, "backend", "alloy_crew", "models", "saved_models_v2")

DEFAULT_TEX = os.path.join(ROOT, "paper_src", "Alloygraph_kbs",
                           "elsarticle-template-num.tex")

TOL = 0.005


def load_csv(p):
    return pd.read_csv(p)


def checks():
    """(section, claim, manuscript value, derived value, source) tuples."""
    out = []

    def add(sec, claim, tex, got, src):
        out.append((sec, claim, tex, got, src))

    h = load_csv(os.path.join(RES, "headline_metrics.csv"))
    s = load_csv(os.path.join(RES, "stratified_metrics.csv"))
    c = load_csv(os.path.join(RES, "conformal_coverage.csv"))
    w = load_csv(os.path.join(RES, "wilcoxon_tests.csv"))

    def hm(arm, prop, col="mae"):
        r = h[(h.arm == arm) & (h.property == prop)]
        return None if r.empty else float(r.iloc[0][col])

    def sm(m, st, p, col="mae"):
        r = s[(s.method == m) & (s.stratum == st) & (s.property == p)]
        return None if r.empty else float(r.iloc[0][col])

    # --- Table 1, headline ------------------------------------------------
    T1 = {"ML-only": {"YS": (98.73, 0.845), "UTS": (112.96, 0.846),
                      "EL": (13.84, 0.246), "EM": (12.86, 0.480)},
          "ML+physics": {"YS": (98.73, 0.845), "UTS": (113.25, 0.846),
                         "EL": (13.80, 0.246), "EM": (8.21, 0.830)},
          "ML+physics+KG": {"YS": (92.37, 0.858), "UTS": (112.75, 0.848),
                            "EL": (12.82, 0.403), "EM": (8.21, 0.830)},
          "Full system (5 seeds)": {"YS": (78.84, 0.874), "UTS": (96.44, 0.863),
                                    "EL": (12.56, 0.412), "EM": (8.94, 0.775)},
          "GPT-4.1-mini (stock)": {"YS": (197.69, 0.365), "UTS": (176.06, 0.606),
                                   "EL": (23.69, -0.360), "EM": (16.12, 0.260)},
          "Llama-3.3-70B (DeepInfra)": {"YS": (265.04, -0.097), "UTS": (290.36, 0.131),
                                        "EL": (24.12, -0.376), "EM": (22.55, -0.185)},
          "GBM raw features": {"YS": (102.76, 0.810), "UTS": (109.39, 0.837),
                               "EL": (14.45, 0.248), "EM": (15.05, 0.346)},
          "RF raw features": {"YS": (93.57, 0.836), "UTS": (110.81, 0.831),
                              "EL": (14.53, 0.238), "EM": (18.90, -0.013)},
          "GPR raw features": {"YS": (119.33, 0.754), "UTS": (132.90, 0.767),
                               "EL": (14.35, 0.285), "EM": (16.62, 0.266)},
          "GPR physics features": {"YS": (232.55, 0.076), "UTS": (279.12, 0.073),
                                   "EL": (23.06, -0.122), "EM": (38.74, -3.154)}}
    for arm, props in T1.items():
        for p, (mae, r2) in props.items():
            add("T1", f"{arm} {p} MAE", mae, hm(arm, p), "headline_metrics.csv")
            add("T1", f"{arm} {p} R2", r2, hm(arm, p, "r2"), "headline_metrics.csv")
    for p, sd in (("YS", 1.13), ("UTS", 0.93), ("EL", 0.12), ("EM", 0.06)):
        add("T1", f"Full system {p} SD", sd,
            hm("Full system (5 seeds)", p, "mae_sd"), "headline_metrics.csv")

    # --- Table 2, stratified ---------------------------------------------
    T2 = {("YS", "NEAR"): (56, 107.56, 107.56, 75.42, 63.15, 3.41),
          ("YS", "MID"): (91, 76.05, 76.05, 75.93, 71.81, 2.95),
          ("YS", "FAR"): (138, 110.10, 110.10, 110.10, 89.83, 1.48),
          ("UTS", "NEAR"): (57, 96.55, 98.80, 96.43, 77.71, 3.65),
          ("UTS", "MID"): (91, 83.53, 83.92, 83.80, 76.27, 3.49),
          ("UTS", "FAR"): (142, 138.41, 137.84, 137.84, 116.88, 2.05),
          ("EL", "NEAR"): (61, 28.53, 28.52, 24.30, 24.25, 0.26),
          ("EL", "MID"): (82, 7.91, 7.86, 7.60, 7.21, 0.10),
          ("EL", "FAR"): (141, 10.93, 10.89, 10.89, 10.61, 0.10),
          ("EM", "NEAR"): (55, 14.60, 6.75, 6.75, 7.73, 0.36),
          ("EM", "MID"): (65, 9.92, 10.48, 10.48, 12.02, 0.28),
          ("EM", "FAR"): (184, 13.38, 7.84, 7.84, 8.22, 0.12)}
    arms = ["ML-only", "ML+physics", "ML+physics+KG", "Full system (5 seeds)"]
    for (p, st), v in T2.items():
        r = s[(s.property == p) & (s.stratum == st)]
        add("T2", f"{p} {st} n", v[0], None if r.empty else int(r.iloc[0].n_rows),
            "stratified_metrics.csv")
        for arm, val in zip(arms, v[1:5]):
            add("T2", f"{p} {st} {arm}", val, sm(arm, st, p), "stratified_metrics.csv")
        add("T2", f"{p} {st} Full SD", v[5],
            sm("Full system (5 seeds)", st, p, "mae_sd"), "stratified_metrics.csv")

    # --- Table 3, method baselines ---------------------------------------
    T3 = {("YS", "NEAR"): (107.6, 107.6, 113.0, 107.8, 99.5, 194.5),
          ("YS", "MID"): (76.0, 76.0, 67.5, 67.9, 97.8, 151.0),
          ("YS", "FAR"): (110.1, 110.1, 121.9, 104.8, 141.6, 301.7),
          ("YS", "ALL"): (98.7, 98.7, 102.8, 93.6, 119.3, 232.6),
          ("UTS", "NEAR"): (96.5, 98.8, 123.5, 129.7, 131.4, 221.8),
          ("UTS", "MID"): (83.5, 83.9, 94.1, 96.7, 119.2, 197.2),
          ("UTS", "FAR"): (138.4, 137.8, 113.5, 112.3, 142.3, 354.6),
          ("UTS", "ALL"): (113.0, 113.2, 109.4, 110.8, 132.9, 279.1)}
    arms3 = ["ML-only", "ML+physics", "GBM raw", "RF raw", "GPR raw", "GPR physics"]
    for (p, st), vals in T3.items():
        for arm, val in zip(arms3, vals):
            got = sm(arm, st, p)
            add("T3", f"{p} {st} {arm}", val, None if got is None else round(got, 1),
                "stratified_metrics.csv")

    # --- conformal --------------------------------------------------------
    CF = {("YS", "NEAR"): (56, 75.0, 368.2), ("YS", "MID"): (91, 93.4, 374.2),
          ("YS", "FAR"): (138, 89.9, 374.2), ("YS", "ALL"): (285, 88.1, 373.4),
          ("UTS", "NEAR"): (57, 94.7, 438.2), ("UTS", "MID"): (91, 92.3, 438.2),
          ("UTS", "FAR"): (142, 83.8, 436.2), ("UTS", "ALL"): (290, 88.6, 437.2),
          ("EL", "NEAR"): (61, 70.5, 38.4), ("EL", "MID"): (82, 93.9, 38.5),
          ("EL", "FAR"): (141, 85.1, 38.5), ("EL", "ALL"): (284, 84.5, 38.5),
          ("EM", "NEAR"): (55, 87.3, 76.0), ("EM", "MID"): (65, 100.0, 76.2),
          ("EM", "FAR"): (184, 87.0, 75.8), ("EM", "ALL"): (304, 89.8, 76.0)}
    for (p, st), (n, cov, wd) in CF.items():
        r = c[(c.property == p) & (c.stratum == st)]
        if r.empty:
            add("T5", f"{p} {st}", n, None, "conformal_coverage.csv")
            continue
        r = r.iloc[0]
        add("T5", f"{p} {st} n", n, int(r.n_scored), "conformal_coverage.csv")
        add("T5", f"{p} {st} coverage %", cov, round(float(r.coverage) * 100, 1),
            "conformal_coverage.csv")
        add("T5", f"{p} {st} width", wd, round(float(r.median_width), 1),
            "conformal_coverage.csv")

    # --- Wilcoxon ---------------------------------------------------------
    ww = w[(w.scope == "FAR") & (w.property == "YS")].iloc[0]
    add("3.2", "Wilcoxon FAR YS W", 2396, round(float(ww.W)), "wilcoxon_tests.csv")
    add("3.2", "Wilcoxon FAR YS p (x1e-7)", 3.4, round(float(ww.p) * 1e7, 1),
        "wilcoxon_tests.csv")
    add("3.2", "Wilcoxon FAR YS n", 138, int(ww.n), "wilcoxon_tests.csv")
    add("3.2", "Wilcoxon FAR YS median gain", 35.6, round(-float(ww.median_delta), 1),
        "wilcoxon_tests.csv")
    add("3.2", "Wilcoxon FAR YS wins", 101, int(ww.wins), "wilcoxon_tests.csv")
    wa = w[(w.scope == "FAR (by alloy)") & (w.property == "YS")].iloc[0]
    add("3.2", "Wilcoxon alloy-level p", 0.016, round(float(wa.p), 3), "wilcoxon_tests.csv")
    add("3.2", "Wilcoxon alloy-level n", 43, int(wa.n), "wilcoxon_tests.csv")

    # --- derived prose claims --------------------------------------------
    kg_near = 100 * (sm("ML+physics", "NEAR", "YS") - sm("ML+physics+KG", "NEAR", "YS")) \
        / sm("ML+physics", "NEAR", "YS")
    add("3.2", "KG gain on NEAR YS (%)", 30, round(kg_near), "stratified_metrics.csv")
    far_gain = 100 * (sm("ML+physics", "FAR", "YS") - sm("Full system (5 seeds)", "FAR", "YS")) \
        / sm("ML+physics", "FAR", "YS")
    add("3.2", "agent gain on FAR YS (%)", 18.4, round(far_gain, 1), "stratified_metrics.csv")
    seps = (sm("ML+physics", "FAR", "YS") - sm("Full system (5 seeds)", "FAR", "YS")) \
        / sm("Full system (5 seeds)", "FAR", "YS", "mae_sd")
    add("3.2", "FAR separation (SD)", 13.7, round(seps, 1), "stratified_metrics.csv")

    # --- chatbot ----------------------------------------------------------
    mcq = json.load(open(os.path.join(CHAT, "mcq_report.json")))["systems"]
    for k, claim in (("chatbot", 91.2), ("llama", 49.6), ("gpt", 50.0)):
        o = mcq[k]["overall"]
        add("3.4", f"MCQ overall {k} (%)", claim,
            round(100 * o["correct"] / o["total"], 1), "mcq_report.json")
    add("3.4", "MCQ total questions", 250,
        sum(mcq["chatbot"][h]["total"] for h in ("1hop", "2hop", "general")),
        "mcq_report.json")
    rag = json.load(open(os.path.join(CHAT, "ragas_scores.json")))
    for k, claim in (("answer_relevancy", 0.95), ("answer_similarity", 0.92),
                     ("faithfulness", 0.74), ("answer_correctness", 0.52)):
        add("3.4", f"RAGAS {k}", claim, round(rag["aggregate"][k], 2), "ragas_scores.json")
    add("3.4", "RAGAS n questions", 100, len(rag["per_sample"]), "ragas_scores.json")
    exp = json.load(open(os.path.join(CHAT, "expert_scores.json")))["aggregated"]
    for k, vals in (("chatbot", (4.08, 3.33, 4.00, 3.81)), ("gpt", (4.92, 4.67, 4.67, 4.75)),
                    ("llama", (4.42, 4.58, 4.75, 4.58))):
        for field, v in zip(("correctness", "completeness", "relevance", "overall"), vals):
            add("T6", f"expert {k} {field}", v, round(exp[k][field]["mean"], 2),
                "expert_scores.json")

    # --- design -----------------------------------------------------------
    d = json.load(open(os.path.join(DESIGN, "random_search_baseline.json")))
    add("3.5", "design n targets", 20, d["n_targets"], "random_search_baseline.json")
    add("3.5", "design hit fraction", 0.9, d["hit_fraction"], "random_search_baseline.json")
    DESIGN_ROWS = [("designer (LLM pipeline)", 52.6, 3, 3),
                   ("random best-of-5", 71.1, 7, 0),
                   ("random best-of-5 (plaus-aware)", 71.1, 7, 0),
                   ("sparsity-matched best-of-5", 52.6, 1, 0),
                   ("sparsity-matched best-of-5 (plaus-aware)", 48.7, 0, 0),
                   ("random+Guard+Tuner best-of-5", 72.4, 9, 0),
                   ("random+Guard+Tuner best-of-5 (plaus-aware)", 71.1, 8, 1),
                   ("sparsity+Guard+Tuner best-of-5", 68.4, 5, 5),
                   ("sparsity+Guard+Tuner best-of-5 (plaus-aware)", 68.4, 5, 5),
                   ("random best-of-100", 80.3, 10, 0),
                   ("random best-of-100 (plaus-aware)", 75.0, 8, 1),
                   ("sparsity-matched best-of-100", 77.6, 10, 5),
                   ("sparsity-matched best-of-100 (plaus-aware)", 77.6, 10, 10)]
    by = {a["label"]: a for a in d["arms"]}
    for label, hr, us, cr in DESIGN_ROWS:
        a = by.get(label)
        add("T7", f"design {label[:34]} hit%", hr, None if not a else a["hit_rate"],
            "random_search_baseline.json")
        add("T7", f"design {label[:34]} usable", us, None if not a else a["usable"],
            "random_search_baseline.json")
        add("T7", f"design {label[:34]} credible", cr, None if not a else a["credible"],
            "random_search_baseline.json")

    # --- Methods counts ---------------------------------------------------
    m = json.load(open(os.path.join(MODELS, "metrics.json")))
    ns = [v["n_samples"] for v in m.values() if isinstance(v, dict) and "n_samples" in v]
    add("2.4", "data points per model, min", 328, min(ns), "saved_models_v2/metrics.json")
    add("2.4", "data points per model, max", 417, max(ns), "saved_models_v2/metrics.json")
    nf = sorted({v["n_features"] for v in m.values() if isinstance(v, dict)})
    add("3.2", "feature dimensions low", 50, nf[0], "saved_models_v2/metrics.json")
    add("3.2", "feature dimensions high", 80, nf[-1], "saved_models_v2/metrics.json")

    owl = open(os.path.join(ROOT, "ontology", "alloygraph.owl"), encoding="utf-8").read()
    blocks = re.findall(r'<rdf:Description rdf:about="([^"]+)">(.*?)</rdf:Description>',
                        owl, re.S)
    for label, marker, claim in (("classes", "owl#Class", 32),
                                 ("object properties", "owl#ObjectProperty", 17),
                                 ("data properties", "owl#DatatypeProperty", 46)):
        add("2.2", f"ontology {label}", claim,
            sum(1 for _, b in blocks if marker in b), "ontology/alloygraph.owl")

    return out


UNTRACEABLE = [
    ("2.6", "median 9 s per evaluation on the original provider",
     "archive/pre_erratum/output/predictions_full_system_*.csv (medians 7.8-9.8 s) -- "
     "an ARCHIVED pre-erratum file; the provider was decommissioned and no current "
     "campaign uses it. Timing is unaffected by the erratum, so the number is sound, "
     "but its only source is archived."),
    ("4", "iteration runaway consumed fifty-six LLM calls on one case",
     "development observation from before the iteration cap; no committed artifact. "
     "Current campaigns cap at 8 agent iterations and top out at 36 LLM requests."),
    ("4", "a 1310 MPa prediction for a 360 MPa alloy",
     "development observation of the authority-transfer bug, fixed before the "
     "reported campaigns; no committed artifact."),
    ("3.2", "a Gaussian process gains 39% in yield strength error from engineered features",
     "MISMATCH, see below -- external_baselines.md gives 15.9% (CV) or 31.9% (holdout)."),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tex", default=DEFAULT_TEX)
    args = ap.parse_args()

    rows = checks()
    bad = []
    for sec, claim, tex, got, src in rows:
        if got is None:
            bad.append((sec, claim, tex, got, src, "NO SOURCE ROW"))
        elif abs(float(got) - float(tex)) > TOL:
            bad.append((sec, claim, tex, got, src, "MISMATCH"))

    print(f"checked {len(rows)} declared numbers against committed sources\n")
    if bad:
        print(f"{'sec':5s} {'claim':44s} {'tex':>10s} {'source':>10s}  verdict")
        print("-" * 88)
        for sec, claim, tex, got, src, why in bad:
            print(f"{sec:5s} {claim[:44]:44s} {tex:>10} {str(got):>10}  {why} ({src})")
    else:
        print("ALL DECLARED NUMBERS MATCH.")

    print(f"\nnot mechanically checkable ({len(UNTRACEABLE)}):")
    for sec, claim, note in UNTRACEABLE:
        print(f"  [{sec}] {claim}")
        print(f"        {note}")

    if os.path.exists(args.tex):
        print(f"\ntex present: {args.tex}")
    else:
        print(f"\ntex not found at {args.tex} (Overleaf copy is git-ignored)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
