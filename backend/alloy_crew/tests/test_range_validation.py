"""The ingestion range check: what it catches, and what it provably cannot."""

import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(ROOT, "backend"))

from pipeline.range_validation import FEATURE_RANGES, check_features  # noqa: E402
from alloy_crew.models.feature_engineering import compute_alloy_features  # noqa: E402

TRAINING = os.path.join(ROOT, "backend", "alloy_crew", "models", "training_data")
EVAL = os.path.join(ROOT, "evaluation", "prediction", "data")


def scan(path):
    hits = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            if not (rec.get("composition") or {}):
                continue
            try:
                feats = compute_alloy_features(rec["composition"])
            except Exception:
                continue
            hits.extend(check_features(rec["alloy"], feats))
    return hits


def test_ranges_match_the_ontology_declarations():
    """The table here must not drift from what build_ontology writes into the OWL."""
    src = open(os.path.join(ROOT, "backend", "pipeline", "build_ontology.py"),
               encoding="utf-8").read()
    for lo, hi, prop in FEATURE_RANGES.values():
        assert prop in src, f"{prop} is not declared in build_ontology.py"
        assert f"min_inclusive={lo}" in src, f"{prop}: min {lo} not in build_ontology.py"
        assert f"max_inclusive={hi}" in src, f"{prop}: max {hi} not in build_ontology.py"


def test_catches_the_three_upstream_defects():
    """Uncorrected upstream: exactly the two closure errata and the PE16 erratum.

    CMSX-2 and CMSX-3 carry the nickel content that leaves their compositions at
    67 wt% (ledger T01/T02) and NIMONIC PE16 the transposed titanium (E01). All
    three surface as Md_avg above the declared 1.05.
    """
    hits = scan(os.path.join(TRAINING, "final_alloy_data_enriched_v2.jsonl"))
    assert {h["alloy"] for h in hits} == {"CMSX-2*(SC)", "CMSX-3*(SC)", "NIMONIC* PE16"}
    assert all(h["feature"] == "Md_avg" for h in hits)


def test_corrected_corpora_are_clean():
    """Nothing fires on the data the published results were computed from."""
    for path in (os.path.join(TRAINING, "train_77alloys_v2.jsonl"),
                 os.path.join(EVAL, "SSS.jsonl"),
                 os.path.join(EVAL, "precip.jsonl"),
                 os.path.join(EVAL, "sc_ds.jsonl")):
        assert scan(path) == [], f"unexpected range violation in {os.path.basename(path)}"


def test_cannot_reach_erratum_e02():
    """E02 is a measurement, and no declared range covers measurements.

    Documents the limit rather than leaving it to be discovered: the ontology
    constrains four computed variant features, none of them a measured strength,
    so the -435 MPa yield strength on RGT* 13 is invisible here by construction.
    data_sanity_sweep.py is the instrument that sees it.
    """
    assert not any("Strength" in f or "strength" in f for f in FEATURE_RANGES)
    # A variant whose measured yield strength is impossible still passes.
    comp = {"Ni": 55.0, "Cr": 20.0, "Co": 13.0, "Mo": 4.0, "Ti": 2.4, "Al": 1.4}
    assert check_features("synthetic", compute_alloy_features(comp)) == []
