#!/usr/bin/env python3
"""
γ/γ' Lattice Mismatch (δ) Validation Against Literature Values

Companion to gamma_prime_validation.py. Guards the lattice mismatch
computation in feature_engineering.py, which had no validation of any kind
until this script was added.

WHY THIS EXISTS
---------------
Commit 62d6ef6 normalised the γ and γ' phase compositions to sum to 100%.
That was correct in itself, but it exposed a latent bug: Ni was missing from
LATTICE_COEFFS and picked up the dict's 0.1 fallback, giving the reference
element a spurious expansion coefficient. Because γ' is Ni₃X (~67-75 at% Ni)
and γ is ~54 at% Ni, the spurious term did not cancel between phases — it
added roughly +0.36% to δ on every γ'-strengthened alloy and tripled the mean
error against literature. No test caught it.

ON THE LITERATURE VALUES
------------------------
δ is not a single number per alloy. It varies with temperature (γ and γ' have
different thermal expansion coefficients, so δ typically becomes more negative
on heating), with heat treatment, and with γ' particle size. Reported values
for the same alloy commonly spread by ±0.05-0.10%.

The targets below are therefore nominal room-temperature values with an
explicit uncertainty band, and scoring uses that band rather than pretending
to a point measurement. Each entry carries a confidence tag:

  high   - sign and approximate magnitude are well established in the
           superalloy literature and consistent across sources
  medium - sign is well established, magnitude less tightly constrained

Do not treat these as single-source measurements. Before publishing any
number derived from this script, verify the targets against the primary
sources for the specific alloys used.
"""

import argparse
import json
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from alloy_crew.models.feature_engineering import (  # noqa: E402
    LATTICE_COEFFS,
    calculate_lattice_mismatch,
    estimate_gamma_prime_vol_pct,
    estimate_partitioning,
    wt_to_at_percent,
)

# =============================================================================
# BENCHMARK SET
# delta_lit: nominal RT γ/γ' mismatch (%).  band: +/- tolerance on that value.
# =============================================================================

BENCHMARK = {
    "Waspaloy": {
        "delta_lit": +0.20, "band": 0.10, "confidence": "high",
        "note": "Wrought disc alloy, small positive misfit, spherical γ'.",
        "composition": {"Ni": 58.0, "Cr": 19.4, "Co": 13.5, "Mo": 4.3,
                        "Ti": 3.0, "Al": 1.4, "Fe": 0.4},
    },
    "IN718": {
        "delta_lit": +0.10, "band": 0.10, "confidence": "medium",
        "note": "γ'/γ misfit is small and positive, but IN718 is strengthened "
                "mainly by γ'' (Ni3Nb, DO22). This model estimates γ' only, so "
                "IN718 is a weak test of δ and is reported for completeness.",
        "composition": {"Ni": 52.5, "Cr": 19.0, "Fe": 18.5, "Mo": 3.0,
                        "Nb": 5.1, "Al": 0.5, "Ti": 0.9},
    },
    "CMSX-4": {
        "delta_lit": -0.25, "band": 0.10, "confidence": "high",
        "note": "2nd-gen SC. Negative misfit at RT is a defining feature of "
                "Re-bearing SC alloys — Re and W partition to γ and expand it.",
        "composition": {"Ni": 61.7, "Cr": 6.5, "Co": 9.0, "Mo": 0.6, "W": 6.0,
                        "Al": 5.6, "Ti": 1.0, "Ta": 6.5, "Re": 3.0, "Hf": 0.1},
    },
    "Rene N5": {
        "delta_lit": -0.20, "band": 0.10, "confidence": "high",
        "note": "2nd-gen SC, negative misfit, comparable to CMSX-4.",
        "composition": {"Ni": 63.1, "Cr": 7.0, "Co": 7.5, "Mo": 1.5, "W": 5.0,
                        "Al": 6.2, "Ta": 6.5, "Re": 3.0, "Hf": 0.15},
    },
    "CMSX-10": {
        "delta_lit": -0.30, "band": 0.12, "confidence": "medium",
        "note": "3rd-gen SC with ~6% Re. More negative than CMSX-4; the sign "
                "and the ordering vs CMSX-4 are the meaningful assertions.",
        "composition": {"Ni": 69.6, "Cr": 2.0, "Co": 3.0, "Mo": 0.4, "W": 5.0,
                        "Al": 5.7, "Ti": 0.2, "Ta": 8.0, "Re": 6.0, "Nb": 0.1},
    },
    "Udimet 720": {
        "delta_lit": +0.10, "band": 0.12, "confidence": "medium",
        "note": "Wrought disc alloy, near-zero to slightly positive misfit.",
        "composition": {"Ni": 55.0, "Cr": 16.0, "Co": 15.0, "Mo": 3.0,
                        "W": 1.25, "Al": 2.5, "Ti": 5.0},
    },
    "MAR-M 247": {
        "delta_lit": +0.10, "band": 0.12, "confidence": "medium",
        "note": "Cast/DS blade alloy, small positive misfit at RT.",
        "composition": {"Ni": 59.5, "Cr": 8.4, "Co": 10.0, "Mo": 0.7, "W": 10.0,
                        "Al": 5.5, "Ti": 1.0, "Ta": 3.0, "Hf": 1.5},
    },
}

# =============================================================================
# VARIANTS — each is (label, coefficient overrides, fallback for unlisted el)
# =============================================================================

VARIANTS = {
    "pre-fix": {
        "label": "pre-fix (Ni falls through to 0.1 default)",
        "overrides": {"Ni": 0.1},
        "fallback": 0.1,
    },
    "fixed": {
        "label": "fixed (Ni = 0.0, unlisted elements = 0.0)",
        "overrides": {},
        "fallback": 0.0,
    },
}


def lattice_parameter(comp_at, coeffs, fallback):
    """Mirror of feature_engineering.calculate_lattice_parameter, but with the
    coefficient table and fallback injectable so variants are comparable."""
    return 3.524 + sum(
        (amt / 100.0) * coeffs.get(el, fallback)
        for el, amt in comp_at.items()
    )


def delta_for(composition, coeffs, fallback):
    """Compute δ for one composition under a given coefficient table."""
    at = wt_to_at_percent(composition)
    gp = estimate_gamma_prime_vol_pct(at)
    c_gamma, c_gamma_prime = estimate_partitioning(at, gp)
    a_g = lattice_parameter(c_gamma, coeffs, fallback)
    a_gp = lattice_parameter(c_gamma_prime, coeffs, fallback)
    if a_g + a_gp == 0:
        return 0.0, gp
    return round(200.0 * (a_gp - a_g) / (a_gp + a_g), 3), gp


def build_coeffs(variant_key, extra_overrides=None):
    v = VARIANTS[variant_key]
    coeffs = dict(LATTICE_COEFFS)
    coeffs.update(v["overrides"])
    if extra_overrides:
        coeffs.update(extra_overrides)
    return coeffs, v["fallback"]


def run_variant(variant_key, extra_overrides=None, verbose=True):
    coeffs, fallback = build_coeffs(variant_key, extra_overrides)
    label = VARIANTS[variant_key]["label"]
    if extra_overrides:
        label += " + " + ", ".join(f"{k}={v}" for k, v in extra_overrides.items())

    if verbose:
        print("=" * 78)
        print(f"VARIANT: {label}")
        print("=" * 78)
        print(f"{'alloy':<13}{'lit':>8}{'band':>7}{'pred':>9}{'err':>8}"
              f"{'sign':>7}{'in-band':>9}  conf")
        print("-" * 78)

    results, abs_errs, sign_ok, in_band = [], [], 0, 0
    for name, d in BENCHMARK.items():
        lit, band = d["delta_lit"], d["band"]
        pred, gp = delta_for(d["composition"], coeffs, fallback)
        err = pred - lit
        s_ok = (pred > 0) == (lit > 0)
        b_ok = abs(err) <= band
        sign_ok += s_ok
        in_band += b_ok
        abs_errs.append(abs(err))
        if verbose:
            print(f"{name:<13}{lit:>+8.2f}{band:>7.2f}{pred:>+9.3f}{err:>+8.3f}"
                  f"{('OK' if s_ok else 'WRONG'):>7}{('yes' if b_ok else 'no'):>9}"
                  f"  {d['confidence']}")
        results.append({
            "alloy": name, "delta_lit": lit, "band": band,
            "delta_pred": pred, "error": round(err, 3),
            "sign_correct": s_ok, "within_band": b_ok,
            "gamma_prime_vol_pct": gp, "confidence": d["confidence"],
        })

    n = len(results)
    mae = sum(abs_errs) / n
    metrics = {
        "variant": label, "n_alloys": n, "mae": round(mae, 4),
        "sign_accuracy": f"{sign_ok}/{n}", "within_band": f"{in_band}/{n}",
    }
    if verbose:
        print("-" * 78)
        print(f"  MAE = {mae:.3f} %   sign correct = {sign_ok}/{n}   "
              f"within band = {in_band}/{n}")
        print()
    return results, metrics


def exposure(coeffs, fallback):
    """Count evaluation-set alloys crossing the two thresholds that matter:
      |δ| > 0.5  -> mismatch_boost = min(|δ|, 0.5)*100 saturates (no longer
                    discriminative in the physics YS estimate)
      |δ| > 0.8  -> designer Guard cuts Ti by 30%; quick_check flags CRITICAL
    """
    data_dir = PROJECT_ROOT / "evaluation" / "prediction" / "data"
    alloys = []
    for fn in ("precip.jsonl", "sc_ds.jsonl", "SSS.jsonl"):
        p = data_dir / fn
        if not p.exists():
            continue
        with open(p) as fh:
            for line in fh:
                if line.strip():
                    d = json.loads(line)
                    if d.get("composition"):
                        alloys.append((d.get("alloy", "?"), d["composition"]))
    n5 = n8 = 0
    mx = 0.0
    for _, comp in alloys:
        try:
            d, _ = delta_for(comp, coeffs, fallback)
        except Exception:
            continue
        n5 += abs(d) > 0.5
        n8 += abs(d) > 0.8
        mx = max(mx, abs(d))
    return len(alloys), n5, n8, mx


def main():
    ap = argparse.ArgumentParser(description="Validate γ/γ' lattice mismatch")
    ap.add_argument("--compare", action="store_true",
                    help="Run every variant side by side instead of just the "
                         "current code")
    ap.add_argument("--set-coeff", action="append", default=[], metavar="EL=VAL",
                    help="Override one Vegard coefficient, e.g. --set-coeff W=0.47. "
                         "For testing a literature correction WITHOUT editing the "
                         "model. Repeatable.")
    ap.add_argument("--max-mae", type=float, default=0.40,
                    help="Fail (exit 1) if the fixed variant exceeds this MAE")
    ap.add_argument("--json-out", type=str, default=None,
                    help="Write full results to this path")
    args = ap.parse_args()

    extra = {}
    for spec in args.set_coeff:
        el, _, val = spec.partition("=")
        extra[el.strip()] = float(val)

    print()
    print("γ/γ' LATTICE MISMATCH VALIDATION")
    print(f"benchmark: {len(BENCHMARK)} alloys with literature RT δ values")
    if extra:
        print(f"coefficient overrides: {extra}")
    print()

    keys = list(VARIANTS) if args.compare else ["fixed"]
    all_metrics, payload = [], {}
    for k in keys:
        results, metrics = run_variant(k, extra)
        all_metrics.append(metrics)
        payload[k] = {"results": results, "metrics": metrics}

    if args.compare:
        print("=" * 78)
        print("SUMMARY")
        print("=" * 78)
        print(f"{'variant':<48}{'MAE':>8}{'sign':>9}{'in-band':>10}")
        for m in all_metrics:
            print(f"{m['variant']:<48}{m['mae']:>8.3f}"
                  f"{m['sign_accuracy']:>9}{m['within_band']:>10}")
        print()

    coeffs, fallback = build_coeffs("fixed", extra)
    n, n5, n8, mx = exposure(coeffs, fallback)
    print("=" * 78)
    print(f"DOWNSTREAM EXPOSURE — {n}-alloy evaluation set (fixed variant)")
    print("=" * 78)
    print(f"  |δ| > 0.5 (mismatch_boost saturates) : {n5}/{n}")
    print(f"  |δ| > 0.8 (designer Guard / CRITICAL): {n8}/{n}")
    print(f"  max |δ|                              : {mx:.2f}")
    print()
    payload["exposure"] = {"n_alloys": n, "gt_0_5": n5, "gt_0_8": n8,
                           "max_abs_delta": round(mx, 3)}

    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"  results written to {args.json_out}")

    fixed_mae = next(m["mae"] for m in all_metrics if m["variant"].startswith("fixed"))
    if fixed_mae > args.max_mae:
        print(f"FAIL: MAE {fixed_mae:.3f} exceeds --max-mae {args.max_mae}")
        return 1
    print(f"PASS: MAE {fixed_mae:.3f} within --max-mae {args.max_mae}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
