"""Light-touch alloy composition optimizer (two-tier).

Tier 1 — Guard:
    Catches fundamentally broken compositions from the LLM and applies
    surgical fixes: wrong alloy class, missing strengtheners, Cr/Co out
    of range, Critical TCP, excessive γ', lattice mismatch.

Tier 2 — Tuner:
    Makes small (±3 wt%) adjustments to performance-sensitive elements
    (Al, Ti, Mo, W, Ta, Nb) to close property deficits against targets.
    Uses ML model predictions for sensitivity analysis — the same models
    that Phase 3 evaluates against — so the tuner sees the same deficits.
    Max 15 steps.  GP formers (Al/Ti/Ta/Nb) can be reduced for EL deficit
    only when YS+UTS already exceed targets (5% margin).

Philosophy:
    The LLM chooses the alloy architecture (class, element balance,
    strengthening strategy).  The optimizer only adjusts performance
    knobs within tight guardrails.  No overshoot factors, no fixed
    trace additions (LLM decides C/B/Zr), no rewriting of the
    composition.
"""

import logging
import math

from .config.alloy_parameters import (
    SSS, TCP_RANK,
    classify_tcp_risk, get_params, get_coeff_gp,
    get_temperature_factor, get_alloy_class, get_sss_physics_ys,
    get_em_temp_factor, compress_uts_ys_ratio,
)
from .models.feature_engineering import (
    compute_alloy_features,
    calculate_em_rule_of_mixtures, MD_VALUES,
)

logger = logging.getLogger(__name__)

# ── Hard physical bounds (wt%) ─────────────────────────────────────
ELEMENT_BOUNDS = {
    "Ni": (40.0, 75.0),
    "Cr": (5.0, 25.0),
    "Co": (0.0, 25.0),
    "Mo": (0.0, 10.0),
    "W":  (0.0, 12.0),
    "Al": (0.0, 7.0),
    "Ti": (0.0, 6.0),
    "Ta": (0.0, 12.0),
    "Nb": (0.0, 6.0),
    "Re": (0.0, 7.0),
    "Fe": (0.0, 20.0),
    "Hf": (0.0, 2.0),
    "C":  (0.0, 0.3),
    "B":  (0.0, 0.05),
    "Zr": (0.0, 0.3),
    "Ru": (0.0, 5.0),
    "V":  (0.0, 3.0),
    "Mn": (0.0, 2.0),
    "Si": (0.0, 1.0),
    "Cu": (0.0, 3.0),
}

# Wrought-specific caps for the tuner (guard has its own logic)
WROUGHT_TUNE_CAPS = {
    "Mo": 4.5,   # >5% → Critical TCP for most disc compositions
    "W":  4.0,   # 5% + high Cr → Critical TCP
    "Ta": 4.0,   # Max ~2% in wrought, 4% with margin
}

# ── Tuner settings ─────────────────────────────────────────────────
TUNABLE_ELEMENTS = {"Al", "Ti", "Mo", "W", "Ta", "Nb"}
MAX_TUNE_DEVIATION = 3.0    # ±3 wt% from LLM's original per element
MAX_TUNE_STEPS = 15
TUNE_STEP_SIZE = 0.5        # wt% per gradient step
SENSITIVITY_DELTA = 0.5     # wt% perturbation for finite differences
CONVERGENCE_TOL = 0.02      # 2% of target → converged


# ── Helpers ────────────────────────────────────────────────────────

def _normalise(composition: dict) -> dict:
    """Enforce physical element bounds and rebalance via Ni."""
    comp = {}
    for el, val in composition.items():
        if val <= 0.001 and el != "Ni":
            continue
        lo, hi = ELEMENT_BOUNDS.get(el, (0, 20))
        comp[el] = max(lo, min(hi, val))

    non_ni = sum(v for k, v in comp.items() if k != "Ni")
    ni_needed = 100.0 - non_ni
    ni_lo, ni_hi = ELEMENT_BOUNDS["Ni"]
    comp["Ni"] = max(ni_lo, min(ni_hi, ni_needed))

    actual_total = sum(comp.values())
    if abs(actual_total - 100.0) > 0.1:
        scale = 100.0 / actual_total if actual_total > 0 else 1.0
        comp = {k: v * scale for k, v in comp.items()}

    return {k: round(v, 4 if v < 0.1 else 2) for k, v in comp.items() if v > 0.001}


def _get_physics_predictions(composition: dict, temperature_c: int,
                             processing: str) -> dict:
    """Compute physics-based property predictions.

    Returns flat dict with Yield Strength, Tensile Strength, Elongation,
    Elastic Modulus, Gamma Prime.
    """
    features = compute_alloy_features(composition)
    alloy_class = get_alloy_class(composition, processing)
    params = get_params(processing)

    gp = features.get("gamma_prime_estimated_vol_pct", 0)
    sss_wt = features.get("SSS_total_wt_pct", 0)
    delta = features.get("lattice_mismatch_pct", 0)

    if alloy_class == "sss":
        physics_ys_rt, _ = get_sss_physics_ys(composition, processing)
        temp_factor = get_temperature_factor(temperature_c, "sss")
        physics_ys = physics_ys_rt * temp_factor

        if processing == "cast":
            uts_ys_ratio = SSS["UTS_YS_RATIO_TYPICAL_CAST"]
            physics_el = SSS["EL_TYPICAL_CAST"]
        else:
            uts_ys_ratio = SSS["UTS_YS_RATIO_TYPICAL_WROUGHT"]
            physics_el = SSS["EL_TYPICAL_WROUGHT"]

        if temperature_c > 500:
            t_excess = temperature_c - 500
            uts_ys_ratio = 1.3 + (uts_ys_ratio - 1.3) * math.exp(-t_excess / 300)

        if temperature_c > SSS["EL_TEMP_TRANSITION"]:
            delta_t = temperature_c - SSS["EL_TEMP_TRANSITION"]
            physics_el = min(65.0, physics_el * math.exp(SSS["EL_TEMP_FACTOR"] * delta_t))

        physics_uts = physics_ys * uts_ys_ratio
    else:
        from .config.alloy_parameters import GP_TEMP, UTS_YS_RATIO
        base_ni = params["BASE_NI"] + params.get("HALL_PETCH_BOOST", 0)
        sss_contribution = params["SSS_CONTRIBUTION_FACTOR"] * sss_wt
        coeff_gp = get_coeff_gp(processing, "standard")
        # Cap mismatch boost at 0.5%: above that coherency degrades
        # (consistent with estimate_physics_ys in quick_check_tool.py)
        mismatch_boost = min(abs(delta), 0.5) * 100.0

        physics_ys_rt = base_ni + sss_contribution + (coeff_gp * gp) + mismatch_boost

        ac = "sc_ds" if alloy_class == "sc_ds" else "gp"
        temp_factor = get_temperature_factor(temperature_c, ac, gp_fraction=gp if ac == "gp" else None)
        physics_ys = physics_ys_rt * temp_factor

        if processing in ["wrought", "forged"]:
            ratio = (UTS_YS_RATIO["WROUGHT_HIGH_GP_EXPECTED"]
                     if gp > 40 else UTS_YS_RATIO["WROUGHT_BASE"])
        else:
            ratio = UTS_YS_RATIO["CAST_BASE"] + (gp / 100) * UTS_YS_RATIO["CAST_GP_FACTOR"]

        ratio = compress_uts_ys_ratio(ratio, temperature_c)
        physics_uts = physics_ys * ratio

        base_el = params["BASE_DUCTILITY"]
        physics_el = max(params["MIN_ELONGATION"], base_el - 0.8 * gp)

        if temperature_c >= 650:
            delta_t = temperature_c - 650
            el_factor = 1.0 + GP_TEMP["EL_TEMP_FACTOR"] * delta_t
            physics_el = min(60.0, physics_el * el_factor)

    em_rt = calculate_em_rule_of_mixtures(composition)
    em_temp_factor = get_em_temp_factor(temperature_c)
    physics_em = em_rt * em_temp_factor

    return {
        "Yield Strength": round(physics_ys, 1),
        "Tensile Strength": round(physics_uts, 1),
        "Elongation": round(physics_el, 1),
        "Elastic Modulus": round(physics_em, 1),
        "Gamma Prime": round(gp, 1),
    }


def _get_ml_predictions(composition: dict, temperature_c: int,
                        processing: str) -> dict:
    """Get ML model predictions (same models as Phase 3 evaluation).

    Returns flat dict with Yield Strength, Tensile Strength, Elongation,
    Elastic Modulus, Gamma Prime.  YS/UTS/El/EM come from the trained ML
    models; Gamma Prime comes from physics (ML doesn't predict GP).
    """
    from .models.predictor import AlloyPredictor

    predictor = AlloyPredictor.get_shared_predictor()
    report_df = predictor.predict(
        composition_wt=composition,
        extra_params={"processing": processing},
        temperatures=[temperature_c],
    )

    row = report_df.iloc[0]

    # GP from physics (ML doesn't predict it)
    features = compute_alloy_features(composition)
    gp = features.get("gamma_prime_estimated_vol_pct", 0)

    return {
        "Yield Strength": round(float(row["ys"]), 1),
        "Tensile Strength": round(float(row["uts"]), 1),
        "Elongation": round(float(row["el"]), 1),
        "Elastic Modulus": round(float(row["em"]), 1),
        "Gamma Prime": round(gp, 1),
    }


def _get_blended_predictions(composition: dict, temperature_c: int,
                             processing: str) -> dict:
    """Blended ML + empirical predictions matching the evaluator's proposal system."""
    ml = _get_ml_predictions(composition, temperature_c, processing)
    features = compute_alloy_features(composition)
    gp = features.get("gamma_prime_estimated_vol_pct", 0)
    alloy_class = get_alloy_class(composition, processing)

    ml_ys = ml["Yield Strength"]
    ml_uts = ml["Tensile Strength"]
    ml_el = ml["Elongation"]
    ml_em = ml["Elastic Modulus"]

    if alloy_class == "sss":
        # SSS: 30% ML + 70% physics (matches Proposal 1 blend)
        physics_ys, _ = get_sss_physics_ys(composition, processing)
        temp_factor = get_temperature_factor(temperature_c, "sss")
        physics_ys *= temp_factor
        blended_ys = 0.30 * ml_ys + 0.70 * physics_ys

        ml_ratio = ml_uts / ml_ys if ml_ys > 0 else 1.5
        ml_ratio = max(1.0, min(ml_ratio, 1.8))
        ratio = compress_uts_ys_ratio(ml_ratio, temperature_c)
        blended_uts = blended_ys * ratio

        blended_el = ml_el  # SSS ductility: trust ML

    else:
        # γ' alloys: use empirical formula (matches Proposal 1B)
        if processing in ("wrought", "forged"):
            empirical_ys_rt = 520 + 13 * gp
        else:
            empirical_ys_rt = 400 + 10 * gp

        temp_factor = get_temperature_factor(temperature_c, "gp", gp_fraction=gp)
        empirical_ys = empirical_ys_rt * temp_factor

        # 20% ML + 80% empirical (same as evaluator's moderate blend)
        blended_ys = 0.20 * ml_ys + 0.80 * empirical_ys

        # UTS from blended YS × temperature-compressed ML ratio
        ml_ratio = ml_uts / ml_ys if ml_ys > 0 else 1.4
        ml_ratio = max(1.0, min(ml_ratio, 1.8))
        ratio = compress_uts_ys_ratio(ml_ratio, temperature_c)
        blended_uts = blended_ys * ratio

        # EL: blend ML with empirical (ML tends to overpredict for high-γ')
        if processing in ("wrought", "forged"):
            empirical_el = max(10.0, 28 - 0.28 * gp)
        else:
            empirical_el = max(4.0, 18 - 0.25 * gp)
        blended_el = 0.50 * ml_el + 0.50 * empirical_el

    # EM: always blend with Reuss bound (matches evaluator's EM enforcement)
    em_reuss = calculate_em_rule_of_mixtures(composition)
    em_temp_factor = get_em_temp_factor(temperature_c)
    physics_em = em_reuss * em_temp_factor
    blended_em = 0.30 * ml_em + 0.70 * physics_em

    return {
        "Yield Strength": round(blended_ys, 1),
        "Tensile Strength": round(blended_uts, 1),
        "Elongation": round(blended_el, 1),
        "Elastic Modulus": round(blended_em, 1),
        "Gamma Prime": round(gp, 1),
    }


def _compute_sensitivity(composition: dict, element: str, prop: str,
                         temperature_c: int, processing: str) -> float:
    """∂prop/∂element via central finite differences (blended predictions)."""
    delta = SENSITIVITY_DELTA
    current = composition.get(element, 0)

    comp_up = composition.copy()
    comp_up[element] = current + delta
    non_ni = sum(v for k, v in comp_up.items() if k != "Ni")
    comp_up["Ni"] = 100.0 - non_ni

    comp_down = composition.copy()
    comp_down[element] = max(0.0, current - delta)
    non_ni = sum(v for k, v in comp_down.items() if k != "Ni")
    comp_down["Ni"] = 100.0 - non_ni

    pred_up = _get_blended_predictions(comp_up, temperature_c, processing)
    pred_down = _get_blended_predictions(comp_down, temperature_c, processing)

    actual_delta = (current + delta) - max(0.0, current - delta)
    if actual_delta < 0.01:
        return 0.0
    return (pred_up.get(prop, 0) - pred_down.get(prop, 0)) / actual_delta


# ══════════════════════════════════════════════════════════════════
# TIER 1: GUARD — catch and fix fundamentally broken compositions
# ══════════════════════════════════════════════════════════════════

def _guard(
    composition: dict,
    targets: dict,
    processing: str,
    temperature_c: int,
) -> dict:
    """Surgical fixes for compositions that are fundamentally wrong.

    Checks (in order):
      1. Wrong alloy class for target strength
      2. Missing critical strengtheners for high-strength wrought targets
      3. Cr/Co out of range for processing route
      4. Excess Fe (dead weight for disc alloys)
      5. Critical TCP risk
      6. Excessive γ' for wrought processing
      7. Excessive lattice mismatch

    Returns:
        dict with 'composition', 'fixes' (list of descriptions)
    """
    comp = composition.copy()
    fixes = []

    ys_target = targets.get("Yield Strength", 0)
    alloy_class = get_alloy_class(comp, processing)

    al = comp.get("Al", 0)
    ti = comp.get("Ti", 0)
    ta = comp.get("Ta", 0)
    nb = comp.get("Nb", 0)
    gp_formers = al + ti + ta + 0.35 * nb

    # ── 1. Wrong alloy class ─────────────────────────────────────
    # YS > 900 is impossible with SSS (tops out ~500 MPa at RT).
    # YS 600-900: either SSS or moderate-γ' could work — don't force.
    if ys_target > 900 and alloy_class == "sss" and gp_formers < 2.0:
        if al < 2.0:
            comp["Al"] = 3.0
            fixes.append(f"Al {al:.1f}→3.0% (need γ' for YS>{ys_target})")
        if ti < 1.0:
            comp["Ti"] = 2.0
            fixes.append(f"Ti {ti:.1f}→2.0% (need γ' for YS>{ys_target})")

    # ── 2. Missing strengtheners for high-strength wrought ───────
    # Only seed elements if the LLM's composition is truly deficient
    # in that category.  Don't add on top of adequate strengthening.
    if ys_target > 900 and processing in ("wrought", "forged"):
        mo_w_total = comp.get("Mo", 0) + comp.get("W", 0)
        nb_ta_total = comp.get("Nb", 0) + comp.get("Ta", 0)

        # SSS strengtheners: need at least ~3% combined Mo+W
        if mo_w_total < 3.0:
            if comp.get("Mo", 0) < 1.0:
                old_mo = comp.get("Mo", 0)
                comp["Mo"] = 3.0
                fixes.append(f"Mo {old_mo:.1f}→3.0% (disc alloy SSS)")
            if comp.get("W", 0) < 0.5:
                old_w = comp.get("W", 0)
                comp["W"] = 2.0
                fixes.append(f"W {old_w:.1f}→2.0% (disc alloy SSS)")

        # Secondary γ' formers: need at least ~1% combined Nb+Ta
        if nb_ta_total < 1.0 and gp_formers < 6.0:
            if comp.get("Nb", 0) < 0.3:
                comp["Nb"] = 1.5
                fixes.append("Nb→1.5% (disc alloy γ' former)")
            if comp.get("Ta", 0) < 0.3:
                comp["Ta"] = 1.0
                fixes.append("Ta→1.0% (modern disc alloy addition)")

    # ── 3. Cr/Co/refractory range for wrought ───────────────────
    if processing in ("wrought", "forged"):
        cr = comp.get("Cr", 0)
        # Cr>16% + Mo/W>4% pushes TCP to Elevated/Critical for disc alloys.
        # High-Cr alloys only work when Mo+W is very low (<5%).
        # Disc alloys with Mo+W>5% need Cr≤14% for Low TCP.
        if cr >= 18:
            comp["Cr"] = 14.0
            fixes.append(f"Cr {cr:.1f}→14.0% (TCP risk: Cr>18% + refractory)")
        elif cr < 8:
            comp["Cr"] = 12.0
            fixes.append(f"Cr {cr:.1f}→12.0% (oxidation resistance)")

        co = comp.get("Co", 0)
        if co < 15.0:
            comp["Co"] = 15.0
            fixes.append(f"Co {co:.1f}→15.0% (modern wrought disc minimum)")

        # Ta for η phase suppression — modern wrought disc alloys need
        # ≥1% Ta (typically 1.5-2% in production alloys).  Independent of Nb+Ta total
        # check above because Ta serves a distinct role (η suppression)
        # beyond secondary γ' formation.
        ta = comp.get("Ta", 0)
        if ta < 0.5:
            comp["Ta"] = 1.5
            fixes.append(f"Ta {ta:.1f}→1.5% (η suppression, modern disc alloy)")

        # Wrought refractory caps — enforce before TCP check so that
        # the TCP fix loop doesn't waste attempts on smaller elements
        # while Mo=9% and W=5% dominate Md.
        for el, cap, reason in [
            ("Mo", 4.5, "wrought Mo limit"),
            ("W", 4.0, "wrought W limit"),
            ("Ta", 4.0, "wrought Ta limit"),
            ("Re", 2.0, "wrought Re limit"),
        ]:
            val = comp.get(el, 0)
            if val > cap:
                comp[el] = cap
                fixes.append(f"{el} {val:.1f}→{cap}% ({reason})")

    # ── 4. Fe dead weight for disc alloys ────────────────────────
    if ys_target > 900 and comp.get("Fe", 0) > 2.0:
        old_fe = comp["Fe"]
        comp["Fe"] = max(0.0, old_fe - 3.0)
        fixes.append(f"Fe {old_fe:.1f}→{comp['Fe']:.1f}% (free budget)")

    # Rebalance Ni after structural fixes
    if fixes:
        comp = _normalise(comp)

    # ── 5. Critical/Elevated TCP ─────────────────────────────────
    # Reduce TCP risk to Low or Moderate.  Cr-first strategy:
    #
    # Per-1% Md_avg reduction: Mo≈0.0062, W≈0.0049, Cr≈0.0042.
    # These are close, but Mo+W provide critical SSS strengthening
    # (~15-20 MPa YS per 1%) while Cr above ~10% has minimal YS
    # impact.  So we prefer reducing Cr first (floor=10%) to
    # preserve Mo+W strengthening budget.
    #
    # Cr floor = 10%: adequate oxidation resistance for wrought disc
    # alloys.  Below 10% risks hot corrosion in turbine environments.
    CR_TCP_FLOOR = 10.0
    for attempt in range(7):
        features = compute_alloy_features(comp)
        md_avg = features.get("Md_avg", 0)
        md_gamma = features.get("Md_gamma", 0)
        tcp_level = classify_tcp_risk(md_gamma, md_avg)

        if tcp_level in ("Low", "Moderate"):
            break

        # Strategy 1: Reduce Cr first (preserves Mo/W strengthening).
        # Each 1% Cr reduction saves ~0.004 Md_avg.  Reducing Cr from
        # 15→10% saves 0.020 — often enough to drop Elevated→Moderate.
        cr_val = comp.get("Cr", 0)
        if cr_val > CR_TCP_FLOOR + 0.5:
            new_cr = max(CR_TCP_FLOOR, cr_val - 2.0)
            fixes.append(f"TCP fix: Cr {cr_val:.1f}→{new_cr:.1f}% (Cr-first, floor={CR_TCP_FLOOR}%)")
            comp["Cr"] = new_cr
            comp = _normalise(comp)
            continue

        # Strategy 2: Cr exhausted — fall back to highest-Md element.
        # Protect Ta ≤ 2%: high Md (2.224) but negligible bulk impact at
        # small amounts.  Ta is critical for η suppression in modern alloys.
        # Protect Nb ≤ 1.5%: Md=2.117 but at ≤1.5% bulk impact is ~+0.042,
        # and Nb is a valuable γ' former + strengthener.  Aligned with agent
        # backstory limit (Nb ≤ 1.5% for wrought).
        best_el, best_md = None, 0
        for el in ["Re", "W", "Mo", "Ta", "Nb"]:
            if comp.get(el, 0) > 0.5:
                if el == "Ta" and comp.get("Ta", 0) <= 2.0:
                    continue
                if el == "Nb" and comp.get("Nb", 0) <= 1.5:
                    continue
                el_md = MD_VALUES.get(el, 0)
                if el_md > best_md:
                    best_md = el_md
                    best_el = el

        if best_el is None:
            break

        old_val = comp[best_el]
        reduction = min(1.0, old_val * 0.3)
        comp[best_el] = max(0.0, old_val - reduction)
        fixes.append(
            f"TCP fix: {best_el} {old_val:.1f}→{comp[best_el]:.1f}% "
            f"(Md={best_md:.3f}, attempt {attempt + 1})"
        )
        comp = _normalise(comp)

    # ── 6. γ' > 50% for wrought ─────────────────────────────────
    # Loop until GP ≤ 50% — a single 0.80 scale pass can't fix GP=85%.
    if processing in ("wrought", "forged"):
        for _gp_pass in range(5):  # max 5 passes (0.80^5 = 0.33× at most)
            features = compute_alloy_features(comp)
            gp = features.get("gamma_prime_estimated_vol_pct", 0)
            if gp <= 50:
                break
            excess_ratio = (gp - 45.0) / max(gp, 1)
            scale = max(0.80, 1.0 - excess_ratio)
            for el in ["Al", "Ti", "Ta", "Nb"]:
                if comp.get(el, 0) > 0.3:
                    old = comp[el]
                    comp[el] = round(old * scale, 2)
            fixes.append(f"GP fix: γ'={gp:.0f}%>50%, scaled formers by {scale:.2f}")
            comp = _normalise(comp)

    # ── 6b. γ' > 60% for cast ────────────────────────────────────
    elif processing == "cast":
        for _gp_pass in range(5):
            features = compute_alloy_features(comp)
            gp = features.get("gamma_prime_estimated_vol_pct", 0)
            if gp <= 60:
                break
            excess_ratio = (gp - 55.0) / max(gp, 1)
            scale = max(0.80, 1.0 - excess_ratio)
            for el in ["Al", "Ti", "Ta", "Nb"]:
                if comp.get(el, 0) > 0.3:
                    old = comp[el]
                    comp[el] = round(old * scale, 2)
            fixes.append(f"GP fix: γ'={gp:.0f}%>60% cast, scaled formers by {scale:.2f}")
            comp = _normalise(comp)

    # ── 7. Lattice mismatch > 0.8% ──────────────────────────────
    features = compute_alloy_features(comp)
    delta = features.get("lattice_mismatch_pct", 0)
    if abs(delta) > 0.8 and comp.get("Ti", 0) > 1.0:
        old_ti = comp["Ti"]
        comp["Ti"] = round(max(1.0, old_ti * 0.7), 2)
        fixes.append(f"Mismatch fix: Ti {old_ti:.1f}→{comp['Ti']:.1f}% (δ={delta:.2f}%)")
        comp = _normalise(comp)

    if fixes:
        logger.info(f"Guard applied {len(fixes)} fixes: {fixes}")

    return {"composition": comp, "fixes": fixes}


# ══════════════════════════════════════════════════════════════════
# TIER 2: TUNER — light-touch performance adjustments
# ══════════════════════════════════════════════════════════════════

def _tune(
    composition: dict,
    original_composition: dict,
    targets: dict,
    temperature_c: int,
    processing: str,
    max_steps: int = MAX_TUNE_STEPS,
) -> dict:
    """Fine-tune performance-sensitive elements within ±2% of LLM values.

    Only adjusts elements in TUNABLE_ELEMENTS that are already present
    (> 0.1 wt%) in the composition.  Uses ML model sensitivity (same
    models as Phase 3) to pick the best element and direction for each
    deficit — so the tuner sees the same gaps that the evaluator reports.

    Args:
        composition: Current composition (after guard fixes).
        original_composition: LLM's original composition (for deviation limits).
        targets: Target properties dict.
        temperature_c: Service temperature (°C).
        processing: "cast" or "wrought".
        max_steps: Maximum tuning iterations.

    Returns:
        dict with 'composition', 'predicted_properties', 'steps_used', 'log'.
    """
    comp = composition.copy()
    log = []

    # Build per-element bounds: ±2% from LLM's original, capped by
    # physical bounds and wrought processing caps.
    tune_bounds = {}
    for el in TUNABLE_ELEMENTS:
        current = comp.get(el, 0)
        if current < 0.1:
            continue  # Don't add new elements — that's the guard's job

        orig = original_composition.get(el, current)
        lo = max(0.0, orig - MAX_TUNE_DEVIATION)
        hi = orig + MAX_TUNE_DEVIATION

        # Physical bounds
        phys_lo, phys_hi = ELEMENT_BOUNDS.get(el, (0, 20))
        lo = max(lo, phys_lo)
        hi = min(hi, phys_hi)

        # Wrought processing caps
        if processing in ("wrought", "forged") and el in WROUGHT_TUNE_CAPS:
            hi = min(hi, WROUGHT_TUNE_CAPS[el])

        # If the guard REDUCED a GP former, cap the tuner's upper bound
        # at the guarded value to prevent undoing GP/mismatch/TCP fixes.
        if el in ("Al", "Ti", "Ta", "Nb"):
            orig_val = original_composition.get(el, 0)
            if current < orig_val - 0.05:  # guard reduced this element
                hi = min(hi, current)

        tune_bounds[el] = (lo, hi)

    if not tune_bounds:
        logger.info("Tuner: no tunable elements present, skipping")
        predicted = _get_blended_predictions(comp, temperature_c, processing)
        features = compute_alloy_features(comp)
        return _build_tune_result(comp, predicted, features, 0, log)

    # Track starting TCP rank — tuner must never worsen it.
    # The guard ensures TCP is at most Moderate; the tuner should not
    # undo the guard's work by increasing Mo/W for marginal EM gains.
    starting_features = compute_alloy_features(comp)
    starting_tcp = classify_tcp_risk(
        starting_features.get("Md_gamma", 0), starting_features.get("Md_avg", 0))
    starting_tcp_rank = TCP_RANK.get(starting_tcp, 4)
    logger.info(f"Tuner starting TCP: {starting_tcp} (rank {starting_tcp_rank})")

    for step in range(max_steps):
        predicted = _get_blended_predictions(comp, temperature_c, processing)

        # Compute property deficits using blended predictions.
        deficits = []
        for prop in ["Yield Strength", "Tensile Strength", "Elongation", "Elastic Modulus"]:
            target = targets.get(prop, 0)
            if target <= 0:
                continue
            actual = predicted.get(prop, 0)
            if actual < target * (1 - CONVERGENCE_TOL):  # 2% tolerance
                deficit_frac = (target - actual) / max(target, 1)
                deficits.append((prop, deficit_frac))

        # Log
        features_now = compute_alloy_features(comp)
        tcp_now = classify_tcp_risk(
            features_now.get("Md_gamma", 0), features_now.get("Md_avg", 0))
        step_info = {
            "step": step,
            "YS": predicted.get("Yield Strength", 0),
            "UTS": predicted.get("Tensile Strength", 0),
            "EL": predicted.get("Elongation", 0),
            "EM": predicted.get("Elastic Modulus", 0),
            "GP": predicted.get("Gamma Prime", 0),
            "Md_avg": round(features_now.get("Md_avg", 0), 3),
            "TCP": tcp_now,
            "deficits": len(deficits),
        }
        log.append(step_info)

        if step % 3 == 0:
            logger.info(
                f"Tune step {step}: YS={predicted.get('Yield Strength', 0):.0f}, "
                f"UTS={predicted.get('Tensile Strength', 0):.0f}, "
                f"GP={predicted.get('Gamma Prime', 0):.1f}%, "
                f"Md={features_now.get('Md_avg', 0):.3f} ({tcp_now}), "
                f"deficits={len(deficits)}"
            )

        if not deficits:
            logger.info(f"Tuner converged at step {step}")
            break

        # Sort by deficit magnitude (largest first)
        deficits.sort(key=lambda x: x[1], reverse=True)
        prop_to_fix, deficit_frac = deficits[0]

        # Find best tunable element to adjust (increase or decrease)
        candidates = []
        for el, (lo, hi) in tune_bounds.items():
            current = comp.get(el, 0)
            sens = _compute_sensitivity(comp, el, prop_to_fix, temperature_c, processing)

            if sens > 0.1:
                # Increasing this element helps
                headroom = hi - current
                if headroom > 0.05:
                    adjustment = min(TUNE_STEP_SIZE, headroom)
                    candidates.append((el, "increase", sens, adjustment))

            elif sens < -0.1:
                # Decreasing this element helps — but restrict GP formers.
                # The guard handles catastrophic GP reductions (γ'>50%,
                # mismatch).  For EM deficit: reducing Al (70 GPa, lowest
                # modulus) destroys 10-25% of GP for marginal EM gain — block.
                if el in ("Al", "Ti", "Ta", "Nb"):
                    if prop_to_fix != "Elongation":
                        continue
                    ys_ok = predicted.get("Yield Strength", 0) >= targets.get("Yield Strength", 0) * 1.05
                    uts_ok = predicted.get("Tensile Strength", 0) >= targets.get("Tensile Strength", 0) * 1.05
                    if not (ys_ok and uts_ok):
                        continue
                reducible = current - lo
                if reducible > 0.05:
                    adjustment = min(TUNE_STEP_SIZE, reducible)
                    candidates.append((el, "decrease", abs(sens), adjustment))

        if not candidates:
            logger.info(
                f"Tuner: no adjustable elements for {prop_to_fix} "
                f"(deficit={deficit_frac:.1%}) at step {step}"
            )
            break

        # Pick candidate with highest impact, but reject if it worsens TCP.
        # Try candidates in order until one passes the TCP check.
        candidates.sort(key=lambda x: x[2] * x[3], reverse=True)
        applied = False
        for el, direction, sens, adjustment in candidates:
            if adjustment < 0.02:
                continue  # Too small to matter

            old_val = comp.get(el, 0)
            if direction == "increase":
                comp[el] = old_val + adjustment
            else:
                comp[el] = max(0.0, old_val - adjustment)

            # Rebalance Ni
            non_ni = sum(v for k, v in comp.items() if k != "Ni")
            comp["Ni"] = 100.0 - non_ni

            # TCP constraint: never worsen TCP beyond starting level
            features_check = compute_alloy_features(comp)
            tcp_check = classify_tcp_risk(
                features_check.get("Md_gamma", 0), features_check.get("Md_avg", 0))
            tcp_check_rank = TCP_RANK.get(tcp_check, 4)
            if tcp_check_rank > starting_tcp_rank:
                # Undo — this step would worsen TCP
                comp[el] = old_val
                non_ni = sum(v for k, v in comp.items() if k != "Ni")
                comp["Ni"] = 100.0 - non_ni
                logger.debug(
                    f"Tune: rejected {el} {direction} {adjustment:.2f}% — "
                    f"would worsen TCP ({starting_tcp}→{tcp_check})"
                )
                continue

            applied = True
            logger.debug(
                f"Tune: {el} {direction} {adjustment:.2f}% "
                f"({old_val:.2f}→{comp[el]:.2f}) for {prop_to_fix} "
                f"(sens={sens:.2f}, deficit={deficit_frac:.1%})"
            )
            break

        if not applied:
            logger.info(
                f"Tuner: all candidates rejected (TCP constraint) for "
                f"{prop_to_fix} at step {step}"
            )
            break

    # Final state
    comp = {k: round(v, 4 if v < 0.1 else 2) for k, v in comp.items() if v > 0.001}
    final_pred = _get_blended_predictions(comp, temperature_c, processing)
    features = compute_alloy_features(comp)

    return _build_tune_result(comp, final_pred, features, len(log), log)


def _build_tune_result(comp, predicted, features, steps_used, log):
    """Build the tuner result dict."""
    return {
        "composition": comp,
        "predicted_properties": {
            "Yield Strength": predicted.get("Yield Strength", 0),
            "Tensile Strength": predicted.get("Tensile Strength", 0),
            "Elongation": predicted.get("Elongation", 0),
            "Elastic Modulus": predicted.get("Elastic Modulus", 0),
            "Gamma Prime": predicted.get("Gamma Prime", 0),
            "Density": round(features.get("density_calculated_gcm3", 8.0), 2),
        },
        "steps_used": steps_used,
        "log": log,
    }


def _check_converged(predicted: dict, targets: dict) -> bool:
    """Check if all target properties are met within convergence tolerance."""
    for prop in ["Yield Strength", "Tensile Strength", "Elongation", "Elastic Modulus"]:
        target = targets.get(prop, 0)
        if target <= 0:
            continue
        actual = predicted.get(prop, 0)
        if actual < target * (1 - CONVERGENCE_TOL):
            return False
    return True


# ══════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════

def optimize(
    initial_composition: dict,
    targets: dict,
    temperature_c: int = 900,
    processing: str = "cast",
    max_steps: int = MAX_TUNE_STEPS,
) -> dict:
    """Two-tier optimization: Guard + Tune.

    Same API as the previous heavy optimizer for backward compatibility.

    Args:
        initial_composition: LLM's composition (wt%, should sum to ~100).
        targets: Target properties, e.g. {"Yield Strength": 1000}.
        temperature_c: Service temperature (°C).
        processing: "cast" or "wrought".
        max_steps: Max tuning steps (default 10).

    Returns:
        dict with composition, predicted_properties, features, tcp_risk,
        converged, steps_used, optimization_log, guard_fixes.
    """
    original_comp = initial_composition.copy()

    # Tier 1: Guard — fix fundamentally broken compositions
    guard_result = _guard(initial_composition, targets, processing, temperature_c)
    guarded_comp = guard_result["composition"]

    # Tier 2: Tune — light-touch adjustments within ±2%
    tune_result = _tune(
        composition=guarded_comp,
        original_composition=original_comp,
        targets=targets,
        temperature_c=temperature_c,
        processing=processing,
        max_steps=max_steps,
    )

    final_comp = tune_result["composition"]
    features = compute_alloy_features(final_comp)
    tcp_level = classify_tcp_risk(
        features.get("Md_gamma", 0), features.get("Md_avg", 0))

    return {
        "composition": final_comp,
        "predicted_properties": tune_result["predicted_properties"],
        "features": {
            "Md_avg": round(features.get("Md_avg", 0), 3),
            "Md_gamma": round(features.get("Md_gamma", 0), 3),
            "lattice_mismatch_pct": round(features.get("lattice_mismatch_pct", 0), 3),
            "gamma_prime_estimated_vol_pct": round(
                features.get("gamma_prime_estimated_vol_pct", 0), 1),
            "density_calculated_gcm3": round(
                features.get("density_calculated_gcm3", 8.0), 2),
        },
        "tcp_risk": tcp_level,
        "converged": _check_converged(tune_result["predicted_properties"], targets),
        "steps_used": tune_result["steps_used"],
        "optimization_log": tune_result["log"],
        "guard_fixes": guard_result["fixes"],
    }
