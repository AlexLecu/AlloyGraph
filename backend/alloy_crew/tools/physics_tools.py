from typing import Any, Type
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import json
import logging

# Import SSS constants from metallurgy_tools (single source of truth)
from .metallurgy_tools import (
    SSS_ALLOY_AL_TI_TA_MAX,
    SSS_YS_MIN_RT,
    SSS_YS_MAX_RT,
    SSS_POTENCY,
    SSS_TEMP_TRANSITION,
    SSS_TEMP_DECAY_SLOW,
    SSS_TEMP_DECAY_FAST,
    SSS_TEMP_MIN_FACTOR,
)

logger = logging.getLogger(__name__)

# =============================================================================
# PHYSICS CORRECTION CONSTANTS
# =============================================================================

# Base YS and γ' coefficients
WROUGHT_BASE_YS = 180
CAST_BASE_YS = 150
WROUGHT_GP_COEFF = 17
CAST_GP_COEFF = 10

# Hall-Petch grain boundary strengthening
WROUGHT_HALL_PETCH = 80
CAST_HALL_PETCH = 25

# Atomic radii for misfit calculations (Å)
ATOMIC_RADII = {
    "Ni": 1.246, "Co": 1.251, "Fe": 1.241, "Cr": 1.249,
    "Mo": 1.363, "W": 1.370, "Re": 1.375, "V": 1.316,
    "Al": 1.432, "Ti": 1.462, "Ta": 1.430, "Nb": 1.429,
}
SSS_SCALING_FACTOR = 1200

# γ'' (Gamma Double Prime) detection thresholds
GAMMA_PP_NB_MIN = 3.0
GAMMA_PP_AL_TI_MAX = 3.0
GAMMA_PP_NB_RATIO_MIN = 1.0
GAMMA_PP_NB_SOLUBILITY = 1.5
GAMMA_PP_FORMATION_FACTOR = 6.0
GAMMA_PP_COEFF_WROUGHT = 30
GAMMA_PP_COEFF_CAST = 25

# Adaptive YS Tolerance Thresholds
YS_TOLERANCE_LOW = 200
YS_TOLERANCE_MEDIUM = 300
YS_TOLERANCE_HIGH = 400

# Suggested Value Factors (blending between ML and physics)
SUGGESTED_VALUE_FACTOR_LOW = 0.5
SUGGESTED_VALUE_FACTOR_MEDIUM = 0.7
SUGGESTED_VALUE_FACTOR_HIGH = 0.8

# UTS/YS Ratio Constants
EXPECTED_RATIO_BASE = 1.2
EXPECTED_RATIO_GP_FACTOR = 0.5

# UTS/YS Ratio Bounds (Low Confidence)
UTS_RATIO_MIN_TIGHT = 1.15
UTS_RATIO_MAX_TIGHT = 1.45
UTS_RATIO_MIN_OFFSET_TIGHT = 0.10
UTS_RATIO_MAX_OFFSET_TIGHT = 0.15

# UTS/YS Ratio Bounds (Medium Confidence)
UTS_RATIO_MIN_MEDIUM = 1.12
UTS_RATIO_MAX_MEDIUM = 1.48
UTS_RATIO_MIN_OFFSET_MEDIUM = 0.12
UTS_RATIO_MAX_OFFSET_MEDIUM = 0.18

# UTS/YS Ratio Bounds (High Confidence)
UTS_RATIO_MIN_LOOSE = 1.05
UTS_RATIO_MAX_LOOSE = 1.6
UTS_RATIO_MIN_OFFSET_LOOSE = 0.2
UTS_RATIO_MAX_OFFSET_LOOSE = 0.3

# SSS-specific UTS/YS ratio bounds
SSS_UTS_YS_RATIO_MIN = 1.5
SSS_UTS_YS_RATIO_MAX = 2.3
SSS_ELONGATION_FACTOR = 2.0
SSS_YS_TOLERANCE = 80

# Elastic Modulus Bounds
WROUGHT_EM_MIN = 200
WROUGHT_EM_MAX = 225
CAST_EM_MIN = 180
CAST_EM_MAX = 215

# EM Adjustments
EM_CO_FE_REDUCTION_MIN = 10
EM_CO_FE_REDUCTION_MAX = 5
EM_TIGHT_OFFSET_MIN = 10
EM_TIGHT_OFFSET_MAX = 5
EM_MEDIUM_WROUGHT_MIN = 200
EM_MEDIUM_WROUGHT_MAX = 223
EM_MEDIUM_CAST_MIN = 185
EM_MEDIUM_CAST_MAX = 213
EM_LOOSE_OFFSET = 10

# Strength-Ductility Tradeoff
YS_VERY_HIGH = 1300
YS_HIGH = 1000
YS_MODERATE = 700
MAX_EL_VERY_HIGH_STRENGTH = 10
MAX_EL_HIGH_STRENGTH = 15
MAX_EL_MODERATE_STRENGTH = 25
MAX_EL_LOW_STRENGTH = 40

# Confidence/Distance Thresholds
KG_DISTANCE_FAR = 10
KG_DISTANCE_MEDIUM = 5
KG_DISTANCE_CLOSE = 3.0
KG_DISTANCE_WEAK = 7.0

# Compositional Thresholds
CO_HIGH_THRESHOLD = 15
FE_HIGH_THRESHOLD = 10
RE_HIGH_THRESHOLD = 3.0
CO_VERY_HIGH_THRESHOLD = 20
FE_VERY_HIGH_THRESHOLD = 15

# Severity Thresholds
SEVERITY_HIGH_PCT_YS_OVER = 20
SEVERITY_HIGH_PCT_YS_UNDER = 15
SEVERITY_HIGH_PCT_UTS_OVER = 8
SEVERITY_HIGH_PCT_UTS_UNDER = 5
SEVERITY_HIGH_PCT_EM = 8
SEVERITY_MEDIUM_PCT_EL = 50


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _detect_gamma_double_prime(composition: dict) -> tuple[bool, float, str]:
    """Detect if alloy is γ'' (Ni₃Nb) strengthened vs γ' (Ni₃Al) strengthened."""
    nb = composition.get("Nb", 0)
    al = composition.get("Al", 0)
    ti = composition.get("Ti", 0)
    al_ti = al + ti

    is_gamma_pp = (
        nb >= GAMMA_PP_NB_MIN and
        al_ti <= GAMMA_PP_AL_TI_MAX and
        (nb / (al_ti + 0.01)) >= GAMMA_PP_NB_RATIO_MIN
    )

    if is_gamma_pp:
        excess_nb = max(0, nb - GAMMA_PP_NB_SOLUBILITY)
        gamma_pp_vol = min(excess_nb * GAMMA_PP_FORMATION_FACTOR, 25.0)
        phase_note = "γ'' (Ni₃Nb) dominated"
        return True, round(gamma_pp_vol, 1), phase_note
    else:
        phase_note = "γ' (Ni₃Al) dominated"
        return False, 0.0, phase_note


def _is_sss_alloy(composition: dict) -> tuple[bool, str]:
    """Detect if alloy is primarily solid solution strengthened (SSS)."""
    al = composition.get("Al", 0)
    ti = composition.get("Ti", 0)
    ta = composition.get("Ta", 0)
    al_ti_ta = al + ti + ta

    if al_ti_ta < SSS_ALLOY_AL_TI_TA_MAX:
        return True, f"Al+Ti+Ta={al_ti_ta:.1f}% < {SSS_ALLOY_AL_TI_TA_MAX}%"
    return False, f"Al+Ti+Ta={al_ti_ta:.1f}% (γ' dominated)"


def _sss_temperature_factor(temp_c: float) -> float:
    """Calculate temperature correction factor for SSS."""
    import math

    if temp_c <= 25.0:
        return 1.0

    if temp_c <= SSS_TEMP_TRANSITION:
        factor = 1.0 - SSS_TEMP_DECAY_SLOW * (temp_c - 25.0)
    else:
        factor_at_transition = 1.0 - SSS_TEMP_DECAY_SLOW * (SSS_TEMP_TRANSITION - 25.0)
        delta_t = temp_c - SSS_TEMP_TRANSITION
        factor = factor_at_transition * math.exp(-delta_t / SSS_TEMP_DECAY_FAST)

    return max(factor, SSS_TEMP_MIN_FACTOR)


def _calculate_sss_contribution(
    composition: dict,
    temp_c: float = 25.0
) -> tuple[float, dict, float]:
    """Calculate solid solution strengthening contribution using Labusch-Nabarro model."""
    r_ni = ATOMIC_RADII.get("Ni", 1.246)
    sss_total = 0.0
    breakdown = {}

    for element, wt_pct in composition.items():
        if element in SSS_POTENCY and wt_pct > 0.1:
            potency = SSS_POTENCY[element]
            r_el = ATOMIC_RADII.get(element, r_ni)

            # Atomic size misfit: δ = |r_el - r_Ni| / r_Ni
            delta_r = abs(r_el - r_ni) / r_ni

            # Labusch model: c^0.67 × δ^1.3 × potency
            concentration_factor = (wt_pct / 100.0) ** 0.67
            misfit_factor = delta_r ** 1.3
            contribution = concentration_factor * misfit_factor * potency * SSS_SCALING_FACTOR

            if contribution > 1.0:
                sss_total += contribution
                breakdown[element] = round(contribution, 1)

    temp_factor = _sss_temperature_factor(temp_c)
    sss_total_corrected = sss_total * temp_factor

    return round(sss_total_corrected, 1), breakdown, round(temp_factor, 3)


def _get_confidence_tier(confidence_level: str, kg_distance: float) -> str:
    """Determine confidence tier based on confidence level and KG match distance."""
    if confidence_level in ["LOW", "VERY LOW"] or kg_distance > KG_DISTANCE_FAR:
        return "LOW"
    elif confidence_level == "MEDIUM" or kg_distance > KG_DISTANCE_MEDIUM:
        return "MEDIUM"
    return "HIGH"


# =============================================================================
# PHYSICS CORRECTIONS PROPOSAL TOOL
# =============================================================================

class PhysicsCorrectionsProposalInput(BaseModel):
    """Input schema for PhysicsCorrectionsProposalTool."""
    properties_json: str = Field(..., description="JSON string of predicted properties (YS, UTS, Elongation, EM, Density, Gamma Prime)")
    composition_json: str = Field(..., description="JSON string of alloy composition (element wt%)")
    confidence_level: str = Field(..., description="Confidence level: HIGH, MEDIUM, LOW, or VERY LOW")
    processing: str = Field(default="wrought", description="Processing route: wrought, cast, or forged")
    kg_match_distance: float = Field(default=999.0, description="Distance to nearest KG match (999 = no match)")


class PhysicsCorrectionsProposalTool(BaseTool):
    name: str = "PhysicsCorrectionsProposalTool"
    description: str = (
        "Proposes physics-based corrections for predicted properties that violate known metallurgical relationships. "
        "Returns proposals with severity levels and reasoning - agent decides whether to apply them."
    )
    args_schema: Type[BaseModel] = PhysicsCorrectionsProposalInput

    def _run(
        self,
        properties_json: str,
        composition_json: str,
        confidence_level: str,
        processing: str = "wrought",
        kg_match_distance: float = 999.0,
        **kwargs: Any
    ) -> str:
        """Generate physics-based correction proposals."""
        try:
            properties = json.loads(properties_json)
            composition = json.loads(composition_json)
            proposals = []

            # Skip corrections for excellent KG matches
            if kg_match_distance < 1.0:
                return json.dumps({
                    "status": "SKIPPED",
                    "proposals": [],
                    "recommendation": "TRUST_KG",
                    "reasoning": f"Excellent KG match (distance={kg_match_distance:.2f}) - trusting experimental data over physics formula.",
                    "confidence_tier": "VERY_HIGH"
                })

            # Extract properties
            ys = properties.get("Yield Strength", 0)
            uts = properties.get("Tensile Strength", 0)
            el = properties.get("Elongation", 0)
            em = properties.get("Elastic Modulus", 0)
            gp = properties.get("Gamma Prime", 0)

            # Extract key composition elements
            co_content = composition.get("Co", 0)
            fe_content = composition.get("Fe", 0)
            re_content = composition.get("Re", 0)

            # Detect alloy type (SSS vs γ' dominated)
            is_sss, sss_reason = _is_sss_alloy(composition)

            # === YIELD STRENGTH CHECK ===
            if ys > 0:
                is_gamma_pp, gamma_pp_vol, phase_note = _detect_gamma_double_prime(composition)

                # Component 1: Base lattice strength
                if processing in ["wrought", "forged"]:
                    sigma_base = WROUGHT_BASE_YS
                    sigma_hp = WROUGHT_HALL_PETCH
                    processing_note = "wrought"
                    gp_coefficient = WROUGHT_GP_COEFF
                    gpp_coefficient = GAMMA_PP_COEFF_WROUGHT
                else:
                    sigma_base = CAST_BASE_YS
                    sigma_hp = CAST_HALL_PETCH
                    processing_note = "cast"
                    gp_coefficient = CAST_GP_COEFF
                    gpp_coefficient = GAMMA_PP_COEFF_CAST

                # Component 2: Precipitation strengthening (γ' or γ'')
                if is_gamma_pp:
                    sigma_ppt = gpp_coefficient * gamma_pp_vol
                    sigma_gp_minor = gp_coefficient * gp * 0.3 if gp > 0 else 0
                    sigma_ppt += sigma_gp_minor
                    ppt_label = "γ''"
                    ppt_detail = f"σ_γ''={gpp_coefficient}×{gamma_pp_vol:.1f}%"
                    if sigma_gp_minor > 0:
                        ppt_detail += f" + σ_γ'(minor)={sigma_gp_minor:.0f}"
                else:
                    sigma_ppt = gp_coefficient * gp if gp > 0 else 0
                    ppt_label = "γ'"
                    ppt_detail = f"σ_γ'={gp_coefficient}×{gp:.1f}%"

                # Component 3: Solid Solution Strengthening (SSS)
                sigma_sss, sss_breakdown, sss_temp_factor = _calculate_sss_contribution(composition)

                # Total physics-based YS estimate
                physics_ys = sigma_base + sigma_ppt + sigma_sss + sigma_hp

                strength_breakdown = {
                    "base_lattice": round(sigma_base, 0),
                    "precipitation": round(sigma_ppt, 0),
                    "precipitation_type": ppt_label,
                    "solid_solution": round(sigma_sss, 0),
                    "hall_petch": round(sigma_hp, 0),
                    "total": round(physics_ys, 0),
                    "sss_contributors": sss_breakdown,
                    "sss_temp_factor": sss_temp_factor,
                    "phase_type": phase_note,
                    "is_sss_alloy": is_sss,
                    "sss_classification": sss_reason
                }
                if is_gamma_pp:
                    strength_breakdown["gamma_pp_vol_pct"] = gamma_pp_vol

                tier = _get_confidence_tier(confidence_level, kg_match_distance)
                if tier == "LOW":
                    tolerance = YS_TOLERANCE_LOW
                    suggested_value_factor = SUGGESTED_VALUE_FACTOR_LOW
                    reason_suffix = " ML is extrapolating outside training data → trusting physics formula."
                elif tier == "MEDIUM":
                    tolerance = YS_TOLERANCE_MEDIUM
                    suggested_value_factor = SUGGESTED_VALUE_FACTOR_MEDIUM
                    reason_suffix = " Moderate confidence → applying physics constraint."
                else:
                    tolerance = YS_TOLERANCE_HIGH
                    suggested_value_factor = SUGGESTED_VALUE_FACTOR_HIGH
                    reason_suffix = ""

                deviation_pct = abs(ys - physics_ys) / physics_ys * 100 if physics_ys > 0 else 0

                sss_detail = ""
                if sss_breakdown:
                    top_sss = sorted(sss_breakdown.items(), key=lambda x: x[1], reverse=True)[:3]
                    sss_detail = f" SSS contributors: {', '.join(f'{el}={v:.0f}' for el, v in top_sss)} MPa."

                decomposition_str = (
                    f"σ_base={sigma_base:.0f} + σ_{ppt_label}={sigma_ppt:.0f} + "
                    f"σ_SSS={sigma_sss:.0f} + σ_HP={sigma_hp:.0f} = {physics_ys:.0f} MPa"
                )

                if ys > physics_ys + tolerance:
                    suggested_ys = round(physics_ys + (ys - physics_ys) * suggested_value_factor, 0)
                    severity = "high" if deviation_pct > SEVERITY_HIGH_PCT_YS_OVER else "medium"
                    proposals.append({
                        "property": "Yield Strength",
                        "current_value": ys,
                        "physics_min": round(physics_ys - tolerance, 0),
                        "physics_max": round(physics_ys + tolerance, 0),
                        "physics_typical": round(physics_ys, 0),
                        "suggested_value": suggested_ys,
                        "deviation_pct": round(deviation_pct, 1),
                        "severity": severity,
                        "strength_breakdown": strength_breakdown,
                        "reasoning": (
                            f"Decomposed strength model for {processing_note}: {decomposition_str}. "
                            f"Predicted {ys:.0f} MPa is {deviation_pct:.1f}% above physics estimate.{sss_detail}{reason_suffix}"
                        ),
                        "literature": "Pollock & Tin (2006), Reed (2006), Labusch (1970)"
                    })
                elif ys < physics_ys - tolerance:
                    # Only generate UPWARD YS proposals for SSS alloys
                    if is_sss:
                        suggested_ys = round(physics_ys - (physics_ys - ys) * suggested_value_factor, 0)
                        severity = "high" if deviation_pct > SEVERITY_HIGH_PCT_YS_UNDER else "medium"
                        proposals.append({
                            "property": "Yield Strength",
                            "current_value": ys,
                            "physics_min": round(physics_ys - tolerance, 0),
                            "physics_max": round(physics_ys + tolerance, 0),
                            "physics_typical": round(physics_ys, 0),
                            "suggested_value": suggested_ys,
                            "deviation_pct": round(deviation_pct, 1),
                            "severity": severity,
                            "strength_breakdown": strength_breakdown,
                            "reasoning": (
                                f"Decomposed strength model for {processing_note}: {decomposition_str}. "
                                f"Predicted {ys:.0f} MPa is {deviation_pct:.1f}% below physics estimate.{sss_detail}{reason_suffix}"
                            ),
                            "literature": "Pollock & Tin (2006), Reed (2006), Labusch (1970)"
                        })

                # SSS-specific corrections
                if is_sss:
                    sss_physics_ys = sigma_base + sigma_sss + sigma_hp
                    sss_physics_ys = max(SSS_YS_MIN_RT, min(SSS_YS_MAX_RT, sss_physics_ys))

                    needs_correction = False
                    if ys > SSS_YS_MAX_RT:
                        deviation_pct = (ys - SSS_YS_MAX_RT) / SSS_YS_MAX_RT * 100
                        correction_direction = "over"
                        needs_correction = True
                    elif ys < SSS_YS_MIN_RT:
                        deviation_pct = (SSS_YS_MIN_RT - ys) / SSS_YS_MIN_RT * 100
                        correction_direction = "under"
                        needs_correction = True
                    elif abs(ys - sss_physics_ys) > SSS_YS_TOLERANCE:
                        deviation_pct = abs(ys - sss_physics_ys) / sss_physics_ys * 100
                        correction_direction = "over" if ys > sss_physics_ys else "under"
                        needs_correction = deviation_pct > 15

                    if needs_correction:
                        corrected_ys = round(sss_physics_ys + (ys - sss_physics_ys) * 0.3, 0)
                        corrected_ys = max(SSS_YS_MIN_RT, min(SSS_YS_MAX_RT, corrected_ys))

                        proposals.append({
                            "property": "Yield Strength",
                            "current_value": ys,
                            "physics_min": SSS_YS_MIN_RT,
                            "physics_max": SSS_YS_MAX_RT,
                            "physics_estimate": round(sss_physics_ys, 0),
                            "suggested_value": corrected_ys,
                            "deviation_pct": round(deviation_pct, 1),
                            "severity": "high",
                            "correction_type": "SSS_ALLOY_CORRECTION",
                            "correction_direction": correction_direction,
                            "strength_breakdown": strength_breakdown,
                            "reasoning": (
                                f"SSS alloy detected ({sss_reason}). ML predicted YS={ys:.0f} MPa is "
                                f"{correction_direction}-predicted. Physics model (σ_base + σ_SSS + σ_HP) "
                                f"estimates {sss_physics_ys:.0f} MPa. SSS alloys range: {SSS_YS_MIN_RT}-{SSS_YS_MAX_RT} MPa. "
                                f"Correcting to {corrected_ys:.0f} MPa."
                            ),
                            "literature": "Davis (1997), Labusch (1970)"
                        })

                    # Flag if γ' was wrongly predicted for SSS alloy
                    if gp > 5:
                        proposals.append({
                            "property": "Gamma Prime",
                            "current_value": gp,
                            "physics_max": 2.0,
                            "suggested_value": 0.0,
                            "deviation_pct": round(gp, 1),
                            "severity": "high",
                            "correction_type": "SSS_GAMMA_PRIME_OVERRIDE",
                            "reasoning": (
                                f"SSS alloy ({sss_reason}) cannot form significant γ'. "
                                f"ML predicted {gp:.1f}% γ', but Al+Ti+Ta < 2% means no γ' precipitation. "
                                f"Setting γ' = 0%."
                            ),
                            "literature": "Reed (2006), Pollock & Tin (2006)"
                        })

            # === UTS/YS RATIO CHECK ===
            if uts > 0 and ys > 0:
                ratio = uts / ys

                if is_sss:
                    min_ratio = SSS_UTS_YS_RATIO_MIN
                    max_ratio = SSS_UTS_YS_RATIO_MAX
                    expected_ratio = 1.8
                    ratio_suffix = f" SSS alloy ({sss_reason}) → using SSS-specific bounds."
                    alloy_type_label = "SSS"
                    literature_ref = "Fleischer (1963), Davis (1997)"
                else:
                    expected_ratio = EXPECTED_RATIO_BASE + (gp / 100) * EXPECTED_RATIO_GP_FACTOR
                    tier = _get_confidence_tier(confidence_level, kg_match_distance)
                    if tier == "LOW":
                        min_ratio = max(UTS_RATIO_MIN_TIGHT, expected_ratio - UTS_RATIO_MIN_OFFSET_TIGHT)
                        max_ratio = min(UTS_RATIO_MAX_TIGHT, expected_ratio + UTS_RATIO_MAX_OFFSET_TIGHT)
                        ratio_suffix = " ML extrapolating → enforcing typical UTS/YS ratio."
                    elif tier == "MEDIUM":
                        min_ratio = max(UTS_RATIO_MIN_MEDIUM, expected_ratio - UTS_RATIO_MIN_OFFSET_MEDIUM)
                        max_ratio = min(UTS_RATIO_MAX_MEDIUM, expected_ratio + UTS_RATIO_MAX_OFFSET_MEDIUM)
                        ratio_suffix = " Moderate confidence → applying empirical ratio constraint."
                    else:
                        min_ratio = max(UTS_RATIO_MIN_LOOSE, expected_ratio - UTS_RATIO_MIN_OFFSET_LOOSE)
                        max_ratio = min(UTS_RATIO_MAX_LOOSE, expected_ratio + UTS_RATIO_MAX_OFFSET_LOOSE)
                        ratio_suffix = ""
                    alloy_type_label = "γ'"
                    literature_ref = "ASM Handbook Vol 2"

                if ratio > max_ratio:
                    deviation_pct = (ratio - max_ratio) / max_ratio * 100
                    severity = "high" if deviation_pct > SEVERITY_HIGH_PCT_UTS_OVER else "medium"
                    suggested_uts = int(ys * max_ratio)
                    proposals.append({
                        "property": "Tensile Strength",
                        "current_value": uts,
                        "physics_min": int(ys * min_ratio),
                        "physics_max": int(ys * max_ratio),
                        "suggested_value": suggested_uts,
                        "current_ratio": round(ratio, 2),
                        "max_ratio": round(max_ratio, 2),
                        "deviation_pct": round(deviation_pct, 1),
                        "severity": severity,
                        "is_sss_alloy": is_sss,
                        "reasoning": (
                            f"UTS/YS ratio is {ratio:.2f}, exceeding typical maximum {max_ratio:.2f} for {alloy_type_label} alloys. "
                            f"Expected ratio ~{expected_ratio:.2f}. "
                            f"Suggested UTS: {suggested_uts:.0f} MPa (ratio {max_ratio:.2f}).{ratio_suffix}"
                        ),
                        "literature": literature_ref
                    })
                elif ratio < min_ratio:
                    deviation_pct = (min_ratio - ratio) / min_ratio * 100
                    severity = "high" if deviation_pct > SEVERITY_HIGH_PCT_UTS_UNDER else "medium"
                    suggested_uts = int(ys * min_ratio)
                    proposals.append({
                        "property": "Tensile Strength",
                        "current_value": uts,
                        "physics_min": int(ys * min_ratio),
                        "physics_max": int(ys * max_ratio),
                        "suggested_value": suggested_uts,
                        "current_ratio": round(ratio, 2),
                        "min_ratio": round(min_ratio, 2),
                        "deviation_pct": round(deviation_pct, 1),
                        "severity": severity,
                        "is_sss_alloy": is_sss,
                        "reasoning": (
                            f"UTS/YS ratio is {ratio:.2f}, below typical minimum {min_ratio:.2f} for {alloy_type_label} alloys.{ratio_suffix}"
                        ),
                        "literature": literature_ref
                    })

            # === ELASTIC MODULUS CHECK ===
            if em > 0:
                if processing in ["wrought", "forged"]:
                    base_em_min = WROUGHT_EM_MIN
                    base_em_max = WROUGHT_EM_MAX
                else:
                    base_em_min = CAST_EM_MIN
                    base_em_max = CAST_EM_MAX

                if co_content > CO_HIGH_THRESHOLD or fe_content > FE_HIGH_THRESHOLD:
                    base_em_min -= EM_CO_FE_REDUCTION_MIN
                    base_em_max -= EM_CO_FE_REDUCTION_MAX
                    note = f" (Co={co_content:.1f}% or Fe={fe_content:.1f}% reduces EM)"
                else:
                    note = ""

                tier = _get_confidence_tier(confidence_level, kg_match_distance)
                if tier == "LOW":
                    min_em = base_em_min + EM_TIGHT_OFFSET_MIN
                    max_em = base_em_max - EM_TIGHT_OFFSET_MAX
                    em_suffix = " ML extrapolating → enforcing strict EM range."
                elif tier == "MEDIUM":
                    if processing in ["wrought", "forged"]:
                        min_em = EM_MEDIUM_WROUGHT_MIN
                        max_em = EM_MEDIUM_WROUGHT_MAX
                    else:
                        min_em = EM_MEDIUM_CAST_MIN
                        max_em = EM_MEDIUM_CAST_MAX
                    em_suffix = " Moderate confidence → enforcing typical EM constraint."
                else:
                    min_em = base_em_min - EM_LOOSE_OFFSET
                    max_em = base_em_max + EM_LOOSE_OFFSET
                    em_suffix = ""

                if em < min_em:
                    deviation_pct = (min_em - em) / min_em * 100
                    severity = "high" if deviation_pct > SEVERITY_HIGH_PCT_EM else "medium"
                    proposals.append({
                        "property": "Elastic Modulus",
                        "current_value": em,
                        "physics_min": min_em,
                        "physics_max": max_em,
                        "suggested_value": min_em,
                        "deviation_pct": round(deviation_pct, 1),
                        "severity": severity,
                        "reasoning": (
                            f"EM {em:.1f} GPa is below typical minimum {min_em} GPa for {processing} Ni-superalloys{note}.{em_suffix}"
                        ),
                        "literature": "Pollock & Tin (2006)"
                    })
                elif em > max_em:
                    deviation_pct = (em - max_em) / max_em * 100
                    severity = "high" if deviation_pct > SEVERITY_HIGH_PCT_EM else "medium"
                    proposals.append({
                        "property": "Elastic Modulus",
                        "current_value": em,
                        "physics_min": min_em,
                        "physics_max": max_em,
                        "suggested_value": max_em,
                        "deviation_pct": round(deviation_pct, 1),
                        "severity": severity,
                        "reasoning": (
                            f"EM {em:.1f} GPa exceeds typical maximum {max_em} GPa for {processing} Ni-superalloys{note}.{em_suffix}"
                        ),
                        "literature": "Pollock & Tin (2006)"
                    })

            # === ELONGATION CHECK ===
            if el > 0 and ys > 0:
                if ys > YS_VERY_HIGH:
                    base_max_el = MAX_EL_VERY_HIGH_STRENGTH
                    category = f"very high strength (>{YS_VERY_HIGH} MPa)"
                elif ys > YS_HIGH:
                    base_max_el = MAX_EL_HIGH_STRENGTH
                    category = f"high strength (>{YS_HIGH} MPa)"
                elif ys > YS_MODERATE:
                    base_max_el = MAX_EL_MODERATE_STRENGTH
                    category = f"moderate strength (>{YS_MODERATE} MPa)"
                else:
                    base_max_el = MAX_EL_LOW_STRENGTH
                    category = "moderate strength"

                if is_sss:
                    max_el = base_max_el * SSS_ELONGATION_FACTOR
                    alloy_type_note = f" SSS alloy ({sss_reason}) → higher ductility allowed."
                    literature_ref = "Davis (1997), Fleischer (1963)"
                else:
                    max_el = base_max_el
                    alloy_type_note = ""
                    literature_ref = "Materials science principles"

                if el > max_el:
                    deviation_pct = (el - max_el) / max_el * 100
                    severity = "medium" if deviation_pct > SEVERITY_MEDIUM_PCT_EL else "low"
                    proposals.append({
                        "property": "Elongation",
                        "current_value": el,
                        "physics_max": max_el,
                        "suggested_value": max_el,
                        "deviation_pct": round(deviation_pct, 1),
                        "severity": severity,
                        "is_sss_alloy": is_sss,
                        "reasoning": (
                            f"Elongation {el:.1f}% is unusually high for {category} alloys. "
                            f"Typical maximum ductility for YS={ys:.0f} MPa is ~{max_el:.0f}%.{alloy_type_note}"
                        ),
                        "literature": literature_ref
                    })

            # Build context
            context = {
                "confidence_level": confidence_level,
                "kg_match_distance": kg_match_distance,
                "has_kg_match": kg_match_distance < KG_DISTANCE_FAR,
                "is_exploratory": kg_match_distance > KG_DISTANCE_FAR,
                "alloy_classification": {
                    "is_sss_alloy": is_sss,
                    "sss_reason": sss_reason,
                    "strengthening_type": "SSS-dominated" if is_sss else "γ'-dominated"
                },
                "special_elements": {
                    "high_rhenium": re_content > RE_HIGH_THRESHOLD,
                    "high_cobalt": co_content > CO_VERY_HIGH_THRESHOLD,
                    "high_iron": fe_content > FE_VERY_HIGH_THRESHOLD
                },
                "recommendation": self._get_correction_recommendation(
                    confidence_level, kg_match_distance, len(proposals)
                )
            }

            # Auto-apply high-severity corrections
            corrected_properties = properties.copy()
            corrections_applied = []

            for proposal in proposals:
                prop_name = proposal.get("property")
                severity = proposal.get("severity", "low")
                suggested = proposal.get("suggested_value")
                correction_type = proposal.get("correction_type", "")
                original = corrected_properties.get(prop_name)

                is_downward = suggested is not None and original is not None and suggested < original
                is_sss_correction = correction_type.startswith("SSS_")
                should_apply = is_sss_correction or (severity == "high" and is_downward)

                if should_apply and suggested is not None and prop_name:
                    corrected_properties[prop_name] = suggested
                    corrections_applied.append({
                        "property": prop_name,
                        "original": original,
                        "corrected": suggested,
                        "reason": proposal.get("reasoning", "")[:200],
                        "correction_type": correction_type
                    })

            result = {
                "proposals": proposals,
                "context": context,
                "corrected_properties": corrected_properties,
                "corrections_applied": corrections_applied,
                "total_proposals": len(proposals),
                "high_severity_count": sum(1 for p in proposals if p["severity"] == "high"),
                "medium_severity_count": sum(1 for p in proposals if p["severity"] == "medium"),
                "auto_corrections_count": len(corrections_applied)
            }

            return json.dumps(result, indent=2)

        except Exception as e:
            logger.error(f"Failed to generate physics correction proposals: {e}")
            return json.dumps({
                "error": str(e),
                "proposals": [],
                "context": {"error": "Failed to generate proposals"}
            })

    def _get_correction_recommendation(
        self,
        confidence_level: str,
        kg_distance: float,
        num_proposals: int
    ) -> str:
        """Provide guidance on how aggressively to apply corrections."""
        if kg_distance < KG_DISTANCE_CLOSE:
            return "Strong KG match - trust fusion more, only apply corrections for high severity violations."
        elif kg_distance < KG_DISTANCE_WEAK:
            return "Weak KG match - apply medium/high severity corrections with caution."
        elif confidence_level in ["LOW", "VERY LOW"]:
            return "No KG match and low confidence - apply corrections aggressively. ML is extrapolating."
        elif num_proposals >= 3:
            return "Multiple physics violations detected - prediction quality questionable. Apply corrections."
        else:
            return "Moderate confidence - review each proposal case-by-case."
