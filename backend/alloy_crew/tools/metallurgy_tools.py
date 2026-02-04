from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type, Dict, Any, Literal, List
import json

from ..models.feature_engineering import compute_alloy_features
from ..config.alloy_parameters import get_params, get_coeff_gp, get_ml_weight, COMMON
import logging

logger = logging.getLogger(__name__)

def validate_property_bounds(properties: Dict[str, Any]) -> list[str]:
    """Validate that predicted properties are within physically reasonable bounds."""
    errors = []
    
    ys = properties.get('Yield Strength', 0)
    uts = properties.get('Tensile Strength', 0)
    el = properties.get('Elongation', 0)
    density = properties.get('Density', 0)
    gp = properties.get('Gamma Prime', 0)
    em = properties.get('Elastic Modulus', 0)
    
    # Physical impossibilities
    if ys > uts and uts > 0:
        errors.append(f"Yield Strength ({ys} MPa) > UTS ({uts} MPa) - physically impossible")
    
    # Known superalloy limits
    if ys > 2000:
        errors.append(f"Yield Strength ({ys} MPa) exceeds known superalloy limits (~2000 MPa)")
    if uts > 2500:
        errors.append(f"Tensile Strength ({uts} MPa) exceeds known superalloy limits (~2500 MPa)")
    
    # Elongation bounds
    if el < 0:
        errors.append(f"Elongation ({el}%) cannot be negative")
    if el > 100:
        errors.append(f"Elongation ({el}%) exceeds 100% - physically impossible")
    
    # Elastic Modulus bounds for Ni-based superalloys (typically 180-220 GPa, hard limits 150-250)
    if em > 0:
        if em < 90 or em > 300:
            errors.append(f"Elastic Modulus ({em} GPa) outside physically reasonable range for Ni-superalloys (90-300 GPa)")
        elif em < 100 or em > 250:
            errors.append(f"Elastic Modulus ({em} GPa) outside typical Ni-superalloy range (100-250 GPa) - verify composition")
    
    # Density bounds for Ni-based superalloys (typically 7.5-9.5 g/cm³)
    if density > 0:
        if density < 7.0 or density > 10.0:
            errors.append(f"Density ({density} g/cm³) out of typical Ni-superalloy range (7.5-9.5)")
    
    # Gamma Prime volume fraction bounds (0-70% typical)
    if gp > 0:
        if gp > 75:
            errors.append(f"Gamma Prime ({gp}%) exceeds typical maximum (~70%)")
    
    return errors

def validate_property_coherency(properties: Dict[str, Any], composition: Dict[str, float]) -> list[str]:
    """Validate property consistency and composition-property alignment."""
    warnings = []

    ys = properties.get("Yield Strength", 0)
    uts = properties.get("Tensile Strength", 0)
    el = properties.get("Elongation", 0)
    em = properties.get("Elastic Modulus", 0)
    density = properties.get("Density", 8.5)
    gp = properties.get("Gamma Prime", 0)

    re_wt = composition.get("Re", 0)
    w_wt = composition.get("W", 0)
    ta_wt = composition.get("Ta", 0)
    al_wt = composition.get("Al", 0)
    ti_wt = composition.get("Ti", 0)

    heavy_refractories = re_wt + w_wt + ta_wt
    gp_formers = al_wt + ti_wt

    # High Strength Requires Adequate γ' Fraction
    if ys > 0 and gp > 0:
        if ys > 1200 and gp < 40:
            warnings.append(
                f"⚠️ Coherency Warning: High yield strength ({ys:.0f} MPa) typically requires γ' > 40% "
                f"(current: {gp:.1f}%). Precipitation hardening may be insufficient."
            )
        elif ys > 1400 and gp < 50:
            warnings.append(
                f"⚠️ Coherency Warning: Exceptional yield strength ({ys:.0f} MPa) requires γ' > 50% "
                f"(current: {gp:.1f}%). Verify composition has sufficient Al+Ti."
            )

    # Rule 2: Density vs Refractory Content
    if density > 0 and heavy_refractories > 0:
        baseline_density = 8.2
        expected_density = baseline_density + (heavy_refractories / 100) * 2.5

        if abs(density - expected_density) > 0.8:
            warnings.append(
                f"⚠️ Coherency Warning: Density anomaly detected. "
                f"Predicted: {density:.2f} g/cm³, Expected for {heavy_refractories:.1f}% refractories: ~{expected_density:.2f} g/cm³. "
                f"Check if ML model correctly accounts for Re/W/Ta content."
            )

    # High Ductility with Heavy Refractories is Rare
    if el > 25 and heavy_refractories > 10:
        warnings.append(
            f"⚠️ Coherency Warning: Unusual combination - High elongation ({el:.1f}%) with heavy refractories "
            f"({heavy_refractories:.1f}%). Re/W typically reduce ductility. Verify if composition is exploratory."
        )

    if el > 30 and re_wt > 6:
        warnings.append(
            f"⚠️ Coherency Warning: High Re content ({re_wt:.1f}%) rarely compatible with elongation > 30%. "
            f"Current prediction: {el:.1f}%. This may indicate extrapolation beyond training data."
        )

    # Rule 4: Elastic Modulus vs Composition
    # Decreases with Al (70 GPa), Ti (116 GPa)
    if em > 0:
        expected_em = calculate_em_rule_of_mixtures(composition)

        if abs(em - expected_em) > 30:
            warnings.append(
                f"⚠️ Coherency Warning: Elastic modulus mismatch. "
                f"Predicted: {em:.0f} GPa, Rule-of-mixtures estimate: {expected_em:.0f} GPa (Δ={abs(em - expected_em):.0f}). "
                f"Large deviation suggests compositional effects beyond linear mixing."
            )

        if not (180 <= em <= 230) and gp_formers < 10:
            warnings.append(
                f"⚠️ Coherency Warning: Elastic modulus ({em:.0f} GPa) outside typical Ni-alloy range (180-230 GPa) "
                f"and composition doesn't justify deviation (Al+Ti={gp_formers:.1f}%)."
            )

    # Rule 5: UTS/YS Ratio Sanity Check
    # Ratio < 1.05 suggests insufficient work hardening capacity
    # Ratio > 1.6 unusual for high-strength alloys
    if ys > 0 and uts > 0:
        ratio = uts / ys
        if ratio < 1.05:
            warnings.append(
                f"⚠️ Coherency Warning: UTS/YS ratio ({ratio:.2f}) is unusually low. "
                f"UTS ({uts:.0f}) barely exceeds YS ({ys:.0f}), suggesting limited work hardening."
            )
        elif ratio > 1.6:
            warnings.append(
                f"⚠️ Coherency Warning: UTS/YS ratio ({ratio:.2f}) is unusually high. "
                f"Typical superalloy ratio is 1.1-1.4. Verify if composition has unique hardening mechanism."
            )

    # Rule 6: Gamma Prime Fraction vs Formers
    if gp > 0 and gp_formers > 0:
        expected_gp = (al_wt + ti_wt + 0.7 * ta_wt) * 3.5

        if abs(gp - expected_gp) > 25 and expected_gp > 20:
            warnings.append(
                f"⚠️ Coherency Warning: γ' volume fraction mismatch. "
                f"Predicted: {gp:.1f}%, Expected from formers (Al+Ti+Ta={gp_formers:.1f}%): ~{expected_gp:.0f}%. "
                f"Check if phase fraction calculation is accurate."
            )

    return warnings

def calculate_em_rule_of_mixtures(composition: Dict[str, float]) -> float:
    """Calculate Elastic Modulus using rule of mixtures."""
    elemental_moduli = {
        "Ni": 200.0,
        "Cr": 279.0,
        "Co": 209.0,
        "Al": 70.0,
        "Ti": 116.0,
        "Mo": 329.0,
        "W": 411.0,
        "Fe": 211.0
    }

    em = sum(composition.get(element, 0) / 100.0 * modulus
             for element, modulus in elemental_moduli.items())

    return em

# Physics Enforcement Layer (Hard Constraints)
def enforce_physics_constraints(
    properties: Dict[str, Any],
    composition: Dict[str, float],
    temperature_c: float = 20,
    processing: str = "cast",
    confidence_level: str = "MEDIUM",
    kg_distance: float = 999
) -> tuple[Dict[str, Any], list[str]]:
    """Enforce physics constraints by correcting extreme deviations."""
    corrections = []
    props = properties.copy()

    ys = props.get("Yield Strength", 0)
    uts = props.get("Tensile Strength", 0)
    gp = props.get("Gamma Prime", 0)
    el = props.get("Elongation", 0)

    # Skip if we have a strong KG match (trust experimental data)
    if kg_distance < COMMON["KG_SKIP_THRESHOLD"]:
        logger.info(f"Physics enforcement skipped: KG match distance={kg_distance:.2f}")
        return props, []

    params = get_params(processing)

    # YS must be consistent with γ' content
    # IMPORTANT: Only apply DOWNWARD corrections (caps) to prevent inflating predictions
    # The decomposed physics model tends to OVER-estimate for high-γ' alloys
    if gp > 5 and ys > 0:
        base_ys = params["ENFORCE_BASE_YS"]
        gp_coeff = params["ENFORCE_GP_COEFF"]

        physics_ys_rt = base_ys + gp_coeff * gp

        # Temperature derating (empirical: ~0.4 MPa/°C for superalloys above RT)
        temp_derating = max(0, (temperature_c - 20) * 0.4)
        physics_ys = max(200, physics_ys_rt - temp_derating)

        # Calculate deviation
        deviation_pct = abs(ys - physics_ys) / physics_ys * 100

        # Thresholds from config
        if confidence_level in ["LOW", "VERY LOW"] or kg_distance > 10:
            threshold_pct = COMMON["THRESHOLD_LOW_CONF"]
        elif confidence_level == "MEDIUM":
            threshold_pct = COMMON["THRESHOLD_MED_CONF"]
        else:
            threshold_pct = COMMON["THRESHOLD_HIGH_CONF"]

        is_over_prediction = ys > physics_ys

        if deviation_pct > threshold_pct and is_over_prediction:
            # Blend factors from config
            if confidence_level in ["LOW", "VERY LOW"]:
                blend_factor = COMMON["BLEND_LOW_CONF"]
            else:
                blend_factor = COMMON["BLEND_HIGH_CONF"]

            corrected_ys = round(ys * (1 - blend_factor) + physics_ys * blend_factor, 1)

            corrections.append(
                f"YS: {ys:.0f}→{corrected_ys:.0f} MPa (physics expects ~{physics_ys:.0f} for γ'={gp:.1f}% at {temperature_c}°C, "
                f"deviation was {deviation_pct:.0f}%)"
            )
            props["Yield Strength"] = corrected_ys
            ys = corrected_ys  # Update for UTS calculation

            logger.info(f"Physics enforcement: YS corrected {properties.get('Yield Strength'):.0f}→{corrected_ys:.0f} MPa")

    # Constraint 2: UTS must maintain valid ratio with YS
    if ys > 0 and uts > 0:
        ratio = uts / ys

        al = composition.get("Al", composition.get("al", 0)) or 0
        ti = composition.get("Ti", composition.get("ti", 0)) or 0
        ta = composition.get("Ta", composition.get("ta", 0)) or 0
        is_sss_alloy = (al + ti + ta) < 2.0

        is_sc_ds, sc_ds_reason = _is_sc_ds_alloy(composition)

        if is_sc_ds and temperature_c < 400:
            base_ratio = 1.12
            expected_ratio = base_ratio
            min_ratio = 1.05
            max_ratio = 1.20
            logger.info(f"SC/DS alloy detected at RT: {sc_ds_reason} - using low UTS/YS ratio bounds")
        elif is_sss_alloy:
            base_ratio = 2.0
            expected_ratio = base_ratio
            min_ratio = 1.6
            max_ratio = 2.4
        elif processing in ["wrought", "forged"]:
            base_ratio = 1.40
            expected_ratio = base_ratio + (gp / 100) * 0.15
            min_ratio = 1.30
            max_ratio = 1.60

            if gp > 40:
                max_ratio = 1.35
                expected_ratio = min(expected_ratio, 1.30)
                logger.info(f"High-γ' wrought alloy (γ'={gp:.1f}%) - using tight UTS/YS ratio bounds (max 1.35)")
        else:

            base_ratio = 1.15
            expected_ratio = base_ratio + (gp / 100) * 0.2
            min_ratio = 1.08
            max_ratio = min(1.5, expected_ratio + 0.15)

        if ratio < min_ratio or ratio > max_ratio:
            target_ratio = max(min_ratio, min(max_ratio, expected_ratio))
            corrected_uts = round(ys * target_ratio, 1)

            # Custom message for high-γ' wrought alloys
            if processing in ["wrought", "forged"] and gp > 40 and ratio > 1.35:
                corrections.append(
                    f"UTS: {uts:.0f}→{corrected_uts:.0f} MPa (ratio {ratio:.2f}→{target_ratio:.2f}). "
                    f"High-γ' wrought alloys (γ'={gp:.1f}%) have limited work hardening (typical ratio 1.25-1.35)"
                )
            else:
                corrections.append(
                    f"UTS: {uts:.0f}→{corrected_uts:.0f} MPa (ratio {ratio:.2f}→{target_ratio:.2f}, expected ~{expected_ratio:.2f} for {processing})"
                )
            props["Tensile Strength"] = corrected_uts

            logger.info(f"Physics enforcement: UTS corrected {properties.get('Tensile Strength'):.0f}→{corrected_uts:.0f} MPa")

    # Constraint 3: Elongation sanity bounds
    if el > 0:
        # High γ' alloys have lower ductility (upper bounds)
        if gp > 60 and el > 20:
            corrected_el = min(el, 18.0)
            if corrected_el != el:
                corrections.append(f"Elongation: {el:.1f}→{corrected_el:.1f}% (high γ' reduces ductility)")
                props["Elongation"] = corrected_el
                el = corrected_el
        elif gp > 40 and el > 30:
            corrected_el = min(el, 25.0)
            if corrected_el != el:
                corrections.append(f"Elongation: {el:.1f}→{corrected_el:.1f}% (moderate γ' limits ductility)")
                props["Elongation"] = corrected_el
                el = corrected_el

        # Wrought γ' alloys have HIGHER minimum ductility (lower bounds)
        # Wrought processing gives finer grains and better ductility
        # NOTE: Skip for SSS alloys - they have different ductility characteristics
        # and are handled in apply_sss_corrections
        if processing in ["wrought", "forged"] and not is_sss_alloy:
            # Wrought γ' alloys with low-moderate γ' should have good ductility
            if gp < 25 and el < 20:
                min_el = 22.0 - (gp * 0.3)  # ~20% at γ'=7%, ~15% at γ'=25%
                if el < min_el:
                    corrected_el = min_el
                    corrections.append(f"Elongation: {el:.1f}→{corrected_el:.1f}% (wrought γ' alloys have better ductility)")
                    props["Elongation"] = round(corrected_el, 1)
            elif gp < 40 and el < 15:
                min_el = 15.0
                corrected_el = min_el
                corrections.append(f"Elongation: {el:.1f}→{corrected_el:.1f}% (wrought γ' processing improves ductility)")
                props["Elongation"] = round(corrected_el, 1)

    em = props.get("Elastic Modulus", 0)
    if em > 0 and temperature_c > 50:
        em_rt = em
        temp_delta = temperature_c - 20
        em_reduction_rate = 0.00032
        em_reduction_factor = 1.0 - (temp_delta * em_reduction_rate)
        em_reduction_factor = max(0.5, em_reduction_factor)  # Cap at 50% reduction (very high T)

        corrected_em = round(em_rt * em_reduction_factor, 1)

        if abs(corrected_em - em) > 5:
            corrections.append(
                f"EM: {em:.1f}→{corrected_em:.1f} GPa (temperature correction: -{(1-em_reduction_factor)*100:.1f}% at {temperature_c}°C)"
            )
            props["Elastic Modulus"] = corrected_em
            logger.info(f"Physics enforcement: EM corrected {em:.1f}→{corrected_em:.1f} GPa for T={temperature_c}°C")

    # Constraint 5: UTS nudge for wrought γ' alloys at ELEVATED temperature only
    ys = props.get("Yield Strength", 0)
    uts = props.get("Tensile Strength", 0)
    if ys > 0 and uts > 0 and temperature_c > 400 and gp > 10:
        ratio = uts / ys
        # For wrought γ' alloys at elevated T, expected ratio ~1.50-1.60
        if processing in ["wrought", "forged"]:
            # At elevated T, ratio is more predictable: ~1.50-1.60 regardless of γ' fraction
            expected_ratio = 1.55
            min_acceptable = 1.45

            if ratio < min_acceptable:
                # Nudge toward expected ratio (partial correction)
                nudge_factor = 0.5  # 50% toward expected
                target_ratio = ratio + (expected_ratio - ratio) * nudge_factor
                corrected_uts = round(ys * target_ratio, 1)
                if corrected_uts > uts:  # Only correct upward
                    corrections.append(
                        f"UTS: {uts:.0f}→{corrected_uts:.0f} MPa (elevated temp γ' ratio: {ratio:.2f}→{target_ratio:.2f})"
                    )
                    props["Tensile Strength"] = corrected_uts
                    logger.info(f"Physics enforcement: UTS nudged {uts:.0f}→{corrected_uts:.0f} MPa at T={temperature_c}°C")

    if corrections:
        logger.info(f"Physics enforcement applied {len(corrections)} corrections")

    return props, corrections

# =============================================================================
# SSS (Solid Solution Strengthening) Alloy Constants
# =============================================================================
SSS_ALLOY_AL_TI_TA_MAX = 2.0
SSS_YS_MIN_RT = 240
SSS_YS_MAX_RT = 500
SSS_YS_TYPICAL = 375
SSS_GP_MAX = 5.0
SSS_EM_MIN = 200.0
SSS_EM_MAX = 220.0
SSS_EM_TYPICAL = 212.0
SSS_EL_MIN_WROUGHT = 35.0
SSS_EL_MAX_WROUGHT = 65.0
SSS_EL_TYPICAL_WROUGHT = 52.0
SSS_EL_MIN_CAST = 5.0
SSS_EL_MAX_CAST = 20.0
SSS_EL_TYPICAL_CAST = 10.0

# SSS potency factors (MPa per wt%)
SSS_POTENCY = {
    "Re": 18.0, "W": 12.0, "Mo": 10.0, "Nb": 8.0, "Ta": 7.0,
    "Ti": 6.0, "Cr": 6.5, "Fe": 6.0, "Co": 2.0, "Al": 1.0,
    "Mn": 0.5, "Si": 0.3,
}

# SSS strength model parameters
SSS_SIGMA_BASE = 120
SSS_SIGMA_HP_WROUGHT = 40
SSS_SIGMA_HP_CAST = 20
SSS_BLEND_FACTOR = 0.7
SSS_CAST_REDUCTION = 0.80

# SSS temperature degradation parameters
SSS_TEMP_TRANSITION = 600.0
SSS_TEMP_DECAY_SLOW = 0.00030
SSS_TEMP_DECAY_FAST = 280.0
SSS_TEMP_MIN_FACTOR = 0.12
SSS_EL_TEMP_TRANSITION = 500.0
SSS_EL_TEMP_FACTOR = 0.0019

# =============================================================================
# γ' (Gamma Prime) Alloy Temperature Degradation Constants
# =============================================================================
GP_ALLOY_AL_TI_TA_MIN = 2.0
GP_TEMP_STAGE1_END = 750.0
GP_TEMP_STAGE2_END = 900.0
GP_TEMP_DECAY_LINEAR = 0.00020
GP_TEMP_DECAY_TAU1 = 400.0
GP_TEMP_DECAY_TAU2 = 80.0
GP_TEMP_MIN_FACTOR = 0.10

# =============================================================================
# SC/DS (Single Crystal / Directionally Solidified) Alloy Constants
# =============================================================================
SC_DS_TEMP_TRANSITION = 850.0
SC_DS_TEMP_DECAY_TAU = 250.0
SC_DS_TEMP_MIN_FACTOR = 0.35
SC_DS_RE_MIN = 2.0
SC_DS_TA_W_MIN = 10.0
SC_DS_TA_ALONE_MIN = 10.0
SC_DS_TA_W_HIGH = 11.0

def _calculate_sss_physics_ys(composition: dict, processing: str = "wrought") -> tuple:
    """Calculate physics-based YS for SSS alloys using Labusch-Nabarro model."""
    sigma_base = SSS_SIGMA_BASE
    sigma_sss = 0.0
    sss_contributions = []

    for element, potency in SSS_POTENCY.items():
        content = composition.get(element, composition.get(element.lower(), 0)) or 0
        if content > 0:
            contribution = potency * content
            sigma_sss += contribution
            if contribution > 5:
                sss_contributions.append(f"{element}:{contribution:.0f}")

    sigma_hp = SSS_SIGMA_HP_WROUGHT if processing == "wrought" else SSS_SIGMA_HP_CAST
    physics_ys = sigma_base + sigma_sss + sigma_hp

    if processing == "cast":
        physics_ys = physics_ys * SSS_CAST_REDUCTION
        cast_note = f" × {SSS_CAST_REDUCTION} (cast)"
    else:
        cast_note = ""

    physics_ys = max(SSS_YS_MIN_RT, min(SSS_YS_MAX_RT, physics_ys))
    breakdown = f"σ_base={sigma_base} + σ_SSS={sigma_sss:.0f} [{'+'.join(sss_contributions[:4])}] + σ_HP={sigma_hp}{cast_note}"

    return physics_ys, breakdown

def _sss_temperature_degradation(temp_c: float) -> float:
    """Calculate temperature degradation factor for SSS alloys."""
    import math

    if temp_c <= 25:
        return 1.0

    if temp_c <= SSS_TEMP_TRANSITION:
        factor = 1.0 - SSS_TEMP_DECAY_SLOW * (temp_c - 25)
    else:
        factor_at_transition = 1.0 - SSS_TEMP_DECAY_SLOW * (SSS_TEMP_TRANSITION - 25)
        delta_t = temp_c - SSS_TEMP_TRANSITION
        factor = factor_at_transition * math.exp(-delta_t / SSS_TEMP_DECAY_FAST)

    return max(factor, SSS_TEMP_MIN_FACTOR)

def _is_sc_ds_alloy(composition: dict) -> tuple:
    """Detect if alloy is Single Crystal (SC) or Directionally Solidified (DS)."""
    re = composition.get("Re", composition.get("re", 0)) or 0
    ru = composition.get("Ru", composition.get("ru", 0)) or 0
    ta = composition.get("Ta", composition.get("ta", 0)) or 0
    w = composition.get("W", composition.get("w", 0)) or 0

    if re >= SC_DS_RE_MIN:
        return True, f"Re={re:.1f}% (2nd+ gen SC indicator)"
    if ru >= 1.0 and re >= 1.0:
        return True, f"Ru={ru:.1f}%, Re={re:.1f}% (4th gen SC indicator)"
    if (ta + w) >= SC_DS_TA_W_MIN and re >= 1.0:
        return True, f"Ta+W={ta+w:.1f}%, Re={re:.1f}% (SC/DS composition)"
    if (ta + w) >= SC_DS_TA_W_HIGH and ta >= 5.0:
        return True, f"Ta+W={ta+w:.1f}%, Ta={ta:.1f}% (1st gen SC composition)"
    if ta >= SC_DS_TA_ALONE_MIN:
        return True, f"Ta={ta:.1f}% (1st gen SC indicator)"

    return False, ""

def _sc_ds_temperature_degradation(temp_c: float) -> float:
    """Calculate temperature degradation factor for SC/DS alloys."""
    import math

    if temp_c <= 25:
        return 1.0

    if temp_c <= SC_DS_TEMP_TRANSITION:
        factor = 1.0 - 0.00006 * (temp_c - 25)
    else:
        factor_at_transition = 1.0 - 0.00006 * (SC_DS_TEMP_TRANSITION - 25)
        delta_t = temp_c - SC_DS_TEMP_TRANSITION
        factor = factor_at_transition * math.exp(-delta_t / SC_DS_TEMP_DECAY_TAU)

    return max(factor, SC_DS_TEMP_MIN_FACTOR)

def _gp_temperature_degradation(temp_c: float) -> float:
    """Calculate temperature degradation factor for polycrystalline γ' alloys."""
    import math

    if temp_c <= 25:
        return 1.0

    if temp_c <= GP_TEMP_STAGE1_END:
        factor = 1.0 - GP_TEMP_DECAY_LINEAR * (temp_c - 25)
    elif temp_c <= GP_TEMP_STAGE2_END:
        factor_at_stage1 = 1.0 - GP_TEMP_DECAY_LINEAR * (GP_TEMP_STAGE1_END - 25)
        delta_t = temp_c - GP_TEMP_STAGE1_END
        factor = factor_at_stage1 * math.exp(-delta_t / GP_TEMP_DECAY_TAU1)
    else:
        factor_at_stage1 = 1.0 - GP_TEMP_DECAY_LINEAR * (GP_TEMP_STAGE1_END - 25)
        factor_at_stage2 = factor_at_stage1 * math.exp(-(GP_TEMP_STAGE2_END - GP_TEMP_STAGE1_END) / GP_TEMP_DECAY_TAU1)
        delta_t = temp_c - GP_TEMP_STAGE2_END
        factor = factor_at_stage2 * math.exp(-delta_t / GP_TEMP_DECAY_TAU2)

    return max(factor, GP_TEMP_MIN_FACTOR)

def apply_gp_temperature_corrections(properties: dict, composition: dict, temperature_c: int = 20, processing: str = "wrought") -> tuple:
    """Apply temperature-dependent corrections for γ' precipitation-strengthened alloys."""
    import math
    corrections = []
    corrected = properties.copy()

    # Check if γ' alloy (Al+Ti+Ta >= 2%)
    al = composition.get("Al", composition.get("al", 0)) or 0
    ti = composition.get("Ti", composition.get("ti", 0)) or 0
    ta = composition.get("Ta", composition.get("ta", 0)) or 0
    al_ti_ta = al + ti + ta

    logger.debug(f"γ' Check: Al={al}, Ti={ti}, Ta={ta}, Sum={al_ti_ta:.2f}%")

    if al_ti_ta < GP_ALLOY_AL_TI_TA_MIN:
        return corrected, corrections

    # Check if SC/DS alloy
    is_sc_ds, sc_ds_reason = _is_sc_ds_alloy(composition)
    if is_sc_ds:
        logger.debug(f"SC/DS ALLOY DETECTED: {sc_ds_reason}")

    temp_threshold = 850 if is_sc_ds else 750
    if temperature_c <= temp_threshold:
        return corrected, corrections

    logger.info(f"{'SC/DS' if is_sc_ds else 'POLYCRYSTALLINE'} γ' ALLOY at HIGH TEMP ({temperature_c}°C) - applying corrections")

    # === YIELD STRENGTH TEMPERATURE CORRECTION ===
    ys_ml = corrected.get("Yield Strength", 0) or 0
    if ys_ml > 0:
        temp_factor = _sc_ds_temperature_degradation(temperature_c) if is_sc_ds else _gp_temperature_degradation(temperature_c)
        ys_corrected = round(ys_ml * temp_factor)
        ys_min_at_temp = max(100, 800 * temp_factor)
        ys_corrected = max(ys_corrected, ys_min_at_temp)

        if abs(ys_ml - ys_corrected) > 30:
            corrected["Yield Strength"] = ys_corrected
            corrections.append(
                f"γ' YS temperature degradation: {ys_ml:.0f} → {ys_corrected:.0f} MPa "
                f"(T={temperature_c}°C, factor={temp_factor:.2f})"
            )

    # === UTS TEMPERATURE CORRECTION ===
    uts_ml = corrected.get("Tensile Strength", 0) or 0
    if uts_ml > 0:
        temp_factor = _sc_ds_temperature_degradation(temperature_c) if is_sc_ds else _gp_temperature_degradation(temperature_c)
        uts_corrected = round(uts_ml * temp_factor)

        ys_new = corrected.get("Yield Strength", ys_ml)

        # Temperature-dependent minimum UTS/YS ratio
        if temperature_c >= 900:
            min_ratio = 1.01  # Near-equal at extreme temps
        elif temperature_c >= 800:
            # Linear interpolation: 900°C→1.01, 800°C→1.1
            min_ratio = 1.1 - (temperature_c - 800) * 0.0009
        elif temperature_c >= 650:
            # Linear interpolation: 800°C→1.1, 650°C→1.2
            min_ratio = 1.2 - (temperature_c - 650) * 0.00067
        else:
            min_ratio = 1.2  # Standard minimum for moderate temps

        if uts_corrected < ys_new * min_ratio:
            uts_corrected = round(ys_new * min_ratio)

        if abs(uts_ml - uts_corrected) > 30:
            corrected["Tensile Strength"] = uts_corrected
            corrections.append(
                f"γ' UTS temperature degradation: {uts_ml:.0f} → {uts_corrected:.0f} MPa "
                f"(T={temperature_c}°C, min ratio={min_ratio:.2f})"
            )

    # === ELONGATION CORRECTION AT HIGH TEMPS ===
    el = corrected.get("Elongation", 0) or 0
    if el > 0 and temperature_c > 650:
        delta_t = temperature_c - 650
        el_factor = 1.0 + 0.0018 * delta_t
        el_corrected = round(el * el_factor, 1)
        el_corrected = min(60.0, el_corrected)

        if el_corrected > el + 2:
            corrected["Elongation"] = el_corrected
            corrections.append(
                f"γ' Elongation high-temp increase: {el:.1f} → {el_corrected:.1f}% (T={temperature_c}°C)"
            )

    logger.info(f"γ' corrections complete: {len(corrections)} applied")
    return corrected, corrections

def apply_sss_corrections(properties: dict, composition: dict, temperature_c: int = 20, processing: str = "wrought") -> tuple:
    """Apply physics-based SSS (Solid Solution Strengthening) corrections."""
    import math
    corrections = []
    corrected = properties.copy()

    # Check if SSS alloy
    al = composition.get("Al", composition.get("al", 0)) or 0
    ti = composition.get("Ti", composition.get("ti", 0)) or 0
    ta = composition.get("Ta", composition.get("ta", 0)) or 0
    al_ti_ta = al + ti + ta

    logger.debug(f"SSS Check: Al={al}, Ti={ti}, Ta={ta}, Sum={al_ti_ta:.2f}%")

    if al_ti_ta >= SSS_ALLOY_AL_TI_TA_MAX:
        return corrected, corrections

    logger.info(f"SSS ALLOY DETECTED (Al+Ti+Ta={al_ti_ta:.1f}%) - applying corrections")

    # 1. Correct γ' if wrongly predicted
    gp = corrected.get("Gamma Prime", 0) or 0
    gp_as_pct = gp * 100 if gp < 1.0 else gp
    if gp_as_pct > SSS_GP_MAX:
        corrected["Gamma Prime"] = 0.0
        corrections.append(
            f"SSS alloy γ' override: {gp_as_pct:.1f}% → 0% (Al+Ti+Ta={al_ti_ta:.1f}% < {SSS_ALLOY_AL_TI_TA_MAX}%)"
        )

    # 2. Apply physics-based YS correction with temperature degradation
    ys_ml = corrected.get("Yield Strength", 0) or 0
    physics_ys_rt, breakdown = _calculate_sss_physics_ys(composition, processing)

    temp_factor = _sss_temperature_degradation(temperature_c)
    physics_ys = physics_ys_rt * temp_factor

    ys_max_at_temp = SSS_YS_MAX_RT * temp_factor
    ys_min_at_temp = max(20, SSS_YS_MIN_RT * temp_factor)

    logger.debug(f"SSS Physics: {breakdown} = {physics_ys_rt:.0f} MPa (RT), {physics_ys:.0f} MPa @ {temperature_c}°C")

    if ys_ml > ys_max_at_temp:
        corrected_ys = round(physics_ys)
        corrected["Yield Strength"] = corrected_ys
        corrections.append(
            f"SSS YS temperature correction: {ys_ml:.0f} → {corrected_ys:.0f} MPa "
            f"(T={temperature_c}°C, factor={temp_factor:.2f})"
        )
    elif ys_ml < ys_min_at_temp:
        corrected_ys = round(physics_ys)
        corrected["Yield Strength"] = corrected_ys
        corrections.append(
            f"SSS YS under-prediction: {ys_ml:.0f} → {corrected_ys:.0f} MPa (below min {ys_min_at_temp:.0f} MPa)"
        )
    elif temperature_c > 100:
        blend_phys, blend_ml = 0.7, 0.3
        blended_ys = round(blend_phys * physics_ys + blend_ml * ys_ml)
        blended_ys = max(ys_min_at_temp, min(ys_max_at_temp, blended_ys))

        if abs(blended_ys - ys_ml) > 20:
            corrected["Yield Strength"] = blended_ys
            corrections.append(
                f"SSS high-temp blend: {ys_ml:.0f} → {blended_ys:.0f} MPa (T={temperature_c}°C)"
            )
    else:
        # Room temperature blending - composition-dependent
        cr = composition.get("Cr", composition.get("cr", 0)) or 0

        if physics_ys > ys_ml:
            if cr >= 20.0:
                blend_ml, blend_phys = 0.10, 0.90
                blend_note = "high-Cr SSS"
            else:
                # Standard SSS - moderate blend
                blend_ml, blend_phys = 0.5, 0.5
                blend_note = "physics-higher"
        else:
            # Physics predicts lower - trust ML more (physics may overpredict)
            blend_ml, blend_phys = 0.8, 0.2
            blend_note = "ML-higher"

        blended_ys = round(blend_ml * ys_ml + blend_phys * physics_ys)
        blended_ys = max(SSS_YS_MIN_RT, min(SSS_YS_MAX_RT, blended_ys))

        if abs(blended_ys - ys_ml) > 15:
            corrected["Yield Strength"] = blended_ys
            corrections.append(
                f"SSS YS blend ({blend_note}): {ys_ml:.0f} → {blended_ys:.0f} MPa "
                f"({int(blend_ml*100)}% ML + {int(blend_phys*100)}% physics={physics_ys:.0f})"
            )

    # 3. Correct UTS to maintain valid UTS/YS ratio for SSS alloys
    if "Yield Strength" in corrected and corrected["Yield Strength"] > 0:
        ys_new = corrected["Yield Strength"]
        uts = corrected.get("Tensile Strength", 0)

        print(f"   🔍 SSS UTS ratio check: YS={ys_new:.0f}, UTS={uts:.0f}, ratio={uts/ys_new if uts > 0 else 0:.2f}")

        if uts > 0:
            ratio = uts / ys_new
            SSS_RATIO_MIN, SSS_RATIO_MAX, SSS_RATIO_TYPICAL = 1.6, 2.4, 2.0

            if ratio < SSS_RATIO_MIN:
                old_uts = uts
                corrected["Tensile Strength"] = round(ys_new * SSS_RATIO_TYPICAL)
                corrections.append(
                    f"SSS UTS/YS ratio fix: {old_uts:.0f} → {corrected['Tensile Strength']:.0f} MPa (ratio {ratio:.2f} → {SSS_RATIO_TYPICAL})"
                )
            elif ratio > SSS_RATIO_MAX:
                old_uts = uts
                corrected["Tensile Strength"] = round(ys_new * SSS_RATIO_MAX)
                corrections.append(
                    f"SSS UTS/YS ratio cap: {old_uts:.0f} → {corrected['Tensile Strength']:.0f} MPa (ratio {ratio:.2f} → {SSS_RATIO_MAX})"
                )

    # 4. Correct Elastic Modulus for SSS alloys
    em = corrected.get("Elastic Modulus", 0) or 0
    if em > 0:
        em_temp_factor = max(0.65, 1.0 - 0.00032 * max(0, temperature_c - 25))
        em_typical_at_temp = round(SSS_EM_TYPICAL * em_temp_factor, 1)

        if temperature_c > 200:
            old_em = em
            corrected["Elastic Modulus"] = em_typical_at_temp
            if abs(old_em - em_typical_at_temp) > 5:
                corrections.append(
                    f"SSS EM temperature correction: {old_em:.1f} → {em_typical_at_temp:.1f} GPa (T={temperature_c}°C)"
                )
        elif em > SSS_EM_MAX or em < SSS_EM_MIN * 0.8:
            old_em = em
            corrected["Elastic Modulus"] = SSS_EM_TYPICAL
            corrections.append(
                f"SSS EM correction: {old_em:.1f} → {SSS_EM_TYPICAL:.1f} GPa (typical SSS range)"
            )

    # 5. Correct Elongation for SSS alloys
    el = corrected.get("Elongation", 0) or 0
    if el > 0:
        if processing == "cast":
            el_min, el_max, el_typical = SSS_EL_MIN_CAST, SSS_EL_MAX_CAST, SSS_EL_TYPICAL_CAST
        else:
            el_min, el_max, el_typical = SSS_EL_MIN_WROUGHT, SSS_EL_MAX_WROUGHT, SSS_EL_TYPICAL_WROUGHT

        if temperature_c > SSS_EL_TEMP_TRANSITION:
            delta_t = temperature_c - SSS_EL_TEMP_TRANSITION
            el_at_temp = el_typical * math.exp(SSS_EL_TEMP_FACTOR * delta_t)
            el_at_temp = min(150.0, el_at_temp)

            old_el = el
            corrected["Elongation"] = round(el_at_temp, 1)
            if abs(old_el - el_at_temp) > 5:
                corrections.append(
                    f"SSS Elongation high-temp: {old_el:.1f} → {el_at_temp:.1f}% (T={temperature_c}°C)"
                )
        elif el < el_min or el > el_max:
            # Out of bounds - correct to typical
            old_el = el
            corrected["Elongation"] = el_typical
            corrections.append(
                f"SSS Elongation out-of-bounds: {old_el:.1f} → {el_typical:.1f}% ({'cast' if processing == 'cast' else 'wrought'})"
            )
        elif abs(el - el_typical) > 10 and el < el_typical * 0.85:
            old_el = el
            blended_el = round(0.60 * el_typical + 0.40 * el, 1)
            corrected["Elongation"] = blended_el
            corrections.append(
                f"SSS Elongation blend toward typical: {old_el:.1f} → {blended_el:.1f}% (60% typical={el_typical:.0f}% + 40% ML)"
            )

    logger.info(f"SSS corrections complete: {len(corrections)} applied")
    return corrected, corrections

class MetallurgyVerifierInput(BaseModel):
    """Input for metallurgy verification."""
    composition: Dict[str, float] = Field(..., description="Alloy composition dict.")
    anchored_properties_json: str = Field(..., description="JSON output from the DataFusionTool containing 'anchored_properties'.")
    temperature_c: float = Field(..., description="Temperature in Celsius.")
    alloy_type: Literal['high_strength', 'high_corrosion', 'standard'] = Field('standard', description="LLM-inferred alloy class based on Cr/Ti/Al levels.")

class MetallurgyVerifierTool(BaseTool):
    name: str = "MetallurgyVerifierTool"
    description: str = (
        "Applies metallurgical laws. Requires 'alloy_type' inferred by Agent analysis of Cr/Ti/Al levels."
    )
    args_schema: Type[BaseModel] = MetallurgyVerifierInput

    def _run(self, composition: Dict[str, float], anchored_properties_json: str, temperature_c: float, alloy_type: str = 'standard', **kwargs: Any) -> str:
        try:
            clean_json = anchored_properties_json.replace("```json", "").replace("```", "").strip()
            if "{" in clean_json:
                 start = clean_json.find("{")
                 end = clean_json.rfind("}") + 1
                 clean_json = clean_json[start:end]
            input_data = json.loads(clean_json)
            props = input_data.get('anchored_properties') or input_data.get('properties') or input_data

        except Exception as e:
            logger.warning(f"Failed to parse anchored properties JSON: {e}")
            input_data = {}
            props = {}

        composition = {k: float(v) for k, v in composition.items()}
        features = compute_alloy_features(composition)

        gp = features["gamma_prime_estimated_vol_pct"]
        md_avg = features.get("Md_avg", 0.0)
        md_gamma = features.get("Md_gamma", md_avg) or 0.0
        density = features["density_calculated_gcm3"]
        sss_wt = features["SSS_total_wt_pct"]
        delta = features.get("lattice_mismatch_pct", 0.0)
        vec = features.get("VEC_avg", 8.0)

        try:
            raw_ys = float(props.get('Yield Strength', 0))
            raw_ts = float(props.get('Tensile Strength', 0))
            raw_el = float(props.get('Elongation', 0))
            raw_em = float(props.get('Elastic Modulus', 0))
            
            alloy_type = str(alloy_type).lower().strip().replace(" ", "_")
            if "corrosion" in alloy_type: 
                alloy_type = "high_corrosion"
            elif "strength" in alloy_type: 
                alloy_type = "high_strength"
            else: 
                alloy_type = "standard"

            processing = (input_data.get("processing") or input_data.get("family") or "unknown").lower()
            
            if processing == "unknown":
                if composition.get("B", 0) > 0.005 and composition.get("Zr", 0) > 0.03:
                    processing = "cast"
                else:
                    processing = "wrought"
            
            if "cast" in processing or "nimocast" in str(input_data).lower():
                processing = "cast"
            elif "wrought" in processing or "forged" in processing:
                processing = "wrought"

            # Initialize warnings list early (used throughout the function)
            warnings = []

            # === PROCESSING-COMPOSITION COMPATIBILITY CHECK ===
            # High γ' alloys (>40%) are almost always cast - warn if specified as wrought
            # Low γ' alloys (<20%) are typically wrought - warn if specified as cast
            original_processing = processing
            processing_warning = None

            if processing == "wrought" and gp > 40:
                processing_warning = (
                    f"ℹ️ Composition has {gp:.1f}% γ' - high for conventional wrought processing, "
                    f"but valid for P/M (powder metallurgy) alloys."
                )
                warnings.append(processing_warning)
                logger.info(processing_warning)
            elif processing == "cast" and gp < 15 and composition.get("Fe", 0) > 10:
                processing_warning = (
                    f"⚠️ Composition has only {gp:.1f}% γ' with high Fe ({composition.get('Fe', 0):.1f}%) - "
                    f"this looks like a wrought alloy. Keeping as specified but results may be inaccurate."
                )
                warnings.append(processing_warning)
                logger.warning(processing_warning)

            # Get parameters from centralized config
            params = get_params(processing)

            base_ductility = params["BASE_DUCTILITY"]
            hall_petch_boost = params["HALL_PETCH_BOOST"]
            base_ni = params["BASE_NI"] + hall_petch_boost
            sss_contribution = params["SSS_CONTRIBUTION_FACTOR"] * sss_wt
            BASE_STRENGTH = base_ni + sss_contribution

            # Gamma prime strengthening coefficient from config
            COEFF_GP = get_coeff_gp(processing, alloy_type)
            COEFF_SSS = params["COEFF_SSS"]

            # Lattice Mismatch Strengthening check
            mismatch_boost = abs(delta) * 100.0

            ys_physics = BASE_STRENGTH + (COEFF_GP * gp) + (COEFF_SSS * sss_wt) + mismatch_boost

            el_physics = base_ductility - (0.8 * gp) - (0.5 * sss_wt)
            min_el = params["MIN_ELONGATION"]
            if el_physics < min_el:
                el_physics = min_el

            # Elastic Modulus - Physics-based calculation
            em_physics = calculate_em_rule_of_mixtures(composition)

            if 'metallurgy_metrics' not in input_data: input_data['metallurgy_metrics'] = {}
            input_data['metallurgy_metrics']['gamma_prime_vol'] = gp
            input_data['metallurgy_metrics']['lattice_mismatch'] = delta
            input_data['metallurgy_metrics']['vec'] = vec

            fusion_meta = input_data.get("fusion_meta", {})
            is_kg_anchored = fusion_meta.get("is_kg_anchored", False)
            confidence = input_data.get("confidence", {})
            confidence_level = confidence.get("level", "MEDIUM") if isinstance(confidence, dict) else "MEDIUM"

            # Determine ML vs physics weighting based on KG anchoring and confidence
            # Weights come from centralized config (alloy_parameters.py)
            if is_kg_anchored:
                # Strong KG match - trust the fused values completely
                final_ys = raw_ys
                final_el = raw_el
                final_em = raw_em
            else:
                # Get ML weight from config based on processing and confidence
                ml_weight = get_ml_weight(processing, confidence_level)
                final_ys = (raw_ys * ml_weight) + (ys_physics * (1 - ml_weight))
                final_el = (raw_el * ml_weight) + (el_physics * (1 - ml_weight))
                final_em = (raw_em * ml_weight) + (em_physics * (1 - ml_weight))
            
            penalties_list = []

            # 1. Gamma Prime / SSS Balance
            if gp < 5.0 and "solid_solution" not in processing:
                 if sss_wt < 10.0:
                     reason = f"Gamma Prime ({gp:.1f}%) and Solid Solution Strengthening ({sss_wt:.1f}%) are both too low for effective creep resistance."
                     penalties_list.append({
                         "name": "Strengthening Balance",
                         "value": f"GP={gp:.1f}%, SSS={sss_wt:.1f}%",
                         "reason": reason
                     })
                     warnings.append(reason)
            
            # 2. Lattice Mismatch (Coherency)
            if abs(delta) > 0.8:
                reason = f"High Lattice Mismatch ({delta:.2f}%) exceeds the 0.5% threshold for stable coherency. Risk of interfacial dislocation formation."
                penalties_list.append({
                    "name": "Coherency Warning",
                    "value": f"{delta:.2f}%",
                    "reason": reason
                })
                warnings.append(reason)
            
            # 3. TCP Risk (Using Matrix Md) - Tiered approach
            if md_gamma > 1.05:
                 # Critical risk - strongly discouraged
                 reason = f"Matrix Md ({md_gamma:.3f}) exceeds 1.05 - CRITICAL TCP phase risk. Sigma/Mu phases highly likely without careful heat treatment."
                 penalties_list.append({
                     "name": "TCP Risk - Critical",
                     "value": f"Md_gamma={md_gamma:.3f}",
                     "reason": reason
                 })
                 warnings.append(reason)
            elif md_gamma > 0.98:
                 reason = f"Matrix Md ({md_gamma:.3f}) is elevated (0.98-1.05 range). TCP phase formation possible but manageable."
                 penalties_list.append({
                     "name": "TCP Risk - Elevated",
                     "value": f"Md_gamma={md_gamma:.3f}",
                     "reason": reason
                 })
                 warnings.append(reason)
            elif md_gamma > 0.96:
                 # Moderate risk - only warn if combined with other factors
                 if md_avg > 0.985:
                     reason = f"Moderate Stability Concern: Matrix Md ({md_gamma:.3f}) with Global Md ({md_avg:.3f}) approaching stability limits."
                     penalties_list.append({
                         "name": "Phase Stability",
                         "value": f"Md={md_gamma:.3f}",
                         "reason": reason
                     })
                     warnings.append(reason)
                 
            # 4. Strength Scaling for TS if not KG anchored
            if not is_kg_anchored:
                 strength_scale = final_ys / (raw_ys + 0.1)
                 final_ts = raw_ts * strength_scale
            else:
                 final_ts = raw_ts

            # Validate property bounds and coherency
            verified_props = {
                "Yield Strength": int(final_ys),
                "Tensile Strength": int(final_ts),
                "Elongation": round(final_el, 1),
                "Elastic Modulus": round(final_em, 1),
                "Density": round(density, 2),
                "Gamma Prime": round(gp, 1)
            }

            # Property Coherency Cross-Check
            coherency_warnings = validate_property_coherency(verified_props, composition)
            for cw in coherency_warnings:
                warnings.append(cw)
                name = "Coherency Audit"
                if "Density" in cw: name = "Density Coherency"
                elif "Elastic" in cw: name = "Elastic Modulus Coherency"
                elif "Yield" in cw: name = "Strength/Gamma Prime Coherency"
                elif "Ductility" in cw: name = "Ductility Coherency"
                
                penalties_list.append({
                    "name": name,
                    "value": "Mismatch",
                    "reason": cw.replace("⚠️ Coherency Warning: ", "")
                })

            # Physical Bounds Check
            bounds_errors = validate_property_bounds(verified_props)
            for be in bounds_errors:
                warnings.append(be)
                penalties_list.append({
                    "name": "Physical Limit Violation",
                    "value": "Out of Bounds",
                    "reason": be
                })

            # Calculate Penalty Score for Agents
            penalty_score = 0
            if penalties_list:
                penalty_score += len(penalties_list) * 5  # Reduced from 10 - penalties are now more informational

            # Tiered penalty for TCP risk (Md)
            if md_gamma > 1.05:
                penalty_score += 30
            elif md_gamma > 0.98:
                penalty_score += 5
            # No penalty for Md < 0.98

            if abs(delta) > 1.5:
                penalty_score += 20  # Severe mismatch penalty

            # Ensure property intervals exist (generate if not provided by fusion tool)
            intervals = input_data.get("property_intervals", {})
            
            # Generate intervals for ML-predicted properties if missing
            if "Yield Strength" in verified_props and "Yield Strength" not in intervals:
                ys_val = verified_props["Yield Strength"]
                ys_unc = ys_val * 0.10
                intervals["Yield Strength"] = {
                    "lower": round(ys_val - ys_unc, 1),
                    "upper": round(ys_val + ys_unc, 1),
                    "uncertainty": round(ys_unc, 1)
                }
            
            if "Tensile Strength" in verified_props and "Tensile Strength" not in intervals:
                ts_val = verified_props["Tensile Strength"]
                ts_unc = ts_val * 0.10
                intervals["Tensile Strength"] = {
                    "lower": round(ts_val - ts_unc, 1),
                    "upper": round(ts_val + ts_unc, 1),
                    "uncertainty": round(ts_unc, 1)
                }
            
            if "Elongation" in verified_props and "Elongation" not in intervals:
                el_val = verified_props["Elongation"]
                el_unc = el_val * 0.15
                intervals["Elongation"] = {
                    "lower": round(el_val - el_unc, 1),
                    "upper": round(el_val + el_unc, 1),
                    "uncertainty": round(el_unc, 1)
                }
            
            if "Elastic Modulus" in verified_props and "Elastic Modulus" not in intervals:
                em_val = verified_props["Elastic Modulus"]
                em_unc = em_val * 0.05
                intervals["Elastic Modulus"] = {
                    "lower": round(em_val - em_unc, 1),
                    "upper": round(em_val + em_unc, 1),
                    "uncertainty": round(em_unc, 1)
                }

            # Generate user-friendly summary
            confidence_level = confidence.get("level", "MEDIUM")
            kg_match = confidence.get("matched_alloy", "None")
            kg_distance = confidence.get("similarity_distance", 999)

            if kg_distance < 2.0 and kg_match != "None":
                summary_text = f"Strong match to {kg_match} - high confidence predictions based on experimental data."
            elif kg_distance < 5.0 and kg_match != "None":
                summary_text = f"Similar to {kg_match} - predictions calibrated with experimental reference."
            elif confidence_level == "HIGH":
                summary_text = "High confidence ML predictions within model's training domain."
            elif confidence_level == "MEDIUM":
                summary_text = "Moderate confidence predictions. Review flagged items below."
            else:
                summary_text = "Exploratory composition - predictions have higher uncertainty."

            # Add TCP risk warning if elevated
            tcp_risk = "Critical" if md_gamma > 1.05 else ("Elevated" if md_gamma > 0.98 else ("Moderate" if md_gamma > 0.96 else "Low"))
            if tcp_risk == "Critical":
                summary_text += " TCP phase risk is critical."
            elif tcp_risk == "Elevated":
                summary_text += " TCP phase risk is elevated (common in industrial alloys)."

            # APPLY SSS AND γ' TEMPERATURE CORRECTIONS
            all_corrections = []
            corrections_explanation_parts = []

            # Apply SSS corrections if applicable (Al+Ti+Ta < 2%)
            verified_props, sss_corrections = apply_sss_corrections(
                properties=verified_props,
                composition=composition,
                temperature_c=int(temperature_c),
                processing=processing
            )
            if sss_corrections:
                all_corrections.extend(sss_corrections)
                corrections_explanation_parts.append(
                    f"SSS alloy corrections applied ({len(sss_corrections)}): "
                    f"Physics-based model used for solid solution strengthened alloy."
                )

            # Apply γ' temperature corrections if applicable (Al+Ti+Ta >= 2% AND high temp)
            verified_props, gp_corrections = apply_gp_temperature_corrections(
                properties=verified_props,
                composition=composition,
                temperature_c=int(temperature_c),
                processing=processing
            )
            if gp_corrections:
                all_corrections.extend(gp_corrections)
                corrections_explanation_parts.append(
                    f"γ' temperature corrections applied ({len(gp_corrections)}): "
                    f"High-temperature degradation model for γ' precipitate dissolution."
                )

            # Build corrections_applied list in the expected format
            corrections_applied_list = []
            for corr_str in all_corrections:
                # Parse correction string: "Property: old → new MPa (reason)"
                # Just store as a dict with the description
                corrections_applied_list.append({
                    "property_name": corr_str.split(":")[0].strip() if ":" in corr_str else "Property",
                    "original_value": 0,
                    "corrected_value": 0,  # Will be in verified_props now
                    "correction_reason": corr_str,
                    "physics_constraint": ""
                })

            # Build corrections explanation
            if corrections_explanation_parts:
                corrections_explanation = " ".join(corrections_explanation_parts)
            else:
                corrections_explanation = "No physics corrections needed - predictions within expected bounds."

            # Determine status based on penalty score
            if penalty_score > 50:
                status = "REJECT"
            elif penalties_list and any("Critical" in p.get("name", "") for p in penalties_list):
                status = "REJECT"
            else:
                status = "PASS"

            output_data = {
                "status": status,
                "summary": summary_text,
                "processing": processing,
                "penalty_score": penalty_score,
                "tcp_risk": tcp_risk,
                "properties": verified_props,
                "property_intervals": intervals,

                "metallurgy_metrics": {
                    # Phase Stability
                    "Md (TCP Stability)": round(md_gamma, 3),
                    "TCP Risk": tcp_risk,
                    # Strengthening
                    "γ/γ' Misfit (%)": round(delta, 3),
                    "Refractory Content (wt%)": round(sss_wt, 2),
                    "Matrix + SSS Strength (MPa)": int(BASE_STRENGTH),
                    # Processing Indicators
                    "Al+Ti (weldability)": round(composition.get("Al", 0) + composition.get("Ti", 0), 2),
                    "Cr (oxidation)": round(composition.get("Cr", 0), 1),
                },
                "audit_penalties": penalties_list,
                "warnings": warnings,
                "confidence": confidence,
                "explanation": "",
                # NEW: Corrections fields for merged Physicist output
                "corrections_applied": corrections_applied_list,
                "corrections_explanation": corrections_explanation
            }
            return json.dumps(output_data, indent=2)

        except Exception as e:
            logger.error(f"Physics Constraint Error: {e}")
            return json.dumps({"status": "FAIL", "error": f"Physics Constraint Error: {str(e)}", "properties": {}})

# SHARED UTILITIES FOR LLM OUTPUT CLEANUP

PROPERTY_KEY_MAP = {
    "YS": "Yield Strength",
    "Yield": "Yield Strength",
    "yield_strength": "Yield Strength",
    "UTS": "Tensile Strength",
    "Tensile": "Tensile Strength",
    "tensile_strength": "Tensile Strength",
    "EM": "Elastic Modulus",
    "E": "Elastic Modulus",
    "Modulus": "Elastic Modulus",
    "elastic_modulus": "Elastic Modulus",
    "El": "Elongation",
    "elongation": "Elongation",
    "Ductility": "Elongation",
    "GP": "Gamma Prime",
    "Gamma_Prime": "Gamma Prime",
    "gamma_prime": "Gamma Prime",
    "γ'": "Gamma Prime",
    "density": "Density",
}

VALID_PROPERTIES = {
    "Yield Strength", "Tensile Strength", "Elongation", "Elastic Modulus",
    "Density", "Gamma Prime", "Creep Life", "Fatigue Life", "Oxidation Resistance"
}

VALID_METRICS = {
    "Md (TCP Stability)", "TCP Risk", "γ/γ' Misfit (%)", "Refractory Content (wt%)",
    "Matrix + SSS Strength (MPa)", "Al+Ti (weldability)", "Cr (oxidation)",
    "Md_gamma", "lattice_mismatch_pct", "refractory_total_wt_pct",
    "gamma_prime_vol", "gamma_prime_fraction", "sss_wt_pct", "density_gcm3",
    "kg_md_avg", "kg_tcp_risk", "kg_sss_wt_pct"
}

REQUIRED_METRIC_KEYS = {"Md (TCP Stability)", "γ/γ' Misfit (%)", "Al+Ti (weldability)"}

def compute_fallback_metrics(composition: Dict[str, float]) -> Dict[str, Any]:
    """Compute metallurgy metrics from feature_engineering when LLM output is invalid."""
    features = compute_alloy_features(composition)
    md_val = features.get("Md_gamma", 0)
    return {
        "Md (TCP Stability)": round(md_val, 3),
        "TCP Risk": "Critical" if md_val > 1.05 else (
            "Elevated" if md_val > 0.98 else ("Moderate" if md_val > 0.96 else "Low")
        ),
        "γ/γ' Misfit (%)": round(features.get("lattice_mismatch_pct", 0), 3),
        "Refractory Content (wt%)": round(features.get("refractory_total_wt_pct", 0), 2),
        "Al+Ti (weldability)": round(composition.get("Al", 0) + composition.get("Ti", 0), 2),
        "Cr (oxidation)": round(composition.get("Cr", 0), 1),
    }

def cleanup_llm_output(
    properties: Dict[str, Any],
    property_intervals: Dict[str, Any],
    metallurgy_metrics: Dict[str, Any],
    composition: Dict[str, float]
) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Normalize and clean LLM output: property keys, intervals, and metrics."""
    # 1. Normalize property keys
    clean_props = {}
    for key, value in properties.items():
        norm_key = PROPERTY_KEY_MAP.get(key, key)
        if norm_key in VALID_PROPERTIES:
            clean_props[norm_key] = value

    # Fix Gamma Prime if given as fraction
    gp = clean_props.get("Gamma Prime", 0)
    if gp is not None and 0 < gp < 1:
        clean_props["Gamma Prime"] = round(gp * 100, 1)
        print(f"  ✓ Converted Gamma Prime from fraction ({gp}) to percentage ({clean_props['Gamma Prime']}%)")

    # 2. Normalize intervals
    clean_intervals = {}
    for key, value in property_intervals.items():
        norm_key = PROPERTY_KEY_MAP.get(key, key)
        if norm_key in VALID_PROPERTIES:
            clean_intervals[norm_key] = value

    # 3. Clean metrics (whitelist)
    clean_metrics = {}
    if metallurgy_metrics:
        for key, value in metallurgy_metrics.items():
            if key in VALID_METRICS:
                clean_metrics[key] = value
            else:
                print(f"  ⚠️ Filtered out invalid metric: {key}")

    # Fallback if missing required metrics
    if not any(k in clean_metrics for k in REQUIRED_METRIC_KEYS):
        print("  ⚠️ Missing required metrics - computing from feature_engineering...")
        clean_metrics = compute_fallback_metrics(composition)

    return clean_props, clean_intervals, clean_metrics

VALID_CONFIDENCE_KEYS = {"level", "similarity_distance", "model_confidence", "data_quality"}

def cleanup_confidence(confidence: Dict[str, Any]) -> Dict[str, Any]:
    """Clean LLM confidence output and filter out hallucinated keys."""
    if not confidence or not isinstance(confidence, dict):
        return {"level": "Medium", "similarity_distance": None}

    clean_conf = {}
    for key, value in confidence.items():
        if key in VALID_CONFIDENCE_KEYS:
            clean_conf[key] = value

    # If no valid keys found, return default
    if not clean_conf:
        return {"level": "Medium", "similarity_distance": None}

    # Ensure 'level' exists
    if "level" not in clean_conf:
        clean_conf["level"] = "Medium"

    return clean_conf

def warnings_to_penalties(warnings: List[str]) -> List[dict]:
    """Convert coherency warning strings to AuditPenalty-compatible dicts."""
    penalties = []
    for warning in warnings:
        name = "Coherency Audit"
        if "Density" in warning:
            name = "Density Coherency"
        elif "Elastic" in warning:
            name = "Elastic Modulus Coherency"
        elif "Yield" in warning or "strength" in warning.lower():
            name = "Strength/Gamma Prime Coherency"
        elif "Ductility" in warning or "Elongation" in warning:
            name = "Ductility Coherency"
        penalties.append({
            "name": name,
            "value": "MEDIUM",
            "reason": warning.replace("⚠️ Coherency Warning: ", "")
        })
    return penalties
