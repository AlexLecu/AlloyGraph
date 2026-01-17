from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type, Dict, Any, Literal, List
import json

from ..models.feature_engineering import compute_alloy_features
import logging

logger = logging.getLogger(__name__)

def validate_property_bounds(properties: Dict[str, Any]) -> list[str]:
    """
    Validates that predicted properties are within physically reasonable bounds.
    Returns a list of error messages for properties that violate physical constraints.
    """
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
    
    # Known superalloy limits (Based on literature: Reed 2006, Pollock & Tin 2006)
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
    if em > 0:  # Only check if provided
        if em < 90 or em > 300:
            errors.append(f"Elastic Modulus ({em} GPa) outside physically reasonable range for Ni-superalloys (90-300 GPa)")
        elif em < 100 or em > 250:
            errors.append(f"Elastic Modulus ({em} GPa) outside typical Ni-superalloy range (100-250 GPa) - verify composition")
    
    # Density bounds for Ni-based superalloys (typically 7.5-9.5 g/cm³)
    if density > 0:  # Only check if provided
        if density < 7.0 or density > 10.0:
            errors.append(f"Density ({density} g/cm³) out of typical Ni-superalloy range (7.5-9.5)")
    
    # Gamma Prime volume fraction bounds (0-70% typical)
    if gp > 0:  # Only check if provided
        if gp > 75:
            errors.append(f"Gamma Prime ({gp}%) exceeds typical maximum (~70%)")
    
    return errors


# ============================================================
# Property Coherency Cross-Check
# ============================================================
def validate_property_coherency(properties: Dict[str, Any], composition: Dict[str, float]) -> list[str]:
    """
    Validate that predicted properties are mutually consistent and align with composition.

    Checks cross-property relationships and composition-property correlations
    to catch physically contradictory predictions.
    """
    warnings = []

    # Extract properties
    ys = properties.get("Yield Strength", 0)
    uts = properties.get("Tensile Strength", 0)
    el = properties.get("Elongation", 0)
    em = properties.get("Elastic Modulus", 0)
    density = properties.get("Density", 8.5)
    gp = properties.get("Gamma Prime", 0)

    # Extract key composition elements
    re_wt = composition.get("Re", 0)
    w_wt = composition.get("W", 0)
    ta_wt = composition.get("Ta", 0)
    al_wt = composition.get("Al", 0)
    ti_wt = composition.get("Ti", 0)

    heavy_refractories = re_wt + w_wt + ta_wt
    gp_formers = al_wt + ti_wt

    # ============================================================
    # Rule 1: High Strength Requires Adequate γ' Fraction
    # ============================================================
    # Physical basis: Precipitation strengthening is primary mechanism
    # Literature: Reed (2006) - YS ≈ 5-8 MPa per 1% γ'
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

    # ============================================================
    # Rule 2: Density vs Refractory Content
    # ============================================================
    # Physical basis: Re (21.0 g/cm³), W (19.3 g/cm³), Ta (16.7 g/cm³) >> Ni (8.9 g/cm³)
    # Expected density increases ~0.15-0.25 g/cm³ per 1% refractory
    if density > 0 and heavy_refractories > 0:
        baseline_density = 8.2
        expected_density = baseline_density + (heavy_refractories / 100) * 2.5

        if abs(density - expected_density) > 0.8:
            warnings.append(
                f"⚠️ Coherency Warning: Density anomaly detected. "
                f"Predicted: {density:.2f} g/cm³, Expected for {heavy_refractories:.1f}% refractories: ~{expected_density:.2f} g/cm³. "
                f"Check if ML model correctly accounts for Re/W/Ta content."
            )

    # ============================================================
    # Rule 3: High Ductility with Heavy Refractories is Rare
    # ============================================================
    # Physical basis: Re/W/Mo reduce dislocation mobility → lower elongation
    # Literature: Pollock & Tin (2006) - Re > 6% typically → EL < 15%
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

    # ============================================================
    # Rule 4: Elastic Modulus vs Composition
    # ============================================================
    # Physical basis: E_M increases with W (411 GPa), Mo (329 GPa), Cr (279 GPa)
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

    # ============================================================
    # Rule 5: UTS/YS Ratio Sanity Check
    # ============================================================
    # Physical basis: UTS/YS typically 1.1-1.4 for superalloys
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

    # ============================================================
    # Rule 6: Gamma Prime Fraction vs Formers
    # ============================================================
    # Physical basis: γ' vol% ≈ 3-4 × (Al_wt + Ti_wt + 0.7×Ta_wt)
    # Simplified Sims-Hagel prediction (wide tolerance due to temperature/composition effects)
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
    """
    Calculate Elastic Modulus using rule of mixtures.
    Based on elemental Young's moduli at room temperature.
    """
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


# ============================================================
# Physics Enforcement Layer (Hard Constraints)
# ============================================================
def enforce_physics_constraints(
    properties: Dict[str, Any],
    temperature_c: float = 20,
    processing: str = "cast",
    confidence_level: str = "MEDIUM",
    kg_distance: float = 999
) -> tuple[Dict[str, Any], list[str]]:
    """
    Enforce physics constraints by correcting extreme deviations.

    Unlike validation (which only warns), this function CORRECTS values
    that are physically impossible or highly unlikely.

    Applied AFTER LLM corrections and calibration as a safety net.

    Returns:
        tuple: (corrected_properties, list of corrections applied)
    """
    corrections = []
    props = properties.copy()

    ys = props.get("Yield Strength", 0)
    uts = props.get("Tensile Strength", 0)
    gp = props.get("Gamma Prime", 0)
    el = props.get("Elongation", 0)

    # Skip if we have a strong KG match (trust experimental data)
    if kg_distance < 3.0:
        logger.info(f"Physics enforcement skipped: KG match distance={kg_distance:.2f}")
        return props, []

    # ============================================================
    # Processing-Composition Compatibility Note
    # ============================================================
    # Note: P/M (powder metallurgy) wrought alloys like RR1000 can have high γ' (>40%)
    # Trust user's processing selection - don't auto-correct

    # ============================================================
    # Constraint 1: YS must be consistent with γ' content
    # ============================================================
    # Formula: YS_RT ≈ base + coeff × γ'
    # Temperature derating: ~0.3-0.5 MPa per °C above RT
    if gp > 5 and ys > 0:
        # Base formula (room temperature)
        if processing in ["wrought", "forged"]:
            base_ys = 400
            gp_coeff = 18
        else:
            base_ys = 450
            gp_coeff = 20

        physics_ys_rt = base_ys + gp_coeff * gp

        # Temperature derating (empirical: ~0.4 MPa/°C for superalloys)
        temp_derating = max(0, (temperature_c - 20) * 0.4)
        physics_ys = max(200, physics_ys_rt - temp_derating)

        # Calculate deviation
        deviation_pct = abs(ys - physics_ys) / physics_ys * 100

        # Thresholds based on confidence
        if confidence_level in ["LOW", "VERY LOW"] or kg_distance > 10:
            threshold_pct = 25  # Stricter for low confidence
        elif confidence_level == "MEDIUM":
            threshold_pct = 35
        else:
            threshold_pct = 50  # More lenient for high confidence

        if deviation_pct > threshold_pct:
            # Blend towards physics value
            if confidence_level in ["LOW", "VERY LOW"]:
                blend_factor = 0.8  # 80% physics, 20% ML
            else:
                blend_factor = 0.6  # 60% physics, 40% ML

            corrected_ys = round(ys * (1 - blend_factor) + physics_ys * blend_factor, 1)

            corrections.append(
                f"YS: {ys:.0f}→{corrected_ys:.0f} MPa (physics expects ~{physics_ys:.0f} for γ'={gp:.1f}% at {temperature_c}°C, "
                f"deviation was {deviation_pct:.0f}%)"
            )
            props["Yield Strength"] = corrected_ys
            ys = corrected_ys  # Update for UTS calculation

            logger.info(f"Physics enforcement: YS corrected {properties.get('Yield Strength'):.0f}→{corrected_ys:.0f} MPa")

    # ============================================================
    # Constraint 2: UTS must maintain valid ratio with YS
    # ============================================================
    if ys > 0 and uts > 0:
        ratio = uts / ys

        # Expected ratio based on processing and γ' content
        # Key insight: Wrought alloys have MUCH higher work hardening than cast
        # - Cast: UTS/YS ≈ 1.15-1.25 (limited by coarse microstructure)
        # - Wrought: UTS/YS ≈ 1.40-1.55 (fine grains enable work hardening)
        if processing in ["wrought", "forged"]:
            base_ratio = 1.40  # Wrought base ratio
            expected_ratio = base_ratio + (gp / 100) * 0.15
            min_ratio = 1.30
            max_ratio = 1.60
        else:
            base_ratio = 1.15  # Cast base ratio
            expected_ratio = base_ratio + (gp / 100) * 0.2
            min_ratio = 1.08
            max_ratio = min(1.5, expected_ratio + 0.15)

        if ratio < min_ratio or ratio > max_ratio:
            target_ratio = max(min_ratio, min(max_ratio, expected_ratio))
            corrected_uts = round(ys * target_ratio, 1)

            corrections.append(
                f"UTS: {uts:.0f}→{corrected_uts:.0f} MPa (ratio {ratio:.2f}→{target_ratio:.2f}, expected ~{expected_ratio:.2f} for {processing})"
            )
            props["Tensile Strength"] = corrected_uts

            logger.info(f"Physics enforcement: UTS corrected {properties.get('Tensile Strength'):.0f}→{corrected_uts:.0f} MPa")

    # ============================================================
    # Constraint 3: Elongation sanity bounds
    # ============================================================
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

        # Wrought alloys have HIGHER minimum ductility (lower bounds)
        # Wrought processing gives finer grains and better ductility
        if processing in ["wrought", "forged"]:
            # Wrought alloys with low-moderate γ' should have good ductility
            if gp < 25 and el < 20:
                min_el = 22.0 - (gp * 0.3)  # ~20% at γ'=7%, ~15% at γ'=25%
                if el < min_el:
                    corrected_el = min_el
                    corrections.append(f"Elongation: {el:.1f}→{corrected_el:.1f}% (wrought alloys have better ductility)")
                    props["Elongation"] = round(corrected_el, 1)
            elif gp < 40 and el < 15:
                min_el = 15.0
                corrected_el = min_el
                corrections.append(f"Elongation: {el:.1f}→{corrected_el:.1f}% (wrought processing improves ductility)")
                props["Elongation"] = round(corrected_el, 1)

    if corrections:
        logger.info(f"Physics enforcement applied {len(corrections)} corrections")

    return props, corrections


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

            # === PROCESSING-COMPOSITION COMPATIBILITY CHECK ===
            # High γ' alloys (>40%) are almost always cast - warn if specified as wrought
            # Low γ' alloys (<20%) are typically wrought - warn if specified as cast
            original_processing = processing
            processing_warning = None

            if processing == "wrought" and gp > 40:
                # Note: P/M (powder metallurgy) wrought alloys like RR1000 can have high γ' (>40%)
                # while still being wrought, so we warn but DON'T auto-correct
                processing_warning = (
                    f"ℹ️ Composition has {gp:.1f}% γ' - high for conventional wrought processing, "
                    f"but valid for P/M (powder metallurgy) alloys like RR1000."
                )
                warnings.append(processing_warning)
                logger.info(processing_warning)
            elif processing == "cast" and gp < 15 and composition.get("Fe", 0) > 10:
                processing_warning = (
                    f"⚠️ Composition has only {gp:.1f}% γ' with high Fe ({composition.get('Fe', 0):.1f}%) - "
                    f"this looks like a wrought alloy (e.g., IN718). Keeping as specified but results may be inaccurate."
                )
                warnings.append(processing_warning)
                logger.warning(processing_warning)

            if processing == "cast":
                 base_ductility = 20.0
                 hall_petch_boost = 0.0
            else:
                 base_ductility = 40.0
                 hall_petch_boost = 50.0
            
            base_ni = 120.0 + hall_petch_boost
            sss_contribution = (12.0 * sss_wt)
            BASE_STRENGTH = base_ni + sss_contribution
            
            COEFF_GP = 25.0
            if alloy_type == 'high_strength':
                COEFF_GP = 45.0
            elif alloy_type == 'high_corrosion':
                COEFF_GP = 15.0
            
            COEFF_SSS = 5.0 
            
            # Lattice Mismatch Strengthening check
            # Large mismatch contributes to strength but hurts stability
            mismatch_boost = abs(delta) * 100.0
            
            ys_physics = BASE_STRENGTH + (COEFF_GP * gp) + (COEFF_SSS * sss_wt) + mismatch_boost

            el_physics = base_ductility - (0.8 * gp) - (0.5 * sss_wt)
            
            if processing == "wrought":
                if el_physics < 12.0: el_physics = 12.0
            else:
                if el_physics < 5.0: el_physics = 5.0

            # Elastic Modulus - Physics-based calculation
            em_physics = calculate_em_rule_of_mixtures(composition)

            if 'metallurgy_metrics' not in input_data: input_data['metallurgy_metrics'] = {}
            input_data['metallurgy_metrics']['gamma_prime_vol'] = gp
            input_data['metallurgy_metrics']['lattice_mismatch'] = delta
            input_data['metallurgy_metrics']['vec'] = vec


            fusion_meta = input_data.get("fusion_meta", {})
            is_kg_anchored = fusion_meta.get("is_kg_anchored", False)
            confidence = input_data.get("confidence", {})

            if is_kg_anchored:
                final_ys = raw_ys
                final_el = raw_el
                final_em = raw_em
            else:
                final_ys = (raw_ys * 0.6) + (ys_physics * 0.4)
                final_el = (raw_el * 0.6) + (el_physics * 0.4)
                final_em = (raw_em * 0.6) + (em_physics * 0.4)
            
            warnings = []
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
            # Note: Many proven industrial alloys (IN738LC, GTD-111) operate with Md 0.98-1.05
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
                 # Elevated risk - common in industrial alloys, manageable with proper processing
                 reason = f"Matrix Md ({md_gamma:.3f}) is elevated (0.98-1.05 range). TCP phase formation possible but manageable - common in proven alloys like IN738LC."
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
            # Note: Many successful industrial alloys have Md 0.98-1.05
            if md_gamma > 1.05:
                penalty_score += 30  # Critical - strong discouragement
            elif md_gamma > 0.98:
                penalty_score += 5   # Elevated - minor concern (IN738LC operates here)
            # No penalty for Md < 0.98

            if abs(delta) > 1.5:
                penalty_score += 20  # Severe mismatch penalty


            # Ensure property intervals exist (generate if not provided by fusion tool)
            intervals = input_data.get("property_intervals", {})
            
            # Generate intervals for ML-predicted properties if missing
            if "Yield Strength" in verified_props and "Yield Strength" not in intervals:
                ys_val = verified_props["Yield Strength"]
                ys_unc = ys_val * 0.10  # 10% base uncertainty
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

            output_data = {
                "summary": summary_text,
                "processing": processing,
                "penalty_score": penalty_score,
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
                "explanation": ""
            }
            return json.dumps(output_data, indent=2)

        except Exception as e:
            logger.error(f"Physics Constraint Error: {e}")
            return json.dumps({"status": "FAIL", "error": f"Physics Constraint Error: {str(e)}", "properties": {}})


# =============================================================================
# SHARED UTILITIES FOR LLM OUTPUT CLEANUP
# =============================================================================

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
    """
    Normalize and clean LLM output: property keys, intervals, and metrics.
    Returns: (clean_properties, clean_intervals, clean_metrics)
    """
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
    """
    Clean LLM confidence output - filter out hallucinated keys like 'confidence1', 'confidence2'.
    Returns a valid confidence dict with proper structure.
    """
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
