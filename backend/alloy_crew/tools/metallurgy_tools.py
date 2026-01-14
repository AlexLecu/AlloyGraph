from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type, Dict, Any, Literal
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
            
            confidence = input_data.get("confidence")
            if not isinstance(confidence, dict):
                # Handle simplified input from agents (float/int)
                if isinstance(confidence, (float, int)):
                     confidence = {
                        "score": float(confidence),
                        "level": "MEDIUM" if confidence > 0.6 else "LOW", 
                        "note": "Reconstructed from scalar"
                     }
                else:
                    # Default fallback
                    confidence = {
                        "score": 0.50,
                        "level": "MEDIUM",
                        "kg_weight_used": 0.0,
                        "similarity_distance": 999.0,
                        "temperature_delta": 0.0,
                        "matched_alloy": "None"
                    }

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
            
            # 3. TCP Risk (Using Matrix Md)
            if md_gamma > 0.98:
                 reason = f"Matrix Md ({md_gamma:.3f}) exceeds critical limit of 0.98. High risk of topologically close-packed (TCP) phase formation like Sigma or Mu."
                 penalties_list.append({
                     "name": "TCP Risk",
                     "value": f"Md_gamma={md_gamma:.3f}",
                     "reason": reason
                 })
                 warnings.append(reason)
            elif md_gamma > 0.95 and md_avg > 0.985:
                 reason = f"Elevated Stability Risk: Matrix Md ({md_gamma:.3f}) and Global Md ({md_avg:.3f}) are dangerously close to instability limits."
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
                penalty_score += len(penalties_list) * 10
            
            # Additional penalty for very high Md even if not warned (soft limit)
            if md_gamma > 0.96:
                penalty_score += 15
            
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

            output_data = {
                "summary": f"Physics Audit Complete. Penalty: {penalty_score}. {len(penalties_list)} Penalties detected.",
                "processing": processing,
                "penalty_score": penalty_score,
                "properties": verified_props,
                "property_intervals": intervals,

                "metallurgy_metrics": {
                    "md_gamma_matrix": round(md_gamma, 3),
                    "lattice_mismatch_pct": round(delta, 3),
                    "vec_avg": round(vec, 3),
                    "tcp_risk": "High" if md_gamma > 0.98 else ("Medium" if md_gamma > 0.95 else "Low"),
                    "sss_wt_pct": round(sss_wt, 2),
                    "base_contribution": int(BASE_STRENGTH)
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
