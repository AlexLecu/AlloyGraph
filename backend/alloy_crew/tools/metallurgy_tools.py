from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type, Dict, Any, Literal, List
import json
from ..models.feature_engineering import compute_alloy_features, calculate_em_rule_of_mixtures
from ..config.alloy_parameters import (
    TCP,
    classify_tcp_risk,
)
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
    if ys < 0:
        errors.append(f"Yield Strength ({ys} MPa) is negative - physically impossible")
    if uts < 0:
        errors.append(f"Tensile Strength ({uts} MPa) is negative - physically impossible")
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
    gp_formers = al_wt + ti_wt + ta_wt

    # High Strength Requires Adequate γ' Fraction
    if ys > 0 and gp > 0:
        if ys > 1400 and gp < 50:
            warnings.append(
                f"⚠️ Coherency Warning: Exceptional yield strength ({ys:.0f} MPa) requires γ' > 50% "
                f"(current: {gp:.1f}%). Verify composition has sufficient Al+Ti."
            )
        elif ys > 1200 and gp < 40:
            warnings.append(
                f"⚠️ Coherency Warning: High yield strength ({ys:.0f} MPa) typically requires γ' > 40% "
                f"(current: {gp:.1f}%). Precipitation hardening may be insufficient."
            )

    # Rule 2: Density vs Refractory Content
    if density > 0 and heavy_refractories > 0:
        baseline_density = 8.2
        expected_density = baseline_density + (heavy_refractories / 100) * 5.0

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

def calculate_metallurgy_penalties(
    gp: float, sss_wt: float, delta: float,
    md_gamma: float, md_avg: float, processing: str
) -> tuple[list[dict], list[str], float]:
    """Calculate audit penalties and warnings based on metallurgical metrics.

    Returns: (penalties_list, warnings, penalty_score)
    """
    penalties_list = []
    warnings = []

    # 1. Gamma Prime / SSS Balance
    if gp < 5.0 and "solid_solution" not in processing:
        if sss_wt < 10.0:
            reason = f"Gamma Prime ({gp:.1f}%) and Solid Solution Strengthening ({sss_wt:.1f}%) are both too low for effective creep resistance."
            penalties_list.append({"name": "Strengthening Balance", "value": f"GP={gp:.1f}%, SSS={sss_wt:.1f}%", "reason": reason})
            warnings.append(reason)

    # 2. Lattice Mismatch (Coherency)
    if abs(delta) > 0.8:
        reason = f"High Lattice Mismatch ({delta:.2f}%) exceeds the 0.8% threshold for stable coherency. Risk of interfacial dislocation formation."
        penalties_list.append({"name": "Coherency Warning", "value": f"{delta:.2f}%", "reason": reason})
        warnings.append(reason)

    # 3. TCP Risk (Using Matrix Md) - Tiered approach using centralized thresholds
    tcp_level = classify_tcp_risk(md_gamma, md_avg)
    if tcp_level == "Critical":
        reason = f"Bulk Md ({md_avg:.3f}) exceeds {TCP['MD_CRITICAL']} - CRITICAL TCP phase risk. Sigma/Mu phases highly likely. (Matrix Md={md_gamma:.3f})"
        penalties_list.append({"name": "TCP Risk - Critical", "value": f"Md_avg={md_avg:.3f}", "reason": reason})
        warnings.append(reason)
    elif tcp_level == "Elevated":
        reason = f"Bulk Md ({md_avg:.3f}) is elevated ({TCP['MD_ELEVATED']}-{TCP['MD_CRITICAL']} range). TCP phase formation possible but manageable. (Matrix Md={md_gamma:.3f})"
        penalties_list.append({"name": "TCP Risk - Elevated", "value": f"Md_avg={md_avg:.3f}", "reason": reason})
        warnings.append(reason)
    elif tcp_level == "Moderate":
        reason = f"Moderate Stability Concern: Bulk Md ({md_avg:.3f}) approaching stability limit ({TCP['MD_MODERATE']}). Matrix Md={md_gamma:.3f}."
        penalties_list.append({"name": "Phase Stability", "value": f"Md_avg={md_avg:.3f}", "reason": reason})
        warnings.append(reason)

    # Calculate penalty score
    penalty_score = len(penalties_list) * 5
    if tcp_level == "Critical":
        penalty_score += 30
    elif tcp_level == "Elevated":
        penalty_score += 5
    if abs(delta) > 1.5:
        penalty_score += 20

    return penalties_list, warnings, penalty_score


def compute_metallurgy_validation(
    properties: Dict[str, Any],
    composition: Dict[str, float],
    temperature_c: float,
    processing: str = "cast",
    confidence: Dict[str, Any] = None,
    existing_intervals: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Compute complete metallurgy validation without modifying properties.

    Deterministic validation: computes metrics, penalties, intervals, and TCP risk.
    Does NOT apply corrections or blend ML/Physics predictions.
    Use this after agent-driven correction decisions.
    """
    confidence = confidence or {}

    # 1. Compute alloy features
    features = compute_alloy_features(composition)
    gp = features["gamma_prime_estimated_vol_pct"]
    md_avg = features.get("Md_avg", 0.0)
    md_gamma = features.get("Md_gamma", md_avg) or 0.0
    sss_wt = features["SSS_total_wt_pct"]
    delta = features.get("lattice_mismatch_pct", 0.0)

    # 2. Calculate penalties and warnings
    penalties_list, penalty_warnings, penalty_score = calculate_metallurgy_penalties(
        gp, sss_wt, delta, md_gamma, md_avg, processing
    )
    warnings = penalty_warnings.copy()

    # 3. Physical bounds check
    bounds_errors = validate_property_bounds(properties)
    for be in bounds_errors:
        warnings.append(be)
        penalties_list.append({"name": "Physical Limit Violation", "value": "Out of Bounds", "reason": be})

    # 4. Property coherency check
    coherency_warnings = validate_property_coherency(properties, composition)
    for cw in coherency_warnings:
        warnings.append(cw)
        name = "Coherency Audit"
        if "Density" in cw:
            name = "Density Coherency"
        elif "Elastic" in cw:
            name = "Elastic Modulus Coherency"
        elif "Yield" in cw:
            name = "Strength/Gamma Prime Coherency"
        elif "Ductility" in cw or "Elongation" in cw:
            name = "Ductility Coherency"
        penalties_list.append({"name": name, "value": "Mismatch", "reason": cw.replace("\u26a0\ufe0f Coherency Warning: ", "")})

    penalty_score += len(coherency_warnings) * 5 + len(bounds_errors) * 5

    # 5. Generate property intervals (preserve any upstream intervals)
    intervals = dict(existing_intervals) if existing_intervals else {}
    uncertainty_pcts = {
        "Yield Strength": 0.10, "Tensile Strength": 0.10,
        "Elongation": 0.15, "Elastic Modulus": 0.05
    }
    for prop, unc_pct in uncertainty_pcts.items():
        if prop in properties and prop not in intervals:
            val = properties[prop]
            if val and val != 0:
                unc = val * unc_pct
                intervals[prop] = {
                    "lower": round(val - unc, 1),
                    "upper": round(val + unc, 1),
                    "uncertainty": round(unc, 1)
                }

    # 6. TCP risk and fallback summary
    tcp_risk = classify_tcp_risk(md_gamma, md_avg)
    summary_text = f"TCP risk: {tcp_risk}."

    # 7. Determine status
    if penalty_score > 50:
        status = "REJECT"
    elif penalties_list and any("Critical" in p.get("name", "") for p in penalties_list):
        status = "REJECT"
    else:
        status = "PASS"

    return {
        "status": status,
        "penalty_score": penalty_score,
        "tcp_risk": tcp_risk,
        "property_intervals": intervals,
        "metallurgy_metrics": {
            "Md (TCP Stability)": round(md_gamma, 3),
            "TCP Risk": tcp_risk,
            "\u03b3/\u03b3' Misfit (%)": round(delta, 3),
            "Refractory Content (wt%)": round(sss_wt, 2),
            "Al+Ti (weldability)": round(composition.get("Al", 0) + composition.get("Ti", 0), 2),
            "Cr (oxidation)": round(composition.get("Cr", 0), 1),
        },
        "audit_penalties": penalties_list,
        "warnings": warnings,
        "summary": summary_text,
    }


class MetallurgyVerifierInput(BaseModel):
    """Input for metallurgy verification."""
    composition: Dict[str, float] = Field(..., description="Alloy composition dict.")
    anchored_properties_json: str = Field(..., description="JSON string of predicted properties for verification.")
    temperature_c: float = Field(..., description="Temperature in Celsius.")
    alloy_type: Literal['high_strength', 'high_corrosion', 'standard'] = Field('standard', description="LLM-inferred alloy class based on Cr/Ti/Al levels.")

class MetallurgyVerifierTool(BaseTool):
    name: str = "MetallurgyVerifierTool"
    description: str = (
        "Validates predicted properties against metallurgical laws: physical bounds, "
        "composition-property coherency, TCP risk, and UTS/YS ratio. "
        "Returns validation status, penalties, and warnings — does NOT modify property values."
    )
    args_schema: Type[BaseModel] = MetallurgyVerifierInput

    def _parse_input_json(self, anchored_properties_json: str) -> tuple:
        """Parse and clean the input JSON from upstream tools."""
        try:
            clean_json = anchored_properties_json.replace("```json", "").replace("```", "").strip()
            if "{" in clean_json:
                start = clean_json.find("{")
                end = clean_json.rfind("}") + 1
                clean_json = clean_json[start:end]
            input_data = json.loads(clean_json)
            props = input_data.get('anchored_properties') or input_data.get('properties') or input_data
            return input_data, props
        except Exception as e:
            logger.warning(f"Failed to parse anchored properties JSON: {e}")
            return {}, {}

    def _detect_processing(self, input_data: dict, composition: dict, gp: float) -> tuple:
        """Detect processing type and validate compatibility with composition."""
        processing = (input_data.get("processing") or input_data.get("family") or "unknown").lower()

        # Infer processing if unknown
        if processing == "unknown":
            if composition.get("B", 0) > 0.005 and composition.get("Zr", 0) > 0.03:
                processing = "cast"
            else:
                processing = "wrought"

        # Normalize processing type
        if "cast" in processing or "nimocast" in str(input_data).lower():
            processing = "cast"
        elif "wrought" in processing or "forged" in processing:
            processing = "wrought"

        # Check processing-composition compatibility
        warnings = []
        if processing == "wrought" and gp > 40:
            warnings.append(
                f"ℹ️ Composition has {gp:.1f}% γ' - high for conventional wrought processing, "
                f"but valid for P/M (powder metallurgy) alloys."
            )
        elif processing == "cast" and gp < 15 and composition.get("Fe", 0) > 10:
            warnings.append(
                f"⚠️ Composition has only {gp:.1f}% γ' with high Fe ({composition.get('Fe', 0):.1f}%) - "
                f"this looks like a wrought alloy. Keeping as specified but results may be inaccurate."
            )

        return processing, warnings

    # =========================================================================
    # Main Entry Point
    # =========================================================================

    def _run(self, composition: Dict[str, float], anchored_properties_json: str, temperature_c: float, alloy_type: str = 'standard', **kwargs: Any) -> str:
        """
        Validation-only tool: thin adapter around compute_metallurgy_validation().

        Parses LLM JSON input, detects processing, overrides Density/GP with
        computed values, then delegates all validation to the shared function.
        """
        input_data, props = self._parse_input_json(anchored_properties_json)

        composition = {k: float(v) for k, v in composition.items()}
        features = compute_alloy_features(composition)
        gp = features["gamma_prime_estimated_vol_pct"]
        density = features["density_calculated_gcm3"]

        try:
            processing, processing_warnings = self._detect_processing(input_data, composition, gp)
            for pw in processing_warnings:
                logger.info(pw)

            # Use input properties as-is, but override Density/GP with computed values
            verified_props = {
                "Yield Strength": round(float(props.get('Yield Strength', 0))),
                "Tensile Strength": round(float(props.get('Tensile Strength', 0))),
                "Elongation": round(float(props.get('Elongation', 0)), 1),
                "Elastic Modulus": round(float(props.get('Elastic Modulus', 0)), 1),
                "Density": round(density, 2),
                "Gamma Prime": round(gp, 1)
            }

            confidence = input_data.get("confidence", {})
            result = compute_metallurgy_validation(
                properties=verified_props,
                composition=composition,
                temperature_c=temperature_c,
                processing=processing,
                confidence=confidence,
                existing_intervals=input_data.get("property_intervals", {})
            )

            # Enrich with tool-specific fields
            result["processing"] = processing
            result["properties"] = verified_props
            result["confidence"] = confidence
            if processing_warnings:
                result["warnings"] = processing_warnings + result.get("warnings", [])

            return json.dumps(result, indent=2)

        except Exception as e:
            logger.error(f"Metallurgy Validation Error: {e}")
            return json.dumps({"status": "FAIL", "error": f"Metallurgy Validation Error: {str(e)}", "properties": {}})

# SHARED UTILITIES FOR LLM OUTPUT CLEANUP

PROPERTY_KEY_MAP = {
    "YS": "Yield Strength",
    "Yield": "Yield Strength",
    "yield_strength": "Yield Strength",
    "YieldStrength": "Yield Strength",
    "Yield_Strength": "Yield Strength",
    "UTS": "Tensile Strength",
    "Tensile": "Tensile Strength",
    "tensile_strength": "Tensile Strength",
    "TensileStrength": "Tensile Strength",
    "Tensile_Strength": "Tensile Strength",
    "EM": "Elastic Modulus",
    "E": "Elastic Modulus",
    "Modulus": "Elastic Modulus",
    "elastic_modulus": "Elastic Modulus",
    "ElasticModulus": "Elastic Modulus",
    "Elastic_Modulus": "Elastic Modulus",
    "El": "Elongation",
    "elongation": "Elongation",
    "Ductility": "Elongation",
    "GP": "Gamma Prime",
    "Gamma_Prime": "Gamma Prime",
    "gamma_prime": "Gamma Prime",
    "GammaPrime": "Gamma Prime",
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
    md_avg = features.get("Md_avg", 0)
    return {
        "Md (TCP Stability)": round(md_val, 3),
        "TCP Risk": classify_tcp_risk(md_val, md_avg),
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
        logger.info(f"Converted Gamma Prime from fraction ({gp}) to percentage ({clean_props['Gamma Prime']}%)")

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
                logger.warning(f"Filtered out invalid metric: {key}")

    # Fallback if missing required metrics
    if not any(k in clean_metrics for k in REQUIRED_METRIC_KEYS):
        logger.warning("Missing required metrics - computing from feature_engineering...")
        clean_metrics = compute_fallback_metrics(composition)

    return clean_props, clean_intervals, clean_metrics

VALID_CONFIDENCE_KEYS = {"level", "similarity_distance", "model_confidence", "data_quality", "score", "matched_alloy"}

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
