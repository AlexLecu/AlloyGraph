from typing import Any, Type
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import json
import logging

logger = logging.getLogger(__name__)

# YS vs Gamma Prime Constants
WROUGHT_BASE_YS = 400
WROUGHT_GP_COEFF = 18
CAST_BASE_YS = 450
CAST_GP_COEFF = 20

# Adaptive Tolerance Thresholds (YS)
YS_TOLERANCE_LOW = 120
YS_TOLERANCE_MEDIUM = 180
YS_TOLERANCE_HIGH = 250

# Suggested Value Factors
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


def _get_confidence_tier(confidence_level: str, kg_distance: float) -> str:
    """
    Determine confidence tier based on confidence level and KG match distance.
    
    Returns:
        'LOW', 'MEDIUM', or 'HIGH'
    """
    if confidence_level in ["LOW", "VERY LOW"] or kg_distance > KG_DISTANCE_FAR:
        return "LOW"
    elif confidence_level == "MEDIUM" or kg_distance > KG_DISTANCE_MEDIUM:
        return "MEDIUM"
    return "HIGH"


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
        "Uses empirical rules from literature (Pollock & Tin 2006, Geddes et al. 2010, ASM Handbook). "
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
        """
        Generate physics-based correction proposals.

        Returns JSON with proposed corrections and reasoning.
        """
        try:
            # Parse inputs
            properties = json.loads(properties_json)
            composition = json.loads(composition_json)

            proposals = []

            # SKIP CORRECTIONS FOR EXCELLENT KG MATCHES
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

            # ============================================================
            # PROPOSAL 1: YS vs Gamma Prime Relationship
            # ============================================================
            # Empirical: YS ≈ 400 + 18*GP for wrought, 450 + 20*GP for cast
            # Source: Pollock & Tin (2006), Geddes et al. (2010)
            if gp > 0 and ys > 0:
                if processing in ["wrought", "forged"]:
                    base_ys = WROUGHT_BASE_YS
                    gp_coefficient = WROUGHT_GP_COEFF
                    processing_note = "wrought alloys"
                else:
                    base_ys = CAST_BASE_YS
                    gp_coefficient = CAST_GP_COEFF
                    processing_note = "cast alloys"

                physics_ys = base_ys + gp_coefficient * gp
                
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

                deviation_pct = abs(ys - physics_ys) / physics_ys * 100

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
                        "reasoning": (
                            f"For γ'={gp:.1f}%, typical YS for {processing_note} is {physics_ys:.0f}±{tolerance:.0f} MPa "
                            f"(empirical: YS ≈ {base_ys} + {gp_coefficient}×γ'). "
                            f"Predicted {ys:.0f} MPa is {deviation_pct:.1f}% above physics constraint.{reason_suffix}"
                        ),
                        "literature": "Pollock & Tin (2006), Geddes et al. (2010)"
                    })
                elif ys < physics_ys - tolerance:
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
                        "reasoning": (
                            f"For γ'={gp:.1f}%, typical YS is {physics_ys:.0f}±{tolerance:.0f} MPa. "
                            f"Predicted {ys:.0f} MPa is {deviation_pct:.1f}% below physics constraint.{reason_suffix}"
                        ),
                        "literature": "Pollock & Tin (2006)"
                    })

            # ============================================================
            # PROPOSAL 2: UTS/YS Ratio
            # ============================================================
            # Typical range: 1.1-1.5 for superalloys
            # Higher γ' → higher ratio
            # Source: ASM Handbook Vol 2
            if uts > 0 and ys > 0:
                ratio = uts / ys
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
                        "reasoning": (
                            f"UTS/YS ratio is {ratio:.2f}, exceeding typical maximum {max_ratio:.2f} for superalloys. "
                            f"For γ'={gp:.1f}%, expected ratio is ~{expected_ratio:.2f}. "
                            f"Suggested UTS: {suggested_uts:.0f} MPa (ratio {max_ratio:.2f}).{ratio_suffix}"
                        ),
                        "literature": "ASM Handbook Vol 2"
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
                        "reasoning": (
                            f"UTS/YS ratio is {ratio:.2f}, below typical minimum {min_ratio:.2f} for superalloys.{ratio_suffix}"
                        ),
                        "literature": "ASM Handbook Vol 2"
                    })

            # ============================================================
            # PROPOSAL 3: Elastic Modulus Bounds
            # ============================================================
            # Ni-base: typically 200-220 GPa for wrought, 180-210 for cast
            # High Co/Fe can reduce by ~10 GPa
            # Source: Pollock & Tin (2006)
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

            # ============================================================
            # PROPOSAL 4: Strength-Ductility Tradeoff
            # ============================================================
            # Very high strength → low ductility
            # Source: General materials science
            if el > 0 and ys > 0:
                if ys > YS_VERY_HIGH:
                    max_el = MAX_EL_VERY_HIGH_STRENGTH
                    category = f"very high strength (>{YS_VERY_HIGH} MPa)"
                elif ys > YS_HIGH:
                    max_el = MAX_EL_HIGH_STRENGTH
                    category = f"high strength (>{YS_HIGH} MPa)"
                elif ys > YS_MODERATE:
                    max_el = MAX_EL_MODERATE_STRENGTH
                    category = f"moderate strength (>{YS_MODERATE} MPa)"
                else:
                    max_el = MAX_EL_LOW_STRENGTH
                    category = "moderate strength"

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
                        "reasoning": (
                            f"Elongation {el:.1f}% is unusually high for {category} alloys. "
                            f"Typical maximum ductility for YS={ys:.0f} MPa is ~{max_el}%."
                        ),
                        "literature": "Materials science principles"
                    })

            # ============================================================
            # Add context about confidence level and KG match
            # ============================================================
            context = {
                "confidence_level": confidence_level,
                "kg_match_distance": kg_match_distance,
                "has_kg_match": kg_match_distance < KG_DISTANCE_FAR,
                "is_exploratory": kg_match_distance > KG_DISTANCE_FAR,
                "special_elements": {
                    "high_rhenium": re_content > RE_HIGH_THRESHOLD,
                    "high_cobalt": co_content > CO_VERY_HIGH_THRESHOLD,
                    "high_iron": fe_content > FE_VERY_HIGH_THRESHOLD
                },
                "recommendation": self._get_correction_recommendation(
                    confidence_level, kg_match_distance, len(proposals)
                )
            }

            result = {
                "proposals": proposals,
                "context": context,
                "total_proposals": len(proposals),
                "high_severity_count": sum(1 for p in proposals if p["severity"] == "high"),
                "medium_severity_count": sum(1 for p in proposals if p["severity"] == "medium")
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
