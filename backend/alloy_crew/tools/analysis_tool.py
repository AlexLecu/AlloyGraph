from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Dict, Type, Any, List, Optional
import json
import math
import logging

from ..config.alloy_parameters import (
    get_params, get_coeff_gp, get_temperature_factor,
    get_alloy_class, get_sss_physics_ys,
    SSS, GP_TEMP, SC_DS, UTS_YS_RATIO, ELONGATION,
    classify_tcp_risk, get_em_temp_factor, compress_uts_ys_ratio,
)
from ..models.feature_engineering import (
    compute_alloy_features, calculate_density,
    calculate_em_rule_of_mixtures
)
from ..models.predictor import AlloyPredictor

logger = logging.getLogger(__name__)



# Alias for backward compatibility within this file
_compress_ratio_for_temp = compress_uts_ys_ratio


def _sanitize_for_json(obj):
    """Replace NaN/Infinity with None for safe JSON serialization."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    return obj


class AlloyAnalysisInput(BaseModel):
    """Input for the Alloy Analysis Tool."""
    composition: Dict[str, float] = Field(
        ...,
        description="Dictionary of element symbols and their weight percentages."
    )
    temperature_c: int = Field(
        20,
        description="Target temperature in Celsius."
    )
    processing: str = Field(
        "cast",
        description="Processing type: 'cast' or 'wrought'."
    )
    kg_context: Optional[str] = Field(
        None,
        description="Optional KG search results JSON string."
    )


class AlloyAnalysisTool(BaseTool):
    """Triangulates ML, physics, and KG predictions. Generates correction proposals
    and detects discrepancies for agent investigation."""
    name: str = "AlloyAnalysisTool"
    description: str = (
        "Analyzes an alloy composition and provides raw predictions from multiple sources "
        "(ML, physics, KG) along with PROPOSED corrections. IMPORTANT: Also detects "
        "discrepancies between sources. When discrepancy_detected=True, the agent MUST "
        "investigate using AlloySearchTool before making correction decisions."
    )
    args_schema: Type[BaseModel] = AlloyAnalysisInput

    def _get_ml_predictions(self, composition: Dict[str, float], temperature_c: int, processing: str) -> Dict[str, Any]:
        """Get raw ML predictions."""
        try:
            predictor = AlloyPredictor.get_shared_predictor(model_dir=None)
            options = {"processing": processing}
            report_df = predictor.predict(composition, options, temperatures=[temperature_c])

            if report_df.empty:
                return {"error": "No ML predictions available"}

            row = report_df.iloc[0]
            return {
                "Yield Strength": float(row['ys']),
                "Tensile Strength": float(row['uts']),
                "Elongation": float(row['el']),
                "Elastic Modulus": float(row['em']),
                "source": "ML_model"
            }
        except Exception as e:
            logger.error("ML prediction failed: %s", e)
            return {"error": str(e)}

    def _get_physics_predictions(self, composition: Dict[str, float], temperature_c: int, processing: str) -> tuple:
        """Get physics-based predictions using established metallurgical models.

        Returns:
            tuple: (physics_result dict, alloy_features dict)
        """
        features = compute_alloy_features(composition)
        alloy_class = get_alloy_class(composition, processing)
        params = get_params(processing)

        gp = features.get("gamma_prime_estimated_vol_pct", 0)
        sss_wt = features.get("SSS_total_wt_pct", 0)
        delta = features.get("lattice_mismatch_pct", 0)

        physics_result = {
            "alloy_class": alloy_class,
            "gamma_prime_pct": gp,
            "sss_wt_pct": sss_wt,
            "lattice_mismatch_pct": delta,
            "source": "physics_model"
        }

        if alloy_class == "sss":
            # SSS alloy: Use Labusch-Nabarro model
            physics_ys_rt, breakdown = get_sss_physics_ys(composition, processing)
            temp_factor = get_temperature_factor(temperature_c, "sss")
            physics_ys = physics_ys_rt * temp_factor

            # SSS UTS/YS ratio depends on processing
            if processing == "cast":
                uts_ys_ratio = SSS["UTS_YS_RATIO_TYPICAL_CAST"]
                physics_el = SSS["EL_TYPICAL_CAST"]
            else:
                uts_ys_ratio = SSS["UTS_YS_RATIO_TYPICAL_WROUGHT"]
                physics_el = SSS["EL_TYPICAL_WROUGHT"]

            # SSS UTS/YS ratio temperature decay (converges toward ~1.3, not 1.0)
            # SSS retains more work hardening than γ' alloys
            if temperature_c > 500:
                t_excess = temperature_c - 500
                uts_ys_ratio = 1.3 + (uts_ys_ratio - 1.3) * math.exp(-t_excess / 300)

            # High-T elongation increase
            if temperature_c > SSS["EL_TEMP_TRANSITION"]:
                delta_t = temperature_c - SSS["EL_TEMP_TRANSITION"]
                physics_el = min(65.0, physics_el * math.exp(SSS["EL_TEMP_FACTOR"] * delta_t))

            physics_uts = physics_ys * uts_ys_ratio

            em_rt = calculate_em_rule_of_mixtures(composition)
            em_temp_factor = get_em_temp_factor(temperature_c)
            physics_em = round(em_rt * em_temp_factor, 1)

            physics_result.update({
                "Yield Strength": round(physics_ys, 1),
                "Tensile Strength": round(physics_uts, 1),
                "Elongation": round(physics_el, 1),
                "Elastic Modulus": physics_em,
                "Gamma Prime": 0.0,
                "physics_model": "Labusch-Nabarro SSS",
                "model_breakdown": breakdown,
                "temp_factor": round(temp_factor, 3),
                "expected_uts_ys_ratio": round(uts_ys_ratio, 2)
            })
        else:
            # γ' alloy: Use precipitation hardening model
            base_ni = params["BASE_NI"] + params.get("HALL_PETCH_BOOST", 0)
            sss_contribution = params["SSS_CONTRIBUTION_FACTOR"] * sss_wt
            coeff_gp = get_coeff_gp(processing, "standard")
            mismatch_boost = min(abs(delta), 0.5) * 100.0

            physics_ys_rt = base_ni + sss_contribution + (coeff_gp * gp) + mismatch_boost

            # Temperature degradation
            if alloy_class == "sc_ds":
                temp_factor = get_temperature_factor(temperature_c, "sc_ds")
            else:
                temp_factor = get_temperature_factor(temperature_c, "gp", gp_fraction=gp)

            physics_ys = physics_ys_rt * temp_factor

            # UTS/YS ratio depends on processing and γ'
            if processing in ["wrought", "forged"]:
                if gp > 40:
                    ratio = UTS_YS_RATIO["WROUGHT_HIGH_GP_EXPECTED"]
                else:
                    ratio = UTS_YS_RATIO["WROUGHT_BASE"]
            else:
                ratio = UTS_YS_RATIO["CAST_BASE"] + (gp / 100) * UTS_YS_RATIO["CAST_GP_FACTOR"]

            # Temperature-dependent UTS/YS ratio
            ratio = _compress_ratio_for_temp(ratio, temperature_c)

            physics_uts = physics_ys * ratio

            base_el = params["BASE_DUCTILITY"]
            physics_el = max(params["MIN_ELONGATION"], base_el - 0.8 * gp)

            # High-T elongation increase (precipitate coarsening)
            if temperature_c >= 650:
                delta_t = temperature_c - 650
                el_factor = 1.0 + GP_TEMP["EL_TEMP_FACTOR"] * delta_t
                physics_el = min(120.0, physics_el * el_factor)

            em_rt = calculate_em_rule_of_mixtures(composition)
            em_temp_factor = get_em_temp_factor(temperature_c)
            physics_em = round(em_rt * em_temp_factor, 1)

            physics_result.update({
                "Yield Strength": round(physics_ys, 1),
                "Tensile Strength": round(physics_uts, 1),
                "Elongation": round(physics_el, 1),
                "Elastic Modulus": physics_em,
                "Gamma Prime": round(gp, 1),
                "physics_model": f"{'SC/DS' if alloy_class == 'sc_ds' else 'Polycrystalline'} γ' hardening",
                "temp_factor": round(temp_factor, 3),
                "expected_uts_ys_ratio": round(ratio, 2)
            })

        return physics_result, features

    def _parse_kg_context(self, kg_context: str, target_temp: float, processing: str = "") -> Dict[str, Any]:
        """Parse KG context and extract relevant properties."""
        if not kg_context:
            return {"matched": False}

        try:
            candidates = json.loads(kg_context)
            if not candidates or not isinstance(candidates, list):
                return {"matched": False}

            best = candidates[0]

            # Prefer a candidate with matching processing route.
            if processing:
                proc_lower = processing.lower()
                for candidate in candidates:
                    cand_proc = (candidate.get("processing") or "").lower()
                    cand_dist = candidate.get("_distance", 999)
                    if cand_proc and (proc_lower in cand_proc or cand_proc in proc_lower):
                        if cand_dist < 4.5:
                            best = candidate
                            logger.info(
                                "KG: Preferred processing-compatible '%s' (%s, dist=%.2f) "
                                "over '%s' (%s, dist=%.2f)",
                                candidate.get("name"), cand_proc, cand_dist,
                                candidates[0].get("name"),
                                (candidates[0].get("processing") or "").lower(),
                                candidates[0].get("_distance", 999)
                            )
                            break

            distance = best.get("_distance", 999)

            kg_props = {}
            props = best.get("properties", {})

            for prop_key, target_key in [
                ("YieldStrength", "Yield Strength"),
                ("Yield Strength", "Yield Strength"),
                ("yield_strength", "Yield Strength"),
                ("UTS", "Tensile Strength"),
                ("Tensile Strength", "Tensile Strength"),
                ("TensileStrength", "Tensile Strength"),
                ("tensile_strength", "Tensile Strength"),
                ("Elongation", "Elongation"),
                ("elongation", "Elongation"),
                ("Elasticity", "Elastic Modulus"),
                ("ElasticModulus", "Elastic Modulus"),
                ("Elastic Modulus", "Elastic Modulus"),
            ]:
                prop_str = props.get(prop_key, "")
                if prop_str:
                    val, _ = self._parse_property_string(prop_str, target_temp, is_strength=(target_key != "Elastic Modulus"))
                    if val is not None:
                        kg_props[target_key] = val

            kg_comp = best.get("composition_wt_pct", {})

            return {
                "matched": True,
                "name": best.get("name", "Unknown"),
                "distance": distance,
                "processing": best.get("processing", "unknown"),
                "properties": kg_props,
                "composition_wt_pct": kg_comp,
                "source": "knowledge_graph"
            }
        except Exception as e:
            logger.warning("Failed to parse KG context: %s", e)
            return {"matched": False}

    def _parse_property_string(self, prop_str: str, target_temp: float, is_strength: bool = True) -> tuple:
        """Parse property string like '725.0 MPa @ 538.0C'. Returns closest match within 50°C."""
        best_val, best_temp, best_diff = None, None, float("inf")
        for entry in prop_str.split(','):
            try:
                parts = entry.split('@')
                if len(parts) != 2:
                    continue
                val_str, temp_str = parts
                val = float(val_str.replace('MPa', '').replace('%', '').replace('GPa', '').strip())
                if 'GPa' in entry and is_strength:
                    val *= 1000
                temp = float(temp_str.replace('C', '').strip())
                diff = abs(temp - target_temp)
                if diff < 50 and diff < best_diff:
                    best_val, best_temp, best_diff = val, temp, diff
            except (ValueError, IndexError, TypeError, AttributeError):
                continue
        return (best_val, best_temp)

    def _generate_proposals(
        self,
        ml_pred: Dict[str, Any],
        physics_pred: Dict[str, Any],
        kg_data: Dict[str, Any],
        composition: Dict[str, float],
        temperature_c: int,
        processing: str
    ) -> List[Dict[str, Any]]:
        """Generate correction proposals based on physics rules. Agent decides acceptance."""
        proposals = []
        alloy_class = physics_pred.get("alloy_class", "gp")
        gp = physics_pred.get("gamma_prime_pct", 0)

        ml_ys = ml_pred.get("Yield Strength", 0)
        ml_uts = ml_pred.get("Tensile Strength", 0)
        ml_el = ml_pred.get("Elongation", 0)
        ml_em = ml_pred.get("Elastic Modulus", 0)

        physics_ys = physics_pred.get("Yield Strength", 0)
        physics_uts = physics_pred.get("Tensile Strength", 0)
        physics_el = physics_pred.get("Elongation", 0)

        kg_distance = kg_data.get("distance", 999) if kg_data.get("matched") else 999
        kg_props = kg_data.get("properties", {}) if kg_data.get("matched") else {}

        # === PROPOSAL 1: SSS Alloy Corrections ===
        if alloy_class == "sss":
            if ml_ys > 0 and physics_ys > 0:
                deviation = abs(ml_ys - physics_ys) / physics_ys * 100
                if deviation > 20:
                    # Propose blended value
                    blend_ml, blend_phys = 0.3, 0.7  # Physics-heavy for SSS
                    proposed_ys = blend_ml * ml_ys + blend_phys * physics_ys

                    proposals.append({
                        "property_name": "Yield Strength",
                        "current_value": ml_ys,
                        "proposed_value": round(proposed_ys, 1),
                        "correction_type": "physics",
                        "confidence": "HIGH" if deviation > 40 else "MEDIUM",
                        "reasoning": (
                            f"SSS alloy (Al+Ti+Ta < 2%) detected. ML predicted {ml_ys:.0f} MPa but "
                            f"Labusch-Nabarro physics model suggests {physics_ys:.0f} MPa. "
                            f"SSS alloys rely on solid solution strengthening, not precipitation hardening. "
                            f"Recommending 70% physics + 30% ML blend = {proposed_ys:.0f} MPa."
                        ),
                        "source": "SSS_physics_model"
                    })

                    # Derive UTS from corrected YS using SSS-appropriate ratio
                    if ml_uts > 0:
                        if processing in ["wrought", "forged"]:
                            sss_uts_ratio = SSS["UTS_YS_RATIO_TYPICAL_WROUGHT"]
                        else:
                            sss_uts_ratio = SSS["UTS_YS_RATIO_TYPICAL_CAST"]
                        sss_uts_ratio = _compress_ratio_for_temp(sss_uts_ratio, temperature_c)
                        proposed_sss_uts = proposed_ys * sss_uts_ratio
                        proposals.append({
                            "property_name": "Tensile Strength",
                            "current_value": ml_uts,
                            "proposed_value": round(proposed_sss_uts, 1),
                            "correction_type": "physics",
                            "confidence": "HIGH" if deviation > 40 else "MEDIUM",
                            "reasoning": (
                                f"UTS derived from corrected SSS YS using {processing} ratio "
                                f"({sss_uts_ratio:.2f}, temp-adjusted for {temperature_c}°C). "
                                f"From {ml_uts:.0f} to {proposed_sss_uts:.0f} MPa."
                            ),
                            "source": "SSS_physics_model"
                        })

            # SSS γ' should be 0%
            ml_gp = ml_pred.get("Gamma Prime", 0)
            if ml_gp and ml_gp > 5:
                proposals.append({
                    "property_name": "Gamma Prime",
                    "current_value": ml_gp,
                    "proposed_value": 0.0,
                    "correction_type": "physics",
                    "confidence": "HIGH",
                    "reasoning": (
                        f"SSS alloys have Al+Ti+Ta < 2% and cannot form significant γ' phase. "
                        f"ML predicted {ml_gp:.1f}% but thermodynamically this is impossible. "
                        f"Setting to 0%."
                    ),
                    "source": "SSS_thermodynamics"
                })

            # SSS Elongation depends on processing (cast vs wrought)
            if ml_el > 0 and physics_el > 0:
                if processing == "cast":
                    el_min, el_max = SSS["EL_MIN_CAST"], SSS["EL_MAX_CAST"]
                    process_note = "Cast SSS alloys have limited ductility (5-20%) due to porosity and coarse grains"
                else:
                    el_min, el_max = SSS["EL_MIN_WROUGHT"], SSS["EL_MAX_WROUGHT"]
                    process_note = "Wrought SSS alloys have high ductility (35-65%) from refined microstructure"

                if ml_el < el_min or ml_el > el_max:
                    # Out of bounds — propose physics value directly
                    proposals.append({
                        "property_name": "Elongation",
                        "current_value": ml_el,
                        "proposed_value": physics_el,
                        "correction_type": "physics",
                        "confidence": "HIGH" if abs(ml_el - physics_el) > 20 else "MEDIUM",
                        "reasoning": (
                            f"SSS alloy elongation outside {processing} range [{el_min:.0f}-{el_max:.0f}%]. "
                            f"ML predicted {ml_el:.1f}% but {process_note}. "
                            f"Setting to typical {processing} value: {physics_el:.0f}%."
                        ),
                        "source": "SSS_processing_ductility"
                    })
                else:
                    # Within bounds but significant deviation — blend toward physics
                    el_deviation = abs(ml_el - physics_el) / physics_el * 100
                    if el_deviation > 30:
                        blend_ml, blend_phys = 0.35, 0.65
                        proposed_el = blend_ml * ml_el + blend_phys * physics_el
                        el_confidence = "HIGH" if el_deviation > 40 else "MEDIUM"
                        proposals.append({
                            "property_name": "Elongation",
                            "current_value": ml_el,
                            "proposed_value": round(proposed_el, 1),
                            "correction_type": "physics",
                            "confidence": el_confidence,
                            "reasoning": (
                                f"SSS alloy elongation within bounds [{el_min:.0f}-{el_max:.0f}%] but "
                                f"ML predicted {ml_el:.1f}% vs physics {physics_el:.0f}% "
                                f"({el_deviation:.0f}% deviation). {process_note}. "
                                f"Proposing 65% physics + 35% ML blend = {proposed_el:.1f}%."
                            ),
                            "source": "SSS_elongation_blend"
                        })

        # === PROPOSAL 1B: High-γ' Alloy Empirical Corrections ===
        nb = (composition.get("Nb", 0) or 0)
        al_ti_ta = (composition.get("Al", 0) or 0) + (composition.get("Ti", 0) or 0) + (composition.get("Ta", 0) or 0) + 0.35 * nb
        is_high_gp_alloy = alloy_class == "gp" and gp > 25 and al_ti_ta > 4.0

        kg_has_ys = kg_props.get("Yield Strength") is not None

        if is_high_gp_alloy:
            # Empirical YS correlation for high-γ' alloys: YS ≈ BASE + COEFF × γ'%
            if processing in ["wrought", "forged"]:
                empirical_ys_rt = 520 + 13 * gp
                empirical_ys_min_rt = 470 + 11 * gp  # Lower bound
                empirical_ys_max_rt = 570 + 15 * gp  # Upper bound
            else:
                # Cast γ' alloys have lower strength
                empirical_ys_rt = 400 + 10 * gp
                empirical_ys_min_rt = 350 + 8 * gp
                empirical_ys_max_rt = 450 + 12 * gp

            # Apply temperature degradation to empirical values (they are RT correlations)
            temp_factor = get_temperature_factor(temperature_c, alloy_class, gp_fraction=gp)
            empirical_ys = empirical_ys_rt * temp_factor
            empirical_ys_min = empirical_ys_min_rt * temp_factor
            empirical_ys_max = empirical_ys_max_rt * temp_factor

            if ml_ys > 0 and ml_ys < empirical_ys_min:
                # ML significantly under-predicts compared to empirical range
                underprediction_severity = (empirical_ys_min - ml_ys) / empirical_ys_min
                if underprediction_severity > 0.30:
                    blend_ml, blend_emp = 0.05, 0.95
                elif underprediction_severity > 0.15:
                    blend_ml, blend_emp = 0.10, 0.90
                else:
                    blend_ml, blend_emp = 0.20, 0.80

                proposed_ys = blend_ml * ml_ys + blend_emp * empirical_ys
                proposed_ys = min(proposed_ys, empirical_ys_max)

                if kg_distance <= 2.0 and kg_has_ys:
                    kg_note = f"Close KG match (d={kg_distance:.1f}) with YS data — KG anchoring also active."
                elif kg_distance <= 2.0:
                    kg_note = f"KG match exists (d={kg_distance:.1f}) but lacks YS data at this temperature."
                else:
                    kg_note = f"No strong KG match (distance={kg_distance:.1f})."

                proposals.append({
                    "property_name": "Yield Strength",
                    "current_value": ml_ys,
                    "proposed_value": round(proposed_ys, 1),
                    "correction_type": "physics",
                    "confidence": "HIGH",
                    "reasoning": (
                        f"High-γ' disc alloy detected (γ'={gp:.0f}%, Al+Ti+Ta={al_ti_ta:.1f}%). "
                        f"ML predicted {ml_ys:.0f} MPa but empirical correlation for {gp:.0f}% γ' "
                        f"suggests {empirical_ys_min:.0f}-{empirical_ys_max:.0f} MPa. "
                        f"{kg_note} "
                        f"Proposing {blend_emp*100:.0f}% empirical + {blend_ml*100:.0f}% ML = {proposed_ys:.0f} MPa."
                    ),
                    "source": "high_GP_empirical_model"
                })

                if ml_uts > 0:
                    # Prefer ML's UTS/YS ratio
                    ml_ratio = ml_uts / ml_ys if ml_ys > 0 else 0
                    if processing in ["wrought", "forged"] and gp > 40:
                        min_r = UTS_YS_RATIO["WROUGHT_HIGH_GP_MIN"]
                        max_r = UTS_YS_RATIO["WROUGHT_HIGH_GP_MAX"]
                        default_r = UTS_YS_RATIO["WROUGHT_HIGH_GP_EXPECTED"]
                    elif processing in ["wrought", "forged"]:
                        min_r = UTS_YS_RATIO["WROUGHT_MIN"]
                        max_r = UTS_YS_RATIO["WROUGHT_MAX"]
                        default_r = UTS_YS_RATIO["WROUGHT_BASE"]
                    else:
                        default_r = UTS_YS_RATIO["CAST_BASE"] + (gp / 100) * UTS_YS_RATIO["CAST_GP_FACTOR"]
                        min_r = UTS_YS_RATIO["CAST_MIN"]
                        max_r = default_r + 0.15
                    # Use ML ratio if within physical bounds, else default
                    if min_r <= ml_ratio <= max_r:
                        uts_ratio = ml_ratio
                        ratio_source = f"ML ratio ({ml_ratio:.2f})"
                    else:
                        uts_ratio = default_r
                        ratio_source = f"default ratio ({default_r:.2f}, ML ratio {ml_ratio:.2f} out of [{min_r:.2f}-{max_r:.2f}])"
                    # Temperature-compress the ratio (converges toward 1.0 at high T)
                    uts_ratio = _compress_ratio_for_temp(uts_ratio, temperature_c)
                    proposed_uts = proposed_ys * uts_ratio

                    proposals.append({
                        "property_name": "Tensile Strength",
                        "current_value": ml_uts,
                        "proposed_value": round(proposed_uts, 1),
                        "correction_type": "physics",
                        "confidence": "HIGH",
                        "reasoning": (
                            f"UTS derived from corrected YS using {ratio_source} "
                            f"(ratio={uts_ratio:.2f}, temp-adjusted for {temperature_c}°C). "
                            f"From {ml_uts:.0f} to {proposed_uts:.0f} MPa."
                        ),
                        "source": "high_GP_empirical_model"
                    })

                # Elongation correction for high-γ' alloys
                if ml_el > 0:
                    if processing in ["wrought", "forged"]:
                        empirical_el = max(10.0, 28 - 0.28 * gp)
                        el_min, el_max = 10.0, 20.0
                    else:
                        # Cast high-γ' alloys have even lower ductility
                        empirical_el = max(4.0, 18 - 0.25 * gp)
                        el_min, el_max = 4.0, 12.0

                    if ml_el > el_max:
                        # ML over-predicts ductility for high-γ' alloys
                        proposed_el = min(empirical_el, el_max)
                        proposals.append({
                            "property_name": "Elongation",
                            "current_value": ml_el,
                            "proposed_value": round(proposed_el, 1),
                            "correction_type": "physics",
                            "confidence": "MEDIUM",
                            "reasoning": (
                                f"High-γ' alloys ({gp:.0f}%) have reduced ductility. "
                                f"ML predicted {ml_el:.1f}% but empirical correlation suggests "
                                f"{el_min:.0f}-{el_max:.0f}% for {processing} processing. "
                                f"Proposing {proposed_el:.1f}%."
                            ),
                            "source": "high_GP_empirical_model"
                        })

        # === PROPOSAL 1C: SC/DS Alloy Empirical Corrections ===
        if alloy_class == "sc_ds" and (kg_distance > 2.0 or not kg_has_ys):
            sc_temp_factor = get_temperature_factor(temperature_c, "sc_ds")
            empirical_ys_sc = (560 + 6 * gp) * sc_temp_factor
            empirical_ys_sc_min = (510 + 5 * gp) * sc_temp_factor  # Lower bound
            empirical_ys_sc_max = (610 + 7 * gp) * sc_temp_factor  # Upper bound

            underprediction_severity = (
                (empirical_ys_sc_min - ml_ys) / empirical_ys_sc_min
                if ml_ys > 0 and empirical_ys_sc_min > 0 else 0
            )

            if ml_ys > 0 and underprediction_severity > 0.10:
                if underprediction_severity > 0.25:
                    blend_ml, blend_emp = 0.02, 0.98
                else:
                    blend_ml, blend_emp = 0.05, 0.95

                proposed_ys_sc = blend_ml * ml_ys + blend_emp * empirical_ys_sc
                proposed_ys_sc = min(proposed_ys_sc, empirical_ys_sc_max)

                if kg_distance <= 2.0 and not kg_has_ys:
                    kg_note_sc = f"KG match exists (d={kg_distance:.1f}) but lacks YS data."
                else:
                    kg_note_sc = f"No strong KG match (distance={kg_distance:.1f})."

                proposals.append({
                    "property_name": "Yield Strength",
                    "current_value": ml_ys,
                    "proposed_value": round(proposed_ys_sc, 1),
                    "correction_type": "physics",
                    "confidence": "HIGH",
                    "reasoning": (
                        f"SC/DS alloy detected (γ'={gp:.0f}%). Single crystals have higher strength "
                        f"due to absence of grain boundaries. ML predicted {ml_ys:.0f} MPa but "
                        f"empirical correlation suggests {empirical_ys_sc_min:.0f}-{empirical_ys_sc_max:.0f} MPa. "
                        f"{kg_note_sc} "
                        f"Proposing {blend_emp*100:.0f}% empirical + {blend_ml*100:.0f}% ML = {proposed_ys_sc:.0f} MPa."
                    ),
                    "source": "SC_DS_empirical_model"
                })

                # UTS correction: prefer physics UTS if ratio is within SC/DS bounds
                if ml_uts > 0:
                    sc_ratio_min = SC_DS["UTS_YS_RATIO_MIN"]
                    sc_ratio_max = SC_DS["UTS_YS_RATIO_MAX"]
                    sc_ratio_expected = SC_DS["UTS_YS_RATIO_EXPECTED"]

                    # Check if physics UTS gives a plausible SC/DS ratio
                    if physics_uts > 0 and proposed_ys_sc > 0:
                        physics_ratio = physics_uts / proposed_ys_sc
                    else:
                        physics_ratio = 0

                    if sc_ratio_min <= physics_ratio <= sc_ratio_max:
                        # Physics UTS is within SC/DS bounds — use it
                        proposed_uts_sc = physics_uts
                        confidence = "HIGH"
                        reasoning = (
                            f"Physics UTS ({physics_uts:.0f} MPa) gives ratio "
                            f"{physics_ratio:.2f} with proposed YS ({proposed_ys_sc:.0f} MPa), "
                            f"within SC/DS bounds [{sc_ratio_min:.2f}-{sc_ratio_max:.2f}]. "
                            f"Using physics prediction directly."
                        )
                    else:
                        # Physics ratio out of bounds — use expected ratio
                        proposed_uts_sc = proposed_ys_sc * sc_ratio_expected
                        confidence = "MEDIUM"
                        reasoning = (
                            f"SC/DS UTS/YS ratio constrained to [{sc_ratio_min:.2f}-{sc_ratio_max:.2f}]. "
                            f"Physics ratio ({physics_ratio:.2f}) out of bounds. "
                            f"Using expected ratio {sc_ratio_expected:.2f}: "
                            f"UTS = {proposed_ys_sc:.0f} × {sc_ratio_expected:.2f} = {proposed_uts_sc:.0f} MPa."
                        )

                    proposals.append({
                        "property_name": "Tensile Strength",
                        "current_value": ml_uts,
                        "proposed_value": round(proposed_uts_sc, 1),
                        "correction_type": "physics",
                        "confidence": confidence,
                        "reasoning": reasoning,
                        "source": "SC_DS_UTS_YS_constraint"
                    })

            # SC/DS elongation correction
            if ml_el > 0:
                sc_el_min = SC_DS["ELONGATION_MIN"]
                sc_el_max = SC_DS["ELONGATION_MAX"]
                empirical_el_sc = max(sc_el_min, 22 - 0.15 * gp)

                if ml_el > sc_el_max:
                    proposed_el_sc = min(empirical_el_sc, sc_el_max)
                    proposals.append({
                        "property_name": "Elongation",
                        "current_value": ml_el,
                        "proposed_value": round(proposed_el_sc, 1),
                        "correction_type": "physics",
                        "confidence": "MEDIUM",
                        "reasoning": (
                            f"SC/DS alloys have limited ductility ({sc_el_min:.0f}-{sc_el_max:.0f}%) due to "
                            f"single crystal structure. ML predicted {ml_el:.1f}% but empirical "
                            f"suggests {empirical_el_sc:.1f}% for {gp:.0f}% γ'."
                        ),
                        "source": "SC_DS_empirical_model"
                    })

        # === PROPOSAL 2: UTS/YS Ratio Corrections ===
        # Use proposed corrected values (if any) for ratio check
        effective_ys = ml_ys
        effective_uts = ml_uts
        ys_was_corrected = False
        for p in proposals:
            if p["property_name"] == "Yield Strength" and not ys_was_corrected:
                effective_ys = p["proposed_value"]
                ys_was_corrected = True
            if p["property_name"] == "Tensile Strength":
                effective_uts = p["proposed_value"]

        if effective_ys > 0 and effective_uts > 0:
            ratio = effective_uts / effective_ys

            if alloy_class == "sss":
                # SSS UTS/YS ratio depends on processing
                if processing == "cast":
                    min_ratio = SSS["UTS_YS_RATIO_MIN_CAST"]
                    max_ratio = SSS["UTS_YS_RATIO_MAX_CAST"]
                    expected_ratio = SSS["UTS_YS_RATIO_TYPICAL_CAST"]
                    alloy_note = "Cast SSS alloys have limited work hardening (defects, coarse grains)"
                else:
                    min_ratio = SSS["UTS_YS_RATIO_MIN_WROUGHT"]
                    max_ratio = SSS["UTS_YS_RATIO_MAX_WROUGHT"]
                    expected_ratio = SSS["UTS_YS_RATIO_TYPICAL_WROUGHT"]
                    alloy_note = "Wrought SSS alloys have high work hardening capacity"
            elif alloy_class == "sc_ds":
                min_ratio = SC_DS["UTS_YS_RATIO_MIN"]
                max_ratio = SC_DS["UTS_YS_RATIO_MAX"]
                expected_ratio = SC_DS["UTS_YS_RATIO_EXPECTED"]
                alloy_note = (
                    f"SC/DS alloys have reduced work hardening capacity. "
                    f"UTS/YS ratio typically {min_ratio:.2f}–{max_ratio:.2f} at RT"
                )
            elif processing in ["wrought", "forged"] and gp > 40:
                min_ratio = UTS_YS_RATIO["WROUGHT_HIGH_GP_MIN"]
                max_ratio = UTS_YS_RATIO["WROUGHT_HIGH_GP_MAX"]
                expected_ratio = UTS_YS_RATIO["WROUGHT_HIGH_GP_EXPECTED"]
                alloy_note = f"High-γ' ({gp:.0f}%) wrought alloys have limited work hardening"
            elif processing in ["wrought", "forged"]:
                min_ratio = UTS_YS_RATIO["WROUGHT_MIN"]
                max_ratio = UTS_YS_RATIO["WROUGHT_MAX"]
                expected_ratio = physics_pred.get("expected_uts_ys_ratio", UTS_YS_RATIO["WROUGHT_BASE"])
                alloy_note = f"Wrought γ' precipitation-hardened alloy"
            else:
                min_ratio = UTS_YS_RATIO["CAST_MIN"]
                max_ratio = UTS_YS_RATIO["CAST_BASE"] + (gp / 100) * UTS_YS_RATIO["CAST_GP_FACTOR"] + 0.10
                expected_ratio = physics_pred.get("expected_uts_ys_ratio", UTS_YS_RATIO["CAST_BASE"])
                alloy_note = f"Cast γ' precipitation-hardened alloy"

            # Temperature-dependent ratio bounds for γ' alloys at elevated T
            if temperature_c >= 650 and alloy_class not in ("sss", "sc_ds"):
                expected_ratio = physics_pred.get("expected_uts_ys_ratio", expected_ratio)
                min_ratio = _compress_ratio_for_temp(min_ratio, temperature_c)
                max_ratio = _compress_ratio_for_temp(max_ratio, temperature_c)
                alloy_note += f" (T={temperature_c}°C: ratio bounds compressed)"

            if ratio < min_ratio or ratio > max_ratio:
                target_ratio = max(min_ratio, min(max_ratio, expected_ratio))
                proposed_uts = effective_ys * target_ratio
                ys_note = f" (based on corrected YS={effective_ys:.0f})" if ys_was_corrected else ""

                proposals.append({
                    "property_name": "Tensile Strength",
                    "current_value": ml_uts,
                    "proposed_value": round(proposed_uts, 1),
                    "correction_type": "ratio",
                    "confidence": "HIGH" if abs(ratio - target_ratio) > 0.3 else "MEDIUM",
                    "reasoning": (
                        f"UTS/YS ratio ({ratio:.2f}) outside valid range [{min_ratio:.2f}-{max_ratio:.2f}]{ys_note}. "
                        f"{alloy_note}. Adjusting to ratio {target_ratio:.2f} "
                        f"gives UTS = {proposed_uts:.0f} MPa."
                    ),
                    "source": "UTS_YS_ratio_physics"
                })

        # === PROPOSAL 3: Temperature-Dependent Corrections ===
        if temperature_c > 650 and alloy_class != "sss":
            temp_factor = physics_pred.get("temp_factor", 1.0)
            if temp_factor < 0.9:  # Significant temperature effect
                # Check if ML seems to have not accounted for temperature
                if ml_ys > physics_ys * 1.3:  # ML seems too high for this temperature
                    proposals.append({
                        "property_name": "Yield Strength",
                        "current_value": ml_ys,
                        "proposed_value": round(physics_ys, 1),
                        "correction_type": "physics",
                        "confidence": "MEDIUM",
                        "reasoning": (
                            f"At {temperature_c}°C, temperature degradation factor is {temp_factor:.2f}. "
                            f"ML predicted {ml_ys:.0f} MPa which may not fully account for γ' coarsening "
                            f"and dislocation recovery. Physics model suggests {physics_ys:.0f} MPa."
                        ),
                        "source": "GP_temperature_degradation"
                    })

        # === PROPOSAL 4: KG Anchoring (if strong match with SAME alloy class) ===
        kg_proc = (kg_data.get("processing") or "").lower()
        proc_lower = processing.lower()
        proc_compatible = (proc_lower and kg_proc and
                           (proc_lower in kg_proc or kg_proc in proc_lower))
        max_anchor_dist = 4.5 if proc_compatible else 3.0

        if kg_data.get("matched") and kg_distance < max_anchor_dist:
            kg_name = kg_data.get("name", "Unknown")

            # Check for alloy class mismatch before anchoring.
            # Vector search can match compositionally similar but functionally
            # different alloys. Use γ' fraction difference as discriminator.
            class_mismatch = False
            kg_comp = kg_data.get("composition_wt_pct", {})
            if kg_comp:
                kg_features = compute_alloy_features(kg_comp)
                kg_gp = kg_features.get("gamma_prime_estimated_vol_pct", 0)
                query_gp = physics_pred.get("gamma_prime_pct", gp)
                gp_diff = abs(query_gp - kg_gp)

                if gp_diff > 10:
                    class_mismatch = True
                    logger.warning(
                        "KG class mismatch (gamma prime): query=%.1f%% vs "
                        "KG match '%s'=%.1f%% (diff=%.1f%%). Skipping KG anchoring.",
                        query_gp, kg_name, kg_gp, gp_diff
                    )

            processing_mismatch = not proc_compatible if (proc_lower and kg_proc and kg_proc != "unknown") else False
            if processing_mismatch:
                logger.warning(
                    "KG processing mismatch: query='%s' vs KG match '%s' "
                    "processing='%s'. Skipping KG anchoring.",
                    processing, kg_name, kg_proc
                )

            if not class_mismatch and not processing_mismatch:
                for prop in ["Yield Strength", "Tensile Strength", "Elongation"]:
                    kg_val = kg_props.get(prop)
                    ml_val = ml_pred.get(prop, 0)

                    if kg_val and ml_val:
                        deviation = abs(ml_val - kg_val) / kg_val * 100
                        if deviation > 15:
                            kg_weight = 1.0 / (1.0 + math.exp((kg_distance - 2.5) / 0.5))
                            proposed_val = ml_val * (1 - kg_weight) + kg_val * kg_weight

                            proposals.append({
                                "property_name": prop,
                                "current_value": ml_val,
                                "proposed_value": round(proposed_val, 1),
                                "correction_type": "calibration",
                                "confidence": "HIGH" if kg_distance < 1.5 else "MEDIUM",
                                "reasoning": (
                                    f"Strong KG match to '{kg_name}' (distance={kg_distance:.2f}). "
                                    f"Experimental data shows {prop}={kg_val:.0f}, ML predicted {ml_val:.0f}. "
                                    f"Anchoring with {kg_weight*100:.0f}% KG weight gives {proposed_val:.0f}."
                                ),
                                "source": "KG_anchoring"
                            })

        # === PROPOSAL 5: Elongation Bounds ===
        if ml_el > 0:
            is_cast_poly = processing not in ["wrought", "forged"] and alloy_class != "sc_ds"
            high_cap = ELONGATION["HIGH_GP_MAX_EL_CAST"] if is_cast_poly else ELONGATION["HIGH_GP_MAX_EL"]
            mod_cap = ELONGATION["MOD_GP_MAX_EL_CAST"] if is_cast_poly else ELONGATION["MOD_GP_MAX_EL"]

            if gp > 60 and ml_el > high_cap:
                proposals.append({
                    "property_name": "Elongation",
                    "current_value": ml_el,
                    "proposed_value": high_cap,
                    "correction_type": "bounds",
                    "confidence": "HIGH",
                    "reasoning": (
                        f"High γ' content ({gp:.0f}%) severely limits ductility. "
                        f"ML predicted {ml_el:.1f}% but empirical max for >60% γ' "
                        f"{'cast polycrystalline' if is_cast_poly else 'alloys'} is {high_cap}%."
                    ),
                    "source": "elongation_bounds"
                })
            elif gp > 40 and ml_el > mod_cap:
                proposals.append({
                    "property_name": "Elongation",
                    "current_value": ml_el,
                    "proposed_value": mod_cap,
                    "correction_type": "bounds",
                    "confidence": "MEDIUM",
                    "reasoning": (
                        f"Moderate γ' content ({gp:.0f}%) limits ductility. "
                        f"ML predicted {ml_el:.1f}% but typical max for 40-60% γ' "
                        f"{'cast polycrystalline' if is_cast_poly else 'alloys'} is {mod_cap}%."
                    ),
                    "source": "elongation_bounds"
                })

        # === PROPOSAL 5B: Elastic Modulus Correction ===
        physics_em = physics_pred.get("Elastic Modulus", 0)
        if ml_em > 0 and physics_em > 0:
            em_deviation = abs(ml_em - physics_em) / physics_em * 100
            if em_deviation > 15:
                # EM is more deterministic than strength — weight physics more heavily
                if alloy_class == "sss":
                    blend_ml, blend_phys = 0.20, 0.80
                    model_name = "SSS typical (Reuss-validated)"
                    em_confidence = "HIGH"
                else:
                    blend_ml, blend_phys = 0.30, 0.70
                    model_name = "Reuss bound (harmonic mixing)"
                    em_confidence = "HIGH" if em_deviation > 20 else "MEDIUM"

                proposed_em = blend_ml * ml_em + blend_phys * physics_em
                proposals.append({
                    "property_name": "Elastic Modulus",
                    "current_value": ml_em,
                    "proposed_value": round(proposed_em, 1),
                    "correction_type": "physics",
                    "confidence": em_confidence,
                    "reasoning": (
                        f"ML predicted EM={ml_em:.1f} GPa but {model_name} gives "
                        f"{physics_em:.1f} GPa ({em_deviation:.0f}% deviation). "
                        f"EM depends primarily on atomic bonding stiffness and is well-predicted "
                        f"by composition-weighted models. "
                        f"Proposing {blend_phys*100:.0f}% physics + {blend_ml*100:.0f}% ML "
                        f"blend = {proposed_em:.1f} GPa."
                    ),
                    "source": "EM_physics_blend"
                })

        # === PROPOSAL 6: Calibration Factor (systematic bias) ===
        params = get_params(processing)
        cal_ys = params.get("CAL_YS_FACTOR", 1.0)
        cal_uts = params.get("CAL_UTS_FACTOR", 1.0)

        if cal_ys != 1.0 and ml_ys > 0 and kg_distance > 3.0:
            proposed_ys = ml_ys * cal_ys
            proposals.append({
                "property_name": "Yield Strength",
                "current_value": ml_ys,
                "proposed_value": round(proposed_ys, 1),
                "correction_type": "calibration",
                "confidence": "LOW",
                "reasoning": (
                    f"Systematic calibration factor ({cal_ys:.2f}) for {processing} alloys. "
                    f"Historical validation shows ML tends to over-predict by ~{(1-cal_ys)*100:.0f}% "
                    f"for this processing type. Consider applying if no better reference available."
                ),
                "source": "systematic_calibration"
            })

        # Deduplicate: keep best proposal per property (confidence rank, then source priority)
        source_priority = {"KG_anchoring": 2, "systematic_calibration": 0}
        seen = {}
        for p in proposals:
            name = p["property_name"]
            conf_rank = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}.get(p["confidence"], 0)
            src_rank = source_priority.get(p["source"], 1)
            rank = (conf_rank, src_rank)
            if name not in seen or rank > seen[name][1]:
                seen[name] = (p, rank)
        proposals = [v[0] for v in seen.values()]

        return proposals

    def _detect_discrepancy(
        self,
        ml_pred: Dict[str, Any],
        physics_pred: Dict[str, Any],
        kg_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Detect significant discrepancies between ML and physics predictions."""
        discrepancy = {
            "detected": False,
            "severity": "none",
            "properties_affected": [],
            "details": [],
            "investigation_needed": False,
            "suggested_action": None
        }

        ml_ys = ml_pred.get("Yield Strength", 0)
        physics_ys = physics_pred.get("Yield Strength", 0)
        ml_uts = ml_pred.get("Tensile Strength", 0)
        physics_uts = physics_pred.get("Tensile Strength", 0)

        kg_ys = kg_data.get("properties", {}).get("Yield Strength") if kg_data.get("matched") else None
        kg_distance = kg_data.get("distance", 999) if kg_data.get("matched") else 999

        if ml_ys > 0 and physics_ys > 0:
            ys_diff_pct = abs(ml_ys - physics_ys) / max(ml_ys, physics_ys) * 100

            if ys_diff_pct > 30:
                discrepancy["detected"] = True
                discrepancy["severity"] = "high"
                discrepancy["properties_affected"].append("Yield Strength")
                discrepancy["details"].append({
                    "property": "Yield Strength",
                    "ml_value": ml_ys,
                    "physics_value": physics_ys,
                    "kg_value": kg_ys,
                    "difference_pct": round(ys_diff_pct, 1),
                    "analysis": (
                        f"ML predicts {ml_ys:.0f} MPa, Physics predicts {physics_ys:.0f} MPa "
                        f"({ys_diff_pct:.0f}% difference). "
                        f"{'KG shows ' + str(kg_ys) + ' MPa.' if kg_ys else 'No KG data available.'}"
                    )
                })
            elif ys_diff_pct > 20:
                discrepancy["detected"] = True
                discrepancy["severity"] = "moderate" if discrepancy["severity"] == "none" else discrepancy["severity"]
                discrepancy["properties_affected"].append("Yield Strength")
                discrepancy["details"].append({
                    "property": "Yield Strength",
                    "ml_value": ml_ys,
                    "physics_value": physics_ys,
                    "kg_value": kg_ys,
                    "difference_pct": round(ys_diff_pct, 1),
                    "analysis": (
                        f"Moderate disagreement: ML={ml_ys:.0f} MPa, Physics={physics_ys:.0f} MPa "
                        f"({ys_diff_pct:.0f}% difference)."
                    )
                })

        if ml_uts > 0 and physics_uts > 0:
            uts_diff_pct = abs(ml_uts - physics_uts) / max(ml_uts, physics_uts) * 100

            if uts_diff_pct > 25:
                discrepancy["detected"] = True
                if uts_diff_pct > 35:
                    discrepancy["severity"] = "high"
                elif discrepancy["severity"] == "none":
                    discrepancy["severity"] = "moderate"
                discrepancy["properties_affected"].append("Tensile Strength")
                discrepancy["details"].append({
                    "property": "Tensile Strength",
                    "ml_value": ml_uts,
                    "physics_value": physics_uts,
                    "difference_pct": round(uts_diff_pct, 1),
                    "analysis": f"UTS disagreement: ML={ml_uts:.0f} MPa, Physics={physics_uts:.0f} MPa"
                })

        ml_em = ml_pred.get("Elastic Modulus", 0)
        physics_em = physics_pred.get("Elastic Modulus", 0)
        if ml_em > 0 and physics_em > 0:
            em_diff_pct = abs(ml_em - physics_em) / max(ml_em, physics_em) * 100
            if em_diff_pct > 20:
                discrepancy["detected"] = True
                if discrepancy["severity"] == "none":
                    discrepancy["severity"] = "moderate"
                discrepancy["properties_affected"].append("Elastic Modulus")
                discrepancy["details"].append({
                    "property": "Elastic Modulus",
                    "ml_value": ml_em,
                    "physics_value": physics_em,
                    "difference_pct": round(em_diff_pct, 1),
                    "analysis": (
                        f"EM disagreement: ML={ml_em:.1f} GPa, Physics={physics_em:.1f} GPa "
                        f"({em_diff_pct:.0f}% difference). Physics (Reuss bound) is typically "
                        f"more reliable for EM."
                    )
                })

        if discrepancy["detected"]:
            discrepancy["investigation_needed"] = True

            if kg_distance < 2.0 and kg_ys:
                discrepancy["suggested_action"] = (
                    f"KG has a strong match (distance={kg_distance:.2f}) with YS={kg_ys:.0f} MPa. "
                    f"Compare this experimental value against ML ({ml_ys:.0f}) and Physics ({physics_ys:.0f}) "
                    f"to determine which model is more accurate for this alloy type."
                )
            elif kg_distance < 5.0:
                discrepancy["suggested_action"] = (
                    f"KG has a moderate match (distance={kg_distance:.2f}). Use AlloySearchTool "
                    f"to search for similar alloys and validate which prediction source is more reliable."
                )
            else:
                discrepancy["suggested_action"] = (
                    f"No strong KG match available. Use AlloySearchTool to search for similar alloy "
                    f"compositions or alloys of the same class ({physics_pred.get('alloy_class', 'unknown')}). "
                    f"Compare found experimental values to determine which model to trust."
                )

        return discrepancy

    def _run(
        self,
        composition: Dict[str, float],
        temperature_c: int = 20,
        processing: str = "cast",
        kg_context: Optional[str] = None,
        **kwargs: Any
    ) -> str:
        """Run comprehensive alloy analysis."""
        try:
            ml_predictions = self._get_ml_predictions(composition, temperature_c, processing)
            physics_predictions, features = self._get_physics_predictions(composition, temperature_c, processing)
            kg_data = self._parse_kg_context(kg_context, temperature_c, processing)
            density = calculate_density(composition)

            proposals = self._generate_proposals(
                ml_predictions, physics_predictions, kg_data,
                composition, temperature_c, processing
            )
            discrepancy = self._detect_discrepancy(ml_predictions, physics_predictions, kg_data)
            output = {
                "analysis_type": "agent_driven_corrections",

                "discrepancy": discrepancy,
                "discrepancy_detected": discrepancy["detected"],

                "predictions": {
                    "ml": ml_predictions,
                    "physics": physics_predictions,
                    "kg": kg_data if kg_data.get("matched") else None
                },

                "alloy_analysis": {
                    "class": physics_predictions.get("alloy_class", "unknown"),
                    "class_description": self._get_class_description(physics_predictions.get("alloy_class")),
                    "gamma_prime_pct": physics_predictions.get("gamma_prime_pct", features.get("gamma_prime_estimated_vol_pct", 0)),
                    "sss_wt_pct": physics_predictions.get("sss_wt_pct", 0),
                    "density_gcm3": round(density, 2),
                    "processing": processing,
                    "temperature_c": temperature_c,
                    "kg_match": {
                        "name": kg_data.get("name"),
                        "distance": kg_data.get("distance"),
                        "quality": self._get_match_quality(kg_data.get("distance", 999))
                    } if kg_data.get("matched") else None
                },

                "proposed_corrections": proposals,
                "correction_count": len(proposals),

                "metallurgy_metrics": {
                    "Md_gamma": features.get("Md_gamma", 0),
                    "lattice_mismatch_pct": features.get("lattice_mismatch_pct", 0),
                    "tcp_risk": self._assess_tcp_risk(features.get("Md_gamma", 0), features.get("Md_avg", 0)),
                    "refractory_wt_pct": features.get("refractory_total_wt_pct", 0)
                },

                "agent_instructions": self._get_agent_instructions(discrepancy, proposals)
            }

            return json.dumps(_sanitize_for_json(output), indent=2)

        except Exception as e:
            logger.error("AlloyAnalysisTool error: %s", e)
            return json.dumps({
                "error": str(e),
                "analysis_type": "failed"
            })

    def _get_class_description(self, alloy_class: str) -> str:
        """Get human-readable description of alloy class."""
        descriptions = {
            "sss": "Solid Solution Strengthened (no γ' precipitation, relies on Cr/Mo/W for strength)",
            "gp": "Gamma Prime (γ') Precipitation Strengthened (Al+Ti+Ta forms coherent precipitates)",
            "sc_ds": "Single Crystal / Directionally Solidified (optimized grain structure for creep)"
        }
        return descriptions.get(alloy_class, "Unknown alloy class")

    def _get_match_quality(self, distance: float) -> str:
        """Assess KG match quality."""
        if distance < 1.0:
            return "EXCELLENT (near-identical composition)"
        elif distance < 2.0:
            return "GOOD (similar composition)"
        elif distance < 3.5:
            return "MODERATE (comparable composition)"
        else:
            return "WEAK (exploratory composition)"

    def _assess_tcp_risk(self, md_gamma: float, md_avg: float = 0.0) -> str:
        """Assess TCP phase risk — delegates to centralized classify_tcp_risk."""
        return classify_tcp_risk(md_gamma, md_avg)

    def _get_agent_instructions(self, discrepancy: Dict[str, Any], proposals: List[Dict]) -> str:
        """Generate context-aware instructions for the agent."""
        if discrepancy.get("detected"):
            severity = discrepancy.get("severity", "moderate")
            affected = ", ".join(discrepancy.get("properties_affected", []))

            return (
                f"DISCREPANCY DETECTED ({severity}) for: {affected}. "
                f"Use AlloySearchTool to investigate before deciding. "
                f"Review {len(proposals)} proposals and ACCEPT or REJECT each."
            )
        else:
            return (
                f"Sources agree. Review {len(proposals)} proposed corrections. "
                f"ACCEPT or REJECT each based on the reasoning provided."
            )
