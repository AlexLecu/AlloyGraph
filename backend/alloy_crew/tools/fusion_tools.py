from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type, Dict, Any
import json
import math

import logging

logger = logging.getLogger(__name__)

from ..models.feature_engineering import compute_alloy_features

class DataFusionInput(BaseModel):
    """Input for data fusion."""
    composition: Dict[str, float] = Field(..., description="The composition of the alloy being evaluated.")
    ml_prediction_json: str = Field(..., description="JSON output from ML Property Predictor.")
    rag_context: str = Field(..., description="JSON string from RAG search tools containing known alloy records.")
    target_temperature_c: float = Field(..., description="The temperature at which to evaluate properties (Celsius).")
    processing: str = Field(default="unknown", description="Processing type: 'cast', 'wrought', or 'unknown'.")
    mode: str = Field(default="evaluate", description="Fusion mode: 'design' (trust ML more) or 'evaluate' (trust KG more).")

class DataFusionTool(BaseTool):
    name: str = "DataFusionTool"
    description: str = (
        "Reconciles ML predictions with Knowledge Graph empirical data. "
        "Logic: If RAG results contain a high-similarity match, it anchors the results toward that 'truth'. "
    )
    args_schema: Type[BaseModel] = DataFusionInput

    def _smooth_kg_weight(self, distance: float, pivot: float = 2.5, softness: float = 0.5, mode: str = "evaluate") -> float:
        """
        Calculate smooth KG weight using sigmoid function.
        
        Eliminates discontinuities from hard thresholds by providing gradual transition.
        Weight smoothly decreases as compositional distance increases.
        
        Args:
            distance: Compositional distance to KG candidate
            pivot: Center point of transition (d₀) - 50% weight at this distance
            softness: Controls transition steepness (s) - higher = more gradual
            mode: "evaluate" (trust KG heavily) or "design" (trust ML more)
        
        Returns:
            KG weight between 0 and 1
            
        Examples:
            distance=0.0 (exact): ~1.00 (100% KG)
            distance=2.0: ~0.98 (98% KG)
            distance=2.5: 0.50 (50% KG - pivot point)
            distance=3.0: ~0.02 (2% KG)
            distance=4.0: ~0.001 (0.1% KG, essentially pure ML)
        """
        base_weight = 1.0 / (1.0 + math.exp((distance - pivot) / softness))

        if mode == "design":
            return base_weight * 0.3
        else:
            return base_weight

    def _calculate_confidence(
        self, 
        kg_weight: float,
        similarity_dist: float,
        temp_delta: float,
        matched_alloy_name: str,
        ml_confidence: float = 0.7,
        fusion_agreement: float = 0.5
    ) -> dict:
        """Calculate confidence score from KG match quality and temperature delta."""
        if abs(temp_delta) < 5:
            temp_quality = 1.0
        elif abs(temp_delta) < 50:
            temp_quality = 0.85
        else:
            temp_quality = 0.60
        
        # Simple weighted combination
        final_score = (kg_weight * 0.65) + (temp_quality * 0.15) + (ml_confidence * 0.1) + (fusion_agreement * 0.1)
        
        if final_score >= 0.85:
            confidence_level = "HIGH"
        elif final_score >= 0.60:
            confidence_level = "MEDIUM"
        else:
            confidence_level = "LOW"

        # Fusion Weighting Transparency
        ml_weight = 1.0 - kg_weight
        kg_pct = round(kg_weight * 100, 1)
        ml_pct = round(ml_weight * 100, 1)

        # Generate clear decision rationale
        if similarity_dist < 0.5:
            rationale = f"Excellent match to {matched_alloy_name} (distance={similarity_dist:.2f}) → Heavily anchored to experimental data"
        elif similarity_dist < 2.0:
            rationale = f"Good similarity to {matched_alloy_name} (distance={similarity_dist:.2f}) → Blended experimental + ML"
        elif similarity_dist < 3.5:
            rationale = f"Moderate match to {matched_alloy_name} (distance={similarity_dist:.2f}) → Balanced fusion"
        else:
            rationale = f"Weak match (distance={similarity_dist:.2f}) → ML-dominated prediction"

        return {
            "score": round(final_score, 3),
            "level": confidence_level,
            "breakdown": {
                "kg_confidence": round(kg_weight, 3),
                "ml_confidence": round(ml_confidence, 3),
                "fusion_agreement": round(fusion_agreement, 3),
                "weighting_strategy": "kg_focused" if kg_weight > 0.7 else "balanced"
            },
            "kg_weight_used": round(kg_weight, 3),
            "similarity_distance": round(similarity_dist, 4),
            "temperature_delta": round(abs(temp_delta), 1),
            "matched_alloy": matched_alloy_name,
            # Explicit fusion weighting breakdown
            "fusion_weighting": {
                "kg_contribution_pct": kg_pct,
                "ml_contribution_pct": ml_pct,
                "decision_rationale": rationale,
                "data_source_primary": "Knowledge Graph" if kg_weight > 0.5 else "ML Model"
            }
        }

    def _infer_processing_type(self, composition: Dict[str, float]) -> str:
        """Infers processing type based on chemical heuristics."""
        al_ti = composition.get("Al", 0) + composition.get("Ti", 0)
        if (composition.get("B", 0) > 0.015 or composition.get("Zr", 0) > 0.05) and al_ti > 5.5:
             return "cast"
        if "cast" in str(composition).lower():
             return "cast"
        return "wrought"

    def _calculate_fusion_agreement(self, ml_props: Dict[str, float], kg_props: Dict[str, float]) -> float:
        """Calculate agreement between ML and KG predictions (0-1 scale)."""
        if not kg_props:
            return 0.5
        
        agreements = []
        for prop in ["Yield Strength", "Tensile Strength", "Elongation", "Elastic Modulus"]:
            ml_val = ml_props.get(prop, 0)
            kg_val = kg_props.get(prop, 0)
            
            if ml_val > 0 and kg_val > 0:
                rel_diff = abs(ml_val - kg_val) / max(ml_val, kg_val)
                agreement = max(0.0, 1.0 - rel_diff)
                agreements.append(agreement)
        
        return sum(agreements) / len(agreements) if agreements else 0.5

    def _parse_property_string(self, prop_str: str, target_temp: float, is_strength: bool = True) -> tuple:
        """Parse property string like '725.0 MPa @ 538.0C' and return (value, temp) if match."""
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
                if abs(temp - target_temp) < 50:
                    return (val, temp)
            except:
                continue
        return (None, None)

    def _extract_properties_from_candidate(self, candidate: Dict[str, Any], target_temp: float) -> tuple:
        """Extracts properties from candidate, trying list format first, then string format."""
        extracted = {}
        matched_temp = target_temp
        
        property_map = [
            ("yield_strength", "Yield Strength", "YieldStrength"), 
            ("uts", "Tensile Strength", "UTS"), 
            ("elongation", "Elongation", "Elongation"),
            ("elasticity", "Elastic Modulus", "ElasticModulus")
        ]

        for list_key, target_key, alt_key in property_map:
            data_points = candidate.get(list_key, [])
            if isinstance(data_points, list) and data_points:
                for dp in data_points:
                    try:
                        temp = float(dp.get("temp_c", "21"))
                        if abs(temp - target_temp) < 50:
                            extracted[target_key] = float(dp.get("value", 0))
                            matched_temp = temp
                            break
                    except: 
                        pass
            
            if target_key not in extracted:
                rag_props = candidate.get("properties", {})
                prop_str = rag_props.get(alt_key, "")
                if prop_str:
                    is_strength = (target_key != "Elastic Modulus")
                    val, temp = self._parse_property_string(prop_str, target_temp, is_strength=is_strength)
                    if val is not None:
                        extracted[target_key] = val
                        matched_temp = temp

        return extracted, matched_temp

    def _run(self, composition: Dict[str, float], ml_prediction_json: str, rag_context: str, target_temperature_c: float, **kwargs: Any) -> str:
        try:
            # 1. Parse ML Input
            clean_json = ml_prediction_json.replace("```json", "").replace("```", "").strip()
            if "{" in clean_json:
                 start = clean_json.find("{")
                 end = clean_json.rfind("}") + 1
                 clean_json = clean_json[start:end]
            try:
                raw_pred = json.loads(clean_json)
                if "error" in raw_pred: return json.dumps({"error": f"ML Tool Error: {raw_pred['error']}"})
            except Exception as e:
                return json.dumps({"error": f"DataFusion Failed: ML Parse Error. {e}"})

            # Helper to safely get ML values
            def get_val(d, keys):
                for k in keys:
                    if k in d: return float(d[k])
                    k_fixed = k.lower().replace(" ", "_")
                    for rk in d.keys():
                        if rk.lower().replace(" ", "_") == k_fixed: return float(d[rk])
                return 0.0
            
            raw_ys = get_val(raw_pred, ["Yield Strength", "yield_strength", "ys"])
            raw_ts = get_val(raw_pred, ["Tensile Strength", "tensile_strength", "uts"])
            raw_el = get_val(raw_pred, ["Elongation", "elongation", "el"])
            raw_em = get_val(raw_pred, ["Elastic Modulus", "elastic_modulus", "em", "elasticity"])
            
            ml_conf_dict = raw_pred.get("model_confidence", {})
            if ml_conf_dict and "Yield Strength" in ml_conf_dict:
                ml_confidence = float(ml_conf_dict.get("Yield Strength", 0.7))
            else:
                ml_confidence = 0.70
            
            ml_intervals = raw_pred.get("property_intervals", {})
            def get_uncert(prop):
                if prop in ml_intervals and isinstance(ml_intervals[prop], dict):
                    return float(ml_intervals[prop].get("uncertainty", 0.0))
                for k in ml_intervals.keys():
                    if k.lower().replace(" ", "_") == prop.lower().replace(" ", "_"):
                         if isinstance(ml_intervals[k], dict):
                            return float(ml_intervals[k].get("uncertainty", 0.0))
                return 0.0

            ys_base_uncertainty = get_uncert("Yield Strength")
            ts_base_uncertainty = get_uncert("Tensile Strength")
            el_base_uncertainty = get_uncert("Elongation")
            em_base_uncertainty = get_uncert("Elastic Modulus")

            matched_candidate = None
            similarity_dist = 999.0
            detected_family = "unknown"

            # Extract user's processing choice first (before KG matching)
            input_processing = kwargs.get("processing", "unknown")
            if input_processing == "unknown":
                input_processing = self._infer_processing_type(composition)

            try:
                candidates = json.loads(rag_context)

                logger.info(f"KG Search: Mode={kwargs.get('mode', 'evaluate')}, Found={len(candidates) if candidates else 0} candidates")

                if candidates and isinstance(candidates, list) and len(candidates) > 0:
                    matched_candidate = candidates[0]
                    similarity_dist = matched_candidate.get("_distance", 0.0)

                    # Log selected match
                    logger.info(f"Selected KG Match: {matched_candidate.get('name', 'Unknown')} (dist={similarity_dist})")
                    
                    if similarity_dist == 0.0 and "_distance" not in matched_candidate:
                         logger.warning("KG Match missing _distance field, defaulting to 0.0")
                    
                    raw_proc = matched_candidate.get("processing", "unknown").lower()
                    if any(x in raw_proc for x in ["cast", "crystal", "ds"]):
                        detected_family = "cast"
                    else:
                        detected_family = "wrought"

                    # Check for processing mismatch (user vs KG candidate)
                    cand_proc = matched_candidate.get("processing", "unknown").lower()
                    processing_mismatch = False
                    if input_processing == "cast" and "cast" not in cand_proc:
                        processing_mismatch = True
                    elif input_processing == "wrought" and "cast" in cand_proc:
                        processing_mismatch = True
                    
                    if processing_mismatch and input_processing != "unknown":
                        matched_candidate = None
                        similarity_dist = 999.0

            except Exception as e:
                logger.error(f"KG Search processing failed: {e}")

            # 3. ANCHORING DECISION
            kg_props = {}
            kg_metallurgy = {}
            kg_note = f"No Match (<5.0) found."
            is_valid_match = False
            kg_matched_temp = target_temperature_c  # Default to target

            if matched_candidate:
                kg_props, kg_matched_temp = self._extract_properties_from_candidate(matched_candidate, target_temperature_c)
                
                metallurgy_data = matched_candidate.get("metallurgy", {})
                if metallurgy_data:
                    kg_metallurgy = {
                        "Density": metallurgy_data.get("density_gcm3"),
                        "Gamma Prime": metallurgy_data.get("gamma_prime_vol"),
                        "Md_avg": metallurgy_data.get("md_avg"),
                        "TCP_risk": metallurgy_data.get("tcp_risk"),
                        "SSS_wt_pct": metallurgy_data.get("sss_wt_pct")
                    }
                
                if kg_props:
                    is_valid_match = True
                    kg_note = f"Anchoring to {matched_candidate.get('name')} (Dist: {similarity_dist:.2f})"


            # 4. FUSION WEIGHTING - Smooth sigmoid-based weighting
            if is_valid_match:
                # Get mode from input (design vs evaluate)
                fusion_mode = kwargs.get("mode", "evaluate")
                
                # Use smooth sigmoid weighting based on compositional distance and mode
                kg_weight = self._smooth_kg_weight(similarity_dist, pivot=2.5, softness=0.5, mode=fusion_mode)
                ml_weight = 1.0 - kg_weight
                
                mode_note = f"[{fusion_mode.upper()} MODE]" if fusion_mode == "design" else ""
                kg_note = f"Anchoring to {matched_candidate.get('name')} (Dist: {similarity_dist:.2f}, KG Weight: {kg_weight:.1%}) {mode_note}"
                
                # Calculate fusion agreement between ML and KG predictions
                ml_props = {
                    "Yield Strength": raw_ys,
                    "Tensile Strength": raw_ts,
                    "Elongation": raw_el,
                    "Elastic Modulus": raw_em
                }
                fusion_agreement = self._calculate_fusion_agreement(ml_props, kg_props)
                
                temp_delta = target_temperature_c - kg_matched_temp
                confidence = self._calculate_confidence(
                    kg_weight=kg_weight,
                    similarity_dist=similarity_dist,
                    temp_delta=temp_delta,
                    matched_alloy_name=matched_candidate.get('name', 'Unknown'),
                    ml_confidence=ml_confidence,
                    fusion_agreement=fusion_agreement
                )
            else:
                ml_weight = 1.0
                kg_weight = 0.0

                confidence = {
                    "score": round(ml_confidence * 0.7, 3),
                    "level": "MEDIUM" if ml_confidence > 0.65 else "LOW",
                    "breakdown": {
                        "kg_confidence": 0.0,
                        "ml_confidence": round(ml_confidence, 3),
                        "fusion_agreement": 0.5,
                        "weighting_strategy": "ml_focused"
                    },
                    "kg_weight_used": 0.0,
                    "similarity_distance": 999.0,
                    "temperature_delta": 0.0,
                    "matched_alloy": "None",
                    # Fusion weighting (ML-only case)
                    "fusion_weighting": {
                        "kg_contribution_pct": 0.0,
                        "ml_contribution_pct": 100.0,
                        "decision_rationale": "No similar alloys found in database → Pure ML prediction (exploratory composition)",
                        "data_source_primary": "ML Model"
                    }
                }

            # Calculate fused property values with uncertainty intervals
            final_ys = (raw_ys * ml_weight) + (kg_props.get("Yield Strength", raw_ys) * kg_weight)
            final_ts = (raw_ts * ml_weight) + (kg_props.get("Tensile Strength", raw_ts) * kg_weight)
            final_el = (raw_el * ml_weight) + (kg_props.get("Elongation", raw_el) * kg_weight)
            final_em = (raw_em * ml_weight) + (kg_props.get("Elastic Modulus", raw_em) * kg_weight)
            
            conf_score = float(confidence.get("score", 0.7))
            if conf_score > 0.80:
                uncertainty_multiplier = 0.6  # Narrow intervals for high confidence
            elif conf_score > 0.55:
                uncertainty_multiplier = 1.0  # Base intervals for medium confidence
            else:
                uncertainty_multiplier = 1.8
            
            ys_uncertainty = ys_base_uncertainty * uncertainty_multiplier if ys_base_uncertainty > 0 else final_ys * 0.10
            ts_uncertainty = ts_base_uncertainty * uncertainty_multiplier if ts_base_uncertainty > 0 else final_ts * 0.10
            el_uncertainty = el_base_uncertainty * uncertainty_multiplier if el_base_uncertainty > 0 else final_el * 0.15
            em_uncertainty = em_base_uncertainty * uncertainty_multiplier if em_base_uncertainty > 0 else final_em * 0.05
            
            final_properties_flat = {
                "Yield Strength": round(final_ys, 1),
                "Tensile Strength": round(final_ts, 1),
                "Elongation": round(final_el, 1),
                "Elastic Modulus": round(final_em, 1)
            }
            
            final_intervals = {
                "Yield Strength": {
                    "lower": round(final_ys - ys_uncertainty, 1),
                    "upper": round(final_ys + ys_uncertainty, 1),
                    "uncertainty": round(ys_uncertainty, 1)
                },
                "Tensile Strength": {
                    "lower": round(final_ts - ts_uncertainty, 1),
                    "upper": round(final_ts + ts_uncertainty, 1),
                    "uncertainty": round(ts_uncertainty, 1)
                },
                "Elongation": {
                    "lower": round(max(0.0, final_el - el_uncertainty), 1),
                    "upper": round(final_el + el_uncertainty, 1),
                    "uncertainty": round(el_uncertainty, 1)
                },
                "Elastic Modulus": {
                    "lower": round(final_em - em_uncertainty, 1),
                    "upper": round(final_em + em_uncertainty, 1),
                    "uncertainty": round(em_uncertainty, 1)
                }
            }
            
            # Add KG computed features if available
            if kg_metallurgy.get("Density"):
                dens_val = kg_metallurgy["Density"]
                final_properties_flat["Density"] = dens_val
            if kg_metallurgy.get("Gamma Prime"):
                gp_val = kg_metallurgy["Gamma Prime"]
                final_properties_flat["Gamma Prime"] = gp_val
            
            # FALLBACK: If KG did not provide Density/GP, calculate them now using physics models
            if "Density" not in final_properties_flat or "Gamma Prime" not in final_properties_flat:
                # Calculate features
                calc_features = compute_alloy_features(composition)
                
                if "Density" not in final_properties_flat:
                    dens_val = calc_features.get("density_calculated_gcm3", 8.5)
                    final_properties_flat["Density"] = round(dens_val, 2)
                
                if "Gamma Prime" not in final_properties_flat:
                    # feature_engineering returns fraction (0.0-1.0), convert to %
                    gp_val = calc_features.get("gamma_prime_estimated_vol_pct", 0.0) 
                    final_properties_flat["Gamma Prime"] = round(gp_val, 1)

            metrics = {
                "rag_match_similarity": similarity_dist,
                "data_source": "KG" if is_valid_match else "ML Only"
            }
            
            # Add KG metallurgy metrics if available
            if kg_metallurgy:
                metrics.update({
                    "kg_md_avg": kg_metallurgy.get("Md_avg"),
                    "kg_tcp_risk": kg_metallurgy.get("TCP_risk"),
                    "kg_sss_wt_pct": kg_metallurgy.get("SSS_wt_pct")
                })
            
            final_processing = input_processing if input_processing != "unknown" else detected_family

            output = {
                "summary": f"Data Fusion Complete. Status: {kg_note}",
                "processing": final_processing,
                "anchored_properties": final_properties_flat,
                "property_intervals": final_intervals,
                "metallurgy_metrics": metrics,
                "fusion_meta": {
                     "kg_similarity_max": similarity_dist,
                     "ml_weight": ml_weight,
                     "kg_weight": kg_weight,
                     "data_conflict": False,
                     "is_kg_anchored": is_valid_match
                },
                "confidence": confidence,
                "raw_ml_properties": {
                    "Yield Strength": raw_ys, 
                    "Tensile Strength": raw_ts, 
                    "Elongation": raw_el,
                    "Elastic Modulus": raw_em
                }
            }
            return json.dumps(output)
        except Exception as e:
            return f"Error in DataFusion: {str(e)}"
