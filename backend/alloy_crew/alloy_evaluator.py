from typing import Dict, Any
import json
from crewai import Task, Crew, Process
from .agents import get_evaluation_agents
from .tools.rag_tools import AlloySearchTool
from .schemas import (
    ValidationOutput, ArbitrationOutput,
    PhysicsAuditWithCorrectionsOutput, CorrectedPropertiesOutput,
    AuditPenalty
)
from .tools.calibration_fix import apply_calibration_safe
from .tools.metallurgy_tools import (
    validate_property_coherency, enforce_physics_constraints,
    cleanup_llm_output, cleanup_confidence, warnings_to_penalties,
    compute_fallback_metrics, PROPERTY_KEY_MAP, VALID_PROPERTIES, REQUIRED_METRIC_KEYS,
)


def _slim_kg_context(kg_json_str: str, target_temp: int = 20) -> str:
    """
    Compress KG context to only essential fields for fusion.
    """
    try:
        alloys = json.loads(kg_json_str)
        slim_alloys = []

        for alloy in alloys[:3]:  # Only top 3 matches
            slim = {
                "name": alloy.get("name", "Unknown"),
                "distance": round(alloy.get("_distance", 999), 2),
                "processing": alloy.get("processing", "unknown"),
            }

            props = alloy.get("properties", {})
            slim_props = {}
            for prop_name, prop_values in props.items():
                if isinstance(prop_values, str):
                    parts = prop_values.split(", ")
                    relevant = []
                    for p in parts[:2]:
                        if f"@ {target_temp}" in p or "@ 20" in p or "@ 25" in p:
                            relevant.append(p.split(" @ ")[0])
                    if relevant:
                        slim_props[prop_name] = ", ".join(relevant)

            if slim_props:
                slim["props"] = slim_props

            slim_alloys.append(slim)
        return json.dumps(slim_alloys, separators=(',', ':'))
    except Exception:
        return kg_json_str[:1500] if len(kg_json_str) > 1500 else kg_json_str

class AlloyEvaluationCrew:
    def __init__(self, llm_config=None):
        # Initialize agents with optional local LLM config
        self.agents_map = get_evaluation_agents(llm=llm_config)
        self.validator = self.agents_map['validator']
        self.arbitrator = self.agents_map['arbitrator']
        self.physicist = self.agents_map['physicist']
        self.summarizer = self.agents_map['summarizer']

    @staticmethod
    def validate_composition(composition: Dict[str, float]) -> Dict[str, Any]:
        """
        Validates and cleans composition by removing non-positive values.
        Returns dict with cleaned composition and any validation warnings.
        """
        warnings = []
        
        # Remove non-positive values
        cleaned = {k: max(0.0, float(v)) for k, v in composition.items() if float(v) > 0}
        
        if not cleaned:
            raise ValueError("Composition is empty after removing non-positives.")

        total = sum(cleaned.values())
        
        # Trust user input if close to 100%
        if 99.5 <= total <= 100.5:
            return {"composition": cleaned, "warnings": warnings}
        
        # Collect warnings for significant deviations
        if total < 95.0:
            warnings.append(f"Composition sums to {total:.1f}%, which seems incomplete. Consider adding missing elements.")
        elif total > 105.0:
            warnings.append(f"Composition sums to {total:.1f}%, which exceeds 100%. This may indicate an error.")
        elif abs(total - 100.0) > 2.0:
            warnings.append(f"Composition sums to {total:.1f}% (expected ~100%).")
        
        # Reject if total is way off
        if total < 90.0 or total > 110.0:
            raise ValueError(
                f"Composition total ({total:.1f}%) is outside acceptable range (90-110%). "
                f"Please check your input values."
            )

        return {
            "composition": {k: round(v, 3) for k, v in cleaned.items()},
            "warnings": warnings
        }
            
            
    def run(self, composition: dict, processing: str = "wrought", temperature: int = 900) -> Dict[str, Any]:
        """
        Runs the Physics Audit evaluation.
        Args:
            composition: Dict of element wt%.
            processing: Alloy processing hint.
            temperature: Target temp in C.
        """
        try:
            validation_result = self.validate_composition(composition)
            current_comp = validation_result["composition"]
        except Exception as e:
            return {"status": "FAIL", "stage": "validation", "error": str(e)}
        
        # 0. KG Context (Pre-Agent Lookup for robustness)
        try:
            search_tool = AlloySearchTool()
            kg_raw = search_tool._run(composition=current_comp, limit=3)
            kg_context_str = _slim_kg_context(kg_raw, target_temp=temperature)
        except Exception as e:
            return {"status": "FAIL", "stage": "kg_lookup", "error": str(e)}

        comp_json = json.dumps(current_comp, ensure_ascii=False)
        
        # --- TASK 1: VALIDATION ---
        task_validation = Task(
            description=(
                f"Validate composition at {temperature}°C using AlloyPredictorTool.\n\n"
                f"Execute: AlloyPredictorTool(\n"
                f"  composition={comp_json},\n"
                f"  temperature_c={temperature},\n"
                f"  processing='{processing}'\n"
                f")\n\n"
                "Return ONLY the tool's output. Do NOT invent or modify any values."
            ),
            expected_output="Valid structured output validating the alloy composition.",
            output_pydantic=ValidationOutput,
            agent=self.validator
        )

        # --- TASK 2: ARBITRATION ---
        task_arbitration = Task(
            description=(
                f"Fuse ML predictions with KG data using DataFusionTool.\n\n"
                f"Execute DataFusionTool with these INPUT parameters ONLY:\n"
                f"  composition: {comp_json}\n"
                f"  ml_prediction_json: <JSON string from Validator's ml_prediction>\n"
                f"  rag_context: {kg_context_str}\n"
                f"  target_temperature_c: {temperature}\n"
                f"  processing: '{processing}'\n"
                f"  mode: 'evaluate'  (standard 50/50 ML/KG weighting for evaluation)\n\n"
                "DO NOT add any other parameters. The tool will return properties, property_intervals, and confidence.\n"
                "Return the tool's complete output exactly as provided."
            ),
            expected_output="Fused properties with confidence scores.",
            output_pydantic=ArbitrationOutput,
            agent=self.arbitrator,
            context=[task_validation]
        )

        # --- TASK 3: PHYSICS AUDIT + CORRECTIONS (merged) ---
        task_physics = Task(
            description=(
                f"Audit and correct physical validity of composition using MetallurgyVerifierTool.\n\n"
                "1. Infer alloy_type using priority order:\n"
                "   - 'high_corrosion' if Cr > 21%\n"
                "   - 'high_strength' if Al+Ti > 4% AND Cr ≤ 21%\n"
                "   - 'standard' otherwise\n\n"
                f"2. Execute MetallurgyVerifierTool with these INPUT parameters ONLY:\n"
                f"   composition: {comp_json}\n"
                f"   anchored_properties_json: <JSON with {{properties, processing}} from Arbitrator>\n"
                f"   temperature_c: {temperature}\n"
                "   alloy_type: <inferred_type>\n\n"
                "   DO NOT add output fields like 'status', 'tcp_risk', 'audit_penalties', etc.\n"
                "   Those will be returned by the tool.\n\n"
                "3. PRESERVE the complete tool output including 'corrections_applied'\n\n"
                "4. Add ONLY a 3-5 sentence human-readable 'explanation':\n"
                "   - Identify dominant strengthening mechanism (γ' vs solid solution)\n"
                "   - Note any corrections applied and why\n"
                "   - Suggest suitable applications\n"
                "   - DO NOT modify any numerical values in your explanation"
            ),
            expected_output="Complete physics audit with corrections applied.",
            output_pydantic=PhysicsAuditWithCorrectionsOutput,
            agent=self.physicist,
            context=[task_arbitration]
        )

        # Create and run crew
        evaluation_crew = Crew(
            agents=[self.validator, self.arbitrator, self.physicist],
            tasks=[task_validation, task_arbitration, task_physics],
            process=Process.sequential,
            verbose=True
        )

        try:
            crew_output = evaluation_crew.kickoff()

            # Extract structured output
            physics_output = None
            if hasattr(crew_output, "pydantic") and crew_output.pydantic:
                physics_output = crew_output.pydantic
            elif hasattr(crew_output, "raw"):
                # Fallback: attempt manual parse
                try:
                    data = json.loads(crew_output.raw)
                    physics_output = PhysicsAuditWithCorrectionsOutput(**data)
                except:
                    print(f"⚠️  Failed to parse LLM output as JSON, attempting recovery...")
                    physics_output = None

            if physics_output is None:
                # Try to get from task output
                last_task_output = task_physics.output
                if last_task_output and last_task_output.pydantic:
                    physics_output = last_task_output.pydantic
                else:
                    # Final fallback: construct from validator
                    print("⚠️  Physics task failed, falling back to validator output...")
                    validator_output = getattr(task_validation.output, "pydantic", None)

                    if validator_output:
                        props = validator_output.ml_prediction
                        physics_output = PhysicsAuditWithCorrectionsOutput(
                            status='PASS',
                            processing=processing,
                            penalty_score=0,
                            tcp_risk='LOW',
                            properties=props,
                            property_intervals={},
                            metallurgy_metrics={},
                            audit_penalties=[],
                            confidence={},
                            explanation="Analysis completed with fallback processing due to LLM parsing issues.",
                            corrections_applied=[],
                            corrections_explanation=""
                        )
                    else:
                        raise ValueError("Could not recover output from any pipeline stage.")

            # Convert PhysicsAuditWithCorrectionsOutput to CorrectedPropertiesOutput for compatibility
            corrected_output = CorrectedPropertiesOutput(
                status=getattr(physics_output, 'status', 'PASS'),
                processing=getattr(physics_output, 'processing', processing),
                penalty_score=getattr(physics_output, 'penalty_score', 0),
                tcp_risk=getattr(physics_output, 'tcp_risk', 'LOW'),
                properties=getattr(physics_output, 'properties', {}),
                property_intervals=getattr(physics_output, 'property_intervals', {}),
                metallurgy_metrics=getattr(physics_output, 'metallurgy_metrics', {}),
                audit_penalties=getattr(physics_output, 'audit_penalties', []),
                confidence=getattr(physics_output, 'confidence', {}),
                explanation=getattr(physics_output, 'explanation', ''),
                corrections_applied=getattr(physics_output, 'corrections_applied', []),
                corrections_explanation=getattr(physics_output, 'corrections_explanation', '')
            )

        except Exception as e:
            return {"status": "FAIL", "stage": "crew_execution", "error": str(e)}

        # === PROPERTY RECOVERY ===
        # LLM agents sometimes drop properties. Recover from earlier pipeline stages.
        required_props = ["Yield Strength", "Tensile Strength", "Elongation", "Elastic Modulus", "Density", "Gamma Prime"]
        missing_props = [p for p in required_props if p not in corrected_output.properties or corrected_output.properties.get(p) in [None, 0, "N/A"]]

        if missing_props:
            print(f"⚠️  Missing properties detected: {missing_props}. Attempting recovery...")

            # Try to recover from validator's ml_prediction
            validator_output = getattr(task_validation.output, "pydantic", None)
            if validator_output and hasattr(validator_output, 'ml_prediction'):
                ml_pred = validator_output.ml_prediction
                for prop in missing_props[:]:
                    if prop in ml_pred and ml_pred[prop] not in [None, 0]:
                        corrected_output.properties[prop] = ml_pred[prop]
                        missing_props.remove(prop)
                        print(f"  ✓ Recovered {prop} from ML prediction: {ml_pred[prop]}")

            # Try to recover confidence, intervals, metrics from earlier stages
            if not corrected_output.confidence or corrected_output.confidence == {}:
                physicist_output = getattr(task_physics.output, "pydantic", None)
                if physicist_output and hasattr(physicist_output, 'confidence'):
                    corrected_output.confidence = physicist_output.confidence
                    print("  ✓ Recovered confidence from Physicist output")

            if not corrected_output.property_intervals or corrected_output.property_intervals == {}:
                if validator_output and hasattr(validator_output, 'ml_prediction'):
                    intervals = validator_output.ml_prediction.get("property_intervals", {})
                    if intervals:
                        corrected_output.property_intervals = intervals
                        print("  ✓ Recovered property_intervals from ML prediction")

            # Recover metallurgy_metrics from Physicist output if missing required keys
            metrics = corrected_output.metallurgy_metrics or {}
            has_required_metrics = any(k in metrics for k in REQUIRED_METRIC_KEYS)

            if not has_required_metrics:
                physicist_output = getattr(task_physics.output, "pydantic", None)
                if physicist_output and hasattr(physicist_output, 'metallurgy_metrics'):
                    phys_metrics = physicist_output.metallurgy_metrics
                    if phys_metrics and any(k in phys_metrics for k in REQUIRED_METRIC_KEYS):
                        corrected_output.metallurgy_metrics = phys_metrics
                        print("  ✓ Recovered metallurgy_metrics from Physicist output")
                        has_required_metrics = True

                if not has_required_metrics:
                    corrected_output.metallurgy_metrics = compute_fallback_metrics(composition)
                    print("  ✓ Computed metallurgy_metrics from feature_engineering")

            # Final fallback: compute from feature_engineering
            if missing_props:
                from .models.feature_engineering import compute_alloy_features
                features = compute_alloy_features(composition)
                if "Density" in missing_props and "density_calculated_gcm3" in features:
                    corrected_output.properties["Density"] = round(features["density_calculated_gcm3"], 2)
                    missing_props.remove("Density")
                    print(f"  ✓ Computed Density from features: {corrected_output.properties['Density']}")
                if "Gamma Prime" in missing_props and "gamma_prime_estimated_vol_pct" in features:
                    # SSS alloy check: Al+Ti+Ta < 2% means γ' should be 0%
                    al = composition.get("Al", composition.get("al", 0)) or 0
                    ti = composition.get("Ti", composition.get("ti", 0)) or 0
                    ta = composition.get("Ta", composition.get("ta", 0)) or 0
                    al_ti_ta = al + ti + ta

                    if al_ti_ta < 2.0:
                        corrected_output.properties["Gamma Prime"] = 0.0
                        print(f"  ✓ SSS alloy (Al+Ti+Ta={al_ti_ta:.1f}% < 2%) - Gamma Prime set to 0%")
                    else:
                        corrected_output.properties["Gamma Prime"] = round(features["gamma_prime_estimated_vol_pct"], 1)
                        print(f"  ✓ Computed Gamma Prime from features: {corrected_output.properties['Gamma Prime']}")
                    missing_props.remove("Gamma Prime")

            if missing_props:
                print(f"  ⚠️  Could not recover: {missing_props}")

        corrected_output.properties = apply_calibration_safe(
            corrected_output.properties,
            composition,
            corrected_output
        )

        if corrected_output.corrections_explanation:
            corrected_output.corrections_explanation += "\n\nDatabase-driven calibration applied to account for systematic formula biases in literature equations."
        else:
            corrected_output.corrections_explanation = "Database-driven calibration applied to account for systematic formula biases in literature equations."

        # === PHYSICS ENFORCEMENT (Hard Constraints) ===
        # Programmatic corrections for extreme deviations that LLM agents may have missed
        confidence = corrected_output.confidence if isinstance(corrected_output.confidence, dict) else {}
        kg_distance = confidence.get("similarity_distance", 999)
        confidence_level = confidence.get("level", "MEDIUM")

        corrected_output.properties, physics_corrections = enforce_physics_constraints(
            properties=corrected_output.properties,
            composition=composition,
            temperature_c=temperature,
            processing=processing,
            confidence_level=confidence_level,
            kg_distance=kg_distance
        )

        if physics_corrections:
            print(f"⚡ Physics enforcement applied {len(physics_corrections)} corrections:")
            for corr in physics_corrections:
                print(f"   - {corr}")
            # Add to corrections explanation
            corrected_output.corrections_explanation += "\n\nPhysics enforcement corrections:\n" + "\n".join(f"• {c}" for c in physics_corrections)

        # Re-run coherency checks with POST-calibration values
        non_coherency_penalties = [
            p for p in corrected_output.audit_penalties
            if not any(x in p.name.lower() for x in ["coherency", "mismatch"])
        ]
        fresh_warnings = validate_property_coherency(corrected_output.properties, composition)
        fresh_penalties = [AuditPenalty(**p) for p in warnings_to_penalties(fresh_warnings)]
        corrected_output.audit_penalties = non_coherency_penalties + fresh_penalties

        # Generate summary with POST-calibration values to avoid mismatch
        try:
            comp_str = ", ".join([f"{elem}: {wt:.1f}%" for elem, wt in sorted(composition.items(), key=lambda x: x[1], reverse=True)[:5]])
            props = corrected_output.properties
            confidence = corrected_output.confidence if isinstance(corrected_output.confidence, dict) else {}

            summary_task = Task(
                description=(
                    f"Generate a concise metallurgical analysis for this alloy evaluation:\n\n"
                    f"**Composition**: {comp_str}, ...\n\n"
                    f"**Predicted Properties** (at {temperature}°C):\n"
                    f"- Yield Strength: {props.get('Yield Strength', 'N/A')} MPa\n"
                    f"- Tensile Strength: {props.get('Tensile Strength', 'N/A')} MPa\n"
                    f"- Elongation: {props.get('Elongation', 'N/A')}%\n"
                    f"- Elastic Modulus: {props.get('Elastic Modulus', 'N/A')} GPa\n"
                    f"- Gamma Prime: {props.get('Gamma Prime', 'N/A')} vol%\n"
                    f"- Density: {props.get('Density', 'N/A')} g/cm³\n\n"
                    f"**Physics Audit**:\n"
                    f"- Status: {corrected_output.status}\n"
                    f"- TCP Risk: {corrected_output.tcp_risk}\n"
                    f"- Audit Violations: {len(corrected_output.audit_penalties)}\n"
                    f"- Confidence: {confidence.get('level', 'MEDIUM')} ({confidence.get('score', 0.5):.2f})\n\n"
                    f"Write a 2-3 sentence technical summary explaining:\n"
                    f"1. What makes this alloy strong (or weak) based on its gamma prime content\n"
                    f"2. Any concerns from the physics audit (TCP risk, coherency)\n"
                    f"3. Suitable applications or recommendations\n\n"
                    f"IMPORTANT: Use ONLY the property values provided above. Do not invent different numbers."
                ),
                expected_output="2-3 sentence metallurgical analysis using exact values provided",
                agent=self.summarizer
            )

            summary_crew = Crew(
                agents=[self.summarizer],
                tasks=[summary_task],
                verbose=False
            )

            summary_result = summary_crew.kickoff()
            summary_text = str(summary_result.raw) if hasattr(summary_result, 'raw') else str(summary_result)
            corrected_output.explanation = summary_text

        except Exception as e:
            print(f"⚠️  Could not generate summary: {e}")
            # Keep original explanation but add note about calibration
            if corrected_output.explanation:
                corrected_output.explanation += "\n\n(Note: Final property values shown above reflect database-driven calibration.)"

        # Cleanup LLM output (normalize keys, filter invalid metrics)
        clean_props, clean_intervals, clean_metrics = cleanup_llm_output(
            corrected_output.properties,
            corrected_output.property_intervals,
            corrected_output.metallurgy_metrics or {},
            composition
        )
        corrected_output.properties = clean_props
        corrected_output.property_intervals = clean_intervals
        corrected_output.metallurgy_metrics = clean_metrics
        corrected_output.confidence = cleanup_confidence(corrected_output.confidence)

        # Normalize corrections_applied
        normalized_corrections = []
        for c in corrected_output.corrections_applied:
            if "Correction reason" in c.correction_reason:
                continue
            norm_name = PROPERTY_KEY_MAP.get(c.property_name, c.property_name)
            if norm_name in VALID_PROPERTIES:
                c.property_name = norm_name
            normalized_corrections.append(c)
        corrected_output.corrections_applied = normalized_corrections

        return corrected_output.model_dump()
