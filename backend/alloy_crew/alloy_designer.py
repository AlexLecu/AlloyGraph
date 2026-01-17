import json
from typing import Optional, Dict
from enum import Enum

from crewai import Crew, Task

from .agents import get_design_agents
from .tools.rag_tools import AlloySearchTool
from .tools.calibration_fix import apply_calibration_safe
from .tools.metallurgy_tools import validate_property_coherency, enforce_physics_constraints, cleanup_llm_output, cleanup_confidence, warnings_to_penalties
from .schemas import ValidationOutput, ArbitrationOutput, PhysicsAuditOutput, CorrectedPropertiesOutput, DesignOutput, OptimizationOutput, AuditPenalty

class FailureMode(Enum):
    """Structured classification of design failure reasons."""
    TCP_RISK = "TCP_RISK"
    PROPERTY_SHORTFALL = "PROPERTY_SHORTFALL"
    PHYSICS_VIOLATION = "PHYSICS_VIOLATION"
    COMPOSITION_INVALID = "COMPOSITION_INVALID"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    OTHER = "OTHER"


def round_composition(comp: Dict[str, float], decimals: int = 2) -> Dict[str, float]:
    """Round composition values to specified decimal places."""
    return {k: round(v, decimals) for k, v in comp.items()}


class IterativeDesignCrew:
    def __init__(self, target_props):
        self.target_props = target_props
        self.agents = get_design_agents()

        self.designer = self.agents["designer"]
        self.optimization_advisor = self.agents["optimization_advisor"]
        self.validator = self.agents["validator"]
        self.arbitrator = self.agents["arbitrator"]
        self.physicist = self.agents["physicist"]
        self.corrector = self.agents["corrector"]
        self.summarizer = self.agents["summarizer"]


        self.min_yield = float(target_props.get("Yield Strength", 0))
        self.min_tensile = float(target_props.get("Tensile Strength", 0))
        self.min_elongation = float(target_props.get("Elongation", 0))
        self.min_elastic_modulus = float(target_props.get("Elastic Modulus", 0))
        self.max_density = float(target_props.get("Density", 99.0))
        self.min_gamma_prime = float(target_props.get("Gamma Prime", 0))
        self.failure_history = []

        self._setup_tasks()
        self._setup_crews()

    def _quick_physics_precheck(self, composition: dict) -> tuple[bool, list]:
        """Fast physics validation before full pipeline."""
        from .models.feature_engineering import compute_alloy_features

        try:
            features = compute_alloy_features(composition)
            warnings = []

            md_gamma = features.get("Md_gamma", 0)
            if md_gamma > 0.97:
                warnings.append(f"Critical: Md_gamma={md_gamma:.3f} > 0.97 (TCP phase formation risk)")
            elif md_gamma > 0.95:
                warnings.append(f"Warning: Md_gamma={md_gamma:.3f} > 0.95 (approaching TCP danger zone)")

            delta = features.get("lattice_mismatch_pct", 0)
            if abs(delta) > 0.9:
                warnings.append(f"Critical: Lattice mismatch={delta:.2f}% > 0.9% (coherency risk)")
            elif abs(delta) > 0.7:
                warnings.append(f"Warning: Lattice mismatch={delta:.2f}% > 0.7% (reduced coherency)")

            cr = composition.get("Cr", 0)
            if cr < 5 or cr > 20:
                warnings.append(f"Warning: Cr={cr:.1f}% outside optimal range (5-20%)")

            al = composition.get("Al", 0)
            ti = composition.get("Ti", 0)
            ta = composition.get("Ta", 0)
            gp_formers = al + ti + ta
            if gp_formers < 3:
                warnings.append(f"Warning: Low γ' formers (Al+Ti+Ta={gp_formers:.1f}% < 3%) may limit strength")
            elif gp_formers > 12:
                warnings.append(f"Warning: High γ' formers (Al+Ti+Ta={gp_formers:.1f}% > 12%) may cause instability")

            return (len([w for w in warnings if w.startswith("Critical")]) == 0, warnings)

        except Exception:
            return (True, [])

    def _get_priority_focus(self, feedback: str, iteration: int) -> str:
        """Determine what the Designer should focus on this iteration."""
        if iteration == 0 or not feedback:
            return "Balanced design meeting all targets with proven composition patterns"

        # Analyze feedback for priorities
        feedback_lower = feedback.lower()

        if "tcp" in feedback_lower or "md" in feedback_lower or "phase" in feedback_lower:
            return "TCP risk reduction (lower Re/W/Mo, increase Cr, optimize Md_gamma < 0.95)"

        if "yield" in feedback_lower and "strength" in feedback_lower:
            return "Strength improvement (increase γ' formers: Al, Ti, Ta)"

        if "lattice" in feedback_lower or "mismatch" in feedback_lower:
            return "Coherency optimization (balance Al/Ti ratio, target |δ| < 0.5%)"

        if "confidence" in feedback_lower or "unreliable" in feedback_lower:
            return "Move closer to known alloy space (reduce exploratory elements)"

        return "Address all feedback points systematically"

    def _setup_tasks(self):
        """Define tasks once with placeholders {variables} for dynamic execution."""


        self.task_design = Task(
            description=(
                "🎯 DESIGN OBJECTIVE:\n"
                "Create a Ni-based superalloy composition meeting the targets below.\n\n"

                "📋 TARGETS (at {temperature}°C, {processing}):\n"
                "{target_props_str}\n\n"

                "📍 CURRENT STATUS:\n"
                "{base_comp_str}\n\n"

                "🔄 ITERATION FEEDBACK:\n"
                "{feedback}\n\n"

                "💡 DESIGN STRATEGY (FOCUS THIS ITERATION):\n"
                "{priority_focus}\n\n"

                "✅ SUCCESS CRITERIA:\n"
                "1. All target properties met (within ±10%)\n"
                "2. TCP risk = Low (Md_gamma < 0.95)\n"
                "3. Lattice mismatch < 0.8%\n"
                "4. Cr = 5-20%, γ' formers appropriate for {processing}\n\n"

                "{novelty_msg}\n\n"

                "OUTPUT: JSON with 'reasoning' (2-3 sentences explaining your approach), "
                "'composition' (dict summing to 100%), 'processing' ('{processing}')."
            ),
            expected_output="Structured design with clear reasoning and valid composition.",
            output_pydantic=DesignOutput,
            agent=self.designer,
        )


        self.task_optimization = Task(
            description=(
                "The Designer's composition has failed validation.\n\n"
                "Composition: {composition_json}\n"
                "Target Properties: {target_props_str}\n"
                "Current Properties: {current_props_json}\n"
                "Failure Reasons: {failure_reasons}\n"
                "Processing: {processing}\n\n"
                "Use AlloyOptimizationAdvisor to calculate physics-based suggestions.\n"
                "Return the TOP 3 most effective adjustments with quantified impacts."
            ),
            expected_output="Structured optimization suggestions with priorities and expected impacts.",
            output_pydantic=OptimizationOutput,
            agent=self.optimization_advisor,
            context=[self.task_design],
        )


        self.task_validation = Task(
            description=(
                "Validate composition at {temperature}°C using AlloyPredictorTool.\n"
                "Processing: {processing}\n"
                "Composition: {composition_json}\n\n"
                "Call the tool with composition, temperature_c, and processing parameters."
            ),
            expected_output="Structured ML predictions with confidence.",
            output_pydantic=ValidationOutput,
            agent=self.validator,
            async_execution=False,
        )

        self.task_arbitration = Task(
            description=(
                "Knowledge Graph Context:\n{kg_context}\n\n"
                "Processing: {processing}\n"
                "Take ML predictions from Validator. Run DataFusionTool to anchor against KG.\n"
                "Use mode='design' for fusion (trust ML more for innovation at {temperature}°C).\n"
                "PRESERVE property_intervals and confidence breakdown."
            ),
            expected_output="Fused properties with confidence and intervals.",
            output_pydantic=ArbitrationOutput,
            agent=self.arbitrator,
            context=[self.task_validation],
        )

        self.task_physics = Task(
            description=(
                "Evaluate physical validity.\n"
                "Composition: {composition_json}\n"
                "Processing: {processing}\n"
                "Temperature: {temperature}°C\n"
                "Input: Use the COMPLETE Arbitrator output JSON (including fusion_meta).\n\n"
                "1. ANALYZE THE ALLOY TYPE (LLM REASONING):\n"
                "   - If Cr > 21.0%: likely corrosion-resistant → Set alloy_type='high_corrosion'\n"
                "   - If Cr < 20.0% AND (Ti + Al) > 4.0%: likely high-strength blade → Set alloy_type='high_strength'\n"
                "   - Otherwise: Set alloy_type='standard'\n"
                "2. EXECUTE `MetallurgyVerifierTool` with:\n"
                "   - composition: {composition_json}\n"
                "   - anchored_properties_json: The ENTIRE Arbitrator output as a JSON string\n"
                "   - temperature_c: {temperature}\n"
                "   - alloy_type: Your inferred type from step 1\n"
                "   CRITICAL: Pass the complete JSON containing properties, property_intervals, fusion_meta, confidence.\n"
                "3. The tool returns verified_properties, property_intervals, confidence, and metallurgy_metrics.\n"
                "4. GENERATE EXPERT METALLURGICAL INSIGHT (3-5 sentences):\n"
                "   - Identify dominant strengthening mechanism (Gamma Prime vs Solid Solution)\n"
                "   - Evaluate trade-offs (strength vs ductility, castability, etc.)\n"
                "   - Propose applications based on property profile\n"
                "   - TONE: Professional, variable, avoid AI-sounding repetition\n"
                "5. Return the tool's output with your explanation added.\n"
                "   - Do NOT modify verified_properties, property_intervals, or confidence\n"
                "   - ONLY update the explanation field"
            ),
            expected_output="Final audit with explanation and intervals.",
            output_pydantic=PhysicsAuditOutput,
            agent=self.physicist,
            context=[self.task_arbitration],
        )

        self.task_corrections = Task(
            description=(
                "Apply physics-based corrections to improve prediction accuracy.\n"
                "Composition: {composition_json}\n"
                "Processing: {processing}\n"
                "Temperature: {temperature}°C\n"
                "Input: Use the COMPLETE Physicist output JSON.\n\n"
                "1. EXTRACT from Physicist output:\n"
                "   - properties (dict)\n"
                "   - composition (from input)\n"
                "   - confidence_level: confidence.level (string: HIGH, MEDIUM, LOW, VERY LOW)\n"
                "   - processing (string: wrought, cast, forged)\n"
                "   - kg_match_distance: confidence.similarity_distance (float, default 999)\n"
                "2. EXECUTE PhysicsCorrectionsProposalTool with these parameters.\n"
                "3. REVIEW proposals:\n"
                "   - Follow decision criteria in your backstory\n"
                "   - Apply HIGH severity corrections if confidence is LOW/VERY LOW\n"
                "   - Apply MEDIUM severity if no KG match (distance > 10)\n"
                "   - Skip LOW severity unless multiple issues\n"
                "4. CREATE PropertyCorrection objects for applied corrections:\n"
                "   - property_name, original_value, corrected_value, correction_reason, physics_constraint\n"
                "5. PRESERVE all fields from Physicist:\n"
                "   - status, penalty_score, tcp_risk, metallurgy_metrics, audit_penalties\n"
                "   - property_intervals, confidence, explanation\n"
                "6. ADD corrections_explanation:\n"
                "   - Why corrections were applied (or not)\n"
                "   - Expected accuracy after corrections: '±5-10% for novel alloys'\n"
                "   - Recommendation for experimental validation if needed\n"
                "   - Note: 'Database-driven calibration will be applied automatically in post-processing to account for systematic formula biases'\n"
                "7. RETURN structured CorrectedPropertiesOutput.\n\n"
                "NOTE: After you complete this task, calibration will be automatically applied in post-processing.\n"
                "This fixes systematic biases in physics formulas (e.g., YS = 400+18×γ' overpredicts by ~16%).\n"
                "You don't need to apply calibration yourself - just document which physics corrections you made."
            ),
            expected_output="Final corrected properties with physics constraints applied.",
            output_pydantic=CorrectedPropertiesOutput,
            agent=self.corrector,
            context=[self.task_physics],
        )

    def _setup_crews(self):
        """Instantiate Crews once."""
        self.crew_synthesis = Crew(
            agents=[self.designer],
            tasks=[self.task_design],
            verbose=True,
        )

        self.crew_analysis = Crew(
            agents=[self.validator, self.arbitrator, self.physicist, self.corrector],
            tasks=[self.task_validation, self.task_arbitration, self.task_physics, self.task_corrections],
            verbose=True,
        )

    def _run_novelty_check(self, composition: Optional[dict]) -> str:
        if not composition:
            return ""
        try:
            search_tool = AlloySearchTool()
            rag_result = search_tool._run(composition=composition, limit=1)
            if rag_result and "Error" not in str(rag_result):
                rag_data = json.loads(rag_result)
                if isinstance(rag_data, list) and len(rag_data) > 0:
                    match = rag_data[0]
                    name = match.get("name", "Unknown")
                    return f" [Context: Closest match is **{name}**. If different, this is a NOVEL design.]"
        except Exception:
            pass
        return ""

    def run(self, base_composition=None, input_feedback="", temperature=900, processing="cast", iteration_num=0):
        """Run one iteration of Design (Phase 1) → Validate (Phase 2)."""


        base_comp_str = (
            f"Starting Composition: {json.dumps(base_composition)}"
            if base_composition
            else "No starting comp - create from scratch"
        )

        # Build target string with proper semantics for each property
        target_parts = []
        if self.min_yield > 0:
            target_parts.append(f"- Yield Strength ≥ {self.min_yield} MPa")
        if self.min_tensile > 0:
            target_parts.append(f"- Tensile Strength ≥ {self.min_tensile} MPa")
        if self.min_elongation > 0:
            target_parts.append(f"- Elongation ≥ {self.min_elongation} %")
        if self.min_elastic_modulus > 0:
            target_parts.append(f"- Elastic Modulus ≥ {self.min_elastic_modulus} GPa")
        if self.max_density < 99.0:
            target_parts.append(f"- Density ≤ {self.max_density} g/cm³")

        # Gamma Prime is a target to match, not a minimum threshold
        if self.min_gamma_prime > 0:
            gp_tolerance = max(2.0, self.min_gamma_prime * 0.2)
            target_parts.append(
                f"- Gamma Prime ≈ {self.min_gamma_prime}% (target range: "
                f"{self.min_gamma_prime - gp_tolerance:.1f}-{self.min_gamma_prime + gp_tolerance:.1f}%). "
                f"⚠️ Do NOT maximize γ' - match the target!"
            )

        target_str = "\n".join(target_parts) if target_parts else "No specific targets"
        novelty_msg = self._run_novelty_check(base_composition)
        priority_focus = self._get_priority_focus(input_feedback, iteration_num)

        inputs_synthesis = {
            "base_comp_str": base_comp_str,
            "target_props_str": target_str,
            "feedback": input_feedback or "None (Initial Run)",
            "temperature": temperature,
            "processing": processing,
            "novelty_msg": novelty_msg,
            "priority_focus": priority_focus,
        }

        try:
            self.crew_synthesis.kickoff(inputs=inputs_synthesis)
        except Exception as e:
            return {"error": f"Synthesis Crew Failed: {e}"}

        try:
            d_obj = getattr(self.task_design.output, "pydantic", None)
            if not d_obj:
                return {"error": "Designer failed to return structured output."}
            designer_comp = d_obj.composition
            processed_route = d_obj.processing

            # Validate composition sum
            if not isinstance(designer_comp, dict) or not designer_comp:
                return {"error": "Designer returned invalid composition."}

            total = sum(designer_comp.values())
            if total < 90.0 or total > 110.0:
                return {"error": f"Composition sum ({total:.1f}%) is outside acceptable range (90-110%)."}

            designer_comp = round_composition(designer_comp, decimals=2)

            # Enforce user-specified processing
            if processed_route != processing:
                processed_route = processing
        except Exception as e:
            return {"error": f"Designer output extraction failed: {e}"}

        try:
            kg_raw = AlloySearchTool()._run(composition=designer_comp, limit=3)
            try:
                kg_json = json.loads(kg_raw)
                kg_context = json.dumps(kg_json)
            except Exception:
                kg_context = "[]"
        except Exception:
            kg_context = "[]"

        novelty_new_design = self._run_novelty_check(designer_comp)


        inputs_analysis = {
            "composition_json": json.dumps(designer_comp),
            "temperature": temperature,
            "processing": processed_route,
            "kg_context": kg_context,
        }

        try:
            self.crew_analysis.kickoff(inputs=inputs_analysis)
        except Exception as e:
            return {"error": f"Analysis Crew Failed: {e}"}

        corrected_output = getattr(self.task_corrections.output, "pydantic", None)
        if not corrected_output:
            return {"error": "Corrector did not return structured output."}

        physics_output = corrected_output

        validator_output = getattr(self.task_validation.output, "pydantic", None)
        if validator_output and hasattr(validator_output, 'ml_prediction'):
            ml_truth = validator_output.ml_prediction

            for prop_name in ["Yield Strength", "Tensile Strength", "Elongation", "Elastic Modulus", "Density"]:
                if prop_name in ml_truth and prop_name in physics_output.properties:
                    ml_value = ml_truth[prop_name]
                    phys_value = physics_output.properties[prop_name]

                    if ml_value > 0 and abs(phys_value - ml_value) > max(0.01 * ml_value, 1):
                        diff_pct = abs(phys_value - ml_value) / ml_value * 100
                        print(f"📊 Physics correction applied to {prop_name}: {ml_value:.1f} → {phys_value:.1f} ({diff_pct:+.1f}%)")

        physics_output.properties = apply_calibration_safe(
            physics_output.properties,
            designer_comp,
            physics_output
        )

        # === PHYSICS ENFORCEMENT (Hard Constraints) ===
        confidence = physics_output.confidence if isinstance(physics_output.confidence, dict) else {}
        kg_distance = confidence.get("similarity_distance", 999)
        confidence_level = confidence.get("level", "MEDIUM")

        physics_output.properties, physics_corrections = enforce_physics_constraints(
            properties=physics_output.properties,
            temperature_c=temperature,
            processing=processed_route,
            confidence_level=confidence_level,
            kg_distance=kg_distance
        )

        if physics_corrections:
            print(f"⚡ Physics enforcement applied {len(physics_corrections)} corrections:")
            for corr in physics_corrections:
                print(f"   - {corr}")

        # Re-run coherency checks with POST-calibration values
        non_coherency_penalties = [
            p for p in physics_output.audit_penalties
            if not any(x in p.name.lower() for x in ["coherency", "mismatch"])
        ]
        fresh_warnings = validate_property_coherency(physics_output.properties, designer_comp)
        fresh_penalties = [AuditPenalty(**p) for p in warnings_to_penalties(fresh_warnings)]
        physics_output.audit_penalties = non_coherency_penalties + fresh_penalties

        try:
            summarizer = self.summarizer
            comp_str = ", ".join([f"{elem}: {wt:.1f}%" for elem, wt in sorted(designer_comp.items(), key=lambda x: x[1], reverse=True)[:5]])

            summary_task = Task(
                description=(
                    f"Generate a 3-paragraph summary for this alloy design:\n\n"
                    f"**Target**: Yield Strength ≥ {self.min_yield} MPa, Density ≤ {self.max_density} g/cm³\n\n"
                    f"**Designed Composition**: {comp_str}, ...\n\n"
                    f"**Achieved Properties**:\n"
                    f"- Yield Strength: {physics_output.properties.get('Yield Strength', 'N/A')} MPa\n"
                    f"- Tensile Strength: {physics_output.properties.get('Tensile Strength', 'N/A')} MPa\n"
                    f"- Elongation: {physics_output.properties.get('Elongation', 'N/A')}%\n"
                    f"- Elastic Modulus: {physics_output.properties.get('Elastic Modulus', 'N/A')} GPa\n"
                    f"- Gamma Prime: {physics_output.properties.get('Gamma Prime', 'N/A')} vol%\n\n"
                    f"**Physics Audit**:\n"
                    f"- TCP Risk: {physics_output.tcp_risk}\n"
                    f"- Violations: {len(physics_output.audit_penalties)}\n\n"
                    f"Explain (1) What was designed, (2) Performance vs target, (3) Trade-offs and recommendations."
                ),
                expected_output="3-paragraph technical summary in clear language",
                agent=summarizer
            )
            
            summary_crew = Crew(
                agents=[summarizer],
                tasks=[summary_task],
                verbose=False
            )
            
            summary_result = summary_crew.kickoff()
            summary_text = str(summary_result.raw) if hasattr(summary_result, 'raw') else str(summary_result)
            
        except Exception as e:
            print(f"⚠️  Could not generate summary: {e}")
            summary_text = physics_output.explanation  # Fallback to physicist explanation

        # Cleanup LLM output (normalize keys, filter invalid metrics)
        clean_properties, clean_intervals, clean_metrics = cleanup_llm_output(
            physics_output.properties,
            physics_output.property_intervals,
            physics_output.metallurgy_metrics or {},
            designer_comp
        )

        return {
            "composition": designer_comp,
            "processing": processed_route,
            "properties": clean_properties,
            "property_intervals": clean_intervals,
            "metallurgy_metrics": clean_metrics,
            "confidence": cleanup_confidence(physics_output.confidence),
            "explanation": summary_text,
            "novelty": novelty_new_design,
            "penalty_score": physics_output.penalty_score,
            "tcp_risk": physics_output.tcp_risk,
            "audit_penalties": [p.dict() for p in physics_output.audit_penalties] if physics_output.audit_penalties else [],
            "status": physics_output.status,
        }

    def _is_design_successful(self, result):
        """Determine if a design meets all success criteria."""
        if result.get("error"):
            return False

        if result.get("tcp_risk", "High") == "High":
            return False

        penalties = result.get("audit_penalties", [])
        if any(p.get("name") == "High Md" for p in penalties):
            return False

        props = result.get("properties", {})

        if self.min_yield > 0 and float(props.get("Yield Strength", 0) or 0) < self.min_yield:
            return False
        if self.min_tensile > 0 and float(props.get("Tensile Strength", 0) or 0) < self.min_tensile:
            return False
        if self.min_elongation > 0 and float(props.get("Elongation", 0) or 0) < self.min_elongation:
            return False
        if self.min_elastic_modulus > 0 and float(props.get("Elastic Modulus", 0) or 0) < self.min_elastic_modulus:
            return False
        if self.max_density < 99.0 and float(props.get("Density", 1e9) or 1e9) > self.max_density:
            return False

        if self.min_gamma_prime > 0:
            actual_gp = float(props.get("Gamma Prime", 0) or 0)
            gp_tolerance = max(2.0, self.min_gamma_prime * 0.2)
            gp_min = self.min_gamma_prime - gp_tolerance
            gp_max = self.min_gamma_prime + gp_tolerance

            if actual_gp < gp_min:
                print(f"❌ DESIGN FAILED: Gamma Prime {actual_gp:.1f}% is TOO LOW (target range: {gp_min:.1f}-{gp_max:.1f}%)")
                return False
            if actual_gp > gp_max:
                print(f"❌ DESIGN FAILED: Gamma Prime {actual_gp:.1f}% is TOO HIGH (target range: {gp_min:.1f}-{gp_max:.1f}%)")
                print(f"   ⚠️  You designed a HIGH-γ' turbine blade alloy when user requested LOW-γ' structural alloy!")
                print(f"   ⚠️  MUST reduce Al+Ti+Ta to ~{self.min_gamma_prime / 3:.1f}% total (currently {sum([result.get('composition', {}).get(el, 0) for el in ['Al', 'Ti', 'Ta']]):.1f}%)")
                return False

        return True

    def _classify_failures(self, result) -> Dict[FailureMode, list]:
        """Classify failure reasons into structured categories."""
        failures_by_mode = {mode: [] for mode in FailureMode}

        tcp = result.get("tcp_risk", "Unknown")
        if tcp == "High":
            failures_by_mode[FailureMode.TCP_RISK].append("TCP Risk is HIGH (topologically close-packed phase formation)")
        elif tcp == "Medium":
            failures_by_mode[FailureMode.TCP_RISK].append("TCP Risk is MEDIUM (approaching danger zone)")

        penalties = result.get("audit_penalties", [])
        if penalties:
            for p in penalties:
                name = p.get("name", "Unknown")
                value = p.get("value", "")
                reason = p.get("reason", "")
                failures_by_mode[FailureMode.PHYSICS_VIOLATION].append(
                    f"{name}: {value} - {reason}"
                )
        props = result.get("properties", {})
        if self.min_yield > 0 and float(props.get("Yield Strength", 0) or 0) < self.min_yield:
            failures_by_mode[FailureMode.PROPERTY_SHORTFALL].append(
                f"Yield Strength too low ({props.get('Yield Strength', 0):.0f} < {self.min_yield} MPa)"
            )
        if self.min_tensile > 0 and float(props.get("Tensile Strength", 0) or 0) < self.min_tensile:
            failures_by_mode[FailureMode.PROPERTY_SHORTFALL].append(
                f"Tensile Strength too low ({props.get('Tensile Strength', 0):.0f} < {self.min_tensile} MPa)"
            )
        if self.min_elongation > 0 and float(props.get("Elongation", 0) or 0) < self.min_elongation:
            failures_by_mode[FailureMode.PROPERTY_SHORTFALL].append(
                f"Elongation too low ({props.get('Elongation', 0):.1f} < {self.min_elongation}%)"
            )
        if self.min_elastic_modulus > 0 and float(props.get("Elastic Modulus", 0) or 0) < self.min_elastic_modulus:
            failures_by_mode[FailureMode.PROPERTY_SHORTFALL].append(
                f"Elastic Modulus too low ({props.get('Elastic Modulus', 0):.0f} < {self.min_elastic_modulus} GPa)"
            )
        if self.max_density < 99.0 and float(props.get("Density", 1e9) or 1e9) > self.max_density:
            failures_by_mode[FailureMode.PROPERTY_SHORTFALL].append(
                f"Density too high ({props.get('Density', 0):.2f} > {self.max_density} g/cm³)"
            )

        if self.min_gamma_prime > 0:
            actual_gp = float(props.get("Gamma Prime", 0) or 0)
            gp_tolerance = max(2.0, self.min_gamma_prime * 0.2)
            gp_min = self.min_gamma_prime - gp_tolerance
            gp_max = self.min_gamma_prime + gp_tolerance

            if actual_gp < gp_min:
                failures_by_mode[FailureMode.PROPERTY_SHORTFALL].append(
                    f"Gamma Prime too low ({actual_gp:.1f}% < target {self.min_gamma_prime}% [min {gp_min:.1f}%]). Increase Al+Ti+Ta content."
                )
            elif actual_gp > gp_max:
                formers_total = sum([result.get('composition', {}).get(el, 0) for el in ['Al', 'Ti', 'Ta']])
                failures_by_mode[FailureMode.PROPERTY_SHORTFALL].append(
                    f"🚨 CRITICAL: Gamma Prime {actual_gp:.1f}% >> target {self.min_gamma_prime}% (max {gp_max:.1f}%). "
                    f"You designed a HIGH-γ' TURBINE BLADE ALLOY (wrong class!). "
                    f"Current formers: {formers_total:.1f}% (Al+Ti+Ta). Target: ~{self.min_gamma_prime / 3:.1f}%. "
                    f"REQUIRED: Drastically cut Al+Ti+Ta by ~{formers_total - self.min_gamma_prime / 3:.1f}%, "
                    f"then compensate strength with SSS elements (Mo, W, Nb, Co). "
                    f"Reference alloys: IN718 (18% γ'), Haynes 282 (25% γ'), NIMOCAST 263 (2.7% γ')."
                )

        if result.get("error"):
            error_msg = result["error"]
            if "Chemistry" in error_msg or "composition" in error_msg.lower():
                failures_by_mode[FailureMode.COMPOSITION_INVALID].append(error_msg)
            else:
                failures_by_mode[FailureMode.OTHER].append(error_msg)

        return {mode: msgs for mode, msgs in failures_by_mode.items() if msgs}

    def _classify_design_quality(self, result):
        """Classify design quality based on how well it hits targets."""
        if result.get("error") or not self._is_design_successful(result):
            return "FAILED", ""

        props = result.get("properties", {})
        ys = float(props.get("Yield Strength", 0) or 0)

        if self.min_yield > 0 and ys > 0:
            target = self.min_yield
            optimal_max = target * 1.10
            excessive_threshold = target * 1.30
            overshoot_pct = ((ys / target) - 1) * 100

            if target <= ys <= optimal_max:
                return "OPTIMAL", f"Target hit within optimal range ({ys:.0f} MPa, +{overshoot_pct:.1f}%)"
            elif optimal_max < ys <= excessive_threshold:
                return "ACCEPTABLE", f"Over-engineered: {ys:.0f} MPa (+{overshoot_pct:.0f}% above {target} MPa target)"
            elif ys > excessive_threshold:
                return "EXCESSIVE", f"Significantly over-engineered: {ys:.0f} MPa (+{overshoot_pct:.0f}% above target)"

        return "SUCCESS", ""

    def loop(self, max_iterations=3, start_composition=None, temperature=900, processing="cast"):
        current_comp = start_composition
        feedback = ""
        result = {"error": "No iterations executed."}
        use_direct_application = False

        target_parts = []
        if self.min_yield > 0:
            target_parts.append(f"- Yield Strength ≥ {self.min_yield} MPa")
        if self.min_tensile > 0:
            target_parts.append(f"- Tensile Strength ≥ {self.min_tensile} MPa")
        if self.min_elongation > 0:
            target_parts.append(f"- Elongation ≥ {self.min_elongation} %")
        if self.min_elastic_modulus > 0:
            target_parts.append(f"- Elastic Modulus ≥ {self.min_elastic_modulus} GPa")
        if self.max_density < 99.0:
            target_parts.append(f"- Density ≤ {self.max_density} g/cm³")

        # Gamma Prime is a target to match, not a minimum threshold
        if self.min_gamma_prime > 0:
            gp_tolerance = max(2.0, self.min_gamma_prime * 0.2)
            target_parts.append(
                f"- Gamma Prime ≈ {self.min_gamma_prime}% (target range: "
                f"{self.min_gamma_prime - gp_tolerance:.1f}-{self.min_gamma_prime + gp_tolerance:.1f}%). "
                f"⚠️ Do NOT maximize γ' - match the target!"
            )

        target_str = "\n".join(target_parts) if target_parts else "No specific targets"

        for i in range(max_iterations):
            iteration_num = i + 1
            print(f"\n⚡ ITERATION {iteration_num}/{max_iterations}")

            if use_direct_application and current_comp:
                print("🔬 DIRECT APPLICATION MODE: Using physics-optimized composition directly (bypassing LLM)")
                if not isinstance(current_comp, dict) or not current_comp:
                    print("⚠️ Direct mode composition invalid, falling back to LLM")
                    use_direct_application = False
                    current_comp = None
                    continue

                total = sum(current_comp.values())
                if total < 90.0 or total > 110.0:
                    print(f"⚠️ Direct mode composition sum ({total:.1f}%) out of range, falling back to LLM")
                    use_direct_application = False
                    current_comp = None
                    continue

                result = {
                    "composition": current_comp,
                    "processing": processing,
                    "reasoning": "Direct application of physics-based optimization"
                }

                inputs_validation = {
                    "composition_json": json.dumps(current_comp),
                    "temperature": temperature,
                    "processing": processing,
                    "kg_context": "[]",
                }
                try:
                    self.crew_analysis.kickoff(inputs=inputs_validation)

                    val_output = getattr(self.task_validation.output, "pydantic", None)

                    # Get corrected output (final task in pipeline)
                    corr_output = getattr(self.task_corrections.output, "pydantic", None)
                    if corr_output:
                        phys_output = corr_output  # Use corrected output
                    else:
                        phys_output = getattr(self.task_physics.output, "pydantic", None)  # Fallback

                    if phys_output:
                        result["properties"] = phys_output.properties
                        result["tcp_risk"] = phys_output.tcp_risk
                        result["audit_penalties"] = [p.model_dump() for p in phys_output.audit_penalties]

                        if val_output and hasattr(val_output, 'ml_prediction'):
                            ml_truth = val_output.ml_prediction
                            for prop_name in ["Yield Strength", "Tensile Strength", "Elongation", "Elastic Modulus", "Density"]:
                                if prop_name in ml_truth and prop_name in result["properties"]:
                                    ml_value = ml_truth[prop_name]
                                    corr_value = result["properties"][prop_name]
                                    if ml_value > 0 and abs(corr_value - ml_value) > max(0.01 * ml_value, 1):
                                        diff_pct = abs(corr_value - ml_value) / ml_value * 100
                                        print(f"📊 Correction applied to {prop_name}: {ml_value:.1f} → {corr_value:.1f} ({diff_pct:+.1f}%)")

                        result["properties"] = apply_calibration_safe(
                            result["properties"],
                            current_comp,
                            phys_output
                        )

                        # Physics enforcement for direct mode
                        conf = phys_output.confidence if isinstance(phys_output.confidence, dict) else {}
                        result["properties"], _ = enforce_physics_constraints(
                            properties=result["properties"],
                            temperature_c=temperature,
                            processing=processing,
                            confidence_level=conf.get("level", "MEDIUM"),
                            kg_distance=conf.get("similarity_distance", 999)
                        )
                    else:
                        raw_phys = getattr(self.task_physics.output, "raw", "")
                        if raw_phys:
                             try:
                                phys_data = json.loads(raw_phys)
                                result["tcp_risk"] = phys_data.get("tcp_risk", "Unknown")
                                result["audit_penalties"] = phys_data.get("audit_penalties", [])
                             except:
                                pass
                except Exception as e:
                    print(f"⚠️ Validation error in direct mode: {e}")
                    use_direct_application = False  # Fall back to LLM
            
            if not use_direct_application or not current_comp:

                result = self.run(
                    base_composition=current_comp,
                    input_feedback=feedback,
                    temperature=temperature,
                    processing=processing,
                    iteration_num=iteration_num,
                )

            if "error" in result:
                print(f"❌ Aborted: {result['error']}")
                current_comp = result.get("composition", current_comp)
                feedback = f"Design Failed: {result['error']}. Fix constraints."

                if "Chemistry" in result["error"]:
                    continue
                break

            if result.get("composition") and iteration_num < max_iterations:
                is_valid, precheck_warnings = self._quick_physics_precheck(result["composition"])

                if not is_valid:
                    print("\n⚡ FAST PHYSICS PRE-CHECK: Critical violations detected")
                    for w in precheck_warnings:
                        if w.startswith("Critical"):
                            print(f"   🔴 {w}")
                        else:
                            print(f"   🟡 {w}")

                    print("\n🔧 EARLY OPTIMIZATION: Getting physics-based corrections...")
                    try:
                        optimization_inputs = {
                            "composition_json": json.dumps(result["composition"]),
                            "target_props_str": target_str,
                            "current_props_json": "{}",
                            "failure_reasons": json.dumps([w for w in precheck_warnings if w.startswith("Critical")]),
                            "processing": processing,
                        }

                        crew_opt = Crew(
                            agents=[self.optimization_advisor],
                            tasks=[self.task_optimization],
                            verbose=False,
                        )
                        crew_opt.kickoff(inputs=optimization_inputs)

                        opt_output = getattr(self.task_optimization.output, "pydantic", None)
                        if opt_output and opt_output.recommended_actions:
                            print("📊 EARLY OPTIMIZATION SUGGESTIONS:")
                            for action in opt_output.recommended_actions[:3]:
                                print(f"   • {action}")

                            feedback = (
                                f"⚠️ PRE-VALIDATION FAILURES DETECTED:\n"
                                f"{chr(10).join(f'  • {w}' for w in precheck_warnings if w.startswith('Critical'))}\n\n"
                                f"PHYSICS-BASED CORRECTIONS (Apply immediately):\n"
                                f"{chr(10).join(f'  • {action}' for action in opt_output.recommended_actions[:3])}\n\n"
                                f"Apply these corrections and propose a revised composition that fixes the critical issues above."
                            )

                            print(f"\n🔄 Skipping full validation, applying corrections in next iteration...")
                            # Continue to next iteration with corrective feedback (skip validation)
                            continue
                    except Exception as e:
                        print(f"⚠️ Early optimization failed: {e}, proceeding with full validation")

            print(f"   Proposed: {result['composition']}")
            props = result.get("properties", {})
            tcp = result.get("tcp_risk", "Unknown")
            penalties = len(result.get("audit_penalties", []))
            print(f"   Properties: YS={props.get('Yield Strength', 0)}, TCP={tcp}, Penalties={penalties}")


            if self._is_design_successful(result):
                print("\n✅ SUCCESS! Converged.")
                break

            # 📊 STRUCTURED FAILURE ANALYSIS
            failures_by_mode = self._classify_failures(result)

            # Print structured failure report
            print("\n📊 FAILURE ANALYSIS:")
            for mode, messages in failures_by_mode.items():
                mode_label = mode.value.replace("_", " ")
                print(f"   [{mode_label}]")
                for msg in messages:
                    print(f"      • {msg}")

            # Build flat failure list for backward compatibility
            failures = []
            for mode_messages in failures_by_mode.values():
                failures.extend(mode_messages)

            # --- PRIORITY ENFORCEMENT LOGIC ---
            # Once YS target is met, focus EXCLUSIVELY on TCP risk
            ys_target_met = props.get("Yield Strength", 0) >= self.min_yield if self.min_yield > 0 else True
            tcp_critical = tcp == "High" or tcp == "Medium"
            
            if ys_target_met and tcp_critical:
                # PRIORITY MODE: Only focus on TCP, ignore other property improvements
                print("\n⚡ PRIORITY MODE: YS target met ({:.0f} >= {:.0f} MPa). Focusing ONLY on TCP risk reduction.".format(
                    props.get("Yield Strength", 0), self.min_yield))
                

                tcp_failures = [f for f in failures if "TCP" in f or "Md" in f or "Physics" in f]
                failures_for_advisor = tcp_failures if tcp_failures else ["TCP Risk needs reduction"]
                
                priority_note = (
                    "\n\n⚠️ CRITICAL PRIORITY: YS target already achieved. "
                    "Do NOT attempt to improve yield strength further. "
                    "Focus EXCLUSIVELY on reducing TCP risk by lowering Md. "
                    "Accept slight YS reduction if it eliminates TCP risk."
                )
            else:
                failures_for_advisor = failures
                priority_note = ""
            


            try:
                target_str = (
                    f"- Yield Strength > {self.min_yield} MPa\n"
                    f"- Tensile Strength > {self.min_tensile} MPa\n"
                    f"- Elongation > {self.min_elongation} %\n"
                    f"- Elastic Modulus > {self.min_elastic_modulus} GPa\n"
                    f"- Density < {self.max_density} g/cm3\n"
                    f"- Gamma Prime > {self.min_gamma_prime} %"
                )
                
                optimization_inputs = {
                    "composition_json": json.dumps(result["composition"]),
                    "target_props_str": target_str,
                    "current_props_json": json.dumps(props),
                    "failure_reasons": json.dumps(failures_for_advisor),  # Use filtered failures
                    "processing": processing,
                }
                

                crew_opt = Crew(
                    agents=[self.optimization_advisor],
                    tasks=[self.task_optimization],
                    verbose=True,
                )
                crew_opt.kickoff(inputs=optimization_inputs)
                

                opt_output = getattr(self.task_optimization.output, "pydantic", None)
                if opt_output and opt_output.recommended_actions:
                    print("\n📊 OPTIMIZATION SUGGESTIONS:")
                    for action in opt_output.recommended_actions[:3]:
                        print(f"   • {action}")

                    failure_mode_summary = []
                    for mode, msgs in failures_by_mode.items():
                        if mode == FailureMode.TCP_RISK:
                            failure_mode_summary.append("⚠️ CRITICAL: TCP phase formation risk")
                        elif mode == FailureMode.PROPERTY_SHORTFALL:
                            failure_mode_summary.append(f"⚠️ PROPERTY TARGET MISS: {len(msgs)} target(s) not met")
                        elif mode == FailureMode.PHYSICS_VIOLATION:
                            failure_mode_summary.append(f"⚠️ PHYSICS VIOLATION: {len(msgs)} constraint(s) violated")

                    failure_list = ', '.join(failures_for_advisor)
                    mode_context = "\n".join(failure_mode_summary)
                    suggestions_text = "\n".join(f"  • {action}" for action in opt_output.recommended_actions[:5])
                    feedback = (
                        f"Design REJECTED:\n{mode_context}\n\n"
                        f"SPECIFIC FAILURES:\n{failure_list}\n\n"
                        f"OPTIMIZATION SUGGESTIONS FROM PHYSICS ANALYSIS:\n{suggestions_text}\n\n"
                        f"Propose a NEW composition addressing these issues.\n"
                        f"Use your metallurgical expertise to incorporate these suggestions intelligently.{priority_note}"
                    )
                else:
                    # Fallback if optimization advisor fails
                    mode_context = []
                    for mode in failures_by_mode.keys():
                        mode_context.append(f"- {mode.value.replace('_', ' ')}")
                    mode_str = "\n".join(mode_context) if mode_context else "unspecified issues"
                    feedback = (
                        f"Design FAILED:\nFailure Categories:\n{mode_str}\n\n"
                        f"Details: {', '.join(failures)}\n\n"
                        f"Propose a new composition to fix these issues."
                    )
            except Exception as e:
                print(f"⚠️  Optimization advisor error: {e}")
                failure_str = ", ".join(failures) if failures else "unspecified issues"
                feedback = f"Design FAILED: {failure_str}. Propose a new composition to fix these issues."

        if not self._is_design_successful(result):
            tcp_risk = result.get("tcp_risk", "Unknown")
            penalties = result.get("audit_penalties", [])
            props = result.get("properties", {})

            issues = []
            recommendations = []

            if tcp_risk == "High":
                md_val = result.get("metallurgy_metrics", {}).get("md_gamma_matrix", "?")
                issues.append({
                    "type": "TCP Risk",
                    "severity": "High",
                    "description": f"High risk of TCP phase formation (Md={md_val}, safe limit <0.99). This can cause brittleness.",
                    "recommendation": "Reduce refractory elements (Re, W, Mo) or increase Cr/Co to lower Md value."
                })
            elif tcp_risk == "Medium":
                issues.append({
                    "type": "TCP Risk",
                    "severity": "Medium",
                    "description": "Medium TCP risk - approaching danger zone for phase formation.",
                    "recommendation": "Consider reducing refractory element content."
                })

            if penalties:
                for penalty in penalties:
                    penalty_name = penalty.get("name", "Unknown")
                    penalty_desc = penalty.get("description", "No description")
                    issues.append({
                        "type": "Audit Violation",
                        "severity": "Medium",
                        "description": f"{penalty_name}: {penalty_desc}",
                        "recommendation": "Review composition constraints."
                    })

            if self.min_yield > 0:
                actual_ys = float(props.get("Yield Strength", 0) or 0)
                if actual_ys < self.min_yield:
                    issues.append({
                        "type": "Target Miss",
                        "severity": "Low",
                        "description": f"Yield Strength {actual_ys:.0f} MPa is below target {self.min_yield} MPa.",
                        "recommendation": "Increase γ' formers (Al, Ti) or add solid solution strengtheners (Mo, W)."
                    })

            if self.min_tensile > 0:
                actual_uts = float(props.get("Tensile Strength", 0) or 0)
                if actual_uts < self.min_tensile:
                    issues.append({
                        "type": "Target Miss",
                        "severity": "Low",
                        "description": f"Tensile Strength {actual_uts:.0f} MPa is below target {self.min_tensile} MPa.",
                        "recommendation": "Similar to yield strength - increase strengthening phases."
                    })

            if self.min_gamma_prime > 0:
                actual_gp = float(props.get("Gamma Prime", 0) or 0)
                gp_tolerance = max(2.0, self.min_gamma_prime * 0.2)
                gp_min = self.min_gamma_prime - gp_tolerance
                gp_max = self.min_gamma_prime + gp_tolerance

                if actual_gp < gp_min:
                    issues.append({
                        "type": "Gamma Prime",
                        "severity": "High",
                        "description": f"Gamma Prime {actual_gp:.1f}% is below target range {gp_min:.1f}-{gp_max:.1f}%. Too little γ' for this alloy class.",
                        "recommendation": f"Increase Al, Ti, or Ta content to reach target ~{self.min_gamma_prime}%."
                    })
                elif actual_gp > gp_max:
                    issues.append({
                        "type": "Gamma Prime",
                        "severity": "High",
                        "description": f"Gamma Prime {actual_gp:.1f}% is above target range {gp_min:.1f}-{gp_max:.1f}%. WRONG ALLOY CLASS - this is a high-γ' turbine blade alloy, not the requested low-γ' structural alloy.",
                        "recommendation": f"Reduce Al, Ti, and Ta content significantly to reach target ~{self.min_gamma_prime}%. Current composition has {sum([result.get('composition', {}).get(el, 0) for el in ['Al', 'Ti', 'Ta']]):.1f}% formers, need <4% for low-γ' alloys."
                    })

            if len(issues) > 0:
                recommendations.append(f"Try increasing iterations to {max_iterations + 5}")
                recommendations.append("Consider relaxing conflicting targets")
                if any(issue["type"] == "Gamma Prime" for issue in issues):
                    recommendations.append("Review gamma prime target - different alloy classes have vastly different γ' fractions")

            result["issues"] = issues
            result["recommendations"] = recommendations
            result["design_status"] = "incomplete"
            result["iterations_used"] = max_iterations

            has_high_severity = any(issue["severity"] == "High" for issue in issues)
            if has_high_severity and result.get("status") == "PASS":
                print(f"\n🔴 Overriding status from PASS to REJECT due to HIGH severity design issues")
                result["status"] = "REJECT"

            print(f"\n⚠️ Design completed with {len(issues)} issues after {max_iterations} iterations")
            for issue in issues:
                print(f"   • [{issue['severity']}] {issue['type']}: {issue['description']}")

        else:
            result["design_status"] = "success"
            result["issues"] = []
            result["recommendations"] = []

        return result


if __name__ == "__main__":
    loop = IterativeDesignCrew({"Yield Strength": 1100})
    loop.loop(max_iterations=2)
