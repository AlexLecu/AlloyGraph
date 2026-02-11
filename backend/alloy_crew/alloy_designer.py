import json
import logging
from typing import Optional, Dict
from enum import Enum

logger = logging.getLogger(__name__)

from crewai import Crew, Task

from .agents import get_design_agents
from .tools.rag_tools import AlloySearchTool
from .schemas import DesignOutput, OptimizationOutput
from .alloy_evaluator import AlloyEvaluationCrew
from .config.alloy_parameters import TCP, classify_tcp_risk

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
        self.analyst = self.agents["analyst"]
        self.reviewer = self.agents["reviewer"]
        self.llm = self.agents.get("llm")


        self.min_yield = float(target_props.get("Yield Strength", 0))
        self.min_tensile = float(target_props.get("Tensile Strength", 0))
        self.min_elongation = float(target_props.get("Elongation", 0))
        self.min_elastic_modulus = float(target_props.get("Elastic Modulus", 0))
        self.max_density = float(target_props.get("Density", 99.0))
        self.min_gamma_prime = float(target_props.get("Gamma Prime", 0))
        self.failure_history = []

        # Reuse the evaluation pipeline (shared Analyst + Reviewer agents)
        self.evaluator = AlloyEvaluationCrew(agents={
            'analyst': self.analyst,
            'reviewer': self.reviewer,
            'llm': self.llm,
        })

        self._setup_tasks()
        self._setup_crews()

    def _quick_physics_precheck(self, composition: dict) -> tuple[bool, list]:
        """Fast physics validation before full pipeline."""
        from .models.feature_engineering import compute_alloy_features

        try:
            features = compute_alloy_features(composition)
            warnings = []

            md_gamma = features.get("Md_gamma", 0)
            md_avg = features.get("Md_avg", 0)
            tcp_level = classify_tcp_risk(md_gamma, md_avg)
            if tcp_level in ("Critical", "Elevated"):
                warnings.append(f"Critical: Md_avg={md_avg:.3f} > {TCP['MD_ELEVATED']} (TCP phase formation risk: {tcp_level})")
            elif tcp_level == "Moderate":
                warnings.append(f"Warning: Md_avg={md_avg:.3f} > {TCP['MD_MODERATE']} (approaching TCP danger zone)")

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
            return f"TCP risk reduction (lower Re/W/Mo, increase Cr, optimize Md_gamma < {TCP['MD_DESIGN_TARGET']})"

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
                "2. TCP risk = Low (Md_gamma < {md_target})\n"
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
                "Use AlloyOptimizationAdvisor to calculate physics-based suggestions.\n\n"
                "Return TOP 3 suggestions ranked by:\n"
                "1. Impact magnitude (ΔMd, ΔYS, degree of constraint violation)\n"
                "2. Ease of implementation (single-element adjustments preferred over multi-element)\n"
                "3. Minimal trade-offs (e.g., fixing TCP without sacrificing strength)\n\n"
                "Each suggestion must include:\n"
                "- Specific element adjustment (e.g., 'Reduce Re from 6.0% to 4.5%')\n"
                "- Quantified impact (e.g., 'lowers Md by 0.04')\n"
                "- Trade-off warning if applicable (e.g., 'may reduce YS by 50 MPa')"
            ),
            expected_output="Structured optimization suggestions with priorities and expected impacts.",
            output_pydantic=OptimizationOutput,
            agent=self.optimization_advisor,
            context=[self.task_design],
        )

    def _setup_crews(self):
        """Instantiate Crews once."""
        self.crew_synthesis = Crew(
            agents=[self.designer],
            tasks=[self.task_design],
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
            "md_target": TCP["MD_DESIGN_TARGET"],
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
            if total < 95.0 or total > 105.0:
                return {"error": f"Composition sum ({total:.1f}%) is outside acceptable range (95-105%)."}

            designer_comp = round_composition(designer_comp, decimals=2)

            # Enforce user-specified processing
            if processed_route != processing:
                processed_route = processing
        except Exception as e:
            return {"error": f"Designer output extraction failed: {e}"}

        # Phase 2: Evaluate designed composition via shared pipeline
        novelty_new_design = self._run_novelty_check(designer_comp)

        result = self.evaluator.evaluate_properties(
            composition=designer_comp,
            processing=processed_route,
            temperature=temperature,
            apply_calibration=True,
            summary_context={
                "min_yield": self.min_yield,
                "max_density": self.max_density,
                "min_tensile": self.min_tensile,
                "min_elongation": self.min_elongation,
                "min_elastic_modulus": self.min_elastic_modulus,
                "min_gamma_prime": self.min_gamma_prime,
            },
            extra_output_fields={
                "composition": designer_comp,
                "novelty": novelty_new_design,
            },
        )

        return result

    def _is_design_successful(self, result):
        """Determine if a design meets all success criteria."""
        if result.get("error"):
            return False

        if result.get("tcp_risk", "Critical") in ("Critical", "Elevated"):
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
                logger.error(f"DESIGN FAILED: Gamma Prime {actual_gp:.1f}% is TOO LOW (target range: {gp_min:.1f}-{gp_max:.1f}%)")
                return False
            if actual_gp > gp_max:
                logger.error(f"DESIGN FAILED: Gamma Prime {actual_gp:.1f}% is TOO HIGH (target range: {gp_min:.1f}-{gp_max:.1f}%)")
                logger.error(f"Designed a HIGH-gamma-prime turbine blade alloy when user requested LOW-gamma-prime structural alloy! MUST reduce Al+Ti+Ta to ~{self.min_gamma_prime / 3:.1f}% total (currently {sum([result.get('composition', {}).get(el, 0) for el in ['Al', 'Ti', 'Ta']]):.1f}%)")
                return False

        return True

    def _classify_failures(self, result) -> Dict[FailureMode, list]:
        """Classify failure reasons into structured categories."""
        failures_by_mode = {mode: [] for mode in FailureMode}

        tcp = result.get("tcp_risk", "Unknown")
        if tcp == "Critical":
            failures_by_mode[FailureMode.TCP_RISK].append("TCP Risk is CRITICAL (topologically close-packed phase formation)")
        elif tcp == "Elevated":
            failures_by_mode[FailureMode.TCP_RISK].append("TCP Risk is ELEVATED (approaching danger zone)")

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
            logger.info(f"ITERATION {iteration_num}/{max_iterations}")

            if use_direct_application and current_comp:
                logger.info("DIRECT APPLICATION MODE: Using physics-optimized composition directly (bypassing LLM)")
                if not isinstance(current_comp, dict) or not current_comp:
                    logger.warning("Direct mode composition invalid, falling back to LLM")
                    use_direct_application = False
                    current_comp = None
                    continue

                total = sum(current_comp.values())
                if total < 95.0 or total > 105.0:
                    logger.warning(f"Direct mode composition sum ({total:.1f}%) out of range (95-105%), falling back to LLM")
                    use_direct_application = False
                    current_comp = None
                    continue

                try:
                    result = self.evaluator.evaluate_properties(
                        composition=current_comp,
                        processing=processing,
                        temperature=temperature,
                        apply_calibration=True,
                    )
                    result["composition"] = current_comp
                    result["processing"] = processing
                    result["reasoning"] = "Direct application of physics-based optimization"
                except Exception as e:
                    logger.warning(f"Validation error in direct mode: {e}")
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
                logger.error(f"Aborted: {result['error']}")
                current_comp = result.get("composition", current_comp)
                feedback = f"Design Failed: {result['error']}. Fix constraints."

                if "Chemistry" in result["error"]:
                    continue
                break

            if result.get("composition") and iteration_num < max_iterations:
                is_valid, precheck_warnings = self._quick_physics_precheck(result["composition"])

                if not is_valid:
                    logger.info("FAST PHYSICS PRE-CHECK: Critical violations detected")
                    for w in precheck_warnings:
                        if w.startswith("Critical"):
                            logger.warning(w)
                        else:
                            logger.info(w)

                    logger.info("EARLY OPTIMIZATION: Getting physics-based corrections...")
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
                            logger.info("EARLY OPTIMIZATION SUGGESTIONS:")
                            for action in opt_output.recommended_actions[:3]:
                                logger.info(f"  {action}")

                            feedback = (
                                f"⚠️ PRE-VALIDATION FAILURES DETECTED:\n"
                                f"{chr(10).join(f'  • {w}' for w in precheck_warnings if w.startswith('Critical'))}\n\n"
                                f"PHYSICS-BASED CORRECTIONS (Apply immediately):\n"
                                f"{chr(10).join(f'  • {action}' for action in opt_output.recommended_actions[:3])}\n\n"
                                f"Apply these corrections and propose a revised composition that fixes the critical issues above."
                            )

                            logger.info("Skipping full validation, applying corrections in next iteration...")
                            # Continue to next iteration with corrective feedback (skip validation)
                            continue
                    except Exception as e:
                        logger.warning(f"Early optimization failed: {e}, proceeding with full validation")

            # Track the latest composition so the next iteration builds on it
            current_comp = result.get("composition", current_comp)

            logger.debug(f"Proposed: {result['composition']}")
            props = result.get("properties", {})
            tcp = result.get("tcp_risk", "Unknown")
            penalties = len(result.get("audit_penalties", []))
            logger.debug(f"Properties: YS={props.get('Yield Strength', 0)}, TCP={tcp}, Penalties={penalties}")


            if self._is_design_successful(result):
                logger.info("SUCCESS! Design converged.")
                break

            # 📊 STRUCTURED FAILURE ANALYSIS
            failures_by_mode = self._classify_failures(result)

            # Print structured failure report
            logger.debug("FAILURE ANALYSIS:")
            for mode, messages in failures_by_mode.items():
                mode_label = mode.value.replace("_", " ")
                logger.debug(f"  [{mode_label}]")
                for msg in messages:
                    logger.debug(f"    {msg}")

            # Build flat failure list for backward compatibility
            failures = []
            for mode_messages in failures_by_mode.values():
                failures.extend(mode_messages)

            # --- PRIORITY ENFORCEMENT LOGIC ---
            # Once YS target is met, focus EXCLUSIVELY on TCP risk
            ys_target_met = props.get("Yield Strength", 0) >= self.min_yield if self.min_yield > 0 else True
            tcp_critical = tcp in ("Critical", "Elevated")
            
            if ys_target_met and tcp_critical:
                # PRIORITY MODE: Only focus on TCP, ignore other property improvements
                logger.info("PRIORITY MODE: YS target met ({:.0f} >= {:.0f} MPa). Focusing ONLY on TCP risk reduction.".format(props.get("Yield Strength", 0), self.min_yield))
                

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
                opt_target_str = (
                    f"- Yield Strength > {self.min_yield} MPa\n"
                    f"- Tensile Strength > {self.min_tensile} MPa\n"
                    f"- Elongation > {self.min_elongation} %\n"
                    f"- Elastic Modulus > {self.min_elastic_modulus} GPa\n"
                    f"- Density < {self.max_density} g/cm3\n"
                    f"- Gamma Prime > {self.min_gamma_prime} %"
                )

                optimization_inputs = {
                    "composition_json": json.dumps(result["composition"]),
                    "target_props_str": opt_target_str,
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
                    logger.info("OPTIMIZATION SUGGESTIONS:")
                    for action in opt_output.recommended_actions[:3]:
                        logger.info(f"  {action}")

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
                logger.warning(f"Optimization advisor error: {e}")
                failure_str = ", ".join(failures) if failures else "unspecified issues"
                feedback = f"Design FAILED: {failure_str}. Propose a new composition to fix these issues."

        if not self._is_design_successful(result):
            tcp_risk = result.get("tcp_risk", "Unknown")
            penalties = result.get("audit_penalties", [])
            props = result.get("properties", {})

            issues = []
            recommendations = []

            if tcp_risk == "Critical":
                md_val = result.get("metallurgy_metrics", {}).get("md_gamma_matrix", "?")
                issues.append({
                    "type": "TCP Risk",
                    "severity": "High",
                    "description": f"Critical risk of TCP phase formation (Md={md_val}, safe limit <{TCP['MD_DESIGN_SAFE']}). This can cause brittleness.",
                    "recommendation": "Reduce refractory elements (Re, W, Mo) or increase Cr/Co to lower Md value."
                })
            elif tcp_risk == "Elevated":
                issues.append({
                    "type": "TCP Risk",
                    "severity": "Medium",
                    "description": "Elevated TCP risk - approaching danger zone for phase formation.",
                    "recommendation": "Consider reducing refractory element content."
                })

            if penalties:
                for penalty in penalties:
                    penalty_name = penalty.get("name", "Unknown")
                    penalty_desc = penalty.get("reason", "No description")
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
                logger.warning("Overriding status from PASS to REJECT due to HIGH severity design issues")
                result["status"] = "REJECT"

            logger.warning(f"Design completed with {len(issues)} issues after {max_iterations} iterations")
            for issue in issues:
                logger.warning(f"  [{issue['severity']}] {issue['type']}: {issue['description']}")

        else:
            result["design_status"] = "success"
            result["issues"] = []
            result["recommendations"] = []

        return result


if __name__ == "__main__":
    loop = IterativeDesignCrew({"Yield Strength": 1100})
    loop.loop(max_iterations=2)
