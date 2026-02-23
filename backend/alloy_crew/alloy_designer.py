"""Three-phase alloy design pipeline.

Phase 1 — LLM Creative Synthesis:
    The Designer agent proposes an initial composition using metallurgical
    knowledge, guided by QuickCheckTool for fast physics validation.

Phase 2 — Deterministic Gradient Optimization:
    The DeterministicOptimizer refines the composition using finite-difference
    sensitivities and constrained gradient steps.  No LLM calls.

Phase 3 — LLM Evaluation (once):
    The full Analyst → Reviewer evaluation pipeline runs on the final
    optimized composition to produce explainable, calibrated predictions.

API contract is preserved: ``IterativeDesignCrew(target_props).loop(...)``
returns the same result dict expected by ``app.py`` and ``design.py``.
"""

import gc
import json
import logging
from typing import Optional, Dict

logger = logging.getLogger(__name__)


def _reset_crewai_event_bus():
    """Reset CrewAI event bus to prevent 'Event stack depth limit (100)' crashes.

    Each design iteration spawns multiple CrewAI crews (Synthesis, Evaluation,
    Optimization).  Their events accumulate on a global bus and eventually exceed
    the 100-event depth limit.  This function:
    1. Disables the depth limit (set max_stack_depth=0)
    2. Clears the event stack
    3. Resets legacy event bus attributes
    """
    # Disable the event stack depth limit (0 = no limit)
    try:
        from crewai.events.event_context import EventContextConfig, _event_context_config, _event_id_stack
        config = EventContextConfig(max_stack_depth=0)
        _event_context_config.set(config)
        _event_id_stack.set(())
    except ImportError:
        pass

    # Legacy event bus cleanup (older CrewAI versions)
    for module_path in [
        "crewai.utilities.events",
        "crewai.utilities.event_bus",
        "crewai.telemetry",
    ]:
        try:
            module = __import__(module_path, fromlist=["crewai_event_bus", "event_bus"])
            for attr in ["crewai_event_bus", "event_bus", "_event_bus"]:
                if hasattr(module, attr):
                    bus = getattr(module, attr)
                    if hasattr(bus, "_event_stack"):
                        bus._event_stack.clear()
                    if hasattr(bus, "reset"):
                        bus.reset()
                    if hasattr(bus, "_events"):
                        bus._events.clear()
        except (ImportError, AttributeError):
            pass
    gc.collect()

from crewai import Crew, Task

from .agents import get_design_agents
from .tools.rag_tools import AlloySearchTool
from .alloy_evaluator import AlloyEvaluationCrew
from .config.alloy_parameters import TCP, TCP_RANK, classify_tcp_risk
from .deterministic_optimizer import optimize as deterministic_optimize
from .models.feature_engineering import compute_alloy_features
from .tools.quick_check_tool import estimate_physics_ys, compute_mismatch_drivers


def round_composition(comp: Dict[str, float], decimals: int = 2) -> Dict[str, float]:
    """Round composition values to specified decimal places."""
    return {k: round(v, decimals) for k, v in comp.items()}


def _recover_design_json(text: str) -> Optional[dict]:
    """Try to extract a valid composition dict from malformed LLM output.

    Handles common LLM formatting issues:
    - Multi-line JSON with trailing newlines
    - Markdown code fences (```json ... ```)
    - Extra text after the JSON object
    """
    import re

    if not text:
        return None

    # Strip markdown code fences
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = text.strip()

    # Try to find the outermost JSON object with { ... }
    # Handle nested braces correctly
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    end = -1
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break

    if end == -1:
        return None

    json_str = text[start:end]

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        # Try stripping internal newlines
        json_str = json_str.replace("\n", " ").replace("\r", "")
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            return None

    if not isinstance(data, dict):
        return None

    # Check if this is a composition-wrapper dict (has composition key)
    if "composition" in data and isinstance(data["composition"], dict):
        return data

    # Check if this is a raw composition dict (has Ni key)
    if "Ni" in data and all(isinstance(v, (int, float)) for v in data.values()):
        return {"composition": data, "reasoning": ""}

    return None


class IterativeDesignCrew:
    """Three-phase alloy design pipeline.

    Phase 1: LLM creative synthesis (with QuickCheckTool)
    Phase 2: Deterministic gradient optimization (no LLM)
    Phase 3: Full LLM evaluation on final result (Analyst → Reviewer)
    """

    def __init__(self, target_props):
        self.target_props = target_props
        self.agents = get_design_agents()

        self.designer = self.agents["designer"]
        self.analyst = self.agents["analyst"]
        self.reviewer = self.agents["reviewer"]
        self.llm = self.agents.get("llm")

        self.min_yield = float(target_props.get("Yield Strength", 0))
        self.min_tensile = float(target_props.get("Tensile Strength", 0))
        self.min_elongation = float(target_props.get("Elongation", 0))
        self.min_elastic_modulus = float(target_props.get("Elastic Modulus", 0))
        self.max_density = float(target_props.get("Density", 99.0))
        self.target_gamma_prime = float(target_props.get("Gamma Prime", 0))

        self._search_tool = AlloySearchTool()

        # Reuse the evaluation pipeline (shared Analyst + Reviewer agents)
        self.evaluator = AlloyEvaluationCrew(agents={
            'analyst': self.analyst,
            'reviewer': self.reviewer,
            'llm': self.llm,
        })

        self._setup_tasks()

    # ── Target string builder ───────────────────────────────────────

    def _build_target_string(self, processing: str = "wrought",
                             temperature: int = 20) -> str:
        """Build target property string for task descriptions."""
        from .config.alloy_parameters import get_temperature_factor

        target_parts = []

        # Compute implied γ' from YS target (empirical: YS = BASE + COEFF × γ')
        implied_gp = None
        if self.min_yield > 0:
            temp_factor = get_temperature_factor(temperature, "gp")
            if processing in ("wrought", "forged"):
                ys_rt_needed = self.min_yield / temp_factor if temp_factor > 0 else self.min_yield
                implied_gp = max(0, (ys_rt_needed - 520) / 13)
            else:
                ys_rt_needed = self.min_yield / temp_factor if temp_factor > 0 else self.min_yield
                implied_gp = max(0, (ys_rt_needed - 400) / 10)

            gp_hint = f"  [→ needs γ'≈{implied_gp:.0f}%]" if implied_gp and implied_gp > 5 else ""
            target_parts.append(f"- Yield Strength >= {self.min_yield} MPa{gp_hint}")
        if self.min_tensile > 0:
            target_parts.append(f"- Tensile Strength >= {self.min_tensile} MPa  [follows from YS × ratio]")
        if self.min_elongation > 0:
            # Compute max γ' for this EL target
            if processing in ("wrought", "forged"):
                max_gp_for_el = (28 - self.min_elongation) / 0.28
            else:
                max_gp_for_el = (18 - self.min_elongation) / 0.25
            max_gp_for_el = max(max_gp_for_el, 10)
            target_parts.append(
                f"- Elongation >= {self.min_elongation} %  "
                f"[→ needs γ'<{max_gp_for_el:.0f}%]"
            )
        if self.min_elastic_modulus > 0:
            target_parts.append(
                f"- Elastic Modulus >= {self.min_elastic_modulus} GPa  "
                f"[add W(411GPa) or Mo(329GPa); avoid excess Al(70GPa)]"
            )
        if self.max_density < 99.0:
            target_parts.append(f"- Density <= {self.max_density} g/cm3")
        if self.target_gamma_prime > 0:
            gp_tolerance = max(2.0, self.target_gamma_prime * 0.2)
            target_parts.append(
                f"- Gamma Prime ~ {self.target_gamma_prime}% (target range: "
                f"{self.target_gamma_prime - gp_tolerance:.1f}-"
                f"{self.target_gamma_prime + gp_tolerance:.1f}%). "
                f"Do NOT maximize gamma prime - match the target!"
            )

        # Add γ' balance warning if both YS and EL targets create tension
        if implied_gp and self.min_elongation > 0 and implied_gp > max_gp_for_el:
            target_parts.append(
                f"\nWARNING: YS needs γ'≈{implied_gp:.0f}% but EL needs γ'<{max_gp_for_el:.0f}%. "
                f"Target γ'≈{(implied_gp + max_gp_for_el) / 2:.0f}% as compromise. "
                f"Use SSS strengtheners (Mo, W) to boost YS without more γ'."
            )

        return "\n".join(target_parts) if target_parts else "No specific targets"

    # ── Novelty check ───────────────────────────────────────────────

    def _run_novelty_check(self, composition: Optional[dict]) -> str:
        if not composition:
            return ""
        try:
            rag_result = self._search_tool._run(composition=composition, limit=1)
            if rag_result and "Error" not in str(rag_result):
                rag_data = json.loads(rag_result)
                if isinstance(rag_data, list) and len(rag_data) > 0:
                    match = rag_data[0]
                    name = match.get("name", "Unknown")
                    return f" [Context: Closest match is **{name}**. If different, this is a NOVEL design.]"
        except Exception:
            pass
        return ""

    # ── Task setup ──────────────────────────────────────────────────

    def _setup_tasks(self):
        """Define the Designer synthesis task template (strings only)."""
        self._task_description = (
            "DESIGN OBJECTIVE:\n"
            "Create a Ni-based superalloy composition meeting the targets below.\n\n"

            "TARGETS (at {temperature}C, {processing}):\n"
            "{target_props_str}\n\n"

            "CURRENT STATUS:\n"
            "{base_comp_str}\n\n"

            "ITERATION FEEDBACK:\n"
            "{feedback}\n\n"

            "INSTRUCTIONS:\n"
            "1. Use QuickCheckTool (set temperature_c={temperature}) to validate BEFORE submitting.\n"
            "   Compare ALL estimated properties (YS, UTS, EL, EM) against your targets.\n"
            "2. If QuickCheckTool reports CRITICAL warnings, fix and re-check.\n"
            "3. Focus on getting the right ALLOY CLASS — a deterministic optimizer will\n"
            "   fine-tune ±3% per element afterward, but cannot change alloy class.\n"
            "4. Follow the [→ needs γ'≈X%] and [→ needs γ'<X%] hints in TARGETS.\n"
            "   These tell you exactly what γ' fraction to aim for.\n"
            "5. UTS follows from YS (× ratio). EM: add W/Mo to increase, reduce Al to avoid lowering.\n"
            "   Density: prefer Mo over W; W/Re/Ta/Hf are heavy.\n"
            "6. ALL alloys MUST include grain boundary strengtheners: C, B, and Zr.\n"
            "   Scale amounts to YOUR alloy's YS target and processing:\n"
            "   - Wrought, YS<1100: C=0.02-0.04, B=0.005, Zr=0.03\n"
            "   - Wrought, YS≥1100: C=0.04-0.08, B=0.010-0.015, Zr=0.05-0.08\n"
            "   - Cast: C=0.07-0.15, B=0.015-0.02, Zr=0.06-0.10\n"
            "   Higher C/B strengthens grain boundaries for high-strength alloys.\n\n"

            "SUCCESS CRITERIA:\n"
            "1. QuickCheckTool reports valid=true (no CRITICAL warnings)\n"
            "2. TCP risk = Low (Md_avg < {md_target})\n"
            "3. ALL estimated properties (YS, UTS, EL, EM) within ~10% of targets\n"
            "4. Elements sum to 100.0 wt%\n\n"

            "{novelty_msg}\n\n"

            "OUTPUT: JSON with 'reasoning' (2-3 sentences explaining your approach), "
            "'composition' (dict summing to 100%), 'processing' ('{processing}')."
        )
        self._task_expected_output = "Structured design with clear reasoning and valid composition."

    # ── Phase 1: LLM synthesis ──────────────────────────────────────

    def _phase1_synthesis(
        self,
        start_composition: Optional[dict],
        temperature: int,
        processing: str,
        feedback: Optional[str] = None,
    ) -> dict:
        """Phase 1: LLM designs an initial composition using QuickCheckTool.

        Returns:
            dict with 'composition' key on success, or 'error' key on failure.
        """
        _reset_crewai_event_bus()

        base_comp_str = (
            f"Starting Composition: {json.dumps(start_composition)}"
            if start_composition
            else "No starting comp - create from scratch"
        )

        target_str = self._build_target_string(processing, temperature)
        novelty_msg = self._run_novelty_check(start_composition)

        inputs = {
            "base_comp_str": base_comp_str,
            "target_props_str": target_str,
            "feedback": feedback or "None (Initial Design)",
            "temperature": temperature,
            "processing": processing,
            "novelty_msg": novelty_msg,
            "md_target": TCP["MD_DESIGN_TARGET"],
        }

        # Fresh task each call — NO output_pydantic to avoid CrewAI's strict
        # JSON validation rejecting multi-line LLM output.  We parse the
        # raw text ourselves with _recover_design_json (brace-matching).
        task = Task(
            description=self._task_description,
            expected_output=self._task_expected_output,
            agent=self.designer,
        )

        crew = Crew(
            agents=[self.designer],
            tasks=[task],
            verbose=True,
        )

        try:
            crew.kickoff(inputs=inputs)
        except Exception as e:
            return {"error": f"Synthesis failed: {e}"}

        # Extract composition from raw text output
        raw = getattr(getattr(task, "output", None), "raw", "") or ""
        if not raw:
            return {"error": "Designer returned empty output."}

        recovered = _recover_design_json(raw)
        if not recovered or "composition" not in recovered:
            return {"error": f"Could not extract composition from Designer output: {raw[:200]}"}

        designer_comp = recovered["composition"]
        if isinstance(designer_comp, str):
            try:
                designer_comp = json.loads(designer_comp) if designer_comp else {}
            except json.JSONDecodeError:
                return {"error": f"Invalid composition JSON: {designer_comp[:200]}"}

        reasoning = recovered.get("reasoning", "")
        return self._validate_phase1_composition(designer_comp, processing, reasoning=reasoning)

    def _validate_phase1_composition(
        self, designer_comp: dict, processing: str, reasoning: str = ""
    ) -> dict:
        """Validate and normalise a Phase 1 composition dict.

        Shared by the normal Pydantic extraction path and the JSON recovery
        fallback.  Returns the same dict contract as ``_phase1_synthesis()``.
        """
        if not isinstance(designer_comp, dict) or not designer_comp:
            return {"error": "Designer returned invalid composition."}

        # Strip zero-valued elements
        designer_comp = {k: v for k, v in designer_comp.items() if v > 0}

        total = sum(designer_comp.values())
        if total < 85.0 or total > 110.0:
            return {
                "error": f"Composition sum ({total:.1f}%) is wildly off.",
                "composition": designer_comp,
            }

        # Auto-balance via Ni
        if "Ni" in designer_comp and abs(total - 100.0) > 0.1:
            adjustment = 100.0 - total
            new_ni = designer_comp["Ni"] + adjustment
            if new_ni >= 40.0:
                logger.info(f"Auto-balancing Ni: {designer_comp['Ni']:.2f}% -> {new_ni:.2f}%")
                designer_comp["Ni"] = new_ni
            else:
                return {
                    "error": f"Cannot auto-balance: Ni would drop to {new_ni:.1f}% (<40%).",
                    "composition": designer_comp,
                }

        designer_comp = round_composition(designer_comp, decimals=2)

        # Soft gate: warn about high γ' but let Phase 2 guard handle it.
        # Pre-compute features here so the loop body can reuse them
        # instead of calling compute_alloy_features a second time.
        feats = compute_alloy_features(designer_comp)
        if processing == "wrought":
            gp = feats.get("gamma_prime_estimated_vol_pct", 0)
            if gp > 75:
                al = designer_comp.get("Al", 0)
                ti = designer_comp.get("Ti", 0)
                ta = designer_comp.get("Ta", 0)
                nb = designer_comp.get("Nb", 0)
                gp_index = al + ti + ta + 0.35 * nb
                return {
                    "error": (
                        f"WROUGHT VIOLATION: gamma'={gp:.0f}% far exceeds wrought limit (max ~50%). "
                        f"Formers: Al={al}, Ti={ti}, Ta={ta}, Nb={nb} "
                        f"(GP index Al+Ti+Ta+0.35*Nb={gp_index:.1f}%). Keep GP index < 7%."
                    ),
                    "composition": designer_comp,
                }
            elif gp > 50:
                logger.warning(
                    f"Phase 1: gamma'={gp:.0f}% > 50% wrought limit — "
                    f"Phase 2 optimizer will reduce."
                )

        logger.info(f"Phase 1 complete (recovered): {designer_comp}")
        return {"composition": designer_comp, "reasoning": reasoning, "features": feats}

    # ── Phase 2: Deterministic optimization ─────────────────────────

    def _phase2_optimize(
        self,
        composition: dict,
        temperature: int,
        processing: str,
    ) -> dict:
        """Phase 2: Light-touch Guard + Tune optimization.

        Returns:
            dict with optimized composition, predicted properties, features, tcp_risk.
        """
        logger.info("Phase 2: Starting light-touch optimization (Guard + Tune)...")

        result = deterministic_optimize(
            initial_composition=composition,
            targets=self.target_props,
            temperature_c=temperature,
            processing=processing,
        )

        guard_fixes = result.get("guard_fixes", [])
        if guard_fixes:
            logger.info(f"Phase 2 Guard: {len(guard_fixes)} fixes applied")

        logger.info(
            f"Phase 2 complete: converged={result['converged']}, "
            f"tune_steps={result['steps_used']}, "
            f"TCP={result['tcp_risk']}, "
            f"YS={result['predicted_properties'].get('Yield Strength', 0):.0f}, "
            f"GP={result['predicted_properties'].get('Gamma Prime', 0):.1f}%"
        )

        return result

    # ── Phase 3: Full LLM evaluation ────────────────────────────────

    def _phase3_evaluate(
        self,
        composition: dict,
        temperature: int,
        processing: str,
    ) -> dict:
        """Phase 3: Full Analyst -> Reviewer evaluation pipeline.

        Returns the standard result dict expected by app.py.
        """
        _reset_crewai_event_bus()
        logger.info("Phase 3: Running full LLM evaluation pipeline...")

        novelty_msg = self._run_novelty_check(composition)

        result = self.evaluator.evaluate_properties(
            composition=composition,
            processing=processing,
            temperature=temperature,
            apply_calibration=True,
            summary_context={
                "min_yield": self.min_yield,
                "max_density": self.max_density,
                "min_tensile": self.min_tensile,
                "min_elongation": self.min_elongation,
                "min_elastic_modulus": self.min_elastic_modulus,
                "target_gamma_prime": self.target_gamma_prime,
            },
            extra_output_fields={
                "composition": composition,
                "novelty": novelty_msg,
            },
        )

        return result

    # ── Success checking ────────────────────────────────────────────

    def _is_design_successful(self, result: dict) -> bool:
        """Check if a result meets all success criteria."""
        if result.get("error"):
            return False

        if result.get("tcp_risk", "Unknown") in ("Critical", "Elevated"):
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

        if self.target_gamma_prime > 0:
            actual_gp = float(props.get("Gamma Prime", 0) or 0)
            gp_tolerance = max(2.0, self.target_gamma_prime * 0.2)
            if actual_gp < self.target_gamma_prime - gp_tolerance:
                return False
            if actual_gp > self.target_gamma_prime + gp_tolerance:
                return False

        return True

    # ── Result enrichment for failed designs ────────────────────────

    def _enrich_failure_result(self, result: dict) -> dict:
        """Add issues/recommendations to a failed design result."""
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
                "description": f"Critical TCP phase formation risk (Md={md_val}, safe limit <{TCP['MD_DESIGN_SAFE']}).",
                "recommendation": "Reduce refractory elements (Re, W, Mo) or increase Cr/Co."
            })
        elif tcp_risk == "Elevated":
            issues.append({
                "type": "TCP Risk",
                "severity": "Medium",
                "description": "Elevated TCP risk - approaching danger zone.",
                "recommendation": "Consider reducing refractory element content."
            })

        if penalties:
            for penalty in penalties:
                issues.append({
                    "type": "Audit Violation",
                    "severity": "Medium",
                    "description": f"{penalty.get('name', 'Unknown')}: {penalty.get('reason', '')}",
                    "recommendation": "Review composition constraints."
                })

        for prop_name, target, unit in [
            ("Yield Strength", self.min_yield, "MPa"),
            ("Tensile Strength", self.min_tensile, "MPa"),
            ("Elongation", self.min_elongation, "%"),
            ("Elastic Modulus", self.min_elastic_modulus, "GPa"),
        ]:
            if target > 0:
                actual = float(props.get(prop_name, 0) or 0)
                if actual < target:
                    issues.append({
                        "type": "Target Miss",
                        "severity": "Low",
                        "description": f"{prop_name} {actual:.0f} {unit} < target {target} {unit}.",
                        "recommendation": f"Adjust composition to improve {prop_name}."
                    })

        if self.max_density < 99.0:
            actual_d = float(props.get("Density", 0) or 0)
            if actual_d > self.max_density:
                issues.append({
                    "type": "Target Miss",
                    "severity": "Low",
                    "description": f"Density {actual_d:.2f} > target {self.max_density} g/cm3.",
                    "recommendation": "Reduce heavy elements (W, Ta, Re)."
                })

        if self.target_gamma_prime > 0:
            actual_gp = float(props.get("Gamma Prime", 0) or 0)
            gp_tolerance = max(2.0, self.target_gamma_prime * 0.2)
            gp_min = self.target_gamma_prime - gp_tolerance
            gp_max = self.target_gamma_prime + gp_tolerance

            if actual_gp < gp_min:
                issues.append({
                    "type": "Gamma Prime",
                    "severity": "High",
                    "description": f"Gamma Prime {actual_gp:.1f}% below range {gp_min:.1f}-{gp_max:.1f}%.",
                    "recommendation": f"Increase Al, Ti, or Ta to reach ~{self.target_gamma_prime}%."
                })
            elif actual_gp > gp_max:
                issues.append({
                    "type": "Gamma Prime",
                    "severity": "High",
                    "description": f"Gamma Prime {actual_gp:.1f}% above range {gp_min:.1f}-{gp_max:.1f}%.",
                    "recommendation": f"Reduce Al, Ti, and Ta to reach ~{self.target_gamma_prime}%."
                })

        if issues:
            recommendations.append("Consider relaxing conflicting targets")
            if any(i["type"] == "Gamma Prime" for i in issues):
                recommendations.append("Review gamma prime target - different alloy classes have vastly different fractions")

        result["issues"] = issues
        result["recommendations"] = recommendations
        result["design_status"] = "incomplete"

        has_high = any(i["severity"] == "High" for i in issues)
        if has_high and result.get("status") == "PASS":
            result["status"] = "REJECT"

        return result

    # ── Main loop ───────────────────────────────────────────────────

    def loop(
        self,
        max_iterations: int = 3,
        start_composition: Optional[dict] = None,
        temperature: int = 900,
        processing: str = "cast",
    ) -> dict:
        """Run the 3-phase design pipeline.

        Args:
            max_iterations: Max LLM synthesis attempts for Phase 1 (Phase 2+3
                run once on the best Phase 1 result).
            start_composition: Optional starting composition hint.
            temperature: Service temperature in Celsius.
            processing: "cast" or "wrought".

        Returns:
            Result dict compatible with app.py (composition, properties,
            tcp_risk, confidence, issues, recommendations, etc.).
        """
        logger.info(
            f"=== DESIGN PIPELINE START ===\n"
            f"  Targets: {self.target_props}\n"
            f"  Processing: {processing}, Temperature: {temperature}C\n"
            f"  Max Phase 1 attempts: {max_iterations}"
        )

        # ── PHASE 1: LLM Creative Synthesis ─────────────────────────
        best_phase1 = None
        feedback = None
        phase1_log = []  # Track each attempt for observability
        overshoot_warned = False  # Track if we've given EL overshoot feedback

        # Max γ' compatible with elongation target (empirical heuristic).
        max_gp_for_el = 0  # 0 means no EL constraint
        if self.min_elongation > 0:
            base = 35 if processing != "cast" else 25
            slope = 0.45 if processing != "cast" else 0.35
            # Temperature correction: above 650°C, EL increases ~0.18% per °C
            if temperature > 650:
                base += 0.18 * (temperature - 650)
            max_gp_for_el = max(15, (base - self.min_elongation) / slope)

        for attempt in range(1, max_iterations + 1):
            logger.info(f"--- Phase 1 Attempt {attempt}/{max_iterations} ---")
            _reset_crewai_event_bus()

            phase1_result = self._phase1_synthesis(
                start_composition=start_composition,
                temperature=temperature,
                processing=processing,
                feedback=feedback,
            )

            if "error" in phase1_result:
                error_msg = phase1_result["error"]
                logger.warning(f"Phase 1 attempt {attempt} failed: {error_msg}")
                phase1_log.append({
                    "attempt": attempt, "status": "error",
                    "error": error_msg,
                    "feedback_given": feedback,
                })
                # Use the failed composition as starting point for next attempt
                start_composition = phase1_result.get("composition", start_composition)
                # Give error-specific feedback so the LLM can fix the issue
                if "WROUGHT VIOLATION" in error_msg or "gamma'" in error_msg.lower():
                    feedback = (
                        f"PREVIOUS ATTEMPT REJECTED: {error_msg} "
                        f"For wrought alloys, keep Al+Ti+Ta+0.35*Nb < 7.0 wt% (γ' former index). "
                        f"Target: Al≈3.5, Ti≈2.5, Ta≈0.5, Nb≈1.5 (index=3.5+2.5+0.5+0.53=7.03). "
                        f"This gives γ'≈40-45% which is optimal for wrought."
                    )
                elif "sum" in error_msg.lower() or "total" in error_msg.lower():
                    feedback = (
                        f"PREVIOUS ATTEMPT REJECTED: {error_msg} "
                        f"Ensure all elements sum to exactly 100.0 wt%."
                    )
                else:
                    feedback = (
                        f"PREVIOUS ATTEMPT FAILED: {error_msg} "
                        f"You MUST return valid JSON with exactly 3 fields: "
                        f"'reasoning' (string), 'composition' (dict of element:wt%), "
                        f"'processing' ('wrought' or 'cast'). "
                        f'Example: {{"reasoning": "...", "composition": '
                        f'{{"Ni": 55.0, "Cr": 14.0, "Co": 12.0, "Mo": 3.5, '
                        f'"W": 3.5, "Al": 3.5, "Ti": 2.5, "Nb": 1.5}}, '
                        f'"processing": "{processing}"}}'
                    )
                feedback = f"[Attempt {attempt}/{max_iterations}] {feedback}"
                continue

            # Phase 1 succeeded — quick-validate with physics
            comp = phase1_result["composition"]
            features = phase1_result.get("features") or compute_alloy_features(comp)
            md_gamma = features.get("Md_gamma", 0)
            md_avg = features.get("Md_avg", 0)
            tcp_level = classify_tcp_risk(md_gamma, md_avg)
            gp = features.get("gamma_prime_estimated_vol_pct", 0)
            delta = features.get("lattice_mismatch_pct", 0)

            # Physics YS estimate for strength feasibility
            physics_ys = estimate_physics_ys(comp, processing, temperature)
            ys_target = self.min_yield
            ys_ratio = physics_ys / ys_target if ys_target > 0 else 1.0

            logger.info(
                f"Phase 1 result: TCP={tcp_level}, "
                f"Md_avg={md_avg:.3f}, GP={gp:.1f}%, mismatch={delta:.2f}%, "
                f"est_YS={physics_ys:.0f} ({ys_ratio:.0%} of target), "
                f"comp={comp}"
            )

            phase1_log.append({
                "attempt": attempt, "status": "ok",
                "tcp": tcp_level, "gp": round(gp, 1),
                "mismatch": round(delta, 2),
                "physics_ys": round(physics_ys, 0),
                "ys_ratio": round(ys_ratio, 2),
                "feedback_given": feedback,
            })

            # Keep the best result (prefer better TCP, then EL feasibility, then YS)
            # TCP rank: Low=0 > Moderate=1 > Elevated=2 > Critical=3
            cur_rank = TCP_RANK.get(tcp_level, 4)
            cur_el_ok = max_gp_for_el <= 0 or gp <= max_gp_for_el * 1.2
            if best_phase1 is None:
                best_phase1 = phase1_result
                best_phase1["_physics_ys"] = physics_ys
                best_phase1["_tcp_rank"] = cur_rank
                best_phase1["_gp"] = gp
            else:
                prev_rank = best_phase1.get("_tcp_rank", 4)
                prev_ys = best_phase1.get("_physics_ys", 0)
                prev_gp = best_phase1.get("_gp", 0)
                prev_el_ok = max_gp_for_el <= 0 or prev_gp <= max_gp_for_el * 1.2

                replace = False
                if cur_rank < prev_rank:
                    replace = True  # Better TCP always wins
                elif cur_rank == prev_rank:
                    if cur_el_ok and not prev_el_ok:
                        replace = True  # EL-feasible beats EL-infeasible
                    elif cur_el_ok == prev_el_ok and physics_ys > prev_ys:
                        replace = True  # Same EL status → higher YS wins

                if replace:
                    best_phase1 = phase1_result
                    best_phase1["_physics_ys"] = physics_ys
                    best_phase1["_tcp_rank"] = cur_rank
                    best_phase1["_gp"] = gp

            # Exit condition: Acceptable TCP AND YS within 80% of target
            # Low or Moderate TCP is acceptable — the guard can reduce Moderate.
            # Only Critical/Elevated require another Phase 1 attempt.
            ys_feasible = ys_target <= 0 or physics_ys >= ys_target * 0.80
            tcp_acceptable = tcp_level in ("Low", "Moderate")
            el_risk = max_gp_for_el > 0 and gp > max_gp_for_el * 1.15

            if tcp_acceptable and ys_feasible:
                if el_risk and not overshoot_warned:
                    overshoot_warned = True
                    logger.info(
                        f"Phase 1: YS feasible but γ'={gp:.0f}% > {max_gp_for_el:.0f}% "
                        f"(max for EL≥{self.min_elongation:.0f}%). Giving EL feedback."
                    )
                else:
                    logger.info(f"Phase 1: Acceptable result (TCP={tcp_level}, YS feasible), proceeding to Phase 2")
                    break

            # Generate targeted feedback for next attempt
            if tcp_level in ("Critical", "Elevated"):
                cr_val = comp.get("Cr", 0)
                mo_val = comp.get("Mo", 0)
                w_val = comp.get("W", 0)
                al_val = comp.get("Al", 0)
                ti_val = comp.get("Ti", 0)
                ta_val = comp.get("Ta", 0)
                nb_val = comp.get("Nb", 0)
                # Nb-aware feedback: Nb has Md=2.117, each 1% adds ~0.028 to Md_avg.
                # When Nb>1.5%, it's often the dominant TCP driver — tell LLM to reduce it.
                nb_advice = ""
                if nb_val > 1.5:
                    nb_md_contribution = nb_val * 0.028
                    nb_advice = (
                        f"ALSO REDUCE Nb from {nb_val:.1f}% to ≤1.5% "
                        f"(Nb has Md=2.117, contributing ~{nb_md_contribution:.3f} to Md_avg). "
                    )
                    gp_freeze = (
                        f"Do NOT increase Al/Ti/Ta to compensate — keep them at current levels "
                        f"(Al={al_val:.1f}, Ti={ti_val:.1f}, Ta={ta_val:.1f}). "
                    )
                else:
                    gp_freeze = (
                        f"Do NOT increase Al/Ti/Ta/Nb to compensate — keep them at current levels "
                        f"(Al={al_val:.1f}, Ti={ti_val:.1f}, Ta={ta_val:.1f}, Nb={nb_val:.1f}). "
                    )
                feedback = (
                    f"TCP RISK {tcp_level}: Md_avg={md_avg:.3f} (safe<0.940). "
                    f"Your Cr={cr_val:.1f}, Mo={mo_val:.1f}, W={w_val:.1f} "
                    f"(Mo+W={mo_val+w_val:.1f}%), Nb={nb_val:.1f}%. "
                    f"REDUCE Mo+W total to <6%, keep Cr=12-14%. "
                    f"{nb_advice}"
                    f"IMPORTANT: {gp_freeze}"
                    f"γ'={gp:.0f}% is already adequate. Only adjust Cr/Mo/W/Co{'/Nb' if nb_val > 1.5 else ''}. "
                    f"Reference low-TCP wrought composition: Cr=12.5, Mo=2.7, W=4.3, Nb=1.5, Co=20.7, Ta=1.6, Md_avg=0.928."
                )
            elif ys_target > 0 and not ys_feasible:
                al = comp.get("Al", 0)
                ti = comp.get("Ti", 0)
                nb = comp.get("Nb", 0)

                # Check mismatch situation to give targeted advice
                mismatch_advice = ""
                if abs(delta) > 0.5:
                    drivers = compute_mismatch_drivers(comp, features)
                    ti_driver = next((c for el, c, _ in drivers if el == "Ti"), 0)
                    if ti_driver > 0.1:
                        mismatch_advice = (
                            f" MISMATCH ALERT: Ti({ti:.1f}%) is the primary mismatch driver "
                            f"({ti_driver:+.2f}%). Al has ~4x less mismatch impact per GP contribution. "
                            f"REDUCE Ti to ≤2.5%, INCREASE Al to ≥3.5%. "
                            f"Nb also works (moderate mismatch, strong γ' former)."
                        )
                    else:
                        mismatch_advice = (
                            f" Mismatch={delta:.2f}%. Top drivers: "
                            + ", ".join(f"{el}:{c:+.2f}%" for el, c, _ in drivers[:2])
                            + "."
                        )
                elif abs(delta) < 0.3:
                    # Mismatch is low — can afford more Ti/Nb
                    mismatch_advice = (
                        f" Mismatch is low ({delta:.2f}%) — you have room to add more "
                        f"Ti or Nb for γ' without coherency issues."
                    )

                feedback = (
                    f"STRENGTH INSUFFICIENT: Estimated YS≈{physics_ys:.0f} MPa vs target "
                    f"{ys_target:.0f} MPa ({ys_ratio:.0%}). γ'={gp:.0f}% — need >35% for "
                    f"YS>{ys_target:.0f}.{mismatch_advice} "
                    f"Current: Al={al:.1f}, Ti={ti:.1f}, Nb={nb:.1f}. "
                    f"Target composition: Al≈3.5, Ti≈2.5, Nb≈1.0-1.5, Mo≈3.5, W≈2-3. "
                    f"Reference high-γ' wrought composition: Al=3.5, Ti=2.5, Nb=1.5, Mo=3.5, W=3.0, Co=18, Ta=1.5 → γ'≈40%, YS≈1100."
                )
            elif el_risk:
                al = comp.get("Al", 0)
                ti = comp.get("Ti", 0)
                nb = comp.get("Nb", 0)
                suggested_gp = round((gp + max_gp_for_el) / 2)
                feedback = (
                    f"ELONGATION RISK: γ'={gp:.0f}% is too high for Elongation≥{self.min_elongation:.0f}%. "
                    f"Your YS≈{physics_ys:.0f} already exceeds target {ys_target:.0f} — "
                    f"REDUCE γ' formers slightly to lower γ' toward ~{suggested_gp:.0f}%. "
                    f"Current: Al={al:.1f}, Ti={ti:.1f}, Nb={nb:.1f}. "
                    f"Reduce Al by ~0.5-1.0% and Ti by ~0.5%. Do NOT drop below γ'≈{suggested_gp:.0f}%."
                )
            if feedback:
                feedback = f"[Attempt {attempt}/{max_iterations}] {feedback}"
            start_composition = comp

        if best_phase1 is None:
            return {
                "error": "Phase 1 failed: Designer could not produce a valid composition.",
                "composition": start_composition or {},
                "properties": {},
                "tcp_risk": "Unknown",
                "confidence": {},
                "design_status": "failed",
                "status": "REJECT",
                "issues": [{"type": "Design Failure", "severity": "High",
                            "description": "Designer could not produce a valid composition.",
                            "recommendation": "Try different targets or relax constraints."}],
                "recommendations": ["Relax conflicting targets", "Try a different alloy class"],
                "explanation": "",
                "optimization_log": {
                    "phase1_attempts": len(phase1_log),
                    "phase1_log": phase1_log,
                },
            }

        initial_comp = best_phase1["composition"]
        best_phase1.pop("_physics_ys", None)  # Clean up temp fields
        best_phase1.pop("_gp", None)
        best_phase1.pop("_tcp_rank", None)
        best_phase1.pop("features", None)
        logger.info(f"Phase 1 best composition: {initial_comp}")

        # ── PHASE 2: Deterministic Gradient Optimization ─────────────
        opt_result = self._phase2_optimize(
            composition=initial_comp,
            temperature=temperature,
            processing=processing,
        )

        optimized_comp = opt_result["composition"]
        optimized_comp = round_composition(optimized_comp, decimals=2)
        logger.info(f"Phase 2 optimized composition: {optimized_comp}")

        # ── PHASE 3: Full LLM Evaluation ─────────────────────────────
        result = self._phase3_evaluate(
            composition=optimized_comp,
            temperature=temperature,
            processing=processing,
        )

        # Merge optimization metadata into result
        result["optimization_log"] = {
            "converged": opt_result["converged"],
            "steps_used": opt_result["steps_used"],
            "guard_fixes": opt_result.get("guard_fixes", []),
            "initial_composition": initial_comp,
            "phase1_reasoning": best_phase1.get("reasoning", ""),
            "phase1_attempts": len(phase1_log),
            "phase1_log": phase1_log,
        }
        result["iterations_used"] = len(phase1_log)  # Phase 1 LLM attempts before Phase 2+3

        # ── Final success check ──────────────────────────────────────
        if self._is_design_successful(result):
            result["design_status"] = "success"
            result["issues"] = []
            result["recommendations"] = []
            logger.info("=== DESIGN PIPELINE: SUCCESS ===")
        else:
            result = self._enrich_failure_result(result)
            logger.warning(
                f"=== DESIGN PIPELINE: INCOMPLETE ({len(result.get('issues', []))} issues) ==="
            )

        return result


if __name__ == "__main__":
    loop = IterativeDesignCrew({"Yield Strength": 1100})
    loop.loop(max_iterations=2)
