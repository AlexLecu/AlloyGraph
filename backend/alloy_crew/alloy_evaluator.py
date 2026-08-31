from typing import Dict, Any
from enum import Enum
import json
import logging

logger = logging.getLogger(__name__)
from crewai import Task, Crew, Process
from .agents import get_evaluation_agents
from .tools.rag_tools import AlloySearchTool
from .tools.ml_tools import AlloyPredictorTool
from .tools.analysis_tool import AlloyAnalysisTool as AnalysisTool
from .schemas import (
    PropertyCorrection,
    PhysicsAuditWithCorrectionsOutput,
    AuditPenalty
)
from .tools.metallurgy_tools import (
    cleanup_confidence,
    compute_metallurgy_validation,
    validate_property_bounds,
    PROPERTY_KEY_MAP, VALID_PROPERTIES,
)
from .tools.calibration_fix import apply_calibration_safe
from .config.alloy_parameters import (
    KG_ANCHOR_MAX_DISTANCE,
    CORRECTION_THRESHOLDS, UTS_YS_RATIO, ELONGATION, AGENT_TRUST, SSS,
    is_sss_alloy, is_sc_ds_alloy, get_em_temp_factor,
)
from .models.feature_engineering import compute_alloy_features, calculate_em_rule_of_mixtures


class TrustDecision(Enum):
    TRUST_PROPOSAL = "trust_proposal"


#: How far past the min/max of the evidence anchors an agent value may sit
#: before it is treated as unsupported. 50% is loose enough to allow genuine
#: agent judgement and tight enough to catch invented magnitudes.
ENVELOPE_TOLERANCE = 0.50

#: Marker written into PropertyCorrection.physics_constraint when the envelope
#: guard fires, so occurrences can be counted across a campaign.
ENVELOPE_CORRECTION_TYPE = "evidence_envelope_override"


def _slim_kg_context(kg_json_str: str, target_temp: int = 20) -> str:
    """Compress KG context to essential fields. Extracts RT or target-temp properties."""
    try:
        alloys = json.loads(kg_json_str)
        slim_alloys = []

        for alloy in alloys[:3]:  # Only top 3 matches
            slim = {
                "name": alloy.get("name", "Unknown"),
                "_distance": round(alloy.get("_distance", 999), 2),
                "processing": alloy.get("processing", "unknown"),
            }

            comp_wt = alloy.get("composition_wt_pct", {})
            if comp_wt:
                slim["composition_wt_pct"] = comp_wt

            props = alloy.get("properties", {})
            slim_props = {}
            for prop_name, prop_values in props.items():
                if isinstance(prop_values, str):
                    parts = prop_values.split(", ")
                    relevant = []
                    for p in parts:
                        if " @ " in p:
                            temp_part = p.split(" @ ")[1].replace("C", "").strip()
                            try:
                                temp_val = float(temp_part)
                                if 15 <= temp_val <= 30 or abs(temp_val - target_temp) < 50:
                                    value_part = p.split(" @ ")[0].strip()
                                    relevant.append(f"{value_part} @ {temp_val:.0f}C")
                            except ValueError:
                                pass
                    if relevant:
                        slim_props[prop_name] = ", ".join(relevant[:2])

            if slim_props:
                slim["properties"] = slim_props

            slim_alloys.append(slim)
        return json.dumps(slim_alloys, separators=(',', ':'))
    except Exception:
        return kg_json_str[:1500] if len(kg_json_str) > 1500 else kg_json_str

class AlloyEvaluationCrew:
    def __init__(self, llm_config=None, agents=None, skip_reviewer: bool = False):
        """skip_reviewer runs the Analyst-only ablation.

        The Analyst's output goes straight to deterministic post-processing:
        every guard, the evidence envelope, the trust system and the correction
        reconciliation still run, exactly as in the full pipeline. Nothing else
        changes -- same agents, same prompts, same tools, same anchors. The only
        difference is that the Reviewer task is not in the crew, which isolates
        what the second agent contributes from what the deterministic layer does.

        Runs in this mode are tagged pipeline_stage="analyst_ablation", kept
        distinct from the pre-existing "analyst_only", which means something
        else entirely: the Reviewer ran but its output could not be parsed and
        the code fell back to the Analyst's.
        """
        if agents:
            self.agents_map = agents
        else:
            self.agents_map = get_evaluation_agents(llm=llm_config)
        self.analyst = self.agents_map['analyst']
        self.reviewer = self.agents_map['reviewer']
        self.llm = self.agents_map.get('llm')  # For direct summary call
        self.skip_reviewer = skip_reviewer

    @staticmethod
    def validate_composition(composition: Dict[str, float]) -> Dict[str, Any]:
        """Validates and cleans composition. Removes non-positive values, warns on totals."""
        warnings = []
        cleaned = {k: max(0.0, float(v)) for k, v in composition.items() if float(v) > 0}

        if not cleaned:
            raise ValueError("Composition is empty after removing non-positives.")

        total = sum(cleaned.values())

        if 99.5 <= total <= 100.5:
            return {"composition": cleaned, "warnings": warnings}

        if total < 95.0:
            warnings.append(f"Composition sums to {total:.1f}%, which seems incomplete. Consider adding missing elements.")
        elif total > 105.0:
            warnings.append(f"Composition sums to {total:.1f}%, which exceeds 100%. This may indicate an error.")
        elif abs(total - 100.0) > 2.0:
            warnings.append(f"Composition sums to {total:.1f}% (expected ~100%).")

        if total < 90.0 or total > 110.0:
            raise ValueError(
                f"Composition total ({total:.1f}%) is outside acceptable range (90-110%). "
                f"Please check your input values."
            )

        return {
            "composition": {k: round(v, 3) for k, v in cleaned.items()},
            "warnings": warnings
        }


    @staticmethod
    def _build_kg_summary(kg_context_str: str, processing: str) -> str:
        """Build human-readable KG summary for agent consumption."""
        try:
            alloys = json.loads(kg_context_str)
            if not isinstance(alloys, list) or not alloys:
                return "KG search returned no matches."
        except Exception:
            return "KG unavailable."

        lines = []
        for i, alloy in enumerate(alloys[:3]):
            name = alloy.get("name", "Unknown")
            dist = alloy.get("_distance", 999)
            proc = alloy.get("processing", "unknown")
            proc_match = "same" if proc == processing else f"different ({proc})"

            props = alloy.get("properties", {})
            prop_parts = []
            for prop_name, prop_val in props.items():
                prop_parts.append(f"{prop_name}={prop_val}")

            prop_str = ", ".join(prop_parts) if prop_parts else "no property data"
            rank = ["Closest", "2nd", "3rd"][i]
            # Flag neighbours past the anchoring cutoff. Distance here is
            # Euclidean wt%, unbounded; a match at 9 wt% shares almost no
            # chemistry with the query and its measured values must not be
            # carried across.
            if isinstance(dist, (int, float)) and dist >= KG_ANCHOR_MAX_DISTANCE:
                lines.append(
                    f"{rank}: {name} (dist={dist:.2f}, {proc_match} processing) "
                    f"— TOO FAR TO ANCHOR (cutoff {KG_ANCHOR_MAX_DISTANCE}); "
                    f"reference only, do not adopt: {prop_str}."
                )
            else:
                lines.append(
                    f"{rank}: {name} (dist={dist:.2f}, {proc_match} processing). {prop_str}."
                )

        return " ".join(lines)

    @staticmethod
    def _build_anchor_text(analysis: dict, ml_fallback: dict) -> str:
        """Build compact anchor summary (ML/Physics/KG values) for agent consumption."""
        _UNITS = {
            "Yield Strength": "MPa", "Tensile Strength": "MPa",
            "Elongation": "%", "Elastic Modulus": "GPa", "Gamma Prime": "%",
        }
        _MECH_PROPS = ["Yield Strength", "Tensile Strength", "Elongation", "Elastic Modulus"]

        def _fmt(prop, val):
            return f"  {prop}: {val:.1f} {_UNITS.get(prop, '')}"

        lines = []

        info = analysis.get("alloy_analysis", {})
        gp = info.get("gamma_prime_pct", 0) or 0
        lines.append(
            f"CLASS: {(info.get('class', '?')).upper()} | "
            f"γ'={gp:.1f}% | ρ={info.get('density_gcm3', 0)} g/cm³"
        )

        preds = analysis.get("predictions", {})
        ml = preds.get("ml", {})
        if ml and not ml.get("error"):
            lines.append("ML:")
            lines.extend(_fmt(p, ml[p]) for p in _MECH_PROPS if ml.get(p) is not None)
        elif ml_fallback:
            ml_pred = ml_fallback
            lines.append("ML (fallback):")
            lines.extend(_fmt(p, ml_pred[p]) for p in _MECH_PROPS if ml_pred.get(p) is not None)

        physics = preds.get("physics", {})
        if physics and not physics.get("error"):
            model = physics.get("physics_model", "physics")
            lines.append(f"PHYSICS ({model}):")
            for p in _MECH_PROPS + ["Gamma Prime"]:
                if physics.get(p) is not None:
                    lines.append(_fmt(p, physics[p]))
            if physics.get("model_breakdown"):
                lines.append(f"  Breakdown: {physics['model_breakdown']}")

        kg = preds.get("kg")
        if kg and kg.get("matched"):
            # Show whether the anchoring gate accepted this match. Previously the
            # rejection was logged but never surfaced here, so a neighbour the
            # gate had thrown out still appeared as authoritative "(experimental)"
            # data. On HASTELLOY X at 21C the Analyst copied UDIMET 630 verbatim
            # from a match 9.14 wt% away, returning YS 1310 against a measured 360.
            gate = analysis.get("kg_gate") or {}
            allowed = gate.get("allowed")
            if allowed is False:
                reason = gate.get("reject_detail") or gate.get("reject_code") or "gate rejected"
                lines.append(
                    f"KG: '{kg.get('name', '?')}' (dist={kg.get('distance', 999):.2f}) "
                    f"— REJECTED FOR ANCHORING ({reason})"
                )
                lines.append(
                    "  DO NOT use these values. This alloy is too dissimilar to inform "
                    "the query; they are shown only so you can see what was considered."
                )
            else:
                lines.append(f"KG: '{kg.get('name', '?')}' (dist={kg.get('distance', 999):.2f})")
            for prop, val in kg.get("properties", {}).items():
                if isinstance(val, (int, float)):
                    tag = "(rejected — do not use)" if allowed is False else "(experimental)"
                    lines.append(f"  {prop}: {val:.1f} {_UNITS.get(prop, '')} {tag}")

        disc = analysis.get("discrepancy", {})
        if disc.get("detected"):
            affected = ", ".join(disc.get("properties_affected", []))
            lines.append(f"⚠️ DISCREPANCY ({disc.get('severity', '?').upper()}): {affected}")
            for d in disc.get("details", []):
                lines.append(
                    f"  {d.get('property', '')}: ML={d.get('ml_value', 0):.0f} vs "
                    f"Physics={d.get('physics_value', 0):.0f} ({d.get('difference_pct', 0):.0f}%)"
                )
        else:
            lines.append("✓ Sources agree.")

        proposals = analysis.get("proposed_corrections", [])
        if proposals:
            lines.append(f"PROPOSALS ({len(proposals)}):")
            for p in proposals:
                lines.append(
                    f"  [{p.get('confidence', '?')}] {p['property_name']}: "
                    f"{p['current_value']:.1f} → {p['proposed_value']:.1f} — {p['reasoning']}"
                )

        return "\n".join(lines)

    @staticmethod
    def _evaluate_agent_trust(
        output: PhysicsAuditWithCorrectionsOutput,
        analysis_anchors: dict,
    ) -> Dict[str, tuple]:
        """Safety net: overrides agent values only for ignored HIGH proposals."""
        _conf_rank = {"HIGH": 2, "MEDIUM": 1, "LOW": 0}
        proposals = {}
        for p in (analysis_anchors or {}).get("proposed_corrections", []):
            name = p["property_name"]
            existing = proposals.get(name)
            if existing is None or _conf_rank.get(p.get("confidence"), 0) >= _conf_rank.get(existing.get("confidence"), 0):
                proposals[name] = p

        original_errors = set(validate_property_bounds(dict(output.properties)))

        def _passes_bounds(prop, val):
            test = dict(output.properties)
            test[prop] = val
            new_errors = set(validate_property_bounds(test)) - original_errors
            return not new_errors

        decisions = {}

        for prop in ["Yield Strength", "Tensile Strength", "Elongation", "Elastic Modulus"]:
            current_val = output.properties.get(prop)
            if not isinstance(current_val, (int, float)):
                continue

            proposal = proposals.get(prop)
            if proposal and proposal.get("confidence") == "HIGH":
                proposed_val = proposal.get("proposed_value")
                if isinstance(proposed_val, (int, float)):
                    threshold = CORRECTION_THRESHOLDS.get(prop, 1.0)
                    if abs(current_val - proposed_val) > threshold and _passes_bounds(prop, proposed_val):
                        decisions[prop] = (
                            TrustDecision.TRUST_PROPOSAL, proposed_val,
                            f"HIGH proposal ignored by both agents: {proposal.get('reasoning', '')[:100]}"
                        )
                        continue

        return decisions

    @staticmethod
    def _build_summary_prompt(
        composition: dict,
        output: PhysicsAuditWithCorrectionsOutput,
        temperature: int,
        alloy_class_label: str,
        summary_context: dict = None,
    ) -> str:
        """Build summary prompt for LLM."""
        comp_str = ", ".join(
            f"{el}: {wt:.1f}%" for el, wt in
            sorted(composition.items(), key=lambda x: x[1], reverse=True)[:5]
        )
        props = output.properties
        confidence = output.confidence if isinstance(output.confidence, dict) else {}

        props_block = (
            f"- YS: {props.get('Yield Strength', 'N/A')} MPa | "
            f"UTS: {props.get('Tensile Strength', 'N/A')} MPa\n"
            f"- EL: {props.get('Elongation', 'N/A')}% | "
            f"EM: {props.get('Elastic Modulus', 'N/A')} GPa\n"
            f"- γ': {props.get('Gamma Prime', 'N/A')} vol% | "
            f"Density: {props.get('Density', 'N/A')} g/cm³"
        )

        audit_block = (
            f"Audit: {output.status} | TCP: {output.tcp_risk} | "
            f"Violations: {len(output.audit_penalties)} | "
            f"Confidence: {confidence.get('level', 'MEDIUM')} ({confidence.get('score', 0.5):.2f})"
        )

        reasoning_lines = []
        if output.analyst_reasoning:
            reasoning_lines.append(f"Analyst: {output.analyst_reasoning[:250]}")
        if output.reviewer_assessment:
            reasoning_lines.append(f"Reviewer: {output.reviewer_assessment[:150]}")
        reasoning_block = "\n".join(reasoning_lines)

        if summary_context:
            targets = []
            if summary_context.get("min_yield", 0) > 0:
                targets.append(f"YS>={summary_context['min_yield']}")
            if summary_context.get("min_tensile", 0) > 0:
                targets.append(f"UTS>={summary_context['min_tensile']}")
            if summary_context.get("min_elongation", 0) > 0:
                targets.append(f"EL>={summary_context['min_elongation']}%")
            if summary_context.get("min_elastic_modulus", 0) > 0:
                targets.append(f"EM>={summary_context['min_elastic_modulus']} GPa")
            if summary_context.get("max_density", 99) < 99:
                targets.append(f"Density<={summary_context['max_density']}")
            if summary_context.get("target_gamma_prime", 0) > 0:
                targets.append(f"γ'≈{summary_context['target_gamma_prime']}%")
            target_str = ", ".join(targets) if targets else "None"

            instruction = (
                f"3-paragraph design summary: (1) What was designed, "
                f"(2) Performance vs target, (3) Trade-offs and recommendations."
            )
            header = f"Target: {target_str}\nComposition: {comp_str}\n"
        else:
            instruction = (
                f"2-3 sentence technical summary: (1) Strengthening mechanism, "
                f"(2) Audit concerns, (3) Applications. Class: {alloy_class_label}"
            )
            header = f"Composition: {comp_str}\n"

        return (
            f"{header}Properties at {temperature}°C:\n{props_block}\n"
            f"{audit_block}\n{reasoning_block}\n\n"
            f"{instruction}\n"
            f"Use ONLY the values above — do not invent numbers."
        )

    def evaluate_properties(
        self,
        composition: dict,
        processing: str = "wrought",
        temperature: int = 900,
        *,
        apply_calibration: bool = False,
        summary_context: dict = None,
        extra_output_fields: dict = None,
    ) -> Dict[str, Any]:
        """Shared evaluation pipeline: Pre-compute → Agents → Trust → Validation → Summary."""
        # === PRE-AGENT COMPUTATION ===
        try:
            search_tool = AlloySearchTool()
            kg_raw = search_tool._run(composition=composition, limit=3, processing=processing)
            kg_context_str = _slim_kg_context(kg_raw, target_temp=temperature)
        except Exception as e:
            logger.warning(f"KG lookup failed (non-fatal): {e}. Continuing without KG context.")
            kg_context_str = "KG unavailable — no experimental reference data."

        analysis_anchors = {}
        ml_fallback = None
        try:
            analysis_tool = AnalysisTool()
            analysis_raw = analysis_tool._run(
                composition=composition,
                temperature_c=temperature,
                processing=processing,
                kg_context=kg_context_str
            )
            analysis_anchors = json.loads(analysis_raw) if isinstance(analysis_raw, str) else analysis_raw
            logger.info(f"Pre-computed analysis anchors (discrepancy={analysis_anchors.get('discrepancy_detected', False)})")

            ml_from_analysis = analysis_anchors.get("predictions", {}).get("ml", {})
            if ml_from_analysis and not ml_from_analysis.get("error"):
                alloy_info = analysis_anchors.get("alloy_analysis", {})
                ml_fallback = {
                    **ml_from_analysis,
                    "Density": alloy_info.get("density_gcm3", 0),
                    "Gamma Prime": alloy_info.get("gamma_prime_pct", 0),
                }
        except Exception as e:
            logger.warning(f"Analysis pre-computation failed: {e}")

        if ml_fallback is None:
            try:
                predictor = AlloyPredictorTool()
                ml_raw = predictor._run(
                    composition=composition,
                    temperature_c=temperature,
                    processing=processing
                )
                ml_fallback = json.loads(ml_raw) if isinstance(ml_raw, str) else ml_raw
            except Exception as e:
                logger.warning(f"ML fallback computation failed: {e}")

        comp_json = json.dumps(composition, ensure_ascii=False)
        anchor_text = self._build_anchor_text(analysis_anchors, ml_fallback)
        kg_summary = self._build_kg_summary(kg_context_str, processing)

        alloy_info = analysis_anchors.get("alloy_analysis", {})
        alloy_class_name = (alloy_info.get("class", "") or "").upper()
        gp_pct_ctx = alloy_info.get("gamma_prime_pct", 0) or 0
        is_sss = alloy_class_name == "SSS" or gp_pct_ctx < 5
        alloy_class_label = (
            f"SSS (solid solution strengthened, 0% gamma-prime)"
            if is_sss else
            f"Gamma-prime precipitation hardened ({gp_pct_ctx:.0f} vol%)"
        )

        # === AGENT PIPELINE ===
        task_analysis = Task(
            description=(
                f"COMPOSITION: {comp_json}\n"
                f"TEMPERATURE: {temperature}°C | PROCESSING: {processing}\n"
                f"ALLOY CLASS: {alloy_class_label}\n\n"
                f"=== ANCHOR VALUES ===\n{anchor_text}\n====================\n\n"
                f"=== PRE-COMPUTED KG MATCHES ===\n{kg_summary}\n==============================\n\n"
                f"WORKFLOW:\n"
                f"1. ALWAYS call AlloySearchTool(composition=<composition dict>, processing='{processing}') to find experimental data.\n"
                f"2. Compare KG experimental values with ML and physics anchors.\n"
                f"3. Select the best value for each property using the decision rules below.\n\n"
                f"DECISION RULES:\n"
                f"- KG match (distance < 2.0): treat experimental values as ground truth\n"
                f"- KG match (distance 2.0-{KG_ANCHOR_MAX_DISTANCE}): weight KG evidence — closer = more trusted\n"
                f"- KG match (distance > {KG_ANCHOR_MAX_DISTANCE}): note findings but rely on ML/physics. "
                f"Matches past this cutoff are rejected for anchoring — do NOT adopt their values.\n"
                f"- Sources agree (within 15%) and no close KG match: use ML value\n"
                f"- SSS alloy + disagreement: prefer Physics (Labusch-Nabarro is calibrated)\n"
                f"- γ' alloy + disagreement: use proposed correction if available\n"
                f"- UTS MUST be >= YS. If your UTS < YS, you have an error — fix it.\n\n"
                f"OUTPUT: status='PASS', processing='{processing}', properties={{...}}\n\n"
                f"FIELD DIRECTIVES:\n"
                f"- analyst_reasoning: for EACH property state 'YS=[val] from [source] — [reason]'. Cite numbers.\n"
                f"- investigation_findings: copy KG matches above, then add your own search results with values.\n"
                f"- source_reliability: state '[ML/Physics/KG] most reliable for this class because [reason]'.\n"
                f"- corrections_applied: only for properties where you chose differently from ML.\n"
                f"- corrections_explanation: what changed and why.\n"
                f"- Do NOT fill metallurgy_metrics, property_intervals, or audit_penalties."
            ),
            expected_output="Property predictions with metallurgical reasoning chain.",
            output_pydantic=PhysicsAuditWithCorrectionsOutput,
            agent=self.analyst
        )

        task_review = Task(
            description=(
                f"Validate and correct the Analyst's predictions. ALLOY CLASS: {alloy_class_label}\n\n"
                f"=== REFERENCE ANCHORS ===\n{anchor_text}\n========================\n\n"
                f"WORKFLOW:\n"
                f"1. Call MetallurgyVerifierTool(composition={comp_json}, "
                f"anchored_properties_json=<Analyst's properties as JSON>, temperature_c={temperature}).\n"
                f"2. For EACH violation the verifier reports:\n"
                f"   - Correct the value using proposals, physics, or KG data from the anchors above.\n"
                f"   - OR justify why the violation is acceptable for this alloy class.\n"
                f"3. Check if HIGH-confidence proposals in the anchors were ignored — apply or justify rejection.\n"
                f"4. If you need independent evidence, search KG with AlloySearchTool(processing='{processing}').\n\n"
                f"OUTPUT: status='PASS', processing='{processing}', properties={{...}}\n\n"
                f"FIELD DIRECTIVES:\n"
                f"- reviewer_assessment: for EACH violation state 'Violation: [property] [issue] — corrected [old]->[new] using [evidence]'.\n"
                f"- corrections_applied: ADD your corrections to the Analyst's list (keep existing ones).\n"
                f"- corrections_explanation: update to include both Analyst and your corrections.\n"
                f"- CRITICAL: If UTS < YS, fix it — this is a physical impossibility.\n"
                f"- Preserve analyst_reasoning, investigation_findings, source_reliability from the Analyst.\n"
                f"- Do NOT fill metallurgy_metrics, property_intervals, or audit_penalties."
            ),
            expected_output="Validated and corrected properties with evidence-based decisions.",
            output_pydantic=PhysicsAuditWithCorrectionsOutput,
            agent=self.reviewer,
            context=[task_analysis]
        )

        if self.skip_reviewer:
            # Ablation: the Reviewer task is built above but never enters the
            # crew, so the prompts stay byte-identical to the full pipeline.
            evaluation_crew = Crew(
                agents=[self.analyst],
                tasks=[task_analysis],
                process=Process.sequential,
                verbose=True
            )
        else:
            evaluation_crew = Crew(
                agents=[self.analyst, self.reviewer],
                tasks=[task_analysis, task_review],
                process=Process.sequential,
                verbose=True
            )

        token_usage = {}
        pipeline_stage = "analyst_ablation" if self.skip_reviewer else "reviewer"

        try:
            crew_output = evaluation_crew.kickoff()

            # Token accounting for cost/throughput measurement. CrewOutput
            # carries a UsageMetrics object; store it as a plain dict so it
            # survives model_dump() and CSV serialisation.
            try:
                tu = getattr(crew_output, "token_usage", None)
                if tu is not None:
                    token_usage = tu.model_dump() if hasattr(tu, "model_dump") else dict(tu)
            except Exception:
                token_usage = {}

            output = None
            pipeline_stage = ("analyst_ablation" if self.skip_reviewer
                              else "reviewer")   # full Analyst -> Reviewer path
            if hasattr(crew_output, "pydantic") and crew_output.pydantic:
                output = crew_output.pydantic
            elif hasattr(crew_output, "raw"):
                try:
                    data = json.loads(crew_output.raw)
                    output = PhysicsAuditWithCorrectionsOutput(**data)
                except Exception:
                    logger.warning("Failed to parse Reviewer output as JSON, attempting recovery...")
                    output = None

            if output is None:
                review_task_output = None if self.skip_reviewer else task_review.output
                if review_task_output and review_task_output.pydantic:
                    output = review_task_output.pydantic
                    pipeline_stage = "reviewer_task_output"
                else:
                    logger.warning("Reviewer task failed, falling back to Analyst output...")
                    analyst_task_output = task_analysis.output
                    if analyst_task_output and analyst_task_output.pydantic:
                        output = analyst_task_output.pydantic
                        if not self.skip_reviewer:
                            pipeline_stage = "analyst_only"
                            output.reviewer_assessment = "Review skipped due to parsing failure."
                        else:
                            output.reviewer_assessment = "Reviewer disabled (analyst-only ablation)."
                    elif ml_fallback:
                        logger.warning("Analyst also failed, using ML fallback...")
                        output = PhysicsAuditWithCorrectionsOutput(
                            status='PASS',
                            processing=processing,
                            properties=ml_fallback,
                            explanation="ML-only fallback — agent pipeline failed.",
                            analyst_reasoning="Agent pipeline failed. Using raw ML predictions.",
                            reviewer_assessment="Review not performed.",
                        )
                        pipeline_stage = "ml_fallback"
                    else:
                        raise ValueError("Could not recover output from any pipeline stage.")

        except Exception as e:
            # Root cause of a real data-loss bug: this early return skipped the
            # extra_output_fields merge ~400 lines below, so a caller that passed
            # the composition in (the Designer's Phase 3 does exactly that) got
            # back a result with no composition at all. A rejected Reviewer
            # output then looked indistinguishable from "we designed nothing".
            # Merge the caller's fields here too, so the failure path keeps
            # whatever context the caller supplied.
            failure = {"status": "FAIL", "stage": "crew_execution", "error": str(e)}
            if extra_output_fields:
                failure.update(extra_output_fields)
            return failure

        # === MERGE ANALYST CORRECTIONS ===
        # The Reviewer LLM sometimes produces a fresh PhysicsAuditWithCorrectionsOutput
        # that drops the Analyst's corrections_applied list (~40% of runs).
        # If the Reviewer's list is empty but the Analyst had corrections, merge them
        # so the reconciliation loop (below) can apply documented adjustments.
        if not output.corrections_applied:
            try:
                analyst_out = task_analysis.output
                if analyst_out and analyst_out.pydantic:
                    analyst_corrections = analyst_out.pydantic.corrections_applied or []
                    if analyst_corrections:
                        logger.info(
                            f"[MERGE] Reviewer dropped {len(analyst_corrections)} Analyst corrections — restoring."
                        )
                        output.corrections_applied = analyst_corrections
            except Exception:
                pass

        # === PROPERTY RECOVERY ===
        required_props = ["Yield Strength", "Tensile Strength", "Elongation", "Elastic Modulus", "Density", "Gamma Prime"]

        def _is_truly_missing(prop_name, value):
            if value is None or value == "N/A":
                return True
            if prop_name in ("Gamma Prime", "Elongation"):
                return False  # 0 is a valid value for these
            return value == 0

        missing_props = [p for p in required_props if p not in output.properties or _is_truly_missing(p, output.properties.get(p))]

        if missing_props:
            logger.warning(f"Missing properties detected: {missing_props}. Attempting recovery...")

            if ml_fallback:
                ml_pred = ml_fallback
                for prop in missing_props[:]:
                    if prop in ml_pred and ml_pred[prop] is not None and ml_pred[prop] != "N/A":
                        output.properties[prop] = ml_pred[prop]
                        missing_props.remove(prop)
                        logger.info(f"Recovered {prop} from ML fallback: {ml_pred[prop]}")

                if not output.confidence:
                    conf = ml_pred.get("confidence", {})
                    if conf:
                        output.confidence = conf
                        logger.info("Recovered confidence from ML fallback")

            if missing_props:
                features = compute_alloy_features(composition)
                if "Density" in missing_props and "density_calculated_gcm3" in features:
                    output.properties["Density"] = round(features["density_calculated_gcm3"], 2)
                    missing_props.remove("Density")
                    logger.info(f"Computed Density from features: {output.properties['Density']}")
                if "Gamma Prime" in missing_props and "gamma_prime_estimated_vol_pct" in features:
                    if is_sss_alloy(composition):
                        output.properties["Gamma Prime"] = 0.0
                        logger.info("SSS alloy (is_sss_alloy=True) — Gamma Prime set to 0%")
                    else:
                        output.properties["Gamma Prime"] = round(features["gamma_prime_estimated_vol_pct"], 1)
                        logger.info(f"Computed Gamma Prime from features: {output.properties['Gamma Prime']}")
                    missing_props.remove("Gamma Prime")

            if missing_props:
                logger.warning(f"Could not recover: {missing_props}")

        # === NORMALIZE PROPERTY KEYS ===
        if not output.corrections_explanation and output.corrections_applied:
            output.corrections_explanation = "Analyst-driven corrections applied based on source triangulation."

        normalized_props = {}
        for key, value in output.properties.items():
            norm_key = PROPERTY_KEY_MAP.get(key, key)
            if isinstance(value, str):  # LLMs sometimes return "520" instead of 520
                try:
                    value = float(value)
                except (ValueError, TypeError):
                    pass
            normalized_props[norm_key] = value
        output.properties = normalized_props

        # === CORRECTION RECONCILIATION ===
        # Apply corrections that agents documented but forgot to update in properties.
        for c in (output.corrections_applied or []):
            norm_name = PROPERTY_KEY_MAP.get(c.property_name, c.property_name)
            current = output.properties.get(norm_name)
            if (isinstance(current, (int, float))
                    and isinstance(c.original_value, (int, float))
                    and isinstance(c.corrected_value, (int, float))
                    and c.corrected_value != c.original_value
                    and abs(current - c.original_value) / max(abs(c.original_value), 1) < 0.05):
                logger.info(
                    f"[RECONCILE] {norm_name}: {current:.1f} → {c.corrected_value:.1f} "
                    f"(correction documented but not applied)"
                )
                output.properties[norm_name] = c.corrected_value

        # === SAFETY NET ===
        trust_decisions = self._evaluate_agent_trust(output, analysis_anchors)

        for prop_name, (decision, value, reason) in trust_decisions.items():
            if decision == TrustDecision.TRUST_PROPOSAL:
                logger.info(f"[SAFETY_NET] {prop_name} -> {value:.1f} ({reason})")
                output.properties[prop_name] = value

        # === DETERMINISTIC OVERRIDES (Density + Gamma Prime are composition-determined) ===
        det_features = compute_alloy_features(composition)
        det_density = round(det_features.get("density_calculated_gcm3", 0), 2)
        det_gp = round(det_features.get("gamma_prime_estimated_vol_pct", 0), 1)
        if is_sss_alloy(composition):
            det_gp = 0.0

        agent_density = output.properties.get("Density")
        if det_density > 0 and agent_density != det_density:
            if isinstance(agent_density, (int, float)) and agent_density > 0:
                logger.info(f"[DET_OVERRIDE] Density: {agent_density} → {det_density} g/cm³ (composition-determined)")
            output.properties["Density"] = det_density

        agent_gp = output.properties.get("Gamma Prime")
        if isinstance(agent_gp, (int, float)) and abs((agent_gp or 0) - det_gp) > 1.0:
            logger.info(f"[DET_OVERRIDE] Gamma Prime: {agent_gp} → {det_gp}% (composition-determined)")
        output.properties["Gamma Prime"] = det_gp

        # === UTS DAMPING & EL FLOOR (precip alloys only) ===
        if (ml_fallback
                and not is_sss_alloy(composition)
                and not is_sc_ds_alloy(composition, processing)[0]):
            ml_ys_val = ml_fallback.get("Yield Strength")
            ml_uts_val = ml_fallback.get("Tensile Strength")
            agent_ys_val = output.properties.get("Yield Strength")
            agent_uts_val = output.properties.get("Tensile Strength")

            if (all(isinstance(v, (int, float)) and v > 0
                    for v in [ml_ys_val, ml_uts_val, agent_ys_val, agent_uts_val])):
                ys_change_pct = (agent_ys_val - ml_ys_val) / ml_ys_val
                damped_uts = ml_uts_val * (1 + 0.5 * ys_change_pct)
                if abs(damped_uts - agent_uts_val) > 5:
                    logger.info(
                        f"[UTS_DAMP] Precip UTS: agent={agent_uts_val:.1f} → "
                        f"damped={damped_uts:.1f} (ML_UTS={ml_uts_val:.1f}, "
                        f"YS change={ys_change_pct:+.1%})"
                    )
                    output.properties["Tensile Strength"] = round(damped_uts, 1)

            ml_el_val = ml_fallback.get("Elongation")
            agent_el_val = output.properties.get("Elongation")
            if (isinstance(ml_el_val, (int, float)) and ml_el_val > 0
                    and isinstance(agent_el_val, (int, float)) and agent_el_val > 0):
                el_floor = ml_el_val * 0.80
                if agent_el_val < el_floor:
                    logger.info(
                        f"[EL_DAMP] Precip EL: agent={agent_el_val:.1f}% → "
                        f"floor={el_floor:.1f}% (ML={ml_el_val:.1f}%, max 20% reduction)"
                    )
                    output.properties["Elongation"] = round(el_floor, 1)

        # === UTS/YS ENFORCEMENT ===
        ys_val = output.properties.get("Yield Strength")
        uts_val = output.properties.get("Tensile Strength")

        if (isinstance(ys_val, (int, float)) and isinstance(uts_val, (int, float))
                and ys_val > 0 and uts_val < ys_val):
            min_uts = round(ys_val * 1.05, 1)
            logger.warning(
                f"[UTS_FLOOR] UTS ({uts_val:.1f}) < YS ({ys_val:.1f}) — "
                f"setting UTS = YS × 1.05 = {min_uts:.1f}"
            )
            output.properties["Tensile Strength"] = min_uts
            uts_val = min_uts

        if isinstance(ys_val, (int, float)) and isinstance(uts_val, (int, float)) and ys_val > 0:
            ratio = uts_val / ys_val
            gp_val = det_gp
            if is_sss_alloy(composition):
                max_ratio = SSS["UTS_YS_RATIO_MAX_WROUGHT"] if processing in ["wrought", "forged"] else SSS["UTS_YS_RATIO_MAX_CAST"]
            elif processing in ["wrought", "forged"] and gp_val > 40:
                max_ratio = UTS_YS_RATIO["WROUGHT_HIGH_GP_MAX"]
            elif processing in ["wrought", "forged"]:
                max_ratio = UTS_YS_RATIO["WROUGHT_MAX"]
            else:
                max_ratio = UTS_YS_RATIO["CAST_BASE"] + (gp_val / 100) * UTS_YS_RATIO["CAST_GP_FACTOR"] + 0.10
            if ratio > max_ratio:
                capped_uts = round(ys_val * max_ratio, 1)
                logger.info(
                    f"[UTS_CAP] UTS/YS ratio {ratio:.2f} > max {max_ratio:.2f} — "
                    f"capping UTS: {uts_val:.1f} → {capped_uts:.1f}"
                )
                output.properties["Tensile Strength"] = capped_uts

        # === ELONGATION CAP ===
        el_val = output.properties.get("Elongation")
        if isinstance(el_val, (int, float)) and el_val > 0:
            is_cast_poly = (
                processing not in ["wrought", "forged"]
                and not is_sc_ds_alloy(composition, processing)[0]
            )
            if det_gp > 60:
                cap = ELONGATION["HIGH_GP_MAX_EL_CAST"] if is_cast_poly else ELONGATION["HIGH_GP_MAX_EL"]
                if el_val > cap:
                    logger.info(
                        f"[EL_CAP] γ'={det_gp:.0f}% (>60%) {'cast-poly' if is_cast_poly else ''} — "
                        f"capping EL: {el_val:.1f}% → {cap}%"
                    )
                    output.properties["Elongation"] = cap
            elif det_gp > 40:
                cap = ELONGATION["MOD_GP_MAX_EL_CAST"] if is_cast_poly else ELONGATION["MOD_GP_MAX_EL"]
                if el_val > cap:
                    logger.info(
                        f"[EL_CAP] γ'={det_gp:.0f}% (40-60%) {'cast-poly' if is_cast_poly else ''} — "
                        f"capping EL: {el_val:.1f}% → {cap}%"
                    )
                    output.properties["Elongation"] = cap

        # === EVIDENCE ENVELOPE (YS/UTS may not float free of the anchors) ===
        # The agents may pick any value they can justify, but nothing previously
        # bounded the magnitude of Yield Strength or Tensile Strength. The
        # UTS/YS ratio check passes any internally consistent pair, so a wholly
        # invented pair survives: HASTELLOY X at 21C returned YS 1310 / UTS 1520
        # against ML 394 / physics 354 and a measured 360.
        #
        # A gate-rejected knowledge-graph match is deliberately NOT part of the
        # envelope. Including it would let the very value that caused that
        # failure define its own ceiling and the guard would pass it.
        _preds = (analysis_anchors or {}).get("predictions") or {}
        _ml = _preds.get("ml") if isinstance(_preds.get("ml"), dict) else {}
        _phys = _preds.get("physics") if isinstance(_preds.get("physics"), dict) else {}
        envelope_sources = [d for d in (_ml, _phys) if d]
        kg_anchor = _preds.get("kg")
        kg_gate_state = (analysis_anchors or {}).get("kg_gate") or {}
        if kg_anchor and kg_gate_state.get("allowed") is True:
            envelope_sources.append(kg_anchor.get("properties") or {})

        for prop in ("Yield Strength", "Tensile Strength"):
            agent_val = output.properties.get(prop)
            if not isinstance(agent_val, (int, float)) or agent_val <= 0:
                continue
            anchors = [s.get(prop) for s in envelope_sources
                       if isinstance(s.get(prop), (int, float)) and s.get(prop) > 0]
            if not anchors:
                continue
            lo, hi = min(anchors), max(anchors)
            if lo * (1 - ENVELOPE_TOLERANCE) <= agent_val <= hi * (1 + ENVELOPE_TOLERANCE):
                continue

            fallback = None
            for src in (_ml, _phys):
                if isinstance(src, dict) and isinstance(src.get(prop), (int, float)) and src[prop] > 0:
                    fallback = float(src[prop])
                    break
            if fallback is None:
                continue

            logger.warning(
                "[ENVELOPE] %s=%.1f outside anchor envelope [%.1f, %.1f] "
                "+/-%.0f%% — falling back to %.1f",
                prop, agent_val, lo, hi, ENVELOPE_TOLERANCE * 100, fallback,
            )
            output.properties[prop] = round(fallback, 1)
            output.corrections_applied.append(PropertyCorrection(
                property_name=prop,
                original_value=float(agent_val),
                corrected_value=round(fallback, 1),
                correction_reason=(
                    f"Agent value {agent_val:.0f} lies outside the evidence envelope "
                    f"[{lo:.0f}, {hi:.0f}] by more than {ENVELOPE_TOLERANCE:.0%}; "
                    f"reverted to the ML/physics anchor."
                ),
                physics_constraint=ENVELOPE_CORRECTION_TYPE,
            ))

        # === EM ENFORCEMENT (override if >15% from VRH bound) ===
        # Not applied to single crystals or DS alloys: Voigt-Reuss-Hill averages
        # over randomly oriented grains, which those alloys do not have, so it
        # does not estimate the measured [001] modulus at all. See
        # physics_corrections.CorrectionProfile.skip_em_override_for_sc_ds.
        em_val = output.properties.get("Elastic Modulus")
        _em_sc_ds, _em_sc_reason = is_sc_ds_alloy(composition, processing)
        if _em_sc_ds and isinstance(em_val, (int, float)) and em_val > 0:
            logger.info(
                f"[EM_OVERRIDE_SKIPPED] SC/DS alloy ({_em_sc_reason}) — keeping "
                f"EM={em_val:.1f} GPa; VRH is undefined for single-crystal moduli"
            )
        elif isinstance(em_val, (int, float)) and em_val > 0:
            em_rt = calculate_em_rule_of_mixtures(composition)
            em_temp_factor = get_em_temp_factor(temperature)
            em_physics = round(em_rt * em_temp_factor, 1)
            if em_physics > 0:
                em_deviation = abs(em_val - em_physics) / em_physics
                if em_deviation > 0.15:
                    logger.info(
                        f"[EM_OVERRIDE] Agent EM={em_val:.1f} GPa deviates {em_deviation:.0%} from "
                        f"physics VRH={em_physics:.1f} GPa — overriding"
                    )
                    output.properties["Elastic Modulus"] = em_physics

        # === CALIBRATION (optional, used by design pipeline) ===
        if apply_calibration:
            # Track which properties were corrected by agents or safety net
            corrected_props = set()
            for c in (output.corrections_applied or []):
                corrected_props.add(PROPERTY_KEY_MAP.get(c.property_name, c.property_name))
            for prop_name, (decision, _, _) in trust_decisions.items():
                if decision == TrustDecision.TRUST_PROPOSAL:
                    corrected_props.add(prop_name)

            # Extract deterministic KG distance from pre-computed context
            det_kg_distance = None
            try:
                kg_alloys = json.loads(kg_context_str)
                if isinstance(kg_alloys, list) and kg_alloys:
                    det_kg_distance = kg_alloys[0].get("_distance", None)
            except Exception:
                pass

            # Save agent-corrected values before calibration
            saved_corrections = {p: output.properties[p] for p in corrected_props
                                 if p in output.properties and isinstance(output.properties[p], (int, float))}

            output.properties = apply_calibration_safe(
                output.properties, composition, output,
                kg_distance_override=det_kg_distance,
            )

            # Restore agent-corrected values (skip calibration for these)
            for prop, val in saved_corrections.items():
                if output.properties.get(prop) != val:
                    logger.info(f"[CAL_SKIP] {prop}: keeping agent-corrected {val:.1f} "
                                f"(calibration would have set {output.properties.get(prop)})")
                    output.properties[prop] = val

        # === DETERMINISTIC VALIDATION ===
        validation_result = compute_metallurgy_validation(
            properties=output.properties,
            composition=composition,
            temperature_c=temperature,
            processing=processing,
            confidence=output.confidence if isinstance(output.confidence, dict) else {}
        )

        output.penalty_score = validation_result["penalty_score"]
        output.tcp_risk = validation_result["tcp_risk"]
        output.property_intervals = validation_result["property_intervals"]
        output.metallurgy_metrics = validation_result["metallurgy_metrics"]
        output.audit_penalties = [AuditPenalty(**p) for p in validation_result["audit_penalties"]]
        output.status = validation_result["status"]

        logger.info(f"Deterministic validation: status={validation_result['status']}, "
                    f"penalty={validation_result['penalty_score']}, tcp_risk={validation_result['tcp_risk']}")

        # === GENERATE SUMMARY ===
        try:
            summary_prompt = self._build_summary_prompt(
                composition, output, temperature, alloy_class_label, summary_context
            )

            llm = self.llm
            if llm is None or not hasattr(llm, 'call'):
                llm = getattr(self.analyst, 'llm', None)

            if llm and hasattr(llm, 'call'):
                summary_text = llm.call(summary_prompt)
                output.explanation = str(summary_text)
            else:
                logger.warning("No LLM available for summary generation")

        except Exception as e:
            logger.warning(f"Could not generate summary: {e}")
            if not output.explanation:
                output.explanation = validation_result.get("summary", "")

        # === CLEANUP ===
        output.properties = {k: v for k, v in output.properties.items() if k in VALID_PROPERTIES}
        output.confidence = cleanup_confidence(output.confidence)

        normalized_corrections = []
        for c in output.corrections_applied:
            reason_text = (c.correction_reason or "").strip().lower()
            if any(p.lower() in reason_text for p in AGENT_TRUST["PLACEHOLDER_STRINGS"]):
                continue
            norm_name = PROPERTY_KEY_MAP.get(c.property_name, c.property_name)
            if norm_name in VALID_PROPERTIES:
                c.property_name = norm_name
            normalized_corrections.append(c)
        output.corrections_applied = normalized_corrections

        # === BUILD RESULT ===
        result = output.model_dump()
        result["token_usage"] = token_usage
        # Which stage actually produced these numbers. Without this a row that
        # silently degraded to Analyst-only or to raw ML is indistinguishable
        # from a full Analyst->Reviewer result in the evaluation CSVs.
        result["pipeline_stage"] = pipeline_stage

        if extra_output_fields:
            result.update(extra_output_fields)

        return result

    def run(self, composition: dict, processing: str = "wrought", temperature: int = 900) -> Dict[str, Any]:
        """Public entry point: validates composition then runs evaluate_properties()."""
        try:
            validation_result = self.validate_composition(composition)
            current_comp = validation_result["composition"]
            comp_warnings = validation_result.get("warnings", [])
        except Exception as e:
            return {"status": "FAIL", "stage": "validation", "error": str(e)}

        result = self.evaluate_properties(current_comp, processing, temperature)
        if comp_warnings and isinstance(result, dict):
            result["composition_warnings"] = comp_warnings
        return result
