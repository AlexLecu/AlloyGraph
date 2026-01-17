from crewai.tools import BaseTool
from pydantic import BaseModel
from typing import Dict, Type, Any, List
import json
import logging

logger = logging.getLogger(__name__)

from ..models.feature_engineering import compute_alloy_features, MD_VALUES
from ..schemas import ElementSuggestion, SuggestionGroup, CompositionSensitivityInput

# =============================================================================
# Physical Constants (Evidence-Based)
# =============================================================================

# TCP risk thresholds (Reed 2006, Pollock & Tin 2006)
MD_TCP_THRESHOLD = 1.00  # Elevated risk threshold - suggest optimization above this
MD_HIGH_RISK = 1.05      # Critical risk threshold - strongly recommend optimization

# Strengthening coefficients
GP_STRENGTH_COEFFICIENT = 35.0  # MPa per vol% γ' (Pollock & Tin 2006)
MAX_GP_WROUGHT = 6.0     # Max γ' for wrought processing
MAX_GP_CAST = 10.0       # Max γ' for cast processing

# Element groups with Md values
MD_RAISERS = [("Re", 1.267), ("Mo", 1.55), ("W", 1.655), ("Ta", 2.224), ("Nb", 2.117)]
MD_DEPRESSORS = [("Cr", 1.142), ("Co", 0.777), ("Ni", 0.717)]
GP_FORMERS = ["Al", "Ti", "Ta", "Nb"]
SSS_ELEMENTS = [("Mo", 12.0), ("W", 10.0)]  # (element, MPa per wt%)


class AlloyOptimizationAdvisor(BaseTool):
    name: str = "AlloyOptimizationAdvisor"
    description: str = (
        "Analyzes a failed alloy design and provides physics-based suggestions for compositional adjustments. "
        "Calculates sensitivities (e.g., ∂Md/∂Re, ∂γ'/∂Al) and ranks suggestions by effectiveness with trade-off analysis."
    )
    args_schema: Type[BaseModel] = CompositionSensitivityInput

    def _perturb_composition(self, composition: Dict[str, float], element: str, delta_wt: float = 1.0) -> Dict[str, float]:
        """
        Perturb composition by delta_wt and renormalize to 100%.
        DRY helper to avoid repeated normalization code.
        """
        perturbed = composition.copy()
        perturbed[element] = perturbed.get(element, 0) + delta_wt
        total = sum(perturbed.values())
        return {k: (v / total) * 100.0 for k, v in perturbed.items()}

    def _calculate_md_sensitivity(self, composition: Dict[str, float], element: str) -> float:
        """Calculate dMd/dElement using finite differences."""
        if element not in MD_VALUES:
            return 0.0
        
        base_md = compute_alloy_features(composition)["Md_avg"]
        perturbed = self._perturb_composition(composition, element, delta_wt=1.0)
        new_md = compute_alloy_features(perturbed)["Md_avg"]
        
        return new_md - base_md

    def _calculate_gp_sensitivity(self, composition: Dict[str, float], element: str) -> float:
        """Calculate dγ'/dElement (vol%) using finite differences."""
        if element not in GP_FORMERS:
            return 0.0
        
        base_gp = compute_alloy_features(composition)["gamma_prime_estimated_vol_pct"]
        perturbed = self._perturb_composition(composition, element, delta_wt=1.0)
        new_gp = compute_alloy_features(perturbed)["gamma_prime_estimated_vol_pct"]
        
        return new_gp - base_gp

    def _generate_tcp_suggestions(
        self, 
        composition: Dict[str, float], 
        current_md: float,
        processing: str
    ) -> SuggestionGroup:
        """Generate suggestions to reduce TCP risk (lower Md)."""
        suggestions = []
        
        # Suggest reducing high Md elements
        for elem, md_val in MD_RAISERS:
            current = composition.get(elem, 0)
            if current > 1.0:  # Only suggest if element is present
                sensitivity = self._calculate_md_sensitivity(composition, elem)
                
                # Calculate reduction needed (aim for Md < 1.0 for optimal stability)
                delta_needed = min(current - 0.5, current * 0.3)  # Reduce by 30% or to 0.5%, whichever is less
                expected_md_change = sensitivity * (-delta_needed)
                
                suggestions.append(ElementSuggestion(
                    element=elem,
                    current_wt=round(current, 2),
                    suggested_wt=round(current - delta_needed, 2),
                    delta_wt=round(-delta_needed, 2),
                    reason=f"Reduce {elem} to lower Md (high Md value: {md_val})",
                    expected_md_change=round(expected_md_change, 3),
                    trade_offs=f"May reduce creep resistance" if elem in ["Re", "W"] else ""
                ))
        
        # Suggest increasing Md-depressors
        for elem, md_val in MD_DEPRESSORS:
            current = composition.get(elem, 0)
            if elem == "Cr" and current < 15.0:
                sensitivity = self._calculate_md_sensitivity(composition, elem)
                delta_add = min(3.0, 15.0 - current)
                expected_md_change = sensitivity * delta_add
                
                suggestions.append(ElementSuggestion(
                    element=elem,
                    current_wt=round(current, 2),
                    suggested_wt=round(current + delta_add, 2),
                    delta_wt=round(delta_add, 2),
                    reason=f"Increase {elem} to lower Md (low Md value: {md_val})",
                    expected_md_change=round(expected_md_change, 3),
                    trade_offs="Improves corrosion resistance" if elem == "Cr" else ""
                ))
        
        # Sort by expected impact
        suggestions.sort(key=lambda x: abs(x.expected_md_change), reverse=True)
        
        return SuggestionGroup(
            issue="TCP Risk (High Md)",
            priority="CRITICAL",
            suggestions=suggestions[:5],  # Top 5 suggestions
            rationale=f"Current Md={current_md:.3f}, target <{MD_TCP_THRESHOLD}. Reduce high-Md elements or increase low-Md elements."
        )

    def _generate_strength_suggestions(
        self,
        composition: Dict[str, float],
        current_ys: float,
        target_ys: float,
        processing: str
    ) -> SuggestionGroup:
        """Generate suggestions to increase yield strength."""
        suggestions = []
        delta_needed = target_ys - current_ys
        
        # γ' formers are the primary strengthening mechanism
        features = compute_alloy_features(composition)
        current_gp = features["gamma_prime_estimated_vol_pct"]
        
        # Estimate γ' needed
        gp_boost_needed = delta_needed / GP_STRENGTH_COEFFICIENT
        
        for elem in GP_FORMERS:
            current = composition.get(elem, 0)
            sensitivity = self._calculate_gp_sensitivity(composition, elem)
            
            if sensitivity > 0.1:  # Only suggest if element is effective
                wt_pct_needed = gp_boost_needed / (sensitivity + 0.01)
                wt_pct_needed = min(wt_pct_needed, 3.0)  # Cap at +3 wt%
                
                # Check processing limits
                if processing == "wrought":
                    gp_total = sum(composition.get(e, 0) for e in GP_FORMERS)
                    if gp_total + wt_pct_needed > MAX_GP_WROUGHT:
                        wt_pct_needed = max(0.0, MAX_GP_WROUGHT - gp_total)
                
                if wt_pct_needed > 0.5:
                    suggestions.append(ElementSuggestion(
                        element=elem,
                        current_wt=round(current, 2),
                        suggested_wt=round(current + wt_pct_needed, 2),
                        delta_wt=round(wt_pct_needed, 2),
                        reason=f"Increase γ' former to boost YS (+{int(wt_pct_needed * sensitivity * GP_STRENGTH_COEFFICIENT)} MPa expected)",
                        expected_gp_change=round(wt_pct_needed * sensitivity, 1),
                        trade_offs="May reduce ductility slightly"
                    ))
        
        # Solid solution strengtheners
        for elem, coeff in SSS_ELEMENTS:
            current = composition.get(elem, 0)
            if current < 8.0:  # Room to add
                delta_add = min(2.0, 8.0 - current)
                expected_ys_gain = delta_add * coeff
                
                md_sensitivity = self._calculate_md_sensitivity(composition, elem)
                md_impact = md_sensitivity * delta_add
                
                suggestions.append(ElementSuggestion(
                    element=elem,
                    current_wt=round(current, 2),
                    suggested_wt=round(current + delta_add, 2),
                    delta_wt=round(delta_add, 2),
                    reason=f"Add solid solution strengthener (+{int(expected_ys_gain)} MPa expected)",
                    expected_md_change=round(md_impact, 3),
                    trade_offs=f"CAUTION: Raises Md by ~{abs(md_impact):.3f}" if md_impact > 0.01 else "Neutral for Md"
                ))
        
        return SuggestionGroup(
            issue=f"Low Yield Strength ({int(current_ys)} < {int(target_ys)} MPa)",
            priority="HIGH",
            suggestions=suggestions[:4],
            rationale=f"Need +{int(delta_needed)} MPa. Increase γ' formers (Al, Ti, Ta) or solid solution strengtheners (Mo, W)."
        )

    def _generate_gp_reduction_suggestions(
        self,
        composition: Dict[str, float],
        current_gp: float,
        target_gp: float
    ) -> SuggestionGroup:
        """Generate suggestions to reduce gamma prime fraction (for low-γ' alloys like NIMOCAST 263)."""
        suggestions = []

        gp_excess = current_gp - target_gp

        # Suggest reducing formers (Al, Ti, Ta have biggest impact)
        former_priorities = [("Al", "Primary"), ("Ti", "Secondary"), ("Ta", "Tertiary")]

        for elem, priority in former_priorities:
            current = composition.get(elem, 0)
            if current > 0.5:  # Only if element is present
                sensitivity = self._calculate_gp_sensitivity(composition, elem)

                if abs(sensitivity) > 0.1:  # Only if sensitive
                    # Calculate reduction needed
                    wt_reduction = min(current - 0.2, gp_excess / (abs(sensitivity) + 0.01))
                    wt_reduction = max(0.5, min(wt_reduction, current * 0.5))  # Reduce by 50% max

                    expected_gp_change = -wt_reduction * abs(sensitivity)

                    suggestions.append(ElementSuggestion(
                        element=elem,
                        current_wt=round(current, 2),
                        suggested_wt=round(current - wt_reduction, 2),
                        delta_wt=round(-wt_reduction, 2),
                        reason=f"Reduce {elem} to lower γ' fraction (currently {current_gp:.1f}%, target {target_gp:.1f}%)",
                        expected_gp_change=round(expected_gp_change, 1),
                        trade_offs=f"Will reduce YS by ~{int(abs(expected_gp_change) * GP_STRENGTH_COEFFICIENT)} MPa"
                    ))

        return SuggestionGroup(
            issue=f"Gamma Prime Too High ({current_gp:.1f}% >> {target_gp:.1f}%)",
            priority="CRITICAL",
            suggestions=suggestions[:3],
            rationale=f"Current γ'={current_gp:.1f}% far exceeds target {target_gp:.1f}%. This is WRONG alloy class! Reduce Al+Ti+Ta content significantly."
        )

    def _run(
        self,
        composition: Dict[str, float],
        target_properties: Dict[str, float],
        current_properties: Dict[str, float],
        failure_reasons: List[str],
        processing: str = "cast",
        **kwargs: Any
    ) -> str:
        """
        Analyze composition and generate physics-based optimization suggestions.
        """
        try:
            features = compute_alloy_features(composition)
            current_md = features["Md_avg"]
            current_gp = features["gamma_prime_estimated_vol_pct"]
            
            suggestion_groups = []
            
            # Check if TCP is an issue
            tcp_issues = [r for r in failure_reasons if "TCP" in r or "Md" in r]
            if tcp_issues or current_md > MD_HIGH_RISK:
                tcp_group = self._generate_tcp_suggestions(composition, current_md, processing)
                suggestion_groups.append(tcp_group)
            
            # CRITICAL: Check if gamma prime target exists and is violated
            target_gp = target_properties.get("Gamma Prime", 0)
            gp_tolerance = 2.0  # Default minimum tolerance
            
            if target_gp > 0:
                gp_tolerance = max(2.0, target_gp * 0.2)  # Same as in alloy_designer.py
                gp_max = target_gp + gp_tolerance

                if current_gp > gp_max:
                    # γ' is TOO HIGH - this is the MOST CRITICAL issue for low-γ' alloys!
                    gp_group = self._generate_gp_reduction_suggestions(
                        composition, current_gp, target_gp
                    )
                    # Put this FIRST - it's critical!
                    suggestion_groups.insert(0, gp_group)
                elif current_gp < (target_gp - gp_tolerance):
                    # γ' is too LOW - need to add formers (handled by strength suggestions)
                    pass

            # Check if strength is an issue (but not if γ' is way too high)
            current_ys = current_properties.get("Yield Strength", 0)
            target_ys = target_properties.get("Yield Strength", 0)

            # Only suggest adding formers if we're not already way over γ' target
            if target_ys > 0 and current_ys < target_ys:
                # Check if we can afford to add more γ' formers
                can_add_formers = True
                if target_gp > 0 and current_gp > (target_gp + gp_tolerance):
                    can_add_formers = False  # Don't suggest adding Al/Ti if γ' already too high!

                if can_add_formers:
                    strength_group = self._generate_strength_suggestions(
                        composition, current_ys, target_ys, processing
                    )
                    suggestion_groups.append(strength_group)

            # Build output
            output = {
                "status": "OK",
                "current_features": {
                    "Md_avg": current_md,
                    "gamma_prime_vol": current_gp,
                    "TCP_risk": "High" if current_md > MD_HIGH_RISK else "Low"
                },
                "suggestion_groups": [g.model_dump() for g in suggestion_groups],
                "summary": f"Generated {len(suggestion_groups)} suggestion groups with {sum(len(g.suggestions) for g in suggestion_groups)} total options."
            }
            
            return json.dumps(output, indent=2)
            
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            return json.dumps({"status": "ERROR", "error": str(e)})
