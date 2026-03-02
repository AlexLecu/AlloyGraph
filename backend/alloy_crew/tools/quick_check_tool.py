from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type, Any
import json
import logging

from ..schemas import CompositionStr

from ..models.feature_engineering import (
    compute_alloy_features, wt_to_at_percent, estimate_partitioning,
    LATTICE_COEFFS, calculate_lattice_parameter, calculate_em_rule_of_mixtures,
)
from ..config.alloy_parameters import (
    TCP, classify_tcp_risk, is_sss_alloy, get_alloy_class,
    get_params, get_coeff_gp, get_sss_physics_ys, get_temperature_factor,
    get_em_temp_factor, compress_uts_ys_ratio,
)

logger = logging.getLogger(__name__)


def estimate_physics_ys(composition: dict, processing: str = "cast",
                        temperature_c: int = 20) -> float:
    """Conservative YS estimate for Phase 1 gate checks.

    Returns a *conservative* yield strength estimate in MPa.  This is
    intentionally lower than the raw physics model because the full
    evaluation pipeline (Phase 3) blends ML predictions, applies KG
    anchoring, and imposes penalties that systematically reduce YS by
    ~25-30%.  Using the raw physics estimate causes the Phase 1 feedback
    loop to never trigger (every composition looks feasible).

    Key adjustments vs. raw physics:
    - Mismatch boost capped at 0.5% (above that, coherency degrades)
    - CAL_YS_FACTOR applied (0.9 for wrought/cast)
    - ML-gap discount (0.82) to approximate final evaluation values
    """
    features = compute_alloy_features(composition)
    gp = features.get("gamma_prime_estimated_vol_pct", 0)
    sss_wt = features.get("SSS_total_wt_pct", 0)
    delta = features.get("lattice_mismatch_pct", 0)

    if is_sss_alloy(composition):
        ys_rt, _ = get_sss_physics_ys(composition, processing)
        # Apply same conservative discount as γ' path — raw SSS physics
        # also overpredicts vs. final evaluation (ML blending + penalties).
        params = get_params(processing)
        cal_factor = params.get("CAL_YS_FACTOR", 1.0)
        ml_gap_discount = 0.82
        ys_rt *= cal_factor * ml_gap_discount
        temp_factor = get_temperature_factor(temperature_c, "sss")
        return ys_rt * temp_factor

    params = get_params(processing)
    base = params["BASE_NI"] + params.get("HALL_PETCH_BOOST", 0)
    sss_contrib = params["SSS_CONTRIBUTION_FACTOR"] * sss_wt
    coeff_gp = get_coeff_gp(processing, "standard")

    # Cap mismatch boost at 0.5%: above that coherency is lost
    # and precipitate hardening degrades, not improves.
    mismatch_boost = min(abs(delta), 0.5) * 100.0

    ys_rt = base + sss_contrib + (coeff_gp * gp) + mismatch_boost

    # Apply calibration factor from alloy_parameters
    cal_factor = params.get("CAL_YS_FACTOR", 1.0)
    # ML-gap discount: raw physics overpredicts vs. final evaluation by ~25%
    # because it doesn't account for ML blending, penalties, or KG anchoring.
    # Calibrated from Run 5 data: avg(final_ys / physics_ys) ≈ 0.73.
    # Using 0.82 (conservative — not as aggressive as the empirical 0.73)
    # to ensure only genuinely weak compositions trigger feedback.
    ml_gap_discount = 0.82
    ys_rt *= cal_factor * ml_gap_discount

    alloy_class = get_alloy_class(composition, processing)
    ac = "sc_ds" if alloy_class == "sc_ds" else "gp"
    temp_factor = get_temperature_factor(temperature_c, ac, gp_fraction=gp if ac == "gp" else None)

    return ys_rt * temp_factor


def compute_mismatch_drivers(composition: dict, features: dict) -> list:
    """Compute per-element contributions to lattice mismatch.

    Returns list of (element, mismatch_contribution_pct, wt_pct) sorted by
    absolute contribution (descending).  Only elements with |contribution| > 0.01%.
    """
    gp = features.get("gamma_prime_estimated_vol_pct", 0)
    if gp < 1:
        return []

    at_pct = wt_to_at_percent(composition)
    c_gamma, c_gp = estimate_partitioning(at_pct, gp)

    a_g = calculate_lattice_parameter(c_gamma)
    a_gp = calculate_lattice_parameter(c_gp)
    denom = a_gp + a_g
    if denom == 0:
        return []

    contributions = []
    for el in composition:
        if el == "Ni" or composition[el] <= 0:
            continue
        k_a = LATTICE_COEFFS.get(el, 0.1)
        delta_c = (c_gp.get(el, 0) - c_gamma.get(el, 0)) / 100.0
        delta_a = delta_c * k_a
        mismatch_contrib = 200.0 * delta_a / denom
        if abs(mismatch_contrib) > 0.01:
            contributions.append((el, round(mismatch_contrib, 3), composition[el]))

    contributions.sort(key=lambda x: abs(x[1]), reverse=True)
    return contributions


class QuickCheckInput(BaseModel):
    """Input for quick physics validation."""
    composition: CompositionStr = Field(
        ...,
        description="JSON string of alloy composition in wt%, e.g. '{\"Ni\": 60.0, \"Cr\": 19.5}'",
    )
    processing: str = Field("cast", description="Processing route: 'cast' or 'wrought'.")
    temperature_c: int = Field(20, description="Service temperature in Celsius. Affects estimated yield strength.")


class QuickCheckTool(BaseTool):
    """Rapid physics validation for alloy compositions.

    Checks TCP risk (Md thresholds), lattice mismatch, Cr range,
    gamma-prime fraction, and processing compatibility.  Returns a
    structured pass/fail with specific warnings so the Designer can
    fix issues BEFORE submitting a composition.
    """
    name: str = "QuickCheckTool"
    description: str = (
        "Validates an alloy composition against physics constraints BEFORE full evaluation. "
        "Returns gamma prime %, Md values, TCP risk, lattice mismatch, density, and "
        "estimated properties: estimated_ys_mpa, estimated_uts_mpa, estimated_el_pct, estimated_em_gpa. "
        "IMPORTANT: Set temperature_c to your service temperature for accurate estimates. "
        "Compare ALL estimated properties to your targets. "
        "If YS is low, add γ' formers (Al, Ti) or SSS strengtheners (Mo, W). "
        "If EL is low, reduce γ' formers. If EM is low, add W or Mo (high-modulus elements)."
    )
    args_schema: Type[BaseModel] = QuickCheckInput

    def _run(
        self,
        composition: Any = None,
        processing: str = "cast",
        temperature_c: int = 20,
        **kwargs: Any,
    ) -> str:
        if isinstance(composition, str):
            composition = json.loads(composition) if composition else {}
        if not composition:
            return json.dumps({"valid": False, "error": "Empty composition"})

        try:
            features = compute_alloy_features(composition)
        except Exception as e:
            return json.dumps({"valid": False, "error": f"Feature computation failed: {e}"})

        md_gamma = features.get("Md_gamma", 0)
        md_avg = features.get("Md_avg", 0)
        gp = features.get("gamma_prime_estimated_vol_pct", 0)
        delta = features.get("lattice_mismatch_pct", 0)
        density = features.get("density_calculated_gcm3", 0)
        tcp_level = classify_tcp_risk(md_gamma, md_avg)
        alloy_class = get_alloy_class(composition, processing)

        warnings = []

        # TCP risk
        if tcp_level in ("Critical", "Elevated"):
            warnings.append(
                f"CRITICAL: TCP risk={tcp_level} (Md_avg={md_avg:.3f} > {TCP['MD_ELEVATED']}). "
                f"Reduce Re/W/Mo or increase Cr/Co."
            )
        elif tcp_level == "Moderate":
            warnings.append(
                f"WARNING: TCP Moderate (Md_avg={md_avg:.3f} > {TCP['MD_MODERATE']}). "
                f"Approaching stability limit."
            )

        # Lattice mismatch — with per-element drivers
        mismatch_driver_list = compute_mismatch_drivers(composition, features) if abs(delta) > 0.3 else []

        if abs(delta) > 0.5:
            severity = "CRITICAL" if abs(delta) > 0.8 else "WARNING"
            threshold_msg = "coherency loss risk" if abs(delta) > 0.8 else "target < 0.5%"
            driver_strs = [f"{el}({wt:.1f}%): {c:+.3f}%" for el, c, wt in mismatch_driver_list[:3]]
            tip = ""
            # If Ti is a top-2 positive driver, suggest Al substitution
            top_positive = [(el, c) for el, c, _ in mismatch_driver_list[:3] if c > 0.02]
            if any(el == "Ti" for el, _ in top_positive):
                tip = (
                    " FIX: Al boosts γ' with ~4x LESS mismatch than Ti. "
                    "Reduce Ti, increase Al. Also consider Nb (moderate mismatch but strong γ' former)."
                )
            warnings.append(
                f"{severity}: Lattice mismatch={delta:.2f}% > {'0.8' if abs(delta) > 0.8 else '0.5'}% "
                f"({threshold_msg}). Drivers: {', '.join(driver_strs)}.{tip}"
            )

        # Cr range
        cr = composition.get("Cr", 0)
        if cr < 5:
            warnings.append(f"CRITICAL: Cr={cr:.1f}% < 5% (insufficient oxidation resistance).")
        elif cr > 20:
            warnings.append(f"CRITICAL: Cr={cr:.1f}% > 20% (sigma phase risk).")
        elif cr >= 18 and processing == "wrought":
            warnings.append(
                f"WARNING: Cr={cr:.1f}% >= 18% for wrought "
                f"(Cr>18% + Mo/W pushes TCP toward Critical for disc alloys). Target Cr=12-15%."
            )

        # Processing-specific checks
        if processing == "wrought":
            if gp > 50:
                al = composition.get("Al", 0)
                ti = composition.get("Ti", 0)
                ta = composition.get("Ta", 0)
                nb = composition.get("Nb", 0)
                warnings.append(
                    f"CRITICAL: gamma'={gp:.0f}% > 50% wrought limit. Cannot hot-work. "
                    f"Al={al:.1f}, Ti={ti:.1f}, Ta={ta:.1f}, Nb={nb:.1f} — reduce formers."
                )
            gp_index = (composition.get("Al", 0) + composition.get("Ti", 0)
                        + composition.get("Ta", 0) + 0.35 * composition.get("Nb", 0))
            if gp_index > 7.0:
                warnings.append(
                    f"WARNING: GP former index (Al+Ti+Ta+0.35*Nb)={gp_index:.1f}% > 7% (wrought limit)."
                )
        elif processing == "cast":
            if gp > 65:
                warnings.append(f"WARNING: gamma'={gp:.0f}% > 65% (high for cast polycrystalline).")

        # γ' formers check for non-SSS
        if not is_sss_alloy(composition):
            gp_formers = (composition.get("Al", 0) + composition.get("Ti", 0)
                          + composition.get("Ta", 0) + 0.35 * composition.get("Nb", 0))
            if gp_formers < 3:
                warnings.append(
                    f"WARNING: Low gamma' formers (Al+Ti+Ta+0.35Nb={gp_formers:.1f}% < 3%). "
                    f"May limit strength."
                )

        # Estimated yield strength from physics model (at service temperature)
        estimated_ys = estimate_physics_ys(composition, processing, temperature_c)

        # Estimated UTS from YS × ratio (temperature-compressed)
        if processing in ("wrought", "forged"):
            base_ratio = 1.15 if gp > 40 else 1.40
        else:
            base_ratio = 1.10 + (gp / 100) * 0.15 + 0.10
        uts_ratio = compress_uts_ys_ratio(base_ratio, temperature_c)
        estimated_uts = estimated_ys * uts_ratio

        # Estimated Elongation from empirical γ' correlation
        if is_sss_alloy(composition):
            estimated_el = 40.0  # SSS alloys have high ductility
        elif processing in ("wrought", "forged"):
            estimated_el = max(10.0, 28 - 0.28 * gp)
        else:
            estimated_el = max(4.0, 18 - 0.25 * gp)

        # Estimated Elastic Modulus from Reuss bound
        em_reuss = calculate_em_rule_of_mixtures(composition)
        em_temp = get_em_temp_factor(temperature_c)
        estimated_em = em_reuss * em_temp

        has_critical = any(w.startswith("CRITICAL") for w in warnings)

        # Per-element mismatch breakdown (reuse cached computation)
        mismatch_drivers = {el: c for el, c, _ in mismatch_driver_list[:5]} if mismatch_driver_list else {}

        result = {
            "valid": not has_critical,
            "alloy_class": alloy_class,
            "gamma_prime_pct": round(gp, 1),
            "estimated_ys_mpa": round(estimated_ys, 0),
            "estimated_uts_mpa": round(estimated_uts, 0),
            "estimated_el_pct": round(estimated_el, 1),
            "estimated_em_gpa": round(estimated_em, 1),
            "Md_avg": round(md_avg, 3),
            "Md_gamma": round(md_gamma, 3),
            "tcp_risk": tcp_level,
            "lattice_mismatch_pct": round(delta, 3),
            "mismatch_drivers": mismatch_drivers,
            "density_gcm3": round(density, 2),
            "warnings": warnings,
        }
        return json.dumps(result, indent=2)
