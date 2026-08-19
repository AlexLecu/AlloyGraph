"""Deterministic physics corrections shared by the evaluator and the ablations.

This is the single implementation of the post-prediction enforcement rules:
UTS floor, UTS/YS ratio ceiling, elongation caps, and elastic-modulus override.
It was previously written twice -- once inline in ``alloy_evaluator`` (the
production path, applied after the agents) and once copied into
``evaluation/prediction/scripts/generate_predictions.py`` for the
``--ml-deterministic`` ablation, whose comment claimed to mirror the evaluator
but had drifted on three rules.

The ``LEGACY_ABLATION`` profile preserves that drift bit-for-bit so previously
published ablation numbers stay reproducible. ``PRODUCTION`` is what the
evaluator actually applies. See ``CorrectionProfile`` for the differences.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .config.alloy_parameters import (
    ELONGATION,
    SSS,
    UTS_YS_RATIO,
    get_em_temp_factor,
    is_sc_ds_alloy,
    is_sss_alloy,
)
from .models.feature_engineering import calculate_em_rule_of_mixtures

# Minimum UTS/YS ratio: UTS may never fall below YS.
UTS_FLOOR_RATIO = 1.05


@dataclass(frozen=True)
class CorrectionProfile:
    """Which variant of the enforcement rules to apply.

    em_deviation_threshold:
        Override the elastic modulus when the predicted value deviates from the
        Voigt-Reuss-Hill estimate by more than this fraction.
    sss_ratio_processing_aware:
        True  -> SSS alloys use separate wrought/cast UTS/YS ceilings.
        False -> a single flat ceiling of 2.4 regardless of processing, which
                 lets cast SSS alloys keep ratios the evaluator would cap.
    elongation_caps_processing_aware:
        True  -> cast polycrystalline alloys use the tighter *_CAST caps.
        False -> the wrought caps are applied to every alloy.
    skip_em_override_for_sc_ds:
        True  -> leave the elastic modulus alone for single-crystal and
                 directionally-solidified alloys. Voigt-Reuss-Hill is an
                 isotropic polycrystalline aggregate; a single crystal loaded
                 along [001] has no such average, so VRH does not estimate the
                 measured quantity at all. Measured [001] moduli sit near
                 105-130 GPa while VRH returns ~162 GPa for the same
                 compositions, a systematic +53% overshoot.
        False -> apply the override to every class (pre-fix behaviour).
    """

    em_deviation_threshold: float = 0.15
    sss_ratio_processing_aware: bool = True
    elongation_caps_processing_aware: bool = True
    skip_em_override_for_sc_ds: bool = True


#: What ``alloy_evaluator`` applies after the agents.
PRODUCTION = CorrectionProfile()

#: Bug-compatible profile reproducing the ``--ml-deterministic`` ablation as it
#: was published. Differs from PRODUCTION on 49 of the 471 evaluation rows
#: (2 UTS, 12 elongation, 35 elastic modulus).
LEGACY_ABLATION = CorrectionProfile(
    em_deviation_threshold=0.20,
    sss_ratio_processing_aware=False,
    elongation_caps_processing_aware=False,
    skip_em_override_for_sc_ds=False,
)


def max_uts_ys_ratio(composition: Dict[str, float], processing: str,
                     gamma_prime_pct: float,
                     profile: CorrectionProfile = PRODUCTION) -> float:
    """Ceiling on UTS/YS for this alloy class and processing route."""
    wrought = processing in ("wrought", "forged")

    if is_sss_alloy(composition):
        if not profile.sss_ratio_processing_aware:
            return 2.4
        return (SSS["UTS_YS_RATIO_MAX_WROUGHT"] if wrought
                else SSS["UTS_YS_RATIO_MAX_CAST"])

    if wrought:
        if gamma_prime_pct > 40:
            return UTS_YS_RATIO["WROUGHT_HIGH_GP_MAX"]
        return UTS_YS_RATIO["WROUGHT_MAX"]

    return (UTS_YS_RATIO["CAST_BASE"]
            + (gamma_prime_pct / 100.0) * UTS_YS_RATIO["CAST_GP_FACTOR"]
            + 0.10)


def elongation_cap(composition: Dict[str, float], processing: str,
                   gamma_prime_pct: float,
                   profile: CorrectionProfile = PRODUCTION) -> Optional[float]:
    """Maximum plausible elongation, or None when no cap applies."""
    if gamma_prime_pct <= 40:
        return None

    cast_poly = (processing not in ("wrought", "forged")
                 and not is_sc_ds_alloy(composition, processing)[0])
    tighten = profile.elongation_caps_processing_aware and cast_poly

    if gamma_prime_pct > 60:
        return ELONGATION["HIGH_GP_MAX_EL_CAST"] if tighten else ELONGATION["HIGH_GP_MAX_EL"]
    return ELONGATION["MOD_GP_MAX_EL_CAST"] if tighten else ELONGATION["MOD_GP_MAX_EL"]


def vrh_elastic_modulus(composition: Dict[str, float], temperature_c: float) -> float:
    """Temperature-adjusted Voigt-Reuss-Hill elastic modulus, in GPa."""
    return round(calculate_em_rule_of_mixtures(composition) * get_em_temp_factor(temperature_c), 1)


def _num(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def apply_physics_corrections(
    properties: Dict[str, Any],
    composition: Dict[str, float],
    processing: str,
    temperature_c: float,
    gamma_prime_pct: float,
    profile: CorrectionProfile = PRODUCTION,
) -> Tuple[Dict[str, Any], List[str]]:
    """Apply the deterministic enforcement rules.

    Returns a corrected copy of ``properties`` plus human-readable notes for
    each rule that fired. The input mapping is not modified. Applying the
    function twice is a no-op on the second call.
    """
    props = dict(properties)
    notes: List[str] = []

    ys = props.get("Yield Strength")
    uts = props.get("Tensile Strength")

    # 1. UTS floor -- UTS must exceed YS.
    if _num(ys) and _num(uts) and ys > 0 and uts < ys:
        new_uts = round(ys * UTS_FLOOR_RATIO, 1)
        notes.append(f"UTS_FLOOR: {uts:.1f} -> {new_uts:.1f} (was below YS {ys:.1f})")
        props["Tensile Strength"] = new_uts
        uts = new_uts

    # 2. UTS/YS ratio ceiling.
    if _num(ys) and _num(uts) and ys > 0:
        ratio = uts / ys
        max_ratio = max_uts_ys_ratio(composition, processing, gamma_prime_pct, profile)
        if ratio > max_ratio:
            capped = round(ys * max_ratio, 1)
            notes.append(f"UTS_CAP: ratio {ratio:.2f} > {max_ratio:.2f}, UTS {uts:.1f} -> {capped:.1f}")
            props["Tensile Strength"] = capped

    # 3. Elongation cap for high-gamma-prime alloys.
    el = props.get("Elongation")
    if _num(el) and el > 0:
        cap = elongation_cap(composition, processing, gamma_prime_pct, profile)
        if cap is not None and el > cap:
            notes.append(f"EL_CAP: {el:.1f}% -> {cap}% (gamma prime {gamma_prime_pct:.0f}%)")
            props["Elongation"] = cap

    # 4. Elastic modulus override against the VRH estimate.
    #
    # Skipped for single crystals and DS alloys: VRH averages over randomly
    # oriented grains, which a single crystal does not have. Enforcing it
    # replaces a usable prediction with one that is wrong by construction --
    # measured [001] moduli near 105-130 GPa against a VRH estimate near
    # 162 GPa for the same composition.
    em = props.get("Elastic Modulus")
    if _num(em) and em > 0:
        sc_ds, sc_reason = is_sc_ds_alloy(composition, processing)
        if profile.skip_em_override_for_sc_ds and sc_ds:
            notes.append(
                f"EM_OVERRIDE_SKIPPED_SC_DS: kept ML value {em:.1f} GPa; VRH is "
                f"undefined for single-crystal/DS moduli ({sc_reason})"
            )
        else:
            em_physics = vrh_elastic_modulus(composition, temperature_c)
            if em_physics > 0:
                deviation = abs(em - em_physics) / em_physics
                if deviation > profile.em_deviation_threshold:
                    notes.append(f"EM_OVERRIDE: {em:.1f} -> {em_physics:.1f} GPa ({deviation:.0%} off VRH)")
                    props["Elastic Modulus"] = em_physics

    return props, notes
