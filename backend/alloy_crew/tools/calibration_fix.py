import logging

logger = logging.getLogger(__name__)

def get_calibration_factor(composition, confidence_level, kg_distance):
    """
    Apply composition-dependent calibration to physics predictions.

    Rationale:
    - Physics formula YS = 400 + 18×γ' overpredicts for modern alloys by ~15-20%
    - High Cr content reduces strength more than formula accounts for
    - High Co content also reduces (solid solution vs precipitation tradeoff)

    Returns scaling factors for each property.
    """
    cr = composition.get("Cr", 0)
    co = composition.get("Co", 0)
    re = composition.get("Re", 0)

    # Base calibration: Formula overpredicts by ~16% on average
    ys_factor = 0.85

    # Composition adjustments
    if cr > 12:
        ys_factor *= (1.0 - 0.01 * (cr - 12))

    if co > 15:
        ys_factor *= (1.0 - 0.005 * (co - 15))

    if re > 3:
        ys_factor *= (1.0 + 0.03 * re)

    # Confidence-based blending
    if confidence_level in ["LOW", "VERY LOW"] or kg_distance > 10:
        # No KG match: trust calibration fully
        blend_weight = 1.0
    elif confidence_level == "MEDIUM" or kg_distance > 5:
        # Moderate confidence: 70% calibration, 30% original
        blend_weight = 0.7
    else:
        # High confidence with KG match: minimal calibration
        blend_weight = 0.3

    final_ys_factor = 1.0 + blend_weight * (ys_factor - 1.0)
    uts_factor = 1.0 + blend_weight * (0.88 - 1.0)
    
    if cr > 12:
        uts_factor *= (1.0 - 0.008 * (cr - 12))
    if co > 15:
        uts_factor *= (1.0 - 0.004 * (co - 15))
    
    em_factor = 1.0 + blend_weight * (1.05 - 1.0)

    return {
        "Yield Strength": final_ys_factor,
        "Tensile Strength": uts_factor,
        "Elastic Modulus": em_factor,
        "Elongation": 1.0
    }


def apply_calibration(properties, composition, confidence_level, kg_distance):
    """
    Apply calibration factors to corrected properties.
    """
    factors = get_calibration_factor(composition, confidence_level, kg_distance)

    calibrated = properties.copy()
    for prop, factor in factors.items():
        if prop in calibrated and factor != 1.0:
            original = calibrated[prop]
            new_value = original * factor

            if new_value <= 0 or not (abs(new_value) < 1e10):
                logger.warning(f"Calibration produced invalid value for {prop}: {new_value}. Keeping original {original:.1f}")
                continue

            calibrated[prop] = round(new_value, 1)
            logger.info(f"Calibration applied to {prop}: {original:.1f} -> {calibrated[prop]:.1f} (x{factor:.3f})")

    return calibrated


def apply_calibration_safe(properties, composition, physics_output_or_confidence):
    """
    Safely apply calibration with automatic error handling.

    This is a convenience wrapper that handles both full physics_output objects
    and standalone confidence dicts, with built-in exception handling.
    """
    try:
        # Extract confidence info from either object or dict
        if hasattr(physics_output_or_confidence, 'confidence'):
            confidence_dict = physics_output_or_confidence.confidence
            confidence_level = confidence_dict.get("level", "MEDIUM") if isinstance(confidence_dict, dict) else "MEDIUM"
            kg_distance = confidence_dict.get("similarity_distance", 999) if isinstance(confidence_dict, dict) else 999
        else:
            confidence_level = physics_output_or_confidence.get("level", "MEDIUM")
            kg_distance = physics_output_or_confidence.get("similarity_distance", 999)

        return apply_calibration(properties, composition, confidence_level, kg_distance)

    except Exception as e:
        logger.warning(f"Calibration failed: {e}. Continuing with uncalibrated values.")
        return properties.copy()
