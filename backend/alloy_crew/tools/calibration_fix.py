import logging

logger = logging.getLogger(__name__)

def get_calibration_factor(composition, kg_distance, processing="cast"):
    """
    Apply composition and processing-dependent calibration to physics predictions.

    - Cast alloys: formulas overpredict, need reduction
    - Wrought alloys: formulas are closer to reality
    - Strong KG match (distance < 1.5): skip calibration, trust experimental data
    """
    if kg_distance < 1.5:
        logger.info(f"Strong KG match (distance={kg_distance:.2f}) - skipping calibration")
        return {"Yield Strength": 1.0, "Tensile Strength": 1.0, "Elastic Modulus": 1.0, "Elongation": 1.0}

    cr = composition.get("Cr", 0)
    co = composition.get("Co", 0)
    re = composition.get("Re", 0)

    is_wrought = processing in ["wrought", "forged"]

    if is_wrought:
        ys_factor = 1.0
        if cr > 18:
            ys_factor *= (1.0 - 0.005 * (cr - 18))
        uts_factor = 1.15
        el_factor = 1.3
        logger.info(f"Wrought calibration: YS×{ys_factor:.2f}, UTS×{uts_factor:.2f}, El×{el_factor:.2f}")
    else:
        ys_factor = 0.85
        if cr > 12:
            ys_factor *= (1.0 - 0.01 * (cr - 12))
        if co > 15:
            ys_factor *= (1.0 - 0.005 * (co - 15))
        if re > 3:
            ys_factor *= (1.0 + 0.03 * re)

        uts_factor = 0.88
        if cr > 12:
            uts_factor *= (1.0 - 0.008 * (cr - 12))
        if co > 15:
            uts_factor *= (1.0 - 0.004 * (co - 15))

        el_factor = 1.0

    # Blend calibration based on KG match quality
    if kg_distance > 10:
        blend_weight = 1.0
    elif kg_distance > 5:
        blend_weight = 0.8
    elif kg_distance > 3:
        blend_weight = 0.5
    else:
        blend_weight = 0.3

    return {
        "Yield Strength": 1.0 + blend_weight * (ys_factor - 1.0),
        "Tensile Strength": 1.0 + blend_weight * (uts_factor - 1.0),
        "Elastic Modulus": 1.0 + blend_weight * 0.05,
        "Elongation": 1.0 + blend_weight * (el_factor - 1.0)
    }


def apply_calibration(properties, composition, kg_distance, processing="cast"):
    """Apply calibration factors to corrected properties."""
    factors = get_calibration_factor(composition, kg_distance, processing=processing)

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
    """Safely apply calibration with automatic error handling."""
    try:
        if hasattr(physics_output_or_confidence, 'confidence'):
            confidence_dict = physics_output_or_confidence.confidence
            kg_distance = confidence_dict.get("similarity_distance", 999) if isinstance(confidence_dict, dict) else 999
            processing = getattr(physics_output_or_confidence, 'processing', 'cast')
        else:
            kg_distance = physics_output_or_confidence.get("similarity_distance", 999)
            processing = physics_output_or_confidence.get("processing", "cast")

        return apply_calibration(properties, composition, kg_distance, processing=processing)

    except Exception as e:
        logger.warning(f"Calibration failed: {e}. Continuing with uncalibrated values.")
        return properties.copy()
