import logging

from ..config.alloy_parameters import get_params, is_sss_alloy

logger = logging.getLogger(__name__)

def get_calibration_factor(composition, kg_distance, processing="cast"):
    """
    Apply processing-dependent calibration to physics predictions."""
    if kg_distance < 4.5:
        logger.info(f"KG match within anchoring range (distance={kg_distance:.2f}) - skipping calibration")
        return {"Yield Strength": 1.0, "Tensile Strength": 1.0, "Elastic Modulus": 1.0, "Elongation": 1.0}

    params = get_params(processing)

    ys_factor = params["CAL_YS_FACTOR"]
    uts_factor = params["CAL_UTS_FACTOR"]
    el_factor = params["CAL_EL_FACTOR"]

    # Composition-dependent calibration for SSS alloys
    # SSS alloys (Al+Ti+Ta < 2%) tend to be over-predicted due to:
    cr = composition.get("Cr", composition.get("cr", 0)) or 0

    if is_sss_alloy(composition):
        # SSS alloy detected - apply processing-specific calibration
        if processing == "cast":
            logger.info(f"Cast SSS alloy - no extra calibration (physics already applies CAST_REDUCTION)")
        elif processing in ["wrought", "forged"]:
            # Wrought SSS alloys
            if cr < 18.0:
                ys_factor = ys_factor * 0.93
                uts_factor = uts_factor * 0.93
                logger.info(f"Low-Cr wrought SSS alloy (Cr={cr:.1f}%) - applying 0.93× reduction to YS/UTS")
            else:
                logger.info(f"High-Cr wrought SSS alloy (Cr={cr:.1f}%) - using standard calibration")


    # Blend calibration based on KG match quality
    if kg_distance > 10:
        blend_weight = 1.0
    elif kg_distance > 5:
        blend_weight = 0.8
    else:
        blend_weight = 0.5

    return {
        "Yield Strength": 1.0 + blend_weight * (ys_factor - 1.0),
        "Tensile Strength": 1.0 + blend_weight * (uts_factor - 1.0),
        "Elastic Modulus": 1.0,
        "Elongation": 1.0 + blend_weight * (el_factor - 1.0)
    }


def apply_calibration(properties, composition, kg_distance, processing="cast"):
    """Apply calibration factors to corrected properties."""
    factors = get_calibration_factor(composition, kg_distance, processing=processing)

    calibrated = properties.copy()
    for prop, factor in factors.items():
        if prop in calibrated and factor != 1.0:
            original = calibrated[prop]
            if not isinstance(original, (int, float)):
                continue
            new_value = original * factor

            if new_value <= 0 or not (abs(new_value) < 1e10):
                logger.warning(f"Calibration produced invalid value for {prop}: {new_value}. Keeping original {original:.1f}")
                continue

            calibrated[prop] = round(new_value, 1)
            logger.info(f"Calibration applied to {prop}: {original:.1f} -> {calibrated[prop]:.1f} (x{factor:.3f})")

    return calibrated


def apply_calibration_safe(properties, composition, physics_output_or_confidence, kg_distance_override=None):
    """Safely apply calibration with automatic error handling."""
    try:
        if physics_output_or_confidence is None:
            return properties.copy()
        if kg_distance_override is not None:
            kg_distance = kg_distance_override
        elif hasattr(physics_output_or_confidence, 'confidence'):
            confidence_dict = physics_output_or_confidence.confidence
            kg_distance = confidence_dict.get("similarity_distance", 999) if isinstance(confidence_dict, dict) else 999
        else:
            kg_distance = physics_output_or_confidence.get("similarity_distance", 999)

        if hasattr(physics_output_or_confidence, 'processing'):
            processing = physics_output_or_confidence.processing
        elif isinstance(physics_output_or_confidence, dict):
            processing = physics_output_or_confidence.get("processing", "cast")
        else:
            processing = "cast"

        return apply_calibration(properties, composition, kg_distance, processing=processing)

    except Exception as e:
        logger.warning(f"Calibration failed: {e}. Continuing with uncalibrated values.")
        return properties.copy()
