# =============================================================================
# WROUGHT ALLOY PARAMETERS
# =============================================================================

WROUGHT = {
    # Gamma Prime Strengthening Coefficients
    # Formula: YS_physics = BASE_STRENGTH + COEFF_GP * γ' + COEFF_SSS * SSS + mismatch
    "COEFF_GP": 28.0,              # γ' strengthening coefficient
    "COEFF_GP_HIGH_STRENGTH": 33.0,  # For high-strength alloy types
    "COEFF_GP_CORROSION": 18.0,      # For corrosion-resistant types

    # Base Strength Components
    "BASE_NI": 120.0,              # Base nickel contribution
    "HALL_PETCH_BOOST": 50.0,      # Grain refinement boost (wrought has finer grains)
    "COEFF_SSS": 5.0,              # Solid solution strengthening coefficient
    "SSS_CONTRIBUTION_FACTOR": 12.0,  # SSS wt% contribution factor

    # ML/Physics Blending Weights (by confidence level)
    "ML_WEIGHT_HIGH_CONF": 0.70,   # 70% ML, 30% physics for HIGH confidence
    "ML_WEIGHT_MED_CONF": 0.60,    # 60% ML, 40% physics for MEDIUM confidence
    "ML_WEIGHT_LOW_CONF": 0.50,    # 50% ML, 50% physics for LOW confidence

    # Physics Enforcement (enforce_physics_constraints)
    "ENFORCE_BASE_YS": 400,        # Base YS for enforcement formula
    "ENFORCE_GP_COEFF": 18,        # γ' coefficient for enforcement

    # Calibration Factors
    # Combined with UTS/YS ratio constraint (1.35 cap for high-γ' wrought in enforce_physics_constraints)
    "CAL_YS_FACTOR": 0.90,         # YS calibration multiplier (10% reduction)
    "CAL_UTS_FACTOR": 0.90,        # UTS calibration multiplier (10% reduction)
    "CAL_EL_FACTOR": 1.0,          # Elongation calibration multiplier

    # Ductility
    "BASE_DUCTILITY": 40.0,        # Base elongation %
    "MIN_ELONGATION": 12.0,        # Minimum elongation floor
}

# =============================================================================
# CAST ALLOY PARAMETERS
# =============================================================================

CAST = {
    # Gamma Prime Strengthening Coefficients
    "COEFF_GP": 10.0,              # γ' strengthening coefficient (lower than wrought)
    "COEFF_GP_HIGH_STRENGTH": 14.0,
    "COEFF_GP_CORROSION": 7.0,

    # Base Strength Components
    "BASE_NI": 120.0,
    "HALL_PETCH_BOOST": 0.0,       # No grain refinement boost for cast
    "COEFF_SSS": 5.0,
    "SSS_CONTRIBUTION_FACTOR": 12.0,

    # ML/Physics Blending Weights
    "ML_WEIGHT_HIGH_CONF": 0.70,
    "ML_WEIGHT_MED_CONF": 0.60,
    "ML_WEIGHT_LOW_CONF": 0.50,

    # Physics Enforcement
    "ENFORCE_BASE_YS": 400,
    "ENFORCE_GP_COEFF": 10,        # Lower coefficient for cast

    # Calibration Factors
    "CAL_YS_FACTOR": 0.95,         # Slight reduction for cast
    "CAL_UTS_FACTOR": 0.95,
    "CAL_EL_FACTOR": 1.0,

    # Ductility
    "BASE_DUCTILITY": 20.0,        # Lower base ductility for cast
    "MIN_ELONGATION": 5.0,
}

# =============================================================================
# COMMON PARAMETERS (processing-independent)
# =============================================================================

COMMON = {
    # Physics Enforcement Thresholds (% deviation to trigger correction)
    "THRESHOLD_LOW_CONF": 40,      # Stricter for low confidence
    "THRESHOLD_MED_CONF": 50,
    "THRESHOLD_HIGH_CONF": 70,     # More lenient for high confidence

    # Blend factors when physics correction is applied
    "BLEND_LOW_CONF": 0.5,         # 50% physics, 50% ML for low confidence
    "BLEND_HIGH_CONF": 0.3,        # 30% physics, 70% ML for high confidence

    # KG Distance Thresholds
    "KG_SKIP_THRESHOLD": 3.0,      # Skip physics enforcement if KG match < this

    # UTS/YS Ratio Constraints
    "UTS_YS_RATIO_MIN": 1.05,
    "UTS_YS_RATIO_MAX": 1.60,
    "UTS_YS_RATIO_EXPECTED": 1.2,

    # Elastic Modulus Bounds
    "EM_MIN_WROUGHT": 200,
    "EM_MAX_WROUGHT": 225,
    "EM_MIN_CAST": 180,
    "EM_MAX_CAST": 215,
}


def get_params(processing: str) -> dict:
    """Get parameters for the specified processing type."""
    if processing in ["wrought", "forged"]:
        return WROUGHT
    else:
        return CAST


def get_coeff_gp(processing: str, alloy_type: str = "standard") -> float:
    """Get the gamma prime coefficient for the given processing and alloy type."""
    params = get_params(processing)

    if alloy_type == "high_strength":
        return params["COEFF_GP_HIGH_STRENGTH"]
    elif alloy_type == "high_corrosion":
        return params["COEFF_GP_CORROSION"]
    else:
        return params["COEFF_GP"]


def get_ml_weight(processing: str, confidence_level: str) -> float:
    """Get the ML blending weight for the given confidence level."""
    params = get_params(processing)

    if confidence_level == "HIGH":
        return params["ML_WEIGHT_HIGH_CONF"]
    elif confidence_level == "MEDIUM":
        return params["ML_WEIGHT_MED_CONF"]
    else:
        return params["ML_WEIGHT_LOW_CONF"]
