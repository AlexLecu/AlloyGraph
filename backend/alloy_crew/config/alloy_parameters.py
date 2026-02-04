# =============================================================================
# SSS (SOLID SOLUTION STRENGTHENING) ALLOY PARAMETERS
# For alloys with Al+Ti+Ta < 2% (e.g., Haynes 230, Inconel 617)
# =============================================================================

SSS = {
    # Classification threshold
    "AL_TI_TA_MAX": 2.0,           # wt% threshold for SSS classification

    # Yield Strength bounds at room temperature
    "YS_MIN_RT": 240,              # Minimum YS at RT (MPa)
    "YS_MAX_RT": 500,              # Maximum YS at RT (MPa)
    "YS_TYPICAL": 375,             # Typical YS at RT (MPa)

    # Gamma Prime (should be ~0% for SSS alloys)
    "GP_MAX": 5.0,                 # Maximum allowed γ' for SSS (%)

    # Elastic Modulus bounds
    "EM_MIN": 200.0,               # Minimum EM (GPa)
    "EM_MAX": 220.0,               # Maximum EM (GPa)
    "EM_TYPICAL": 212.0,           # Typical EM (GPa)

    # Elongation by processing
    "EL_MIN_WROUGHT": 35.0,        # Min elongation for wrought SSS (%)
    "EL_MAX_WROUGHT": 65.0,        # Max elongation for wrought SSS (%)
    "EL_TYPICAL_WROUGHT": 52.0,    # Typical elongation for wrought SSS (%)
    "EL_MIN_CAST": 5.0,
    "EL_MAX_CAST": 20.0,
    "EL_TYPICAL_CAST": 10.0,

    # UTS/YS ratio for SSS alloys (higher than γ' alloys due to work hardening)
    "UTS_YS_RATIO_MIN": 1.6,
    "UTS_YS_RATIO_MAX": 2.4,
    "UTS_YS_RATIO_TYPICAL": 2.0,

    # SSS potency factors (MPa per wt%) - Labusch-Nabarro model
    "POTENCY": {
        "Re": 18.0, "W": 12.0, "Mo": 10.0, "Nb": 8.0, "Ta": 7.0,
        "Ti": 6.0, "Cr": 6.5, "Fe": 6.0, "Co": 2.0, "Al": 1.0,
        "Mn": 0.5, "Si": 0.3,
    },

    # SSS strength model parameters
    "SIGMA_BASE": 120,             # Base strength (MPa)
    "SIGMA_HP_WROUGHT": 40,        # Hall-Petch for wrought (MPa)
    "SIGMA_HP_CAST": 20,           # Hall-Petch for cast (MPa)
    "BLEND_FACTOR": 0.7,           # Physics/ML blend factor
    "CAST_REDUCTION": 0.80,        # Cast strength reduction factor

    # Temperature degradation parameters
    "TEMP_TRANSITION": 600.0,      # Temperature where decay accelerates (°C)
    "TEMP_DECAY_SLOW": 0.00030,    # Linear decay rate below transition
    "TEMP_DECAY_TAU": 280.0,       # Exponential decay constant above transition
    "TEMP_MIN_FACTOR": 0.12,       # Minimum retention factor at very high T
    "EL_TEMP_TRANSITION": 500.0,   # Elongation temperature transition (°C)
    "EL_TEMP_FACTOR": 0.0019,      # Elongation increase rate with temp
}

# =============================================================================
# GP (GAMMA PRIME) TEMPERATURE DEGRADATION PARAMETERS
# For polycrystalline γ' alloys (Al+Ti+Ta >= 2%)
# =============================================================================

GP_TEMP = {
    # Classification threshold
    "AL_TI_TA_MIN": 2.0,           # wt% threshold for γ' classification

    # Three-stage temperature degradation model
    "STAGE1_END": 750.0,           # End of linear stage (°C)
    "STAGE2_END": 900.0,           # End of first exponential stage (°C)

    # Decay rates
    "DECAY_LINEAR": 0.00020,       # Linear decay rate (per °C)
    "DECAY_TAU1": 400.0,           # First exponential decay constant
    "DECAY_TAU2": 80.0,            # Second exponential decay constant (rapid)
    "MIN_FACTOR": 0.10,            # Minimum retention factor
}

# =============================================================================
# SC/DS (SINGLE CRYSTAL / DIRECTIONALLY SOLIDIFIED) PARAMETERS
# For advanced turbine blade alloys
# =============================================================================

SC_DS = {
    # Temperature degradation (SC/DS retain strength longer)
    "TEMP_TRANSITION": 850.0,      # Higher transition than polycrystalline
    "TEMP_DECAY_TAU": 250.0,       # Slower decay
    "TEMP_MIN_FACTOR": 0.35,       # Better high-temp retention
    "TEMP_DECAY_LINEAR": 0.00006,  # Very slow linear decay

    # Detection thresholds (composition-based)
    "RE_MIN": 2.0,                 # Re content for 2nd+ gen SC
    "TA_W_MIN": 10.0,              # Ta+W threshold with Re
    "TA_ALONE_MIN": 10.0,          # Ta alone threshold for 1st gen
    "TA_W_HIGH": 11.0,             # High Ta+W threshold

    # UTS/YS ratio at room temperature (lower due to single crystal)
    "UTS_YS_RATIO_RT_BASE": 1.12,
    "UTS_YS_RATIO_RT_MIN": 1.05,
    "UTS_YS_RATIO_RT_MAX": 1.20,
}

# =============================================================================
# PROPERTY BOUNDS (physical limits for all superalloys)
# =============================================================================

BOUNDS = {
    # Yield Strength (MPa)
    "YS_MAX": 2000,                # Maximum known superalloy YS

    # Tensile Strength (MPa)
    "UTS_MAX": 2500,               # Maximum known superalloy UTS

    # Elongation (%)
    "EL_MIN": 0,
    "EL_MAX": 100,

    # Elastic Modulus (GPa) - Ni-based superalloys
    "EM_HARD_MIN": 90,             # Absolute minimum
    "EM_HARD_MAX": 300,            # Absolute maximum
    "EM_TYPICAL_MIN": 100,         # Typical range minimum
    "EM_TYPICAL_MAX": 250,         # Typical range maximum
    "EM_EXPECTED_MIN": 180,        # Expected for standard alloys
    "EM_EXPECTED_MAX": 230,

    # Density (g/cm³)
    "DENSITY_MIN": 7.0,
    "DENSITY_MAX": 10.0,
    "DENSITY_TYPICAL_MIN": 7.5,
    "DENSITY_TYPICAL_MAX": 9.5,

    # Gamma Prime (%)
    "GP_MAX": 75,                  # Maximum typical γ'
}

# =============================================================================
# UTS/YS RATIO CONSTRAINTS BY PROCESSING AND CONDITION
# =============================================================================

UTS_YS_RATIO = {
    # Wrought alloys
    "WROUGHT_BASE": 1.40,
    "WROUGHT_MIN": 1.30,
    "WROUGHT_MAX": 1.60,
    "WROUGHT_GP_FACTOR": 0.15,     # Additional ratio per 100% γ'
    "WROUGHT_HIGH_GP_MAX": 1.35,   # Cap for high-γ' (>40%) wrought
    "WROUGHT_HIGH_GP_EXPECTED": 1.30,

    # Cast alloys
    "CAST_BASE": 1.15,
    "CAST_MIN": 1.08,
    "CAST_GP_FACTOR": 0.2,

    # High temperature adjustments
    "ELEVATED_TEMP_THRESHOLD": 400,  # °C
    "ELEVATED_TEMP_EXPECTED": 1.55,
    "ELEVATED_TEMP_MIN": 1.45,

    # Coherency bounds (warning thresholds)
    "COHERENCY_MIN": 1.05,         # Below this = insufficient work hardening
    "COHERENCY_MAX": 1.60,         # Above this = unusual
}

# =============================================================================
# ELONGATION CONSTRAINTS BY PROCESSING AND γ'
# =============================================================================

ELONGATION = {
    # High γ' limits (γ' reduces ductility)
    "HIGH_GP_THRESHOLD": 60,       # γ' % above which ductility is limited
    "HIGH_GP_MAX_EL": 18.0,        # Max elongation for γ' > 60%
    "MOD_GP_THRESHOLD": 40,        # Moderate γ' threshold
    "MOD_GP_MAX_EL": 25.0,         # Max elongation for γ' 40-60%

    # Wrought minimum ductility (wrought should have good ductility)
    "WROUGHT_LOW_GP_THRESHOLD": 25,
    "WROUGHT_LOW_GP_BASE_EL": 22.0,
    "WROUGHT_LOW_GP_FACTOR": 0.3,  # EL = BASE - (GP * FACTOR)
    "WROUGHT_MOD_GP_MIN_EL": 15.0,
}

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


# =============================================================================
# TEMPERATURE DEGRADATION FUNCTIONS
# Unified approach for SSS, GP, and SC/DS alloys
# =============================================================================

def get_temperature_factor(temp_c: float, alloy_class: str) -> float:
    """
    Calculate temperature degradation factor for strength properties.

    Args:
        temp_c: Temperature in Celsius
        alloy_class: One of 'sss', 'gp', 'sc_ds'

    Returns:
        Factor to multiply room-temperature strength (0.0 to 1.0)
    """
    import math

    if temp_c <= 25:
        return 1.0

    if alloy_class == "sss":
        # SSS alloys: linear then exponential decay
        if temp_c <= SSS["TEMP_TRANSITION"]:
            factor = 1.0 - SSS["TEMP_DECAY_SLOW"] * (temp_c - 25)
        else:
            factor_at_trans = 1.0 - SSS["TEMP_DECAY_SLOW"] * (SSS["TEMP_TRANSITION"] - 25)
            delta_t = temp_c - SSS["TEMP_TRANSITION"]
            factor = factor_at_trans * math.exp(-delta_t / SSS["TEMP_DECAY_TAU"])
        return max(factor, SSS["TEMP_MIN_FACTOR"])

    elif alloy_class == "sc_ds":
        # SC/DS alloys: better high-temp retention
        if temp_c <= SC_DS["TEMP_TRANSITION"]:
            factor = 1.0 - SC_DS["TEMP_DECAY_LINEAR"] * (temp_c - 25)
        else:
            factor_at_trans = 1.0 - SC_DS["TEMP_DECAY_LINEAR"] * (SC_DS["TEMP_TRANSITION"] - 25)
            delta_t = temp_c - SC_DS["TEMP_TRANSITION"]
            factor = factor_at_trans * math.exp(-delta_t / SC_DS["TEMP_DECAY_TAU"])
        return max(factor, SC_DS["TEMP_MIN_FACTOR"])

    else:  # gp (polycrystalline γ' alloys)
        # Three-stage degradation model
        if temp_c <= GP_TEMP["STAGE1_END"]:
            factor = 1.0 - GP_TEMP["DECAY_LINEAR"] * (temp_c - 25)
        elif temp_c <= GP_TEMP["STAGE2_END"]:
            factor_s1 = 1.0 - GP_TEMP["DECAY_LINEAR"] * (GP_TEMP["STAGE1_END"] - 25)
            delta_t = temp_c - GP_TEMP["STAGE1_END"]
            factor = factor_s1 * math.exp(-delta_t / GP_TEMP["DECAY_TAU1"])
        else:
            factor_s1 = 1.0 - GP_TEMP["DECAY_LINEAR"] * (GP_TEMP["STAGE1_END"] - 25)
            factor_s2 = factor_s1 * math.exp(-(GP_TEMP["STAGE2_END"] - GP_TEMP["STAGE1_END"]) / GP_TEMP["DECAY_TAU1"])
            delta_t = temp_c - GP_TEMP["STAGE2_END"]
            factor = factor_s2 * math.exp(-delta_t / GP_TEMP["DECAY_TAU2"])
        return max(factor, GP_TEMP["MIN_FACTOR"])


def is_sss_alloy(composition: dict) -> bool:
    """Check if composition is an SSS alloy (Al+Ti+Ta < 2%)."""
    al = composition.get("Al", composition.get("al", 0)) or 0
    ti = composition.get("Ti", composition.get("ti", 0)) or 0
    ta = composition.get("Ta", composition.get("ta", 0)) or 0
    return (al + ti + ta) < SSS["AL_TI_TA_MAX"]


def is_sc_ds_alloy(composition: dict) -> tuple:
    """
    Detect if alloy is Single Crystal (SC) or Directionally Solidified (DS).

    Returns:
        tuple: (is_sc_ds: bool, reason: str)
    """
    re = composition.get("Re", composition.get("re", 0)) or 0
    ru = composition.get("Ru", composition.get("ru", 0)) or 0
    ta = composition.get("Ta", composition.get("ta", 0)) or 0
    w = composition.get("W", composition.get("w", 0)) or 0

    if re >= SC_DS["RE_MIN"]:
        return True, f"Re={re:.1f}% (2nd+ gen SC indicator)"
    if ru >= 1.0 and re >= 1.0:
        return True, f"Ru={ru:.1f}%, Re={re:.1f}% (4th gen SC indicator)"
    if (ta + w) >= SC_DS["TA_W_MIN"] and re >= 1.0:
        return True, f"Ta+W={ta+w:.1f}%, Re={re:.1f}% (SC/DS composition)"
    if (ta + w) >= SC_DS["TA_W_HIGH"] and ta >= 5.0:
        return True, f"Ta+W={ta+w:.1f}%, Ta={ta:.1f}% (1st gen SC composition)"
    if ta >= SC_DS["TA_ALONE_MIN"]:
        return True, f"Ta={ta:.1f}% (1st gen SC indicator)"

    return False, ""


def get_alloy_class(composition: dict) -> str:
    """
    Determine alloy class based on composition.

    Returns:
        One of: 'sss', 'sc_ds', 'gp'
    """
    if is_sss_alloy(composition):
        return "sss"
    is_sc, _ = is_sc_ds_alloy(composition)
    if is_sc:
        return "sc_ds"
    return "gp"


def get_sss_physics_ys(composition: dict, processing: str = "wrought") -> tuple:
    """
    Calculate physics-based YS for SSS alloys using Labusch-Nabarro model.

    Returns:
        tuple: (physics_ys: float, breakdown: str)
    """
    sigma_base = SSS["SIGMA_BASE"]
    sigma_sss = 0.0
    sss_contributions = []

    for element, potency in SSS["POTENCY"].items():
        content = composition.get(element, composition.get(element.lower(), 0)) or 0
        if content > 0:
            contribution = potency * content
            sigma_sss += contribution
            if contribution > 5:
                sss_contributions.append(f"{element}:{contribution:.0f}")

    sigma_hp = SSS["SIGMA_HP_WROUGHT"] if processing == "wrought" else SSS["SIGMA_HP_CAST"]
    physics_ys = sigma_base + sigma_sss + sigma_hp

    if processing == "cast":
        physics_ys = physics_ys * SSS["CAST_REDUCTION"]
        cast_note = f" × {SSS['CAST_REDUCTION']} (cast)"
    else:
        cast_note = ""

    physics_ys = max(SSS["YS_MIN_RT"], min(SSS["YS_MAX_RT"], physics_ys))
    breakdown = f"σ_base={sigma_base} + σ_SSS={sigma_sss:.0f} [{'+'.join(sss_contributions[:4])}] + σ_HP={sigma_hp}{cast_note}"

    return physics_ys, breakdown
