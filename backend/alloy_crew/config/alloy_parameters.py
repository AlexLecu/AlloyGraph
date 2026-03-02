# =============================================================================
# SSS (SOLID SOLUTION STRENGTHENING) ALLOY PARAMETERS
# For alloys with Al+Ti+Ta < 2%
# =============================================================================

SSS = {
    "AL_TI_TA_MAX": 2.0,           # wt% threshold for SSS classification

    "YS_MIN_RT": 240,              # MPa
    "YS_MAX_RT": 500,              # MPa
    "GP_MAX": 5.0,                 # Maximum allowed γ' (%)

    "EM_MIN": 200.0,               # GPa
    "EM_MAX": 220.0,               # GPa
    "EM_TYPICAL": 212.0,           # GPa

    # Elongation (min lowered from 35→28 based on datasheet validation)
    "EL_MIN_WROUGHT": 28.0,        # %
    "EL_MAX_WROUGHT": 65.0,
    "EL_TYPICAL_WROUGHT": 52.0,
    "EL_MIN_CAST": 5.0,
    "EL_MAX_CAST": 20.0,
    "EL_TYPICAL_CAST": 10.0,

    # UTS/YS ratio — wrought (high work hardening)
    "UTS_YS_RATIO_MIN_WROUGHT": 1.6,
    "UTS_YS_RATIO_MAX_WROUGHT": 2.4,
    "UTS_YS_RATIO_TYPICAL_WROUGHT": 2.0,
    # UTS/YS ratio — cast
    "UTS_YS_RATIO_MIN_CAST": 1.3,
    "UTS_YS_RATIO_MAX_CAST": 1.7,
    "UTS_YS_RATIO_TYPICAL_CAST": 1.5,

    # SSS potency factors (MPa/wt%) — Labusch-Nabarro model
    "POTENCY": {
        "Re": 18.0, "W": 12.0, "Mo": 10.0, "Hf": 10.0, "Nb": 8.0, "Ta": 7.0,
        "Ti": 6.0, "Cr": 3.0, "Al": 1.5, "Fe": 1.5, "Co": 2.0,
        "V": 4.0, "Mn": 0.5, "Si": 0.3,
    },

    "SIGMA_BASE": 120,             # Base strength (MPa)
    "SIGMA_HP_WROUGHT": 40,        # Hall-Petch for wrought (MPa)
    "SIGMA_HP_CAST": 20,           # Hall-Petch for cast (MPa)
    "CAST_REDUCTION": 0.68,        # ~32% reduction for cast

    "TEMP_TRANSITION": 600.0,      # °C — decay accelerates above this
    "TEMP_DECAY_SLOW": 0.00055,    # Linear decay rate below transition
    "TEMP_DECAY_TAU": 450.0,       # Exponential decay constant 600–900°C
    "TEMP_TRANSITION_2": 900.0,    # °C — second acceleration above this
    "TEMP_DECAY_TAU2": 150.0,      # Exponential decay constant >900°C
    "TEMP_MIN_FACTOR": 0.12,
    "EL_TEMP_TRANSITION": 500.0,   # °C
    "EL_TEMP_FACTOR": 0.0019,      # Elongation increase rate per °C
}

# =============================================================================
# GP (GAMMA PRIME) TEMPERATURE DEGRADATION PARAMETERS
# For polycrystalline γ' alloys (Al+Ti+Ta >= 2%)
# =============================================================================

GP_TEMP = {
    "AL_TI_TA_MIN": 2.0,           # wt% threshold for γ' classification

    # Three-stage degradation: linear → exp(TAU1) → exp(TAU2)
    "STAGE1_END": 750.0,           # °C (reference at 25% γ')
    "STAGE2_END": 900.0,           # °C (reference solvus)

    "DECAY_LINEAR": 0.00020,       # per °C (linear stage)
    "DECAY_TAU1": 450.0,           # γ' coarsening (stage1→stage2)
    "DECAY_TAU2": 66.0,            # γ' dissolution (>stage2)
    "MIN_FACTOR": 0.02,            # floor factor

    # γ' solvus estimation
    "SOLVUS_BASE": 700.0,          # °C
    "SOLVUS_GP_COEFF": 8.0,        # °C per vol% γ'
    "GP_REF": 25.0,                # reference γ' fraction

    "EL_TEMP_FACTOR": 0.0018,      # Elongation increase per °C above 650
}

# =============================================================================
# SC/DS (SINGLE CRYSTAL / DIRECTIONALLY SOLIDIFIED) PARAMETERS
# For advanced turbine blade alloys
# =============================================================================

SC_DS = {
    # Temperature degradation (SC/DS retain strength longer than polycrystalline)
    "TEMP_TRANSITION": 850.0,      # °C
    "TEMP_DECAY_TAU": 250.0,
    "TEMP_MIN_FACTOR": 0.35,
    "TEMP_DECAY_LINEAR": 0.00006,

    # Detection thresholds
    "RE_MIN": 2.0,                 # 2nd+ gen SC
    "TA_W_MIN": 10.0,              # Ta+W with Re
    "TA_ALONE_MIN": 10.0,          # 1st gen SC
    "TA_W_HIGH": 11.0,

    # UTS/YS ratio
    "UTS_YS_RATIO_MIN": 1.03,
    "UTS_YS_RATIO_MAX": 1.30,
    "UTS_YS_RATIO_EXPECTED": 1.12,

    "ELONGATION_MIN": 3.0,         # %
    "ELONGATION_MAX": 15.0,
    "ELONGATION_EXPECTED": 8.0,
}

# =============================================================================
# TCP PHASE STABILITY (Md THRESHOLDS)
# Morinaga d-electron theory. Single authoritative source.
# =============================================================================

TCP = {
    # Thresholds on Md_avg (bulk), not Md_gamma (matrix).
    # Morinaga (1984) TCP boundary at Md ≈ 0.985. Using Md_gamma causes false positives.
    "MD_CRITICAL": 0.985,     # σ/μ phases highly likely (Morinaga boundary)
    "MD_ELEVATED": 0.960,     # TCP possible under long exposure
    "MD_MODERATE": 0.940,     # Moderate concern, monitor

    # Md_gamma > 0.980 upgrades risk by one level (heavy partitioning edge case)
    "MD_GAMMA_BOOST": 0.980,

    # Design targets
    "MD_DESIGN_TARGET": 0.935,
    "MD_DESIGN_SAFE": 0.955,
}

# Numeric TCP risk ranking (lower = better). Used by designer and optimizer.
TCP_RANK = {"Low": 0, "Moderate": 1, "Elevated": 2, "Critical": 3}


def classify_tcp_risk(md_gamma: float, md_avg: float = 0.0) -> str:
    """Classify TCP risk. Primary: Md_avg (bulk). Secondary: Md_gamma > 0.980 upgrades by one level."""
    if md_avg <= 0:
        md_avg = md_gamma

    if md_avg > TCP["MD_CRITICAL"]:
        base_risk = "Critical"
    elif md_avg > TCP["MD_ELEVATED"]:
        base_risk = "Elevated"
    elif md_avg > TCP["MD_MODERATE"]:
        base_risk = "Moderate"
    else:
        base_risk = "Low"

    if md_gamma > TCP["MD_GAMMA_BOOST"] and base_risk != "Critical":
        risk_levels = ["Low", "Moderate", "Elevated", "Critical"]
        current_idx = risk_levels.index(base_risk)
        return risk_levels[current_idx + 1]

    return base_risk


# =============================================================================
# UTS/YS RATIO CONSTRAINTS BY PROCESSING AND CONDITION
# =============================================================================

UTS_YS_RATIO = {
    # Wrought GP alloys (WROUGHT_MIN lowered for γ" alloys)
    "WROUGHT_BASE": 1.40,
    "WROUGHT_MIN": 1.15,
    "WROUGHT_MAX": 1.60,
    "WROUGHT_HIGH_GP_MAX": 1.45,   # γ' > 40%
    "WROUGHT_HIGH_GP_MIN": 1.25,
    "WROUGHT_HIGH_GP_EXPECTED": 1.38,

    # Cast GP alloys
    "CAST_BASE": 1.15,
    "CAST_MIN": 1.08,
    "CAST_GP_FACTOR": 0.2,

    # Coherency warning thresholds
    "COHERENCY_MIN": 1.05,
    "COHERENCY_MAX": 1.60,
}

# =============================================================================
# ELONGATION CONSTRAINTS BY PROCESSING AND γ'
# =============================================================================

ELONGATION = {
    "HIGH_GP_MAX_EL": 18.0,             # γ' > 60% (SC/DS and wrought)
    "HIGH_GP_MAX_EL_CAST": 10.0,        # γ' > 60% (cast polycrystalline)
    "MOD_GP_MAX_EL": 25.0,              # γ' 40-60% (SC/DS and wrought)
    "MOD_GP_MAX_EL_CAST": 15.0,         # γ' 40-60% (cast polycrystalline)
}

# =============================================================================
# WROUGHT ALLOY PARAMETERS
# =============================================================================

WROUGHT = {
    # γ' strengthening: YS = BASE_NI + COEFF_GP * γ' + SSS + Hall-Petch
    "COEFF_GP": 28.0,
    "COEFF_GP_HIGH_STRENGTH": 33.0,
    "COEFF_GP_CORROSION": 18.0,

    "BASE_NI": 120.0,              # MPa
    "HALL_PETCH_BOOST": 50.0,      # MPa (wrought grain refinement)
    "SSS_CONTRIBUTION_FACTOR": 12.0,

    # ML/Physics blending weights
    "ML_WEIGHT_HIGH_CONF": 0.70,
    "ML_WEIGHT_MED_CONF": 0.60,
    "ML_WEIGHT_LOW_CONF": 0.50,

    "CAL_YS_FACTOR": 0.90,
    "CAL_UTS_FACTOR": 0.90,
    "CAL_EL_FACTOR": 1.0,

    # Ductility
    "BASE_DUCTILITY": 35.0,        # %
    "MIN_ELONGATION": 10.0,
}

# =============================================================================
# CAST ALLOY PARAMETERS
# =============================================================================

CAST = {
    "COEFF_GP": 10.0,
    "COEFF_GP_HIGH_STRENGTH": 14.0,
    "COEFF_GP_CORROSION": 7.0,

    "BASE_NI": 120.0,              # MPa
    "HALL_PETCH_BOOST": 0.0,       # No grain refinement for cast
    "SSS_CONTRIBUTION_FACTOR": 12.0,

    "ML_WEIGHT_HIGH_CONF": 0.70,
    "ML_WEIGHT_MED_CONF": 0.60,
    "ML_WEIGHT_LOW_CONF": 0.50,

    "CAL_YS_FACTOR": 0.95,
    "CAL_UTS_FACTOR": 0.95,
    "CAL_EL_FACTOR": 1.0,

    "BASE_DUCTILITY": 20.0,        # %
    "MIN_ELONGATION": 5.0,
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

# =============================================================================
# TEMPERATURE DEGRADATION FUNCTIONS
# Unified approach for SSS, GP, and SC/DS alloys
# =============================================================================

def get_temperature_factor(temp_c: float, alloy_class: str, gp_fraction: float = None) -> float:
    """
    Calculate temperature degradation factor for strength properties.

    Args:
        temp_c: Temperature in Celsius
        alloy_class: One of 'sss', 'gp', 'sc_ds'
        gp_fraction: Estimated γ' volume percent (only used for 'gp' class).
                     If None, defaults to 25% (Waspaloy calibration point).

    Returns:
        Factor to multiply room-temperature strength (0.0 to 1.0)
    """
    import math

    if temp_c <= 25:
        return 1.0

    if alloy_class == "sss":
        # SSS alloys: linear → exp(τ1) → exp(τ2) three-stage decay
        if temp_c <= SSS["TEMP_TRANSITION"]:
            factor = 1.0 - SSS["TEMP_DECAY_SLOW"] * (temp_c - 25)
        elif temp_c <= SSS["TEMP_TRANSITION_2"]:
            factor_at_trans = 1.0 - SSS["TEMP_DECAY_SLOW"] * (SSS["TEMP_TRANSITION"] - 25)
            delta_t = temp_c - SSS["TEMP_TRANSITION"]
            factor = factor_at_trans * math.exp(-delta_t / SSS["TEMP_DECAY_TAU"])
        else:
            factor_at_trans = 1.0 - SSS["TEMP_DECAY_SLOW"] * (SSS["TEMP_TRANSITION"] - 25)
            factor_at_trans2 = factor_at_trans * math.exp(
                -(SSS["TEMP_TRANSITION_2"] - SSS["TEMP_TRANSITION"]) / SSS["TEMP_DECAY_TAU"]
            )
            delta_t = temp_c - SSS["TEMP_TRANSITION_2"]
            factor = factor_at_trans2 * math.exp(-delta_t / SSS["TEMP_DECAY_TAU2"])
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
        gp = GP_TEMP["GP_REF"]
        gp = max(2.0, min(70.0, gp))
        gp_ref = GP_TEMP["GP_REF"]

        solvus = GP_TEMP["SOLVUS_BASE"] + GP_TEMP["SOLVUS_GP_COEFF"] * gp
        stage1_end = max(500, min(850, GP_TEMP["STAGE1_END"] * min(1.0, math.sqrt(gp / gp_ref))))
        stage2_end = solvus
        tau2 = GP_TEMP["DECAY_TAU2"] * max(0.5, min(3.0, gp / gp_ref))

        if temp_c <= stage1_end:
            factor = 1.0 - GP_TEMP["DECAY_LINEAR"] * (temp_c - 25)
        elif temp_c <= stage2_end:
            factor_s1 = 1.0 - GP_TEMP["DECAY_LINEAR"] * (stage1_end - 25)
            delta_t = temp_c - stage1_end
            factor = factor_s1 * math.exp(-delta_t / GP_TEMP["DECAY_TAU1"])
        else:
            factor_s1 = 1.0 - GP_TEMP["DECAY_LINEAR"] * (stage1_end - 25)
            factor_s2 = factor_s1 * math.exp(-(stage2_end - stage1_end) / GP_TEMP["DECAY_TAU1"])
            delta_t = temp_c - stage2_end
            factor = factor_s2 * math.exp(-delta_t / tau2)

        # SSS residual floor above solvus
        sss_frac = max(0.3, 1.0 - gp / 100.0)
        if temp_c <= SSS["TEMP_TRANSITION"]:
            sss_f = 1.0 - SSS["TEMP_DECAY_SLOW"] * (temp_c - 25)
        elif temp_c <= SSS["TEMP_TRANSITION_2"]:
            f_trans = 1.0 - SSS["TEMP_DECAY_SLOW"] * (SSS["TEMP_TRANSITION"] - 25)
            sss_f = f_trans * math.exp(-(temp_c - SSS["TEMP_TRANSITION"]) / SSS["TEMP_DECAY_TAU"])
        else:
            f_trans = 1.0 - SSS["TEMP_DECAY_SLOW"] * (SSS["TEMP_TRANSITION"] - 25)
            f_trans2 = f_trans * math.exp(
                -(SSS["TEMP_TRANSITION_2"] - SSS["TEMP_TRANSITION"]) / SSS["TEMP_DECAY_TAU"]
            )
            sss_f = f_trans2 * math.exp(-(temp_c - SSS["TEMP_TRANSITION_2"]) / SSS["TEMP_DECAY_TAU2"])
        sss_floor = sss_frac * max(sss_f, SSS["TEMP_MIN_FACTOR"])

        return max(factor, sss_floor, GP_TEMP["MIN_FACTOR"])


def is_sss_alloy(composition: dict) -> bool:
    """
    Check if composition is an SSS alloy.

    SSS alloys lack significant precipitation hardening phases.

    Threshold: Al + Ti + Ta + 0.35*Nb < 2%
    """
    al = composition.get("Al", composition.get("al", 0)) or 0
    ti = composition.get("Ti", composition.get("ti", 0)) or 0
    ta = composition.get("Ta", composition.get("ta", 0)) or 0
    nb = composition.get("Nb", composition.get("nb", 0)) or 0

    precipitate_formers = al + ti + ta + (0.35 * nb)

    return precipitate_formers < SSS["AL_TI_TA_MAX"]


def is_sc_ds_alloy(composition: dict, processing: str = "") -> tuple:
    """
    Detect if alloy is Single Crystal (SC) or Directionally Solidified (DS).

    Returns:
        tuple: (is_sc_ds: bool, reason: str)
    """
    # Wrought/forged alloys cannot be SC/DS regardless of composition
    if processing in ("wrought", "forged"):
        return False, ""

    re = composition.get("Re", composition.get("re", 0)) or 0
    ru = composition.get("Ru", composition.get("ru", 0)) or 0
    ta = composition.get("Ta", composition.get("ta", 0)) or 0
    w = composition.get("W", composition.get("w", 0)) or 0
    c = composition.get("C", composition.get("c", 0)) or 0

    if re >= SC_DS["RE_MIN"]:
        return True, f"Re={re:.1f}% (2nd+ gen SC indicator)"
    if ru >= 1.0 and re >= 1.0:
        return True, f"Ru={ru:.1f}%, Re={re:.1f}% (4th gen SC indicator)"
    if (ta + w) >= SC_DS["TA_W_MIN"] and re >= 1.0:
        return True, f"Ta+W={ta+w:.1f}%, Re={re:.1f}% (SC/DS composition)"

    # C >= 0.06% indicates cast polycrystalline (SC/DS alloys have near-zero C)
    if c >= 0.06:
        return False, ""
    if (ta + w) >= SC_DS["TA_W_HIGH"] and ta >= 5.0:
        return True, f"Ta+W={ta+w:.1f}%, Ta={ta:.1f}% (1st gen SC composition)"
    if ta >= SC_DS["TA_ALONE_MIN"]:
        return True, f"Ta={ta:.1f}% (1st gen SC indicator)"
    if (ta + w) >= 12.0 and w >= 8.0:  # W-rich DS
        return True, f"Ta+W={ta+w:.1f}%, W={w:.1f}% (W-rich DS alloy)"

    return False, ""


def get_alloy_class(composition: dict, processing: str = "") -> str:
    """
    Determine alloy class based on composition and processing route.

    Returns:
        One of: 'sss', 'sc_ds', 'gp'
    """
    if is_sss_alloy(composition):
        return "sss"
    is_sc, _ = is_sc_ds_alloy(composition, processing)
    if is_sc:
        return "sc_ds"
    return "gp"


# =============================================================================
# ELASTIC MODULUS TEMPERATURE DECAY
# =============================================================================

EM_TEMP_DECAY_RATE = 0.00032  # ~0.032% per °C above RT
EM_TEMP_RT_BASELINE = 20      # °C


def get_em_temp_factor(temperature_c: float) -> float:
    """Calculate EM temperature reduction factor (multiply RT value)."""
    delta_t = max(0, temperature_c - EM_TEMP_RT_BASELINE)
    return max(0.50, 1.0 - EM_TEMP_DECAY_RATE * delta_t)


def compress_uts_ys_ratio(rt_ratio: float, temperature_c: float) -> float:
    """Two-stage UTS/YS ratio compression for elevated temperatures"""
    import math
    if temperature_c < 650:
        return rt_ratio
    if temperature_c <= 800:
        t_excess = temperature_c - 650
        return 1.0 + (rt_ratio - 1.0) * max(0.2, 1.0 - 0.003 * t_excess)
    # >800C: compute ratio at 800 then exponential decay
    ratio_at_800 = 1.0 + (rt_ratio - 1.0) * max(0.2, 1.0 - 0.003 * 150)
    t_excess_800 = temperature_c - 800
    return 1.0 + (ratio_at_800 - 1.0) * math.exp(-t_excess_800 / 50)


# =============================================================================
# CORRECTION THRESHOLDS — minimum change to consider a correction meaningful
# =============================================================================

CORRECTION_THRESHOLDS = {
    "Yield Strength": 5.0,       # MPa
    "Tensile Strength": 5.0,     # MPa
    "Elongation": 0.5,           # %
    "Elastic Modulus": 1.0,      # GPa
    "Density": 0.05,             # g/cm³
    "Gamma Prime": 0.5,          # vol%
}

AGENT_TRUST = {
    "NOOP_THRESHOLD": 0.05,         # Correction within 5% of ML baseline → treated as no-op
    "MIN_REASON_LENGTH": 20,        # Characters — shorter reasons are placeholders
    "UNDOCUMENTED_DEVIATION": 0.15, # 15% deviation from ML without documentation triggers review
    "PLACEHOLDER_STRINGS": [        # Known placeholder reasons from LLM output
        "Correction reason",
        "correction applied",
    ],
}


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
