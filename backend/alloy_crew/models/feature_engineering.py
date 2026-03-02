from typing import Dict, Any

from ..config.alloy_parameters import classify_tcp_risk

# Physical constants
ATOMIC_WEIGHTS = {
    "Ni": 58.69, "Co": 58.93, "Cr": 52.00, "Mo": 95.95, "W": 183.84,
    "Al": 26.98, "Ti": 47.87, "Ta": 180.95, "Nb": 92.91, "Re": 186.21,
    "Fe": 55.85, "C": 12.01, "B": 10.81, "Hf": 178.49, "Zr": 91.22,
    "Ru": 101.07, "V": 50.94, "Mn": 54.94, "Si": 28.09, "Cu": 63.55,
    "N": 14.01, "La": 138.91, "Y": 88.91
}

# Morinaga d-electron parameter values (Md)
MD_VALUES = {
    "Ni": 0.717, "Co": 0.777, "Cr": 1.142, "Mo": 1.55, "W": 1.655,
    "Al": 1.9, "Ti": 2.271, "Ta": 2.224, "Nb": 2.117, "Re": 1.267,
    "Fe": 0.858, "Hf": 3.0, "Zr": 2.9, "Ru": 1.006, "V": 1.543
}

# Element densities (g/cm³)
ELEMENT_DENSITIES = {
    "Ni": 8.90, "Co": 8.90, "Cr": 7.19, "Mo": 10.28, "W": 19.25,
    "Al": 2.70, "Ti": 4.50, "Ta": 16.69, "Nb": 8.57, "Re": 21.02,
    "Fe": 7.87, "Hf": 13.31, "Zr": 6.51, "Ru": 12.37, "V": 6.11,
    "Mn": 7.21, "Si": 2.33, "Cu": 8.96, "C": 2.26, "B": 2.34
}

# Elemental elastic moduli (GPa) for rule-of-mixtures calculation
ELEMENTAL_MODULI = {
    "Ni": 200.0, "Cr": 279.0, "Co": 209.0, "Al": 70.0,
    "Ti": 116.0, "Mo": 329.0, "W": 411.0, "Fe": 211.0,
    "Ta": 186.0, "Re": 463.0, "Nb": 105.0, "Hf": 78.0,
    "Mn": 198.0, "Si": 130.0,
}


def calculate_em_rule_of_mixtures(composition: Dict[str, float]) -> float:
    """Calculate Elastic Modulus using Voigt-Reuss-Hill (VRH) average."""
    total_wt = sum(composition.get(el, 0) for el in ELEMENTAL_MODULI if composition.get(el, 0) > 0)
    if total_wt <= 0:
        return 200.0  # Fallback to pure Ni

    # Reuss bound (harmonic mean — lower bound)
    inv_sum = sum(
        (composition.get(element, 0) / total_wt) / modulus
        for element, modulus in ELEMENTAL_MODULI.items()
        if composition.get(element, 0) > 0
    )
    if inv_sum <= 0:
        return 200.0
    reuss = 1.0 / inv_sum

    # Voigt bound (arithmetic mean — upper bound)
    voigt = sum(
        (composition.get(element, 0) / total_wt) * modulus
        for element, modulus in ELEMENTAL_MODULI.items()
        if composition.get(element, 0) > 0
    )

    # Hill average (midpoint)
    vrh = (voigt + reuss) / 2.0
    return round(vrh, 1)


def wt_to_at_percent(composition: Dict[str, float]) -> Dict[str, float]:
    """Convert weight percent to atomic percent."""
    moles = {}
    for el, wt in composition.items():
        if el in ATOMIC_WEIGHTS and wt > 0:
            moles[el] = wt / ATOMIC_WEIGHTS[el]

    total_moles = sum(moles.values())
    if total_moles == 0:
        return {el: 0.0 for el in ATOMIC_WEIGHTS}

    return {el: round(m / total_moles * 100, 2) for el, m in moles.items()}


def calculate_md_avg(at_percent: Dict[str, float]) -> float:
    """Calculate average Md (Morinaga d-electron) parameter."""
    md_sum = sum(
        (at_pct / 100) * MD_VALUES[el]
        for el, at_pct in at_percent.items()
        if el in MD_VALUES
    )
    return round(md_sum, 4)


def calculate_density(composition: Dict[str, float]) -> float:
    """Calculate alloy density using inverse rule of mixtures"""
    total_wt = sum(composition.values())
    if total_wt == 0:
        return 8.0
    inv_sum = sum(
        (wt / total_wt) / ELEMENT_DENSITIES.get(el, 8.0)
        for el, wt in composition.items()
        if ELEMENT_DENSITIES.get(el, 8.0) > 0
    )
    if inv_sum <= 0:
        return 8.0
    return round(1.0 / inv_sum, 3)


def estimate_gamma_prime_vol_pct(at_percent: Dict[str, float]) -> float:
    """
    Estimate gamma prime (γ') volume fraction using solubility-based model
    with Ni-stoichiometry constraint (γ' = Ni₃X).

    Calibrated against: CMSX-4 (70%), René 80 (50%), IN718 (15-20%)
    """
    al = at_percent.get("Al", 0)
    ti = at_percent.get("Ti", 0)
    ta = at_percent.get("Ta", 0)
    nb = at_percent.get("Nb", 0)
    v = at_percent.get("V", 0)
    ni = at_percent.get("Ni", 0)
    co = at_percent.get("Co", 0)
    cr = at_percent.get("Cr", 0)
    fe = at_percent.get("Fe", 0)

    # Weighted γ' formers (calibrated to CALPHAD partitioning)
    effective_formers = 0.75*al + 1.20*ti + 1.30*ta + 1.05*nb + 0.5*v

    # Matrix solubility limit: higher value → more formers dissolve → LESS γ'
    # +Co: expands solubility (slight γ' suppression at high Co)
    # -Fe: lowers solubility → more γ' (calibrated for IN718; see note below)
    # +Cr: γ-matrix stabilizer, raises solubility → less γ' (physically correct)
    solubility_limit = 2.0 + 0.04*co - 0.05*fe + 0.03*cr

    excess_formers = max(0, effective_formers - solubility_limit)

    # Ni-stoichiometry constraint
    ni_limited = ni / 3.6
    effective_excess = min(excess_formers, ni_limited)

    return round(max(0.0, min(85.0, effective_excess * 6.0)), 1)


# === Phase Partitioning & Lattice Parameters ===

# Partitioning coefficients k = C_γ'/C_γ (Reed, Pollock)
PARTITION_COEFFS = {
    "Al": 4.0, "Ti": 6.0, "Ta": 4.0, "Nb": 3.0, "Hf": 2.0, "Ni": 1.1,
    "Co": 0.6, "Cr": 0.1, "Mo": 0.3, "W": 0.4, "Re": 0.1, "Ru": 0.4, "Fe": 0.3, "V": 0.3
}

def estimate_partitioning(at_percent: Dict[str, float], gamma_prime_vol_frac: float) -> tuple[Dict[str, float], Dict[str, float]]:
    """
    Estimate γ and γ' phase compositions using mass balance.
    C_γ = C_alloy / (f*k + 1-f), where k = C_γ'/C_γ
    """
    f = max(0.01, min(0.90, gamma_prime_vol_frac / 100.0))
    c_gamma, c_gamma_prime = {}, {}

    for el, c_alloy in at_percent.items():
        k = PARTITION_COEFFS.get(el, 1.0)
        denom = (f * k) + (1.0 - f) or 1.0
        c_gamma[el] = c_alloy / denom
        c_gamma_prime[el] = c_gamma[el] * k

    # Normalize phase compositions to sum to 100%
    total_gamma = sum(c_gamma.values())
    total_gp = sum(c_gamma_prime.values())
    if total_gamma > 0:
        c_gamma = {el: v / total_gamma * 100 for el, v in c_gamma.items()}
    if total_gp > 0:
        c_gamma_prime = {el: v / total_gp * 100 for el, v in c_gamma_prime.items()}

    return c_gamma, c_gamma_prime


# Lattice parameter coefficients (Å per at% from pure Ni, a₀=3.524Å)
LATTICE_COEFFS = {
    "Al": 0.179, "Ti": 0.422, "Cr": 0.113, "Mo": 0.467, "W": 0.575,
    "Ta": 0.670, "Nb": 0.700, "Re": 0.528, "Co": 0.010, "Fe": 0.050,
    "Hf": 0.850, "V": 0.150, "C": 0.0, "B": 0.0, "Zr": 0.9
}

def calculate_lattice_parameter(composition_at: Dict[str, float]) -> float:
    """Calculate FCC lattice parameter: a = a_Ni + Σ(Xi × ki)"""
    return 3.524 + sum(
        (amt / 100.0) * LATTICE_COEFFS.get(el, 0.1)
        for el, amt in composition_at.items()
    )

def calculate_lattice_mismatch(c_gamma: Dict[str, float], c_gamma_prime: Dict[str, float]) -> float:
    """Calculate lattice mismatch δ = 2(a_γ' - a_γ)/(a_γ' + a_γ) in percent."""
    a_g = calculate_lattice_parameter(c_gamma)
    a_gp = calculate_lattice_parameter(c_gamma_prime)
    if a_g + a_gp == 0:
        return 0.0
    return round(200.0 * (a_gp - a_g) / (a_gp + a_g), 3)


VEC_VALUES = {
    "Ni": 10, "Co": 9, "Fe": 8, "Cr": 6, "Mo": 6, "W": 6,
    "Al": 3, "Ti": 4, "Nb": 5, "Ta": 5, "Re": 7, "Ru": 8,
    "V": 5, "Hf": 4, "Zr": 4, "C": 4, "B": 3
}

def calculate_vec(composition_at: Dict[str, float]) -> float:
    """Calculate Valence Electron Concentration."""
    return round(sum(
        (amt / 100.0) * VEC_VALUES.get(el, 0)
        for el, amt in composition_at.items()
    ), 3)

# === Strengthening Mechanism Coefficients ===

# Metallic radii (Å)
ATOMIC_RADII = {
    "Ni": 1.246, "Co": 1.251, "Fe": 1.241, "Cr": 1.249,
    "Mo": 1.363, "W": 1.370, "Re": 1.375,
    "Al": 1.432, "Ti": 1.462, "Ta": 1.430, "Nb": 1.429, "Hf": 1.564, "V": 1.316
}

# SSS potency (Re > W > Mo > V > Cr > Fe > Co)
SSS_COEFFS = {"Re": 3.0, "W": 2.2, "Mo": 1.0, "V": 0.8, "Cr": 0.3, "Fe": 0.2, "Co": 0.1}

def calculate_solid_solution_strengthening_coeff(c_gamma: Dict[str, float]) -> float:
    """
    Calculate SSS coefficient using Labusch-Nabarro model.
    Uses γ matrix composition since SSS occurs in the matrix.
    """
    r_ni = ATOMIC_RADII["Ni"]
    sss_coeff = 0.0

    for el, at_pct in c_gamma.items():
        if el in SSS_COEFFS and at_pct > 0:
            r_el = ATOMIC_RADII.get(el, r_ni)
            delta_r = abs(r_el - r_ni) / r_ni
            sss_coeff += ((at_pct / 100.0) ** 0.67) * (delta_r ** 1.3) * SSS_COEFFS[el]

    return round(sss_coeff, 4)

def calculate_precipitation_hardening_coeff(at_percent: Dict[str, float], gp_vol_pct: float) -> float:
    """Calculate precipitation hardening: σ_ppt ∝ f^0.5 × (Al + Ti)"""
    al_at = at_percent.get("Al", 0)
    ti_at = at_percent.get("Ti", 0)
    f_factor = (gp_vol_pct / 100.0) ** 0.5
    former_factor = (al_at + ti_at) / 100.0
    return round(f_factor * former_factor * 10.0, 4)

# === High-Temperature Performance ===

def calculate_creep_resistance_param(at_percent: Dict[str, float]) -> float:
    """Calculate creep resistance parameter (Re > Ru > W)."""
    return round(
        2.0 * at_percent.get("Re", 0) +
        1.5 * at_percent.get("Ru", 0) +
        1.0 * at_percent.get("W", 0),
        3
    )

def calculate_oxidation_resistance(composition: Dict[str, float]) -> float:
    """Calculate oxidation resistance from Cr and Al oxide formers."""
    cr_wt = composition.get("Cr", 0)
    al_wt = composition.get("Al", 0)
    total = cr_wt + al_wt
    if total < 1.0:
        return 0.0
    cr_ratio = cr_wt / total
    oxide_formers = min(cr_wt, 15.0) + min(al_wt, 8.0)
    return round(cr_ratio * oxide_formers, 3)


# === Main Entry Point ===

def compute_alloy_features(alloy_or_comp: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute all features for an alloy from composition.
    Accepts either {"Ni": 60, "Al": 5} or {"composition": {"Ni": 60...}}
    """
    if "composition" in alloy_or_comp and isinstance(alloy_or_comp["composition"], dict):
        raw_comp = alloy_or_comp["composition"]
    else:
        raw_comp = alloy_or_comp

    composition = {str(k): float(v) for k, v in raw_comp.items() if isinstance(v, (int, float))}
    at_percent = wt_to_at_percent(composition)

    # Phase calculations
    gp_vol_pct = estimate_gamma_prime_vol_pct(at_percent)
    c_gamma, c_gamma_prime = estimate_partitioning(at_percent, gp_vol_pct)

    # Stability metrics
    md_avg_global = calculate_md_avg(at_percent)
    md_gamma_matrix = calculate_md_avg(c_gamma)
    lattice_mismatch = calculate_lattice_mismatch(c_gamma, c_gamma_prime)
    vec = calculate_vec(at_percent)

    # TCP risk
    tcp_risk = classify_tcp_risk(md_gamma_matrix, md_avg_global)

    # Strengthening mechanisms
    sss_coeff = calculate_solid_solution_strengthening_coeff(c_gamma)
    ppt_coeff = calculate_precipitation_hardening_coeff(at_percent, gp_vol_pct)

    return {
        "atomic_percent": at_percent,
        "gamma_composition_at": {k: round(v, 2) for k, v in c_gamma.items() if v > 0.01},
        "gamma_prime_composition_at": {k: round(v, 2) for k, v in c_gamma_prime.items() if v > 0.01},
        "Md_avg": md_avg_global,
        "Md_gamma": md_gamma_matrix,
        "VEC_avg": vec,
        "TCP_risk": tcp_risk,
        "gamma_prime_estimated_vol_pct": gp_vol_pct,
        "lattice_mismatch_pct": lattice_mismatch,
        "density_calculated_gcm3": calculate_density(composition),
        "SSS_coefficient": sss_coeff,
        "precipitation_hardening_coeff": ppt_coeff,
        "creep_resistance_param": calculate_creep_resistance_param(at_percent),
        "SSS_total_wt_pct": round(sum(composition.get(el, 0) for el in ["Mo", "W", "Nb", "Ta", "Re"]), 2),
        "oxidation_resistance": calculate_oxidation_resistance(composition),
        "refractory_total_wt_pct": round(sum(composition.get(el, 0) for el in ["Mo", "W", "Ta", "Re", "Nb", "Hf"]), 2),
        "GP_formers_wt_pct": round(sum(composition.get(el, 0) for el in ["Al", "Ti", "Ta", "Nb"]), 2),
        "Al_Ti_ratio": round(composition.get("Al", 0) / (composition.get("Ti", 0) + 0.01), 2),
        "Cr_Co_ratio": round(composition.get("Cr", 0) / (composition.get("Co", 0) + 0.01), 2),
        "Cr_Ni_ratio": round(composition.get("Cr", 0) / (composition.get("Ni", 0) + 0.01), 3),
        "Mo_W_ratio": round(composition.get("Mo", 0) / (composition.get("W", 0) + 0.01), 2),
        "Al_Ti_at_ratio": round(at_percent.get("Al", 0) / (at_percent.get("Ti", 0) + 0.01), 2),
        "GP_formers_at_pct": round(sum(at_percent.get(el, 0) for el in ["Al", "Ti", "Ta", "Nb"]), 2),
        "Al_Ti_interaction": round(at_percent.get("Al", 0) * at_percent.get("Ti", 0), 2),
        "Cr_Al_interaction": round(composition.get("Cr", 0) * composition.get("Al", 0), 2),
    }
