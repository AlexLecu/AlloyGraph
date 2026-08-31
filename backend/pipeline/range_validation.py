"""Check computed variant features against the ontology's own range restrictions.

``build_ontology.py`` declares physical-plausibility bounds as OWL
``ConstrainedDatatype`` restrictions on ``Variant``: density 7.0-10.0 g/cm3,
gamma-prime 0-85 vol%, average Md 0.70-1.05, lattice mismatch -2.0 to 2.0%.
HermiT checks them when the schema is built, but nothing checked the *data*:
ingestion computed these features and wrote them to the triplestore without ever
comparing them to the bounds the ontology declares for them.

This closes that gap. The bounds are read from one table here and are the same
numbers ``build_ontology`` writes into the OWL, so the two cannot drift without
the test below failing.

WHAT IT CATCHES, MEASURED RATHER THAN ASSERTED. Run over the uncorrected
upstream corpus it flags three records, each an independently documented
erratum: CMSX-2 and CMSX-3 (Md 1.117, from the nickel content that leaves the
composition at 67 wt% -- ledger T01/T02) and NIMONIC PE16 (Md 1.060, from the
transposed titanium -- ledger E01). Run over the corrected training set and all
three evaluation files it flags nothing.

WHAT IT CANNOT CATCH. Erratum E02, the -435 MPa yield strength on RGT* 13, is a
*measurement*. The ontology declares no range on measured strengths, so no
amount of wiring reaches it; that defect is caught by
``evaluation/prediction/scripts/data_sanity_sweep.py`` instead. A range check
over the declared restrictions and a physical-plausibility sweep over the
measurements are two different instruments, and only the second sees E02.

Violations are logged, not raised. The corrected data produces none, so a hard
gate would reject nothing today while risking a pipeline that refuses to load on
a future record a human should look at first.
"""

import logging
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)

#: feature key -> (min inclusive, max inclusive, the ontology property it mirrors)
#: These are the same numbers build_ontology.py writes as ConstrainedDatatype
#: restrictions on Variant. test_range_validation asserts the two agree.
FEATURE_RANGES: Dict[str, Tuple[float, float, str]] = {
    "density_calculated_gcm3": (7.0, 10.0, "hasDensityCalculated"),
    "gamma_prime_estimated_vol_pct": (0.0, 85.0, "hasGammaPrimeEstimate"),
    "Md_avg": (0.70, 1.05, "hasMdAverage"),
    "lattice_mismatch_pct": (-2.0, 2.0, "hasLatticeMismatchPct"),
}


def check_features(alloy: str, features: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Violations of the declared ranges for one variant. Empty when clean."""
    out = []
    for key, (lo, hi, prop) in FEATURE_RANGES.items():
        value = features.get(key)
        if value is None or not isinstance(value, (int, float)):
            continue
        if lo <= float(value) <= hi:
            continue
        out.append({"alloy": alloy, "feature": key, "value": float(value),
                    "min": lo, "max": hi, "ontology_property": prop})
    return out


def log_violations(violations: List[Dict[str, Any]]) -> int:
    """Log each violation at WARNING. Returns the count, for the caller's summary."""
    for v in violations:
        logger.warning(
            "RANGE VIOLATION [%s] %s = %.4f outside [%s, %s] declared by %s. "
            "The record is ingested anyway; check it against its source.",
            v["alloy"], v["feature"], v["value"], v["min"], v["max"],
            v["ontology_property"])
    return len(violations)
