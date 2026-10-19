"""Conformal prediction intervals for delivered property values.

The interval the platform reports around each predicted property is the 90%
cross-conformal (CV+) interval of the ML ensemble, calibrated on the 77
training alloys with alloy-grouped folds
(evaluation/prediction/scripts/conformal_intervals.py). CV+ widths vary by
under 5% across the evaluation set, so one pair of offsets per property is
stored in saved_models_v2/conformal_offsets.json and applied around the value
the platform delivers.

What the interval means. The CV+ marginal coverage guarantee holds at the
ensemble's own prediction. The platform delivers the value after agent
reasoning and deterministic post-processing, so around that value the
interval is an empirical approximation: measured coverage at the delivered
points is 91.2% (YS), 90.0% (UTS), 85.6% (EL) and 97.7% (EM) on the
88-alloy evaluation set (evaluation/prediction/results/conformal_delivered.md).
"""

import json
import logging
import os
from functools import lru_cache
from typing import Dict, Optional

logger = logging.getLogger(__name__)

OFFSETS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "models", "saved_models_v2", "conformal_offsets.json")

METHOD_LABEL = "conformal_cv_plus_90"


@lru_cache(maxsize=1)
def load_offsets(path: str = OFFSETS_PATH) -> Optional[dict]:
    """The calibration file, or None if it is missing or unreadable."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Conformal offsets unavailable (%s); intervals fall back to "
                       "fixed-percentage bands, which carry no coverage guarantee.", exc)
        return None


def conformal_interval(prop: str, value: float) -> Optional[Dict[str, float]]:
    """90% conformal interval around a delivered value, clipped to the physical domain.

    Returns None for a property without calibration (density, gamma prime), so
    the caller can decide what, if anything, to report for it.
    """
    offsets = load_offsets()
    if not offsets or value is None:
        return None
    entry = offsets.get("properties", {}).get(prop)
    if entry is None:
        return None
    lower = value + entry["lower_offset"]
    upper = value + entry["upper_offset"]
    if entry.get("physical_min") is not None:
        lower = max(lower, entry["physical_min"])
    if entry.get("physical_max") is not None:
        upper = min(upper, entry["physical_max"])
    return {
        "lower": round(lower, 1),
        "upper": round(upper, 1),
        "uncertainty": round((entry["upper_offset"] - entry["lower_offset"]) / 2, 1),
        "method": METHOD_LABEL,
        "nominal_coverage": offsets.get("nominal_coverage", 0.9),
    }
