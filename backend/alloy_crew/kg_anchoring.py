"""Knowledge-graph anchoring primitives.

The sigmoid weight and the gating rules that decide whether a knowledge-graph
neighbour may calibrate an ML prediction. ``AlloyAnalysisTool`` uses these when
it builds the calibration proposals the Analyst agent receives; the
``--ml-physics-kg`` ablation uses the same functions to accept those proposals
deterministically, with no LLM in the loop.

Distances here are Euclidean (L2) on wt%-normalised compositions, as produced by
``rag_tools._composition_distance`` -- not cosine distances, and not bounded
above by 2.
"""

import math
from typing import Optional, Tuple

from .config.alloy_parameters import (
    KG_ANCHOR_MAX_DISTANCE,
    KG_ANCHOR_MAX_DISTANCE_INCOMPATIBLE,
    KG_ANCHOR_MAX_GP_DIFF,
    KG_SIGMOID_MIDPOINT,
    KG_SIGMOID_SLOPE,
)

#: Properties a knowledge-graph neighbour is allowed to calibrate.
ANCHORABLE_PROPERTIES = ("Yield Strength", "Tensile Strength", "Elongation")

#: Source tag carried by proposals produced through this path.
KG_ANCHOR_SOURCE = "KG_anchoring"


def kg_anchor_weight(distance: float) -> float:
    """Weight given to the knowledge-graph value: 1 / (1 + exp((d - m) / s)).

    Decays fast: d=2.0 -> 0.73, d=2.5 -> 0.50, d=3.0 -> 0.27, d=4.5 -> 0.02.
    """
    return 1.0 / (1.0 + math.exp((distance - KG_SIGMOID_MIDPOINT) / KG_SIGMOID_SLOPE))


def processing_compatible(query_processing: str, kg_processing: str) -> bool:
    """True when the two processing routes are the same family.

    Substring matching in both directions, so "wrought" matches "wrought bar".
    An empty or unknown route on either side counts as compatible: absence of
    evidence is not evidence of a mismatch.
    """
    q = (query_processing or "").lower()
    k = (kg_processing or "").lower()
    if not q or not k or k == "unknown":
        return True
    return q in k or k in q


def max_anchor_distance(proc_compatible: bool) -> float:
    """Hard cutoff beyond which no anchoring happens at all."""
    return KG_ANCHOR_MAX_DISTANCE if proc_compatible else KG_ANCHOR_MAX_DISTANCE_INCOMPATIBLE


def anchoring_allowed(
    distance: float,
    query_gamma_prime: float,
    kg_gamma_prime: Optional[float],
    query_processing: str = "",
    kg_processing: str = "",
) -> Tuple[bool, str]:
    """Decide whether this neighbour may calibrate the query.

    Returns ``(allowed, reason)``; ``reason`` explains the refusal and is empty
    when allowed. Mirrors the guards in ``AlloyAnalysisTool._generate_proposals``:
    a distance gate that tightens when the processing route differs, a
    gamma-prime class check, and an outright processing-route mismatch check.
    """
    compatible = processing_compatible(query_processing, kg_processing)

    limit = max_anchor_distance(compatible)
    if not (distance < limit):
        return False, f"distance {distance:.2f} >= cutoff {limit:.2f}"

    if not compatible:
        return False, (f"processing mismatch: query '{query_processing}' "
                       f"vs KG '{kg_processing}'")

    if kg_gamma_prime is not None:
        gp_diff = abs(query_gamma_prime - kg_gamma_prime)
        if gp_diff > KG_ANCHOR_MAX_GP_DIFF:
            return False, (f"gamma-prime class mismatch: query {query_gamma_prime:.1f}% "
                           f"vs KG {kg_gamma_prime:.1f}% (diff {gp_diff:.1f} > "
                           f"{KG_ANCHOR_MAX_GP_DIFF})")

    return True, ""


def blend(ml_value: float, kg_value: float, distance: float) -> float:
    """Sigmoid-weighted blend of an ML prediction and a KG experimental value."""
    w = kg_anchor_weight(distance)
    return ml_value * (1.0 - w) + kg_value * w
