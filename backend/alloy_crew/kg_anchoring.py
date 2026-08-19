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
    """True when both routes are known and name the same family.

    Substring matching in both directions, so "wrought" matches "wrought bar".

    An empty route on either side is NOT treated as compatible. That is
    deliberate and matches ``AlloyAnalysisTool``: an unknown route earns the
    tighter distance cutoff, because there is no evidence the neighbour was
    made the same way. It is still not grounds for outright rejection -- see
    ``processing_mismatch``.
    """
    q = (query_processing or "").lower()
    k = (kg_processing or "").lower()
    return bool(q and k and (q in k or k in q))


def processing_mismatch(query_processing: str, kg_processing: str) -> bool:
    """True only when both routes are known and positively disagree.

    A missing route, or the literal "unknown", is not a mismatch: it downgrades
    the distance cutoff but never rejects on its own.
    """
    q = (query_processing or "").lower()
    k = (kg_processing or "").lower()
    if not (q and k and k != "unknown"):
        return False
    return not (q in k or k in q)


def max_anchor_distance(proc_compatible: bool) -> float:
    """Hard cutoff beyond which no anchoring happens at all."""
    return KG_ANCHOR_MAX_DISTANCE if proc_compatible else KG_ANCHOR_MAX_DISTANCE_INCOMPATIBLE


#: Machine-readable rejection reasons, for ablation accounting.
REJECT_DISTANCE = "distance"
REJECT_PROCESSING = "processing_route"
REJECT_GP_CLASS = "gamma_prime_class"
REJECT_NO_MATCH = "no_kg_match"


def anchoring_allowed(
    distance: float,
    query_gamma_prime: float,
    kg_gamma_prime: Optional[float],
    query_processing: str = "",
    kg_processing: str = "",
    matched: bool = True,
) -> Tuple[bool, str, str]:
    """Decide whether this neighbour may calibrate the query.

    Returns ``(allowed, reason_code, detail)``. Evaluation order and semantics
    mirror ``AlloyAnalysisTool._generate_proposals`` exactly: the distance gate
    (tightened when the processing route is unknown or differs), then an
    outright processing-route mismatch, then the gamma-prime class check.
    ``kg_gamma_prime`` of None means the neighbour carried no composition, in
    which case the class check is skipped -- as it is in the tool.
    """
    if not matched:
        return False, REJECT_NO_MATCH, "no knowledge-graph match returned"

    compatible = processing_compatible(query_processing, kg_processing)

    limit = max_anchor_distance(compatible)
    if not (distance < limit):
        return False, REJECT_DISTANCE, f"distance {distance:.2f} >= cutoff {limit:.2f}"

    if processing_mismatch(query_processing, kg_processing):
        return False, REJECT_PROCESSING, (f"query '{query_processing}' vs KG '{kg_processing}'")

    if kg_gamma_prime is not None:
        gp_diff = abs(query_gamma_prime - kg_gamma_prime)
        if gp_diff > KG_ANCHOR_MAX_GP_DIFF:
            return False, REJECT_GP_CLASS, (
                f"query {query_gamma_prime:.1f}% vs KG {kg_gamma_prime:.1f}% "
                f"(diff {gp_diff:.1f} > {KG_ANCHOR_MAX_GP_DIFF})")

    return True, "", ""


def blend(ml_value: float, kg_value: float, distance: float) -> float:
    """Sigmoid-weighted blend of an ML prediction and a KG experimental value."""
    w = kg_anchor_weight(distance)
    return ml_value * (1.0 - w) + kg_value * w
