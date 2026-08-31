# NEAR stratum: what anchoring reached, and what withheld it

Derived from the committed `seed42_v2prod_ml_physics_kg_*.csv` gate columns
(`kg_gate_allowed`, `kg_reject_code`, `kg_match_name`) against
`seed42_v2prod_ml_deterministic_*.csv`, with the withheld cases re-probed
through the production retrieval to identify which condition stopped them.
A NEAR alloy counts as anchored if any property on any row differs between the
two arms.

| alloy | d | rows | rows changed | outcome |
|---|---:|---:|---:|---|
| MAR-M* 200 | 0.024 | 11 | 3 | **anchored** |
| NIMONIC* 86 | 0.037 | 3 | 1 | **anchored** |
| Rene 95 | 0.107 | 1 | 0 | withheld — divergence below 15% (YS 8.6%, UTS 7.6%, EL 3.6%) |
| ATI AL 601 | 0.311 | 1 | 0 | withheld — **retrieval miss**: true neighbour INCONEL* 601 never returned; the retrieved match was INCONEL* 718 at d = 11.64, rejected on distance |
| Alloy 718 (AMS 5596 sheet) | 0.365 | 1 | 0 | withheld — divergence below 15% (YS 14.4%, UTS 5.2%, EL 1.5%) |
| Haynes 214 | 0.526 | 26 | 14 | **anchored** |
| Inconel X-750 | 0.757 | 1 | 0 | withheld — divergence below 15% (YS 11.0%, UTS 8.3%, EL 10.3%) |
| ATI Altemp 718 | 0.818 | 9 | 1 | **anchored** |
| Alloy 263 (C263) | 0.980 | 1 | 0 | withheld — divergence below 15% (YS 4.8%, UTS 4.3%, EL 9.4%) |
| Udimet 720 | 0.987 | 1 | 0 | withheld — neighbour UDIMET* 720LI carried no parsed property values |
| Haynes 263 | 1.145 | 26 | 12 | **anchored** |
| Rene 100 | 1.600 | 1 | 0 | withheld — **proposal deduplication**: the KG proposal was generated and then outranked, see below |
| Haynes 718 | 1.663 | 17 | 7 | **anchored** |

**Of the 13 NEAR alloys, anchoring reached 6. The remaining 7 were withheld by
the 15% ML-vs-KG divergence threshold (4), proposal deduplication (1), a
retrieval miss (1), and a neighbour with no parsed property values (1).**

None of the seven was withheld by the gamma-prime class guard or by processing-route
exclusion. Both gates exist and fire elsewhere -- the gamma-prime guard rejects
Astroloy's NIMONIC* 105 match at 47.6% against 33.9% -- but neither is what
limits anchoring on the near-duplicates.

The single most consequential finding is the retrieval miss: ATI AL 601 sits
0.31 wt% from INCONEL* 601 and the hybrid recall stage never surfaced it, so the
one NEAR alloy with an almost exact training twin got no anchoring at all. This
is the same failure measured at corpus scale in the retrieval-recall check
(78/88 top-3 recall), and it bounds what anchoring can deliver independently of
any threshold.

**Rene 100: a distance cliff, not the divergence gate.** Its elongation diverges
by 17.4% (ML 10.6% against KG 9.0%), above the threshold, and the anchoring
proposal *was* generated -- instrumenting `kg_anchor_weight` shows it firing at
d = 1.60 for a weight of 0.858. The proposal is then discarded by the
per-property deduplication at `analysis_tool.py:950`, which ranks
`(confidence, source)` in that order:

| proposal | confidence | rank |
|---|---|---|
| KG anchoring (IN-100, d = 1.60) | MEDIUM -- the rule is `HIGH if kg_distance < 1.5 else MEDIUM` | (2, 2) |
| elongation bounds | HIGH, hard-coded | (3, 1) |

`(3,1) > (2,2)` lexicographically, so a generic bounds rule displaces an
experimental value from a near-duplicate 1.6 wt% away. Had the alloy been at
d = 1.49 the KG proposal would have ranked (3, 2) and won.

This is a latent design issue rather than a one-off: any near-duplicate between
d = 1.5 and 4.5 whose property collides with a HIGH-confidence rule loses its
anchoring regardless of how well the knowledge-graph match agrees. It is
recorded, not changed -- the system is frozen for these results.
