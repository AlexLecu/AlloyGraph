# Alloy Designer Pipeline: Current vs Proposed

## CURRENT SYSTEM

```
USER TARGETS
(YS>=1000 MPa, EL>=15%, gamma'~30%, wrought, 900C)
 |
 v
+=====================================================================+
|                        DESIGN LOOP                                  |
|   max_iterations full evaluations + 3 extra for recoverable errors  |
+=====================================================================+
 |
 |  ITERATION N
 |
 v
+------------------------------------------------------------------+
| PHASE 1: LLM SYNTHESIS                                          |
|                                                                  |
|  Designer Agent (LLM)    tools=[] (BLIND - no validation tools)  |
|  +--------------------------------------------------------+     |
|  | Inputs:                                                 |     |
|  |   - Target properties (formatted string)                |     |
|  |   - Previous composition OR "create from scratch"       |     |
|  |   - Feedback from last failure (500+ tokens of prose)   |     |
|  |   - Priority focus (string-matched from own feedback)   |     |
|  |   - Novelty check (KG query, informational only)        |     |
|  |                                                         |     |
|  | Output:                                                 |     |
|  |   - reasoning (text)                                    |     |
|  |   - composition (JSON, elements summing to ~100%)       |     |
|  |   - processing (cast/wrought)                           |     |
|  +--------------------------------------------------------+     |
|                                                                  |
|  Post-processing (deterministic):                                |
|    1. Parse JSON (handle Groq str/dict quirks)                   |
|    2. Strip zero elements                                        |
|    3. Reject if sum wildly off (<85% or >110%)                   |
|    4. Auto-balance Ni to 100%                                    |
|    5. Round to 2 decimals                                        |
|    6. Enforce processing route                                   |
|    7. Hard gate: wrought + gamma'>50% = reject                  |
|                                                                  |
|  LLM CALL #1                                                    |
+------------------------------------------------------------------+
 |
 |  (recoverable errors loop back without counting as evaluation)
 |
 v
+------------------------------------------------------------------+
| FAST PRECHECK (middle iterations only)                           |
|                                                                  |
|  Deterministic physics check:                                    |
|    - TCP risk (Md thresholds)                                    |
|    - Lattice mismatch (>0.9% = critical)                         |
|    - Cr range, gamma' formers                                    |
|    - Wrought gamma' limit                                        |
|                                                                  |
|  If CRITICAL violations:                                         |
|    Skip expensive agent pipeline ----+                           |
|    Run Optimizer on features only    |                           |
|    (no real YS/UTS/EL/EM values)     |  BLIND SPOT:             |
|    Feed back to Designer             |  Optimizer gets Md/delta  |
|    Does NOT count as evaluation      |  not mechanical props     |
+----------------------------------+---+                           |
                                   |                               |
                                   +-------> next iteration -------+
 |
 v
+------------------------------------------------------------------+
| PHASE 2: FULL EVALUATION (shared pipeline)                       |
|                                                                  |
|  Pre-computation (deterministic):                                |
|  +------------------+  +----------+  +----------+                |
|  | KG Vector Search |  | ML Model |  | Physics  |                |
|  | (Weaviate)       |  | (XGBoost)|  | (SSS/GP) |                |
|  +--------+---------+  +----+-----+  +----+-----+                |
|           |                  |             |                      |
|           v                  v             v                      |
|  +---------------------------------------------------+           |
|  | AlloyAnalysisTool (deterministic)                  |           |
|  | - Triangulates ML/Physics/KG                       |           |
|  | - Detects discrepancies                            |           |
|  | - Generates correction PROPOSALS                   |           |
|  |   [HIGH] YS: 650 -> 820 MPa (empirical model)     |           |
|  |   [MED]  EL: 35 -> 22% (high-GP cap)              |           |
|  +---------------------------------------------------+           |
|                                                                  |
|  Agent Pipeline:                                                 |
|  +---------------------------+                                   |
|  | Analyst Agent (LLM)       |  LLM CALL #2                     |
|  | - Calls AlloySearchTool   |  (often finds distant KG matches |
|  | - Compares KG/ML/Physics  |   for synthetic compositions     |
|  | - Picks values + reasoning|   -> KG adds little value here)  |
|  +-------------+-------------+                                   |
|                |                                                 |
|                v                                                 |
|  +---------------------------+                                   |
|  | Reviewer Agent (LLM)      |  LLM CALL #3                     |
|  | - Calls VerifierTool      |                                   |
|  | - Fixes violations        |                                   |
|  | - Validates coherency     |                                   |
|  +-------------+-------------+                                   |
|                |                                                 |
|                v                                                 |
|  Deterministic post-processing:                                  |
|    - Safety net (override ignored HIGH proposals)                |
|    - Density/GP override (composition-determined)                |
|    - UTS >= YS * 1.05 floor                                     |
|    - UTS/YS ratio cap by class                                   |
|    - Elongation cap by GP level                                  |
|    - EM override if >20% from Reuss bound                        |
|    - Calibration (skip for agent-corrected props)                |
|    - compute_metallurgy_validation()                             |
|                                                                  |
|  Summary:                                                        |
|  +---------------------------+                                   |
|  | LLM summary generation    |  LLM CALL #4                     |
|  +---------------------------+                                   |
+------------------------------------------------------------------+
 |
 v
+------------------------------------------------------------------+
| SUCCESS CHECK                                                    |
|                                                                  |
|  All targets met?  ----YES----> RETURN RESULT (best_result)      |
|  TCP Low?                                                        |
|  No High Md penalty?                                             |
|                                                                  |
|  NO                                                              |
+------------------------------------------------------------------+
 |
 v
+------------------------------------------------------------------+
| FAILURE ANALYSIS + OPTIMIZATION                                  |
|                                                                  |
|  _classify_failures():                                           |
|    TCP_RISK, PROPERTY_SHORTFALL, PHYSICS_VIOLATION, etc.         |
|                                                                  |
|  Priority enforcement:                                           |
|    If YS met + TCP critical -> "TCP only mode"                   |
|                                                                  |
|  +-------------------------------+                               |
|  | Optimizer Agent (LLM)         |  LLM CALL #5                 |
|  | - Calls AlloyOptimizationAdvisor tool                         |
|  | - Tool computes EXACT sensitivities:                          |
|  |     dMd/dRe = -0.027/wt%                                     |
|  |     dGP/dAl = +8.2 vol%/wt%                                  |
|  | - Tool returns SPECIFIC suggestions:                          |
|  |     "Re 6.0% -> 4.5% (Md -0.04)"                             |
|  | - Agent reformats as text                                     |
|  +-------------------------------+                               |
|                                                                  |
|  Build feedback (prose):                                         |
|    "Design REJECTED:                                             |
|     CRITICAL: TCP phase formation risk                           |
|     PROPERTY TARGET MISS: 2 targets not met                     |
|     CURRENT VALUES vs TARGETS:                                   |
|       YS: 850 / 1000 MPa (85% of target, deficit: +150)         |
|     OPTIMIZATION SUGGESTIONS:                                    |
|       - PRIORITY 1: Reduce Re 6.0%->4.5% ...                    |
|       - PRIORITY 2: Increase Al 5.0%->6.5% ...                  |
|     Apply these corrections and propose revised composition."    |
|                                                                  |
+------------------------------------------------------------------+
 |
 v
 LOOP BACK TO PHASE 1 (Designer interprets 500+ tokens of prose
                        and GUESSES a new composition)


TOTAL PER ITERATION:  5 LLM calls
TOTAL FOR 3 ITERATIONS: 15 LLM calls
BOTTLENECK: LLM interpreting numerical suggestions as text
```


## PROPOSED SYSTEM

```
USER TARGETS
(YS>=1000 MPa, EL>=15%, gamma'~30%, wrought, 900C)
 |
 v
+=======================================================================+
| PHASE 1: LLM STRATEGIC DESIGN (one-shot, what LLMs are good at)     |
|                                                                       |
|  Designer Agent (LLM)  tools=[QuickCheckTool, AlloySearchTool]        |
|  +------------------------------------------------------------------+|
|  | "Design a wrought Ni superalloy with ~30% gamma prime,           ||
|  |  YS>=1000 MPa at 900C"                                           ||
|  |                                                                   ||
|  | LLM reasoning:                                                    ||
|  |   "30% GP wrought disc alloy -> Waspaloy/U720-class              ||
|  |    Al ~2.5%, Ti ~2.0%, Cr ~19% for oxidation                     ||
|  |    Mo+W for SSS, keep Md < 0.94                                  ||
|  |    Reference: Waspaloy (Ni-19.5Cr-13.5Co-4.3Mo-1.3Al-3Ti)"      ||
|  |                                                                   ||
|  | Can verify with QuickCheckTool before submitting:                 ||
|  |   "Md=0.932 OK, GP=28% OK, mismatch=0.3% OK"                    ||
|  +------------------------------------------------------------------+|
|                                                                       |
|  Output: initial_composition + strategic_reasoning                    |
|                                                                       |
|  LLM CALL #1 (only LLM call in the optimization loop)               |
+=======================================================================+
 |
 v
+=======================================================================+
| PHASE 2: DETERMINISTIC OPTIMIZATION (no LLM, pure computation)       |
|                                                                       |
|  Uses YOUR EXISTING code:                                             |
|    - compute_alloy_features()                                         |
|    - _get_physics_predictions()                                       |
|    - _get_ml_predictions()                                            |
|    - _calculate_md_sensitivity()                                      |
|    - _calculate_gp_sensitivity()                                      |
|    - classify_tcp_risk()                                              |
|                                                                       |
|  +----------------------------------------------------------------+  |
|  |                                                                |  |
|  |  while iteration < max_iterations and not converged:           |  |
|  |                                                                |  |
|  |    1. EVALUATE (deterministic)                                 |  |
|  |       features = compute_alloy_features(composition)           |  |
|  |       ml_props = ml_predict(composition, temp, processing)     |  |
|  |       physics_props = physics_predict(composition, temp, proc) |  |
|  |       props = blend(ml_props, physics_props)  # weighted       |  |
|  |       apply deterministic overrides (UTS>=YS, EL cap, EM, ..) |  |
|  |                                                                |  |
|  |    2. CHECK TARGETS                                            |  |
|  |       deficits = {                                             |  |
|  |         "YS":  target_ys - props["YS"],    # +150 MPa needed  |  |
|  |         "TCP": md_avg - 0.940,             # -0.02 needed     |  |
|  |         "GP":  props["GP"] - target_gp,    # +2% over         |  |
|  |       }                                                        |  |
|  |       if all_met(deficits): break  # CONVERGED                |  |
|  |                                                                |  |
|  |    3. COMPUTE GRADIENTS (your existing sensitivity code)       |  |
|  |       for each element in composition:                         |  |
|  |         d_md  = calculate_md_sensitivity(comp, element)        |  |
|  |         d_gp  = calculate_gp_sensitivity(comp, element)        |  |
|  |         d_ys  = d_gp * coeff_gp  # YS response via GP         |  |
|  |                                                                |  |
|  |    4. RANK & SELECT best adjustments                           |  |
|  |       Score each element change by:                            |  |
|  |         impact_on_deficits / side_effects                      |  |
|  |       Priority: fix constraint violations first (TCP, GP),     |  |
|  |                 then optimize targets (YS, EL)                 |  |
|  |                                                                |  |
|  |    5. APPLY ADJUSTMENTS (direct, precise)                      |  |
|  |       composition["Re"] -= 1.5    # exact, not LLM-guessed    |  |
|  |       composition["Al"] += 0.8                                 |  |
|  |       composition["Ni"] = balance  # Ni: bal.                  |  |
|  |       enforce_constraints(composition)                         |  |
|  |         - sum = 100%                                           |  |
|  |         - Cr in [5, 20]                                        |  |
|  |         - Ni >= 40%                                            |  |
|  |         - wrought: GP_formers <= 7%                            |  |
|  |                                                                |  |
|  |    6. LOG iteration (for final report)                         |  |
|  |       history.append({composition, props, deficits, changes})  |  |
|  |                                                                |  |
|  +----------------------------------------------------------------+  |
|                                                                       |
|  0 LLM calls, runs in seconds                                        |
|  Typically converges in 5-15 iterations (cheap, ~50ms each)           |
+=======================================================================+
 |
 v
+=======================================================================+
| PHASE 3: LLM EVALUATION & EXPLANATION (what LLMs are good at)       |
|                                                                       |
|  Run ONCE on the final converged composition:                         |
|                                                                       |
|  +---------------------------+                                        |
|  | Analyst Agent (LLM)       |  LLM CALL #2                         |
|  | - KG search NOW valuable  |  (final comp might match real alloy) |
|  | - Triangulate ML/Physics  |                                       |
|  | - Transparent reasoning   |                                       |
|  +-------------+-------------+                                        |
|                |                                                      |
|                v                                                      |
|  +---------------------------+                                        |
|  | Reviewer Agent (LLM)      |  LLM CALL #3                         |
|  | - Validate final result   |                                       |
|  | - Check for issues the    |                                       |
|  |   optimizer can't see     |                                       |
|  | - Quality assessment      |                                       |
|  +-------------+-------------+                                        |
|                |                                                      |
|  Deterministic validation:                                            |
|    - All existing post-processing                                     |
|    - compute_metallurgy_validation()                                  |
|                                                                       |
|  +---------------------------+                                        |
|  | LLM Summary               |  LLM CALL #4                         |
|  | - Design rationale        |                                       |
|  | - Optimization trajectory |                                       |
|  | - Trade-off analysis      |                                       |
|  | - Comparison to known     |                                       |
|  |   alloys from KG          |                                       |
|  +---------------------------+                                        |
|                                                                       |
|  If Reviewer finds issues optimizer missed:                           |
|    Option A: Re-run Phase 2 with adjusted constraints                 |
|    Option B: Flag as advisory (rare edge case)                        |
+=======================================================================+
 |
 v
RETURN RESULT
  - composition
  - predicted properties (ML + physics + KG-anchored)
  - optimization history (what changed each iteration and why)
  - LLM explanation and assessment
  - issues / recommendations


TOTAL: 4 LLM calls (vs 15)
OPTIMIZATION: deterministic, fast, precise
LLM USED FOR: strategy (Phase 1) + explanation (Phase 3)
```


## SIDE-BY-SIDE COMPARISON

```
                    CURRENT                          PROPOSED
                    -------                          --------

Initial Design:     LLM (blind, no tools)            LLM (with QuickCheck tool)

Optimization:       LLM interprets prose feedback     Direct gradient application
                    "Reduce Re 6%->4.5%"             composition["Re"] -= 1.5
                    -> LLM might set Re to 5%         -> Re = 4.5 exactly

Iterations:         3 full evals (expensive)          5-15 cheap evals (ms each)
                    5 LLM calls each = 15 total       0 LLM calls in loop

Evaluation:         Every iteration (wasteful for      Once on final result
                    synthetic compositions with        (KG search meaningful now -
                    distant KG matches)                converged comp may match
                                                       a real alloy)

LLM calls total:    15 (3 iterations)                  4 (fixed)

Latency:            3-5 min (LLM-bound)               30-60 sec (1 LLM design +
                                                       fast optimization +
                                                       1 LLM evaluation)

Precision:          LLM may partially apply            Exact numerical adjustments
                    suggestions or hallucinate

Feedback loop:      Indirect (compute gradient ->      Direct (compute gradient ->
                    text -> LLM -> new comp)           apply gradient -> new comp)

Explainability:     Reasoning spread across            Clean separation:
                    15 LLM outputs                     optimization log (exact) +
                                                       LLM explanation (narrative)

Code reuse:         ~80% of existing code              ~90% of existing code
                    (agents, tools, eval pipeline)     (sensitivity calcs, physics,
                                                       ML, eval pipeline, agents)

What changes:       Replace loop() internals           loop() becomes deterministic
                    Keep all tools, models, agents     Keep all agents for Phase 1+3
                    Keep evaluation pipeline           Keep full eval pipeline
```
