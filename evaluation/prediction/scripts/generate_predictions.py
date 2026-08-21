#!/usr/bin/env python3
"""
AlloyGraph Prediction Generator

Generates predictions for evaluation with multiple ablation modes:
  --ml-only            : Raw ML model output (no agents, no physics enforcement)
  --ml-deterministic   : ML + physics enforcement (UTS/YS caps, EM VRH, EL caps) — no agents
  --ml-physics-kg      : ML + physics + KG anchoring, deterministically applied — no agents
  --llm-only           : Raw LLM prediction from composition (no ML, no KG, no agents)
  (default)            : Full system (ML + physics + KG + multi-agent pipeline)

Features:
- Fresh evaluator instance per alloy to avoid state corruption
- Rate limit handling with exponential backoff
- Intermediate result saving
- Comprehensive output with all metadata

Usage:
    python generate_predictions.py --dataset sss
    python generate_predictions.py --dataset precip --ml-only
    python generate_predictions.py --dataset precip --ml-deterministic
    python generate_predictions.py --dataset sss --llm-only
    python generate_predictions.py --dataset sss --llm-only --model openai
    python generate_predictions.py --dataset sss --llm-only --model openai/gpt-4.1-mini
    python generate_predictions.py --dataset custom --custom-path /path/to/data.jsonl
"""

import sys
import os
import json
import time
import logging
import argparse
import random
import re
import gc
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# Disable telemetry before importing crewai
os.environ['CREWAI_TELEMETRY_OPT_OUT'] = 'true'
os.environ['OTEL_SDK_DISABLED'] = 'true'
os.environ['LITELLM_LOG'] = 'ERROR'

# Suppress noisy logs
logger = logging.getLogger(__name__)

logging.getLogger('LiteLLM').setLevel(logging.CRITICAL)
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('crewai').setLevel(logging.WARNING)

import pandas as pd
import numpy as np

# Add backend and scripts to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)  # prediction/
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))  # AlloyGraph/
BACKEND_DIR = os.path.join(PROJECT_ROOT, 'backend')
sys.path.insert(0, BACKEND_DIR)
sys.path.insert(0, SCRIPT_DIR)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def seed_everything(seed):
    """Seed Python and numpy RNGs and reduce run-to-run variation.

    What this does and does NOT guarantee — measured, not assumed:

      - Python `random` and `numpy.random` are seeded. This is what makes any
        sampling or ordering decision in the harness repeatable.

      - Predictions are reproducible to ~1e-13 relative, NOT bit-exact. Across
        5 seeded ML_ONLY runs the pred_ys values agreed to 13 significant
        figures but differed in the last digit (e.g. 394.95063667307 vs
        394.95063667306994). The cause is summation order in parallel
        reductions inside XGBoost/BLAS, which a seed does not control. This is
        ~15 orders of magnitude below the precision of an MPa prediction, so it
        does not affect any reported metric; do not expect `diff` on two CSVs
        from the same seed to come back empty.

      - Thread pinning below reduces but does not remove that variation: numpy
        (and its BLAS) is imported at module load, before main() calls this, so
        the BLAS variables arrive too late to bind. They do apply to XGBoost,
        which is imported lazily inside the prediction functions. Exporting
        OMP_NUM_THREADS=1 in the shell before launching is the only way to pin
        every backend.

      - PYTHONHASHSEED is set for any *subprocess*; this process fixed its hash
        seed at interpreter start, so it would need a re-exec to take effect
        here.

      - Remote LLM calls are NOT made deterministic by a local seed. --seed
        additionally drives sampling temperature to 0.0 and forwards the seed
        to litellm, which passes it to providers that honour a `seed` parameter
        (OpenAI, Groq). Both are best-effort, and neither is guaranteed.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)

    for var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ[var] = '1'

    random.seed(seed)
    np.random.seed(seed)


# ---------------------------------------------------------------------------
# CrewAI state management
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Shared rate-limit gate
# ---------------------------------------------------------------------------

class RateLimitGate:
    """Process-wide pause shared by all workers.

    When any worker sees a 429 the whole pool stops issuing new requests for a
    backoff window, rather than each worker backing off alone and continuing to
    hammer the endpoint. Backoff doubles per consecutive trip and resets once a
    request succeeds.
    """

    def __init__(self, base_wait=30.0, max_wait=300.0):
        self._open = threading.Event()
        self._open.set()                 # set == traffic allowed
        self._lock = threading.Lock()
        self._base = base_wait
        self._max = max_wait
        self._consecutive = 0
        self.trips = 0

    def configure(self, base_wait=None, max_wait=None):
        """Apply CLI-supplied backoff bounds to the already-constructed gate."""
        with self._lock:
            if base_wait is not None:
                self._base = float(base_wait)
            if max_wait is not None:
                self._max = float(max_wait)

    def wait(self):
        """Block while the gate is closed."""
        self._open.wait()

    def trip(self):
        """Record a 429 and close the gate for a backoff window."""
        with self._lock:
            if not self._open.is_set():
                return                   # another worker is already backing off
            self._consecutive += 1
            self.trips += 1
            wait = min(self._base * (2 ** (self._consecutive - 1)), self._max)
            self._open.clear()
        print(f"  [rate limit] pausing all workers for {wait:.0f}s "
              f"(trip #{self.trips})")
        time.sleep(wait)
        self._open.set()

    def succeeded(self):
        with self._lock:
            self._consecutive = 0


RATE_GATE = RateLimitGate()


def reset_event_context():
    """Clear the per-execution event ContextVars.

    These are ContextVars, and in CPython each thread carries its own context,
    so this is thread-local and safe to call from a worker. It is the part that
    actually prevents the _event_id_stack overflow.
    """
    try:
        from crewai.events import event_context
        event_context._event_id_stack.set(())
        event_context._last_event_id.set(None)
        event_context._triggering_event_id.set(None)
    except (ImportError, AttributeError) as e:
        print(f"  Warning: Could not reset event_context: {e}")


def reset_crewai_state():
    """Full reset: event ContextVars plus the global event bus.

    MAIN THREAD ONLY. crewai_event_bus is a process-wide singleton, so flushing
    it from a worker would discard events belonging to other in-flight
    evaluations. Workers should call reset_event_context() instead.
    """
    reset_event_context()

    try:
        from crewai.events.event_bus import crewai_event_bus
        try:
            crewai_event_bus.flush(timeout=5.0)
        except Exception:
            pass
        with crewai_event_bus._futures_lock:
            crewai_event_bus._pending_futures.clear()
    except (ImportError, AttributeError) as e:
        print(f"  Warning: Could not reset event_bus: {e}")

    gc.collect()


# ---------------------------------------------------------------------------
# Prediction modes
# ---------------------------------------------------------------------------

def run_full_system(composition, processing, temperature, max_retries=3, base_wait=30, llm_config=None):
    """Run full-system evaluation with fresh evaluator and retry logic."""
    from alloy_crew.alloy_evaluator import AlloyEvaluationCrew

    for attempt in range(max_retries):
        try:
            RATE_GATE.wait()
            reset_event_context()
            evaluator = AlloyEvaluationCrew(llm_config=llm_config)
            result = evaluator.run(
                composition=composition,
                processing=processing,
                temperature=temperature
            )
            del evaluator
            gc.collect()
            RATE_GATE.succeeded()
            return result

        except Exception as e:
            error_str = str(e).lower()
            is_rate_limit = any(x in error_str for x in [
                'rate', 'limit', '429', 'quota', 'too many', 'throttl'
            ])
            # A provider outage or dropped connection is transient and affects
            # every worker at once, so it must close the shared gate exactly like
            # a 429. Treating it as a generic per-row error is what turned a
            # DeepInfra blip into 159 consecutive failures: each row burned its
            # retries against a dead endpoint and the pool shredded the queue in
            # minutes instead of waiting for recovery.
            is_transient_provider = any(x in error_str for x in [
                'connection error', 'connection reset', 'connection aborted',
                'internalservererror', 'internal server error', 'service unavailable',
                'bad gateway', 'timeout', 'timed out', 'temporarily unavailable',
                '500', '502', '503', '504',
            ])
            is_event_stack = 'event stack' in error_str or 'depth limit' in error_str

            if is_rate_limit or is_transient_provider:
                # Close the shared gate so every worker backs off together.
                if is_transient_provider and not is_rate_limit:
                    print(f"  Provider error ({str(e)[:60]}...), pausing all workers")
                RATE_GATE.trip()
            elif is_event_stack:
                # reset_event_context(), not reset_crewai_state(): this runs on a
                # pool worker, and flushing the process-wide crewai_event_bus from
                # here would discard other workers' in-flight events (and with them
                # their token_usage and pipeline_stage bookkeeping).
                print(f"  Event stack overflow, performing deep reset...")
                reset_event_context()
                gc.collect()
                time.sleep(2)
                reset_event_context()
                time.sleep(3)
            else:
                if attempt < max_retries - 1:
                    print(f"  Error: {str(e)[:80]}... retrying in 10s")
                    time.sleep(10)
                else:
                    raise

    # Final attempt — still on a worker thread, so thread-local reset only.
    reset_event_context()
    evaluator = AlloyEvaluationCrew(llm_config=llm_config)
    return evaluator.run(composition=composition, processing=processing, temperature=temperature)


def run_ml_only(composition, processing, temperature):
    """Run ML-only prediction without LLM agents."""
    from alloy_crew.models.predictor import AlloyPredictor
    from alloy_crew.models.feature_engineering import compute_alloy_features
    from alloy_crew.config.alloy_parameters import is_sss_alloy

    predictor = AlloyPredictor.get_shared_predictor()
    result_df = predictor.predict(
        composition,
        extra_params={'processing': processing},
        temperatures=[temperature]
    )

    if result_df.empty:
        return {'status': 'FAIL', 'error': 'Empty prediction result'}

    row = result_df.iloc[0]
    features = compute_alloy_features(composition)
    density = round(features.get("density_calculated_gcm3", 0), 2)
    gp = 0.0 if is_sss_alloy(composition) else round(features.get("gamma_prime_estimated_vol_pct", 0), 1)

    return {
        'properties': {
            'Yield Strength': float(row.get('ys', 0)) if 'ys' in row else None,
            'Tensile Strength': float(row.get('uts', 0)) if 'uts' in row else None,
            'Elongation': float(row.get('el', 0)) if 'el' in row else None,
            'Elastic Modulus': float(row.get('em', 0)) if 'em' in row else None,
            'Density': density,
            'Gamma Prime': gp,
        },
        'confidence': {'level': 'MEDIUM', 'score': 0.5},
        'status': 'SUCCESS',
    }


def _ml_plus_physics(composition, processing, temperature):
    """ML ensemble prediction followed by the deterministic physics corrections.

    Shared by --ml-deterministic and --ml-physics-kg so the two modes cannot
    drift apart. Returns ``(properties, gamma_prime_pct, correction_notes)``,
    or ``(None, 0.0, [])`` when the predictor yields nothing. All three
    branches must keep the same arity: callers unpack three names, so a
    short tuple raises before the ``props is None`` guard can run.
    """
    from alloy_crew.models.predictor import AlloyPredictor
    from alloy_crew.models.feature_engineering import compute_alloy_features
    from alloy_crew.config.alloy_parameters import is_sss_alloy
    from alloy_crew.physics_corrections import apply_physics_corrections, PRODUCTION

    predictor = AlloyPredictor.get_shared_predictor()
    result_df = predictor.predict(
        composition, extra_params={'processing': processing}, temperatures=[temperature]
    )
    if result_df.empty:
        return None, 0.0, []

    row = result_df.iloc[0]
    features = compute_alloy_features(composition)
    density = round(features.get("density_calculated_gcm3", 0), 2)
    gp = 0.0 if is_sss_alloy(composition) else round(
        features.get("gamma_prime_estimated_vol_pct", 0), 1)

    props = {
        'Yield Strength': float(row.get('ys', 0)) if 'ys' in row else None,
        'Tensile Strength': float(row.get('uts', 0)) if 'uts' in row else None,
        'Elongation': float(row.get('el', 0)) if 'el' in row else None,
        'Elastic Modulus': float(row.get('em', 0)) if 'em' in row else None,
        'Density': density,
        'Gamma Prime': gp,
    }

    # PRODUCTION = the rules alloy_evaluator actually applies. This harness used
    # to carry a drifted copy (flat 2.4 SSS ratio, wrought elongation caps on
    # cast alloys, 20% EM threshold). See physics_corrections for both profiles.
    props, notes = apply_physics_corrections(
        props, composition, processing, temperature, gp, profile=PRODUCTION,
    )
    return props, gp, notes


def run_ml_deterministic(composition, processing, temperature):
    """ML + deterministic physics corrections (no agents, no KG).

    Applies:
    - Density & gamma prime from composition (not ML)
    - UTS >= YS * 1.05 floor
    - UTS/YS ratio ceiling (processing & gamma-prime aware)
    - Elongation caps for high gamma-prime alloys
    - EM override if >15% from the Voigt-Reuss-Hill average
    - compute_metallurgy_validation for TCP risk & penalties
    """
    from alloy_crew.tools.metallurgy_tools import compute_metallurgy_validation

    props, gp, notes = _ml_plus_physics(composition, processing, temperature)
    if props is None:
        return {'status': 'FAIL', 'error': 'Empty prediction result'}
    em_skipped = any(n.startswith('EM_OVERRIDE_SKIPPED_SC_DS') for n in notes)

    # --- Step 3: Metallurgical validation (TCP, penalties, intervals) ---
    validation = compute_metallurgy_validation(
        properties=props,
        composition=composition,
        temperature_c=temperature,
        processing=processing,
    )

    return {
        'properties': props,
        'confidence': {'level': 'MEDIUM', 'score': 0.5},
        'status': 'SUCCESS',
        'tcp_risk': validation.get('tcp_risk', 'N/A'),
        'validation_status': validation.get('status', 'UNKNOWN'),
        'penalty_score': validation.get('penalty_score', 0),
        'em_override_skipped_sc_ds': em_skipped,
    }


def run_ml_physics_kg(composition, processing, temperature):
    """ML + deterministic physics + KG anchoring, with no agents.

    Fills the gap between --ml-deterministic and the full system: it isolates
    what knowledge-graph calibration contributes on its own, without any LLM
    reasoning on top.

    Pipeline:
      1. ML ensemble prediction (identical to --ml-only)
      2. deterministic physics corrections (identical to --ml-deterministic)
      3. KG retrieval through the production path -- AlloySearchTool then
         _slim_kg_context, exactly as alloy_evaluator does it
      4. AlloyAnalysisTool builds the calibration proposals the Analyst would
         receive; we deterministically accept those tagged KG_anchoring and
         ignore the rest, so the delta against --ml-deterministic is purely
         knowledge-graph anchoring and not the empirical physics proposals
      5. physics corrections re-applied, since a KG override can otherwise push
         UTS back above its ratio ceiling. The rules are idempotent, and this
         matches production, where enforcement runs after the agents.
    """
    from alloy_crew.physics_corrections import apply_physics_corrections, PRODUCTION
    from alloy_crew.kg_anchoring import KG_ANCHOR_SOURCE
    from alloy_crew.tools.metallurgy_tools import compute_metallurgy_validation

    props, gp, notes = _ml_plus_physics(composition, processing, temperature)
    if props is None:
        return {'status': 'FAIL', 'error': 'Empty prediction result'}
    em_skipped = any(n.startswith('EM_OVERRIDE_SKIPPED_SC_DS') for n in notes)

    # --- Step 3: KG retrieval (same calls, same order, as alloy_evaluator) ---
    kg_match, kg_applied, gate = None, [], {}
    try:
        from alloy_crew.tools.rag_tools import AlloySearchTool
        from alloy_crew.tools.analysis_tool import AlloyAnalysisTool
        from alloy_crew.alloy_evaluator import _slim_kg_context

        kg_raw = AlloySearchTool()._run(composition=composition, limit=3, processing=processing)
        kg_context = _slim_kg_context(kg_raw, target_temp=temperature)

        analysis = AlloyAnalysisTool()._run(
            composition=composition,
            temperature_c=temperature,
            processing=processing,
            kg_context=kg_context,
        )
        analysis = json.loads(analysis) if isinstance(analysis, str) else analysis

        kg_match = (analysis.get('alloy_analysis') or {}).get('kg_match')
        gate = analysis.get('kg_gate') or {}

        # --- Step 4: accept only the KG-anchoring proposals, no LLM ---
        for prop in analysis.get('proposed_corrections') or []:
            if prop.get('source') != KG_ANCHOR_SOURCE:
                continue
            name, value = prop.get('property_name'), prop.get('proposed_value')
            if name in props and isinstance(value, (int, float)):
                kg_applied.append(f"{name}: {props[name]} -> {value}")
                props[name] = value
    except Exception as e:
        logger.warning(f"KG anchoring unavailable ({e}); falling back to ML+physics only")

    # --- Step 5: re-enforce physics after the KG overrides ---
    props, _ = apply_physics_corrections(
        props, composition, processing, temperature, gp, profile=PRODUCTION,
    )

    validation = compute_metallurgy_validation(
        properties=props, composition=composition,
        temperature_c=temperature, processing=processing,
    )

    return {
        'properties': props,
        'confidence': {'level': 'MEDIUM', 'score': 0.5},
        'status': 'SUCCESS',
        'tcp_risk': validation.get('tcp_risk', 'N/A'),
        'validation_status': validation.get('status', 'UNKNOWN'),
        'penalty_score': validation.get('penalty_score', 0),
        'kg_match_name': (kg_match or {}).get('name'),
        'kg_match_distance': (kg_match or {}).get('distance'),
        # The harness takes len() of this, so it must stay a list.
        'corrections_applied': kg_applied,
        # Rejection accounting: which gate stopped anchoring, and the weight
        # earned when it did fire.
        'kg_gate_allowed': gate.get('allowed'),
        'kg_reject_code': gate.get('reject_code'),
        'kg_reject_detail': gate.get('reject_detail'),
        'kg_weight': gate.get('weight'),
        'em_override_skipped_sc_ds': em_skipped,
    }


LLM_ONLY_MODELS = {
    # Retired by Groq on 2026-08-16; kept for accounts that still have access.
    "groq": "groq/llama-3.3-70b-versatile",
    "deepinfra": "deepinfra/meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "together": "together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "openai-mini": "openai/gpt-4.1-mini",
    "openai-ft": "openai/ft:gpt-4.1-mini-2025-04-14:digital-science-dimensions::CoGgLDPB",
}
#: Groq decommissioned llama-3.3-70b-versatile on 2026-08-16, so the "groq"
#: alias above no longer resolves. Default to a provider that still serves a
#: model, and keep the dead alias only for accounts that retain access.
LLM_ONLY_DEFAULT = "openai-mini"


LLM_SYSTEM_PROMPT = (
    "You are an expert materials scientist specializing in nickel-based superalloys. "
    "Predict mechanical properties accurately based on composition, processing, and temperature. "
    "Reason briefly, then answer as JSON."
)


#: Counts how often a temperature series had to be resolved by nearest match
#: rather than an exact hit. Read after a run for the reporting caveat.
SERIES_RESOLUTION = {"scalar": 0, "series_exact": 0, "series_nearest": 0}

#: Rows that needed a re-draw because the model returned a null answer, and
#: rows that never produced one. Read after a run for the reporting caveat.
DEGENERATE_RESPONSES = {"retried": 0, "unrecovered": 0}

#: Providers that reject ``seed`` outright. litellm raises UnsupportedParamsError
#: rather than dropping it, so the parameter cannot simply be sent hopefully.
#: DeepInfra is one: its Llama endpoint exposes no seed control at all.
#:
#: A run against such a provider is therefore NOT seeded. Temperature 0.0 still
#: makes it near-deterministic, but bit-identical reproduction is not available
#: and the report must say so rather than implying a seed was honoured.
SEED_UNSUPPORTED = {"seed_dropped": False, "provider": None}


#: No solid is stiffer than about 1220 GPa (diamond); superalloys sit near 200.
#: A value above this can only be MPa, so it is converted rather than scored.
EM_MPA_THRESHOLD_GPA = 1000.0

#: Counts unit conversions applied, for the reporting caveat.
UNIT_NORMALISATIONS = {"em_mpa_to_gpa": 0}


def _normalise_elastic_modulus(value):
    """Convert an elastic modulus quoted in MPa to GPa.

    The prompt asks for ``{"elastic_modulus": <number>}`` and never states a
    unit, so a model answering 210000 is answering in MPa and is not wrong --
    reading it as 210000 GPa is our parsing error. Two of 471 rows in the
    August stock run and four of 466 in the February archive do this, and left
    unconverted they move the baseline's elastic-modulus MAE from 16 GPa to
    1397. Scoring that would misreport a unit convention as a modelling
    failure, in the same way that scoring a null answer as 0 MPa would.
    """
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if value > EM_MPA_THRESHOLD_GPA:
            UNIT_NORMALISATIONS["em_mpa_to_gpa"] += 1
            return value / 1000.0
    return value


def _is_degenerate(preds):
    """True when a response carries no non-zero number for any property.

    The fine-tuned model intermittently answers
    ``{"yield_strength": 0.0, "uts": 0.0, "elongation": 0.0, "elasticity": 0.0}``
    or the same shape filled with nulls. A yield strength of exactly zero
    alongside a zero modulus is not a physical claim, it is the model failing
    to answer, and scoring it as a prediction of 0 MPa would measure a decoding
    failure rather than the model's knowledge. These are treated like an
    unparseable response and re-drawn.

    The rate is not a quirk of our settings: the February protocol reproduced
    exactly -- same prompt, temperature 0.3, no seed -- now returns this on 38%
    of sampled rows, against none in the archived run. The behaviour of this
    fine-tuned model as served has changed since the archived baseline was
    collected.
    """
    for value in preds.values():
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value != 0:
            return False
    return True


def _scalar_at_temperature(value, temperature):
    """Reduce a model answer to one number at the requested temperature.

    The fine-tuned model was trained on targets shaped as
    ``{"yield_strength": [{"temp_c": "21", "value": 740.0}, ...]}``, so at
    sampling temperature 0.0 it frequently reproduces that whole series
    instead of the single value the prompt asks for. Writing the list straight
    into the CSV would silently corrupt the column, so the series is resolved
    here.

    Exact temperature match wins. Failing that the nearest entry is taken --
    reading the model's own answer at the closest point it reported, which is
    what a person scoring the response by hand would do. Nothing is
    interpolated: no number is produced that the model did not state.
    """
    if not isinstance(value, list):
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            SERIES_RESOLUTION["scalar"] += 1
        return value

    points = []
    for entry in value:
        if not isinstance(entry, dict):
            continue
        try:
            t = float(entry.get("temp_c", entry.get("temperature", "nan")))
            v = float(entry.get("value", entry.get("val", "nan")))
        except (TypeError, ValueError):
            continue
        if v == v and t == t:  # both non-NaN
            points.append((t, v))
    if not points:
        return None

    target = float(temperature)
    exact = [v for t, v in points if abs(t - target) < 1e-6]
    if exact:
        SERIES_RESOLUTION["series_exact"] += 1
        return exact[0]
    SERIES_RESOLUTION["series_nearest"] += 1
    return min(points, key=lambda tv: abs(tv[0] - target))[1]


def run_llm_only(composition, processing, temperature, model_key=None, max_retries=6,
                 seed=None, sampling_temperature=0.3):
    """Run LLM-only prediction: prompt an LLM with composition, no ML/KG/agents.

    When ``seed`` is set the sampling temperature is driven to 0.0 by the caller
    and the seed is forwarded to litellm for providers that honour it.
    """
    from litellm import completion as llm_completion

    model_key = model_key or LLM_ONLY_DEFAULT
    model_name = LLM_ONLY_MODELS.get(model_key, model_key)  # allow raw litellm model strings too

    comp_str = ", ".join(
        f"{elem}: {wt}%" for elem, wt in sorted(composition.items(), key=lambda x: -x[1])
    )

    prompt = f"""Given the following nickel-based superalloy composition and conditions, predict its mechanical properties.

COMPOSITION (wt%):
{comp_str}

PROCESSING: {processing}
TEMPERATURE: {temperature}°C

Predict the following properties. Reason briefly about the alloy class and expected behavior, then respond with JSON:
{{"yield_strength": <number>, "uts": <number>, "elongation": <number>, "elastic_modulus": <number>}}"""

    completion_kwargs = {
        "temperature": sampling_temperature,
        "max_tokens": 512,
    }
    for attempt in range(max_retries):
        if seed is not None and not SEED_UNSUPPORTED["seed_dropped"]:
            # Honoured by OpenAI; rejected outright by DeepInfra (see
            # SEED_UNSUPPORTED). Re-draws walk the seed deterministically, so a
            # run stays reproducible while a null answer can still be retried --
            # at a fixed seed the model returns the identical null every time.
            completion_kwargs["seed"] = seed + 1000 * attempt
        elif seed is not None:
            # Seed unavailable: vary nothing, but a re-draw is still worth
            # making because sampling is not perfectly deterministic upstream.
            completion_kwargs.pop("seed", None)
        try:
            response = llm_completion(
                model=model_name,
                messages=[
                    {"role": "system", "content": LLM_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                **completion_kwargs,
            )
            content = response.choices[0].message.content.strip()

            # Try JSON extraction (handle nested braces, trailing commas)
            parsed = None
            json_match = re.search(r'\{(?:[^{}]|\{[^{}]*\})*\}', content, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                # Clean common LLM JSON issues: trailing commas, single quotes
                json_str = re.sub(r',\s*}', '}', json_str)
                json_str = json_str.replace("'", '"')
                try:
                    parsed = json.loads(json_str)
                except json.JSONDecodeError:
                    pass

            if parsed:
                # Flexible key matching — models may use different key names
                def find_val(d, *keys):
                    for k in keys:
                        for dk, dv in d.items():
                            if k in dk.lower().replace(' ', '_'):
                                return dv
                    return None

                preds = {
                    'Yield Strength': _scalar_at_temperature(
                        find_val(parsed, 'yield', 'ys'), temperature),
                    'Tensile Strength': _scalar_at_temperature(
                        find_val(parsed, 'uts', 'tensile', 'ultimate'), temperature),
                    'Elongation': _scalar_at_temperature(
                        find_val(parsed, 'elong', 'el'), temperature),
                    'Elastic Modulus': _normalise_elastic_modulus(
                        _scalar_at_temperature(
                            find_val(parsed, 'elastic', 'modulus', 'em'), temperature)),
                }
            else:
                # Fallback: extract numbers in order
                numbers = re.findall(r'[\d.]+', content)
                if len(numbers) >= 4:
                    preds = {
                        'Yield Strength': float(numbers[0]),
                        'Tensile Strength': float(numbers[1]),
                        'Elongation': float(numbers[2]),
                        'Elastic Modulus': _normalise_elastic_modulus(float(numbers[3])),
                    }
                else:
                    raise ValueError(f"Could not parse LLM response: {content[:200]}")

            if _is_degenerate(preds):
                if attempt < max_retries - 1:
                    DEGENERATE_RESPONSES["retried"] += 1
                    continue
                DEGENERATE_RESPONSES["unrecovered"] += 1
                return {'status': 'FAIL',
                        'error': 'model returned a null answer on every attempt'}

            return {
                'properties': preds,
                'confidence': {'level': 'LOW', 'score': 0.3},
                'status': 'SUCCESS',
            }

        except Exception as e:
            error_str = str(e).lower()
            if "does not support parameters" in error_str and "seed" in error_str:
                # Drop the seed for the rest of the run and retry immediately.
                # Recorded so the report can state the run was not seeded.
                if not SEED_UNSUPPORTED["seed_dropped"]:
                    SEED_UNSUPPORTED["seed_dropped"] = True
                    SEED_UNSUPPORTED["provider"] = model_name.split("/")[0]
                    print(f"  [seed] {model_name.split('/')[0]} rejects seed; "
                          f"continuing unseeded at temperature {sampling_temperature}")
                completion_kwargs.pop("seed", None)
                continue
            print(f"  [DEBUG] Error (attempt {attempt+1}): {str(e)[:200]}")
            if 'rate' in error_str or '429' in error_str or 'too many' in error_str:
                wait_time = 30 * (attempt + 1)
                print(f"  Rate limit, waiting {wait_time}s...")
                time.sleep(wait_time)
            elif attempt < max_retries - 1:
                print(f"  Error: {str(e)[:80]}... retrying")
                time.sleep(5)
            else:
                raise

    return {'status': 'FAIL', 'error': 'Max retries exceeded'}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_actual_values(alloy_data, temperature):
    """Extract actual property values at given temperature."""
    actuals = {}
    prop_map = {
        'yield_strength': 'actual_ys',
        'uts': 'actual_uts',
        'elongation': 'actual_el',
        'elasticity': 'actual_em'
    }

    for prop_name, col_name in prop_map.items():
        for entry in alloy_data.get(prop_name, []):
            temp = float(entry.get('temp_c', 999))
            if abs(temp - temperature) < 5:
                actuals[col_name] = entry.get('value')
                break

    return actuals


def get_all_temperatures(alloy_data):
    """Get all temperatures with any property data."""
    temps = set()
    for prop in ['yield_strength', 'uts', 'elongation', 'elasticity']:
        for entry in alloy_data.get(prop, []):
            temps.add(float(entry.get('temp_c', 20)))
    return sorted(temps)


def get_llm_config(llm_choice, temperature=0.1):
    """Get LLM configuration based on user choice.

    ``temperature`` is driven to 0.0 by --seed so that full-system runs are as
    close to reproducible as the provider allows.
    """
    # Same shim as agents._resolve_llm: crewai's litellm path leaves the
    # cache_breakpoint marker on messages and Groq rejects it.
    from alloy_crew.agents import _CacheBreakpointSafeLLM as LLM

    if llm_choice == 'openai':
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")
        return LLM(model="gpt-4o-mini", api_key=api_key, temperature=temperature)
    elif llm_choice == 'groq':
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment")
        return LLM(model=LLM_ONLY_MODELS['groq'], api_key=api_key, temperature=temperature)
    else:
        return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='Generate AlloyGraph predictions')

    # Dataset selection
    parser.add_argument('--dataset', type=str, default='sss',
                        choices=['sss', 'precip', 'sc_ds', 'other', 'all',
                                 'holdout', 'matweb', 'custom'],
                        help='Dataset to evaluate (default: sss)')
    parser.add_argument('--custom-path', type=str, default=None,
                        help='Custom dataset path (requires --dataset custom)')

    # Filtering
    parser.add_argument('--num', type=int, default=None,
                        help='Number of alloys to evaluate (default: all)')
    parser.add_argument('--skip', type=int, default=0,
                        help='Skip first N alloys')
    parser.add_argument('--temp', type=int, default=None,
                        help='Evaluate only at specific temperature (e.g., 20)')

    # Mode selection (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--ml-only', action='store_true',
                            help='ML-only predictions (no physics caps, no agents)')
    mode_group.add_argument('--ml-deterministic', action='store_true',
                            help='ML + physics enforcement (UTS/YS caps, EM Reuss, EL caps) — no agents')
    mode_group.add_argument('--ml-physics-kg', action='store_true',
                            help='ML + physics + KG anchoring, no agents (needs Weaviate)')
    mode_group.add_argument('--llm-only', action='store_true',
                            help='LLM-only predictions (no ML model, no KG, no agents)')

    # LLM provider (for full system mode)
    parser.add_argument('--llm', type=str, default=None,
                        choices=['openai', 'groq', 'auto'],
                        help='LLM provider for full system mode (default: auto)')

    # Model selection (for --llm-only mode)
    parser.add_argument('--model', type=str, default=None,
                        help=f'Model for --llm-only mode. Aliases: {", ".join(LLM_ONLY_MODELS.keys())}. '
                             f'Or pass a raw litellm model string (e.g., openai/gpt-4.1). Default: {LLM_ONLY_DEFAULT}')

    # Reproducibility
    parser.add_argument('--concurrency', type=int, default=1,
                        help='Parallel evaluations for full-system mode (default 1). '
                             'Raise to shorten a campaign; all workers share one '
                             'rate-limit gate that closes on any 429.')
    parser.add_argument('--resume', action='store_true',
                        help='Skip alloy/temperature rows already present in the '
                             'output CSV and append the rest. Requires --output.')
    parser.add_argument('--rate-base-wait', type=float, default=30.0,
                        help='Initial backoff in seconds when a 429 is seen (doubles '
                             'per consecutive trip, capped at --rate-max-wait).')
    parser.add_argument('--rate-max-wait', type=float, default=300.0,
                        help='Ceiling on the 429 backoff window.')
    parser.add_argument('--seed', type=int, default=None,
                        help='Seed Python/numpy RNGs, drive LLM sampling temperature to 0.0, '
                             'and forward the seed to litellm where the provider supports it. '
                             'Recorded in the output filename and in a `seed` column.')

    # Output
    parser.add_argument('--output', type=str, default=None,
                        help='Output filename')
    parser.add_argument('--delay', type=float, default=None,
                        help='Delay between alloys in seconds (default: 30 for full, 5 for llm-only, 0 for ml-only/ml-deterministic)')

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _evaluate_row(item, method, args, llm_only_model_key, llm_sampling_temp, llm_config,
                  base_wait=30.0):
    """Run one alloy/temperature evaluation. Returns (row, error_dict)."""
    i = item["i"]
    alloy_data = item["alloy_data"]
    alloy_name = item["alloy_name"]
    composition = item["composition"]
    processing = item["processing"]
    temp = item["temp"]
    start_time = time.time()
    try:
        if method == 'ML_ONLY':
            eval_result = run_ml_only(composition, processing, int(temp))
        elif method == 'ML_DETERMINISTIC':
            eval_result = run_ml_deterministic(composition, processing, int(temp))
        elif method == 'ML_PHYSICS_KG':
            eval_result = run_ml_physics_kg(composition, processing, int(temp))
        elif method == 'LLM_ONLY':
            eval_result = run_llm_only(
                composition, processing, int(temp),
                model_key=llm_only_model_key,
                seed=args.seed,
                sampling_temperature=llm_sampling_temp,
            )
        else:
            eval_result = run_full_system(
                composition=composition,
                processing=processing,
                temperature=int(temp),
                base_wait=base_wait,
                llm_config=llm_config
            )

        elapsed = time.time() - start_time

        if eval_result.get('status') == 'FAIL':
            return None, {
                'alloy': alloy_name,
                'temperature': temp,
                'error': eval_result.get('error', 'Unknown'),
                'stage': eval_result.get('stage', 'unknown'),
            }

        # Extract results
        props = eval_result.get('properties', {})
        confidence = eval_result.get('confidence', {})
        actuals = get_actual_values(alloy_data, temp)

        row = {
            'alloy': alloy_name,
            'temperature': temp,
            'processing': processing,
            'method': method,
            'status': eval_result.get('status', 'UNKNOWN'),

            'pred_ys': props.get('Yield Strength'),
            'actual_ys': actuals.get('actual_ys'),
            'pred_uts': props.get('Tensile Strength'),
            'actual_uts': actuals.get('actual_uts'),
            'pred_el': props.get('Elongation'),
            'actual_el': actuals.get('actual_el'),
            'pred_em': props.get('Elastic Modulus'),
            'actual_em': actuals.get('actual_em'),

            'pred_density': props.get('Density'),
            'pred_gamma_prime': props.get('Gamma Prime'),
            'confidence_level': confidence.get('level', 'UNKNOWN'),
            'tcp_risk': eval_result.get('tcp_risk', 'N/A'),
            'corrections_applied': len(eval_result.get('corrections_applied', [])),
            'eval_time_sec': round(elapsed, 1),
            'seed': args.seed if args.seed is not None else '',
        }

        if method == 'FULL_SYSTEM':
            row['pipeline_stage'] = eval_result.get('pipeline_stage', 'unknown')
            row['envelope_overrides'] = sum(
                1 for c in (eval_result.get('corrections_applied') or [])
                if isinstance(c, dict)
                and c.get('physics_constraint') == 'evidence_envelope_override'
            )

        tu = eval_result.get('token_usage') or {}
        if tu:
            row['prompt_tokens'] = tu.get('prompt_tokens')
            row['completion_tokens'] = tu.get('completion_tokens')
            row['total_tokens'] = tu.get('total_tokens')
            row['llm_requests'] = tu.get('successful_requests')

        if method in ('ML_DETERMINISTIC', 'ML_PHYSICS_KG'):
            row['em_override_skipped_sc_ds'] = eval_result.get(
                'em_override_skipped_sc_ds', False)

        if method == 'ML_PHYSICS_KG':
            row.update({
                'kg_match_name': eval_result.get('kg_match_name'),
                'kg_match_distance': eval_result.get('kg_match_distance'),
                'kg_gate_allowed': eval_result.get('kg_gate_allowed'),
                'kg_reject_code': eval_result.get('kg_reject_code'),
                'kg_reject_detail': eval_result.get('kg_reject_detail'),
                'kg_weight': eval_result.get('kg_weight'),
            })

        return row, None
    except Exception as e:
        return None, {"alloy": alloy_name, "temperature": temp, "error": str(e)}

def main():
    args = parse_args()

    # RATE_GATE is a module-level singleton built before argparse runs, so the
    # tuning flags have to be applied here or they are silently inert.
    RATE_GATE.configure(base_wait=args.rate_base_wait, max_wait=args.rate_max_wait)

    # The default output name embeds a fresh timestamp, so it can never match a
    # previous run's file and --resume would quietly restart from zero.
    if args.resume and not args.output:
        raise SystemExit(
            "error: --resume requires --output. The default filename embeds a "
            "timestamp, so there is no prior file for resume to match."
        )

    # Reproducibility: seed before anything else touches an RNG
    if args.seed is not None:
        seed_everything(args.seed)

    # LLM sampling temperature: 0.0 when a seed is requested, else the defaults
    llm_sampling_temp = 0.0 if args.seed is not None else 0.3
    crew_sampling_temp = 0.0 if args.seed is not None else 0.1

    # Full-system agents resolve their own LLM unless --llm is passed, and that
    # path used to ignore crew_sampling_temp entirely -- a seeded run announced
    # 0.0 while actually sampling at the 0.1 default. Publish the temperature so
    # agents._resolve_llm honours it for whichever provider it selects.
    if args.seed is not None:
        os.environ["ALLOYGRAPH_LLM_TEMPERATURE"] = str(crew_sampling_temp)

    # Determine mode
    if args.ml_only:
        method = 'ML_ONLY'
    elif args.ml_deterministic:
        method = 'ML_DETERMINISTIC'
    elif args.ml_physics_kg:
        method = 'ML_PHYSICS_KG'
    elif args.llm_only:
        method = 'LLM_ONLY'
    else:
        method = 'FULL_SYSTEM'

    # Default delay per mode
    if args.delay is not None:
        delay = args.delay
    elif method in ('ML_ONLY', 'ML_DETERMINISTIC', 'ML_PHYSICS_KG'):
        delay = 0.0
    elif method == 'LLM_ONLY':
        delay = 3.0
    else:
        delay = 30.0

    # Resolve dataset path
    data_dir = os.path.join(BASE_DIR, 'data')
    preprocess_dir = os.path.join(BACKEND_DIR, 'superalloy_preprocess', 'output_data')

    dataset_paths = {
        'sss': os.path.join(data_dir, 'SSS.jsonl'),
        'precip': os.path.join(data_dir, 'precip.jsonl'),
        'sc_ds': os.path.join(data_dir, 'sc_ds.jsonl'),
        'other': os.path.join(data_dir, 'other.jsonl'),
        'all': os.path.join(data_dir, 'all_categorized.jsonl'),
        'holdout': os.path.join(preprocess_dir, 'evaluation_holdout_set.jsonl'),
        'matweb': os.path.join(preprocess_dir, 'matweb_alloys.jsonl'),
    }

    if args.dataset == 'custom':
        if not args.custom_path:
            print("Error: --custom-path required when using --dataset custom")
            return
        data_path = args.custom_path
    else:
        data_path = dataset_paths[args.dataset]

    if not os.path.exists(data_path):
        print(f"Error: Dataset not found at {data_path}")
        print(f"Available datasets in {data_dir}:")
        for f in sorted(os.listdir(data_dir)):
            print(f"  {f}")
        return

    # Load data
    print(f"\nLoading {os.path.basename(data_path)}...")
    with open(data_path, 'r') as f:
        all_alloys = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded {len(all_alloys)} alloys")

    # Apply filters
    alloys = all_alloys[args.skip:]
    if args.num:
        alloys = alloys[:args.num]

    # LLM config (full system only)
    llm_config = None
    llm_name = 'auto'
    if method == 'FULL_SYSTEM' and args.llm:
        llm_config = get_llm_config(args.llm, temperature=crew_sampling_temp)
        llm_name = args.llm

    # Resolve LLM-only model
    llm_only_model_key = args.model or LLM_ONLY_DEFAULT
    llm_only_model_name = LLM_ONLY_MODELS.get(llm_only_model_key, llm_only_model_key)

    print(f"Mode: {method}")
    if method == 'FULL_SYSTEM':
        print(f"LLM provider: {llm_name}")
    elif method == 'LLM_ONLY':
        print(f"LLM model: {llm_only_model_name}")
    print(f"Alloys: {len(alloys)}")
    print(f"Delay: {delay}s")
    if args.seed is not None:
        print(f"Seed: {args.seed} (LLM sampling temperature forced to {crew_sampling_temp})")
    else:
        print("Seed: none (run is not reproducible)")
    print("=" * 70)

    # Prepare output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(BASE_DIR, 'output')
    os.makedirs(output_dir, exist_ok=True)

    if args.output:
        output_file = os.path.join(output_dir, args.output)
    else:
        mode_suffix = method.lower()
        if method == 'LLM_ONLY':
            # Include model name so different runs are distinguishable
            model_tag = llm_only_model_key.replace("/", "_").replace(":", "_")
            mode_suffix = f"llm_only_{model_tag}"
        # Include the seed so repeated runs at different seeds do not collide
        seed_tag = f'_seed{args.seed}' if args.seed is not None else ''
        output_file = os.path.join(
            output_dir, f'predictions_{mode_suffix}{seed_tag}_{timestamp}.csv'
        )

    # Run evaluations
    results = []
    errors = []
    eval_times = []

    # Flatten to one work item per alloy/temperature row.
    work = []
    for i, alloy_data in enumerate(alloys):
        alloy_name = alloy_data.get('alloy', f'Alloy_{i}')
        composition = alloy_data.get('composition', {})
        processing = alloy_data.get('processing', 'cast')

        if not composition or sum(composition.values()) < 90:
            errors.append({'alloy': alloy_name, 'error': 'Invalid composition',
                           'composition_sum': sum(composition.values()) if composition else 0})
            print(f"[{i+1}/{len(alloys)}] {alloy_name}: SKIP (invalid composition)")
            continue

        temps = get_all_temperatures(alloy_data)
        if args.temp is not None:
            temps = [t for t in temps if abs(t - args.temp) < 5]
        if not temps:
            errors.append({'alloy': alloy_name, 'error': 'No matching temperatures'})
            continue

        for temp in temps:
            work.append({'i': i, 'alloy_data': alloy_data, 'alloy_name': alloy_name,
                         'composition': composition, 'processing': processing, 'temp': temp})

    # Resume: drop rows already present in the output file.
    if args.resume and os.path.exists(output_file):
        prior = pd.read_csv(output_file)
        done = {(str(r.alloy), round(float(r.temperature), 1))
                for r in prior.itertuples() if pd.notna(r.temperature)}
        before = len(work)
        work = [w for w in work if (str(w['alloy_name']), round(float(w['temp']), 1)) not in done]
        results.extend(prior.to_dict('records'))
        print(f"Resume: {len(done)} rows already in {os.path.basename(output_file)}, "
              f"{before - len(work)} skipped, {len(work)} remaining")
    elif args.resume:
        print(f"Resume: no existing {os.path.basename(output_file)}, starting fresh")

    total = len(work)
    print(f"Rows to evaluate: {total}\n")

    write_lock = threading.Lock()
    completed = [0]

    def _record(row, err):
        with write_lock:
            completed[0] += 1
            n = completed[0]
            if err:
                errors.append(err)
                print(f"  [{n}/{total}] EXCEPTION {err['alloy']} @ {err['temperature']}C: "
                      f"{str(err['error'])[:70]}")
            else:
                results.append(row)
                if isinstance(row.get('eval_time_sec'), (int, float)):
                    eval_times.append(row['eval_time_sec'])
                ys = row.get('pred_ys')
                ys_str = f"{ys:.0f}" if isinstance(ys, (int, float)) else "N/A"
                print(f"  [{n}/{total}] {row['alloy'][:34]} @ {row['temperature']}C "
                      f"| {row.get('eval_time_sec')}s | YS {ys_str} "
                      f"(actual {row.get('actual_ys', 'N/A')})")
            # Checkpoint after every row so a crash never loses completed work.
            if results:
                pd.DataFrame(results).to_csv(output_file, index=False)

    concurrency = max(1, int(args.concurrency))
    if concurrency > 1 and method != 'FULL_SYSTEM':
        print(f"NOTE: --concurrency only applies to full-system mode; running sequentially")
        concurrency = 1

    if concurrency == 1:
        for item in work:
            row, err = _evaluate_row(item, method, args, llm_only_model_key,
                                     llm_sampling_temp, llm_config, base_wait=delay)
            _record(row, err)
            if method == 'FULL_SYSTEM':
                reset_crewai_state()
            if delay > 0:
                time.sleep(delay)
    else:
        print(f"Concurrency: {concurrency} workers, shared rate-limit gate")
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {pool.submit(_evaluate_row, item, method, args,
                                   llm_only_model_key, llm_sampling_temp, llm_config,
                                   delay): item
                       for item in work}
            for fut in as_completed(futures):
                item = futures[fut]
                try:
                    row, err = fut.result()
                except Exception as e:
                    row, err = None, {'alloy': item['alloy_name'],
                                      'temperature': item['temp'], 'error': str(e)}
                _record(row, err)
        # Global bus cleanup once, on the main thread, after the pool drains.
        if method == 'FULL_SYSTEM':
            reset_crewai_state()


    # Final summary
    print("\n" + "=" * 70)
    print(f"PREDICTION GENERATION COMPLETE ({method})")
    print("=" * 70)
    print(f"\nResults: {len(results)} predictions")
    print(f"Errors: {len(errors)} failures")

    if eval_times:
        print(f"Total time: {sum(eval_times)/60:.1f} minutes")
        print(f"Avg per evaluation: {np.mean(eval_times):.1f}s")

    if results:
        df = pd.DataFrame(results)
        df.to_csv(output_file, index=False)
        print(f"\nSaved: {output_file}")

        # Quick accuracy check
        print("\n" + "-" * 40)
        for pred_col, actual_col, label in [
            ('pred_ys', 'actual_ys', 'YS'),
            ('pred_uts', 'actual_uts', 'UTS'),
            ('pred_el', 'actual_el', 'EL'),
            ('pred_em', 'actual_em', 'EM'),
        ]:
            valid = df[[pred_col, actual_col]].dropna()
            valid = valid[valid[actual_col] != 0]
            if len(valid) > 0:
                mape = (valid[pred_col] - valid[actual_col]).abs().div(valid[actual_col]).mean() * 100
                bias = (valid[pred_col] - valid[actual_col]).div(valid[actual_col]).mean() * 100
                print(f"  {label:4} | n={len(valid):3} | MAPE: {mape:5.1f}% | Bias: {bias:+5.1f}%")

    if errors:
        errors_file = output_file.replace('.csv', '_errors.csv')
        pd.DataFrame(errors).to_csv(errors_file, index=False)
        print(f"Errors: {errors_file}")

    return output_file


if __name__ == '__main__':
    main()
