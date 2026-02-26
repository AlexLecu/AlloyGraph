#!/usr/bin/env python3
"""
AlloyGraph Prediction Generator

Generates predictions for evaluation with multiple ablation modes:
  --ml-only            : Raw ML model output (no agents, no physics enforcement)
  --ml-deterministic   : ML + physics enforcement (UTS/YS caps, EM Reuss, EL caps) — no agents
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
import re
import gc
from datetime import datetime

# Disable telemetry before importing crewai
os.environ['CREWAI_TELEMETRY_OPT_OUT'] = 'true'
os.environ['OTEL_SDK_DISABLED'] = 'true'
os.environ['LITELLM_LOG'] = 'ERROR'

# Suppress noisy logs
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
# CrewAI state management
# ---------------------------------------------------------------------------

def reset_crewai_state():
    """Reset CrewAI event context and bus state to prevent stack overflow.

    The main issue is the _event_id_stack ContextVar in crewai.events.event_context
    which accumulates when crew executions fail without emitting ending events.
    """
    try:
        from crewai.events import event_context
        event_context._event_id_stack.set(())
        event_context._last_event_id.set(None)
        event_context._triggering_event_id.set(None)
    except (ImportError, AttributeError) as e:
        print(f"  Warning: Could not reset event_context: {e}")

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
            reset_crewai_state()
            evaluator = AlloyEvaluationCrew(llm_config=llm_config)
            result = evaluator.run(
                composition=composition,
                processing=processing,
                temperature=temperature
            )
            del evaluator
            gc.collect()
            return result

        except Exception as e:
            error_str = str(e).lower()
            is_rate_limit = any(x in error_str for x in [
                'rate', 'limit', '429', 'quota', 'too many', 'throttl'
            ])
            is_event_stack = 'event stack' in error_str or 'depth limit' in error_str

            if is_rate_limit:
                wait_time = base_wait * (attempt + 1)
                print(f"  Rate limit, waiting {wait_time}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
            elif is_event_stack:
                print(f"  Event stack overflow, performing deep reset...")
                reset_crewai_state()
                gc.collect()
                time.sleep(2)
                reset_crewai_state()
                time.sleep(3)
            else:
                if attempt < max_retries - 1:
                    print(f"  Error: {str(e)[:80]}... retrying in 10s")
                    time.sleep(10)
                else:
                    raise

    # Final attempt
    reset_crewai_state()
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


def run_ml_deterministic(composition, processing, temperature):
    """ML predictions + deterministic physics enforcement (no LLM agents).

    Applies the same physics caps as the full evaluator:
    - Density & gamma prime from composition (not ML)
    - UTS >= YS * 1.05 floor
    - UTS/YS ratio ceiling (processing & gamma-prime aware)
    - Elongation caps for high gamma-prime alloys
    - EM override if >20% from Reuss bound
    - compute_metallurgy_validation for TCP risk & penalties
    """
    from alloy_crew.models.predictor import AlloyPredictor
    from alloy_crew.models.feature_engineering import (
        compute_alloy_features, calculate_em_rule_of_mixtures
    )
    from alloy_crew.config.alloy_parameters import (
        is_sss_alloy, get_em_temp_factor, UTS_YS_RATIO, ELONGATION
    )
    from alloy_crew.tools.metallurgy_tools import compute_metallurgy_validation

    # --- Step 1: Raw ML predictions (same as run_ml_only) ---
    predictor = AlloyPredictor.get_shared_predictor()
    result_df = predictor.predict(
        composition,
        extra_params={'processing': processing},
        temperatures=[temperature]
    )

    if result_df.empty:
        return {'status': 'FAIL', 'error': 'Empty prediction result'}

    row = result_df.iloc[0]

    # Composition-determined density & gamma prime
    features = compute_alloy_features(composition)
    density = round(features.get("density_calculated_gcm3", 0), 2)
    gp = 0.0 if is_sss_alloy(composition) else round(features.get("gamma_prime_estimated_vol_pct", 0), 1)

    props = {
        'Yield Strength': float(row.get('ys', 0)) if 'ys' in row else None,
        'Tensile Strength': float(row.get('uts', 0)) if 'uts' in row else None,
        'Elongation': float(row.get('el', 0)) if 'el' in row else None,
        'Elastic Modulus': float(row.get('em', 0)) if 'em' in row else None,
        'Density': density,
        'Gamma Prime': gp,
    }

    # --- Step 2: Physics enforcement (mirrors alloy_evaluator.py) ---
    ys_val = props['Yield Strength']
    uts_val = props['Tensile Strength']

    # UTS floor: must be >= YS * 1.05
    if (isinstance(ys_val, (int, float)) and isinstance(uts_val, (int, float))
            and ys_val > 0 and uts_val < ys_val):
        props['Tensile Strength'] = round(ys_val * 1.05, 1)
        uts_val = props['Tensile Strength']

    # UTS/YS ratio ceiling
    if isinstance(ys_val, (int, float)) and isinstance(uts_val, (int, float)) and ys_val > 0:
        ratio = uts_val / ys_val
        if is_sss_alloy(composition):
            max_ratio = 2.4
        elif processing in ["wrought", "forged"] and gp > 40:
            max_ratio = UTS_YS_RATIO["WROUGHT_HIGH_GP_MAX"]
        elif processing in ["wrought", "forged"]:
            max_ratio = UTS_YS_RATIO["WROUGHT_MAX"]
        else:
            max_ratio = UTS_YS_RATIO["CAST_BASE"] + (gp / 100) * UTS_YS_RATIO["CAST_GP_FACTOR"] + 0.10
        if ratio > max_ratio:
            props['Tensile Strength'] = round(ys_val * max_ratio, 1)

    # Elongation caps
    el_val = props.get('Elongation')
    if isinstance(el_val, (int, float)) and el_val > 0:
        if gp > 60 and el_val > ELONGATION["HIGH_GP_MAX_EL"]:
            props['Elongation'] = ELONGATION["HIGH_GP_MAX_EL"]
        elif gp > 40 and el_val > ELONGATION["MOD_GP_MAX_EL"]:
            props['Elongation'] = ELONGATION["MOD_GP_MAX_EL"]

    # EM Reuss bound enforcement (override if >20% deviation)
    em_val = props.get('Elastic Modulus')
    if isinstance(em_val, (int, float)) and em_val > 0:
        em_rt = calculate_em_rule_of_mixtures(composition)
        em_temp_factor = get_em_temp_factor(temperature)
        em_physics = round(em_rt * em_temp_factor, 1)
        if em_physics > 0:
            em_deviation = abs(em_val - em_physics) / em_physics
            if em_deviation > 0.20:
                props['Elastic Modulus'] = em_physics

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
    }


LLM_ONLY_MODELS = {
    "groq": "groq/llama-3.3-70b-versatile",
    "openai-mini": "openai/gpt-4.1-mini",
    "openai-ft": "openai/ft:gpt-4.1-mini-2025-04-14:digital-science-dimensions::CoGgLDPB",
}
LLM_ONLY_DEFAULT = "groq"


LLM_SYSTEM_PROMPT = (
    "You are an expert materials scientist specializing in nickel-based superalloys. "
    "Predict mechanical properties accurately based on composition, processing, and temperature. "
    "Reason briefly, then answer as JSON."
)


def run_llm_only(composition, processing, temperature, model_key=None, max_retries=3):
    """Run LLM-only prediction: prompt an LLM with composition, no ML/KG/agents."""
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

    for attempt in range(max_retries):
        try:
            response = llm_completion(
                model=model_name,
                messages=[
                    {"role": "system", "content": LLM_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=512,
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
                    'Yield Strength': find_val(parsed, 'yield', 'ys'),
                    'Tensile Strength': find_val(parsed, 'uts', 'tensile', 'ultimate'),
                    'Elongation': find_val(parsed, 'elong', 'el'),
                    'Elastic Modulus': find_val(parsed, 'elastic', 'modulus', 'em'),
                }
            else:
                # Fallback: extract numbers in order
                numbers = re.findall(r'[\d.]+', content)
                if len(numbers) >= 4:
                    preds = {
                        'Yield Strength': float(numbers[0]),
                        'Tensile Strength': float(numbers[1]),
                        'Elongation': float(numbers[2]),
                        'Elastic Modulus': float(numbers[3]),
                    }
                else:
                    raise ValueError(f"Could not parse LLM response: {content[:200]}")

            return {
                'properties': preds,
                'confidence': {'level': 'LOW', 'score': 0.3},
                'status': 'SUCCESS',
            }

        except Exception as e:
            error_str = str(e).lower()
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


def get_llm_config(llm_choice):
    """Get LLM configuration based on user choice."""
    from crewai import LLM

    if llm_choice == 'openai':
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")
        return LLM(model="gpt-4o-mini", api_key=api_key, temperature=0.1)
    elif llm_choice == 'groq':
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment")
        return LLM(model=LLM_ONLY_MODEL, api_key=api_key, temperature=0.1)
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

    # Output
    parser.add_argument('--output', type=str, default=None,
                        help='Output filename')
    parser.add_argument('--delay', type=float, default=None,
                        help='Delay between alloys in seconds (default: 30 for full, 5 for llm-only, 0 for ml-only/ml-deterministic)')

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Determine mode
    if args.ml_only:
        method = 'ML_ONLY'
    elif args.ml_deterministic:
        method = 'ML_DETERMINISTIC'
    elif args.llm_only:
        method = 'LLM_ONLY'
    else:
        method = 'FULL_SYSTEM'

    # Default delay per mode
    if args.delay is not None:
        delay = args.delay
    elif method in ('ML_ONLY', 'ML_DETERMINISTIC'):
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
        llm_config = get_llm_config(args.llm)
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
        output_file = os.path.join(output_dir, f'predictions_{mode_suffix}_{timestamp}.csv')

    # Run evaluations
    results = []
    errors = []
    eval_times = []

    for i, alloy_data in enumerate(alloys):
        alloy_name = alloy_data.get('alloy', f'Alloy_{i}')
        composition = alloy_data.get('composition', {})
        processing = alloy_data.get('processing', 'cast')

        # Validate composition
        if not composition or sum(composition.values()) < 90:
            errors.append({
                'alloy': alloy_name,
                'error': 'Invalid composition',
                'composition_sum': sum(composition.values()) if composition else 0
            })
            print(f"[{i+1}/{len(alloys)}] {alloy_name}: SKIP (invalid composition)")
            continue

        # Get temperatures
        temps = get_all_temperatures(alloy_data)
        if args.temp is not None:
            temps = [t for t in temps if abs(t - args.temp) < 5]

        if not temps:
            errors.append({'alloy': alloy_name, 'error': 'No matching temperatures'})
            continue

        # Evaluate at each temperature
        for temp in temps:
            print(f"\n[{i+1}/{len(alloys)}] {alloy_name} @ {temp} C ({processing}) [{method}]")

            start_time = time.time()

            try:
                if method == 'ML_ONLY':
                    eval_result = run_ml_only(composition, processing, int(temp))
                elif method == 'ML_DETERMINISTIC':
                    eval_result = run_ml_deterministic(composition, processing, int(temp))
                elif method == 'LLM_ONLY':
                    eval_result = run_llm_only(composition, processing, int(temp), model_key=llm_only_model_key)
                else:
                    eval_result = run_full_system(
                        composition=composition,
                        processing=processing,
                        temperature=int(temp),
                        base_wait=delay,
                        llm_config=llm_config
                    )

                elapsed = time.time() - start_time
                eval_times.append(elapsed)

                if eval_result.get('status') == 'FAIL':
                    errors.append({
                        'alloy': alloy_name,
                        'temperature': temp,
                        'error': eval_result.get('error', 'Unknown'),
                        'stage': eval_result.get('stage', 'unknown')
                    })
                    print(f"  FAILED: {eval_result.get('error')}")
                    continue

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
                }

                results.append(row)

                ys_pred = props.get('Yield Strength')
                ys_str = f"{ys_pred:.0f}" if isinstance(ys_pred, (int, float)) else "N/A"
                ys_actual = actuals.get('actual_ys', 'N/A')
                print(f"  Done in {elapsed:.1f}s | YS: {ys_str} (actual: {ys_actual})")

            except Exception as e:
                elapsed = time.time() - start_time
                errors.append({
                    'alloy': alloy_name,
                    'temperature': temp,
                    'error': str(e)
                })
                print(f"  EXCEPTION: {e}")

        # Save intermediate results
        if results:
            pd.DataFrame(results).to_csv(output_file, index=False)

        # Reset state (full system only) and delay
        if method == 'FULL_SYSTEM':
            reset_crewai_state()
        if delay > 0 and i < len(alloys) - 1:
            time.sleep(delay)

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
