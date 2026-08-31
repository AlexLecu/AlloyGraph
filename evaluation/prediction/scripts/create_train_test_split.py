#!/usr/bin/env python3
"""
Create Proper Train/Test Split from Training Data

This script creates a stratified train/test split from the enriched training data,
ensuring balanced representation of processing types and alloy categories.

Output:
- training_set.jsonl (80%) - For ML training
- evaluation_interpolation_set.jsonl (20%) - For evaluation (same distribution as training)
"""

import json
import os
import random
from collections import defaultdict
from datetime import datetime

# Set seed for reproducibility
SEED = 42
random.seed(SEED)

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)  # prediction/
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))  # AlloyGraph/
TRAINING_DATA_DIR = os.path.join(PROJECT_ROOT, 'backend', 'alloy_crew', 'models', 'training_data')
TRAINING_DATA = os.path.join(TRAINING_DATA_DIR, 'final_alloy_data_enriched.jsonl')
OUTPUT_DIR = TRAINING_DATA_DIR  # Output to same folder as source data


def classify_alloy(alloy_data: dict) -> str:
    """Classify alloy by strengthening mechanism."""
    comp = alloy_data.get('composition', {})
    al = comp.get('Al', 0)
    ti = comp.get('Ti', 0)
    nb = comp.get('Nb', 0)
    mo = comp.get('Mo', 0)
    w = comp.get('W', 0)

    al_ti = al + ti

    # Single crystal / DS alloys (high Al+Ti, high refractory)
    if al_ti > 7:
        return 'single_crystal_type'

    # γ'' dominated (IN718 type)
    if nb >= 3.0 and al_ti < 3.0:
        return 'gamma_double_prime'

    # Standard γ' strengthened
    if al_ti >= 3.0:
        return 'gamma_prime_high'
    elif al_ti >= 1.5:
        return 'gamma_prime_medium'

    # Solid solution
    if mo + w > 10:
        return 'solid_solution'

    return 'other'


def has_room_temp_ys(alloy_data: dict) -> bool:
    """Check if alloy has room temperature YS data."""
    for entry in alloy_data.get('yield_strength', []):
        temp = float(entry.get('temp_c', 999))
        if 15 <= temp <= 30:
            return True
    return False


def create_stratified_split(alloys: list, test_ratio: float = 0.2) -> tuple:
    """
    Create stratified train/test split.

    Stratify by:
    1. Processing type (cast/wrought)
    2. Alloy class (γ'/γ''/solid solution)
    """
    # Group alloys by strata
    strata = defaultdict(list)

    for alloy in alloys:
        processing = alloy.get('processing', 'unknown')
        alloy_class = classify_alloy(alloy)
        key = f"{processing}_{alloy_class}"
        strata[key].append(alloy)

    train_set = []
    test_set = []

    print("Stratification:")
    for stratum, alloy_list in sorted(strata.items()):
        random.shuffle(alloy_list)
        n_test = max(1, int(len(alloy_list) * test_ratio))

        test_set.extend(alloy_list[:n_test])
        train_set.extend(alloy_list[n_test:])

        print(f"  {stratum}: {len(alloy_list)} total, {n_test} test, {len(alloy_list) - n_test} train")

    return train_set, test_set


def main():
    print("=" * 70)
    print("CREATING STRATIFIED TRAIN/TEST SPLIT")
    print("=" * 70)
    print(f"Random seed: {SEED}")

    # Load training data
    print(f"\nLoading: {TRAINING_DATA}")
    with open(TRAINING_DATA, 'r') as f:
        alloys = [json.loads(line) for line in f if line.strip()]

    print(f"Total alloys: {len(alloys)}")

    # Include ALL alloys - evaluate at whatever temperatures they have data
    # (Don't filter by RT data - evaluate at available temperatures)
    alloys_with_ys = [a for a in alloys if a.get('yield_strength')]  # Has any YS data
    alloys_no_ys = [a for a in alloys if not a.get('yield_strength')]  # No YS at all

    print(f"With any YS data: {len(alloys_with_ys)}")
    print(f"Without YS data: {len(alloys_no_ys)}")

    if alloys_no_ys:
        print("  (Alloys without YS will still be included in training for KG)")

    # Create stratified split from alloys with YS data
    print()
    train_set, test_set = create_stratified_split(alloys_with_ys, test_ratio=0.3)

    # Add alloys without YS to training set (useful for KG, can't evaluate)
    train_set.extend(alloys_no_ys)

    print(f"\nSplit results:")
    print(f"  Training: {len(train_set)} alloys")
    print(f"  Test: {len(test_set)} alloys")

    # Validate split
    train_cast = sum(1 for a in train_set if a.get('processing') == 'cast')
    test_cast = sum(1 for a in test_set if a.get('processing') == 'cast')
    print(f"\nProcessing balance:")
    print(f"  Train: {train_cast} cast, {len(train_set) - train_cast} wrought")
    print(f"  Test: {test_cast} cast, {len(test_set) - test_cast} wrought")

    # Save splits
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_path = os.path.join(OUTPUT_DIR, 'train_split_70.jsonl')
    with open(train_path, 'w') as f:
        for alloy in train_set:
            f.write(json.dumps(alloy) + '\n')
    print(f"\nSaved: {train_path}")

    test_path = os.path.join(OUTPUT_DIR, 'test_split_30.jsonl')
    with open(test_path, 'w') as f:
        for alloy in test_set:
            f.write(json.dumps(alloy) + '\n')
    print(f"Saved: {test_path}")

    # Print test set alloys
    print("\n" + "=" * 70)
    print("TEST SET ALLOYS (for evaluation)")
    print("=" * 70)
    for alloy in sorted(test_set, key=lambda x: x['alloy']):
        name = alloy['alloy']
        processing = alloy.get('processing', '?')
        alloy_class = classify_alloy(alloy)

        # Get RT YS
        rt_ys = None
        for entry in alloy.get('yield_strength', []):
            if 15 <= float(entry.get('temp_c', 999)) <= 30:
                rt_ys = entry['value']
                break

        print(f"  {name:30} | {processing:7} | {alloy_class:20} | YS={rt_ys}")

    # Create summary
    summary = {
        'created': datetime.now().isoformat(),
        'seed': SEED,
        'total_alloys': len(alloys),
        'alloys_with_rt_ys': len(alloys_with_ys),
        'train_size': len(train_set),
        'test_size': len(test_set),
        'train_cast': train_cast,
        'test_cast': test_cast,
        'test_alloys': [a['alloy'] for a in test_set]
    }

    summary_path = os.path.join(OUTPUT_DIR, 'split_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {summary_path}")

    return train_set, test_set


if __name__ == '__main__':
    main()
