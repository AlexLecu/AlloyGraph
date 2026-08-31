#!/usr/bin/env python3
"""
Create Clean Evaluation Dataset

This script curates a high-quality evaluation dataset from available JSONL files,
applying rigorous filtering criteria to ensure methodologically sound evaluation.

Criteria:
1. Ni-base superalloys only (Ni > 45%)
2. Complete room temperature data (YS, UTS, Elongation at 20°C)
3. Valid composition (sum ~100%)
4. Known alloy classification (cast/wrought)
"""

import json
import os
from collections import defaultdict
from datetime import datetime

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # prediction/
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))  # AlloyGraph/
DATA_DIR = os.path.join(PROJECT_ROOT, 'backend', 'superalloy_preprocess', 'output_data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'data')


# Classification rules for alloy types
def classify_alloy(composition: dict, alloy_name: str) -> dict:
    """
    Classify alloy by strengthening mechanism and type.

    Returns dict with:
        - is_superalloy: bool
        - strengthening: 'precipitation' | 'solid_solution' | 'unknown'
        - precipitate_type: 'gamma_prime' | 'gamma_double_prime' | 'mixed' | None
        - exclusion_reason: str or None
    """
    ni = composition.get('Ni', 0)
    cr = composition.get('Cr', 0)
    al = composition.get('Al', 0)
    ti = composition.get('Ti', 0)
    nb = composition.get('Nb', 0)
    mo = composition.get('Mo', 0)
    w = composition.get('W', 0)
    co = composition.get('Co', 0)
    fe = composition.get('Fe', 0)
    cu = composition.get('Cu', 0)

    result = {
        'is_superalloy': True,
        'strengthening': 'unknown',
        'precipitate_type': None,
        'exclusion_reason': None,
        'alloy_class': 'standard'
    }

    # Check if it's a Ni-base alloy
    if ni < 40:
        result['is_superalloy'] = False
        result['exclusion_reason'] = f'Low Ni content ({ni:.1f}%), not Ni-base'
        return result

    # Check for pure/near-pure nickel
    if ni > 95:
        result['is_superalloy'] = False
        result['exclusion_reason'] = f'Pure nickel ({ni:.1f}%), not a superalloy'
        result['alloy_class'] = 'pure_nickel'
        return result

    # Check for Ni-Cu alloys (Monel type)
    if cu > 25:
        result['is_superalloy'] = False
        result['exclusion_reason'] = f'Ni-Cu alloy ({cu:.1f}% Cu), Monel-type'
        result['alloy_class'] = 'nickel_copper'
        return result

    # Check for Fe-Ni alloys (Incoloy type)
    if fe > 20 and ni < 55:
        result['is_superalloy'] = False
        result['exclusion_reason'] = f'Fe-Ni alloy ({fe:.1f}% Fe, {ni:.1f}% Ni)'
        result['alloy_class'] = 'iron_nickel'
        return result

    # Determine strengthening mechanism
    al_ti = al + ti

    # γ'' (gamma double prime) dominated - IN718 type
    if nb >= 3.0 and al_ti < 3.0 and (nb / (al_ti + 0.01)) >= 1.0:
        result['strengthening'] = 'precipitation'
        result['precipitate_type'] = 'gamma_double_prime'
        result['alloy_class'] = 'in718_type'

    # γ' (gamma prime) dominated - standard superalloy
    elif al_ti >= 2.0:
        result['strengthening'] = 'precipitation'
        result['precipitate_type'] = 'gamma_prime'
        result['alloy_class'] = 'gamma_prime_strengthened'

    # Solid solution strengthened (Hastelloy type)
    elif mo + w > 10 and al_ti < 1.5:
        result['strengthening'] = 'solid_solution'
        result['precipitate_type'] = None
        result['alloy_class'] = 'solid_solution'

    # Low Al+Ti but some precipitation possible
    elif al_ti >= 0.5:
        result['strengthening'] = 'precipitation'
        result['precipitate_type'] = 'gamma_prime'
        result['alloy_class'] = 'low_gamma_prime'

    else:
        result['strengthening'] = 'solid_solution'
        result['alloy_class'] = 'solid_solution'

    return result


def validate_composition(composition: dict) -> tuple[bool, str]:
    """Validate that composition sums to ~100% and has required elements."""
    total = sum(composition.values())

    if total < 90:
        return False, f'Composition sum too low: {total:.1f}%'
    if total > 110:
        return False, f'Composition sum too high: {total:.1f}%'

    if 'Ni' not in composition:
        return False, 'Missing Ni content'

    return True, ''


def has_room_temp_data(alloy_data: dict) -> tuple[bool, dict]:
    """
    Check if alloy has complete room temperature (20°C) mechanical properties.

    Returns:
        (has_data: bool, available_props: dict)
    """
    props_found = {}

    for prop_name in ['yield_strength', 'uts', 'elongation']:
        entries = alloy_data.get(prop_name, [])
        for entry in entries:
            temp = float(entry.get('temp_c', 999))
            if 15 <= temp <= 30:  # Room temperature range
                props_found[prop_name] = entry.get('value')
                break

    # Require at least YS and UTS
    has_ys = 'yield_strength' in props_found
    has_uts = 'uts' in props_found

    return (has_ys and has_uts), props_found


def load_alloys(filepath: str) -> list:
    """Load alloys from JSONL file."""
    alloys = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                alloys.append(json.loads(line))
    return alloys


def create_clean_dataset():
    """Main function to create curated evaluation dataset."""

    print("=" * 70)
    print("CREATING CLEAN EVALUATION DATASET")
    print("=" * 70)

    # Load all available datasets
    datasets = {
        'holdout': os.path.join(DATA_DIR, 'evaluation_holdout_set.jsonl'),
        'matweb': os.path.join(DATA_DIR, 'matweb_alloys.jsonl'),
        'original': os.path.join(DATA_DIR, 'matweb_original_alloys.jsonl'),
    }

    all_alloys = []
    seen_names = set()

    # Load holdout first (priority)
    for ds_name, ds_path in datasets.items():
        if os.path.exists(ds_path):
            alloys = load_alloys(ds_path)
            print(f"\n[{ds_name}] Loaded {len(alloys)} alloys from {os.path.basename(ds_path)}")

            for alloy in alloys:
                name = alloy.get('alloy', 'Unknown')
                if name not in seen_names:
                    alloy['_source'] = ds_name
                    all_alloys.append(alloy)
                    seen_names.add(name)

    print(f"\nTotal unique alloys: {len(all_alloys)}")

    # Quality report
    report = {
        'timestamp': datetime.now().isoformat(),
        'total_alloys': len(all_alloys),
        'included': [],
        'excluded': [],
        'statistics': defaultdict(int)
    }

    clean_alloys = []

    for alloy in all_alloys:
        name = alloy.get('alloy', 'Unknown')
        composition = alloy.get('composition', {})
        processing = alloy.get('processing', 'unknown')
        source = alloy.get('_source', 'unknown')

        # Validate composition
        valid_comp, comp_reason = validate_composition(composition)
        if not valid_comp:
            report['excluded'].append({
                'alloy': name,
                'reason': comp_reason,
                'source': source
            })
            report['statistics']['invalid_composition'] += 1
            continue

        # Classify alloy
        classification = classify_alloy(composition, name)

        if not classification['is_superalloy']:
            report['excluded'].append({
                'alloy': name,
                'reason': classification['exclusion_reason'],
                'source': source,
                'class': classification['alloy_class']
            })
            report['statistics'][f'excluded_{classification["alloy_class"]}'] += 1
            continue

        # Check room temperature data
        has_data, props = has_room_temp_data(alloy)
        if not has_data:
            report['excluded'].append({
                'alloy': name,
                'reason': f'Missing RT data. Found: {list(props.keys())}',
                'source': source
            })
            report['statistics']['missing_data'] += 1
            continue

        # Add classification info to alloy
        alloy['_classification'] = classification
        alloy['_rt_properties'] = props

        clean_alloys.append(alloy)
        report['included'].append({
            'alloy': name,
            'processing': processing,
            'source': source,
            'strengthening': classification['strengthening'],
            'precipitate_type': classification['precipitate_type'],
            'alloy_class': classification['alloy_class'],
            'rt_ys': props.get('yield_strength'),
            'rt_uts': props.get('uts'),
            'rt_el': props.get('elongation')
        })
        report['statistics'][f'included_{classification["alloy_class"]}'] += 1
        report['statistics'][f'processing_{processing}'] += 1

    # Summary
    print("\n" + "=" * 70)
    print("CURATION RESULTS")
    print("=" * 70)
    print(f"\nIncluded: {len(clean_alloys)} alloys")
    print(f"Excluded: {len(report['excluded'])} alloys")

    print("\nBy alloy class:")
    for key, count in sorted(report['statistics'].items()):
        if key.startswith('included_') or key.startswith('excluded_'):
            print(f"  {key}: {count}")

    print("\nBy processing:")
    for key, count in sorted(report['statistics'].items()):
        if key.startswith('processing_'):
            print(f"  {key}: {count}")

    # Save clean dataset
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    output_path = os.path.join(OUTPUT_DIR, 'evaluation_clean.jsonl')
    with open(output_path, 'w') as f:
        for alloy in clean_alloys:
            # Remove internal fields before saving
            alloy_clean = {k: v for k, v in alloy.items() if not k.startswith('_')}
            f.write(json.dumps(alloy_clean) + '\n')

    print(f"\nSaved: {output_path}")

    # Save quality report
    report_path = os.path.join(OUTPUT_DIR, 'data_quality_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    print(f"Saved: {report_path}")

    # Print exclusion reasons
    print("\n" + "=" * 70)
    print("EXCLUDED ALLOYS")
    print("=" * 70)
    for item in report['excluded']:
        print(f"  {item['alloy']}: {item['reason']}")

    return clean_alloys, report


if __name__ == '__main__':
    clean_alloys, report = create_clean_dataset()
