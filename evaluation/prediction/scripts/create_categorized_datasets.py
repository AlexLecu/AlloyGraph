#!/usr/bin/env python3
"""
Create categorized evaluation datasets.

Categories:
1. precip_poly - Precipitation-hardened polycrystalline (primary target)
2. precip_sc_ds - Precipitation-hardened single crystal/DS
3. solid_solution - Solid-solution strengthened
4. other - Pure Ni, Monel, Co-base, Fe-Ni base

Outputs to: evaluation_v2/data_clean/
"""

import json
import os
from collections import defaultdict
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)  # prediction/
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))  # AlloyGraph/
OUTPUT_DIR = os.path.join(BASE_DIR, "data_clean")

# Source datasets
SOURCES = {
    "test_split_30": os.path.join(PROJECT_ROOT, "backend", "alloy_crew", "models", "training_data", "test_split_30.jsonl"),
    "matweb_original": os.path.join(PROJECT_ROOT, "backend", "superalloy_preprocess", "output_data", "matweb_original_alloys.jsonl"),
    "matweb_alloys": os.path.join(PROJECT_ROOT, "backend", "superalloy_preprocess", "output_data", "matweb_alloys.jsonl"),
}

#: Declared scope exclusions. Every record a source offers and the evaluation
#: set does not take is named in this file, with the rule that removes it or --
#: where no rule can -- an explicit declaration. Rules R1/R2/R4 are applied here;
#: entries marked "declared" or "historical" are matched by alloy name because
#: no composition test reaches them without removing alloys that are kept.
#: See docs/data_curation.md.
SCOPE_EXCLUSIONS = os.path.join(BASE_DIR, "data", "scope_exclusions.json")

#: R2. Above this the strengthening phase is bulk beta-NiAl, not gamma-prime.
NIAL_AL_WT = 7.0

#: R4. Composition distance below which two records are the same alloy.
DUPLICATE_D = 0.01


def load_scope_exclusions(path=SCOPE_EXCLUSIONS):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _distance(a, b):
    """Euclidean distance on wt%-normalised composition, as rag_tools computes it."""
    keys = set(a) | set(b)
    return sum((float(a.get(k) or 0) - float(b.get(k) or 0)) ** 2 for k in keys) ** 0.5


def scope_verdict(alloy_data, accepted, by_name):
    """(excluded, entry_id, reason) for one record.

    Rules first, so a record a rule reaches is reported as reached by the rule
    rather than by its declaration.
    """
    comp = alloy_data.get("composition") or {}
    name = alloy_data.get("alloy", "")

    if not comp:
        return True, by_name.get(name, {}).get("id"), "R1 unfeaturisable: composition is empty"
    if float(comp.get("Al") or 0) >= NIAL_AL_WT:
        return True, by_name.get(name, {}).get("id"), (
            f"R2 NiAl intermetallic: Al = {float(comp['Al']):.1f} wt%")
    for kept_name, kept_comp in accepted:
        if kept_comp and _distance(comp, kept_comp) < DUPLICATE_D:
            return True, by_name.get(name, {}).get("id"), (
                f"R4 exact duplicate of {kept_name}")
    entry = by_name.get(name)
    if entry:
        return True, entry["id"], f"{entry['status']}: {entry['rationale'][:70]}..."
    return False, None, ""


# SC/DS indicators in alloy names
SC_DS_INDICATORS = ['(SC)', '(DS)', 'CMSX', 'PWA 14', 'PWA*14', 'Rene N', 'RENÉ* N',
                    'TMS-', 'DD5', 'DD6', 'DD9', 'DD98', 'RR30']


def classify_alloy(alloy_data):
    """
    Classify an alloy into one of four categories.

    Returns: (category, reason)
    """
    alloy_name = alloy_data.get('alloy', '')
    comp = alloy_data.get('composition', {})

    ni = comp.get('Ni', 0)
    al = comp.get('Al', 0)
    ti = comp.get('Ti', 0)
    cu = comp.get('Cu', 0)
    co = comp.get('Co', 0)
    fe = comp.get('Fe', 0)

    al_ti = al + ti

    # Check for SC/DS first
    is_sc_ds = any(ind in alloy_name for ind in SC_DS_INDICATORS)

    # Category 4: Other (non-Ni-base)
    if ni > 95:
        return "other", "Pure Nickel"
    if cu > 20:
        return "other", "Ni-Cu (Monel)"
    if ni < 40:
        if co > ni:
            return "other", "Co-base"
        else:
            return "other", "Fe-Ni base"

    # Category 3: Solid-solution (no significant precipitation)
    if al_ti < 2:
        return "solid_solution", f"Al+Ti={al_ti:.1f}%"

    # Category 2: SC/DS precipitation-hardened
    if is_sc_ds:
        if al_ti >= 6:
            return "precip_sc_ds", "SC/DS γ' high"
        else:
            return "precip_sc_ds", "SC/DS γ'/γ''"

    # Category 1: Polycrystalline precipitation-hardened
    if al_ti >= 6:
        return "precip_poly", "γ' high"
    else:
        return "precip_poly", "γ'/γ''"


def load_and_categorize():
    """Load all datasets and categorize alloys."""

    categorized = defaultdict(list)
    all_alloys = []
    stats = defaultdict(lambda: defaultdict(int))

    scope = load_scope_exclusions()
    by_name = {e["alloy"]: e for e in scope["entries"]}
    accepted = []          # (name, composition) of everything kept, for R4
    excluded_log = []

    for source_name, source_path in SOURCES.items():
        if not os.path.exists(source_path):
            print(f"Warning: {source_path} not found, skipping")
            continue

        print(f"\nProcessing {source_name}...")

        with open(source_path, 'r') as f:
            for line in f:
                if not line.strip():
                    continue

                try:
                    alloy_data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                excluded, entry_id, why = scope_verdict(alloy_data, accepted, by_name)
                if excluded:
                    entry = by_name.get(alloy_data.get('alloy', ''), {})
                    disposition = entry.get('disposition', 'dropped')
                    excluded_log.append((alloy_data.get('alloy', '?'), source_name,
                                         entry_id, why, disposition))
                    stats[source_name]['excluded'] += 1
                    if disposition != 'other':
                        continue
                    category, reason = 'other', why
                else:
                    category, reason = classify_alloy(alloy_data)
                    accepted.append((alloy_data.get('alloy', '?'),
                                     alloy_data.get('composition') or {}))

                # Add metadata
                alloy_data['_category'] = category
                alloy_data['_category_reason'] = reason
                alloy_data['_source'] = source_name

                categorized[category].append(alloy_data)
                all_alloys.append(alloy_data)
                stats[source_name][category] += 1

                print(f"  {category:<15} | {reason:<20} | {alloy_data.get('alloy', 'Unknown')[:40]}")

    print(f"\n{len(excluded_log)} record(s) excluded by scope_exclusions.json:")
    for name, src, entry_id, why, disposition in excluded_log:
        print(f"  {entry_id or '??':4s} {name[:44]:46s} [{src}] -> {disposition}")
        print(f"       {why}")

    return categorized, all_alloys, stats


def write_datasets(categorized, all_alloys, stats):
    """Write categorized datasets to files."""

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Write individual category files
    for category, alloys in categorized.items():
        output_path = os.path.join(OUTPUT_DIR, f"{category}.jsonl")
        with open(output_path, 'w') as f:
            for alloy in alloys:
                f.write(json.dumps(alloy) + '\n')
        print(f"\nWrote {len(alloys)} alloys to {category}.jsonl")

    # Write combined file with all alloys
    combined_path = os.path.join(OUTPUT_DIR, "all_categorized.jsonl")
    with open(combined_path, 'w') as f:
        for alloy in all_alloys:
            f.write(json.dumps(alloy) + '\n')
    print(f"Wrote {len(all_alloys)} alloys to all_categorized.jsonl")

    # Write manifest/summary
    manifest = {
        "created": datetime.now().isoformat(),
        "description": "Categorized evaluation datasets for AlloyGraph",
        "categories": {
            "precip_poly": {
                "name": "Precipitation-hardened Polycrystalline",
                "description": "Primary target: γ'/γ'' strengthened, polycrystalline Ni-base superalloys",
                "criteria": "Ni > 40%, Al+Ti >= 2%, not SC/DS",
                "count": len(categorized.get("precip_poly", []))
            },
            "precip_sc_ds": {
                "name": "Precipitation-hardened SC/DS",
                "description": "Single crystal and directionally solidified superalloys",
                "criteria": "Ni > 40%, Al+Ti >= 2%, SC/DS processing",
                "count": len(categorized.get("precip_sc_ds", []))
            },
            "solid_solution": {
                "name": "Solid-Solution Strengthened",
                "description": "Corrosion-resistant alloys without significant precipitation",
                "criteria": "Ni > 40%, Al+Ti < 2%",
                "count": len(categorized.get("solid_solution", []))
            },
            "other": {
                "name": "Other Nickel Alloys",
                "description": "Pure Ni, Monel, Co-base, Fe-Ni base alloys",
                "criteria": "Ni < 40% or Cu > 20% or Ni > 95%",
                "count": len(categorized.get("other", []))
            }
        },
        "sources": dict(stats),
        "files": {
            "precip_poly.jsonl": "Primary evaluation set",
            "precip_sc_ds.jsonl": "SC/DS alloys (different physics)",
            "solid_solution.jsonl": "Out-of-scope test (different mechanism)",
            "other.jsonl": "Non-Ni-base alloys",
            "all_categorized.jsonl": "Combined file with _category field"
        },
        "total_alloys": len(all_alloys)
    }

    manifest_path = os.path.join(OUTPUT_DIR, "manifest.json")
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote manifest.json")

    return manifest


def print_summary(stats, manifest):
    """Print summary statistics."""

    print("\n" + "=" * 70)
    print("CATEGORIZED DATASET SUMMARY")
    print("=" * 70)

    print("\nBy Source:")
    print("-" * 50)
    for source, cats in stats.items():
        print(f"\n  {source}:")
        for cat, count in sorted(cats.items()):
            print(f"    {cat:<20}: {count}")

    print("\n\nBy Category (Total):")
    print("-" * 50)
    for cat, info in manifest["categories"].items():
        print(f"  {cat:<20}: {info['count']:>3} alloys - {info['name']}")

    print(f"\n  {'TOTAL':<20}: {manifest['total_alloys']:>3} alloys")

    print("\n\nOutput Files:")
    print("-" * 50)
    for filename, desc in manifest["files"].items():
        print(f"  {filename:<30} - {desc}")

    print("\n" + "=" * 70)


def main():
    print("Creating categorized evaluation datasets...")
    print("=" * 70)

    categorized, all_alloys, stats = load_and_categorize()
    manifest = write_datasets(categorized, all_alloys, stats)
    print_summary(dict(stats), manifest)

    print(f"\nDatasets saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
