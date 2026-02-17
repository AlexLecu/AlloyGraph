"""
Generate chatbot evaluation dataset from KG ground truth.

Reads train_77alloys.jsonl (the alloys in Weaviate) and produces
question/ground_truth pairs across 5 question types:
  1. Property lookup  — specific value for a specific alloy
  2. Composition       — elemental makeup of an alloy
  3. Ranking           — which alloy has the highest/lowest X
  4. Target search     — find alloy with property near a value
  5. Comparison        — compare two alloys on a property

Output: evaluation_chatbot/data/questions.jsonl
"""

import json
import re
import random
import os
from collections import Counter
from pathlib import Path

# ── Paths ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent
GROUND_TRUTH = ROOT / "backend" / "alloy_crew" / "models" / "training_data" / "train_77alloys.jsonl"

random.seed(42)  # reproducibility


# ── Load ground truth ───────────────────────────────────────────────────

def _core_name(alloy_name: str) -> str:
    """Extract core alloy name, stripping processing suffixes and symbols."""
    name = alloy_name.lower().replace("*", "").strip()
    name = re.sub(r"\(forged\)|\(cast\)", "", name)
    for word in ["wrought", "cast", "bar", "sheet", "plate", "forged"]:
        name = re.sub(rf"\b{word}\b", "", name)
    return name.strip()


def load_alloys(path: Path) -> list[dict]:
    alloys = []
    with open(path) as f:
        for line in f:
            alloys.append(json.loads(line))

    # Skip alloys that have multiple variants (cast + wrought, etc.)
    # to avoid ambiguity in questions and ground truth
    core_counts = Counter(_core_name(a["alloy"]) for a in alloys)
    multi_variant = {core for core, count in core_counts.items() if count > 1}
    if multi_variant:
        before = len(alloys)
        alloys = [a for a in alloys if _core_name(a["alloy"]) not in multi_variant]
        print(f"Skipped {before - len(alloys)} entries with multiple variants: "
              f"{sorted(multi_variant)}")

    return alloys


def get_rt_value(measurements: list[dict]) -> float | None:
    """Get room temperature value (20-25°C) from a list of measurements."""
    for m in measurements:
        temp = m["temp_c"].strip()
        if temp in ("20", "21", "22", "25"):
            return m["value"]
    return None


def get_value_at_temp(measurements: list[dict], target_temp: int) -> float | None:
    """Get value at a specific temperature."""
    for m in measurements:
        if int(m["temp_c"].strip()) == target_temp:
            return m["value"]
    return None


def top_elements(composition: dict, n: int = 5) -> list[tuple[str, float]]:
    """Return top-n elements by weight percent."""
    sorted_comp = sorted(composition.items(), key=lambda x: x[1], reverse=True)
    return sorted_comp[:n]


# ── Question templates ──────────────────────────────────────────────────

PROPERTY_LOOKUP_TEMPLATES = [
    "What is the {property_name} of {alloy}?",
    "What's the {property_name} of {alloy}?",
    "Tell me the {property_name} of {alloy}.",
    "How much {property_name} does {alloy} have?",
]

PROPERTY_LOOKUP_TEMP_TEMPLATES = [
    "What is the {property_name} of {alloy} at {temp}°C?",
    "What's the {property_name} of {alloy} at {temp}°C?",
]

COMPOSITION_TEMPLATES = [
    "What is the composition of {alloy}?",
    "What elements are in {alloy}?",
    "Tell me about the composition of {alloy}.",
]

RANKING_TEMPLATES_HIGHEST = [
    "Which alloy has the highest {property_name}?",
    "What alloy has the greatest {property_name}?",
    "Top 3 alloys by {property_name}.",
]

RANKING_TEMPLATES_LOWEST = [
    "Which alloy has the lowest {property_name}?",
    "What is the lightest alloy?" if "density" in "{property_name}" else "Which alloy has the lowest {property_name}?",
]

TARGET_TEMPLATES = [
    "Find an alloy with {property_name} around {value} {unit}.",
    "Which alloy has a {property_name} close to {value} {unit}?",
    "I need an alloy with approximately {value} {unit} {property_name}.",
]

COMPARISON_TEMPLATES = [
    "Compare {alloy1} and {alloy2} in terms of {property_name}.",
    "How does the {property_name} of {alloy1} compare to {alloy2}?",
    "What's the difference in {property_name} between {alloy1} and {alloy2}?",
]

# Property metadata
PROPERTIES = {
    "yield_strength": {
        "name": "yield strength",
        "field": "yield_strength",
        "unit": "MPa",
    },
    "uts": {
        "name": "tensile strength",
        "field": "uts",
        "unit": "MPa",
    },
    "elongation": {
        "name": "elongation",
        "field": "elongation",
        "unit": "%",
    },
    "elasticity": {
        "name": "elastic modulus",
        "field": "elasticity",
        "unit": "GPa",
    },
    "density": {
        "name": "density",
        "field": "density",
        "unit": "g/cm³",
    },
}


# ── Question generators ────────────────────────────────────────────────

def generate_property_lookup(alloys: list[dict]) -> list[dict]:
    """Type 1: Property lookup questions (RT and elevated temperature)."""
    questions = []
    qid = 0

    for prop_key, prop_meta in PROPERTIES.items():
        if prop_key == "density":
            # Density comes from computed_features, not measurements
            for alloy in alloys:
                cf = alloy.get("computed_features", {})
                density = cf.get("density_calculated_gcm3")
                if density is None:
                    continue
                qid += 1
                template = random.choice(PROPERTY_LOOKUP_TEMPLATES)
                questions.append({
                    "id": f"prop_{prop_key}_{qid:03d}",
                    "type": "property_lookup",
                    "question": template.format(
                        property_name=prop_meta["name"],
                        alloy=alloy["alloy"],
                    ),
                    "ground_truth": (
                        f"The {prop_meta['name']} of {alloy['alloy']} is "
                        f"{density:.2f} {prop_meta['unit']}."
                    ),
                    "ground_truth_value": round(density, 2),
                    "ground_truth_unit": prop_meta["unit"],
                    "alloy": alloy["alloy"],
                    "property": prop_key,
                    "temperature_c": None,
                })
            continue

        # Measurement-based properties
        for alloy in alloys:
            measurements = alloy.get(prop_meta["field"], [])
            if not measurements:
                continue

            # Room temperature question
            rt_val = get_rt_value(measurements)
            if rt_val is not None:
                qid += 1
                template = random.choice(PROPERTY_LOOKUP_TEMPLATES)
                questions.append({
                    "id": f"prop_{prop_key}_{qid:03d}",
                    "type": "property_lookup",
                    "question": template.format(
                        property_name=prop_meta["name"],
                        alloy=alloy["alloy"],
                    ),
                    "ground_truth": (
                        f"The {prop_meta['name']} of {alloy['alloy']} at room temperature "
                        f"is {rt_val:.0f} {prop_meta['unit']}."
                    ),
                    "ground_truth_value": rt_val,
                    "ground_truth_unit": prop_meta["unit"],
                    "alloy": alloy["alloy"],
                    "property": prop_key,
                    "temperature_c": 21,
                })

            # Elevated temperature question (pick one non-RT temp if available)
            elevated = [
                m for m in measurements
                if int(m["temp_c"].strip()) > 25
            ]
            if elevated:
                m = random.choice(elevated)
                temp = int(m["temp_c"].strip())
                qid += 1
                template = random.choice(PROPERTY_LOOKUP_TEMP_TEMPLATES)
                questions.append({
                    "id": f"prop_{prop_key}_{qid:03d}",
                    "type": "property_lookup",
                    "question": template.format(
                        property_name=prop_meta["name"],
                        alloy=alloy["alloy"],
                        temp=temp,
                    ),
                    "ground_truth": (
                        f"The {prop_meta['name']} of {alloy['alloy']} at {temp}°C "
                        f"is {m['value']:.0f} {prop_meta['unit']}."
                    ),
                    "ground_truth_value": m["value"],
                    "ground_truth_unit": prop_meta["unit"],
                    "alloy": alloy["alloy"],
                    "property": prop_key,
                    "temperature_c": temp,
                })

    return questions


def generate_composition(alloys: list[dict]) -> list[dict]:
    """Type 2: Composition questions."""
    questions = []
    candidates = [a for a in alloys if a.get("composition")]

    for i, alloy in enumerate(candidates):
        comp = alloy["composition"]
        top = top_elements(comp)
        top_str = ", ".join(f"{el}: {val:.1f}%" for el, val in top)

        template = random.choice(COMPOSITION_TEMPLATES)
        questions.append({
            "id": f"comp_{i+1:03d}",
            "type": "composition",
            "question": template.format(alloy=alloy["alloy"]),
            "ground_truth": (
                f"The main elements in {alloy['alloy']} (wt%) are: {top_str}. "
                f"Processing: {alloy['processing']}."
            ),
            "ground_truth_composition": comp,
            "alloy": alloy["alloy"],
        })

    return questions


def generate_ranking(alloys: list[dict]) -> list[dict]:
    """Type 3: Ranking questions (highest/lowest)."""
    questions = []
    qid = 0

    for prop_key, prop_meta in PROPERTIES.items():
        if prop_key == "density":
            scored = []
            for a in alloys:
                cf = a.get("computed_features", {})
                d = cf.get("density_calculated_gcm3")
                if d is not None:
                    scored.append((a["alloy"], d))
        else:
            scored = []
            for a in alloys:
                val = get_rt_value(a.get(prop_meta["field"], []))
                if val is not None:
                    scored.append((a["alloy"], val))

        if len(scored) < 5:
            continue

        # Highest
        scored_high = sorted(scored, key=lambda x: x[1], reverse=True)
        top3 = scored_high[:3]
        top3_str = "; ".join(f"{name}: {val:.0f} {prop_meta['unit']}" for name, val in top3)

        qid += 1
        template = random.choice(RANKING_TEMPLATES_HIGHEST)
        questions.append({
            "id": f"rank_{prop_key}_{qid:03d}",
            "type": "ranking",
            "question": template.format(property_name=prop_meta["name"]),
            "ground_truth": (
                f"The top 3 alloys by {prop_meta['name']} (room temperature) are: {top3_str}."
            ),
            "ground_truth_ranking": [
                {"alloy": name, "value": val} for name, val in top3
            ],
            "property": prop_key,
            "direction": "highest",
        })

        # Lowest
        scored_low = sorted(scored, key=lambda x: x[1])
        bot3 = scored_low[:3]
        bot3_str = "; ".join(f"{name}: {val:.0f} {prop_meta['unit']}" for name, val in bot3)

        qid += 1
        template = random.choice(RANKING_TEMPLATES_LOWEST)
        q_text = template.format(property_name=prop_meta["name"])
        # Special case: "lightest alloy" for density
        if prop_key == "density":
            q_text = random.choice([
                "Which alloy has the lowest density?",
                "What is the lightest superalloy?",
            ])
        questions.append({
            "id": f"rank_{prop_key}_{qid:03d}",
            "type": "ranking",
            "question": q_text,
            "ground_truth": (
                f"The 3 alloys with the lowest {prop_meta['name']} are: {bot3_str}."
            ),
            "ground_truth_ranking": [
                {"alloy": name, "value": val} for name, val in bot3
            ],
            "property": prop_key,
            "direction": "lowest",
        })

    return questions


def generate_target(alloys: list[dict]) -> list[dict]:
    """Type 4: Target search questions."""
    questions = []
    qid = 0

    for prop_key, prop_meta in PROPERTIES.items():
        if prop_key == "density":
            scored = []
            for a in alloys:
                cf = a.get("computed_features", {})
                d = cf.get("density_calculated_gcm3")
                if d is not None:
                    scored.append((a["alloy"], d))
        else:
            scored = []
            for a in alloys:
                val = get_rt_value(a.get(prop_meta["field"], []))
                if val is not None:
                    scored.append((a["alloy"], val))

        if len(scored) < 5:
            continue

        # Pick 3 target values: low / median / high quartile (round to nice numbers)
        values = sorted([v for _, v in scored])
        q1 = values[len(values) // 4]
        median_val = values[len(values) // 2]
        q3 = values[3 * len(values) // 4]

        if prop_key == "density":
            targets = [round(q1, 1), round(median_val, 1), round(q3, 1)]
        elif prop_key == "elongation":
            targets = [round(q1 / 5) * 5, round(median_val / 5) * 5, round(q3 / 5) * 5]
        else:
            targets = [round(q1 / 50) * 50, round(median_val / 50) * 50, round(q3 / 50) * 50]
        # Deduplicate in case rounding produces the same value
        targets = list(dict.fromkeys(t for t in targets if t > 0))

        for target_val in targets:
            if target_val <= 0:
                continue
            # Find the 3 closest alloys
            by_distance = sorted(scored, key=lambda x: abs(x[1] - target_val))
            closest3 = by_distance[:3]
            fmt = ".2f" if prop_key == "density" else ".0f"
            closest_str = "; ".join(
                f"{name}: {val:{fmt}} {prop_meta['unit']}" for name, val in closest3
            )

            qid += 1
            template = random.choice(TARGET_TEMPLATES)
            tv_str = f"{target_val:.1f}" if prop_key == "density" else f"{target_val:.0f}"
            questions.append({
                "id": f"target_{prop_key}_{qid:03d}",
                "type": "target_search",
                "question": template.format(
                    property_name=prop_meta["name"],
                    value=tv_str,
                    unit=prop_meta["unit"],
                ),
                "ground_truth": (
                    f"Alloys with {prop_meta['name']} closest to "
                    f"{tv_str} {prop_meta['unit']}: {closest_str}."
                ),
                "ground_truth_closest": [
                    {"alloy": name, "value": val} for name, val in closest3
                ],
                "target_value": target_val,
                "property": prop_key,
            })

    return questions


def generate_comparison(alloys: list[dict]) -> list[dict]:
    """Type 5: Comparison questions."""
    questions = []
    qid = 0

    for prop_key, prop_meta in PROPERTIES.items():
        if prop_key == "density":
            alloys_with_val = []
            for a in alloys:
                cf = a.get("computed_features", {})
                d = cf.get("density_calculated_gcm3")
                if d is not None:
                    alloys_with_val.append((a["alloy"], d))
        else:
            alloys_with_val = []
            for a in alloys:
                val = get_rt_value(a.get(prop_meta["field"], []))
                if val is not None:
                    alloys_with_val.append((a["alloy"], val))

        if len(alloys_with_val) < 2:
            continue

        # Pick 4 random pairs per property
        pairs = []
        shuffled = list(alloys_with_val)
        random.shuffle(shuffled)
        for i in range(0, min(8, len(shuffled) - 1), 2):
            pairs.append((shuffled[i], shuffled[i + 1]))

        for (name1, val1), (name2, val2) in pairs:
            qid += 1
            template = random.choice(COMPARISON_TEMPLATES)
            diff = val1 - val2
            if prop_key == "density":
                gt = (
                    f"{name1} has a {prop_meta['name']} of {val1:.2f} {prop_meta['unit']} "
                    f"and {name2} has {val2:.2f} {prop_meta['unit']}. "
                    f"Difference: {abs(diff):.2f} {prop_meta['unit']}."
                )
            else:
                gt = (
                    f"{name1} has a {prop_meta['name']} of {val1:.0f} {prop_meta['unit']} "
                    f"and {name2} has {val2:.0f} {prop_meta['unit']} at room temperature. "
                    f"Difference: {abs(diff):.0f} {prop_meta['unit']}."
                )

            questions.append({
                "id": f"comp_{prop_key}_{qid:03d}",
                "type": "comparison",
                "question": template.format(
                    property_name=prop_meta["name"],
                    alloy1=name1,
                    alloy2=name2,
                ),
                "ground_truth": gt,
                "alloy1": name1,
                "alloy1_value": val1,
                "alloy2": name2,
                "alloy2_value": val2,
                "property": prop_key,
            })

    return questions


# ── Balanced sampling ───────────────────────────────────────────────────

SAMPLE_TARGETS = {
    "property_lookup": 40,  # ~40 property lookups (mix of RT + elevated)
    "composition": 20,       # ~20 composition questions
    "ranking": 10,           # all ranking questions (5 props × 2 directions)
    "target_search": 13,     # all target questions
    "comparison": 17,        # ~17 comparison questions
}


def sample_balanced(questions: list[dict]) -> list[dict]:
    """Create a balanced evaluation subset from the full question pool."""
    by_type: dict[str, list[dict]] = {}
    for q in questions:
        by_type.setdefault(q["type"], []).append(q)

    sampled = []
    for qtype, target_n in SAMPLE_TARGETS.items():
        pool = by_type.get(qtype, [])
        if len(pool) <= target_n:
            sampled.extend(pool)
        else:
            # For property lookups, ensure diversity across properties and alloys
            if qtype == "property_lookup":
                sampled.extend(_diverse_sample_properties(pool, target_n))
            else:
                sampled.extend(random.sample(pool, target_n))

    return sampled


def _diverse_sample_properties(pool: list[dict], n: int) -> list[dict]:
    """Sample property lookup questions ensuring coverage of all property types
    and a mix of RT / elevated temperature."""
    by_prop: dict[str, list[dict]] = {}
    for q in pool:
        by_prop.setdefault(q["property"], []).append(q)

    per_prop = max(1, n // len(by_prop))
    remainder = n - per_prop * len(by_prop)

    sampled = []
    for prop, qs in by_prop.items():
        # Split into RT and elevated
        rt = [q for q in qs if q.get("temperature_c") in (20, 21, 22, 25, None)]
        elevated = [q for q in qs if q not in rt]

        # Take half from RT, half from elevated (when available)
        n_rt = min(len(rt), (per_prop + 1) // 2)
        n_elev = min(len(elevated), per_prop - n_rt)
        n_rt = min(len(rt), per_prop - n_elev)  # rebalance if elevated was short

        sampled.extend(random.sample(rt, n_rt) if n_rt <= len(rt) else rt)
        sampled.extend(random.sample(elevated, n_elev) if n_elev <= len(elevated) else elevated)

    # Fill remainder from the full pool (avoiding duplicates)
    sampled_ids = {q["id"] for q in sampled}
    remaining = [q for q in pool if q["id"] not in sampled_ids]
    if remainder > 0 and remaining:
        sampled.extend(random.sample(remaining, min(remainder, len(remaining))))

    return sampled[:n]


# ── Save helper ─────────────────────────────────────────────────────────

def save_jsonl(questions: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for q in questions:
            f.write(json.dumps(q) + "\n")


# ── Main ────────────────────────────────────────────────────────────────

OUTPUT_FULL = ROOT / "evaluation_chatbot" / "data" / "questions_full.jsonl"
OUTPUT_EVAL = ROOT / "evaluation_chatbot" / "data" / "questions.jsonl"


def main():
    alloys = load_alloys(GROUND_TRUTH)
    print(f"Loaded {len(alloys)} alloys from {GROUND_TRUTH.name}")

    # Generate all question types
    q_property = generate_property_lookup(alloys)
    q_composition = generate_composition(alloys)
    q_ranking = generate_ranking(alloys)
    q_target = generate_target(alloys)
    q_comparison = generate_comparison(alloys)

    all_questions = q_property + q_composition + q_ranking + q_target + q_comparison

    print(f"\nFull question pool:")
    print(f"  Property lookup:  {len(q_property)}")
    print(f"  Composition:      {len(q_composition)}")
    print(f"  Ranking:          {len(q_ranking)}")
    print(f"  Target search:    {len(q_target)}")
    print(f"  Comparison:       {len(q_comparison)}")
    print(f"  ── Total:         {len(all_questions)}")

    # Save full pool
    save_jsonl(all_questions, OUTPUT_FULL)
    print(f"\nFull pool saved to {OUTPUT_FULL}")

    # Create balanced evaluation subset
    eval_set = sample_balanced(all_questions)

    # Count by type
    type_counts: dict[str, int] = {}
    for q in eval_set:
        type_counts[q["type"]] = type_counts.get(q["type"], 0) + 1

    print(f"\nEvaluation subset (balanced):")
    for qtype, count in sorted(type_counts.items()):
        print(f"  {qtype}: {count}")
    print(f"  ── Total: {len(eval_set)}")

    save_jsonl(eval_set, OUTPUT_EVAL)
    print(f"Evaluation set saved to {OUTPUT_EVAL}")


if __name__ == "__main__":
    main()
