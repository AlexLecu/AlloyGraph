"""Shared utilities for evaluation dataset generators."""

import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
GROUND_TRUTH = ROOT / "backend" / "alloy_crew" / "models" / "training_data" / "train_77alloys.jsonl"

PROPERTIES = {
    "yield_strength": {"name": "yield strength", "field": "yield_strength", "unit": "MPa"},
    "uts":            {"name": "tensile strength", "field": "uts", "unit": "MPa"},
    "elongation":     {"name": "elongation", "field": "elongation", "unit": "%"},
    "elasticity":     {"name": "elastic modulus", "field": "elasticity", "unit": "GPa"},
    "density":        {"name": "density", "field": "density", "unit": "g/cm³"},
}


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
    core_counts = Counter(_core_name(a["alloy"]) for a in alloys)
    multi_variant = {core for core, count in core_counts.items() if count > 1}
    if multi_variant:
        before = len(alloys)
        alloys = [a for a in alloys if _core_name(a["alloy"]) not in multi_variant]
        print(f"Skipped {before - len(alloys)} entries with multiple variants: "
              f"{sorted(multi_variant)}")

    return alloys


def get_rt_value(measurements: list[dict]) -> float | None:
    """Get room temperature value (20-25 C) from a list of measurements."""
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
