"""
Generate open-ended questions for RAGAS evaluation (chatbot only).

Produces ragas_questions.jsonl with 100 questions across 6 types:
  1. Descriptive profile    (20)
  2. Explanatory / why      (18)
  3. Comparative / class    (18)
  4. Recommendation         (18)
  5. Multi-property profile (14)
  6. Temperature trend      (12)

Each question has a `ground_truth` reference answer synthesised from
train_77alloys.jsonl so RAGAS can score FactualCorrectness, Faithfulness,
ResponseRelevancy, and LLMContextRecall.

Usage:
    python generate_ragas.py
"""

import json
import random
from collections import defaultdict
from pathlib import Path

from utils import load_alloys, get_rt_value, get_value_at_temp, PROPERTIES

random.seed(42)

ROOT = Path(__file__).resolve().parent.parent.parent.parent
GROUND_TRUTH = ROOT / "backend" / "alloy_crew" / "models" / "training_data" / "train_77alloys.jsonl"
OUTPUT = ROOT / "evaluation" / "chatbot" / "data" / "ragas_questions.jsonl"


# ── Helpers ────────────────────────────────────────────────────────────

def _fmt(val: float, pk: str) -> str:
    unit = PROPERTIES[pk]["unit"]
    if pk == "density":
        return f"{val:.2f} {unit}"
    if pk == "elongation":
        return f"{val:.0f}{unit}"
    return f"{val:.0f} {unit}"


def _top_elements(comp: dict, n: int = 5) -> str:
    top = sorted(comp.items(), key=lambda x: x[1], reverse=True)[:n]
    return ", ".join(f"{el}: {v:.1f}%" for el, v in top)


def _prop_at_temps(alloy: dict, pk: str) -> list[tuple[int, float]]:
    """Return [(temp, value), ...] sorted by temp."""
    if pk == "density":
        d = alloy.get("computed_features", {}).get("density_calculated_gcm3")
        return [(21, d)] if d else []
    field = PROPERTIES[pk]["field"]
    pairs = []
    for m in alloy.get(field, []):
        pairs.append((int(m["temp_c"].strip()), m["value"]))
    return sorted(pairs, key=lambda x: x[0])


def _alloy_class(alloy: dict) -> str:
    """Classify alloy into descriptive category."""
    name = alloy["alloy"]
    if "(SC)" in name or "(DS)" in name:
        return "single-crystal/DS"
    return alloy.get("processing", "unknown")


# ── Type 1: Descriptive profile ───────────────────────────────────────

def _build_descriptive_ref(alloy: dict, focus: str) -> str:
    """Build a comprehensive reference for a descriptive question."""
    name = alloy["alloy"]
    proc = alloy.get("processing", "")
    cf = alloy.get("computed_features", {})
    lines = [f"{name} is a {proc} nickel-based superalloy."]

    # Composition
    comp = alloy.get("composition", {})
    if comp:
        lines.append(f"Main elements (wt%): {_top_elements(comp)}.")

    # Key computed features
    gp = cf.get("gamma_prime_estimated_vol_pct")
    density = cf.get("density_calculated_gcm3")
    tcp = cf.get("TCP_risk")
    if gp is not None:
        lines.append(f"Estimated γ' volume fraction: {gp:.1f}%.")
    if density is not None:
        lines.append(f"Density: {density:.2f} g/cm³.")
    if tcp:
        lines.append(f"TCP risk: {tcp}.")

    if focus == "high_temp":
        # Show properties at elevated temperatures
        for pk in ["yield_strength", "uts", "elongation"]:
            pairs = _prop_at_temps(alloy, pk)
            elevated = [(t, v) for t, v in pairs if t > 500]
            if elevated:
                vals = ", ".join(f"{_fmt(v, pk)} at {t}°C" for t, v in elevated)
                lines.append(f"{PROPERTIES[pk]['name'].capitalize()}: {vals}.")
    elif focus == "room_temp":
        for pk in ["yield_strength", "uts", "elongation", "elasticity"]:
            val = get_rt_value(alloy.get(PROPERTIES[pk]["field"], []))
            if val is not None:
                lines.append(f"RT {PROPERTIES[pk]['name']}: {_fmt(val, pk)}.")
    else:  # "all_temps"
        for pk in ["yield_strength", "uts"]:
            pairs = _prop_at_temps(alloy, pk)
            if pairs:
                vals = ", ".join(f"{_fmt(v, pk)} at {t}°C" for t, v in pairs)
                lines.append(f"{PROPERTIES[pk]['name'].capitalize()}: {vals}.")

    return " ".join(lines)


def gen_descriptive(alloys: list[dict], target: int = 20) -> list[dict]:
    """Type 1: Describe the properties of alloy X."""
    templates = [
        ("Describe the mechanical properties of {alloy} at elevated temperatures.",
         "high_temp"),
        ("What are the room-temperature mechanical properties of {alloy}?",
         "room_temp"),
        ("Describe the overall property profile of {alloy} across all temperatures.",
         "all_temps"),
        ("What are the key characteristics and properties of {alloy}?",
         "room_temp"),
    ]

    candidates = [a for a in alloys
                  if len(a.get("yield_strength", [])) >= 2]
    random.shuffle(candidates)

    questions = []
    for i, alloy in enumerate(candidates):
        if len(questions) >= target:
            break
        tmpl, focus = templates[i % len(templates)]
        q_text = tmpl.format(alloy=alloy["alloy"])
        ref = _build_descriptive_ref(alloy, focus)
        questions.append({
            "id": f"ragas_desc_{len(questions)+1:03d}",
            "subtype": "descriptive",
            "question": q_text,
            "ground_truth": ref,
        })
    return questions


# ── Type 2: Explanatory (why) ─────────────────────────────────────────

def gen_explanatory(alloys: list[dict], target: int = 18) -> list[dict]:
    """Type 2: Why might alloy A be preferred over alloy B?"""
    # Build pairs with clear performance differences
    templates = [
        "Why might {a1} be preferred over {a2} for high-temperature structural applications?",
        "What advantages does {a1} have over {a2} for applications above 800°C?",
        "Why would an engineer choose {a1} instead of {a2} for high-strength applications?",
    ]

    candidates = []
    for i, a1 in enumerate(alloys):
        for a2 in alloys[i+1:]:
            ys1 = get_rt_value(a1.get("yield_strength", []))
            ys2 = get_rt_value(a2.get("yield_strength", []))
            if ys1 and ys2 and abs(ys1 - ys2) > 200:
                stronger = (a1, ys1, a2, ys2) if ys1 > ys2 else (a2, ys2, a1, ys1)
                candidates.append(stronger)

    random.shuffle(candidates)
    questions = []

    for a_strong, ys_s, a_weak, ys_w in candidates:
        if len(questions) >= target:
            break

        name_s = a_strong["alloy"]
        name_w = a_weak["alloy"]
        proc_s = a_strong.get("processing", "")
        proc_w = a_weak.get("processing", "")

        # Build reference with quantitative comparison
        ref_parts = [
            f"{name_s} ({proc_s}) has a significantly higher room-temperature "
            f"yield strength ({_fmt(ys_s, 'yield_strength')}) compared to "
            f"{name_w} ({proc_w}, {_fmt(ys_w, 'yield_strength')}).",
        ]

        # Add UTS comparison
        uts_s = get_rt_value(a_strong.get("uts", []))
        uts_w = get_rt_value(a_weak.get("uts", []))
        if uts_s and uts_w:
            ref_parts.append(
                f"UTS: {name_s} {_fmt(uts_s, 'uts')} vs {name_w} {_fmt(uts_w, 'uts')}.")

        # Add elongation comparison (trade-off)
        el_s = get_rt_value(a_strong.get("elongation", []))
        el_w = get_rt_value(a_weak.get("elongation", []))
        if el_s and el_w:
            ref_parts.append(
                f"Elongation: {name_s} {_fmt(el_s, 'elongation')} vs "
                f"{name_w} {_fmt(el_w, 'elongation')}.")

        # Gamma prime comparison
        gp_s = a_strong.get("computed_features", {}).get("gamma_prime_estimated_vol_pct")
        gp_w = a_weak.get("computed_features", {}).get("gamma_prime_estimated_vol_pct")
        if gp_s is not None and gp_w is not None:
            ref_parts.append(
                f"γ' fraction: {name_s} {gp_s:.1f}% vs {name_w} {gp_w:.1f}%.")

        tmpl = templates[len(questions) % len(templates)]
        q_text = tmpl.format(a1=name_s, a2=name_w)

        questions.append({
            "id": f"ragas_expl_{len(questions)+1:03d}",
            "subtype": "explanatory",
            "question": q_text,
            "ground_truth": " ".join(ref_parts),
        })
    return questions


# ── Type 3: Comparative (class) ───────────────────────────────────────

def gen_comparative(alloys: list[dict], target: int = 18) -> list[dict]:
    """Type 3: How do class A alloys compare to class B?"""
    # Group alloys
    cast = [a for a in alloys if a["processing"] == "cast"
            and "(SC)" not in a["alloy"] and "(DS)" not in a["alloy"]]
    wrought = [a for a in alloys if a["processing"] == "wrought"]
    sc_ds = [a for a in alloys
             if "(SC)" in a["alloy"] or "(DS)" in a["alloy"]]

    # High vs low gamma prime
    high_gp = [a for a in alloys
               if (a.get("computed_features", {})
                   .get("gamma_prime_estimated_vol_pct", 0)) > 40]
    low_gp = [a for a in alloys
              if (a.get("computed_features", {})
                  .get("gamma_prime_estimated_vol_pct", 0)) < 15
              and (a.get("computed_features", {})
                   .get("gamma_prime_estimated_vol_pct", 0)) > 0]

    # High-Cr vs low-Cr
    high_cr = [a for a in alloys if a.get("composition", {}).get("Cr", 0) > 20]
    low_cr = [a for a in alloys if a.get("composition", {}).get("Cr", 0) < 12
              and a.get("composition", {}).get("Cr", 0) > 0]

    def _class_stats(group: list[dict], pk: str,
                     temp: int | None = None) -> tuple[float, float, int]:
        """Return (mean, std, count) for a property in a group."""
        vals = []
        for a in group:
            if temp is None:
                v = get_rt_value(a.get(PROPERTIES[pk]["field"], []))
            else:
                v = get_value_at_temp(a.get(PROPERTIES[pk]["field"], []), temp)
            if v is not None:
                vals.append(v)
        if not vals:
            return 0, 0, 0
        mean = sum(vals) / len(vals)
        std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
        return mean, std, len(vals)

    comparisons = [
        (cast, "cast polycrystalline alloys", wrought, "wrought alloys"),
        (sc_ds, "single-crystal/DS alloys", wrought, "wrought alloys"),
        (sc_ds, "single-crystal/DS alloys", cast, "cast polycrystalline alloys"),
        (high_gp, "high γ' (>40 vol%) alloys", low_gp, "low γ' (<15 vol%) alloys"),
        (high_cr, "high-Cr (>20 wt%) alloys", low_cr, "low-Cr (<12 wt%) alloys"),
    ]

    prop_temps = [
        ("yield_strength", None, "room-temperature yield strength"),
        ("yield_strength", 871, "yield strength at 871°C"),
        ("uts", None, "room-temperature tensile strength"),
        ("elongation", None, "room-temperature elongation"),
        ("density", None, "density"),
    ]

    combos = [(c, pt) for c in comparisons for pt in prop_temps]
    random.shuffle(combos)

    questions = []
    for (g1, n1, g2, n2), (pk, temp, prop_desc) in combos:
        if len(questions) >= target:
            break
        m1, s1, c1 = _class_stats(g1, pk, temp)
        m2, s2, c2 = _class_stats(g2, pk, temp)
        if c1 < 3 or c2 < 3:
            continue

        temp_str = f" at {temp}°C" if temp else " at room temperature"
        q_text = (f"How do {n1} compare to {n2} in terms of "
                  f"{PROPERTIES[pk]['name']}{temp_str}?")

        ref = (f"Among {n1} (n={c1}), the average {PROPERTIES[pk]['name']} is "
               f"{_fmt(m1, pk)}{temp_str}. "
               f"Among {n2} (n={c2}), the average is {_fmt(m2, pk)}. ")
        if m1 > m2:
            ref += f"{n1.capitalize()} have {((m1/m2)-1)*100:.0f}% higher average."
        else:
            ref += f"{n2.capitalize()} have {((m2/m1)-1)*100:.0f}% higher average."

        questions.append({
            "id": f"ragas_comp_{len(questions)+1:03d}",
            "subtype": "comparative",
            "question": q_text,
            "ground_truth": ref,
        })
    return questions


# ── Type 4: Recommendation ────────────────────────────────────────────

def gen_recommendation(alloys: list[dict], target: int = 18) -> list[dict]:
    """Type 4: Which alloy would suit application requiring X?"""
    requirements = [
        {"desc": "yield strength above 900 MPa at room temperature with good ductility (elongation > 15%)",
         "filters": [("yield_strength", None, 900, "gt"), ("elongation", None, 15, "gt")]},
        {"desc": "high tensile strength at 871°C (above 700 MPa)",
         "filters": [("uts", 871, 700, "gt")]},
        {"desc": "yield strength above 600 MPa at 871°C",
         "filters": [("yield_strength", 871, 600, "gt")]},
        {"desc": "density below 8.0 g/cm³ with yield strength above 800 MPa at RT",
         "filters": [("density", None, 8.0, "lt"), ("yield_strength", None, 800, "gt")]},
        {"desc": "high elongation (above 40%) at room temperature",
         "filters": [("elongation", None, 40, "gt")]},
        {"desc": "tensile strength above 1200 MPa at room temperature",
         "filters": [("uts", None, 1200, "gt")]},
        {"desc": "yield strength above 800 MPa at 760°C",
         "filters": [("yield_strength", 760, 800, "gt")]},
        {"desc": "good high-temperature strength (YS > 500 MPa at 982°C)",
         "filters": [("yield_strength", 982, 500, "gt")]},
        {"desc": "low density (below 7.8 g/cm³) for lightweight applications",
         "filters": [("density", None, 7.8, "lt")]},
        {"desc": "high elastic modulus (above 215 GPa) at room temperature",
         "filters": [("elasticity", None, 215, "gt")]},
        {"desc": "excellent ductility at 871°C (elongation > 30%)",
         "filters": [("elongation", 871, 30, "gt")]},
        {"desc": "high tensile strength (UTS > 1000 MPa) with elongation above 20% at RT",
         "filters": [("uts", None, 1000, "gt"), ("elongation", None, 20, "gt")]},
        {"desc": "yield strength above 700 MPa at 649°C",
         "filters": [("yield_strength", 649, 700, "gt")]},
        {"desc": "yield strength above 400 MPa at 982°C with density below 8.5 g/cm³",
         "filters": [("yield_strength", 982, 400, "gt"), ("density", None, 8.5, "lt")]},
        {"desc": "high tensile strength at 760°C (above 900 MPa)",
         "filters": [("uts", 760, 900, "gt")]},
        {"desc": "elongation above 25% at room temperature with YS above 500 MPa",
         "filters": [("elongation", None, 25, "gt"), ("yield_strength", None, 500, "gt")]},
        {"desc": "tensile strength above 600 MPa at 982°C",
         "filters": [("uts", 982, 600, "gt")]},
        {"desc": "yield strength above 1000 MPa at room temperature for structural applications",
         "filters": [("yield_strength", None, 1000, "gt")]},
    ]

    random.shuffle(requirements)
    questions = []

    for req in requirements:
        if len(questions) >= target:
            break

        def _get_val(a, pk, temp):
            if pk == "density":
                return a.get("computed_features", {}).get("density_calculated_gcm3")
            if temp is None:
                return get_rt_value(a.get(PROPERTIES[pk]["field"], []))
            return get_value_at_temp(a.get(PROPERTIES[pk]["field"], []), temp)

        # Find matching alloys
        matches = []
        for a in alloys:
            passes = True
            vals = {}
            for pk, temp, thresh, direction in req["filters"]:
                v = _get_val(a, pk, temp)
                if v is None:
                    passes = False
                    break
                if direction == "gt" and v <= thresh:
                    passes = False
                    break
                if direction == "lt" and v >= thresh:
                    passes = False
                    break
                vals[pk] = (v, temp)
            if passes:
                matches.append((a["alloy"], vals))

        if not matches:
            continue

        # Build reference with top candidates
        ref_parts = [f"Based on the database, {len(matches)} alloys meet these requirements."]
        for name, vals in matches[:3]:
            val_strs = []
            for pk, (v, temp) in vals.items():
                temp_str = f" at {temp}°C" if temp else ""
                val_strs.append(f"{PROPERTIES[pk]['name']} {_fmt(v, pk)}{temp_str}")
            ref_parts.append(f"{name}: {', '.join(val_strs)}.")

        q_text = (f"Which alloy would be best suited for an application requiring "
                  f"{req['desc']}?")

        questions.append({
            "id": f"ragas_rec_{len(questions)+1:03d}",
            "subtype": "recommendation",
            "question": q_text,
            "ground_truth": " ".join(ref_parts),
        })
    return questions


# ── Type 5: Multi-property profile ────────────────────────────────────

def gen_multi_property(alloys: list[dict], target: int = 14) -> list[dict]:
    """Type 5: Full property profile of alloy X at temperature T."""
    temps_to_try = [21, 538, 649, 760, 871]
    candidates = []
    for a in alloys:
        for temp in temps_to_try:
            props_available = 0
            for pk in ["yield_strength", "uts", "elongation", "elasticity"]:
                val = get_value_at_temp(a.get(PROPERTIES[pk]["field"], []), temp)
                if val is not None:
                    props_available += 1
            if props_available >= 3:
                candidates.append((a, temp, props_available))

    # Sort by most properties, then shuffle within groups
    candidates.sort(key=lambda x: -x[2])
    questions = []
    seen_alloys: set[str] = set()

    for alloy, temp, _ in candidates:
        if len(questions) >= target:
            break
        if alloy["alloy"] in seen_alloys:
            continue

        name = alloy["alloy"]
        proc = alloy.get("processing", "")
        temp_str = "room temperature (21°C)" if temp == 21 else f"{temp}°C"

        ref_parts = [f"{name} is a {proc} alloy. Properties at {temp_str}:"]
        for pk in ["yield_strength", "uts", "elongation", "elasticity"]:
            val = get_value_at_temp(alloy.get(PROPERTIES[pk]["field"], []), temp)
            if val is not None:
                ref_parts.append(f"{PROPERTIES[pk]['name']}: {_fmt(val, pk)};")

        # Add density and gamma prime
        cf = alloy.get("computed_features", {})
        d = cf.get("density_calculated_gcm3")
        gp = cf.get("gamma_prime_estimated_vol_pct")
        if d:
            ref_parts.append(f"density: {d:.2f} g/cm³;")
        if gp is not None:
            ref_parts.append(f"γ' fraction: {gp:.1f}%.")

        q_text = (f"Give me the complete property profile of {name} at {temp_str}.")
        questions.append({
            "id": f"ragas_prof_{len(questions)+1:03d}",
            "subtype": "multi_property",
            "question": q_text,
            "ground_truth": " ".join(ref_parts),
        })
        seen_alloys.add(name)

    return questions


# ── Type 6: Temperature trend ─────────────────────────────────────────

def gen_temperature_trend(alloys: list[dict], target: int = 12) -> list[dict]:
    """Type 6: How does property X of alloy Y change with temperature?"""
    prop_keys = ["yield_strength", "uts", "elongation"]
    candidates = []
    for a in alloys:
        for pk in prop_keys:
            pairs = _prop_at_temps(a, pk)
            if len(pairs) >= 4:
                candidates.append((a, pk, pairs))

    random.shuffle(candidates)
    questions = []
    seen: set[tuple] = set()

    for alloy, pk, pairs in candidates:
        if len(questions) >= target:
            break
        key = (alloy["alloy"], pk)
        if key in seen:
            continue

        name = alloy["alloy"]
        trend_str = ", ".join(f"{_fmt(v, pk)} at {t}°C" for t, v in pairs)

        # Compute retention from RT to highest available temp
        rt_val = pairs[0][1]
        ht_val = pairs[-1][1]
        ht_temp = pairs[-1][0]
        if rt_val > 0:
            retention = ht_val / rt_val
            retention_note = (f" The alloy retains {retention:.0%} of its "
                              f"RT value at {ht_temp}°C.")
        else:
            retention_note = ""

        ref = (f"{name} {PROPERTIES[pk]['name']} across temperatures: "
               f"{trend_str}.{retention_note}")

        q_text = (f"How does the {PROPERTIES[pk]['name']} of {name} "
                  f"change with temperature?")
        questions.append({
            "id": f"ragas_trend_{len(questions)+1:03d}",
            "subtype": "temperature_trend",
            "question": q_text,
            "ground_truth": ref,
        })
        seen.add(key)

    return questions


# ── Save ───────────────────────────────────────────────────────────────

def save_jsonl(data: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


# ── Main ───────────────────────────────────────────────────────────────

def main():
    alloys = load_alloys(GROUND_TRUTH)
    print(f"Loaded {len(alloys)} alloys")

    q_desc = gen_descriptive(alloys, target=20)
    q_expl = gen_explanatory(alloys, target=18)
    q_comp = gen_comparative(alloys, target=18)
    q_rec = gen_recommendation(alloys, target=18)
    q_prof = gen_multi_property(alloys, target=14)
    q_trend = gen_temperature_trend(alloys, target=12)

    all_questions = q_desc + q_expl + q_comp + q_rec + q_prof + q_trend

    print(f"\nRAGAS Questions ({len(all_questions)} total):")
    print(f"  descriptive:       {len(q_desc)}")
    print(f"  explanatory:       {len(q_expl)}")
    print(f"  comparative:       {len(q_comp)}")
    print(f"  recommendation:    {len(q_rec)}")
    print(f"  multi_property:    {len(q_prof)}")
    print(f"  temperature_trend: {len(q_trend)}")

    save_jsonl(all_questions, OUTPUT)
    print(f"\nSaved to {OUTPUT.name}")


if __name__ == "__main__":
    main()
