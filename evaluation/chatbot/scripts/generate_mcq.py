"""
Generate Multiple-Choice Questions (MCQ) for chatbot evaluation.

Produces two files:
  - mcq_1hop_questions.jsonl  (100 questions requiring a single KG lookup)
  - mcq_2hop_questions.jsonl  (100 questions requiring two KG lookups / reasoning steps)

Each question has 4 options (A-D), exactly 1 correct answer, and 3 plausible
distractors drawn from real alloy data.

Usage:
    python generate_mcq.py
"""

import json
import random
from collections import defaultdict
from pathlib import Path

from utils import load_alloys, get_rt_value, get_value_at_temp, PROPERTIES

random.seed(42)

ROOT = Path(__file__).resolve().parent.parent.parent.parent
GROUND_TRUTH = ROOT / "backend" / "alloy_crew" / "models" / "training_data" / "train_77alloys.jsonl"
OUT_1HOP = ROOT / "evaluation" / "chatbot" / "data" / "mcq_1hop_questions.jsonl"
OUT_2HOP = ROOT / "evaluation" / "chatbot" / "data" / "mcq_2hop_questions.jsonl"

# ── Question templates ─────────────────────────────────────────────────

PROP_TEMPLATES_RT = [
    "What is the {prop} of {alloy} at room temperature?",
    "What is the room-temperature {prop} of {alloy}?",
    "At room temperature, what is the {prop} of {alloy}?",
]
PROP_TEMPLATES_TEMP = [
    "What is the {prop} of {alloy} at {temp}°C?",
    "At {temp}°C, what is the {prop} of {alloy}?",
]
COMP_TEMPLATES = [
    "What is the {element} content (wt%) of {alloy}?",
    "How much {element} does {alloy} contain (wt%)?",
]
PROC_TEMPLATES = [
    "What processing method is used for {alloy}?",
    "How is {alloy} processed?",
]
DENSITY_TEMPLATES = [
    "What is the density of {alloy}?",
    "What is the calculated density of {alloy} in g/cm³?",
]
TCP_TEMPLATES = [
    "What is the TCP phase risk classification of {alloy}?",
    "What TCP risk level does {alloy} have?",
]
GP_TEMPLATES = [
    "What is the estimated gamma prime (γ') volume fraction of {alloy}?",
    "What percentage of γ' phase does {alloy} contain?",
]


# ── DistractorPool ─────────────────────────────────────────────────────

class DistractorPool:
    """Pre-index alloy data for efficient distractor generation."""

    def __init__(self, alloys: list[dict]):
        self.alloys = alloys
        self.alloy_by_name: dict[str, dict] = {}
        # (prop_key, temp_int) -> [(alloy_name, value)]
        self.prop_index: dict[tuple, list[tuple[str, float]]] = defaultdict(list)
        self.comp_index: dict[str, dict[str, float]] = {}       # alloy -> {El: wt%}
        self.features: dict[str, dict] = {}                      # alloy -> computed_features
        self.processing: dict[str, str] = {}                     # alloy -> cast/wrought
        self._build()

    def _build(self):
        for a in self.alloys:
            name = a["alloy"]
            self.alloy_by_name[name] = a
            self.comp_index[name] = a.get("composition", {})
            self.features[name] = a.get("computed_features", {})
            self.processing[name] = a.get("processing", "")

            for prop_key, meta in PROPERTIES.items():
                if prop_key == "density":
                    d = self.features[name].get("density_calculated_gcm3")
                    if d is not None:
                        self.prop_index[("density", None)].append((name, d))
                    continue
                for m in a.get(meta["field"], []):
                    t = int(m["temp_c"].strip())
                    self.prop_index[(prop_key, t)].append((name, m["value"]))

    # ── Numeric distractors ────────────────────────────────────────────

    def numeric_distractors(self, correct: float, prop_key: str, alloy: str,
                            temp: int | None, n: int = 3,
                            unit: str = "") -> list[dict]:
        """Return n plausible wrong numeric values with source annotations."""
        candidates: list[dict] = []

        # Strategy 1: same alloy, different temperature
        if temp is not None:
            a = self.alloy_by_name.get(alloy)
            if a and prop_key != "density":
                field = PROPERTIES[prop_key]["field"]
                for m in a.get(field, []):
                    t = int(m["temp_c"].strip())
                    if t != temp and not self._too_close(m["value"], correct):
                        candidates.append({
                            "value": m["value"],
                            "source": f"{alloy} {prop_key} at {t}°C",
                        })

        # Strategy 2: same alloy, different property (for strength properties)
        if prop_key in ("yield_strength", "uts"):
            other = "uts" if prop_key == "yield_strength" else "yield_strength"
            a = self.alloy_by_name.get(alloy)
            if a:
                field = PROPERTIES[other]["field"]
                for m in a.get(field, []):
                    t = int(m["temp_c"].strip())
                    if (temp is None or t == temp) and not self._too_close(m["value"], correct):
                        candidates.append({
                            "value": m["value"],
                            "source": f"{alloy} {other} at {t}°C",
                        })

        # Strategy 3: different alloy, same property at same temp
        for other_name, other_val in self.prop_index.get((prop_key, temp), []):
            if other_name != alloy and not self._too_close(other_val, correct):
                candidates.append({
                    "value": other_val,
                    "source": f"{other_name} {prop_key}",
                })

        # Sort by closeness to correct (most confusing first)
        candidates.sort(key=lambda c: abs(c["value"] - correct))

        # Deduplicate by value
        seen_vals: set[float] = set()
        unique: list[dict] = []
        for c in candidates:
            rounded = round(c["value"], 2)
            if rounded not in seen_vals:
                seen_vals.add(rounded)
                unique.append(c)

        # Strategy 4: perturbation fallback
        if len(unique) < n:
            for factor in [0.80, 1.25, 0.70, 1.35, 0.60, 1.50]:
                val = round(correct * factor, 2)
                if val > 0 and not self._too_close(val, correct) and round(val, 2) not in seen_vals:
                    seen_vals.add(round(val, 2))
                    unique.append({"value": val, "source": "perturbation"})

        return unique[:n]

    def alloy_distractors(self, correct_alloy: str,
                          filter_fn=None, n: int = 3) -> list[str]:
        """Return n wrong alloy names. If filter_fn given, all must pass it."""
        pool = [name for name in self.alloy_by_name
                if name != correct_alloy
                and (filter_fn is None or filter_fn(name))]
        random.shuffle(pool)
        return pool[:n]

    @staticmethod
    def _too_close(val: float, ref: float, tol: float = 0.005) -> bool:
        if ref == 0:
            return abs(val) < 0.01
        return abs(val - ref) / abs(ref) < tol


# ── Helpers ────────────────────────────────────────────────────────────

def _format_numeric(val: float, prop_key: str) -> str:
    """Format a numeric value with appropriate precision and unit."""
    unit = PROPERTIES[prop_key]["unit"]
    if prop_key == "density":
        return f"{val:.2f} {unit}"
    if prop_key == "elongation":
        return f"{val:.0f}{unit}"
    return f"{val:.0f} {unit}"


def _format_gp(val: float) -> str:
    return f"{val:.1f}%"


def _append_mcq(lst: list, item: dict | None):
    """Append MCQ to list if not None (i.e., if it passed dedup check)."""
    if item is not None:
        lst.append(item)


def _build_mcq(qid: str, subtype: str, question_text: str,
               correct_val: str, distractors: list[str]) -> dict | None:
    """Build a shuffled MCQ dict with correct answer placed randomly.
    Returns None if distractors contain duplicates or match the correct answer."""
    # Final safety check: ensure all 4 options are unique
    unique_dists = []
    seen = {correct_val}
    for d in distractors:
        if d not in seen:
            seen.add(d)
            unique_dists.append(d)
    if len(unique_dists) < 3:
        return None
    options_list = [correct_val] + unique_dists[:3]
    random.shuffle(options_list)
    correct_letter = chr(65 + options_list.index(correct_val))  # A/B/C/D
    options = {chr(65 + i): v for i, v in enumerate(options_list)}
    opts_text = "\n".join(f"{k}) {v}" for k, v in options.items())
    return {
        "id": qid,
        "subtype": subtype,
        "question": f"{question_text}\n\n{opts_text}",
        "options": options,
        "correct_answer": correct_letter,
    }


# ── 1-Hop Generators ──────────────────────────────────────────────────

def gen_1hop_property(alloys: list[dict], pool: DistractorPool,
                      target: int = 35) -> list[dict]:
    """Property value at specific temperature."""
    candidates = []
    for a in alloys:
        for pk, meta in PROPERTIES.items():
            if pk == "density":
                continue
            field = meta["field"]
            for m in a.get(field, []):
                temp = int(m["temp_c"].strip())
                candidates.append((a["alloy"], pk, temp, m["value"]))

    random.shuffle(candidates)
    questions = []
    seen_alloy_prop: set[tuple] = set()

    for alloy, pk, temp, val in candidates:
        if len(questions) >= target:
            break
        key = (alloy, pk)
        if key in seen_alloy_prop:
            continue

        dists = pool.numeric_distractors(val, pk, alloy, temp)
        if len(dists) < 3:
            continue

        is_rt = temp in (20, 21, 22, 25)
        if is_rt:
            tmpl = random.choice(PROP_TEMPLATES_RT)
            q_text = tmpl.format(prop=PROPERTIES[pk]["name"], alloy=alloy)
        else:
            tmpl = random.choice(PROP_TEMPLATES_TEMP)
            q_text = tmpl.format(prop=PROPERTIES[pk]["name"], alloy=alloy, temp=temp)

        correct_str = _format_numeric(val, pk)
        dist_strs = [_format_numeric(d["value"], pk) for d in dists[:3]]

        _append_mcq(questions, _build_mcq(
            f"mcq_1hop_prop_{len(questions)+1:03d}", "property_at_temp",
            q_text, correct_str, dist_strs,
        ))
        seen_alloy_prop.add(key)

    return questions


def gen_1hop_composition(alloys: list[dict], pool: DistractorPool,
                         target: int = 20) -> list[dict]:
    """Element content (wt%) of a specific alloy."""
    # Pick interesting elements that vary across alloys
    elements = ["Cr", "Co", "Mo", "W", "Al", "Ti", "Fe", "Ta", "Nb", "Re"]
    candidates = []
    for a in alloys:
        comp = a.get("composition", {})
        for el in elements:
            if el in comp and comp[el] >= 0.5:  # meaningful amount
                candidates.append((a["alloy"], el, comp[el]))

    random.shuffle(candidates)
    questions = []
    seen: set[tuple] = set()

    for alloy, el, val in candidates:
        if len(questions) >= target:
            break
        if (alloy, el) in seen:
            continue

        # Distractors: same element in other alloys
        other_vals = [(n, c.get(el, 0))
                      for n, c in pool.comp_index.items()
                      if n != alloy and c.get(el, 0) >= 0.1
                      and not DistractorPool._too_close(c[el], val)]
        other_vals.sort(key=lambda x: abs(x[1] - val))
        if len(other_vals) < 3:
            continue

        correct_str = f"{val:.1f}%"
        # Deduplicate distractors by formatted string (two alloys can share same wt%)
        seen_strs = {correct_str}
        dist_strs = []
        for _, v in other_vals:
            s = f"{v:.1f}%"
            if s not in seen_strs:
                seen_strs.add(s)
                dist_strs.append(s)
            if len(dist_strs) >= 3:
                break
        if len(dist_strs) < 3:
            continue

        tmpl = random.choice(COMP_TEMPLATES)
        q_text = tmpl.format(element=el, alloy=alloy)

        _append_mcq(questions, _build_mcq(
            f"mcq_1hop_comp_{len(questions)+1:03d}", "composition_element",
            q_text, correct_str, dist_strs,
        ))
        seen.add((alloy, el))

    return questions


def gen_1hop_processing(alloys: list[dict], pool: DistractorPool,
                        target: int = 10) -> list[dict]:
    """Processing method identification."""
    options_all = ["wrought", "cast", "single crystal", "directionally solidified"]
    candidates = [(a["alloy"], a["processing"]) for a in alloys
                   if a["processing"] in ("cast", "wrought")]
    random.shuffle(candidates)
    questions = []
    # Balance cast vs wrought
    cast_count = wrought_count = 0
    for alloy, proc in candidates:
        if len(questions) >= target:
            break
        if proc == "cast" and cast_count >= target // 2:
            continue
        if proc == "wrought" and wrought_count >= target // 2:
            continue

        distractors = [o for o in options_all if o != proc][:3]
        tmpl = random.choice(PROC_TEMPLATES)
        q_text = tmpl.format(alloy=alloy)

        _append_mcq(questions, _build_mcq(
            f"mcq_1hop_proc_{len(questions)+1:03d}", "processing_method",
            q_text, proc, distractors,
        ))
        if proc == "cast":
            cast_count += 1
        else:
            wrought_count += 1

    return questions


def gen_1hop_density(alloys: list[dict], pool: DistractorPool,
                     target: int = 15) -> list[dict]:
    """Density lookup."""
    candidates = []
    for a in alloys:
        d = a.get("computed_features", {}).get("density_calculated_gcm3")
        if d is not None:
            candidates.append((a["alloy"], d))
    random.shuffle(candidates)

    questions = []
    for alloy, val in candidates:
        if len(questions) >= target:
            break
        dists = pool.numeric_distractors(val, "density", alloy, None)
        if len(dists) < 3:
            continue

        correct_str = f"{val:.2f} g/cm³"
        dist_strs = [f"{d['value']:.2f} g/cm³" for d in dists[:3]]
        tmpl = random.choice(DENSITY_TEMPLATES)
        q_text = tmpl.format(alloy=alloy)

        _append_mcq(questions, _build_mcq(
            f"mcq_1hop_dens_{len(questions)+1:03d}", "density",
            q_text, correct_str, dist_strs,
        ))
    return questions


def gen_1hop_tcp(alloys: list[dict], pool: DistractorPool,
                 target: int = 10) -> list[dict]:
    """TCP risk classification."""
    tcp_levels = ["Low", "Moderate", "Elevated", "Critical"]
    by_level: dict[str, list[str]] = defaultdict(list)
    for a in alloys:
        risk = a.get("computed_features", {}).get("TCP_risk", "")
        if risk in tcp_levels:
            by_level[risk].append(a["alloy"])

    questions = []
    # Sample from each level proportionally, with more from populated levels
    all_candidates = []
    for level, names in by_level.items():
        for name in names:
            all_candidates.append((name, level))
    random.shuffle(all_candidates)

    seen: set[str] = set()
    for alloy, risk in all_candidates:
        if len(questions) >= target:
            break
        if alloy in seen:
            continue

        distractors = [l for l in tcp_levels if l != risk][:3]
        tmpl = random.choice(TCP_TEMPLATES)
        q_text = tmpl.format(alloy=alloy)

        _append_mcq(questions, _build_mcq(
            f"mcq_1hop_tcp_{len(questions)+1:03d}", "tcp_risk",
            q_text, risk, distractors,
        ))
        seen.add(alloy)
    return questions


def gen_1hop_gamma_prime(alloys: list[dict], pool: DistractorPool,
                         target: int = 10) -> list[dict]:
    """Gamma prime volume fraction."""
    candidates = []
    for a in alloys:
        gp = a.get("computed_features", {}).get("gamma_prime_estimated_vol_pct")
        if gp is not None and gp > 0:
            candidates.append((a["alloy"], gp))
    random.shuffle(candidates)

    questions = []
    for alloy, val in candidates:
        if len(questions) >= target:
            break
        # Distractors: gamma prime values from other alloys
        others = [(n, f.get("gamma_prime_estimated_vol_pct", 0))
                  for n, f in pool.features.items()
                  if n != alloy and f.get("gamma_prime_estimated_vol_pct", 0) > 0
                  and not DistractorPool._too_close(
                      f["gamma_prime_estimated_vol_pct"], val)]
        others.sort(key=lambda x: abs(x[1] - val))
        if len(others) < 3:
            continue

        correct_str = _format_gp(val)
        # Deduplicate distractors by formatted string (two alloys can share same γ'%)
        seen_strs = {correct_str}
        dist_strs = []
        for _, v in others:
            s = _format_gp(v)
            if s not in seen_strs:
                seen_strs.add(s)
                dist_strs.append(s)
            if len(dist_strs) >= 3:
                break
        if len(dist_strs) < 3:
            continue
        tmpl = random.choice(GP_TEMPLATES)
        q_text = tmpl.format(alloy=alloy)

        _append_mcq(questions, _build_mcq(
            f"mcq_1hop_gp_{len(questions)+1:03d}", "gamma_prime",
            q_text, correct_str, dist_strs,
        ))
    return questions


# ── 2-Hop Generators ──────────────────────────────────────────────────

def gen_2hop_comp_filter_rank(alloys: list[dict], pool: DistractorPool,
                              target: int = 18) -> list[dict]:
    """Among alloys with element > X%, which has highest/lowest property?"""
    filter_configs = [
        ("Cr", 20, "greater than 20%"),
        ("Cr", 15, "greater than 15%"),
        ("Co", 15, "greater than 15%"),
        ("Co", 10, "greater than 10%"),
        ("Mo", 5, "greater than 5%"),
        ("W", 3, "greater than 3%"),
        ("Al", 4, "greater than 4%"),
        ("Ti", 3, "greater than 3%"),
        ("Fe", 5, "greater than 5%"),
        ("Nb", 3, "greater than 3%"),
    ]
    prop_keys = ["yield_strength", "uts", "elongation", "density"]
    directions = ["highest", "lowest"]

    all_combos = [(fc, pk, d)
                  for fc in filter_configs
                  for pk in prop_keys
                  for d in directions]
    random.shuffle(all_combos)

    questions = []
    for (element, threshold, desc), prop_key, direction in all_combos:
        if len(questions) >= target:
            break

        # Filter alloys by composition
        filtered = []
        for a in alloys:
            comp = a.get("composition", {})
            if comp.get(element, 0) > threshold:
                if prop_key == "density":
                    d = a.get("computed_features", {}).get("density_calculated_gcm3")
                    if d is not None:
                        filtered.append((a["alloy"], d))
                else:
                    val = get_rt_value(a.get(PROPERTIES[prop_key]["field"], []))
                    if val is not None:
                        filtered.append((a["alloy"], val))

        if len(filtered) < 4:
            continue

        filtered.sort(key=lambda x: x[1], reverse=(direction == "highest"))
        correct_alloy, correct_val = filtered[0]
        distractor_alloys = [name for name, _ in filtered[1:4]]

        q_text = (f"Among alloys with {element} content {desc}, "
                  f"which has the {direction} {PROPERTIES[prop_key]['name']} "
                  f"at room temperature?")

        _append_mcq(questions, _build_mcq(
            f"mcq_2hop_cfr_{len(questions)+1:03d}", "comp_filter_rank",
            q_text, correct_alloy, distractor_alloys,
        ))
    return questions


def gen_2hop_dual_threshold(alloys: list[dict], pool: DistractorPool,
                            target: int = 18) -> list[dict]:
    """Which alloy has BOTH property1 > X AND property2 > Y?"""
    threshold_configs = [
        ("yield_strength", 1000, "yield strength above 1000 MPa",
         "elongation", 15, "elongation above 15%"),
        ("yield_strength", 900, "yield strength above 900 MPa",
         "elongation", 10, "elongation above 10%"),
        ("uts", 1200, "tensile strength above 1200 MPa",
         "elongation", 15, "elongation above 15%"),
        ("yield_strength", 800, "yield strength above 800 MPa",
         "density", 8.2, "density above 8.2 g/cm³"),
        ("uts", 1000, "tensile strength above 1000 MPa",
         "elongation", 20, "elongation above 20%"),
        ("yield_strength", 700, "yield strength above 700 MPa",
         "elongation", 20, "elongation above 20%"),
        ("uts", 900, "tensile strength above 900 MPa",
         "density", 7.5, "density below 7.5 g/cm³"),
        ("elongation", 30, "elongation above 30%",
         "uts", 700, "tensile strength above 700 MPa"),
        ("yield_strength", 600, "yield strength above 600 MPa",
         "elongation", 25, "elongation above 25%"),
        ("uts", 800, "tensile strength above 800 MPa",
         "yield_strength", 600, "yield strength above 600 MPa"),
    ]

    random.shuffle(threshold_configs)
    questions = []

    for pk1, t1, desc1, pk2, t2, desc2 in threshold_configs:
        if len(questions) >= target:
            break

        def _get_val(a, pk):
            if pk == "density":
                return a.get("computed_features", {}).get("density_calculated_gcm3")
            return get_rt_value(a.get(PROPERTIES[pk]["field"], []))

        # Find alloys satisfying BOTH conditions
        both = []
        only_first = []
        only_second = []
        for a in alloys:
            v1 = _get_val(a, pk1)
            v2 = _get_val(a, pk2)
            if v1 is None or v2 is None:
                continue
            # Handle "below" for density threshold
            passes1 = (v1 < t1) if "below" in desc1 else (v1 > t1)
            passes2 = (v2 < t2) if "below" in desc2 else (v2 > t2)
            if passes1 and passes2:
                both.append(a["alloy"])
            elif passes1 and not passes2:
                only_first.append(a["alloy"])
            elif not passes1 and passes2:
                only_second.append(a["alloy"])

        if not both or len(only_first) + len(only_second) < 3:
            continue

        correct_alloy = random.choice(both)
        # Distractors: alloys that fail at least one condition
        dist_pool = only_first + only_second
        random.shuffle(dist_pool)
        distractors = dist_pool[:3]
        if len(distractors) < 3:
            continue

        q_text = (f"Which of these alloys has BOTH {desc1} AND {desc2} "
                  f"at room temperature?")

        _append_mcq(questions, _build_mcq(
            f"mcq_2hop_dual_{len(questions)+1:03d}", "dual_threshold",
            q_text, correct_alloy, distractors,
        ))

    # Generate more by varying thresholds if needed
    if len(questions) < target:
        extra_configs = [
            ("yield_strength", 500, "yield strength above 500 MPa",
             "elongation", 30, "elongation above 30%"),
            ("uts", 1100, "tensile strength above 1100 MPa",
             "yield_strength", 900, "yield strength above 900 MPa"),
            ("elongation", 20, "elongation above 20%",
             "density", 8.0, "density below 8.0 g/cm³"),
            ("yield_strength", 800, "yield strength above 800 MPa",
             "uts", 1100, "tensile strength above 1100 MPa"),
            ("uts", 700, "tensile strength above 700 MPa",
             "elongation", 40, "elongation above 40%"),
            ("yield_strength", 1100, "yield strength above 1100 MPa",
             "uts", 1300, "tensile strength above 1300 MPa"),
            ("elongation", 10, "elongation above 10%",
             "yield_strength", 900, "yield strength above 900 MPa"),
            ("uts", 1300, "tensile strength above 1300 MPa",
             "elongation", 10, "elongation above 10%"),
        ]
        for pk1, t1, desc1, pk2, t2, desc2 in extra_configs:
            if len(questions) >= target:
                break

            def _get_val(a, pk):
                if pk == "density":
                    return a.get("computed_features", {}).get("density_calculated_gcm3")
                return get_rt_value(a.get(PROPERTIES[pk]["field"], []))

            both = []
            only_first = []
            only_second = []
            for a in alloys:
                v1 = _get_val(a, pk1)
                v2 = _get_val(a, pk2)
                if v1 is None or v2 is None:
                    continue
                passes1 = (v1 < t1) if "below" in desc1 else (v1 > t1)
                passes2 = (v2 < t2) if "below" in desc2 else (v2 > t2)
                if passes1 and passes2:
                    both.append(a["alloy"])
                elif passes1:
                    only_first.append(a["alloy"])
                elif passes2:
                    only_second.append(a["alloy"])

            if not both or len(only_first) + len(only_second) < 3:
                continue

            correct_alloy = random.choice(both)
            dist_pool = only_first + only_second
            random.shuffle(dist_pool)
            distractors = dist_pool[:3]
            if len(distractors) < 3:
                continue

            q_text = (f"Which of these alloys has BOTH {desc1} AND {desc2} "
                      f"at room temperature?")

            _append_mcq(questions, _build_mcq(
                f"mcq_2hop_dual_{len(questions)+1:03d}", "dual_threshold",
                q_text, correct_alloy, distractors,
            ))

    return questions


def gen_2hop_pairwise(alloys: list[dict], pool: DistractorPool,
                      target: int = 18) -> list[dict]:
    """At temperature T, which of these 4 alloys has the highest/lowest property?"""
    temps = [427, 538, 649, 760, 871, 982, 1093]
    prop_keys = ["yield_strength", "uts", "elongation", "elasticity"]
    directions = ["highest", "lowest"]
    combos = [(pk, t, d) for pk in prop_keys for t in temps for d in directions]
    random.shuffle(combos)

    questions = []
    for pk, temp, direction in combos:
        if len(questions) >= target:
            break

        # Find alloys with data at this temperature
        scored = []
        for a in alloys:
            val = get_value_at_temp(a.get(PROPERTIES[pk]["field"], []), temp)
            if val is not None:
                scored.append((a["alloy"], val))

        if len(scored) < 4:
            continue

        scored.sort(key=lambda x: x[1], reverse=(direction == "highest"))
        correct_alloy, correct_val = scored[0]
        rest = scored[1:]
        random.shuffle(rest)
        distractors = [name for name, _ in rest[:3]]

        q_text = (f"At {temp}°C, which of these alloys has the {direction} "
                  f"{PROPERTIES[pk]['name']}?")

        _append_mcq(questions, _build_mcq(
            f"mcq_2hop_pair_{len(questions)+1:03d}", "pairwise_comparison",
            q_text, correct_alloy, distractors,
        ))
    return questions


def gen_2hop_temp_retention(alloys: list[dict], pool: DistractorPool,
                            target: int = 16) -> list[dict]:
    """Which alloy retains the most/least of its RT strength at temp T?

    Only uses YS and UTS (not elongation, which can increase at high temp).
    """
    temps = [538, 649, 760, 871, 982]
    prop_keys = ["yield_strength", "uts"]
    directions = ["most", "least"]
    combos = [(pk, t, d) for pk in prop_keys for t in temps for d in directions]
    random.shuffle(combos)

    questions = []
    for pk, temp, direction in combos:
        if len(questions) >= target:
            break

        scored = []
        for a in alloys:
            field = PROPERTIES[pk]["field"]
            rt = get_rt_value(a.get(field, []))
            ht = get_value_at_temp(a.get(field, []), temp)
            if rt and ht and rt > 0:
                retention = ht / rt
                scored.append((a["alloy"], retention, rt, ht))

        if len(scored) < 4:
            continue

        scored.sort(key=lambda x: x[1], reverse=(direction == "most"))
        correct_alloy, correct_ret, correct_rt, correct_ht = scored[0]
        rest = scored[1:]
        random.shuffle(rest)
        distractors = [name for name, _, _, _ in rest[:3]]

        q_text = (f"Which of these alloys retains the {direction} of its "
                  f"room-temperature {PROPERTIES[pk]['name']} at {temp}°C?")

        _append_mcq(questions, _build_mcq(
            f"mcq_2hop_ret_{len(questions)+1:03d}", "temp_retention",
            q_text, correct_alloy, distractors,
        ))
    return questions


def gen_2hop_filtered_ranking(alloys: list[dict], pool: DistractorPool,
                              target: int = 14) -> list[dict]:
    """Among cast/wrought alloys, which has the highest/lowest property?"""
    filter_configs = [
        ("cast", lambda a: a["processing"] == "cast", "cast alloys"),
        ("wrought", lambda a: a["processing"] == "wrought", "wrought alloys"),
    ]
    prop_keys = ["yield_strength", "uts", "elongation", "density"]
    directions = ["highest", "lowest"]

    combos = [(fc, pk, d)
              for fc in filter_configs
              for pk in prop_keys
              for d in directions]
    random.shuffle(combos)

    questions = []
    for (proc_key, filter_fn, desc), pk, direction in combos:
        if len(questions) >= target:
            break

        scored = []
        for a in alloys:
            if not filter_fn(a):
                continue
            if pk == "density":
                val = a.get("computed_features", {}).get("density_calculated_gcm3")
            else:
                val = get_rt_value(a.get(PROPERTIES[pk]["field"], []))
            if val is not None:
                scored.append((a["alloy"], val))

        if len(scored) < 4:
            continue

        scored.sort(key=lambda x: x[1], reverse=(direction == "highest"))
        correct_alloy, correct_val = scored[0]
        distractors = [name for name, _ in scored[1:4]]

        q_text = (f"Among {desc}, which has the {direction} "
                  f"{PROPERTIES[pk]['name']} at room temperature?")

        _append_mcq(questions, _build_mcq(
            f"mcq_2hop_filt_{len(questions)+1:03d}", "filtered_ranking",
            q_text, correct_alloy, distractors,
        ))
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
    pool = DistractorPool(alloys)

    # ── 1-Hop ──
    q1_prop = gen_1hop_property(alloys, pool, target=35)
    q1_comp = gen_1hop_composition(alloys, pool, target=20)
    q1_proc = gen_1hop_processing(alloys, pool, target=10)
    q1_dens = gen_1hop_density(alloys, pool, target=15)
    q1_tcp = gen_1hop_tcp(alloys, pool, target=10)
    q1_gp = gen_1hop_gamma_prime(alloys, pool, target=10)

    all_1hop = q1_prop + q1_comp + q1_proc + q1_dens + q1_tcp + q1_gp
    random.shuffle(all_1hop)

    print(f"\n1-Hop MCQ ({len(all_1hop)} total):")
    print(f"  property_at_temp:   {len(q1_prop)}")
    print(f"  composition_element:{len(q1_comp)}")
    print(f"  processing_method:  {len(q1_proc)}")
    print(f"  density:            {len(q1_dens)}")
    print(f"  tcp_risk:           {len(q1_tcp)}")
    print(f"  gamma_prime:        {len(q1_gp)}")

    # ── 2-Hop ──
    # (cross_property removed: unanswerable without KG, unfair to vanilla LLMs)
    q2_cfr = gen_2hop_comp_filter_rank(alloys, pool, target=23)
    q2_dual = gen_2hop_dual_threshold(alloys, pool, target=18)
    q2_pair = gen_2hop_pairwise(alloys, pool, target=23)
    q2_ret = gen_2hop_temp_retention(alloys, pool, target=20)
    q2_filt = gen_2hop_filtered_ranking(alloys, pool, target=16)

    all_2hop = q2_cfr + q2_dual + q2_pair + q2_ret + q2_filt
    random.shuffle(all_2hop)

    print(f"\n2-Hop MCQ ({len(all_2hop)} total):")
    print(f"  comp_filter_rank:   {len(q2_cfr)}")
    print(f"  dual_threshold:     {len(q2_dual)}")
    print(f"  pairwise_comparison:{len(q2_pair)}")
    print(f"  temp_retention:     {len(q2_ret)}")
    print(f"  filtered_ranking:   {len(q2_filt)}")

    save_jsonl(all_1hop, OUT_1HOP)
    save_jsonl(all_2hop, OUT_2HOP)
    print(f"\nSaved {len(all_1hop)} 1-hop MCQs to {OUT_1HOP.name}")
    print(f"Saved {len(all_2hop)} 2-hop MCQs to {OUT_2HOP.name}")


if __name__ == "__main__":
    main()
