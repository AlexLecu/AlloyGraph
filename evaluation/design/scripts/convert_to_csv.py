"""Convert design_evaluation_results.jsonl to expert-ready CSV.

Produces a clean spreadsheet for materials scientist review with:
  - Design context (temperature, processing route)
  - Composition (wt%), only elements actually used
  - Target vs predicted properties with % achievement and confidence intervals
  - Density and γ' with target columns (when constrained)
  - Phase stability metrics (Md, TCP, γ/γ' misfit)
  - Phase 1 and Phase 2 optimizer diagnostics
  - Agent corrections and key issues
"""
import csv
import json
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
JSONL_PATH = RESULTS_DIR / "design_evaluation_results.jsonl"
CSV_PATH = RESULTS_DIR / "design_evaluation_results.csv"

# Elements in conventional superalloy ordering (base → major → minor → trace)
ELEMENTS = ["Ni", "Cr", "Co", "Mo", "W", "Al", "Ti", "Ta", "Nb", "Re", "Hf",
            "Fe", "C", "B", "Zr", "V", "Mn", "Si", "Cu", "Ru"]

PROPERTIES = ["Yield Strength", "Tensile Strength", "Elongation", "Elastic Modulus"]
UNITS = {"Yield Strength": "MPa", "Tensile Strength": "MPa",
         "Elongation": "%", "Elastic Modulus": "GPa"}
SHORT = {"Yield Strength": "YS", "Tensile Strength": "UTS",
         "Elongation": "El", "Elastic Modulus": "EM"}

records = []
with open(JSONL_PATH) as f:
    for line in f:
        line = line.strip()
        if line:
            records.append(json.loads(line))

# --- Detect which elements are actually used across all alloys ---
used_elements = []
for el in ELEMENTS:
    if any(r.get("composition", {}).get(el, 0) > 0 for r in records):
        used_elements.append(el)

# --- Header ---
header = ["Alloy", "Processing", "Temperature (°C)", "Status"]

# Composition — only elements present in at least one alloy
header += [f"{el} (wt%)" for el in used_elements]

# Targets and predictions side-by-side with % achievement
for p in PROPERTIES:
    s, u = SHORT[p], UNITS[p]
    header += [f"Target {s} ({u})", f"Predicted {s} ({u})",
               f"{s} Achievement (%)", f"{s} 95% CI ({u})"]

# Density and γ' (with optional targets)
header += ["Target Density (g/cm³)", "Predicted Density (g/cm³)",
           "Target γ' (vol%)", "Predicted γ' (vol%)"]

# Phase stability
header += ["Md_avg", "Md_matrix", "TCP Risk", "γ/γ' Misfit (%)",
           "Refractory (wt%)", "Al+Ti (wt%)"]

# Optimizer diagnostics
header += ["Phase 1 Attempts", "Phase 1 Best TCP",
           "Guard Fixes", "Tuner Converged", "Tuner Steps"]

# Assessment
header += ["Corrections Applied", "Key Issues", "Design Time (s)"]

# --- Rows ---
rows = []
for r in records:
    err = r.get("error")
    status = "FAILED" if err else r.get("design_status", "unknown").upper()
    proc = r.get("processing", "")
    temp = r.get("temperature", "")

    row = [r["id"], proc, temp, status]

    # Composition — omit zero elements for readability
    comp = r.get("composition", {})
    for el in used_elements:
        v = comp.get(el, 0.0)
        row.append(round(v, 2) if v > 0 else "")

    # Target vs predicted with achievement % and CI
    targets = r.get("targets", {})
    props = r.get("properties", {})
    intervals = r.get("property_intervals", {})
    for p in PROPERTIES:
        target_val = targets.get(p, "")
        pred_val = props.get(p, "")
        # Achievement %
        if target_val and pred_val and isinstance(target_val, (int, float)) and isinstance(pred_val, (int, float)) and target_val > 0:
            achievement = round(pred_val / target_val * 100, 1)
        else:
            achievement = ""
        # Round prediction
        if isinstance(pred_val, (int, float)):
            pred_val = round(pred_val, 1)
        # Confidence interval
        iv = intervals.get(p, {})
        lo = iv.get("lower", 0)
        hi = iv.get("upper", 0)
        ci_str = f"{round(lo, 1)}–{round(hi, 1)}" if lo and hi else ""
        row += [target_val if target_val else "", pred_val, achievement, ci_str]

    # Density and γ' (with targets when specified)
    target_density = targets.get("Density", "")
    pred_density = props.get("Density", "")
    target_gp = targets.get("Gamma Prime", "")
    pred_gp = props.get("Gamma Prime", "")
    if isinstance(pred_density, (int, float)):
        pred_density = round(pred_density, 2)
    if isinstance(pred_gp, (int, float)):
        pred_gp = round(pred_gp, 1)
    row += [target_density if target_density else "",
            pred_density,
            target_gp if target_gp else "",
            pred_gp]

    # Phase stability
    met = r.get("metallurgy_metrics", {})
    penalties = r.get("audit_penalties", [])
    # Extract Md_avg from metallurgy_metrics or penalty text
    md_avg_val = ""
    for pen in penalties:
        val_str = pen.get("value", "")
        if "Md_avg=" in val_str:
            try:
                md_avg_val = val_str.split("Md_avg=")[1].split(")")[0].strip()
            except (IndexError, ValueError):
                pass
            break
    # Also check optimization_log for computed Md
    if not md_avg_val:
        # Md_avg not in penalties means Low TCP — compute from features if available
        opt_log = r.get("optimization_log", {})
        p1_log = opt_log.get("phase1_log", [])
        # Use guard_fixes or just leave empty
        pass

    row += [
        md_avg_val if md_avg_val else met.get("Md (TCP Stability)", ""),
        met.get("Md (TCP Stability)", ""),
        r.get("tcp_risk", met.get("TCP Risk", "")),
        met.get("γ/γ' Misfit (%)", ""),
        met.get("Refractory Content (wt%)", ""),
        met.get("Al+Ti (weldability)", ""),
    ]

    # Optimizer diagnostics
    opt = r.get("optimization_log", {})
    p1_log = opt.get("phase1_log", [])
    p1_attempts = opt.get("phase1_attempts", len(p1_log) if p1_log else "")
    # Best TCP from Phase 1 log
    best_tcp_p1 = ""
    tcp_rank = {"Low": 0, "Moderate": 1, "Elevated": 2, "Critical": 3}
    for entry in p1_log:
        if entry.get("status") == "ok":
            tcp_val = entry.get("tcp", "")
            if not best_tcp_p1 or tcp_rank.get(tcp_val, 4) < tcp_rank.get(best_tcp_p1, 4):
                best_tcp_p1 = tcp_val
    guard_fixes = opt.get("guard_fixes", [])
    guard_summary = f"{len(guard_fixes)} fixes" if guard_fixes else "None"
    converged = opt.get("converged", "")
    steps = opt.get("steps_used", "")

    row += [p1_attempts, best_tcp_p1, guard_summary,
            converged if converged != "" else "", steps]

    # Corrections — concise summary
    corrections = r.get("corrections_applied", [])
    if corrections:
        parts = []
        for c in corrections:
            pname = (c["property_name"]
                     .replace("Yield Strength", "YS")
                     .replace("Tensile Strength", "UTS")
                     .replace("Elastic Modulus", "EM")
                     .replace("Elongation", "El"))
            orig = c.get("original_value", "?")
            corr = c.get("corrected_value", "?")
            if isinstance(orig, float):
                orig = round(orig, 1)
            if isinstance(corr, float):
                corr = round(corr, 1)
            parts.append(f"{pname}: {orig}→{corr}")
        row.append("; ".join(parts))
    else:
        row.append("None")

    # Key issues — concise, deduplicated
    tcp_risk = r.get("tcp_risk", met.get("TCP Risk", ""))
    issues_parts = []
    if err:
        issues_parts.append(f"FAILED: {err}")
    if tcp_risk in ("Elevated", "Critical"):
        issues_parts.append(f"TCP {tcp_risk}")
    for pen in penalties:
        name = pen.get("name", "")
        if "Coherency" in name:
            misfit = met.get("γ/γ' Misfit (%)", "?")
            issues_parts.append(f"Misfit {misfit}% > 0.8%")
    # Target misses
    for p in PROPERTIES:
        t = targets.get(p)
        v = props.get(p)
        if t and v and isinstance(t, (int, float)) and isinstance(v, (int, float)):
            if v < t:
                pct = round(v / t * 100)
                if pct < 90:
                    issues_parts.append(f"{SHORT[p]} {pct}% of target")
    row.append("; ".join(issues_parts) if issues_parts else "None")

    # Elapsed time
    row.append(r.get("elapsed_s", ""))

    rows.append(row)

with open(CSV_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(rows)

print(f"Wrote {len(rows)} alloys to {CSV_PATH}")
print(f"Columns: {len(header)}")
print(f"Elements included: {', '.join(used_elements)}")
