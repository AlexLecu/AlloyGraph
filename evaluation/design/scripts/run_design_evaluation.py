"""
Batch evaluation of the alloy designer.

Runs a set of target property specifications through the IterativeDesignCrew
and records the resulting compositions, predicted properties, and audit info.

Usage:
    cd AlloyGraph
    python -m evaluation.design.scripts.run_design_evaluation
    # or
    python evaluation/design/scripts/run_design_evaluation.py
"""

import gc
import json
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path  (scripts/ → design/ → evaluation/ → project root)
ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

from backend.alloy_crew.alloy_designer import IterativeDesignCrew


def reset_crewai_state():
    """Reset CrewAI event bus to prevent 'Event stack depth limit (100)' crashes.

    The designer spawns multiple CrewAI crews per iteration (Designer, Evaluator,
    Optimizer).  Their events accumulate on a global bus and eventually exceed the
    100-event depth limit, killing the Optimization Advisor.  This function:
    1. Disables the depth limit (set max_stack_depth=0)
    2. Clears the event stack
    3. Resets legacy event bus attributes
    """
    # Disable the event stack depth limit (0 = no limit)
    try:
        from crewai.events.event_context import EventContextConfig, _event_context_config, _event_id_stack
        config = EventContextConfig(max_stack_depth=0)
        _event_context_config.set(config)
        _event_id_stack.set(())
    except ImportError:
        pass

    # Legacy event bus cleanup (older CrewAI versions)
    for module_path in [
        "crewai.utilities.events",
        "crewai.utilities.event_bus",
        "crewai.telemetry",
    ]:
        try:
            module = __import__(module_path, fromlist=["crewai_event_bus", "event_bus"])
            for attr in ["crewai_event_bus", "event_bus", "_event_bus"]:
                if hasattr(module, attr):
                    bus = getattr(module, attr)
                    if hasattr(bus, "_event_stack"):
                        bus._event_stack.clear()
                    if hasattr(bus, "reset"):
                        bus.reset()
                    if hasattr(bus, "_events"):
                        bus._events.clear()
        except (ImportError, AttributeError):
            pass
    gc.collect()

OUTPUT_DIR = ROOT / "evaluation" / "design" / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_FILE = OUTPUT_DIR / "design_evaluation_results.jsonl"

# ── Target specifications ────────────────────────────────────────────────
TARGETS = [
    {
        "id": "Alloy_A",
        "target_props": {"Yield Strength": 1080, "Tensile Strength": 1550, "Elastic Modulus": 200},
        "temperature": 20, "processing": "wrought",
    },
    {
        "id": "Alloy_B",
        "target_props": {"Yield Strength": 1150, "Tensile Strength": 1640, "Elastic Modulus": 210},
        "temperature": 20, "processing": "wrought",
    },
    {
        "id": "Alloy_C",
        "target_props": {"Yield Strength": 1180, "Tensile Strength": 1610, "Elastic Modulus": 220},
        "temperature": 20, "processing": "wrought",
    },
    {
        "id": "Alloy_D",
        "target_props": {"Yield Strength": 1230, "Tensile Strength": 1600, "Elastic Modulus": 210},
        "temperature": 20, "processing": "wrought",
    },
    {
        "id": "Alloy_E",
        "target_props": {"Yield Strength": 1080, "Tensile Strength": 1560, "Elastic Modulus": 220},
        "temperature": 20, "processing": "wrought",
    },
    {
        "id": "Alloy_F",
        "target_props": {"Yield Strength": 1150, "Tensile Strength": 1600, "Elastic Modulus": 210},
        "temperature": 20, "processing": "wrought",
    },
    {
        "id": "Alloy_G",
        "target_props": {"Yield Strength": 1090, "Tensile Strength": 1590, "Elastic Modulus": 210},
        "temperature": 20, "processing": "wrought",
    },
    {
        "id": "Alloy_H",
        "target_props": {"Yield Strength": 1040, "Tensile Strength": 1520, "Elastic Modulus": 210},
        "temperature": 20, "processing": "wrought",
    },
    # ── Diverse targets ─────────────────────────────────────────────────
    # Achievable wrought RT — moderate-strength γ' with high ductility
    {
        "id": "Alloy_I",
        "target_props": {"Yield Strength": 830, "Tensile Strength": 1240, "Elongation": 25, "Elastic Modulus": 210},
        "temperature": 20, "processing": "wrought",
    },
    # High-temperature wrought — 760°C turbine disc service
    {
        "id": "Alloy_J",
        "target_props": {"Yield Strength": 750, "Tensile Strength": 900, "Elongation": 15, "Elastic Modulus": 170},
        "temperature": 760, "processing": "wrought",
    },
    # High-temperature wrought — 871°C extreme disc/ring
    {
        "id": "Alloy_K",
        "target_props": {"Yield Strength": 520, "Tensile Strength": 620, "Elongation": 20, "Elastic Modulus": 150},
        "temperature": 871, "processing": "wrought",
    },
    # Cast γ' RT — investment-cast turbine component
    {
        "id": "Alloy_L",
        "target_props": {"Yield Strength": 850, "Tensile Strength": 1100, "Elongation": 8, "Elastic Modulus": 210},
        "temperature": 20, "processing": "cast",
    },
    # Cast moderate-temperature — 760°C turbine vane/nozzle
    {
        "id": "Alloy_M",
        "target_props": {"Yield Strength": 700, "Tensile Strength": 870, "Elongation": 8, "Elastic Modulus": 175},
        "temperature": 760, "processing": "cast",
    },
    # Cast high-temperature — 871°C turbine blade conditions
    {
        "id": "Alloy_N",
        "target_props": {"Yield Strength": 580, "Tensile Strength": 720, "Elongation": 10, "Elastic Modulus": 160},
        "temperature": 871, "processing": "cast",
    },
    # Balanced wrought — high ductility + moderate strength
    {
        "id": "Alloy_O",
        "target_props": {"Yield Strength": 900, "Tensile Strength": 1350, "Elongation": 25, "Elastic Modulus": 215},
        "temperature": 20, "processing": "wrought",
    },
    # Lightweight wrought disc — density-constrained design
    {
        "id": "Alloy_P",
        "target_props": {"Yield Strength": 950, "Tensile Strength": 1400, "Density": 8.0, "Elastic Modulus": 200},
        "temperature": 20, "processing": "wrought",
    },
    # ── γ' and density constrained targets ─────────────────────────────
    # Wrought disc with explicit γ' target — Waspaloy-class (~25% γ')
    {
        "id": "Alloy_Q",
        "target_props": {"Yield Strength": 800, "Tensile Strength": 1250, "Elongation": 25, "Gamma Prime": 25, "Density": 8.2},
        "temperature": 20, "processing": "wrought",
    },
    # High-γ' wrought disc — LSHR/ME3-class (~40% γ'), lightweight
    {
        "id": "Alloy_R",
        "target_props": {"Yield Strength": 1100, "Tensile Strength": 1550, "Gamma Prime": 40, "Density": 8.1, "Elastic Modulus": 210},
        "temperature": 20, "processing": "wrought",
    },
    # Cast with γ' + density — IN-738-class (~45% γ')
    {
        "id": "Alloy_S",
        "target_props": {"Yield Strength": 850, "Tensile Strength": 1050, "Elongation": 7, "Gamma Prime": 45, "Density": 8.3},
        "temperature": 20, "processing": "cast",
    },
    # High-temperature wrought with γ' — 760°C disc, controlled γ' for ductility
    {
        "id": "Alloy_T",
        "target_props": {"Yield Strength": 700, "Tensile Strength": 950, "Elongation": 18, "Gamma Prime": 30, "Elastic Modulus": 175},
        "temperature": 760, "processing": "wrought",
    },
]

MAX_ITERATIONS = 5


def run_one(spec: dict) -> dict:
    """Run the designer for a single target spec and return the result record."""
    alloy_id = spec["id"]
    target = spec["target_props"]
    temp = spec["temperature"]
    proc = spec["processing"]

    print(f"\n{'='*60}")
    print(f"  Designing {alloy_id}  |  {proc} @ {temp}°C")
    print(f"  Targets: {target}")
    print(f"{'='*60}")

    reset_crewai_state()

    t0 = time.time()
    engine = IterativeDesignCrew(target)
    result = engine.loop(
        max_iterations=MAX_ITERATIONS,
        start_composition=None,
        temperature=temp,
        processing=proc,
    )
    elapsed = time.time() - t0

    record = {
        "id": alloy_id,
        "targets": target,
        "temperature": temp,
        "processing": proc,
        "elapsed_s": round(elapsed, 1),
        "composition": result.get("composition", {}),
        "properties": result.get("properties", {}),
        "property_intervals": result.get("property_intervals", {}),
        "tcp_risk": result.get("tcp_risk", "Unknown"),
        "penalty_score": result.get("penalty_score", 0),
        "audit_penalties": result.get("audit_penalties", []),
        "confidence": result.get("confidence", {}),
        "design_status": result.get("design_status", "unknown"),
        "metallurgy_metrics": result.get("metallurgy_metrics", {}),
        "explanation": result.get("explanation", ""),
        "error": result.get("error"),
        # Agent reasoning and corrections (same data as webapp)
        "analyst_reasoning": result.get("analyst_reasoning", ""),
        "reviewer_assessment": result.get("reviewer_assessment", ""),
        "corrections_applied": result.get("corrections_applied", []),
        # Design loop metadata
        "iterations_used": result.get("iterations_used"),
        "optimization_log": result.get("optimization_log", {}),
        "issues": result.get("issues", []),
        "recommendations": result.get("recommendations", []),
    }

    # Quick summary
    props = record["properties"]
    status = "OK" if record.get("error") is None else "FAIL"
    print(f"\n  Result ({status}, {elapsed:.0f}s):")
    print(f"    YS={props.get('Yield Strength','?')}  UTS={props.get('Tensile Strength','?')}  "
          f"EM={props.get('Elastic Modulus','?')}  El={props.get('Elongation','?')}  "
          f"TCP={record['tcp_risk']}")

    return record


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run alloy design evaluation")
    parser.add_argument("--only", nargs="+", metavar="ID",
                        help="Run only these alloy IDs (e.g. --only Alloy_Q Alloy_R)")
    parser.add_argument("--fresh", action="store_true",
                        help="Ignore existing results for selected alloys (re-run them)")
    args = parser.parse_args()

    # Filter targets if --only specified
    targets = TARGETS
    if args.only:
        targets = [s for s in TARGETS if s["id"] in args.only]
        missing = set(args.only) - {s["id"] for s in targets}
        if missing:
            print(f"WARNING: Unknown alloy IDs: {missing}")
        if not targets:
            print("No matching alloy IDs found.")
            return

    # Resume support: skip already-completed alloys
    done_ids = set()
    if RESULTS_FILE.exists() and not args.fresh:
        with open(RESULTS_FILE) as f:
            for line in f:
                if line.strip():
                    done_ids.add(json.loads(line)["id"])
    elif RESULTS_FILE.exists() and args.fresh and args.only:
        # Remove old results for selected alloys, keep the rest
        kept = []
        with open(RESULTS_FILE) as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    if rec["id"] not in args.only:
                        kept.append(line.strip())
                    else:
                        print(f"  Removing old result for {rec['id']}")
        with open(RESULTS_FILE, "w") as f:
            for line in kept:
                f.write(line + "\n")

    remaining = [s for s in targets if s["id"] not in done_ids]

    if not remaining:
        print("All selected alloys already evaluated. Use --fresh to re-run.")
        return

    if done_ids & {s["id"] for s in targets}:
        skipped = done_ids & {s["id"] for s in targets}
        print(f"Skipping already done: {skipped}")

    print(f"Running {len(remaining)} alloy(s): {[s['id'] for s in remaining]}")

    with open(RESULTS_FILE, "a") as f:
        for i, spec in enumerate(remaining):
            record = run_one(spec)
            f.write(json.dumps(record) + "\n")
            f.flush()

    print(f"\nResults saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
