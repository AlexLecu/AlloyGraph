"""
De-anonymise and analyse the expert grading sheet.

Reads the filled-in grading CSV + the randomisation key, maps
System A/B/C back to chatbot / llama / gpt, and produces:
  - Per-system aggregate scores (mean ± std for each criterion)
  - Per-question breakdown
  - Statistical significance tests (Friedman + Wilcoxon signed-rank)
  - Saves results to expert_scores.json

Usage:
    python score_expert_grading.py [--grading-csv PATH]
"""

import argparse
import csv
import json
import statistics
from pathlib import Path

from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "results"

DEFAULT_GRADING_CSV = Path.home() / "Downloads" / "exam_questions_grading - grading_sheet.csv"
GRADING_KEY = OUTPUT / "expert_grading_key.json"
SCORES_OUT = OUTPUT / "expert_scores.json"

CRITERIA = ["correctness", "completeness", "relevance"]


def load_grading_key() -> dict:
    with open(GRADING_KEY) as f:
        return json.load(f)


def parse_grading_csv(csv_path: Path) -> list[dict]:
    """Parse the multiline grading CSV into clean rows."""
    rows = []
    warnings = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Need at least 2 of 3 criteria filled to include the row
            scores = {}
            filled = 0
            for c in CRITERIA:
                val = row.get(c, "").strip()
                if val:
                    try:
                        scores[c] = float(val)
                        filled += 1
                    except ValueError:
                        pass
            if filled == 0:
                continue
            if filled < 3:
                qid = row["question_id"].strip()
                sys_label = row["system"].strip()
                missing = [c for c in CRITERIA if c not in scores]
                warnings.append(f"  {qid} {sys_label}: missing {missing}")
            entry = {
                "question_id": row["question_id"].strip(),
                "system_label": row["system"].strip(),
            }
            for c in CRITERIA:
                entry[c] = scores.get(c)  # None if missing
            rows.append(entry)
    if warnings:
        print(f"WARNING: {len(warnings)} rows with partial scores:")
        for w in warnings:
            print(w)
    return rows
    return rows


def de_anonymise(rows: list[dict], key: dict) -> list[dict]:
    """Replace 'System A/B/C' with actual system names using the key."""
    for row in rows:
        qid = row["question_id"]
        label = row["system_label"].replace("System ", "")  # "A", "B", "C"
        qid_key = key.get(qid, {})
        row["system"] = qid_key.get(label, f"unknown-{label}")
    return rows


def aggregate_by_system(rows: list[dict]) -> dict:
    """Compute mean ± std per criterion per system."""
    by_system = {}
    for row in rows:
        sys = row["system"]
        if sys not in by_system:
            by_system[sys] = {c: [] for c in CRITERIA}
        for c in CRITERIA:
            if row[c] is not None:
                by_system[sys][c].append(row[c])

    results = {}
    for sys, scores in sorted(by_system.items()):
        results[sys] = {}
        all_scores = []
        for c in CRITERIA:
            vals = scores[c]
            all_scores.extend(vals)
            results[sys][c] = {
                "mean": round(statistics.mean(vals), 3),
                "std": round(statistics.stdev(vals), 3) if len(vals) > 1 else 0,
                "n": len(vals),
            }
        results[sys]["overall"] = {
            "mean": round(statistics.mean(all_scores), 3),
            "std": round(statistics.stdev(all_scores), 3) if len(all_scores) > 1 else 0,
        }
    return results


def per_question_breakdown(rows: list[dict]) -> list[dict]:
    """Group scores by question, showing all 3 systems side by side."""
    by_q = {}
    for row in rows:
        qid = row["question_id"]
        if qid not in by_q:
            by_q[qid] = {}
        valid = [row[c] for c in CRITERIA if row[c] is not None]
        avg = round(statistics.mean(valid), 2) if valid else 0
        by_q[qid][row["system"]] = {
            **{c: row[c] for c in CRITERIA},
            "avg": avg,
        }

    breakdown = []
    for qid in sorted(by_q.keys()):
        breakdown.append({"question_id": qid, "systems": by_q[qid]})
    return breakdown


def statistical_tests(rows: list[dict]) -> dict:
    """Run Friedman test + pairwise Wilcoxon signed-rank tests.

    These are appropriate for paired, ordinal (Likert) data.
    """
    # Build paired score vectors: one average score per question per system
    by_q_sys = {}
    for row in rows:
        qid = row["question_id"]
        sys = row["system"]
        valid = [row[c] for c in CRITERIA if row[c] is not None]
        avg = statistics.mean(valid) if valid else 0
        by_q_sys[(qid, sys)] = avg

    questions = sorted(set(qid for qid, _ in by_q_sys))
    systems = sorted(set(sys for _, sys in by_q_sys))

    # Build aligned vectors (same question order)
    vectors = {}
    for sys in systems:
        vectors[sys] = [by_q_sys.get((q, sys), 0) for q in questions]

    results = {"n_questions": len(questions), "systems": systems}

    # Friedman test (non-parametric repeated measures)
    if len(systems) >= 3:
        stat, p = stats.friedmanchisquare(*[vectors[s] for s in systems])
        results["friedman"] = {"statistic": round(stat, 4), "p_value": round(p, 6)}

    # Pairwise Wilcoxon signed-rank tests
    pairwise = []
    for i in range(len(systems)):
        for j in range(i + 1, len(systems)):
            s1, s2 = systems[i], systems[j]
            try:
                stat, p = stats.wilcoxon(vectors[s1], vectors[s2])
                pairwise.append({
                    "pair": f"{s1} vs {s2}",
                    "statistic": round(stat, 4),
                    "p_value": round(p, 6),
                    "mean_diff": round(
                        statistics.mean(vectors[s1]) - statistics.mean(vectors[s2]), 3
                    ),
                })
            except ValueError as e:
                pairwise.append({
                    "pair": f"{s1} vs {s2}",
                    "error": str(e),
                })
    results["pairwise_wilcoxon"] = pairwise
    return results


def print_results(aggregated: dict, test_results: dict, breakdown: list[dict]):
    """Pretty-print the analysis."""
    print("\n" + "=" * 70)
    print("EXPERT EVALUATION RESULTS (de-anonymised)")
    print("=" * 70)

    # System scores table
    print(f"\n{'System':<12} {'Correct':>10} {'Complete':>10} {'Relevant':>10} {'Overall':>10}")
    print("-" * 55)
    for sys, scores in aggregated.items():
        print(
            f"{sys:<12} "
            f"{scores['correctness']['mean']:>5.2f}±{scores['correctness']['std']:<4.2f}"
            f"{scores['completeness']['mean']:>5.2f}±{scores['completeness']['std']:<4.2f}"
            f"{scores['relevance']['mean']:>5.2f}±{scores['relevance']['std']:<4.2f}"
            f"{scores['overall']['mean']:>5.2f}±{scores['overall']['std']:<4.2f}"
        )

    # Statistical tests
    if "friedman" in test_results:
        fr = test_results["friedman"]
        sig = "***" if fr["p_value"] < 0.001 else "**" if fr["p_value"] < 0.01 else "*" if fr["p_value"] < 0.05 else "ns"
        print(f"\nFriedman test: χ²={fr['statistic']:.2f}, p={fr['p_value']:.4f} {sig}")

    if test_results.get("pairwise_wilcoxon"):
        print("\nPairwise Wilcoxon signed-rank tests:")
        for pw in test_results["pairwise_wilcoxon"]:
            if "error" in pw:
                print(f"  {pw['pair']}: {pw['error']}")
            else:
                sig = "***" if pw["p_value"] < 0.001 else "**" if pw["p_value"] < 0.01 else "*" if pw["p_value"] < 0.05 else "ns"
                print(f"  {pw['pair']}: W={pw['statistic']:.1f}, p={pw['p_value']:.4f} {sig}, "
                      f"mean diff={pw['mean_diff']:+.2f}")

    # Per-question breakdown
    print(f"\nPer-question breakdown (avg of correctness/completeness/relevance):")
    print(f"{'Question':<14} {'chatbot':>10} {'gpt':>10} {'llama':>10}")
    print("-" * 46)
    for item in breakdown:
        qid = item["question_id"]
        scores = item["systems"]
        parts = [f"{qid:<14}"]
        for sys in ["chatbot", "gpt", "llama"]:
            if sys in scores:
                parts.append(f"{scores[sys]['avg']:>10.2f}")
            else:
                parts.append(f"{'—':>10}")
        print("".join(parts))

    # Summary
    print(f"\nQuestions evaluated: {test_results['n_questions']}")


def main():
    parser = argparse.ArgumentParser(description="Score expert grading sheet")
    parser.add_argument("--grading-csv", type=Path, default=DEFAULT_GRADING_CSV,
                        help="Path to the filled-in grading CSV")
    args = parser.parse_args()

    print(f"Loading grading key from {GRADING_KEY}")
    key = load_grading_key()

    print(f"Parsing grading CSV from {args.grading_csv}")
    rows = parse_grading_csv(args.grading_csv)
    print(f"  Parsed {len(rows)} scored rows")

    rows = de_anonymise(rows, key)
    aggregated = aggregate_by_system(rows)
    breakdown = per_question_breakdown(rows)
    test_results = statistical_tests(rows)

    print_results(aggregated, test_results, breakdown)

    # Save to JSON
    output = {
        "source": str(args.grading_csv),
        "total_scored_rows": len(rows),
        "aggregated": aggregated,
        "statistical_tests": test_results,
        "per_question": breakdown,
    }
    with open(SCORES_OUT, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {SCORES_OUT}")


if __name__ == "__main__":
    main()
