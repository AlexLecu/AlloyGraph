"""
Generate a blind grading spreadsheet for the materials scientist.

Reads the 12 expert exam responses from all 3 systems, anonymises them
as System A / B / C (shuffled per question), and writes:
  1. grading_sheet.csv  — the spreadsheet the expert fills in
  2. grading_key.json   — the mapping from A/B/C back to system names

Usage:
    python generate_expert_grading.py
"""

import csv
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "results"

EXPERT_QUESTIONS = ROOT / "data" / "expert_questions.jsonl"
EXPERT_RESP = {
    "chatbot": OUTPUT / "expert_responses_chatbot.jsonl",
    "llama":   OUTPUT / "expert_responses_llama.jsonl",
    "gpt":     OUTPUT / "expert_responses_gpt.jsonl",
}

GRADING_SHEET = OUTPUT / "expert_grading_sheet.csv"
GRADING_KEY   = OUTPUT / "expert_grading_key.json"

SEED = 42  # reproducible shuffle


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    random.seed(SEED)

    questions = load_jsonl(EXPERT_QUESTIONS)
    responses = {}
    for sys_name, path in EXPERT_RESP.items():
        for rec in load_jsonl(path):
            responses[(sys_name, rec["id"])] = rec["answer"]

    # Build rows: one row per (question, system) with anonymised labels
    rows = []
    key = {}  # question_id → {A: system, B: system, C: system}

    systems = list(EXPERT_RESP.keys())

    for q in questions:
        qid = q["id"]
        labels = ["A", "B", "C"]
        shuffled = systems[:]
        random.shuffle(shuffled)
        label_map = dict(zip(labels, shuffled))
        key[qid] = {lbl: sys for lbl, sys in label_map.items()}

        for lbl in labels:
            sys_name = label_map[lbl]
            answer = responses.get((sys_name, qid), "[NO RESPONSE]")
            rows.append({
                "question_id": qid,
                "question": q["question"],
                "reference_answer": q.get("reference_answer", ""),
                "system": f"System {lbl}",
                "answer": answer,
                "correctness": "",
                "completeness": "",
                "relevance": "",
            })

    # Write CSV
    fieldnames = [
        "question_id", "question", "reference_answer",
        "system", "answer",
        "correctness", "completeness", "relevance",
    ]
    with open(GRADING_SHEET, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Write key (do NOT share with the grader!)
    with open(GRADING_KEY, "w") as f:
        json.dump(key, f, indent=2)

    print(f"Grading sheet: {GRADING_SHEET}  ({len(rows)} rows)")
    print(f"Answer key:    {GRADING_KEY}")
    print()
    print("Instructions for the grader:")
    print("  - Score each answer on a 1-5 Likert scale:")
    print("    1 = Very poor, 2 = Poor, 3 = Adequate, 4 = Good, 5 = Excellent")
    print("  - Correctness: Are the facts and technical details accurate?")
    print("  - Completeness: Does the answer cover all key points?")
    print("  - Relevance: Does the answer address the specific question asked?")
    print("  - Use the reference_answer column as a guide (not all have one)")
    print("  - System labels are randomised — do not try to identify which is which")


if __name__ == "__main__":
    main()
