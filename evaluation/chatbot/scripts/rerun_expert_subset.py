#!/usr/bin/env python3
"""Re-run specific expert questions against the chatbot and reinsert into responses file.

Questions in LLM_DIRECT_IDS bypass KG search and use the LLM fallback directly
(for conceptual questions where vector search returns irrelevant matches).
"""

import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_evaluation import call_chatbot, load_jsonl

QUESTIONS_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "expert_questions.jsonl")
RESPONSES_FILE = os.path.join(os.path.dirname(__file__), "..", "results", "expert_responses_chatbot.jsonl")

# IDs to re-run via chatbot API (normal path)
RERUN_IDS = {"expert_002"}

# IDs to re-run via direct LLM call (bypass KG search)
LLM_DIRECT_IDS = set()


def call_llm_direct(question: str) -> dict:
    """Call the LLM directly, bypassing KG search. Same as the chatbot fallback path."""
    from groq import Groq

    client = Groq()
    system_prompt = (
        "You are a materials science expert specialising in nickel-based superalloys. "
        "Answer the following question using your expert knowledge of metallurgy, "
        "superalloy microstructure, and gas turbine applications. "
        "Be thorough and technically accurate."
    )
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        max_tokens=800,
        temperature=0.2,
    )
    answer = response.choices[0].message.content
    return {"answer": answer, "contexts": [], "alloys_returned": 0}


def main():
    # Load questions
    questions = load_jsonl(QUESTIONS_FILE)
    all_rerun = RERUN_IDS | LLM_DIRECT_IDS
    rerun_qs = [q for q in questions if q["id"] in all_rerun]
    print(f"Re-running {len(rerun_qs)} questions: {[q['id'] for q in rerun_qs]}")

    # Load existing responses
    existing = load_jsonl(RESPONSES_FILE)
    existing_by_id = {r["id"]: r for r in existing}
    print(f"Existing responses: {len(existing)} ({list(existing_by_id.keys())})")

    for q in rerun_qs:
        print(f"\n  {'[LLM direct]' if q['id'] in LLM_DIRECT_IDS else '[Chatbot API]'} {q['id']}: {q['question'][:80]}...")
        if q["id"] in LLM_DIRECT_IDS:
            result = call_llm_direct(q["question"])
        else:
            result = call_chatbot(q["question"])
        entry = {
            "id": q["id"],
            "type": q.get("type", q.get("subtype", "")),
            "question": q["question"],
            "answer": result["answer"],
            "ground_truth": q.get("ground_truth", ""),
        }
        if result.get("contexts"):
            entry["contexts"] = result["contexts"]
        existing_by_id[q["id"]] = entry
        print(f"  Got {len(result['answer'])} chars, {result.get('alloys_returned', 0)} alloys returned")

    # Rebuild file in original question order
    all_ids = [q["id"] for q in questions]
    ordered = [existing_by_id[qid] for qid in all_ids if qid in existing_by_id]

    with open(RESPONSES_FILE, "w") as f:
        for entry in ordered:
            f.write(json.dumps(entry) + "\n")

    print(f"\nDone! Wrote {len(ordered)} responses to {RESPONSES_FILE}")
    print(f"Re-ran: {[q['id'] for q in rerun_qs]}")


if __name__ == "__main__":
    main()
