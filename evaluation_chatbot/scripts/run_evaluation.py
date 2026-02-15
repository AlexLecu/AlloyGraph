"""
Chatbot evaluation pipeline using RAGAS.

Compares 3 systems:
  1. Chatbot (Llama 3.3 70B + Knowledge Graph)
  2. Llama 3.3 70B vanilla (same model, no KG) via Groq
  3. GPT-4o vanilla (no KG) via OpenAI

Phase 1: Collect responses from all 3 systems.
Phase 2: Score with RAGAS (GPT-4o as judge).
Phase 3: Generate a 3-way comparison report.

Prerequisites:
  - Backend running (Flask + Weaviate) at CHATBOT_URL
  - pip install -r requirements.txt
  - OPENAI_API_KEY and GROQ_API_KEY in .env

Usage:
  python run_evaluation.py --phase collect    # Phase 1: collect responses
  python run_evaluation.py --phase score      # Phase 2: score with RAGAS
  python run_evaluation.py --phase report     # Phase 3: generate report
  python run_evaluation.py                    # Run all phases
"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
import requests
from groq import Groq

# Load .env from project root (two levels up from this script)
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

# ── Paths ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
QUESTIONS = ROOT / "data" / "questions.jsonl"
RESPONSES_CHATBOT = ROOT / "output" / "responses_chatbot.jsonl"
RESPONSES_LLAMA = ROOT / "output" / "responses_llama.jsonl"
RESPONSES_GPT = ROOT / "output" / "responses_gpt.jsonl"
SCORES_CHATBOT = ROOT / "output" / "scores_chatbot.json"
SCORES_LLAMA = ROOT / "output" / "scores_llama.json"
SCORES_GPT = ROOT / "output" / "scores_gpt.json"
REPORT = ROOT / "output" / "report.json"

# ── Config ──────────────────────────────────────────────────────────────
CHATBOT_URL = os.getenv("CHATBOT_URL", "http://localhost:5001")
GROQ_MODEL = "llama-3.3-70b-versatile"
GPT_MODEL = "gpt-4o"
BASELINE_SYSTEM_PROMPT = (
    "You are a materials science expert specializing in nickel-based superalloys. "
    "Answer the user's question based on your knowledge. "
    "Be concise, include units (MPa, g/cm³, %, °C), and state when you are unsure."
)


# ── Helpers ─────────────────────────────────────────────────────────────

def load_questions() -> list[dict]:
    with open(QUESTIONS) as f:
        return [json.loads(line) for line in f]


def save_jsonl(data: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


# ── Phase 1: Collect responses ──────────────────────────────────────────

def call_chatbot(question: str) -> dict:
    """Send a question to the chatbot API and parse the NDJSON stream.

    Returns: {"answer": str, "contexts": list[str], "alloys_returned": int}
    """
    try:
        resp = requests.post(
            f"{CHATBOT_URL}/api/chat",
            json={"prompt": question, "sessionId": "eval", "history": []},
            stream=True,
            timeout=60,
        )
        resp.raise_for_status()
    except requests.RequestException as e:
        return {"answer": f"[ERROR: {e}]", "contexts": [], "alloys_returned": 0}

    answer_parts = []
    contexts = []
    alloy_count = 0

    for line in resp.iter_lines(decode_unicode=True):
        if not line.strip():
            continue
        try:
            chunk = json.loads(line)
        except json.JSONDecodeError:
            continue

        if chunk.get("type") == "data":
            alloys = chunk.get("alloys", [])
            alloy_count = len(alloys)
            # Convert alloy data to text contexts for RAGAS
            for alloy in alloys:
                ctx = _format_alloy_context(alloy)
                if ctx:
                    contexts.append(ctx)

        elif chunk.get("type") in ("chunk", "text_chunk", "string_chunk"):
            content = chunk.get("content", "")
            if content:
                answer_parts.append(content)

        elif chunk.get("type") == "error":
            answer_parts.append(f"[ERROR: {chunk.get('content', '')}]")

    return {
        "answer": "".join(answer_parts),
        "contexts": contexts,
        "alloys_returned": alloy_count,
    }


def _format_alloy_context(alloy: dict) -> str:
    """Format an alloy dict (from the data chunk) into a text context string."""
    name = alloy.get("name", "Unknown")
    processing = alloy.get("processing_method", "")
    lines = [f"Alloy: {name} ({processing})"]

    # Composition
    comp = alloy.get("composition", {})
    if comp:
        sorted_comp = sorted(comp.items(), key=lambda x: x[1], reverse=True)
        comp_str = ", ".join(f"{el}: {val:.1f}%" for el, val in sorted_comp[:8])
        lines.append(f"Composition (wt%): {comp_str}")

    # Density
    density = alloy.get("density_gcm3")
    if density:
        lines.append(f"Density: {density:.2f} g/cm³")

    # Properties
    props = alloy.get("properties", [])
    if props:
        for p in props:
            ptype = p.get("property_type", "")
            val = p.get("value")
            unit = p.get("unit", "")
            temp = p.get("temperature_c")
            if val is not None:
                temp_str = f" @ {temp}°C" if temp is not None else ""
                lines.append(f"{ptype}: {val:.0f} {unit}{temp_str}")

    return "\n".join(lines)


def call_llama(question: str) -> dict:
    """Send a question to vanilla Llama 3.3 70B via Groq (no KG context).

    Returns: {"answer": str}
    """
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    try:
        completion = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": BASELINE_SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            temperature=0.2,
            max_tokens=800,
        )
        return {"answer": completion.choices[0].message.content}
    except Exception as e:
        return {"answer": f"[ERROR: {e}]"}


def call_gpt(question: str) -> dict:
    """Send a question to vanilla GPT-4o via OpenAI (no KG context).

    Returns: {"answer": str}
    """
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    try:
        completion = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {"role": "system", "content": BASELINE_SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            temperature=0.2,
            max_tokens=800,
        )
        return {"answer": completion.choices[0].message.content}
    except Exception as e:
        return {"answer": f"[ERROR: {e}]"}


def _collect_system(name: str, call_fn, questions: list[dict], out_path: Path,
                     has_contexts: bool = False):
    """Collect responses for a single system (resume-safe)."""
    responses = []
    if out_path.exists():
        responses = load_jsonl(out_path)
        print(f"Resuming {name}: {len(responses)}/{len(questions)} already done")

    for i, q in enumerate(questions):
        if i < len(responses):
            continue

        print(f"  [{i+1}/{len(questions)}] {name}: {q['question'][:60]}...")
        result = call_fn(q["question"])
        entry = {
            "id": q["id"],
            "type": q["type"],
            "question": q["question"],
            "ground_truth": q["ground_truth"],
            "answer": result["answer"],
            "contexts": result.get("contexts", []) if has_contexts else [],
        }
        if has_contexts:
            entry["alloys_returned"] = result.get("alloys_returned", 0)
        responses.append(entry)
        save_jsonl(responses, out_path)
        time.sleep(0.5)

    print(f"{name} responses: {len(responses)} saved to {out_path.name}")


def phase_collect():
    """Phase 1: Collect responses from chatbot, Llama baseline, and GPT baseline."""
    questions = load_questions()
    print(f"Loaded {len(questions)} questions")

    _collect_system("Chatbot", call_chatbot, questions, RESPONSES_CHATBOT, has_contexts=True)
    _collect_system("Llama-3.3-70B", call_llama, questions, RESPONSES_LLAMA)
    _collect_system("GPT-4o", call_gpt, questions, RESPONSES_GPT)


# ── Phase 2: Score with RAGAS ───────────────────────────────────────────

def _score_system(name: str, responses_path: Path, scores_path: Path,
                  metrics, evaluator_llm, has_contexts: bool = False):
    """Score a single system with RAGAS and save results."""
    print(f"\nScoring {name} responses...")
    data = load_jsonl(responses_path)
    samples = []
    for r in data:
        if has_contexts:
            contexts = r.get("contexts", []) or ["No context retrieved."]
        else:
            contexts = ["No knowledge graph context available."]
        samples.append({
            "user_input": r["question"],
            "response": r["answer"],
            "retrieved_contexts": contexts,
            "reference": r["ground_truth"],
        })

    from ragas import EvaluationDataset, evaluate
    dataset = EvaluationDataset.from_list(samples)
    result = evaluate(dataset=dataset, metrics=metrics, llm=evaluator_llm)

    # Build per-sample scores
    df = result.to_pandas()
    metric_keys = [m.__class__.__name__ for m in metrics]
    # Map class names to column names in the DataFrame
    col_map = {
        "FactualCorrectness": "factual_correctness",
        "Faithfulness": "faithfulness",
        "ResponseRelevancy": "response_relevancy",
        "LLMContextRecall": "llm_context_recall",
    }

    scores = {
        "system": name,
        "aggregate": {k: round(v, 4) for k, v in result.items()},
        "per_sample": [],
    }
    for i, row in df.iterrows():
        sample = {
            "id": data[i]["id"],
            "type": data[i]["type"],
            "question": data[i]["question"],
        }
        for cls_name in metric_keys:
            col = col_map.get(cls_name, cls_name.lower())
            sample[col] = round(row.get(col, 0), 4)
        scores["per_sample"].append(sample)

    with open(scores_path, "w") as f:
        json.dump(scores, f, indent=2)
    print(f"{name} scores saved: {scores['aggregate']}")


def phase_score():
    """Phase 2: Score responses with RAGAS using GPT-4o as judge."""
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        Faithfulness,
        FactualCorrectness,
        ResponseRelevancy,
        LLMContextRecall,
    )
    from langchain_openai import ChatOpenAI

    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o"))

    # Chatbot gets all 4 metrics (has retrieved contexts)
    chatbot_metrics = [FactualCorrectness(), Faithfulness(),
                       ResponseRelevancy(), LLMContextRecall()]
    # Baselines get 2 metrics (no real contexts)
    baseline_metrics = [FactualCorrectness(), ResponseRelevancy()]

    _score_system("Chatbot (Llama+KG)", RESPONSES_CHATBOT, SCORES_CHATBOT,
                  chatbot_metrics, evaluator_llm, has_contexts=True)
    _score_system("Llama-3.3-70B", RESPONSES_LLAMA, SCORES_LLAMA,
                  baseline_metrics, evaluator_llm)
    _score_system("GPT-4o", RESPONSES_GPT, SCORES_GPT,
                  baseline_metrics, evaluator_llm)


# ── Phase 3: Generate report ────────────────────────────────────────────

def phase_report():
    """Phase 3: Generate 3-way comparison report."""
    systems = {}
    for label, path in [("chatbot", SCORES_CHATBOT),
                        ("llama", SCORES_LLAMA),
                        ("gpt", SCORES_GPT)]:
        with open(path) as f:
            systems[label] = json.load(f)

    display_names = {
        "chatbot": "Chatbot+KG",
        "llama": "Llama-3.3",
        "gpt": "GPT-4o",
    }

    # ── Aggregate comparison ──
    print("\n" + "=" * 70)
    print("CHATBOT EVALUATION REPORT (3-way comparison)")
    print("=" * 70)

    print("\n── Overall Scores ──")
    header = f"{'Metric':<25}"
    for key in systems:
        header += f" {display_names[key]:>12}"
    print(header)
    print("-" * 65)

    shared_metrics = ["factual_correctness", "response_relevancy"]
    for metric in shared_metrics:
        line = f"{metric:<25}"
        for key in systems:
            val = systems[key]["aggregate"].get(metric, 0)
            line += f" {val:>12.4f}"
        print(line)

    # Chatbot-only metrics
    chatbot_only = ["faithfulness", "llm_context_recall"]
    for metric in chatbot_only:
        cb = systems["chatbot"]["aggregate"].get(metric, 0)
        line = f"{metric:<25} {cb:>12.4f}"
        for _ in ["llama", "gpt"]:
            line += f" {'N/A':>12}"
        print(line)

    # ── Per-type breakdown ──
    print("\n── Factual Correctness by Question Type ──")
    header = f"{'Type':<20}"
    for key in systems:
        header += f" {display_names[key]:>12}"
    print(header)
    print("-" * 60)

    by_type = {}
    for key, data in systems.items():
        by_type[key] = {}
        for s in data["per_sample"]:
            by_type[key].setdefault(s["type"], []).append(s["factual_correctness"])

    all_types = sorted(set().union(*(bt.keys() for bt in by_type.values())))
    for qtype in all_types:
        line = f"{qtype:<20}"
        for key in systems:
            vals = by_type[key].get(qtype, [])
            avg = sum(vals) / len(vals) if vals else 0
            line += f" {avg:>12.4f}"
        print(line)

    # ── Save report ──
    report = {
        "timestamp": datetime.now().isoformat(),
        "num_questions": len(systems["chatbot"]["per_sample"]),
        "aggregate": {key: data["aggregate"] for key, data in systems.items()},
        "by_type": {},
    }
    for qtype in all_types:
        entry = {"count": len(by_type["chatbot"].get(qtype, []))}
        for key in systems:
            vals = by_type[key].get(qtype, [])
            entry[f"{key}_factual_correctness"] = round(
                sum(vals) / len(vals), 4) if vals else 0
        report["by_type"][qtype] = entry

    with open(REPORT, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {REPORT.name}")


# ── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Chatbot evaluation pipeline")
    parser.add_argument(
        "--phase",
        choices=["collect", "score", "report", "all"],
        default="all",
        help="Which phase to run (default: all)",
    )
    args = parser.parse_args()

    if args.phase in ("collect", "all"):
        phase_collect()

    if args.phase in ("score", "all"):
        phase_score()

    if args.phase in ("report", "all"):
        phase_report()


if __name__ == "__main__":
    main()
