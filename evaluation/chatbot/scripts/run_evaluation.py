"""
Chatbot evaluation pipeline.

Phases:
  1. collect      — Collect chatbot responses for 100 open-ended questions
  2. score        — Score responses with RAGAS (faithfulness, factual correctness, etc.)
  3. report       — Combine automated accuracy + RAGAS into final report
  4. mcq-collect  — Collect MCQ responses from all 3 systems (chatbot, Llama, GPT)
  5. mcq-score    — Score MCQ responses (deterministic, no LLM judge)
  6. ragas-collect — Collect RAGAS responses from chatbot only
  7. ragas-score  — Score RAGAS responses using GPT-4o as judge

Prerequisites:
  - Backend running (Flask + Weaviate) at CHATBOT_URL
  - pip install -r requirements.txt
  - OPENAI_API_KEY and GROQ_API_KEY in .env

Usage:
  python run_evaluation.py --phase mcq-collect
  python run_evaluation.py --phase mcq-score
  python run_evaluation.py --phase ragas-collect
  python run_evaluation.py --phase ragas-score
  python run_evaluation.py --phase all
"""

import argparse
import glob
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
import requests
from groq import Groq

# Load .env from project root (scripts/ → chatbot/ → evaluation/ → project root)
load_dotenv(Path(__file__).resolve().parent.parent.parent.parent / ".env")

# ── Paths ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent

# Main questions
QUESTIONS = ROOT / "data" / "questions.jsonl"
RESPONSES_CHATBOT = ROOT / "results" / "responses_chatbot.jsonl"
RESPONSES_LLAMA = ROOT / "results" / "responses_llama.jsonl"
RESPONSES_GPT = ROOT / "results" / "responses_gpt.jsonl"
SCORES_CHATBOT = ROOT / "results" / "scores_chatbot.json"
SCORES_LLAMA = ROOT / "results" / "scores_llama.json"
SCORES_GPT = ROOT / "results" / "scores_gpt.json"
REPORT = ROOT / "results" / "report.json"

# Track 1: MCQ
MCQ_1HOP = ROOT / "data" / "mcq_1hop_questions.jsonl"
MCQ_2HOP = ROOT / "data" / "mcq_2hop_questions.jsonl"
MCQ_1HOP_RESP = {
    "chatbot": ROOT / "results" / "mcq_1hop_responses_chatbot.jsonl",
    "llama":   ROOT / "results" / "mcq_1hop_responses_llama.jsonl",
    "gpt":     ROOT / "results" / "mcq_1hop_responses_gpt.jsonl",
}
MCQ_2HOP_RESP = {
    "chatbot": ROOT / "results" / "mcq_2hop_responses_chatbot.jsonl",
    "llama":   ROOT / "results" / "mcq_2hop_responses_llama.jsonl",
    "gpt":     ROOT / "results" / "mcq_2hop_responses_gpt.jsonl",
}
MCQ_GENERAL = ROOT / "data" / "mcq_general.jsonl"
MCQ_GENERAL_RESP = {
    "chatbot": ROOT / "results" / "mcq_general_responses_chatbot.jsonl",
    "llama":   ROOT / "results" / "mcq_general_responses_llama.jsonl",
    "gpt":     ROOT / "results" / "mcq_general_responses_gpt.jsonl",
}
MCQ_REPORT = ROOT / "results" / "mcq_report.json"

# Track 2: RAGAS (chatbot only)
RAGAS_QUESTIONS = ROOT / "data" / "ragas_questions.jsonl"
RAGAS_RESPONSES = ROOT / "results" / "ragas_responses.jsonl"
RAGAS_SCORES = ROOT / "results" / "ragas_scores.json"
RAGAS_REPORT = ROOT / "results" / "ragas_report.json"

# Track 3: Expert exam (blind evaluation by materials scientist)
EXPERT_QUESTIONS = ROOT / "data" / "expert_questions.jsonl"
EXPERT_RESP = {
    "chatbot": ROOT / "results" / "expert_responses_chatbot.jsonl",
    "llama":   ROOT / "results" / "expert_responses_llama.jsonl",
    "gpt":     ROOT / "results" / "expert_responses_gpt.jsonl",
}

# Training data (for context reconstruction)
TRAINING_DATA = ROOT.parent.parent / "backend" / "alloy_crew" / "models" / "training_data" / "train_77alloys.jsonl"

# ── Config ──────────────────────────────────────────────────────────────
CHATBOT_URL = os.getenv("CHATBOT_URL", "http://localhost:5001")
GROQ_MODEL = "llama-3.3-70b-versatile"
GPT_MODEL = "gpt-4o"
BASELINE_SYSTEM_PROMPT = (
    "You are a materials science expert specializing in nickel-based superalloys. "
    "Answer the user's question based on your knowledge. "
    "Be concise, include units (MPa, g/cm³, %, °C), and state when you are unsure."
)

MCQ_SYSTEM_PROMPT = (
    "You are a materials science expert specializing in nickel-based superalloys. "
    "You will be given a multiple-choice question with 4 options (A, B, C, D). "
    "You MUST respond with ONLY the letter of your chosen answer followed by a "
    "brief explanation. Format: ANSWER: X\nExplanation: ..."
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


_groq_client = None


def _get_groq_client():
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return _groq_client


def call_llama(question: str) -> dict:
    """Send a question to vanilla Llama 3.3 70B via Groq (no KG context).

    Returns: {"answer": str}
    """
    client = _get_groq_client()

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


def _collect_system(name: str, call_fn, questions: list[dict], out_path: Path):
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
            "type": q.get("type", q.get("subtype", "")),
            "question": q["question"],
            "answer": result["answer"],
            "ground_truth": q.get("ground_truth", ""),
        }
        # Save contexts from chatbot retrieval if available
        if result.get("contexts"):
            entry["contexts"] = result["contexts"]
        responses.append(entry)
        # Append single entry instead of rewriting entire file
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
        time.sleep(0.5)

    print(f"{name} responses: {len(responses)} saved to {out_path.name}")


def phase_collect():
    """Phase 1: Collect responses from chatbot, Llama baseline, and GPT baseline."""
    questions = load_questions()
    print(f"Loaded {len(questions)} questions")

    _collect_system("Chatbot", call_chatbot, questions, RESPONSES_CHATBOT)
    _collect_system("Llama-3.3-70B", call_llama, questions, RESPONSES_LLAMA)
    _collect_system("GPT-4o", call_gpt, questions, RESPONSES_GPT)


# ── Phase 2: Score with RAGAS ───────────────────────────────────────────

def _score_system(name: str, responses_path: Path, scores_path: Path,
                  metrics, evaluator_llm, has_contexts: bool = False,
                  questions_path: Path | None = None):
    """Score a single system with RAGAS and save results."""
    print(f"\nScoring {name} responses...")
    data = load_jsonl(responses_path)

    # If responses lack ground_truth, merge from questions file
    if data and "ground_truth" not in data[0] and questions_path:
        questions = load_jsonl(questions_path)
        gt_map = {q["id"]: q for q in questions}
        for r in data:
            q = gt_map.get(r["id"], {})
            r["ground_truth"] = q.get("ground_truth", "")
            if "type" not in r:
                r["type"] = q.get("type", q.get("subtype", ""))

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
            "reference": r.get("ground_truth", ""),
        })

    from ragas import EvaluationDataset, evaluate
    dataset = EvaluationDataset.from_list(samples)
    result = evaluate(dataset=dataset, metrics=metrics, llm=evaluator_llm)

    # Build per-sample scores
    df = result.to_pandas()

    # Auto-detect metric columns (everything except input/output fields)
    input_cols = {"user_input", "retrieved_contexts", "response", "reference"}
    df_col_map = {}
    for col in df.columns:
        if col not in input_cols:
            # Clean column name: "factual_correctness(mode=f1)" → "factual_correctness"
            clean = col.split("(")[0].strip().replace(" ", "_")
            df_col_map[col] = clean
    aggregate = {}
    for df_col, out_name in df_col_map.items():
        if df_col in df.columns:
            aggregate[out_name] = round(df[df_col].mean(skipna=True), 4)

    scores = {
        "system": name,
        "aggregate": aggregate,
        "per_sample": [],
    }
    for i, (_, row) in enumerate(df.iterrows()):
        sample = {
            "id": data[i]["id"],
            "type": data[i].get("type", data[i].get("subtype", "")),
            "question": data[i]["question"],
        }
        for df_col, out_name in df_col_map.items():
            if df_col in df.columns:
                val = row.get(df_col, 0)
                sample[out_name] = round(float(val), 4) if val is not None else 0.0
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
                  chatbot_metrics, evaluator_llm, has_contexts=True,
                  questions_path=QUESTIONS)
    _score_system("Llama-3.3-70B", RESPONSES_LLAMA, SCORES_LLAMA,
                  baseline_metrics, evaluator_llm, questions_path=QUESTIONS)
    _score_system("GPT-4o", RESPONSES_GPT, SCORES_GPT,
                  baseline_metrics, evaluator_llm, questions_path=QUESTIONS)


# ── Phase 3: Generate combined report ──────────────────────────────────

def _find_latest_automated(output_dir: Path, prefix: str) -> Path | None:
    """Find the most recent automated score file by timestamp."""
    pattern = str(output_dir / f"{prefix}_*.json")
    files = sorted(glob.glob(pattern))
    return Path(files[-1]) if files else None


def phase_report():
    """Generate combined evaluation report.

    Merges two data sources (any subset available):
      1. Automated accuracy metrics
      2. RAGAS quality metrics (from --phase score)
    """
    output_dir = ROOT / "results"
    report = {"timestamp": datetime.now().isoformat(), "sections": {}}
    found_any = False

    # ── Section 1: Automated Accuracy ────────────────────────────────
    auto_summary_path = _find_latest_automated(output_dir, "automated_summary")

    if auto_summary_path:
        found_any = True
        with open(auto_summary_path) as f:
            auto_summary = json.load(f)

        report["sections"]["automated_accuracy"] = auto_summary

        print("\n" + "=" * 70)
        print("1. AUTOMATED ACCURACY (deterministic)")
        print("=" * 70)
        print(f"\n  Overall Score: {auto_summary['overall_score']:.1%}"
              f"  ({auto_summary['total_questions']} questions)")
        print(f"\n  {'Type':<20} {'Count':>6} {'Score':>8}")
        print(f"  {'─' * 38}")
        for qtype in ["property_lookup", "composition", "ranking",
                       "target_search", "comparison"]:
            info = auto_summary["by_type"].get(qtype)
            if info:
                print(f"  {qtype:<20} {info['count']:>6} {info['avg_score']:>8.1%}")
                if qtype == "property_lookup":
                    print(f"    Exact match: {info.get('exact_match_rate', 0):.0%}  "
                          f"Within 5%: {info.get('within_5pct_rate', 0):.0%}  "
                          f"Within 10%: {info.get('within_10pct_rate', 0):.0%}")
    else:
        print("\n  [Automated accuracy] Not found")

    # ── Section 2: RAGAS Quality ─────────────────────────────────────
    if SCORES_CHATBOT.exists():
        found_any = True
        with open(SCORES_CHATBOT) as f:
            ragas_data = json.load(f)

        report["sections"]["ragas_quality"] = ragas_data["aggregate"]

        print("\n" + "=" * 70)
        print("2. RAGAS QUALITY METRICS (GPT-4o judge)")
        print("=" * 70)
        print(f"\n  {'Metric':<28} {'Score':>8}")
        print(f"  {'─' * 38}")
        for metric, val in ragas_data["aggregate"].items():
            print(f"  {metric:<28} {val:>8.4f}")

        # Per-type breakdown for factual correctness
        by_type = {}
        for s in ragas_data.get("per_sample", []):
            by_type.setdefault(s["type"], []).append(
                s.get("factual_correctness", 0)
            )
        if by_type:
            print(f"\n  Factual Correctness by Type:")
            print(f"  {'Type':<20} {'Count':>6} {'Score':>8}")
            print(f"  {'─' * 38}")
            for qtype in sorted(by_type):
                vals = by_type[qtype]
                avg = sum(vals) / len(vals) if vals else 0
                print(f"  {qtype:<20} {len(vals):>6} {avg:>8.4f}")
    else:
        print("\n  [RAGAS quality] Not found — run --phase score first")

    # ── Save combined report ─────────────────────────────────────────
    if not found_any:
        print("\nNo evaluation data found. Run the evaluation phases first.")
        return

    with open(REPORT, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n{'=' * 70}")
    print(f"Combined report saved to {REPORT.name}")
    print(f"{'=' * 70}")


# ── Track 1: MCQ ───────────────────────────────────────────────────────

def call_llama_mcq(question: str) -> dict:
    """Send MCQ to Llama with MCQ-specific system prompt, temperature=0."""
    client = _get_groq_client()
    try:
        completion = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": MCQ_SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            temperature=0.0,
            max_tokens=300,
        )
        return {"answer": completion.choices[0].message.content}
    except Exception as e:
        return {"answer": f"[ERROR: {e}]"}


def call_gpt_mcq(question: str) -> dict:
    """Send MCQ to GPT-4o with MCQ-specific system prompt, temperature=0."""
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    try:
        completion = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {"role": "system", "content": MCQ_SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            temperature=0.0,
            max_tokens=300,
        )
        return {"answer": completion.choices[0].message.content}
    except Exception as e:
        return {"answer": f"[ERROR: {e}]"}


def extract_mcq_answer(response_text: str,
                       options: dict | None = None) -> str | None:
    """Extract A/B/C/D from an MCQ response.

    Tries regex patterns first, then falls back to matching the
    response text against option values when no letter is found.
    """
    if not response_text:
        return None

    text = response_text.strip()
    upper = text.upper()

    # ── Regex strategies (look for explicit letter) ──────────────

    # "ANSWER: B" or "answer is: B" or "correct answer is B"
    m = re.search(r'(?:CORRECT\s+)?ANSWER\s*(?:IS)?[:\s]*\(?([A-D])\)?', upper)
    if m:
        return m.group(1)

    # Letter at start: "B) ..." or "(B) ..."
    m = re.match(r'^\(?([A-D])\)?[\s\).\-:]', upper)
    if m:
        return m.group(1)

    # "**B)**" or "**B)** value" (bold letter in chatbot responses)
    m = re.search(r'\*\*([A-D])\)', text)
    if m:
        return m.group(1).upper()

    # "B) **value**" pattern
    m = re.search(r'\b([A-D])\)\s+\*\*', text)
    if m:
        return m.group(1).upper()

    # Letter on its own line (after "answer is:\n\nB) ...")
    m = re.search(r'\n\s*([A-D])\)', text)
    if m:
        return m.group(1).upper()

    # Single letter response
    if len(upper) == 1 and upper in "ABCD":
        return upper

    # ── Value matching fallback ──────────────────────────────────
    # When the chatbot answers with the value/name but no letter,
    # match against the option values.  Strip * from both sides
    # so markdown bold and trademark asterisks don't break matching.
    # Use first-mention (earliest position) to pick the chatbot's
    # top-ranked answer when all options appear in the text.
    if options:
        clean = re.sub(r'\*+', '', upper)
        best_letter = None
        best_pos = len(clean) + 1
        for letter, value in options.items():
            val_clean = re.sub(r'\*+', '', value.strip().upper())
            pos = clean.find(val_clean)
            if pos != -1 and pos < best_pos:
                best_letter = letter
                best_pos = pos
        if best_letter:
            return best_letter

    return None


def score_mcq(questions: list[dict], responses: list[dict]) -> dict:
    """Score MCQ responses, returning accuracy metrics."""
    results = {
        "total": len(questions),
        "correct": 0,
        "incorrect": 0,
        "unparseable": 0,
        "by_subtype": {},
        "per_question": [],
    }

    for q, r in zip(questions, responses):
        extracted = extract_mcq_answer(r["answer"], q.get("options"))
        is_correct = (extracted == q["correct_answer"]) if extracted else False

        if extracted is None:
            results["unparseable"] += 1
        elif is_correct:
            results["correct"] += 1
        else:
            results["incorrect"] += 1

        subtype = q.get("subtype", "unknown")
        if subtype not in results["by_subtype"]:
            results["by_subtype"][subtype] = {"correct": 0, "total": 0}
        results["by_subtype"][subtype]["total"] += 1
        if is_correct:
            results["by_subtype"][subtype]["correct"] += 1

        results["per_question"].append({
            "id": q["id"],
            "subtype": subtype,
            "extracted": extracted,
            "correct_answer": q["correct_answer"],
            "is_correct": is_correct,
        })

    results["accuracy"] = (results["correct"] / results["total"]
                           if results["total"] > 0 else 0)
    for sub in results["by_subtype"].values():
        sub["accuracy"] = sub["correct"] / sub["total"] if sub["total"] else 0

    return results


def phase_mcq_collect():
    """Track 1: Collect MCQ responses from all 3 systems."""
    systems = [
        ("Chatbot-MCQ", call_chatbot),
        ("Llama-MCQ", call_llama_mcq),
        ("GPT-MCQ", call_gpt_mcq),
    ]

    for label, questions_path, resp_map in [
        ("1-hop", MCQ_1HOP, MCQ_1HOP_RESP),
        ("2-hop", MCQ_2HOP, MCQ_2HOP_RESP),
        ("general", MCQ_GENERAL, MCQ_GENERAL_RESP),
    ]:
        questions = load_jsonl(questions_path)
        print(f"\n── MCQ {label}: {len(questions)} questions ──")

        for sys_name, call_fn in systems:
            key = sys_name.split("-")[0].lower()
            out_path = resp_map[key]
            _collect_system(sys_name, call_fn, questions, out_path)


def phase_mcq_score():
    """Track 1: Score MCQ responses (deterministic, no LLM judge)."""
    report = {"timestamp": datetime.now().isoformat(), "systems": {}}

    for hop_label, q_path, resp_map in [
        ("1hop", MCQ_1HOP, MCQ_1HOP_RESP),
        ("2hop", MCQ_2HOP, MCQ_2HOP_RESP),
        ("general", MCQ_GENERAL, MCQ_GENERAL_RESP),
    ]:
        questions = load_jsonl(q_path)
        for sys_key, resp_path in resp_map.items():
            if not resp_path.exists():
                print(f"Skipping {sys_key} {hop_label}: no responses found")
                continue
            responses = load_jsonl(resp_path)
            scores = score_mcq(questions, responses)

            if sys_key not in report["systems"]:
                report["systems"][sys_key] = {}
            report["systems"][sys_key][hop_label] = scores

            print(f"{sys_key} {hop_label}: {scores['accuracy']:.1%} "
                  f"({scores['correct']}/{scores['total']})")

    # Compute combined accuracy
    for sys_key, hop_data in report["systems"].items():
        total_c = sum(h.get("correct", 0) for h in hop_data.values())
        total_n = sum(h.get("total", 0) for h in hop_data.values())
        report["systems"][sys_key]["overall"] = {
            "correct": total_c,
            "total": total_n,
            "accuracy": total_c / total_n if total_n else 0,
        }

    with open(MCQ_REPORT, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nMCQ report saved to {MCQ_REPORT.name}")


# ── Track 2: RAGAS (chatbot only) ─────────────────────────────────────

def phase_ragas_collect():
    """Track 2: Collect RAGAS responses from chatbot only."""
    questions = load_jsonl(RAGAS_QUESTIONS)
    print(f"Loaded {len(questions)} RAGAS questions")
    _collect_system("Chatbot-RAGAS", call_chatbot, questions, RAGAS_RESPONSES)


def _reconstruct_contexts(responses: list[dict]) -> None:
    """Reconstruct KG contexts from training data for responses missing them."""
    alloys_data = load_jsonl(TRAINING_DATA)

    for r in responses:
        if r.get("contexts"):
            continue
        answer = r.get("answer", "")
        gt = r.get("ground_truth", "")
        # Find alloys mentioned in the answer or ground truth
        matched = []
        for a in alloys_data:
            name = a["alloy"]
            if name in answer or name in gt or name.replace("*", "") in answer:
                matched.append(a)
        # Build context strings from matched alloys
        contexts = []
        for a in matched[:5]:
            lines = [f"Alloy: {a['alloy']} ({a.get('processing', '')})"]
            comp = a.get("composition", {})
            if comp:
                top = sorted(comp.items(), key=lambda x: x[1], reverse=True)[:8]
                lines.append("Composition (wt%): " +
                             ", ".join(f"{el}: {v:.1f}%" for el, v in top))
            cf = a.get("computed_features", {})
            d = cf.get("density_calculated_gcm3")
            if d:
                lines.append(f"Density: {d:.2f} g/cm\u00b3")
            gp = cf.get("gamma_prime_estimated_vol_pct")
            if gp is not None:
                lines.append(f"\u03b3' fraction: {gp:.1f}%")
            tcp = cf.get("TCP_risk")
            if tcp:
                lines.append(f"TCP risk: {tcp}")
            for field in ["yield_strength", "uts", "elongation", "elasticity"]:
                measurements = a.get(field, [])
                if measurements:
                    vals = ", ".join(f"{m['value']} @ {m['temp_c']}\u00b0C"
                                     for m in measurements)
                    lines.append(f"{field}: {vals}")
            contexts.append("\n".join(lines))
        r["contexts"] = contexts if contexts else ["No context retrieved."]


def phase_ragas_score():
    """Track 2: Score RAGAS responses using GPT-4o as judge."""
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        AnswerCorrectness,
        ResponseRelevancy,
        AnswerSimilarity,
        Faithfulness,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
    )
    from langchain_openai import ChatOpenAI

    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o"))
    metrics = [
        AnswerCorrectness(),
        ResponseRelevancy(),
        AnswerSimilarity(),
        Faithfulness(),
        LLMContextPrecisionWithReference(),
        LLMContextRecall(),
    ]

    # Load responses and enrich with ground_truth + contexts
    data = load_jsonl(RAGAS_RESPONSES)
    questions = load_jsonl(RAGAS_QUESTIONS)
    gt_map = {q["id"]: q for q in questions}
    for r in data:
        q = gt_map.get(r["id"], {})
        if "ground_truth" not in r:
            r["ground_truth"] = q.get("ground_truth", "")
        if "type" not in r:
            r["type"] = q.get("type", q.get("subtype", ""))
    if data and not data[0].get("contexts"):
        print("Reconstructing contexts from training data...")
        _reconstruct_contexts(data)

    samples = []
    for r in data:
        contexts = r.get("contexts", []) or ["No context retrieved."]
        samples.append({
            "user_input": r["question"],
            "response": r["answer"],
            "retrieved_contexts": contexts,
            "reference": r.get("ground_truth", ""),
        })

    from ragas import EvaluationDataset, evaluate
    dataset = EvaluationDataset.from_list(samples)
    result = evaluate(dataset=dataset, metrics=metrics, llm=evaluator_llm)

    df = result.to_pandas()

    # Auto-detect metric columns (everything except input/output fields)
    input_cols = {"user_input", "retrieved_contexts", "response", "reference"}
    df_col_map = {}
    for col in df.columns:
        if col not in input_cols:
            # Clean column name: "factual_correctness(mode=f1)" → "factual_correctness"
            clean = col.split("(")[0].strip().replace(" ", "_")
            df_col_map[col] = clean
    # Compute aggregates from actual DataFrame columns
    aggregate = {}
    for df_col, out_name in df_col_map.items():
        if df_col in df.columns:
            aggregate[out_name] = round(df[df_col].mean(skipna=True), 4)

    scores = {
        "system": "Chatbot-RAGAS",
        "aggregate": aggregate,
        "per_sample": [],
    }
    for i, (_, row) in enumerate(df.iterrows()):
        sample = {
            "id": data[i]["id"],
            "type": data[i].get("type", data[i].get("subtype", "")),
            "question": data[i]["question"],
        }
        for df_col, out_name in df_col_map.items():
            if df_col in df.columns:
                val = row.get(df_col, 0)
                sample[out_name] = round(float(val), 4) if val is not None else 0.0
        scores["per_sample"].append(sample)

    with open(RAGAS_SCORES, "w") as f:
        json.dump(scores, f, indent=2)
    print(f"Chatbot-RAGAS scores saved: {scores['aggregate']}")


# ── Track 3: Expert exam ──────────────────────────────────────────────

EXPERT_SYSTEM_PROMPT = (
    "You are a materials science expert specializing in nickel-based superalloys. "
    "Answer the following exam question in detail, drawing on your knowledge of "
    "metallurgy, microstructure, and processing. Be thorough and precise."
)


def call_llama_expert(question: str) -> dict:
    """Send expert question to Llama with expert system prompt."""
    client = _get_groq_client()
    try:
        completion = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": EXPERT_SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            temperature=0.2,
            max_tokens=1500,
        )
        return {"answer": completion.choices[0].message.content}
    except Exception as e:
        return {"answer": f"[ERROR: {e}]"}


def call_gpt_expert(question: str) -> dict:
    """Send expert question to GPT-4o with expert system prompt."""
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    try:
        completion = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {"role": "system", "content": EXPERT_SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            temperature=0.2,
            max_tokens=1500,
        )
        return {"answer": completion.choices[0].message.content}
    except Exception as e:
        return {"answer": f"[ERROR: {e}]"}


def phase_expert_collect():
    """Track 3: Collect expert exam responses from all 3 systems."""
    questions = load_jsonl(EXPERT_QUESTIONS)
    print(f"Loaded {len(questions)} expert exam questions")

    systems = [
        ("Chatbot", call_chatbot, EXPERT_RESP["chatbot"]),
        ("Llama-Expert", call_llama_expert, EXPERT_RESP["llama"]),
        ("GPT-Expert", call_gpt_expert, EXPERT_RESP["gpt"]),
    ]
    for name, call_fn, out_path in systems:
        _collect_system(name, call_fn, questions, out_path)


# ── Main ────────────────────────────────────────────────────────────────

ALL_PHASES = [
    "collect", "score", "report",
    "mcq-collect", "mcq-score",
    "ragas-collect", "ragas-score",
    "expert-collect",
    "all",
]


def main():
    parser = argparse.ArgumentParser(description="Chatbot evaluation pipeline")
    parser.add_argument(
        "--phase",
        choices=ALL_PHASES,
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

    if args.phase in ("mcq-collect", "all"):
        phase_mcq_collect()

    if args.phase in ("mcq-score", "all"):
        phase_mcq_score()

    if args.phase in ("ragas-collect", "all"):
        phase_ragas_collect()

    if args.phase in ("ragas-score", "all"):
        phase_ragas_score()

    if args.phase in ("expert-collect", "all"):
        phase_expert_collect()


if __name__ == "__main__":
    main()
