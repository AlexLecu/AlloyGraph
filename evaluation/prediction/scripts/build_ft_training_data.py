#!/usr/bin/env python3
"""Build fine-tuning data from the exact training set this system's ML models use.

WHY NOT THE COMMITTED SPLIT. The archived fine-tune was trained on a 106-alloy
corpus split 75/10/21. That corpus is not the set the ML ensemble was trained
on: its train+val covers only 63 of our 77 alloys, holds 14 of them out in its
test split, and adds 26 alloys our system has never seen. A baseline trained on
different data measures a different experiment -- any accuracy gap confounds
architecture with training corpus.

This script rebuilds the fine-tuning file from
``models/training_data/train_77alloys_v2.jsonl``, the same 77 rows behind
``saved_models_v2``, in the original prompt/completion format. The resulting
baseline and this system then differ only in architecture.

FORMAT. Identical to the archived corpus, so the comparison to the February
model stays meaningful:

    system     Predict mechanical properties based off chemical compositions
               and processing style
    user       Alloy: <name>; Processing: <route>; Composition: <dict>
    assistant  {"yield_strength": [{"temp_c": ..., "value": ...}, ...],
                "uts": [...], "elongation": [...], "elasticity": [...]}

NO VALIDATION FILE. Holding alloys out would break the property that makes this
comparison controlled -- the training set must equal the ML ensemble's, exactly.
With 77 examples and a fixed epoch count there is no schedule to tune, so
validation would cost tokens without informing a decision.

ALLOYS WITH NO MEASURED PROPERTY ARE DROPPED. Three of the 77 (M-21, RENE 77,
UDIMET 720LI) carry no value for any of the four targets. They contribute zero
rows to every ML target, so the ML ensemble never trains on them either and
dropping them preserves the equivalence exactly. Keeping them would be actively
harmful: their completion is four empty arrays, which is supervision teaching
the model to answer with nothing -- the precise failure mode that made the
February model unusable.

NIMONIC PE16 is not in the training set at all; it is an evaluation alloy. The
erratum therefore does not touch this file.

Usage:
    python build_ft_training_data.py
    python build_ft_training_data.py --out /tmp/ft.jsonl
"""

import argparse
import json
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))

SOURCE = os.path.join(PROJECT_ROOT, "backend", "alloy_crew", "models",
                      "training_data", "train_77alloys_v2.jsonl")
DEFAULT_OUT = os.path.join(BASE_DIR, "data", "ft_training_77alloys.jsonl")

#: Verbatim from the archived corpus.
SYSTEM_PROMPT = ("Predict mechanical properties based off chemical compositions "
                 "and processing style")

PROPERTY_KEYS = ("yield_strength", "uts", "elongation", "elasticity")


def build_example(record):
    """One training example in the archived corpus's format."""
    user = (f"Alloy: {record['alloy']}; Processing: {record['processing']}; "
            f"Composition: {record['composition']}")
    completion = {k: (record.get(k) or []) for k in PROPERTY_KEYS}
    return {"messages": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
        {"role": "assistant", "content": json.dumps(completion)},
    ]}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default=SOURCE)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    records = [json.loads(line) for line in open(args.source)]
    usable = [r for r in records if any(r.get(k) for k in PROPERTY_KEYS)]
    dropped = [r["alloy"] for r in records if r not in usable]
    examples = [build_example(r) for r in usable]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        for ex in examples:
            fh.write(json.dumps(ex) + "\n")

    print(f"{len(examples)} examples written to {args.out} (from {len(records)} rows)")
    if dropped:
        print(f"  dropped, no measured property in any target: {dropped}")
    try:
        import tiktoken
        enc = tiktoken.get_encoding("o200k_base")
        total = sum(sum(len(enc.encode(m["content"])) for m in ex["messages"])
                    + 3 * len(ex["messages"]) + 3 for ex in examples)
        print(f"  training tokens: {total:,}  "
              f"(3 epochs -> {total * 3:,} billed, ${total * 3 * 5.0 / 1e6:.3f})")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
