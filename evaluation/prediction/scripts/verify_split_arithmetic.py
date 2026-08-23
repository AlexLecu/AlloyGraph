#!/usr/bin/env python3
"""Account for every handbook record between the corpus and the evaluation set.

The corpus, the training set and the evaluation set are quoted in the paper as
106 / 77 / 88, and those numbers do not obviously add up. They do, but only
once two things are said plainly: a "record" is an alloy under one processing
route and product form, not an alloy, and the evaluation set is not the holdout.

This script rebuilds both steps and prints the disposition of every record that
falls out, so the arithmetic in Methods can be checked rather than trusted.

Step 1, the split (create_train_test_split.py):

    106 upstream records  =  77 training  +  27 holdout  +  2 dropped

Step 2, categorisation into the shipped evaluation files
(create_categorized_datasets.py): the 88-alloy evaluation set is the holdout
plus MatWeb records, minus everything routed to the out-of-scope ``other``
category -- and minus four holdout records that are absent for no recorded
reason. Those four are the finding this script exists to keep visible.

Exit status is 1 if the step-1 arithmetic stops closing.

Usage:
    python verify_split_arithmetic.py
"""

import importlib.util
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))
TRAINING_DIR = os.path.join(PROJECT_ROOT, "backend", "alloy_crew", "models", "training_data")
DATA_DIR = os.path.join(BASE_DIR, "data")

UPSTREAM = "final_alloy_data_enriched_v2.jsonl"
TRAIN = "train_77alloys_v2.jsonl"
HOLDOUT = "test_split_30.jsonl"
SHIPPED = ("SSS", "precip", "sc_ds")


def load(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def key(r):
    return (r.get("alloy"), r.get("processing"), r.get("form"))


def comp_sum(r):
    c = r.get("composition") or {}
    return sum(float(v) for v in c.values() if v)


def main():
    up = load(os.path.join(TRAINING_DIR, UPSTREAM))
    tr = load(os.path.join(TRAINING_DIR, TRAIN))
    ho = load(os.path.join(TRAINING_DIR, HOLDOUT))
    ev = [a for s in SHIPPED for a in load(os.path.join(DATA_DIR, f"{s}.jsonl"))]

    K_up, K_tr, K_ho = {key(r) for r in up}, {key(r) for r in tr}, {key(r) for r in ho}

    print("STEP 1 -- the split")
    print(f"  upstream records          {len(up):4d}   ({len({r['alloy'] for r in up})} distinct alloy names)")
    print(f"  training                  {len(tr):4d}   ({len({r['alloy'] for r in tr})} distinct)")
    print(f"  holdout (test_split_30)   {len(ho):4d}   ({len({r['alloy'] for r in ho})} distinct)")
    print(f"  train and holdout overlap {len(K_tr & K_ho):4d}")

    dropped = K_up - K_tr - K_ho
    print(f"\n  in neither: {len(dropped)}")
    for k in sorted(dropped, key=str):
        r = next(r for r in up if key(r) == k)
        n_el = len(r.get("composition") or {})
        siblings = [kk for kk in K_up if kk[0] == k[0] and kk != k]
        print(f"    {k[0]!r} ({k[1]}, {k[2]}): {n_el} elements, composition sums to "
              f"{comp_sum(r):.1f} wt%")
        for sk in siblings:
            print(f"        sibling {sk[2]!r} record is in training: {sk in K_tr}")

    closes = len(tr) + len(ho) + len(dropped) == len(up)
    print(f"\n  {len(tr)} + {len(ho)} + {len(dropped)} = {len(up)}  -> "
          f"{'closes' if closes else 'DOES NOT CLOSE'}")

    print("\nSTEP 2 -- holdout to evaluation set")
    ev_names = {a["alloy"] for a in ev}
    ho_shipped = {r["alloy"] for r in ho if r["alloy"] in ev_names}
    absent = [r for r in ho if r["alloy"] not in ev_names]
    print(f"  evaluation alloys          {len(ev_names):4d}")
    print(f"  of them from the holdout   {len(ho_shipped):4d}")
    print(f"  holdout records not shipped{len(absent):4d}")

    spec = importlib.util.spec_from_file_location(
        "cc", os.path.join(SCRIPT_DIR, "create_categorized_datasets.py"))
    cc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cc)

    print()
    for r in sorted(absent, key=lambda r: r["alloy"]):
        cat, why = cc.classify_alloy(r)
        n_el = len(r.get("composition") or {})
        if n_el == 0:
            verdict = "no composition; classifier reads Ni=0 and routes it to 'other'"
        elif cat == "other":
            verdict = "out-of-scope category"
        else:
            verdict = f"NO RECORDED REASON -- classifies as {cat}, absent from every shipped file"
        print(f"  {r['alloy']:22s} {n_el} elements, sum {comp_sum(r):6.1f} wt%  -> {verdict}")

    unexplained = [r for r in absent
                   if (r.get("composition") or {}) and cc.classify_alloy(r)[0] != "other"]
    print(f"\n  unexplained: {len(unexplained)} of {len(absent)}")
    return 0 if closes else 1


if __name__ == "__main__":
    sys.exit(main())
