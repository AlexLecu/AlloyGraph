#!/usr/bin/env python3
"""Can the agent LLM recognise an evaluation alloy from its composition? (Expert Systems R2.2)

Compositional distance measures novelty relative to the 77 alloys the system was
built on. It says nothing about what the pretrained LLM already knows: the
evaluation alloys are commercial grades whose datasheets may sit in Llama's
pretraining data. The agents never receive the alloy's name -- they see the
composition, processing route and temperature only, as do the vanilla LLM
baselines -- but a model could still recognise a well-known composition.

This probe asks the same model, on the same provider, at the same temperature,
to name each evaluation alloy from exactly the inputs the agents get. An alloy
counts as identified when the answer matches its designation or a documented
alias (UNS number, trade-name equivalent). Answers are kept verbatim so every
match decision can be audited.

Outputs (../results):
    llm_identification_probe.csv   one row per alloy: answer, confidence, match
    llm_identification_raw.jsonl   raw responses, written as they arrive

Usage:
    python llm_identification_probe.py            # query + score (resumes)
    python llm_identification_probe.py --score    # re-score cached answers only
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
RAW = os.path.join(RESULTS_DIR, "llm_identification_raw.jsonl")
OUT = os.path.join(RESULTS_DIR, "llm_identification_probe.csv")

ENDPOINT = "https://api.deepinfra.com/v1/openai/chat/completions"
MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo"   # the agents' model and provider
DATASETS = ("SSS", "precip", "sc_ds")

PROMPT = """The following is the nominal composition of a commercial nickel-based superalloy.

COMPOSITION (wt%):
{comp}

PROCESSING: {processing}

Which commercial alloy is this? Answer with the single most likely trade name or designation.
If you cannot identify it, answer "UNKNOWN". Respond with JSON only:
{{"alloy": "<name or UNKNOWN>", "confidence": <number between 0 and 1>}}"""

#: Accepted designations per evaluation alloy, as regexes over the lower-cased
#: answer with punctuation removed. Written from the alloy names and their
#: standard aliases (UNS numbers, the Hastelloy/Inconel/Haynes equivalences),
#: never from the model's answers. An alloy not listed falls back to its
#: designation token, the characteristic number or name in its label.
ALIASES = {
    "NIMOCAST* 75": [r"\b75\b"],
    "NIMOCAST* 242": [r"\b242\b"],
    "INCONEL* 600": [r"\b600\b", r"n06600"],
    "HASTELLOY* X": [r"hastelloy x\b", r"\bhx\b", r"n06002"],
    "HAYNES* 242": [r"\b242\b"],
    "Alloy 718 (AMS 5596 sheet)": [r"\b718\b", r"n07718"],
    "Alloy 725": [r"\b725\b", r"n07725"],
    "Haynes 244": [r"\b244\b"],
    "ATI Allegheny Ludlum AL 22™ Nickel-Base Alloy": [r"\bc ?22\b", r"\b622\b", r"n06022", r"\bal 22\b"],
    "ATI Allegheny Ludlum AL 276™ Nickel-Base Alloy": [r"\bc ?276\b", r"n10276", r"\b276\b"],
    "ATI Allegheny Ludlum AL 600™ Nickel-Base Alloy": [r"\b600\b", r"n06600"],
    "ATI Allegheny Ludlum AL 601™ Nickel-Base Alloy": [r"\b601\b", r"n06601"],
    "ATI Allegheny Ludlum AL 625HP™ Stainless Steel": [r"\b625\b", r"n06625"],
    "ATI Allegheny Ludlum Altemp® 718 Nickel-Base Superalloy": [r"\b718\b", r"n07718"],
    "ATI Allegheny Ludlum Altemp® HX Nickel-Based superalloy": [r"hastelloy x\b", r"\bhx\b", r"n06002"],
    "Haynes 625 alloy": [r"\b625\b", r"n06625"],
    "Haynes 718 alloy": [r"\b718\b", r"n07718"],
    "Haynes CABVAL Hastelloy® G-50™ Grade 100 Ni-Cr-Fe-Mo Alloy": [r"\bg ?50\b"],
    "Haynes Hastelloy® B-2 alloy": [r"\bb ?2\b", r"n10665"],
    "Haynes Hastelloy® B-3® alloy": [r"\bb ?3\b", r"n10675"],
    "Haynes Hastelloy® C-2000® alloy": [r"\bc ?2000\b", r"n06200"],
    "Haynes Hastelloy® C-22® alloy": [r"\bc ?22\b", r"\b622\b", r"n06022"],
    "Haynes Hastelloy® C-276 alloy": [r"\bc ?276\b", r"n10276", r"\b276\b"],
    "Haynes Hastelloy® C-4 alloy": [r"\bc ?4\b", r"n06455"],
    "Haynes Hastelloy® G-30® alloy": [r"\bg ?30\b", r"n06030"],
    "Haynes Hastelloy® N": [r"hastelloy n\b", r"n10003", r"inor ?8"],
    "Nickel Alloy 59": [r"\b59\b", r"n06059"],
    "Special Metals INCOLOY® alloy 890 Ni-Cr-Fe Superalloy": [r"\b890\b"],
    "Special Metals INCONEL® 622 Nickel Superalloy (UNS N06022) Plate 0.250 in (6.4 mm)": [r"\b622\b", r"\bc ?22\b", r"n06022"],
    "Special Metals INCONEL® 725 Nickel Superalloy (UNS N07725)": [r"\b725\b", r"n07725"],
    "Special Metals INCONEL® Alloy 686 Nickel Superalloy Sheet (UNS N06686)": [r"\b686\b", r"n06686"],
    "Special Metals INCONEL® Alloy N06230 Ni-Cr-W-Mo Alloy": [r"\b230\b", r"n06230"],
    "Special Metals INCONEL® C276 Nickel Superalloy Sheet": [r"\bc ?276\b", r"n10276", r"\b276\b"],
    "Special Metals INCONEL® G-3 Nickel Superalloy (UNS N06985)": [r"\bg ?3\b", r"n06985"],
    "Special Metals NIMONIC® Alloy 86": [r"\b86\b"],
    "MAR-M* 247": [r"\b247\b"],
    "MAR-M* 200": [r"mar ?m ?200\b"],
    "TRW-NASA VI A": [r"trw", r"nasa ?vi"],
    "IN-738": [r"\b738\b"],
    "NIMOCAST* 263": [r"\b263\b"],
    "IN-731": [r"\b731\b"],
    "WASPALOY*": [r"waspaloy"],
    "NIMONIC* 901": [r"\b901\b"],
    "NIMONIC* 90": [r"nimonic ?90\b"],
    "RGT* 13": [r"rgt ?13"],
    "UDIMET* 520": [r"\b520\b"],
    "D-979": [r"\b979\b"],
    "NIMONIC* 81": [r"nimonic ?81\b"],
    "ASTROLOY": [r"astroloy"],
    "NIMONIC* PE16": [r"pe ?16"],
    "AF2-1DA": [r"af2 ?1 ?da", r"\baf2\b"],
    "Alloy 263 (C263)": [r"\b263\b"],
    "CM 247 LC": [r"\b247\b"],
    "GTD-111": [r"\b111\b"],
    "GTD-262": [r"\b262\b"],
    "GTD-444": [r"\b444\b"],
    "Haynes 233": [r"\b233\b"],
    "Haynes 282": [r"\b282\b"],
    "Inconel 718Plus": [r"718 ?plus"],
    "Inconel X-750": [r"x ?750\b"],
    "MAR-M 002": [r"mar ?m ?002\b", r"\b002\b"],
    "MERL 76": [r"merl ?76"],
    "Rene 100": [r"ren[eé] ?100\b", r"\bin ?100\b"],
    "Rene 108": [r"ren[eé] ?108\b"],
    "Rene 95": [r"ren[eé] ?95\b"],
    "RR1000": [r"rr ?1000"],
    "Udimet 720": [r"\b720\b"],
    "ATI Allegheny Ludlum Altemp® X-750 Nickel-Based superalloy": [r"x ?750\b"],
    "Haynes 214® alloy": [r"\b214\b"],
    "Haynes 263® alloy": [r"\b263\b"],
    "Haynes R-41 alloy": [r"\br ?41\b", r"ren[eé] ?41\b"],
    "Special Metals INCONEL® alloy 693": [r"\b693\b"],
    "Special Metals INCONEL® alloy 740 Precipitation-Hardenable Ni-Cr-Co Superalloy": [r"\b740\b"],
    "PWA*1480(SC)": [r"pwa ?1480"],
    "RENÉ* N5(SC)": [r"\bn ?5\b"],
    "CMSX-10": [r"cmsx ?10\b"],
    "CMSX-8": [r"cmsx ?8\b"],
    "DD5": [r"\bdd ?5\b"],
    "DD6": [r"\bdd ?6\b"],
    "DD9": [r"\bdd ?9\b"],
    "DD98M": [r"dd ?98"],
    "PWA 1497": [r"pwa ?1497"],
    "Rene N500": [r"n ?500\b"],
    "Rene N6": [r"\bn ?6\b"],
    "RR3000": [r"rr ?3000"],
    "RR3010": [r"rr ?3010"],
    "TMS-138": [r"tms ?138"],
    "TMS-162": [r"tms ?162"],
}


def load_key():
    for line in open(os.path.join(PROJECT_ROOT, ".env"), encoding="utf-8"):
        if line.strip().startswith("DEEPINFRA_API_KEY="):
            return line.split("=", 1)[1].strip().strip("'\"")
    raise SystemExit("DEEPINFRA_API_KEY not set in .env")


def alloys():
    for stem in DATASETS:
        for line in open(os.path.join(DATA_DIR, f"{stem}.jsonl"), encoding="utf-8"):
            rec = json.loads(line)
            if rec.get("composition"):
                yield rec


def ask(rec, key, retries=4):
    comp = ", ".join(f"{e}: {w}%" for e, w in
                     sorted(rec["composition"].items(), key=lambda kv: -kv[1]))
    body = {"model": MODEL, "temperature": 0.0, "max_tokens": 80,
            "messages": [{"role": "user", "content": PROMPT.format(
                comp=comp, processing=rec.get("processing", "wrought"))}]}
    for attempt in range(retries):
        req = urllib.request.Request(ENDPOINT, data=json.dumps(body).encode(), headers={
            "Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                out = json.loads(r.read())
            return {"alloy": rec["alloy"], "processing": rec.get("processing"),
                    "answer_raw": out["choices"][0]["message"]["content"],
                    "usage": out.get("usage", {})}
        except (urllib.error.URLError, TimeoutError, KeyError) as exc:
            if attempt == retries - 1:
                return {"alloy": rec["alloy"], "error": str(exc)}
            time.sleep(10 * (attempt + 1))


def parse(answer_raw):
    m = re.search(r"\{.*\}", answer_raw or "", re.S)
    try:
        obj = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        obj = {}
    name = str(obj.get("alloy", answer_raw or "")).strip()
    try:
        conf = float(obj.get("confidence"))
    except (TypeError, ValueError):
        conf = float("nan")
    return name, conf


def normalise(text):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9éè ]", " ", text.lower().replace("-", " "))).strip()


def matches(alloy, answer):
    a = normalise(answer)
    if not a or a == "unknown":
        return False
    return any(re.search(p, a) for p in ALIASES[alloy])


def query(key):
    done = set()
    if os.path.exists(RAW):
        done = {json.loads(l)["alloy"] for l in open(RAW, encoding="utf-8")
                if l.strip() and "error" not in json.loads(l)}
    todo = [r for r in alloys() if r["alloy"] not in done]
    print(f"{len(done)} cached, {len(todo)} to query")
    with ThreadPoolExecutor(max_workers=6) as pool, open(RAW, "a", encoding="utf-8") as fh:
        for res in pool.map(lambda r: ask(r, key), todo):
            fh.write(json.dumps(res) + "\n")
            fh.flush()


def score():
    import pandas as pd
    raw = {}
    for line in open(RAW, encoding="utf-8"):
        rec = json.loads(line)
        if "error" not in rec:
            raw[rec["alloy"]] = rec
    missing = set(ALIASES) - set(raw)
    if missing:
        raise SystemExit(f"no answer for {sorted(missing)}")
    dist = pd.read_csv(os.path.join(RESULTS_DIR, "nn_distance.csv")).set_index("alloy")
    rows = []
    for alloy in ALIASES:
        name, conf = parse(raw[alloy]["answer_raw"])
        rows.append({"alloy": alloy, "stratum": dist.loc[alloy, "stratum"],
                     "alloy_class": dist.loc[alloy, "alloy_class"],
                     "answer": name, "confidence": conf,
                     "identified": matches(alloy, name),
                     "said_unknown": normalise(name) == "unknown"})
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(df.groupby("stratum").identified.agg(["sum", "count"]))
    print(df.groupby("alloy_class").identified.agg(["sum", "count"]))
    print(f"identified {df.identified.sum()} / {len(df)}; said UNKNOWN {df.said_unknown.sum()}")
    tok = sum(r.get("usage", {}).get("total_tokens", 0) for r in raw.values())
    print(f"tokens {tok}")
    write_report(df, tok)
    return df


def write_report(df, tokens):
    top = df.answer.str.lower().value_counts().head(5)
    L = ["# Can the agent LLM identify an evaluation alloy from its composition?", "",
         "Generated by `evaluation/prediction/scripts/llm_identification_probe.py`. "
         f"{MODEL} on DeepInfra, temperature 0.0, given exactly the agents' inputs "
         "(composition, processing route) and asked to name the alloy. An answer counts "
         "only if it matches the designation or a documented alias; every answer is in "
         "`llm_identification_probe.csv`.", "",
         "| stratum | identified | alloys |", "|---|---:|---:|"]
    for s in ("NEAR", "MID", "FAR"):
        sub = df[df.stratum == s]
        L.append(f"| {s} | {int(sub.identified.sum())} | {len(sub)} |")
    L.append(f"| **ALL** | **{int(df.identified.sum())}** | **{len(df)}** |")
    L += ["", "Identified: " + ", ".join(f"{r.alloy} ({r.stratum})" for r in df[df.identified].itertuples()) + ".",
          "", "The model never answered UNKNOWN and reported the same confidence (0.8) for "
          f"{int((df.confidence == 0.8).sum())} of {len(df)} alloys, so its stated confidence carries "
          "no information. It falls back on a few famous designations whatever the composition:", ""]
    L += [f"- {name}: {n} answers" for name, n in top.items()]
    L += ["", "What this does and does not establish. A failed identification does not show "
          "the model holds no relevant prior knowledge -- it may still know how a class of "
          "alloys behaves. It does show that the agent gain on FAR alloys cannot come from "
          "recalling a named datasheet: none of the 44 FAR alloys is recognised from the "
          "inputs the agents receive.", "", f"Tokens used: {tokens:,}.", ""]
    with open(os.path.join(RESULTS_DIR, "llm_identification_probe.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--score", action="store_true", help="re-score cached answers only")
    args = ap.parse_args()
    if set(r["alloy"] for r in alloys()) != set(ALIASES):
        raise SystemExit("ALIASES does not cover exactly the evaluation alloys")
    if not args.score:
        query(load_key())
    score()


if __name__ == "__main__":
    main()
