#!/usr/bin/env python3
"""Write paper_assets/MANIFEST.md, listing each asset and its provenance.

The per-asset rows are generated rather than typed, so an asset that is added,
renamed or dropped cannot silently fall out of the manifest. The prose sections
live in MANIFEST_TEMPLATE.md next to this script.

Usage:
    python write_manifest.py
"""

import argparse
import hashlib
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DEFAULT_ASSETS = os.path.join(PROJECT_ROOT, "paper_assets")
TEMPLATE = os.path.join(SCRIPT_DIR, "MANIFEST_TEMPLATE.md")

#: asset -> (producing script, data it consumes)
PROVENANCE = {
    "figures/nn_distance_hist.pdf": (
        "evaluation/paper_assets/make_figures.py",
        "evaluation/prediction/results/nn_distance.csv"),
    "figures/parity_stratified.pdf": (
        "evaluation/paper_assets/make_figures.py",
        "evaluation/prediction/output/stageb_seed4[2-6]_full_system_*.csv "
        "(5-seed mean); nn_distance.csv (strata); "
        "stratified_metrics.csv (annotated MAE)"),
    "figures/accuracy_vs_distance.pdf": (
        "evaluation/paper_assets/make_figures.py",
        "evaluation/prediction/output/seed42_v2prod_ml_deterministic_*.csv, "
        "seed42_v2prod_ml_physics_kg_*.csv, stageb_seed4[2-6]_full_system_*.csv; "
        "nn_distance.csv (bands)"),
    "figures/coverage_by_temperature.pdf": (
        "evaluation/paper_assets/make_figures.py",
        "evaluation/prediction/results/conformal_intervals.csv"),
    "figures/mcq_accuracy.pdf": (
        "evaluation/paper_assets/make_figures.py",
        "evaluation/chatbot/results/mcq_report.json"),
    "tables/T1_main.tex": (
        "evaluation/paper_assets/make_tables.py",
        "evaluation/prediction/results/headline_metrics.csv"),
    "tables/T2_stratified.tex": (
        "evaluation/paper_assets/make_tables.py",
        "evaluation/prediction/results/stratified_metrics.csv"),
    "tables/T3_provenance.tex": (
        "evaluation/paper_assets/make_tables.py",
        "evaluation/prediction/results/stratified_metrics.csv (percentages); "
        "owl_reasoning_ablation.md (ontology row)"),
    "tables/T4_design.tex": (
        "evaluation/paper_assets/make_tables.py",
        "evaluation/design/results/random_search_baseline.json"),
    "tables/T5_conformal.tex": (
        "evaluation/paper_assets/make_tables.py",
        "evaluation/prediction/results/conformal_coverage.csv"),
    "tables/T6_baselines.tex": (
        "evaluation/paper_assets/make_tables.py",
        "evaluation/prediction/results/stratified_metrics.csv"),
}


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(65536), b""):
            h.update(block)
    return h.hexdigest()[:12]


def rows_for(assets_dir, prefix):
    lines, missing = [], []
    for asset, (script, data) in PROVENANCE.items():
        if not asset.startswith(prefix):
            continue
        path = os.path.join(assets_dir, asset)
        name = os.path.basename(asset)
        if not os.path.exists(path):
            missing.append(asset)
            lines.append(f"| `{name}` | `{script}` | {data} — **NOT BUILT** |")
            continue
        size = os.path.getsize(path) / 1024
        lines.append(f"| `{name}` ({size:.0f} kB, `sha256:{digest(path)}`) "
                     f"| `{script}` | {data} |")
    return lines, missing


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--assets", default=DEFAULT_ASSETS)
    args = ap.parse_args()

    fig_rows, fig_missing = rows_for(args.assets, "figures/")
    tab_rows, tab_missing = rows_for(args.assets, "tables/")

    text = open(TEMPLATE).read()
    text = text.replace("{FIGURE_ROWS}", "\n".join(fig_rows))
    text = text.replace("{TABLE_ROWS}", "\n".join(tab_rows))

    # Anything present in the folder that no rule claims: report it rather than
    # letting an unexplained file ship with the manuscript.
    claimed = set(PROVENANCE)
    found = set()
    for sub in ("figures", "tables"):
        d = os.path.join(args.assets, sub)
        if os.path.isdir(d):
            found |= {f"{sub}/{f}" for f in os.listdir(d) if not f.startswith(".")}
    orphans = sorted(found - claimed)
    if orphans:
        text += ("\n## Unclaimed files\n\nPresent in `paper_assets/` with no "
                 "entry above. Either add provenance or delete them:\n\n"
                 + "\n".join(f"- `{o}`" for o in orphans) + "\n")

    out = os.path.join(args.assets, "MANIFEST.md")
    os.makedirs(args.assets, exist_ok=True)
    with open(out, "w") as fh:
        fh.write(text)

    print(f"{len(fig_rows)} figures, {len(tab_rows)} tables -> {out}")
    if fig_missing or tab_missing:
        print("  NOT BUILT:", ", ".join(fig_missing + tab_missing))
    if orphans:
        print("  unclaimed:", ", ".join(orphans))


if __name__ == "__main__":
    main()
