# Superseded evaluation artefacts

Everything in this directory was produced **before** one or more of three
changes that altered the numbers. It is kept for provenance — these files back
the March 2026 submission — and must not be cited alongside the current
results, because in several places the two disagree about the same quantity.

Nothing here is deleted and nothing here is authoritative.

## What changed underneath these files

| # | change | commit | effect |
|---|---|---|---|
| 1 | ML models retrained on corrected δ features (`saved_models` → `saved_models_v2`) | `58e0bfb` | all ML-derived predictions |
| 2 | `PRODUCTION` correction profile replaces the drifted ablation copy; VRH elastic-modulus override gated off for SC/DS alloys | `9a46e63`, `86b11eb` | 49 of 471 rows; EM most of all |
| 3 | NIMONIC PE16 erratum, Ti 12.0 → 1.2 wt% | `dff5dd5` | 10 rows in every arm, and the FAR/ALL strata that contain them |

## Contents

### `results/{all,SSS,SC_DS,precip}/`

The consolidated six-arm comparison behind the March submission's Table 1
(`full_system`, `ml_only`, `ml_deterministic`, `llm_only`, `gpt4.1`,
`gpt4.1_ft`). Superseded by all three changes above.

These files carry a second defect independent of the changes: **the arms do
not share a row set.** `ml_deterministic` holds 471 rows, the three LLM arms
hold 466 (missing RGT\* 13 at five temperatures), and `full_system` holds 461 —
missing RGT\* 13 *and* all ten NIMONIC PE16 rows. Any metric computed by
averaging each file independently therefore scores the proposed system on a
strictly easier subset than the baselines it is compared against, PE16 being
the single worst alloy for every ML arm. Current results are computed on a
common row set for exactly this reason.

### `output/predictions_*.csv`

Timestamped raw runs from 26 Feb – 2 Mar 2026, including partial and
duplicated attempts. Superseded by all three changes. Retained because the
consolidated `results/` files above were built from them.

### `output/{full_system,llm_only,ml_deterministic,ml_only}_{sss,precip,sc_ds}.csv`

The per-dataset March runs that fed the consolidated tables. Same vintage and
same limitations as `predictions_*`.

### `output/seed42_v1_*.csv`

Seed-42 ML-only and ML+deterministic runs against the **v1** models, kept as
the direct before/after against `saved_models_v2`. Superseded by change 1.

### `output/seed42_v2_ml_deterministic_*.csv`

v2 models but the **`LEGACY_ABLATION`** correction profile, which differs from
what the evaluator actually applies on 49 rows (2 UTS, 12 elongation, 35
elastic modulus). Superseded by change 2; replaced by
`seed42_v2prod_ml_deterministic_*`. The profile itself is still defined in
`backend/alloy_crew/physics_corrections.py` so these numbers stay reproducible.

## The three LLM baselines have not been re-run

`gpt4.1`, `gpt4.1_ft` and `llm_only` exist **only** in this directory. Every
one of them was scored on NIMONIC PE16's incorrect composition, and re-running
them requires an OpenAI key that is not present in the working tree.

This matters most for `gpt4.1_ft`, the strongest baseline. On elastic modulus
it reaches 8.26 GPa MAE with the defective rows included and 8.17 GPa without,
against 8.22 / 8.26 for ML+physics — so **which method wins that property
depends entirely on whether the known data error is left in the baseline's
input.** Until the re-run happens, no elastic-modulus ranking against this
baseline should be published.

`evaluation/prediction/results/ft_baseline_analysis.md` reads
`results/all/gpt4.1_ft.csv` from this directory and excludes the PE16 rows for
that reason. Re-point `ft_baseline_analysis.py --ft-results` at a fresh run and
pass `--keep-erratum-rows` once the baseline has been re-executed.

### Re-run command

```bash
for ds in sss precip sc_ds; do
    python evaluation/prediction/scripts/generate_predictions.py \
        --llm-only --model openai-ft --dataset $ds --seed 42 \
        --output evaluation/prediction/output/seed42_ft_${ds}.csv
done
```

Run the three category datasets separately, **not** `--dataset all`. That option
reads `data/all_categorized.jsonl`, an untracked 28 Feb file which still carries
PE16 at Ti = 12.0 wt% and diverges from the tracked per-category files on three
further alloys. The erratum was applied to `precip.jsonl` only, because that is
what the campaign actually consumed. Using `--dataset all` would silently
reintroduce the defect this re-run exists to remove.

Run all 471 rows rather than patching the ten PE16 rows into the archived file.
The archived run used sampling temperature 0.3 with no seed, so it is not
reproducible row-by-row and a patched hybrid would mix two sampling regimes;
`--seed 42` drives the temperature to 0.0 and makes the new baseline
reproducible in a way the old one never was.

## Current authoritative results

| table | file |
|---|---|
| stratified MAE/R² by NEAR/MID/FAR | `results/stratified_metrics.csv` |
| nearest-neighbour distances and strata | `results/nn_distance.csv`, `results/nn_distance_strata.csv` |
| ML-only memorisation check | `results/memorisation_check.csv` |
| fine-tune overlap and its effect | `results/ft_overlap.csv`, `results/ft_stratified_metrics.csv` |
| conformal coverage | `results/conformal_coverage.csv`, `results/conformal_intervals.csv` |
| Waspaloy calibration sensitivity | `results/waspaloy_sensitivity.csv` |
| raw predictions, current arms | `output/seed42_v2_ml_only_*`, `output/seed42_v2prod_*`, `output/stageb_seed4[2-6]_*` |
