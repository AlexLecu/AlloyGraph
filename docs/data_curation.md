# Data curation and errata

Where the alloy data comes from, every correction applied to it since, and how
to check that the committed files are what they claim to be.

## Provenance chain

```
CAST_Wrought_Alloys.xlsx                 hand-annotated from published datasheets
  backend/superalloy_preprocess/annotated_data/
        |
        v
all_alloys.jsonl  ->  final_alloy_data_enriched_v2.jsonl        upstream, uncorrected
  backend/alloy_crew/models/training_data/
        |
        +--> train_77alloys_v2.jsonl          77 alloys, corrected  -> saved_models_v2
        |
        +--> evaluation/prediction/data/{SSS,precip,sc_ds}.jsonl   88 alloys, corrected
```

The upstream files are a faithful transcription of the annotation spreadsheet,
including its defects. Corrections live in the ledger, not in the upstream
files, so the raw extraction stays auditable.

## The errata ledger

`evaluation/prediction/data/errata_ledger.json` lists **every** correction, with
the value before and after and the reason. Sixteen entries:

| class | n | what it fixes |
|---|---|---|
| `closure` | 6 | composition not summing to ~100 wt% — a transposed digit, a dropped zero, an omitted element |
| `sign` | 6 | a negative strength or modulus |
| `magnitude` | 3 | a misplaced decimal or transposed leading digit |
| `withdrawn` | 1 | impossible, and the source cannot say what it should be |

Fourteen apply to the training set and were **pre-existing**: made by hand when
the 77-alloy set was built, by no committed script, and undocumented until now.
Two apply to the evaluation set and were made during the revision:

- **E01** NIMONIC PE16, Ti 12.0 → 1.2 wt%
- **E02** RGT* 13, yield strength at 871 °C withdrawn

### Why the pre-existing edits are recorded rather than reversed

They are correct. Each is verifiable against something other than the editor's
judgement:

- every `closure` edit restores a composition that summed to 67–110 wt% back to
  within 0.5 wt% of 100
- every `sign` edit produces a value that continues the alloy's own temperature
  series and sits below the tensile strength at the same point
- both `magnitude` edits replace a value that is physically impossible for a
  nickel-base alloy (a 15.6 GPa modulus, a tensile strength below the alloy's
  own yield strength) with one consistent with its series

What was wrong was not the edits but that they were invisible. The models were
fitted on a training set nobody could regenerate.

### The RGT* 4 / RGT* 13 pair

`T14` and `E02` are the same defect in the same spreadsheet column: RGT* 4 reads
`-205` and RGT* 13 reads `-435`, and no other alloy in the corpus has a negative
strength.

They are resolved differently, deliberately. RGT* 4 was already corrected to
+205 in the committed training set before this revision began; that edit is
sound on the same grounds as the other `sign` entries and is left in place and
documented. RGT* 13 was **not** corrected, because the source cannot settle it:
the 982 °C column is empty for that row, leaving open the reading that the dash
means "no data" and 435 belongs to 982 °C. The RGT* 4 precedent is a project
decision, not source evidence, and was not treated as licence to guess.

If the original handbook is consulted and confirms 435 MPa at 871 °C, E02 can be
changed from a withdrawal to a correction: edit the ledger entry, re-run
`apply_errata.py --apply`, and regenerate.

## What the 88 evaluation alloys are

Each evaluation record carries a `_source` tag. The split is exactly:

| `_source` | n | what it is |
|---|---:|---|
| `test_split_30` | 22 | stratified 30% holdout of the curated corpus, split before any model was fitted (`create_train_test_split.py`) |
| `matweb_original` | 33 | MatWeb records, first extraction pass |
| `matweb_alloys` | 33 | MatWeb records, second extraction pass |

None of the 88 shares a name with any of the 77 training alloys. The two MatWeb
passes are extraction batches, not different kinds of source: both are MatWeb
transcriptions of manufacturer datasheets, so the split by *kind* of source is
**22 curated-corpus holdout against 66 datasheet-derived**.

**Manufacturer is not recorded.** No field in any evaluation or upstream file
names one, and deriving it from the alloy-name string does not work: 31 of the
88 carry no vendor token at all, and the ones that do are not reliable --
HASTELLOY X is a Haynes product but the string says nothing, while UDIMET has
changed hands. Any per-manufacturer count in the paper would be an invention.

## The 106 / 77 / 88 arithmetic

Check it with `verify_split_arithmetic.py`. Two things have to be said plainly
before it closes.

**A "record" is an alloy under one processing route and product form, not an
alloy.** The upstream file holds **106 records over 100 distinct alloy names** --
RENÉ\* 41 appears as Bar and as Sheet, and so on. Quoting 106 as a count of
alloys is wrong.

**The evaluation set is not the holdout.** The stratified split produced 27
holdout records; the 88-alloy evaluation set draws 22 of them plus 66 MatWeb
records.

### Step 1, the split: 106 = 77 + 27 + 2

Two records go to neither side:

| record | composition | why |
|---|---|---|
| RENÉ\* 41 (wrought, Sheet) | empty, 0 wt% | cannot be featurised; the Bar record of the same alloy carries a full composition and is in training |
| TD Nickel (wrought, Sheet) | empty, 0 wt% | same |

Training and holdout do not overlap at all.

### Step 2, holdout to evaluation set: 27 = 22 + 5

Five holdout records never reach a shipped file. **One is explained, four are
not.**

| record | composition | disposition |
|---|---|---|
| UNITEMP\* AF2-1DA | empty, 0 wt% | classifier reads Ni = 0, routes it to `other` (out of scope). Same defect as the two above |
| INCONEL\* MA758 | 7 elements, 103.8 wt% | **no recorded reason** — classifies `solid_solution` |
| MC-102\* | 9 elements, 99.7 wt% | **no recorded reason** — classifies `solid_solution` |
| TD NiCr | 2 elements, 98.0 wt% | **no recorded reason** — classifies `solid_solution` |
| NX188(DS) | 4 elements, 100.0 wt% | **no recorded reason** — classifies `precip_sc_ds` |

The four have valid compositions, full yield-strength, tensile-strength and
elongation series, and classify cleanly under the committed classifier. They are
not duplicates of shipped alloys: nearest-neighbour distances run 0.92 to 12.5.
Re-running `create_categorized_datasets.py` on the committed inputs puts all
four into a shipped category, so **the committed evaluation set cannot be
reproduced from the committed scripts** — it is five records short of what they
produce.

### Why no scope rule was encoded

Encoding the exclusions as a rule was attempted and **abandoned, deliberately.**
No principled, composition-based rule covers all four, and the reason is in the
data rather than in the wording.

**MC-102\* is not out of scope by any criterion, and an earlier note here was
wrong to group it with the ODS alloys.** Its composition is
Ni 64, Cr 20, Mo 6, W 2.5, Nb 6, Ta 0.6, closing to 99.7 wt%, with six
temperatures of yield strength, tensile strength and elongation. It is a
conventional cast Ni--Cr--Mo--Nb alloy of the 625 family: its nearest neighbour
in the corpus is Cast Alloy 625 at d = 5.13. By the project's own
γ′-former index, `Al+Ti+Ta+0.35*Nb` = **2.70**, it is a γ′/γ″-class alloy --
`create_categorized_datasets.classify_alloy` labels it solid-solution only
because its test reads Al and Ti and ignores Nb and Ta. There is no scope reason
to exclude it.

**For two of the others the deciding property is not recorded at all.** MA758 is
dispersion strengthened by Y₂O₃ and TD NiCr by ThO₂, and neither oxide appears
in the stored composition: MA758 reads as Ni--30Cr--3Nb and TD NiCr as
Ni--20Cr and nothing else. "ODS" is therefore knowable only from the alloy
name, and a name-matching rule is the approach already rejected for
manufacturer attribution.

Candidate composition rules were tested against the shipped 88 and the 77
training alloys:

| candidate rule | catches | collateral |
|---|---|---|
| `Al >= 7 wt%` | NX188 | none, in either set |
| `< 3 elements` | TD NiCr | removes **TD Nickel from training** |
| `composition sum > 103 wt%` | MA758 | removes **12 shipped evaluation alloys** |
| `Mo >= 15 wt%` | NX188 | removes **12 shipped evaluation alloys** |
| any of the above | — | **none catches MC-102\*** |

Only the NiAl rule is clean. The rest either change the training set, which is
frozen, or delete alloys the paper reports on.

So the four remain what they are: an undocumented legacy exclusion, now
measured and named rather than guessed at. Two defensible ways forward, neither
taken here without a decision:

1. **Declare them.** A named list with a per-entry rationale, in the pattern
   `errata_ledger.json` already uses, plus the `Al >= 7 wt%` rule for NX188.
   Honest about being a list rather than a rule, and makes the evaluation set
   reproducible from the scripts.
2. **Re-admit MC-102\***, which nothing justifies excluding, and declare only
   the three that have a stated reason. This changes the evaluation set to 89
   alloys and every metric with it, so it is a decision for before the next
   results freeze, not after.

### One authoritative path

`data/all_categorized.jsonl` and `data/manifest.json` were snapshots of an
earlier run that nothing regenerated: the manifest counted 111 records against a
file holding 106, that file carried 8 records no category file holds, and it was
missing RGT\* 13, which *is* shipped in `precip.jsonl`. This was a live second
path, not a leftover -- `generate_predictions.py --dataset all` reads it.

Both are now **derived views**, rebuilt from the category files by
`rebuild_categorized_index.py`, which is concatenation and nothing else:

```bash
python evaluation/prediction/scripts/rebuild_categorized_index.py --check
```

fails if they ever drift again. The four category files stay authoritative, and
`SSS.jsonl`, `precip.jsonl` and `sc_ds.jsonl` were not touched -- no shipped
number moved. The combined index is now 99 records: the 88 evaluated plus the 11
in `other`.

`*.jsonl` is globally git-ignored; `SSS.jsonl`, `precip.jsonl` and `sc_ds.jsonl`
are force-added exceptions and are the tracked source of truth.
`other.jsonl` and the combined index are not tracked, so on a fresh clone the
index rebuilds to the 88 evaluated records alone. Nothing evaluates `other`, so
no result depends on it.

A further eight MatWeb records are classifiable but unshipped, several of them
apparent name duplicates of shipped alloys (`TRW-NASA VIA` beside
`TRW-NASA VI A`, `MM-200` beside `MAR-M\* 200`). Not investigated here.

## Checks

Run both before trusting any published number.

```bash
python evaluation/prediction/scripts/verify_training_provenance.py
```

Rebuilds the training set from the upstream file plus the ledger and compares it
cell for cell against the committed file. It currently passes on all 77 records,
which is the claim that the committed training set is reproducible from
committed sources. It fails if anyone hand-edits the training data again.

```bash
python evaluation/prediction/scripts/data_sanity_sweep.py
```

Sweeps all 165 training and evaluation records for measurements that cannot be
real: negative or zero strengths and moduli, yield strength above tensile
strength by more than rounding, elongation beyond superplastic range,
compositions far from closure. Exits non-zero if any remain; it currently finds
none.

It also lists 48 rows that fall outside the window `train_ml_models` accepts but
are perfectly real — a superalloy does yield at 9 MPa near 1200 °C and does
elongate 159% once γ′ dissolves. Those are reported separately, because that
window is a training convenience and not a statement about physics. Training
silently discards them; the sweep makes that visible.

## Applying an erratum to existing results

Prediction CSVs each carry their own snapshot of the ground truth in the
`actual_*` columns, so a correction made afterwards leaves them stale.

```bash
python evaluation/prediction/scripts/apply_errata.py            # dry run
python evaluation/prediction/scripts/apply_errata.py --apply
```

It touches only the cells named in its `ERRATA` list and never reads or writes a
`pred_*` column. Re-running the arms is unnecessary: a model sees composition,
processing and temperature, none of which a ground-truth correction changes.

Do not replace this with a wholesale re-sync of `actual_*` from the data files.
That was tried: the evaluation data records each property at its own measured
temperature (elongation at 650 °C where yield strength is at 649 °C), the
harness matches within 5 °C, and an exact re-match reported 902 false
differences and would have deleted hundreds of good measurements.

## What is still open

`RGT* 4` and the other thirteen pre-existing edits are documented but were never
independently re-checked against the original datasheets — only against internal
consistency. If the handbook is available, spot-checking the six `closure` edits
would be worth an hour.

No retraining has been done for any of this. The models remain those fitted on
`train_77alloys_v2.jsonl` as committed, which is exactly what the ledger now
reproduces.
