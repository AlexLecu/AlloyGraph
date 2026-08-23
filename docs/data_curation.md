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
