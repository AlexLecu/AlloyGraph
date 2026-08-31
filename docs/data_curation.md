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

### The scope exclusion list

Every record a source offers and the evaluation set declines is now named in
`evaluation/prediction/data/scope_exclusions.json`, the same contract as the
errata ledger: a change means an entry, not new logic. Thirteen entries under
three rules.

```bash
python evaluation/prediction/scripts/verify_evaluation_provenance.py
```

rebuilds the four category files from the sources plus the list plus the two
evaluation errata. **Membership and order reproduce exactly** in all four,
which is what the list governs, and the script fails if that stops being true.

**Rules.** Each was tested against the shipped 88 and the 77 training alloys
before being adopted, and each removes nothing that is currently kept:

| rule | test | catches |
|---|---|---|
| R1 unfeaturisable | composition is empty | UNITEMP\* AF2-1DA |
| R2 NiAl intermetallic | Al ≥ 7.0 wt% | NX188(DS) |
| R4 exact duplicate | composition distance < 0.01 to a record already accepted | MM-200, G-50 Grade 140, INCONEL 622 Sheet |

R1 matters beyond its one record: the untouched classifier reached the same
outcome for the wrong reason, reading the absent nickel as Ni = 0 and filing the
record as Fe-Ni base.

**Declared, because no rule can reach them.** Seven entries rest on alloy
identity. For MA758 and TD NiCr the deciding property is not in the data at all:
they are dispersion strengthened by Y₂O₃ and ThO₂ respectively, and neither
oxide appears in the stored composition. Every composition test that catches
them does collateral damage — `sum > 103 wt%` removes 12 shipped alloys and
UDIMET\* 630 from the frozen training set; `< 3 elements` removes TD Nickel from
training. The rest are near-duplicates that R4 misses (INCONEL HX at d = 0.42,
TRW-NASA VIA at d = 3.40, both the same alloy as a kept record) or records whose
composition never closes (AL 825 at 67.9 wt%, INCOLOY 908 at 61.2 wt%); no
closure threshold separates those from INCOLOY 803, which is shipped at 65.6.

**Historical, and disclosed.** One entry, `X05`, has no defensible reason.

### MC-102\*: an exclusion with no justification, measured rather than assumed

MC-102\* is in scope on every criterion. Ni 64 / Cr 20 / Mo 6 / W 2.5 / Nb 6 /
Ta 0.6, closing to 99.7 wt%, with yield strength, tensile strength and
elongation at six temperatures; a conventional cast Ni--Cr--Mo--Nb alloy of the
625 family whose nearest training neighbour is Cast Alloy 625 at d = 5.13. By
the project's own γ′-former index, `Al+Ti+Ta+0.35*Nb` = 2.70, it is γ′ class;
`classify_alloy` files it as solid-solution only because that test reads Al and
Ti and ignores Nb and Ta.

It is kept out **only** so the committed results stay reproducible, and the cost
of that is measured, not asserted to be small. d = 5.13 puts it in **FAR**, the
stratum the headline agent claim rests on. Including its six rows:

| property | scope | ML+physics+KG shipped | with MC-102\* | change |
|---|---|---:|---:|---:|
| YS | FAR | 110.10 | 108.50 | **−1.5%** |
| YS | ALL | 92.37 | 91.95 | −0.5% |
| UTS | FAR | 137.84 | 134.29 | −2.6% |
| UTS | ALL | 112.75 | 111.48 | −1.1% |
| EL | FAR | 10.89 | 11.38 | **+4.4%** |
| EM | either | 7.84 | 7.84 | none — no measured modulus |

Every strength number moves **in the platform's favour**, so the exclusion is
not flattering the result. Elongation is the exception and worth stating: the
arms predict roughly 30% against a measured 5% at room temperature, which is the
already-known weakness on cast ductility rather than a new one. Full table in
`evaluation/prediction/results/mc102_sensitivity.md`.

Re-admit at the next results freeze: the evaluation set becomes 89 alloys and
every metric moves.

### The category files are the artefact of record for their values

Membership reproduces from the sources; **cell values do not**, in 20 of 99
records over 397 cells. The missing step was searched for and is not in this
repository.

**It is not a newer MatWeb extraction.** That was the first guess and it is
wrong. MatWeb's own record for AL 276, in every scrape on disk
(`backend/scrape/Data/{all_materials,materials3,processed_materials}.jsonl`) and
in `matweb_unique_standardized.xlsx`, carries a single room-temperature bound:

```
yield_strength_mpa: [{temp_c: null, min: 283.0, qualifier: ">=", raw: ">= 283 MPa @Strain 0.2 %"}]
```

The shipped file carries 415 / 380 / 345 / 315 MPa at 21 / 93 / 204 / 316 °C --
a 70 / 200 / 400 / 600 °F ladder off a manufacturer datasheet, which no MatWeb
summary page ever held. The enrichment came from reading datasheets directly
between the January extraction and the March evaluation files, and no
intermediate was kept.

Searched and excluded: `.new/`, `Data/`, `figshare-data/`, `alloygraph-data.zip`,
`evaluation/_archive/`, `backend/_archive/`, `backend/scrape/Data*`, every
`output_data/` file, the standardised MatWeb spreadsheet, and the git history of
`SSS.jsonl` (one commit, 2026-03-18, already carrying the series).
`figshare-data/evaluation/sss.jsonl` matches the shipped values exactly and is a
**byte-identical copy** of `SSS.jsonl`, not a source.

So the three evaluated category files are the artefact of record for their
values, the way `kg.png` is for figure 2: reproducible in structure, not
regenerable in content. They are tracked for that reason.

**What the 397 cells are:**

| kind | cells | what it is |
|---|---:|---|
| added | 236 | the source has nothing; the shipped file has a measurement |
| removed | 148 | the source has a value the shipped file drops |
| changed | 13 | both present and different |

Net, the shipped files hold **88 more measurements** than the sources. Most of
the 148 removals are the elasticity clean-up (79 of them): `clean_elasticity.py`
strips shear-modulus values that MatWeb mixed into Young's-modulus fields. The
13 changes are the same defect handled by substitution rather than deletion --
AL 600 at 76 → 207 GPa, INCONEL 725 at 78 → 204, C276 at 79 → 205 are all
G → E corrections -- plus three elongation values differing in the first decimal.

**The deltas pass the standing check.** `data_sanity_sweep.py` sweeps all 165
training and evaluation records, these 20 included, and reports **0 physically
impossible measurements**: no negative or zero strengths, no yield above
tensile beyond rounding, no impossible elongation, no composition far from
closure. The enrichment is not smuggling anything past the guard rails.

**What this means for reuse.** Regenerating the evaluation set from the
committed sources gives the right 88 alloys with poorer property coverage, not a
different alloy set. Anyone reproducing the paper's numbers must use the
committed category files, which is why they are tracked and why
`verify_evaluation_provenance.py` gates on membership and prints the value delta
rather than implying the files are derivable.

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
