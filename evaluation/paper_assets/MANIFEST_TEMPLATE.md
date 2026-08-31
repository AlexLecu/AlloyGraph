# Manuscript asset manifest

Every asset in `paper_assets/` and what produced it. The folder is git-ignored:
the scripts are committed, the outputs are not, so an asset is always
regenerable and can never drift from the data behind it.

Regenerate everything:

```bash
python evaluation/paper_assets/make_figures.py
python evaluation/paper_assets/make_tables.py
python evaluation/paper_assets/write_manifest.py
```

Both generators read only committed results. Nothing recomputes a metric that a
results table already holds, so a figure and a table cannot disagree.

## Figures

All vector PDF, no raster content, Helvetica embedded as subsetted TrueType so
text stays selectable and editable. Sized for a single journal column (3.5 in)
at 8 pt base type; `mcq_accuracy.pdf` is the one exception, set to the full text
width (7.16 in) because twelve grouped bars each carrying a value label cannot
fit a column without the labels colliding. The manuscript already runs that
figure across both columns. Palette is Okabe--Ito, distinguishable under
deuteranopia, protanopia and tritanopia, and separable in greyscale.

| Asset | Script | Data consumed |
|---|---|---|
{FIGURE_ROWS}

## Tables

LaTeX with `booktabs`, no vertical rules. Each file is a complete float
(`table` or `table*`) with caption, label and footnote, ready to `\input`.
Verified to compile with `pdflatex` against `booktabs`.

| Asset | Script | Data consumed |
|---|---|---|
{TABLE_ROWS}

## Architecture diagrams — manual export required

The manuscript carries three architecture figures. Two are draw.io diagrams
with committed, uncompressed-XML sources; the third has no source and cannot be
regenerated at all. Which file is which was not obvious and cost an audit, so it
is written down here.

| Figure in the paper | Source | Rendered PNG (tracked) |
|---|---|---|
| System architecture | `docs/paper/figures/fig1_system_architecture_big.drawio` (`alloygraph-arch-v10`) | `docs/paper/figures/system_architecture.pdf` (vector) and `.png` |
| HAYNES 230 knowledge graph | **none — screenshot, see below** | `docs/paper/figures/kg.png` |
| Prediction / design pipeline | `docs/paper/figures/fig3_evaluation_pipeline.drawio` (`pipeline-v4`) | `docs/paper/figures/agent_pipeline.png` |

Both sources were checked label-for-label against the rendered figure and are
current. The alloy counts are the quick tell: the source and the paper figure
both read "88 Ni superalloys", and any export reading **99** is stale.

Figure 1 was revised again in the same way: the Data Sources box now reads
"88 Ni superalloys · Held-out test data (stratified by novelty)". The previous
wording, "Independent test data", contradicted the paper's own central finding --
13 of those 88 are vendor-renamed near-duplicates of training alloys, which is
the reason the evaluation is stratified at all. The detail text inside the
inner boxes also went from 14 px to 16 px; the cropped export is 5500x3049
before and after, so nothing overflowed.

Figure 1 carries three corrections made in `a076a77`, so an export predating it
is stale in three more places: the triplestore holds **77** alloy variants and
not 106, the orchestration band names no vendor (the provider chain is
DeepInfra/Together/Groq/OpenAI, so "(Groq API)" was both stale and misleading),
and the ensemble uses **50-80** engineered features, which is what
`saved_models_v2/metrics.json` records -- `n_features` 80 for YS and UTS, 50 for
EL and EM -- against the "25+" the figure had claimed.

Two further draw.io files are **not used by the manuscript** and are kept only
as alternates. Do not export from them:

```
docs/paper/figures/fig1_system_architecture_compact_UNUSED.drawio   terse-caption variant of fig 1
docs/paper/figures/fig2_agent_workflow.drawio                       landscape two-panel alternate
```

`assets/system_architecture.png` is the same bytes as the tracked copy above,
serving the README. If figure 1 is re-exported, replace both.

### Figure 2 (HAYNES 230 knowledge graph) has no source file

`kg.png` is a **screenshot of the GraphDB Visual Graph view**, not a drawing.
Node labels in it are raw resource IRIs minted by the ingest pipeline
(`HAYNES__230_cast_YieldStrength_0851cc23`, `MPa_ae5f17e2`), which is how it can
be identified as a live view rather than an illustration.

To reconstruct it, ingest the data and open the variant node:

```
http://localhost:7200/graphs-visualizations?uri=\
  https://w3id.org/alloygraph/res/variant/HAYNES__230_cast
```

repository `AlloyGraph` (`GRAPHDB_REPO`), populated by
`backend/pipeline/run_pipeline_docker.py`. The IRI follows
`enrich_graphdb.safe_iri`, which maps `*` and space to `_`, so alloy
`HAYNES* 230` in processing route `cast` becomes `HAYNES__230_cast`.

**It will not come back identical, and cannot.** Measurement and quantity nodes
are minted by `mint_res_uuid` (`enrich_graphdb.py:81,218`) with a random
`uuid4` suffix, so every re-ingest gives every measurement a different IRI and
therefore a different label in the view. Node placement is force-directed and
unseeded as well. The rendered PNG is the artefact of record; the recipe above
reproduces the *structure*, not the image.

**`docs/paper/figures/kg.png` is therefore the artefact of record**, and is
tracked for that reason rather than as a convenience. The other two figures can
be rebuilt from their `.drawio` sources if the PNG is lost; this one cannot be
rebuilt from anything. It is the only copy that exists in a commit --
`paper_src/` is git-ignored, so the manuscript's own copy is not backed by
version control.

If figure 2 is ever regenerated, the new screenshot replaces this file, and the
figure in the paper changes with it: node labels carry fresh uuid suffixes and
the force-directed layout lands differently. Treat a re-capture as a new figure
needing a fresh caption check, not as a refresh.

**Export is scriptable.** The draw.io desktop app ships a CLI wrapper; install
it once with `brew install --cask drawio`, which links `drawio` onto the path.
Re-export figure 1 after any edit to its source:

```bash
SRC=docs/paper/figures/fig1_system_architecture_big.drawio
drawio --export --format pdf --crop --transparent \
  --output docs/paper/figures/system_architecture.pdf "$SRC"
drawio --export --format png --crop --width 5500 \
  --output docs/paper/figures/system_architecture.png "$SRC"
cp docs/paper/figures/system_architecture.pdf paper_src/Alloygraph_kbs/images/
cp docs/paper/figures/system_architecture.png assets/system_architecture.png
```

**The manuscript now compiles the PDF, not the PNG** -- `paper_src/images/`
holds only `system_architecture.pdf`, so that is the copy a re-export must
refresh. The PNG survives for one reader: the README. Four files move together
(source, tracked PDF, manuscript PDF, README PNG), and missing one is exactly
how the "99 Ni superalloys" export went stale.

`--crop` sizes the canvas to the content bounding box, which makes it a free
overflow test: if a font change pushes text outside its box, the exported
dimensions grow. Compare them across a re-export before trusting a type
change.

`system_architecture.pdf` is the vector version and is tracked: KBS R2.1 asks
for vector artwork, so it, not the PNG, is what should go to the publisher. It
carries no raster content and embeds every font as a subset. One caveat: five
faces are subsetted Helvetica (CID TrueType), but the glyphs Helvetica lacks --
`·`, `→`, `γ`, `′`, `±` -- fall back to a **Type 3** LucidaGrande subset. Type 3
here is vector glyph procedures rather than bitmaps, so it is resolution
independent, but some publishers' preflight flags Type 3 on sight. If Elsevier
objects, the fix is to set those characters in a face that has them rather than
to rasterise.

## Notes that belong with the numbers

**The "raw-feature models degrade to ~170 MPa on FAR" claim is not supported,
and T6 shows what is.** No raw-feature arm is anywhere near 170 on FAR yield
strength: GBM 121.9, RF 104.8, GPR 141.6, against 110.1 for the internal arms.
**RF on raw features beats the internal arms on FAR** (104.8 against 110.1), and
on FAR tensile strength both tree baselines beat them clearly (112.3 and 113.5
against 137.8) -- RF also beats the full system there (116.9). The only arm near
or above 170 is the Gaussian process on *engineered* features (301.7), which is
the physics-feature arm, not a raw one, and its failure is extrapolation rather
than weak features: it is the best arm in cross-validation.

The defensible claim is narrower: distance from the training set costs every arm
accuracy, and the full system holds the largest FAR margin on yield strength
(89.8 against 104.8 for the best baseline). It does not hold that margin on
tensile strength.

**Significance of the headline comparison** is in
`evaluation/prediction/results/wilcoxon_tests.csv`, from
`evaluation/prediction/scripts/wilcoxon_tests.py`: paired Wilcoxon signed-rank
on per-case absolute errors, full system (five-seed mean) against ML+physics+KG.
Strength improvements are significant, ductility and stiffness are not. Reported
at both case level and alloy level, because 285 cases come from 88 alloys and
are not independent -- the alloy-level test is the conservative one.

**`accuracy_vs_distance.pdf` is banded, and the banded numbers do not support
the "clean step at d = 2.0" reading.** Panel (a) is the relative yield-strength
error reduction from adding KG anchoring to ML+physics, per distance band;
panel (b) is the same bands in absolute MAE for three arms. Values are
recomputed from the prediction CSVs and asserted against `BAND_EXPECTED` in
`make_figures.py`, which refuses to draw if they move.

| band | alloys | n | ML+physics | + KG | gain | full system (5 seeds) |
|---|---:|---:|---:|---:|---:|---:|
| 0–0.5 | 5 | 12 | 36.71 | 38.80 | **-5.7%** | 72.77 ± 0.49 |
| 0.5–1 | 5 | 24 | 137.70 | 96.22 | +30.1% | 59.78 ± 1.25 |
| 1–1.5 | 1 | 12 | 63.95 | 33.25 | +48.0% | 35.17 ± 9.60 |
| 1.5–2 | 2 | 8 | 188.83 | 131.25 | +30.5% | 100.81 ± 11.73 |
| 2–3 | 10 | 25 | 90.51 | 90.51 | +0.0% | 92.53 ± 2.55 |
| 3–4.5 | 21 | 66 | 70.57 | 70.41 | **+0.24%** | 63.96 ± 2.94 |
| ≥4.5 | 44 | 138 | 110.10 | 110.10 | +0.0% | 89.83 ± 1.32 |

Two statements that have been made about these bands are false at this
resolution, and both are visible in the figure:

1. *"Every band below 2.0 gains at least 25%."* The **0–0.5 band loses 5.7%** --
   anchoring makes the nearest near-duplicates slightly worse. Those are the
   alloys ML already predicts best (36.71 MPa, the lowest MAE of any band), so
   there is little to win and something to lose.
2. *"No band above 2.0 gains more than 0.2%."* The 3–4.5 band gains **0.24%**.
   Trivial in size, but it is not zero, and it is not zero for a reason: KG
   anchoring fires at d = 3.679 on Haynes 625.

**The deeper problem is that these bands are not samples.** KG anchoring changes
a yield-strength prediction for **6 of 86 alloys and 21 of 285 rows**. For the
other 80 alloys the two arms are bit-identical, so every band average is a
handful of alloys diluted by rows where nothing happened:

| alloy | d | rows changed | MAE ML+physics | MAE + KG |
|---|---:|---:|---:|---:|
| MAR-M* 200 | 0.024 | 1 | 39.67 | **149.20** |
| NIMONIC* 86 | 0.037 | 1 | 84.86 | 0.40 |
| Haynes 214 | 0.526 | 9 | 141.32 | 30.71 |
| Haynes 263 | 1.145 | 5 | 97.24 | 23.56 |
| Haynes 718 | 1.663 | 3 | 203.34 | 49.80 |
| Haynes 625 | 3.679 | 2 | 110.29 | 104.80 |

So the "step function" is six alloys, one of which (MAR-M* 200) anchoring makes
**276% worse** and one of which sits well above the supposed d = 2.0 boundary.
The honest claim is that anchoring fires rarely, that when it fires on a true
near-duplicate it helps a great deal, and that it can also misfire. A claim
about a threshold in distance is not supported by six points.

Panel (b) carries its own correction. The full-system curve does **not** sit
below the other arms everywhere: it is worst of the three in the 0–0.5 band
(72.77 against 36.71) and marginally worst in 2–3 (92.53 against 90.51). It is
clearly below beyond d = 4.5 (89.83 against 110.10), which is the part of the
agent story that survives.

**The parity panels carry +/-10% and +/-20% tolerance bands.** Two nested
light-grey fills around the perfect-prediction diagonal, no edges, alpha 0.07
each, drawn beneath the points, so the inner +/-10% region reads darker because
it carries both. They give the reader a fixed relative-error reference that MAE
in absolute units cannot: 100 MPa is a different thing at 300 MPa than at
1200 MPa.

Read them per panel, not across panels. Each panel is scaled robustly to its
own data, so the elastic-modulus axis starts near 100 GPa rather than zero and
its bands look wide -- the percentages are still exact, since the bands are
defined off the diagonal through the origin, but visually comparing band widths
between panels means nothing.

**The bands are not currently explained anywhere in the figure**, which has no
room for a fourth legend entry. The manuscript caption must say what they are.

**Parity MAE annotations are read from `stratified_metrics.csv`, not recomputed.**
The plotted points are the row-wise mean prediction over the five campaign
seeds; averaging predictions first cancels independent seed noise, so those
points give a slightly lower error (78.6 MPa on yield strength) than the mean
of the per-seed errors the tables report (80.9 MPa). The tables' definition is
the one quoted on the figure, so figure and table agree.

**RGT* 13 at 871 °C has been resolved — this note previously said otherwise.**
That row carried a yield strength of **-435 MPa**. It was withdrawn as erratum
E02 in commit `173a4ac`, after the annotated source proved unable to settle what
the value should have been: the dash may be a stray sign on 435 MPa, or a "no
data" marker run together with the empty 982 °C column. The measurement is gone
rather than guessed; UTS and elongation at 871 °C are retained, so the row still
scores on its other properties.

Yield-strength `n` therefore reads **285, not 286**, and the headline figures
moved by one decimal: FAR yield strength -18.3% to **-18.4%**, pooled yield
strength -14.7% to **-14.6%**. The full system's margin over the KG arm on
pooled yield strength is now **13.53 MPa** (92.37 to 78.84) -- exactly the
"without the row" figure this note had anticipated.

Total rows are unchanged at **471**: the erratum withdrew a cell, not a case.
Every correction is declared in `evaluation/prediction/data/errata_ledger.json`;
see `docs/data_curation.md`.

**The MCQ figure is now generated from the committed report, and guards
against drift.** `images/mcq_accuracy.png` in the manuscript was drawn by
`evaluation/chatbot/notebooks/mcq_analysis.ipynb` (cell 7), which **hard-codes**
its twelve percentages rather than reading them from anything -- so the figure
could have disagreed with the run behind it and nothing would have caught it.
`mcq_accuracy.pdf` recomputes all twelve from
`evaluation/chatbot/results/mcq_report.json` and refuses to draw if any of them
round to something other than the published value, naming the cell that moved.
All twelve currently agree, so the vector figure is the same figure: 1-Hop
100/35/33, 2-Hop 79/43/42, General 98/92/100, Overall 91/50/50, for
Chatbot + KG / Llama 3.3 70B / GPT-4o over 250 questions per system.

The one deliberate departure from the PNG is palette. The original used Paul
Tol muted; this uses Okabe--Ito to match the other figures, and draws "Overall"
in light grey with a hatch rather than a fourth hue, because it is an aggregate
of the other three bars and should not carry the visual weight of a peer.

**The first coverage bin is "<=400 degC", not "0-400".** The evaluation set
reaches -196 degC, so an axis starting at 0 named a lower edge the data does not
have. The bin now runs from -273 and the label says so; the four cryogenic rows
(two elongation, two tensile strength) join the bin they always belonged to
rather than being silently dropped. Coverage in that bin is computed over five
or more rows per property, as everywhere else, so no estimate rests on the four
alone. The per-stratum conformal table (T5) is unaffected.

**GPR on engineered features is worse than GPR on raw features, and that is
real.** It scores better in-distribution (CV MAE 105 against 125 MPa on yield
strength, holdout 97 against 142) and much worse on the evaluation set (228
against 124). Column alignment between the training and evaluation matrices was
checked and is exact; the cause is extrapolation, with 7 of 14 engineered
features putting more than 5% of evaluation rows outside the training range and
`Md_gamma` putting 55.7% outside. A stationary Gaussian-process kernel over
features that are themselves nonlinear functions of composition extrapolates
badly. It is a useful negative result: in-distribution validation badly
mispredicts out-of-distribution behaviour here.
