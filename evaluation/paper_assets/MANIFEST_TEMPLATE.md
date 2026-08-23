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
| System architecture | `docs/paper/figures/fig1_system_architecture_big.drawio` (`alloygraph-arch-v10`) | `docs/paper/figures/system_architecture.png` |
| HAYNES 230 knowledge graph | **none — screenshot, see below** | `docs/paper/figures/kg.png` |
| Prediction / design pipeline | `docs/paper/figures/fig3_evaluation_pipeline.drawio` (`pipeline-v4`) | `docs/paper/figures/agent_pipeline.png` |

Both sources were checked label-for-label against the rendered figure and are
current. The alloy counts are the quick tell: the source and the paper figure
both read "88 Ni superalloys", and any export reading **99** is stale.

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

**These could not be exported here.** Neither the draw.io desktop application
nor its CLI is installed on this machine, and no scriptable export path exists
without one. Rendering the XML through a third-party library would silently
change the layout the diagrams were drawn with, which is worse than not
exporting them.

To export, either use the desktop app (File > Export as > PDF, with *Crop* and
*Transparent background*), or install the CLI once:

```bash
npm install -g @drawio/export
drawio --export --format pdf --crop --transparent \
  --output paper_assets/figures/fig1_system_architecture.pdf \
  docs/paper/figures/fig1_system_architecture_big.drawio
```

Export as PDF rather than PNG: KBS R2.1 asks for vector artwork.

## Notes that belong with the numbers

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

**The sub-zero temperature bin is excluded from the coverage figure.** It holds
two cryogenic rows for two properties, too few to estimate coverage from. The
per-stratum conformal table (T5) is unaffected.

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
