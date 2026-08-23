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
at 8 pt base type. Palette is Okabe--Ito, distinguishable under deuteranopia,
protanopia and tritanopia, and separable in greyscale.

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
| HAYNES 230 knowledge graph | **none — screenshot, see below** | `paper_src/.../images/kg.png` (untracked) |
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

Because `paper_src/` is git-ignored, `kg.png` currently exists in no commit.
Fixing that means either tracking it beside the other two figures or accepting
that figure 2 is unrecoverable if the working copy is lost.

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
