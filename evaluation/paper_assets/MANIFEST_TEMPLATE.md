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

Four draw.io sources exist and are uncompressed XML:

```
docs/paper/figures/fig1_system_architecture.drawio
docs/paper/figures/fig1_system_architecture_big.drawio
docs/paper/figures/fig2_agent_workflow.drawio
docs/paper/figures/fig3_evaluation_pipeline.drawio
```

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
  docs/paper/figures/fig1_system_architecture.drawio
```

Export as PDF rather than PNG: KBS R2.1 asks for vector artwork.

## Notes that belong with the numbers

**Parity MAE annotations are read from `stratified_metrics.csv`, not recomputed.**
The plotted points are the row-wise mean prediction over the five campaign
seeds; averaging predictions first cancels independent seed noise, so those
points give a slightly lower error (78.6 MPa on yield strength) than the mean
of the per-seed errors the tables report (80.9 MPa). The tables' definition is
the one quoted on the figure, so figure and table agree.

**One measured value in the evaluation set is not physical.** RGT* 13 at 871 °C
carries a yield strength of **-435 MPa**. It inflates every arm's yield-strength
MAE by 2.0--2.5 MPa and affects them near-equally, so relative comparisons are
untouched: the full system's margin over the KG arm is 13.94 MPa with the row
and 13.53 MPa without it. It has **not** been corrected, because doing so would
change every committed number after the results were frozen. The parity panels
bound their axes robustly and count the off-scale points rather than hiding
them. This is a second data defect of the same kind as the NIMONIC PE16 erratum
and should be decided on before submission.

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
