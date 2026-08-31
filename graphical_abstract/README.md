# Graphical abstract

Submission asset for *Advanced Engineering Informatics* (Elsevier). Regenerate with:

```bash
python graphical_abstract/make_graphical_abstract.py          # 600 dpi rasters
python graphical_abstract/make_graphical_abstract.py --dpi 300
```

| file | role |
|---|---|
| `alloygraph_graphical_abstract.pdf` | primary deliverable, vector, text as embedded TrueType |
| `alloygraph_graphical_abstract.tiff` | 600 dpi, LZW, 3071 x 1228 px |
| `alloygraph_graphical_abstract.png` | on-screen preview |
| `make_graphical_abstract.py` | the generator; the figure has no other source |

## Journal spec

| requirement | this asset |
|---|---|
| landscape, 2.5:1 | 13.000 x 5.200 cm, ratio 2.50000 |
| at least 1328 x 531 px | 3071 x 1228 px at 600 dpi (1535 x 614 at 300) |
| readable at 13 x 5 cm print | drawn 1:1 at 13 cm, so no rescaling; body type floor 7.0 pt |

The figure is drawn at its final printed size rather than large-and-shrunk, so
the point sizes in the script are the point sizes on paper. `check_fit` measures
every label against the box that owns it with the real renderer and reports any
overflow, so a wording change that no longer fits fails loudly rather than
silently colliding; the script prints `fit: all 40 labels inside their
containers` on a clean run.

One exception to the 7 pt floor: the grey `evaluated by compositional novelty`
caption under the evaluation heading is 6.0 pt (`FS_TINY`), deliberately set
below body size so it reads as a subordinate gloss on the panel rather than as
content. It is the only element below 7 pt.

Palette and type follow the manuscript's existing figures — the pale band /
saturated stroke scheme of `docs/paper/figures/fig1_system_architecture_big.drawio`
and the Okabe-Ito NEAR/MID/FAR colours of `evaluation/paper_assets/make_figures.py`,
so a stratum is the same colour here as in the results figures.

## Provenance of the numbers

All from committed results, not restated from memory:

| claim | source | value |
|---|---|---|
| NEAR, KG anchoring, -30% YS error, known alloys | `evaluation/prediction/results/nn_distance_stratified_results.md` | MAE 107.56 -> 75.42 MPa over ML+physics (-29.9%) |
| FAR, agent triangulation, -18% YS error | `evaluation/prediction/results/analyst_ablation.md` | MAE 110.10 -> 89.83 MPa over ML+physics+KG (-18.4%, five-seed mean, sigma 1.32) |
| physics: elastic modulus, -36% | `nn_distance_stratified_results.md`, ALL row | MAE 12.86 -> 8.21 GPa over ML-only (-36.2%, n = 304) |
| 13 of 88 vendor-renamed near-duplicates | same, "The NEAR alloys" table | 13 NEAR alloys, d < 2.0 |

The modulus row quotes the **pooled** figure the paper text uses, not a stratum
row. It is deliberately not stratified: physics helps NEAR (-54%) and FAR
(-41%) but makes MID slightly worse (9.92 -> 10.48 GPa), so a per-stratum claim
would need a caveat the panel has no room for.

"106 alloy records" and "near-duplicates" are the manuscript's own wording --
106 records cover 100 distinct alloys, and the NEAR overlap is near-duplication,
not identity.
