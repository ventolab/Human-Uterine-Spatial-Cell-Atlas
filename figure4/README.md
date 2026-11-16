## Analyses for figure 4

Transfers the basalis-to-lumen axis bins from Xenium 480-plex onto stage-matched epithelial and stromal scRNA-seq cells, then models gene expression along the axis.

| Path | What it does | Panels |
|---|---|---|
| `mapping_epithelium/01_run_tacco_ensembl.py` | TACCO run once per Xenium section (Xenium epithelial cells as reference, scRNA-seq epithelial cells of the same stage as query); scores are averaged across sections to give one axis bin per cell | 4a |
| `mapping_epithelium/01_run_tacco.py` | Same transfer, with all sections of a stage concatenated into one reference | ED 4a-c |
| `mapping_epithelium/01_run_xenium_mapping.py`, `utils.py` | Same transfer with ISS-Patcher (`--mode unique\|concat\|ensemble`) | ED 4a-c |
| `mapping_epithelium/02_prepare_for_gradients_modelling_tacco.py` | Pseudobulk counts per donor × axis bin | – |
| `mapping_epithelium/03_run_glm_tacco.py` | Per-gene mixed models on log2-CPM: axis slope (Test A, Wald) and axis × phase interaction (Test B, LRT), BH-corrected; proliferative and secretory only | Supp Table 6 |
| `mapping_epithelium/04_visualize_trend_tacco_multisample_ensemble.ipynb` | Spline plots of pseudobulk trends along the axis | exploratory |
| `mapping_mesenchyme/01_run_tacco_ensembl.py`, `01_run_tacco.py`, `01_run_xenium_mapping.py`, `utils.py` | As above, for mesenchymal cells | 4a; ED 4a-c |
| `mapping_mesenchyme/axis_stromal/02_prepare_for_gradients_modelling_tacco.py` | Pseudobulk for stromal fibroblasts | – |
| `mapping_mesenchyme/axis_stromal/03_run_glm_tacco.py` | Mixed models as above; menstrual, proliferative and secretory | Supp Table 6 |
| `mapping_mesenchyme/axis_stromal/04_visualize_trend_tacco_multisample_ensemble.ipynb` | Spline plots of HSD11B1, HSD11B2 and OGN | 4e, 4f |
| `mapping_mesenchyme/compare_methods_gex_correlation.py` | Correlation between scRNA-seq and Xenium mean profiles per axis bin, for TACCO and ISS-Patcher (concat and ensemble), in both lineages | ED 4a-c |
| `mapping_mesenchyme/visualize_lineage_bins.ipynb` | Dot plots of hedgehog pathway genes in stroma and IHH in epithelium, by axis bin and stage | 4c |
| `mapping_mesenchyme/04_visualize_trend_xenium.ipynb` | Spline plots from the Xenium mesenchymal pseudobulk: GLI1/PTCH1 and SFRP5 | ED 4d, 6c |
| `mapping_mesenchyme/plot_secretory_fib.py` | TACCO-predicted early, mid and late secretory fibroblasts on A30 | 4g |

### Run order

Needs, from `figure1/`, the TissueTag outputs (`<sample>_annotated_new_axis.h5ad`) and the TACCO cell-type predictions from `benchmark_spatial_mapping/tacco/run_benchmark.py`. These are used to select epithelial and mesenchymal cells on Xenium.

1. For each lineage: `01_run_tacco_ensembl.py`, then `02`, then `03`, then the `04` notebook. For mesenchyme, run `02` to `04` from `axis_stromal/`; they read `../tacco_annotation_ensembl`.
2. `visualize_lineage_bins.ipynb` needs the `tacco_annotation_ensembl/` outputs of both lineages.
3. `compare_methods_gex_correlation.py` needs both lineages' outputs from `01_run_tacco.py`, `01_run_tacco_ensembl.py` and `01_run_xenium_mapping.py --mode concat` and `--mode ensemble`.

Paths point to internal Sanger storage. Cell-type labels come from the HFRSCA v1 annotations (Cohen et al., 2026).