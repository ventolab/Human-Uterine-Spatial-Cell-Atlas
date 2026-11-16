## Analyses for figure 6

The basalis niche on Xenium 480-plex, and the postmenopausal validation dataset (INCLIVA).

| Path | What it does | Panels |
|---|---|---|
| `basalis_niche/plot_celltype_heatmap.py` | Heatmap of the fraction of TACCO-predicted Epi_GlandBas, Mesen_GlandBas and B cells in each compartment, per stage | 6a |
| `basalis_niche/plot_genes_xenium480.py` | Spatial expression of MS4A1, CD8A, CXCL13 and LTF on each Xenium 480-plex section | 6g, ED 6d |
| `incliva/incliva_umap.ipynb` | Re-clusters mast, B and plasma cells in the INCLIVA postmenopausal scRNA-seq data; UMAPs coloured by cell type and by donor | 6h, ED 6e |

### Run order

`plot_celltype_heatmap.py` imports `figure1/benchmark_spatial_mapping/config.py` and reads the TACCO predictions from `figure1/benchmark_spatial_mapping/tacco/run_benchmark.py`. `plot_genes_xenium480.py` reads the TissueTag outputs (`<sample>_annotated_new_axis.h5ad`) from `figure1/`. `incliva_umap.ipynb` starts from a preprocessed INCLIVA object.

Paths point to internal Sanger storage. Cell-type labels come from the HFRSCA v1 annotations (Cohen et al., 2026), which are not included here.
