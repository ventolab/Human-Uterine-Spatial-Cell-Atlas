## Scripts to generate the scRNA-seq and spatial transcriptomics atlas presented in Figure 1

Builds the uterus scRNA-seq object, annotates the Xenium 480-plex H&E sections into compartments, defines the basalis-to-lumen axis, and benchmarks cell-type mapping onto Xenium. The axis and the TACCO predictions made here are reused in figures 4 and 6.

| Path | Description | Panels |
|---|---|---|
| `build_eutopic_object/00_run_scvi.py` | Takes the uterus cells from the HFRSCA v1 freeze, concatenates the datasets, then runs scVI integration (batch: dataset; covariates: donor, cell-cycle phase), UMAP and Leiden | input to 1c, 1e |
| `build_eutopic_object/01_run_tfidf.py` | TF-IDF markers per fine cell type within each lineage | Supp Table 4 |
| `build_eutopic_object/02_plot_umap_fine_ct.py` | UMAP coloured by cell type | 1c |
| `build_eutopic_object/03_barplot.ipynb` | Lineage proportions per stage, including menstrual fluid | 1e |
| `menstrual_fluid_umap/00_run_scvi.py` | scVI (batch: donor) and UMAP of the menstrual fluid FLEX data | Supplementary 2e |
| `menstrual_fluid_umap/01_umap.ipynb` | UMAP plot; dot plot of endometrial identity and decidualisation markers | Supplementary 2e, Supplementary 2f |
| `Lazyslide_automatic_annotations/create_pyramid.py` | Converts a flat H&E TIFF to a pyramidal TIFF for LazySlide | – |
| `Lazyslide_automatic_annotations/<sample>_UNI2_embedding.ipynb` | Tissue detection, tiling, UNI2 tile embeddings and Leiden clustering; exports `<sample>_uni2_tile_clusters.csv` | Supplementary 1b |
| `TissueTag_manual_annotations/TissueTag_Xenium_480_<sample>_new_axis.ipynb` | Paints the UNI2 clusters onto the registered H&E and refines them by hand in TissueTag (myometrium, basalis, functionalis, lumen). Computes distances on a 10 µm grid, the [0,1] basalis-to-lumen axis and its six bins, and writes `<sample>_annotated_new_axis.h5ad` | 1d, Supplementary 1c, Supplementary 2a; bins used in 4a |
| `benchmark_spatial_mapping/config.py` | Shared paths, sample-to-stage map and parameters | – |
| `benchmark_spatial_mapping/iss_patcher/run_benchmark.py` | ISS-Patcher (kNN) transfer of fine cell types from scRNA-seq to each Xenium section; full and downsampled reference | Supplementary 2b-c |
| `benchmark_spatial_mapping/iss_patcher/save_iss_annots.py` | Converts ISS-Patcher outputs to parquet and deletes the h5ad files | – |
| `benchmark_spatial_mapping/tacco/run_benchmark.py` | TACCO (optimal transport) transfer of fine cell types to each Xenium section | Supplementary 2b-c; reused in figures 4 and 6 |
| `benchmark_spatial_mapping/visualize/compute_gex_correlation.py` | Pearson r between scRNA-seq and spatial mean profiles of each predicted cell type, against a random baseline | Supplementary 2b-c |
| `benchmark_spatial_mapping/visualize/plot_gex_correlation.py` | Bar plots of mean r and number of cell types per method | Supplementary 2b-c |
| `benchmark_spatial_mapping/visualize/plot_celltype_spatial.py` | Spatial grid of each predicted cell type, per method and section | Supplementary Note |
| `benchmark_spatial_mapping/visualize/plot_sfrp5.py` | Mesen_GlandBas predictions (ISS-Patcher vs TACCO) next to SFRP5 expression on A66 | Supplementary Note |

Xenium 480-plex sections (one LazySlide and one TissueTag notebook per section):

| Stage | Sections |
|---|---|
| Proliferative | A13, A66, DA72_endo, DA64 |
| Secretory | A30 |
| Exogenous hormones | BZ99, DA45, DA46 |
| Menstrual | DA39, DA50, DA63 |

### Run order

1. `build_eutopic_object/00_run_scvi.py`, then `01`–`03`. The `menstrual_fluid_umap/` scripts run independently.
2. For each section: `create_pyramid.py`, then the LazySlide notebook, then the TissueTag notebook (the annotation step uses an interactive Panel widget).
3. Benchmark (needs steps 1 and 2): `iss_patcher/run_benchmark.py --config all_sc` (add `--downsample_n 1000` for the 1k run), then `save_iss_annots.py --confirm`, then `tacco/run_benchmark.py`, then `visualize/compute_gex_correlation.py`, then `plot_gex_correlation.py`.

Paths point to internal Sanger HPC. Cell-type labels come from the Human Female Reproductive System Cell Atlas v1 (Cohen et al., 2026 BioRxiv).
