"""Shared constants for the ISS-Patcher vs DOT benchmark."""
from pathlib import Path

SC_REF_PATH = "/lustre/scratch125/cellgen/vento/mm58/eutopic_endometrium/build_eutopic_object/concat_uterus_inner.h5ad"
SC_ANNOT_PATH = "/nfs/team292/projects/PanTissue/results/freeze/annotations/concatenated_annotations_postnatal_v2.csv"
CELLTYPE_COL = "fine_celltype"
MENSTRUAL_COL = "Menstrual_stage"  # derive short label: str.split(" ").str[0]
AXIS_COL = "universal_axis"  # new spatial files; legacy names (myometrial_luminal/lumina_axis) renamed on load

# Keys are sample_ids; stage is derived by stripping trailing _N (e.g. "Proliferative_1" → "Proliferative")
SPATIAL_FILES = {
    "Proliferative_1": "/nfs/team292/vl6/Endometriosis/Xenium/A13-UTR-0-TL4-1-S50/A13_annotated_new_axis.h5ad",
    "Proliferative_2": "/nfs/team292/vl6/Endometriosis/Xenium/DA72-END-0-FO-2-S2-ii/DA72_endo_annotated_new_axis.h5ad",
    "Proliferative_3": "/nfs/team292/vl6/Endometriosis/Xenium/DA64-END-0-FO-1-S2-i/DA64_annotated_new_axis.h5ad",  # no universal_axis yet
    "Proliferative_4": "/nfs/team292/vl6/Endometriosis/Xenium/A66-RPT-8-FO-1-S40/A66_annotated_new_axis.h5ad", 
    "Secretory":       "/nfs/team292/vl6/Endometriosis/Xenium/A30-UTR-2-FO-1-S48/A30_annotated_new_axis.h5ad",
    "Hormones_1":      "/nfs/team292/vl6/Endometriosis/Xenium/DA45-END-0-FO-2-S2-i/DA45_annotated_new_axis.h5ad",
    "Hormones_2":      "/nfs/team292/vl6/Endometriosis/Xenium/DA46-END-0-FO-1-S4-i/DA46_annotated_new_axis.h5ad",
    "Hormones_3":      "/nfs/team292/vl6/Endometriosis/Xenium/BZ99-END-0-FO-1-S3/BZ99_annotated_new_axis.h5ad",
    "Menstrual_1":     "/nfs/team292/vl6/Endometriosis/Xenium/DA39-END-0-FO-4-S8b/DA39_S8b_annotated_new_axis.h5ad",
    "Menstrual_2":     "/nfs/team292/vl6/Endometriosis/Xenium/DA50-END-0-FO-1-S2-i/DA50_annotated_new_axis.h5ad",
    "Menstrual_3":     "/nfs/team292/vl6/Endometriosis/Xenium/DA63-END-0-FO-1-S2-ii/DA63_annotated_new_axis.h5ad",
}

DOWNSAMPLE_N = 150

# Annotation-granularity variants for TACCO (fine_celltype/all_sc is the baseline)
ANNOT_CONFIGS = {
    "broad_celltype": {
        "annotation_key":    "broad_celltype",
        "exclude_celltypes": [],
    },
    "fine_no_oLAM": {
        "annotation_key":    "fine_celltype",
        "exclude_celltypes": ["Immune_oLAM", "Immune_uMac_Inf"],
    },
    "fine_no_mac": {
        "annotation_key":    "fine_celltype",
        "exclude_celltypes": ["Immune_oLAM", "Immune_uMac_Inf", "Immune_Mac_Transitional"],
    },
}

OUTPUT_DIR = Path("/lustre/scratch125/cellgen/vento/mm58/eutopic_endometrium/benchmark_knn_vs_dot/outputs")

# KNN parameters for ISS-Patcher
KNN_NEIGHBOURS = 30
KNN_COMPUTATION = "annoy"
