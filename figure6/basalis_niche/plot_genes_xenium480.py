import scanpy as sc
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42

genes_plot = ['MS4A1', 'CD8A', 'CXCL13', 'LTF']

donor_paths = {
    'A66':  '/nfs/team292/vl6/Endometriosis/Xenium/A66-RPT-8-FO-1-S40/A66_annotated_new_axis.h5ad',
    'A13':  '/nfs/team292/vl6/Endometriosis/Xenium/A13-UTR-0-TL4-1-S50/A13_annotated_new_axis.h5ad',
    'DA72': '/nfs/team292/vl6/Endometriosis/Xenium/DA72-END-0-FO-2-S2-ii/DA72_endo_annotated_new_axis.h5ad',
    'A30':  '/nfs/team292/vl6/Endometriosis/Xenium/A30-UTR-2-FO-1-S48/A30_annotated_new_axis.h5ad',
    'DA50': '/nfs/team292/vl6/Endometriosis/Xenium/DA50-END-0-FO-1-S2-i/DA50_annotated_new_axis.h5ad',
    'DA39': '/nfs/team292/vl6/Endometriosis/Xenium/DA39-END-0-FO-4-S8b/DA39_S8b_annotated_new_axis.h5ad',
    'DA63': '/nfs/team292/vl6/Endometriosis/Xenium/DA63-END-0-FO-1-S2-ii/DA63_annotated_new_axis.h5ad',
    'DA45': '/nfs/team292/vl6/Endometriosis/Xenium/DA45-END-0-FO-2-S2-i/DA45_annotated_new_axis.h5ad',
    'DA46': '/nfs/team292/vl6/Endometriosis/Xenium/DA46-END-0-FO-1-S4-i/DA46_annotated_new_axis.h5ad',
    'BZ99': '/nfs/team292/vl6/Endometriosis/Xenium/BZ99-END-0-FO-1-S3/BZ99_annotated_new_axis.h5ad',
}


def plot_genes_spatial(adata, donor, genes, out_dir='figures'):
    fig = sc.pl.embedding(
        adata,
        basis='spatial',
        color=[g for g in genes if g in adata.var_names],
        cmap='Reds',
        vmin=0,
        size=5,
        show=False,
        return_fig=True,
    )
    for ax in fig.axes:
        for coll in ax.collections:
            coll.set_rasterized(True)
    fig.savefig(f'{out_dir}/LA_{donor.lower()}.pdf', bbox_inches='tight', dpi=600)


for donor, path in donor_paths.items():
    adata = sc.read_h5ad(path)
    plot_genes_spatial(adata, donor, genes_plot)
