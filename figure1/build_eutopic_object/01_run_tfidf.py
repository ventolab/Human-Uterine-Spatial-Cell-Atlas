
import importlib.util, sys
import pandas as pd
import scanpy as sc
spec = importlib.util.spec_from_file_location(
    'panatlas_utils',
    '/nfs/team292/projects/PanTissue/code/working/lg18/utils/panatlas_utils.py'
)
panatlas_utils = importlib.util.module_from_spec(spec)
sys.modules['panatlas_utils'] = panatlas_utils
spec.loader.exec_module(panatlas_utils)

from panatlas_utils import quick_markers

def get_markers(adata, groupby, n_top=25):
    # Downsample: max 150 cells per lineage_bin
    cells_down = (
        adata.obs
        .groupby(groupby, observed=True)
        .apply(lambda g: g.sample(min(len(g), 150), random_state=0), include_groups=False)
        .index.get_level_values(-1)
    )
    adata_down = adata[cells_down].copy()

    # Remove deprecated genes
    adata_down = adata_down[:, ~adata_down.var_names.str.contains('DEPRECATED')].copy()
    
    # TF-IDF markers per lineage_bin
    tfidf_markers = quick_markers(
        adata_down,
        cluster_key = groupby,
        n_markers   = n_top,
        fdr         = 0.01,
        express_cut = 0.9,
    )
    
    return tfidf_markers

adata = sc.read_h5ad('/lustre/scratch125/cellgen/vento/mm58/eutopic_endometrium/build_eutopic_object/integrated_scvi_uterus.h5ad')

all_markers = {}

for lineage in adata.obs['lineage'].unique():
    print(f'Processing lineage: {lineage}')
    
    if lineage in ['mesothelial', 'neural']:
        continue
    
    adata_sub = adata[adata.obs['lineage'] == lineage].copy()
    markers = get_markers(adata_sub, groupby='fine_celltype')
    
    all_markers[lineage] = markers



markers_lineage = get_markers(adata, groupby='lineage')
all_markers['by_lineage'] = markers_lineage
        
# save as excel file with one sheet per lineage
with pd.ExcelWriter('tfidf_markers.xlsx') as writer:
    for lineage, markers in all_markers.items():
        markers.to_excel(writer, sheet_name=lineage[:31], index=False)



