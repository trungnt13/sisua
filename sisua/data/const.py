from collections import OrderedDict

UNIVERSAL_RANDOM_SEED = 5218

# Choose 3D or 2D visualization here
TSNE_DIM = 2

# This contains pair of protein markers that
# should not appear together, hence, good for
# inspecting the biological plausibility of the model
PROTEIN_PAIR_COMPARISON = [
    ('CD8', 'CD4'),
    ('CD56', 'NKT'),
    ('CD45RA', 'CD45RO'),
    ('TIGIT', 'CD127'),
]


# Mapping from protein to gene expression
# CCR5
# CCR7
MARKER_GENES = {
    ### For PBMC CITEseq
    'CD14': 'CD14',
    'CD15': 'FUT4',
    'CD16': 'FCGR3A',
    'CD11c': 'ITGAX',
    'CD127': 'IL7R',
    'CD19': 'CD19',
    'CD2': 'CD2',
    'CD25': 'IL2RA',
    'CD3': 'CD3G',
    'CD4': 'CD4',
    'CD45RA': 'PTPRC',
    'CD45RO': 'PTPRC',
    'CD56': 'NCAM1',
    'CD57': 'B3GAT1',
    'CD8': 'CD8A',
    'CD8a': 'CD8A',
    'PD-1': 'PDCD1',
    'TIGIT': 'TIGIT',
    ### For FACS_5
    'CD20': 'MS4A1',
    'CD45': 'PTPRC',
    'CD34': 'CD34',
    'CD10': 'MME',
    # CD19 is above
    ### For FACS_7
    # CD10 is above
    'CD135': 'FLT3',
    # CD34 is above
    'CD38': 'CD38',
    # CD45RA is above
    'CD49F': 'ITGA6',
    'CD90': 'THY1',
}

# ['CD77' 'CCR7' 'CD19' 'CD1a' 'CD2' 'CD27' 'PD-L1;CD274' 'CD28'
#  'PECAM;CD31' 'CD34' 'CD3' 'CD4' 'CD44' 'CD5' 'CD69' 'CD7' 'CD8' 'CD66b'
#  'CTLA4' 'CD26;Adenosine' 'CD16' 'CD366;tim3' 'HLA-A' 'MHCII;HLA-DR'
#  'IL7Ralpha;CD127' 'CD11b' 'CD11c' 'LAMP1' 'CD56' 'PD-1;CD279' 'PD1;CD279'
#  'B220;CD45R' 'CD45RA' 'CD45RO' 'CD138' 'CD62L' 'Siglec-8' 'Ox40;CD134']
