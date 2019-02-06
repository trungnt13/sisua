from collections import OrderedDict

UNIVERSAL_RANDOM_SEED = 5218

# from protein_marker -> HUMAN_symbol
PBMC_markers_to_symbols = OrderedDict([
    ('CD3', 'CD3G'),
    ('CD4', 'CD4'),
    ('CD8', 'CD8A'),
    ('CD2', 'CD2'),
    ('CD45RA', 'PTPRC'),
    ('CD57', 'B3GAT1'),
    ('CD16', 'FCGR3A'),
    ('CD14', 'CD14'),
    ('CD11c', 'ITGAX'),
    ('CD19', 'CD19')
])

PBMC_colors = ['blue', 'lightblue', 'red', 'purple', 'green',
               'sienna', 'yellow', 'magenta', 'limegreen', 'orange']
PBMC_markers = ['.', 'P', "*", '|', "X",
                "s", "^", "+", "x", "_"]
# Choose 3D or 2D visualization here
TSNE_DIM = 2

MARKER_GENES = dict(
    ### For PBMC CITEseq
    CD3='CD3G',
    CD4='CD4',
    CD8='CD8A',
    CD2='CD2',
    CD45RA ='PTPRC',
    CD57 ='B3GAT1',
    CD16 ='FCGR3A',
    CD14 ='CD14',
    CD11c ='ITGAX',
    CD19 ='CD19',
    ### For FACS_5
    CD20='MS4A1',
    CD45='PTPRC',
    CD34='CD34',
    CD10='MME',
    # CD19 is above
    ### For FACS_7
    # CD10 is above
    CD135 ='FLT3',
    # CD34 is above
    CD38 ='CD38',
    # CD45RA is above
    CD49F ='ITGA6',
    CD90 ='THY1',
)
