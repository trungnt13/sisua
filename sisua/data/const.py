from collections import OrderedDict

UNIVERSAL_RANDOM_SEED = 5218
DROPOUT_TEST = 0.4

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
