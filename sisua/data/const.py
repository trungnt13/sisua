from collections import OrderedDict
from typing import List, Text, Tuple

from six import string_types

from odin.utils.ordered_flag import OrderedFlag

UNIVERSAL_RANDOM_SEED = 5218

# Choose 3D or 2D visualization here
TSNE_DIM = 2

# This contains representative pairs of protein markers
# 100 pairs
PROTEIN_PAIR_POSITIVE = [
    ('CD3', 'CD4'), ('CD14', 'CD4'), ('CD19', 'CD45RA'), ('CD14', 'CD19'),
    ('CD3', 'CD8'), ('IgG1', 'IgG2a'), ('IgG2a', 'IgG2b'), ('IgG1', 'IgG2b'),
    ('CD45RO', 'PD-1'), ('CD14', 'IgG2b'), ('CD19', 'IgG2a'), ('CD14', 'IgG2a'),
    ('CD19', 'IgG1'), ('CD19', 'IgG2b'), ('CD14', 'CD8'), ('CD14', 'IgG1'),
    ('CD4', 'IgG2a'), ('CCR7', 'CD19'), ('CD4', 'IgG1'), ('CCR7', 'CD4'),
    ('CD4', 'IgG2b'), ('IgG1', 'PD-1'), ('CD16', 'CD56'), ('CCR7', 'CD14'),
    ('IgG2a', 'PD-1'), ('CD14', 'PD-1'), ('CD4', 'PD-1'), ('CD19', 'PD-1'),
    ('CCR7', 'IgG2a'), ('CCR7', 'CD45RA'), ('IgG2b', 'PD-1'),
    ('CD16', 'CD45RA'), ('CD45RA', 'CD56'), ('CD14', 'CD3'), ('CCR7', 'IgG1'),
    ('CD11c', 'CD14'), ('CCR7', 'IgG2b'), ('CCR7', 'CD3'), ('CD19', 'CD4'),
    ('CD45RO', 'IgG1'), ('CD16', 'CD19'), ('CD19', 'CD8'), ('CD14', 'CD45RO'),
    ('CD45RA', 'CD8'), ('CD127', 'CD3'), ('CD45RA', 'IgG2a'), ('CD8', 'PD-1'),
    ('CD4', 'CD45RO'), ('CD127', 'CD4'), ('CD8', 'IgG2a'), ('CD8', 'IgG1'),
    ('CD45RO', 'CD8'), ('CD11c', 'CD16'), ('CD45RA', 'IgG2b'), ('CD3', 'IgG2a'),
    ('CD14', 'HLA-DR'), ('HLA-DR', 'IgG1'), ('HLA-DR', 'PD-1'), ('CD3', 'IgG1'),
    ('CCR7', 'HLA-DR'), ('CD8', 'HLA-DR'), ('CD19', 'HLA-DR'), ('CD19', 'CD56'),
    ('HLA-DR', 'IgG2a'), ('CD3', 'CD45RO'), ('CCR7', 'CD8'), ('CD8', 'IgG2b'),
    ('CD3', 'PD-1'), ('CD3', 'IgG2b'), ('CD10', 'CD34'), ('CD45RO', 'HLA-DR'),
    ('CD14', 'CD16'), ('HLA-DR', 'IgG2b'), ('CD2', 'CD3'), ('CCR7', 'PD-1'),
    ('CD4', 'HLA-DR'), ('CD25', 'CD45RO'), ('CD25', 'PD-1'), ('CD56', 'TIGIT'),
    ('CCR7', 'CD34'), ('CD16', 'TIGIT'), ('CD45RA', 'IgG1'), ('CD127', 'CD28'),
    ('CD16', 'CD34'), ('CCR7', 'CD10'), ('CCR5', 'CD10'), ('CD28', 'CD4'),
    ('CD27', 'CD28'), ('CCR7', 'CD16'), ('CD34', 'CD56'), ('CD11c', 'CD56'),
    ('PD-1', 'TIGIT'), ('CCR5', 'CCR7'), ('CD3', 'CD45RA'), ('CD11c', 'CD34'),
    ('CD3', 'HLA-DR'), ('CD15', 'IgG2a'), ('CD11c', 'CD8'), ('CD14', 'CD25'),
    ('CD2', 'CD8')
]
PROTEIN_PAIR_NEGATIVE = [
    ('CD45RA', 'CD45RO'), ('CD3', 'CD56'), ('CD16', 'CD3'), ('CD4', 'CD56'),
    ('CD127', 'CD45RA'), ('CD45RA', 'PD-1'), ('CD19', 'CD2'), ('CD127', 'CD56'),
    ('CD11b', 'CD27'), ('CD11c', 'CD3'), ('CD11b', 'CD28'), ('CD127', 'CD16'),
    ('CD16', 'CD4'), ('CD4', 'CD45RA'), ('CD127', 'TIGIT'), ('CD11b', 'CD127'),
    ('CD11c', 'CD2'), ('CD2', 'CD27'), ('CD28', 'CD8'), ('CD2', 'CD45RA'),
    ('CD11b', 'CD4'), ('CD127', 'CD14'), ('CD3', 'TIGIT'), ('CD28', 'CD45RA'),
    ('CD127', 'CD19'), ('CD4', 'CD8'), ('CD27', 'CD8'), ('CD11b', 'CD3'),
    ('CD11b', 'CD5'), ('CD2', 'CD62L'), ('CD2', 'CD31'), ('CD2', 'PD-1;CD279'),
    ('CD2', 'CD69'), ('CD2', 'MHCII'), ('CD5', 'CD56'), ('CD25', 'CD45RA'),
    ('CD16', 'CD2'), ('CD4', 'TIGIT'), ('CCR7', 'CD2'), ('CD45RA', 'CD5'),
    ('CD2', 'CD77'), ('CD26', 'CD8'), ('CD5', 'HLA-A'), ('CD2', 'HLA-A'),
    ('CD44', 'CD45RA'), ('CD5', 'CD7'), ('CD31', 'CD5'), ('CD10', 'CD45'),
    ('CD31', 'CD44'), ('CD5', 'CD8'), ('CD34', 'CD45'), ('CD31', 'CD4'),
    ('CD5', 'CD77'), ('CD27', 'CD56'), ('CD11b', 'CD26'), ('CD11b', 'CD44'),
    ('CD27', 'HLA-A'), ('CD8', 'PD-1;CD279'), ('CD38', 'CD90'),
    ('CD7', 'MHCII'), ('CD366', 'CD5'), ('CD278', 'HLA-DR'), ('CD11b', 'CD278'),
    ('CD366', 'CD44'), ('CD2', 'CD66b'), ('CD127', 'HLA-DR'), ('CD34', 'CD4'),
    ('CD28', 'HLA-DR'), ('CD27', 'HLA-DR'), ('CD3', 'CD69'), ('CD3', 'CD366'),
    ('CD8', 'PD1;CD279'), ('CD44', 'CD7'), ('CD278', 'CD86'), ('CD19', 'CD5'),
    ('CD27', 'CD45RA'), ('CD44', 'CD77'), ('CD62L', 'CD8'), ('CD27', 'MHCII'),
    ('CD2', 'CD28'), ('CD3', 'CD86'), ('CD2', 'CD366'), ('CD44', 'CD56'),
    ('CD26', 'CD45RA'), ('CD127', 'MHCII'), ('CD5', 'MHCII'), ('CD16', 'CD27'),
    ('CD3', 'CD34'), ('CD127', 'CD86'), ('CD16', 'CD5'), ('CD28', 'CD86'),
    ('CD27', 'CD86'), ('CD28', 'CD56'), ('CD2', 'LAMP1'), ('CD14', 'CD27'),
    ('CD127', 'CD2'), ('CD14', 'CD278'), ('CCR7', 'CD44'), ('CD16', 'CD44'),
    ('CD2', 'CD34')
]

# Mapping from protein to gene expression
# CCR5
# CCR7
MARKER_ADT_GENE = {
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

MARKER_ADTS = list(MARKER_ADT_GENE.keys())
MARKER_GENES = list(set(list(MARKER_ADT_GENE.values()) + \
  ['CD8B', 'CD79A', 'LYZ', 'LGALS3', 'S100A8', 'GNLY',
   'KLRB1', 'FCER1A', 'CST3', "MS4A1", "CD19", "MME", "VPREB1", "VPREB3",
   "DNTT", "CD79A", "MZB1", "NKG7", "CD3D", "CD34", "CST3", "LYZ",
   "HBA1", "FCGR3A", "GATA1", "GATA2"]))

# ['CD77' 'CCR7' 'CD19' 'CD1a' 'CD2' 'CD27' 'PD-L1;CD274' 'CD28'
#  'PECAM;CD31' 'CD34' 'CD3' 'CD4' 'CD44' 'CD5' 'CD69' 'CD7' 'CD8' 'CD66b'
#  'CTLA4' 'CD26;Adenosine' 'CD16' 'CD366;tim3' 'HLA-A' 'MHCII;HLA-DR'
#  'IL7Ralpha;CD127' 'CD11b' 'CD11c' 'LAMP1' 'CD56' 'PD-1;CD279' 'PD1;CD279'
#  'B220;CD45R' 'CD45RA' 'CD45RO' 'CD138' 'CD62L' 'Siglec-8' 'Ox40;CD134']
MARKER_ATAC = {
    'GZMK classic promoter': 'chr13:113180223:113181928',
    'GZMK alternative promoter': 'chr13:113182148:113184892',
    'CD68 promoter': 'chr11:69665600:69667000',
    'CD3D promoter': 'chr9:44981200:44982800',
    'CD19 promoter': 'chr7:126414200:126415200',
    'NCR1 promoter': 'chr7:4337400:4337800',
}

_MARKER_ATAC_URL = [
    r"https://www.encodeproject.org/files/ENCFF108APF/@@download/ENCFF108APF.bed.gz"
]

#   # other data types (matrix or dataframe)
#   else:
#     data = read_r_matrix(data)
#     data.to_pickle(outpath)
# # tsv files
# elif '.tsv.gz' in path.lower():
#   data = pd.read_csv(path, compression='gzip', header=0, sep='\t')
#   data.to_pickle(outpath)

# peak2gene=
# (r"https://jeffgranja.s3.amazonaws.com/MPAL-10x/Supplementary_Data/Integration/MPAL-Significant-Peak2Gene-Links.tsv.gz",
#  r"8258d7eb424f5d977bd849ba0cc37c6f"),
# rna2atac=
# (r"https://jeffgranja.s3.amazonaws.com/MPAL-10x/Supplementary_Data/Integration/scATAC-scRNA-mappings.rds",
#  r"c41bddd79cda449058ad46cd53064ac5"),


# ===========================================================================
# Omic Enum
# ===========================================================================
class OMIC(OrderedFlag):
  r""" Enum class to represent all possible OMIC type """

  genomic = 'genomic'
  chromatin = 'chromatin'
  transcriptomic = 'transcriptomic'
  proteomic = 'proteomic'
  celltype = 'celltype'
  disease = 'disease'
  progenitor = 'progenitor'
  pmhc = 'pmhc'  # peptide - major histocompatibility complex
  # reconstructed
  ochromatin = 'ochromatin'
  otranscriptomic = 'otranscriptomic'
  # imputed
  igenomic = 'igenomic'
  ichromatin = 'ichromatin'
  itranscriptomic = 'itranscriptomic'
  iproteomic = 'iproteomic'
  icelltype = 'icelltype'
  idisease = 'idisease'
  iprogenitor = 'iprogenitor'
  ipmhc = 'ipmhc'
  #
  epigenomic = 'epigenomic'
  metabolomic = 'metabolomic'
  microbiomic = 'microbiomic'
  #
  latent = 'latent'

  @property
  def is_imputed(self):
    r""" Return True if the OMIC type is imputation """
    if self in (OMIC.igenomic, OMIC.ichromatin, OMIC.itranscriptomic,
                OMIC.iproteomic, OMIC.icelltype, OMIC.idisease,
                OMIC.iprogenitor, OMIC.ipmhc):
      return True
    return False

  @property
  def markers(self) -> List[Text]:
    r""" Return list of all marker variables' name """
    name = self.name
    if name in (OMIC.proteomic.name, OMIC.iproteomic.name):
      return list(MARKER_ADTS)
    if name in (OMIC.transcriptomic.name, OMIC.itranscriptomic.name):
      return list(MARKER_GENES)
    if name in (OMIC.chromatin.name, OMIC.ichromatin.name):
      return list(MARKER_ATAC)
    return None

  def marker_pairs(self, omic) -> List[Tuple[Text]]:
    name1 = self.name
    name2 = OMIC.parse(omic).name
    if name1 in (OMIC.transcriptomic.name, OMIC.itranscriptomic.name) and \
      name2 in (OMIC.proteomic.name, OMIC.iproteomic.name):
      return [(j, i) for i, j in MARKER_ADT_GENE.items()]
    if name1 in (OMIC.proteomic.name, OMIC.iproteomic.name) and \
      name2 in (OMIC.transcriptomic.name, OMIC.itranscriptomic.name):
      return [(i, j) for i, j in MARKER_ADT_GENE.items()]
    return None

  @classmethod
  def is_omic_type(cls, o):
    o = str(o).lower().strip()
    all_omics = [i.name for i in list(cls)]
    return o in all_omics

  @classmethod
  def _sep(cls):
    return '_'
