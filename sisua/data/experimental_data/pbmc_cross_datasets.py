import os
import pickle
import shutil
from functools import partial, reduce

import numpy as np
import scanpy as sc

from odin.utils import md5_folder
from sisua.data.const import MARKER_GENES
from sisua.data.data_loader.cbmc_CITEseq import read_CITEseq_CBMC
from sisua.data.data_loader.childhood_leukemia_cALL import read_leukemia_BMMC
from sisua.data.data_loader.dataset10x import read_dataset10x
from sisua.data.data_loader.mixed_phenotype_acute_leukemia import \
    read_leukemia_MixedPhenotypes
from sisua.data.data_loader.pbmc8k import read_PBMC8k
from sisua.data.data_loader.pbmc_CITEseq import read_CITEseq_PBMC
from sisua.data.data_loader.pbmcecc import read_PBMCeec
from sisua.data.path import DATA_DIR, EXP_DIR
from sisua.data.single_cell_dataset import SingleCellOMIC
from sisua.data.utils import standardize_protein_name

# ===========================================================================
# Helpers
# ===========================================================================
_DATASETS = dict(
    # PBMC8k is the "pbmc_10k_protein_v3"
    pbmc8k=partial(read_PBMC8k, subset='full', filtered_genes=False),
    pbmcecc=partial(read_PBMCeec, subset='ly', filtered_genes=False),
    pbmcciteseq=partial(read_CITEseq_PBMC, filtered_genes=False),
    cbmcciteseq=partial(read_CITEseq_CBMC, filtered_genes=False),
    call=partial(read_leukemia_BMMC, filtered_genes=False),
    mpal=partial(read_leukemia_MixedPhenotypes,
                 filtered_genes=False,
                 omic='rna'),
    pbmc5k=partial(read_dataset10x,
                   name='5k_pbmc_protein_v3',
                   filtered_genes=False),
    vdj1=partial(read_dataset10x,
                 name='vdj_v1_hs_aggregated_donor1',
                 filtered_genes=False),
    vdj4=partial(read_dataset10x,
                 name='vdj_v1_hs_aggregated_donor4',
                 filtered_genes=False),
)

_MD5 = r"a46190b62f77e08799d93af8c4742237"


def _match_genes(sco: SingleCellOMIC, gene_names: dict):
  var_names = {
      name: i for i, name in enumerate(sco.get_var_names('transcriptomic'))
  }
  ids = [var_names[name] for name in gene_names]
  X = sco.get_omic("transcriptomic")[:, ids]
  return X, ids


# ===========================================================================
# Main
# ===========================================================================
def read_PBMC_crossdataset(name,
                           filtered_genes=True,
                           override=False,
                           verbose=True) -> SingleCellOMIC:
  r""" This create a dataset with shared genes among multiple datasets

    - 'pbmc8k' (6290, 17870)->(6290, 11299) genes
    - 'pbmcecc' (2941, 15634)->(2941, 11299) genes
    - 'pbmcciteseq' (7985, 17006)->(7985, 11299) genes
    - 'cbmcciteseq' (8617, 20400)->(8617, 11299) genes
    - 'call' (37552, 33694)->(37552, 11299) genes
    - 'mpal' (52396, 20287)->(52396, 11299) genes
    - 'pbmc5k' (5247, 33538)->(5247, 11299) genes
    - 'vdj1' (55206, 33538)->(55206, 11299) genes
    - 'vdj4' (36619, 33538)->(36619, 11299) genes

  Total transcriptomic data: 212853(cells) 11299(genes)

  Highly variable genes: 2000

  Arguments:
    name : {'pbmc8k', 'pbmcecc', 'call', 'mpal', 'pbmc5k', 'vdj1', 'vdj4'}
  """
  assert name in _DATASETS, \
    (f"Invalid dataset name='{name}', "
     f"available datasets are: {list(_DATASETS.keys())}")
  preprocessed_path = os.path.join(DATA_DIR, 'PBMC_crossdataset_preprocessed')
  if override and os.path.exists(preprocessed_path):
    shutil.rmtree(preprocessed_path)
    if verbose:
      print(f"Override preprocessed data at path {preprocessed_path}")
  if not os.path.exists(preprocessed_path):
    os.mkdir(preprocessed_path)
  # ******************** preprocessing ******************** #
  if len(os.listdir(preprocessed_path)) == 0 or \
    md5_folder(preprocessed_path) != _MD5:
    datasets = {}
    for i, j in _DATASETS.items():
      ds = j(verbose=verbose)
      datasets[i] = ds
      if verbose:
        print(f"Read dataset='{i}' shape={ds.shape}")
    gene_names = sorted(
        reduce(lambda x, y: x & y,
               (set(i.var_names.values) for i in datasets.values())))
    # this make sure the gene order is random and consistent among all machines
    rand = np.random.RandomState(seed=1)
    rand.shuffle(gene_names)
    # some debugging
    if verbose:
      omics = reduce(lambda x, y: x | y, (i.omics for i in datasets.values()))
      n_samples = {k: v.shape[0] for k, v in datasets.items()}
      print(f"Select {len(gene_names)} common genes "
            f"among {', '.join(datasets.keys())}.")
      print(f"All available OMICs are: {omics}")
      print(f"Amount of samples: {n_samples}")
    # read data from all available OMICs
    indices = {}
    mRNA = []
    for name, sco in datasets.items():
      X, ids = _match_genes(sco, gene_names)
      indices[name] = ids
      mRNA.append(X)
      if verbose:
        print(f"Matching genes for dataset '{name}' "
              f"{sco.X.shape}->{X.shape} genes")
    mRNA = np.concatenate(mRNA, axis=0)
    if verbose:
      print("Total transcriptomic data:",
            f"{mRNA.shape[0]}(cells) {mRNA.shape[1]}(genes)")
    # filter genes seurat
    sco = SingleCellOMIC(mRNA, gene_id=gene_names)
    sc.pp.filter_cells(sco, min_genes=200)
    sc.pp.filter_genes(sco, min_cells=3)
    sc.pp.normalize_total(sco, target_sum=1e4)
    result = sc.pp.filter_genes_dispersion(sco.X,
                                           min_mean=0.0125,
                                           max_mean=3,
                                           min_disp=0.5,
                                           log=False,
                                           n_top_genes=2000)
    gene_subset = result.gene_subset
    # maker sure all marker genes included
    for i, gene in enumerate(gene_names):
      if gene in MARKER_GENES:
        gene_subset[i] = True
    sco._inplace_subset_var(gene_subset)
    top_genes = set(sco.var_names.values)
    if verbose:
      print(f"Filtered highly variable genes: {len(top_genes)}")
    del sco
    # save the indices and top_genes
    with open(os.path.join(preprocessed_path, 'gene_indices'), 'wb') as f:
      pickle.dump([gene_names, indices, top_genes], f)
    print(f"Preprocessed MD5: {md5_folder(preprocessed_path)}")
  # ******************** load the dataset ******************** #
  with open(os.path.join(preprocessed_path, 'gene_indices'), 'rb') as f:
    gene_names, indices, top_genes = pickle.load(f)
  sco = _DATASETS[name](verbose=verbose)
  sco._inplace_subset_var(indices[name])
  if filtered_genes:
    top_indices = [i in top_genes for i in sco.var_names]
    sco._inplace_subset_var(top_indices)
  sco._name += 'x'
  return sco
