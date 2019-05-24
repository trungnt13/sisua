from __future__ import print_function, division, absolute_import

import os
import re
import copy
import pandas as pd
from numbers import Number
from six import string_types

import numpy as np
from odin.fuel import MmapData
from odin.utils import ctext, one_hot, cache_memory
from odin.utils.crypto import md5_checksum
from odin.stats import (train_valid_test_split, describe,
                        sparsity_percentage)

from sisua.data.path import EXP_DIR
from sisua.data.const import UNIVERSAL_RANDOM_SEED
from sisua.data.utils import validating_dataset
from sisua.data.single_cell_dataset import (
    apply_artificial_corruption, get_library_size, SingleCellDataset)

def get_dataset_meta():
  from sisua.data.data_loader.pbmc_CITEseq import read_CITEseq_PBMC
  from sisua.data.data_loader.pbmc10x_pp import read_10xPBMC_PP
  from sisua.data.data_loader.cbmc_CITEseq import read_CITEseq_CBMC
  from sisua.data.data_loader.mnist import read_MNIST
  from sisua.data.data_loader.facs_gene_protein import read_FACS, read_full_FACS
  from sisua.data.data_loader.fashion_mnist import (
      read_fashion_MNIST, read_fashion_MNIST_drop, read_MNIST_drop)
  from sisua.data.data_loader.scvi_datasets import (
      read_Cortex, read_Hemato, read_PBMC, read_Retina)
  from sisua.data.data_loader.pbmc8k import read_PBMC8k
  from sisua.data.data_loader.pbmcecc import read_PBMCeec
  from sisua.data.experimental_data.pbmc_8k_ecc_ly import read_PBMC_ecc_to_8k

  data_meta = {
      # ====== PBMC 10x ====== #
      'pbmcscvae': read_10xPBMC_PP,
      'pbmcscvi': read_PBMC,

      # ====== pbmc 8k ====== #
      'pbmc8k_lyfull': lambda override: read_PBMC8k(subset='ly', override=override, filtered_genes=False),
      'pbmc8k_myfull': lambda override: read_PBMC8k(subset='my', override=override, filtered_genes=False),
      'pbmc8k_ly': lambda override: read_PBMC8k(subset='ly', override=override, filtered_genes=True),
      'pbmc8k_my': lambda override: read_PBMC8k(subset='my', override=override, filtered_genes=True),
      'pbmc8k': lambda override: read_PBMC8k(subset='full', override=override, filtered_genes=True),
      'pbmc8k_full': lambda override: read_PBMC8k(subset='full', override=override, filtered_genes=False),

      # ====== PBMC ECC ====== #
      'pbmcecc_lyfull': lambda override: read_PBMCeec(subset='ly', override=override, filtered_genes=False),
      'pbmcecc_myfull': lambda override: read_PBMCeec(subset='my', override=override, filtered_genes=False),
      'pbmcecc_ly': lambda override: read_PBMCeec(subset='ly', override=override, filtered_genes=True),
      'pbmcecc_my': lambda override: read_PBMCeec(subset='my', override=override, filtered_genes=True),
      'pbmcecc': lambda override: read_PBMCeec(subset='full', override=override, filtered_genes=True),
      'pbmcecc_full': lambda override: read_PBMCeec(subset='full', override=override, filtered_genes=False),

      # ====== cross PBMC ====== #
      'cross8k_lyfull': lambda override: read_PBMC_ecc_to_8k(subset='ly', return_ecc=False, override=override, filtered_genes=False),
      'cross8k_ly': lambda override: read_PBMC_ecc_to_8k(subset='ly', return_ecc=False, override=override, filtered_genes=True),

      'crossecc_lyfull': lambda override: read_PBMC_ecc_to_8k(subset='ly', return_ecc=True, override=override, filtered_genes=False),
      'crossecc_ly': lambda override: read_PBMC_ecc_to_8k(subset='ly', return_ecc=True, override=override, filtered_genes=True),

      # ====== CITEseq ====== #
      'pbmc_citeseq': read_CITEseq_PBMC,
      'cbmc_citeseq': read_CITEseq_CBMC,
      'pbmc5000': lambda override: read_CITEseq_PBMC(override, version_5000genes=True),

      # ====== MNIST ====== #
      'mnist': read_MNIST,
      'mnist_org': read_MNIST,
      'mnist_imp': read_MNIST_drop,

      'fmnist': read_fashion_MNIST,
      'fmnist_org': read_fashion_MNIST,
      'fmnist_imp': read_fashion_MNIST_drop,

      # ====== FACS ====== #
      'facs_7': lambda override: read_full_FACS(override=override),
      'facs_5': lambda override: read_FACS(n_protein=5, override=override),
      'facs_2': lambda override: read_FACS(n_protein=2, override=override),

      # ====== other fun ====== #
      'cortex': read_Cortex,
      'retina': read_Retina,
      'hemato': read_Hemato,
  }
  return data_meta

@cache_memory
def get_dataset(dataset_name, override=False):
  """ Supporting dataset:

  'pbmc_citeseq' :
  'pbmc_10x' :
  'pbmc' :
  'pbmc_5000' :

  'pbmc8k' :
  'pbmc8k_full' :

  'pbmc_lyfull' :
  'pbmc_myfull' :
  'pbmc_ly' :
  'pbmc_my' :

  'cbmc_citeseq' :
  'mnist' :
  'mnist_org' :
  'mnist_imp' :
  'fmnist' :
  'fmnist_org' :
  'fmnist_imp' :
  'facs_7' :
  'facs_5' :
  'facs_2' :
  'facs_corrupt' :
  'cortex' :
  'retina' :
  'hemato' :

  Return
  ------
  dataset: `odin.fuel.dataset.Dataset` contains original data
  cell_gene: instance of `sisua.data.SingleCellDataset` for cell-gene matrix
  cell_protein: instance of `sisua.data.SingleCellDataset` for cell-protein matrix
  """
  data_meta = get_dataset_meta()
  # ====== special case: get all dataset ====== #
  dataset_name = str(dataset_name).lower().strip()
  if dataset_name not in data_meta:
    raise RuntimeError(
        'Cannot find dataset with name: "%s", all dataset include: %s'
        % (dataset_name, ",".join(list(data_meta.keys()))))
  ds = data_meta[dataset_name](override=override)
  validating_dataset(ds)
  # ====== get particular dataset ====== #
  cell_gene = SingleCellDataset(
      data=ds['X'], rowname=ds['X_row'], colname=ds['X_col'])

  cell_prot = SingleCellDataset(
      data=ds['y'], rowname=ds['X_row'], colname=ds['y_col'])

  assert cell_gene.md5 == cell_prot.md5
  # ******************** return ******************** #
  setattr(ds, 'name', dataset_name)
  setattr(cell_gene, 'name', dataset_name)
  setattr(cell_prot, 'name', dataset_name)
  return ds, cell_gene, cell_prot

def get_scvi_dataset(dataset_name):
  """ Convert any SISUA dataset to relevant format for scVI models """
  from scvi.dataset import GeneExpressionDataset
  ds, gene, prot = get_dataset(dataset_name, override=False)
  X = np.concatenate((gene.X_train, gene.X_test), axis=0)
  labels = np.concatenate((prot.X_train, prot.X_test), axis=0)
  means, vars = get_library_size(X)
  is_multi_classes_labels = np.all(np.sum(labels, axis=1) != 1.)
  scvi = GeneExpressionDataset(
      X=X,
      local_means=means,
      local_vars=vars,
      batch_indices=np.zeros(shape=(X.shape[0], 1)),
      labels=None,
      gene_names=gene.X_col,
      cell_types=None)
  if not is_multi_classes_labels:
    scvi.labels = labels
    scvi.cell_types = prot.X_col
  else:
    scvi.labels = labels
    scvi.adt_expression = labels
    scvi.protein_markers = prot.X_col
  return scvi

def get_dataframe(dataset_name, override=False):
  """ Return 2 tuples of the DataFrame instance of:

    * (train_gene, test_gene)
    * (train_prot, test_prot)
  """
  ds, gene_ds, prot_ds = get_dataset(dataset_name=dataset_name, override=override)
  train_gene = pd.DataFrame(data=gene_ds['train'],
                            index=gene_ds.row_name[0],
                            columns=gene_ds.col_name)
  test_gene = pd.DataFrame(data=gene_ds['test'],
                           index=gene_ds.row_name[1],
                           columns=gene_ds.col_name)

  train_prot = pd.DataFrame(data=prot_ds['train'],
                            index=prot_ds.row_name[0],
                            columns=prot_ds.col_name)
  test_prot = pd.DataFrame(data=prot_ds['test'],
                           index=prot_ds.row_name[1],
                           columns=prot_ds.col_name)
  return (train_gene, test_gene), (train_prot, test_prot)

# ===========================================================================
# Some shortcut
# ===========================================================================
def Cortex():
  return get_dataset('cortex')[0]

def PBMCscVI():
  """ The PBMC dataset used in scVI paper """
  return get_dataset('pbmcscvi')[0]

def PBMCscVAE():
  """ The PBMC dataset used in scVAE paper """
  return get_dataset('pbmcscvae')[0]

# ====== PBMC 8k ====== #
def PBMC8k_lymphoid(filtered_genes=True):
  """ lymphoid subset of PBMC 8k"""
  return get_dataset(
      'pbmc8k_ly' if filtered_genes else 'pbmc8k_lyfull')[0]

def PBMC8k_myeloid(filtered_genes=True):
  """ myeloid subset of PBMC 8k"""
  return get_dataset(
      'pbmc8k_my' if filtered_genes else 'pbmc8k_myfull')[0]

def PBMC8k(filtered_genes=True):
  """ PBMC 8k"""
  return get_dataset(
      'pbmc8k' if filtered_genes else 'pbmc8k_full')[0]

# ====== PBMC ecc ====== #
def PBMCecc_lymphoid(filtered_genes=True):
  """ lymphoid subset of PBMC ecc"""
  return get_dataset(
      'pbmcecc_ly' if filtered_genes else 'pbmcecc_lyfull')[0]

def PBMCecc_myeloid(filtered_genes=True):
  """ myeloid subset of PBMC ecc"""
  return get_dataset(
      'pbmcecc_my' if filtered_genes else 'pbmcecc_myfull')[0]

def PBMCecc(filtered_genes=True):
  """ PBMC ecc"""
  return get_dataset(
      'pbmcecc' if filtered_genes else 'pbmcecc_full')[0]

# ====== cross dataset ====== #
def CROSS8k_lymphoid(filtered_genes=True):
  return get_dataset(
      'cross8k_ly' if filtered_genes else 'cross8k_lyfull')[0]

def CROSSecc_lymphoid(filtered_genes=True):
  return get_dataset(
      'crossecc_ly' if filtered_genes else 'crossecc_lyfull')[0]
