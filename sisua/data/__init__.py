from __future__ import absolute_import, division, print_function

import copy
import os
from collections import OrderedDict
from functools import partial
from numbers import Number
from typing import Tuple

import numpy as np

from odin.stats import describe, sparsity_percentage, train_valid_test_split
from odin.utils import cache_memory, catch_warnings_ignore, one_hot
from sisua.data.const import (MARKER_ADT_GENE, MARKER_ADTS, MARKER_ATAC,
                              MARKER_GENES, OMIC, PROTEIN_PAIR_NEGATIVE,
                              PROTEIN_PAIR_POSITIVE, UNIVERSAL_RANDOM_SEED)
from sisua.data.path import CONFIG_PATH, DATA_DIR, EXP_DIR
from sisua.data.single_cell_dataset import SingleCellOMIC
from sisua.data.utils import (apply_artificial_corruption, get_gene_id2name,
                              get_library_size, is_binary_dtype,
                              is_categorical_dtype, standardize_protein_name,
                              validating_dataset)


def get_dataset_meta():
  r"""
  Return:
    a Dictionary : mapping from dataset name -> loading_function()
  """
  from sisua.data.data_loader.childhood_leukemia_cALL import read_leukemia_BMMC
  from sisua.data.data_loader.dataset10x import read_dataset10x
  from sisua.data.data_loader.pbmc_CITEseq import read_CITEseq_PBMC
  from sisua.data.data_loader.cbmc_CITEseq import read_CITEseq_CBMC
  from sisua.data.data_loader.facs_gene_protein import read_FACS, read_full_FACS
  from sisua.data.data_loader.scvi_datasets import (read_Cortex, read_Hemato,
                                                    read_PBMC, read_Retina)
  from sisua.data.data_loader.pbmc8k import read_PBMC8k
  from sisua.data.data_loader.pbmcecc import read_PBMCeec
  from sisua.data.data_loader.centenarian import read_centenarian
  from sisua.data.utils import standardize_protein_name
  from sisua.data.data_loader.mixed_phenotype_acute_leukemia import read_leukemia_MixedPhenotypes
  from sisua.data.experimental_data.pbmc_cross_datasets import read_PBMC_crossdataset
  data_meta = {
      # ====== pbmc 8k ====== #
      "call":
          read_leukemia_BMMC,
      "callall":
          partial(read_leukemia_BMMC, filtered_genes=False),
      # ====== pbmc 8k ====== #
      "mpal":
          partial(read_leukemia_MixedPhenotypes, omic='rna'),
      "mpalall":
          partial(read_leukemia_MixedPhenotypes,
                  omic='rna',
                  filtered_genes=False),
      "mpalatac":
          partial(read_leukemia_MixedPhenotypes, omic='atac'),
      # ====== pbmc 8k ====== #
      # "100yo":
      #     read_centenarian,
      # ====== pbmc 8k ====== #
      '8klyall':
          partial(read_PBMC8k, subset='ly', filtered_genes=False),
      '8kmyall':
          partial(read_PBMC8k, subset='my', filtered_genes=False),
      '8kly':
          partial(read_PBMC8k, subset='ly', filtered_genes=True),
      '8kmy':
          partial(read_PBMC8k, subset='my', filtered_genes=True),
      '8k':
          partial(read_PBMC8k, subset='full', filtered_genes=True),
      '8kall':
          partial(read_PBMC8k, subset='full', filtered_genes=False),
      # ====== PBMC ECC ====== #
      'ecclyall':
          partial(read_PBMCeec, subset='ly', filtered_genes=False),
      'eccly':
          partial(read_PBMCeec, subset='ly', filtered_genes=True),
      # 'eccmyall':
      #     partial(read_PBMCeec, subset='my', filtered_genes=False),
      # 'eccmy':
      #     partial(read_PBMCeec, subset='my', filtered_genes=True),
      # 'ecc':
      #     partial(read_PBMCeec, subset='full', filtered_genes=True),
      # 'eccall':
      #     partial(read_PBMCeec, subset='full', filtered_genes=False),
      # ====== cross PBMC ====== #
      '8kx':
          partial(read_PBMC_crossdataset, name='pbmc8k', filtered_genes=True),
      '8kxall':
          partial(read_PBMC_crossdataset, name='pbmc8k', filtered_genes=False),
      'eccx':
          partial(read_PBMC_crossdataset, name='pbmcecc', filtered_genes=True),
      'eccxall':
          partial(read_PBMC_crossdataset, name='pbmcecc', filtered_genes=False),
      'vdj1x':
          partial(read_PBMC_crossdataset, name='vdj1', filtered_genes=True),
      'vdj1xall':
          partial(read_PBMC_crossdataset, name='vdj1', filtered_genes=False),
      'vdj4x':
          partial(read_PBMC_crossdataset, name='vdj4', filtered_genes=True),
      'vdj4xall':
          partial(read_PBMC_crossdataset, name='vdj4', filtered_genes=False),
      'mpalx':
          partial(read_PBMC_crossdataset, name='mpal', filtered_genes=True),
      'mpalxall':
          partial(read_PBMC_crossdataset, name='mpal', filtered_genes=False),
      'callx':
          partial(read_PBMC_crossdataset, name='call', filtered_genes=True),
      'callxall':
          partial(read_PBMC_crossdataset, name='call', filtered_genes=False),
      'pbmcx':
          partial(read_PBMC_crossdataset,
                  name='pbmcciteseq',
                  filtered_genes=True),
      'cbmcx':
          partial(read_PBMC_crossdataset,
                  name='cbmcciteseq',
                  filtered_genes=True),
      # '8knocd4x':
      #     partial(read_PBMC_crossdataset_remove_protein,
      #             subset='ly',
      #             return_ecc=False,
      #             filtered_genes=True,
      #             remove_protein='CD4'),
      # 'eccnocd4x':
      #     partial(read_PBMC_crossdataset_remove_protein,
      #             subset='ly',
      #             return_ecc=True,
      #             filtered_genes=True,
      #             remove_protein='CD4'),
      # '8knocd8x':
      #     partial(read_PBMC_crossdataset_remove_protein,
      #             subset='ly',
      #             return_ecc=False,
      #             filtered_genes=True,
      #             remove_protein='CD8'),
      # 'eccnocd8x':
      #     partial(read_PBMC_crossdataset_remove_protein,
      #             subset='ly',
      #             return_ecc=True,
      #             filtered_genes=True,
      #             remove_protein='CD8'),
      # '8knocd48x':
      #     partial(read_PBMC_crossdataset_remove_protein,
      #             subset='ly',
      #             return_ecc=False,
      #             filtered_genes=True,
      #             remove_protein=['CD4', 'CD8']),
      # 'eccnocd48x':
      #     partial(read_PBMC_crossdataset_remove_protein,
      #             subset='ly',
      #             return_ecc=True,
      #             filtered_genes=True,
      #             remove_protein=['CD4', 'CD8']),
      # '8konlycd8x':
      #     partial(read_PBMC_crossdataset_remove_protein,
      #             subset='ly',
      #             return_ecc=False,
      #             filtered_genes=True,
      #             remove_protein=['CD3', 'CD4', 'CD16', 'CD56', 'CD19']),
      # ====== CITEseq ====== #
      'pbmcciteseq':
          read_CITEseq_PBMC,
      'cbmcciteseq':
          read_CITEseq_CBMC,
      'pbmc5000':
          partial(read_CITEseq_PBMC, filtered_genes=True),
      # ====== FACS ====== #
      'facs7':
          read_full_FACS,
      'facs5':
          partial(read_FACS, n_protein=5),
      'facs2':
          partial(read_FACS, n_protein=2),
      # ====== other fun ====== #
      'pbmcscvi':
          read_PBMC,
      'cortex':
          read_Cortex,
      'retina':
          read_Retina,
      'hemato':
          read_Hemato,
  }
  # add 10xgenomics data
  for alias, name in [
      ('vdj1', 'vdj_v1_hs_aggregated_donor1'),
      ('vdj2', 'vdj_v1_hs_aggregated_donor2'),
      ('vdj3', 'vdj_v1_hs_aggregated_donor3'),
      ('vdj4', 'vdj_v1_hs_aggregated_donor4'),
      ("neuron10k", "neuron_10k_v3"),
      ("heart10k", "heart_10k_v3"),
      ('memoryt', 'memory_t'),
      ('naivet', 'naive_t'),
      ('regulatoryt', 'regulatory_t'),
      ('cd4t', 'cd4_t_helper'),
      ("4k", "pbmc4k"),
      ("5k", "5k_pbmc_protein_v3"),
      ("10k", "pbmc_10k_protein_v3"),
      ("18k", "pbmc8k"),
  ]:
    data_meta[alias] = partial(read_dataset10x,
                               name=name,
                               filtered_cells=True,
                               filtered_genes=True)
    data_meta[alias + 'all'] = partial(read_dataset10x,
                                       name=name,
                                       filtered_cells=True,
                                       filtered_genes=False)
  # check validate alias for datasets
  import re
  pattern = re.compile('\w*')
  for name in data_meta.keys():
    assert pattern.match(name) and '_' not in name
  return data_meta


def get_dataset_summary(return_html=False):
  import pandas as pd
  all_datasets = []
  for name, fn in sorted(get_dataset_meta().items()):
    ds = fn(override=False)
    info = OrderedDict([
        ('Keyword', name),
        ('#Cells', ds['X'].shape[0]),
        ('#Genes', ds['X'].shape[1]),
        ('#Labels', ds['y'].shape[1]),
        ('Binary', sorted(np.unique(ds['y'].astype('float32'))) == [0., 1.]),
        ('Labels', ', '.join([standardize_protein_name(i) for i in ds['y_col']
                             ])),
    ])
    all_datasets.append(info)
  df = pd.DataFrame(all_datasets)
  if return_html:
    return df.to_html()
  return df


def get_dataset(dataset_name, override=False, verbose=True) -> SingleCellOMIC:
  r""" Check `get_dataset_meta` for more information

  List of all dataset available: ['call', 'callall', 'mpal', 'mpalall',
    'mpalatac', '100yo', '8klyall', '8kmyall', '8kly', '8kmy', '8k',
    '8kall', 'ecclyall', 'eccly', 'eccmyall', 'eccmy', 'ecc', 'eccall',
    '8kx', '8kxall', 'eccx', 'eccxall', 'vdj1x', 'vdj1xall', 'vdj4x',
    'vdj4xall', 'mpalx', 'mpalxall', 'callx', 'callxall', 'pbmcciteseq',
    'cbmcciteseq', 'pbmc5000', 'facs7', 'facs5', 'facs2', 'pbmcscvi',
    'cortex', 'retina', 'hemato', 'vdj1', 'vdj1all', 'vdj2', 'vdj2all',
    'vdj3', 'vdj3all', 'vdj4', 'vdj4all', 'vdjhs3', 'vdjhs3all', 'vdjhs4',
    'vdjhs4all', 'neuron10k', 'neuron10kall', 'heart10k', 'heart10kall',
    'memoryt', 'memorytall', 'naivet', 'naivetall', 'regulatoryt',
    'regulatorytall', 'cd4t', 'cd4tall', '5k', '5kall', '18k', '18kall',
    '4k', '4kall', '10k', '10kall']

  Return:
    mRNA data : `SingleCellOMIC`
    label data: `SingleCellOMIC`. If label data is not availabel, then None

  Example:
    gene, prot = get_dataset("cortex")
    X_train, X_test = gene.split(0.8, seed=1234)
    y_train, y_test = prot.split(0.8, seed=1234)
    X_train.assert_matching_cells(y_train)
    X_test.assert_matching_cells(y_test)
  """
  data_meta = get_dataset_meta()
  # ====== special case: get all dataset ====== #
  dataset_name = str(dataset_name).lower().strip()
  if dataset_name not in data_meta:
    raise RuntimeError(
        'Cannot find dataset with name: "%s", all dataset include: %s' %
        (dataset_name, ", ".join(list(data_meta.keys()))))
  with catch_warnings_ignore(FutureWarning):
    ds = data_meta[dataset_name](override=override, verbose=verbose)
  # ******************** create SCO ******************** #
  if isinstance(ds, SingleCellOMIC):
    return ds
  # ******************** return ******************** #
  validating_dataset(ds)
  with catch_warnings_ignore(FutureWarning):
    sc = SingleCellOMIC(X=ds['X'],
                        cell_id=ds['X_row'],
                        gene_id=ds['X_col'],
                        name=dataset_name)
    if 'y' in ds:
      y = ds['y']
      if is_binary_dtype(y):
        sc.add_omic(OMIC.celltype, y, ds['y_col'])
      else:
        sc.add_omic(OMIC.proteomic, y, ds['y_col'])
  return sc
