from __future__ import absolute_import, division, print_function

import copy
import os
from collections import OrderedDict
from numbers import Number
from typing import Tuple

import numpy as np

from odin.stats import describe, sparsity_percentage, train_valid_test_split
from odin.utils import cache_memory, ctext, one_hot
from sisua.data import normalization_recipes
from sisua.data.const import UNIVERSAL_RANDOM_SEED
from sisua.data.single_cell_dataset import (SingleCellOMIC,
                                            apply_artificial_corruption,
                                            get_library_size)
from sisua.data.utils import (get_gene_id2name, standardize_protein_name,
                              validating_dataset)


def get_dataset_meta():
  """
  Return
  ------
  dictionary : dataset name -> loading_function()
  """
  from sisua.data.data_loader.dataset10x import read_dataset10x_cellexp
  from sisua.data.data_loader.pbmc_CITEseq import read_CITEseq_PBMC
  from sisua.data.data_loader.cbmc_CITEseq import read_CITEseq_CBMC
  from sisua.data.data_loader.mnist import read_MNIST
  from sisua.data.data_loader.fashion_mnist import read_fashion_MNIST
  from sisua.data.data_loader.facs_gene_protein import read_FACS, read_full_FACS
  from sisua.data.data_loader.scvi_datasets import (read_Cortex, read_Hemato,
                                                    read_PBMC, read_Retina)
  from sisua.data.data_loader.pbmc8k import read_PBMC8k
  from sisua.data.data_loader.pbmcecc import read_PBMCeec
  from sisua.data.experimental_data.pbmc_8k_ecc_ly import (
      read_PBMCcross_ecc_8k, read_PBMCcross_remove_protein)
  from sisua.data.data_loader.centenarian import read_centenarian
  data_meta = {
      "centenarian":
          read_centenarian,
      # ====== PBMC 10x ====== #
      "neuron10k":
          lambda override, verbose: read_dataset10x_cellexp(name=
                                                            'neuron_10k_v3',
                                                            spec='filtered',
                                                            override=override,
                                                            verbose=verbose),
      "heart10kv3":
          lambda override, verbose: read_dataset10x_cellexp(name='heart_10k_v3',
                                                            spec='filtered',
                                                            override=override,
                                                            verbose=verbose),
      "cellvdj":
          lambda override, verbose: read_dataset10x_cellexp(name='cellvdj',
                                                            spec='filtered',
                                                            override=override,
                                                            verbose=verbose),
      'memoryt':
          lambda override, verbose: read_dataset10x_cellexp(name='memory_t',
                                                            spec='filtered',
                                                            override=override,
                                                            verbose=verbose),
      'memorytraw':
          lambda override, verbose: read_dataset10x_cellexp(
              name='memory_t', spec='raw', override=override, verbose=verbose),
      'pbmc5k':
          lambda override, verbose: read_dataset10x_cellexp(
              name='5k_pbmc_protein_v3',
              spec='filtered',
              override=override,
              verbose=verbose),
      'pbmc5kgem':
          lambda override, verbose: read_dataset10x_cellexp(
              name='5k_pbmc_protein_v3_nextgem',
              spec='filtered',
              override=override,
              verbose=verbose),
      'pbmc5kraw':
          lambda override, verbose: read_dataset10x_cellexp(
              name='5k_pbmc_protein_v3',
              spec='raw',
              override=override,
              verbose=verbose),
      'pbmc5kgemraw':
          lambda override, verbose: read_dataset10x_cellexp(
              name='5k_pbmc_protein_v3_nextgem',
              spec='raw',
              override=override,
              verbose=verbose),
      'pbmcscvi':
          read_PBMC,

      # ====== pbmc 8k ====== #
      'pbmc8klyfull':
          lambda override, verbose: read_PBMC8k(subset='ly',
                                                override=override,
                                                verbose=verbose,
                                                filtered_genes=False),
      'pbmc8kmyfull':
          lambda override, verbose: read_PBMC8k(subset='my',
                                                override=override,
                                                verbose=verbose,
                                                filtered_genes=False),
      'pbmc8kly':
          lambda override, verbose: read_PBMC8k(subset='ly',
                                                override=override,
                                                verbose=verbose,
                                                filtered_genes=True),
      'pbmc8kmy':
          lambda override, verbose: read_PBMC8k(subset='my',
                                                override=override,
                                                verbose=verbose,
                                                filtered_genes=True),
      'pbmc8k':
          lambda override, verbose: read_PBMC8k(subset='full',
                                                override=override,
                                                verbose=verbose,
                                                filtered_genes=True),
      'pbmc8kfull':
          lambda override, verbose: read_PBMC8k(subset='full',
                                                override=override,
                                                verbose=verbose,
                                                filtered_genes=False),

      # ====== PBMC ECC ====== #
      'pbmcecclyfull':
          lambda override, verbose: read_PBMCeec(subset='ly',
                                                 override=override,
                                                 verbose=verbose,
                                                 filtered_genes=False),
      # 'pbmcecc_myfull': lambda override: read_PBMCeec(subset='my', override=override, filtered_genes=False),
      'pbmceccly':
          lambda override, verbose: read_PBMCeec(subset='ly',
                                                 override=override,
                                                 verbose=verbose,
                                                 filtered_genes=True),
      # 'pbmcecc_my': lambda override: read_PBMCeec(subset='my', override=override, filtered_genes=True),
      # 'pbmcecc': lambda override: read_PBMCeec(subset='full', override=override, filtered_genes=True),
      # 'pbmcecc_full': lambda override: read_PBMCeec(subset='full', override=override, filtered_genes=False),

      # ====== cross PBMC ====== #
      'cross8klyfull':
          lambda override, verbose: read_PBMCcross_ecc_8k(subset='ly',
                                                          return_ecc=False,
                                                          override=override,
                                                          verbose=verbose,
                                                          filtered_genes=False),
      'cross8kly':
          lambda override, verbose: read_PBMCcross_ecc_8k(subset='ly',
                                                          return_ecc=False,
                                                          override=override,
                                                          verbose=verbose,
                                                          filtered_genes=True),
      'crossecclyfull':
          lambda override, verbose: read_PBMCcross_ecc_8k(subset='ly',
                                                          return_ecc=True,
                                                          override=override,
                                                          verbose=verbose,
                                                          filtered_genes=False),
      'crosseccly':
          lambda override, verbose: read_PBMCcross_ecc_8k(subset='ly',
                                                          return_ecc=True,
                                                          override=override,
                                                          verbose=verbose,
                                                          filtered_genes=True),
      'cross8knocd4':
          lambda override, verbose: read_PBMCcross_remove_protein(
              subset='ly',
              return_ecc=False,
              override=override,
              verbose=verbose,
              filtered_genes=True,
              remove_protein='CD4'),
      'crosseccnocd4':
          lambda override, verbose: read_PBMCcross_remove_protein(
              subset='ly',
              return_ecc=True,
              override=override,
              verbose=verbose,
              filtered_genes=True,
              remove_protein='CD4'),
      'cross8knocd8':
          lambda override, verbose: read_PBMCcross_remove_protein(
              subset='ly',
              return_ecc=False,
              override=override,
              verbose=verbose,
              filtered_genes=True,
              remove_protein='CD8'),
      'crosseccnocd8':
          lambda override, verbose: read_PBMCcross_remove_protein(
              subset='ly',
              return_ecc=True,
              override=override,
              verbose=verbose,
              filtered_genes=True,
              remove_protein='CD8'),
      'cross8knocd48':
          lambda override, verbose: read_PBMCcross_remove_protein(
              subset='ly',
              return_ecc=False,
              override=override,
              verbose=verbose,
              filtered_genes=True,
              remove_protein=['CD4', 'CD8']),
      'crosseccnocd48':
          lambda override, verbose: read_PBMCcross_remove_protein(
              subset='ly',
              return_ecc=True,
              override=override,
              filtered_genes=True,
              remove_protein=['CD4', 'CD8']),
      'cross8konlycd8':
          lambda override, verbose: read_PBMCcross_remove_protein(
              subset='ly',
              return_ecc=False,
              override=override,
              verbose=verbose,
              filtered_genes=True,
              remove_protein=['CD3', 'CD4', 'CD16', 'CD56', 'CD19']),
      # ====== CITEseq ====== #
      'pbmcciteseq':
          read_CITEseq_PBMC,
      'cbmcciteseq':
          read_CITEseq_CBMC,
      'pbmc5000':
          lambda override, verbose: read_CITEseq_PBMC(
              override=override, verbose=verbose, version_5000genes=True),

      # ====== MNIST ====== #
      'mnist':
          read_MNIST,
      'fmnist':
          read_fashion_MNIST,

      # ====== FACS ====== #
      'facs7':
          lambda override, verbose: read_full_FACS(override=override,
                                                   verbose=verbose),
      'facs5':
          lambda override, verbose: read_FACS(
              n_protein=5, override=override, verbose=verbose),
      'facs2':
          lambda override, verbose: read_FACS(
              n_protein=2, override=override, verbose=verbose),

      # ====== other fun ====== #
      'cortex':
          read_Cortex,
      'retina':
          read_Retina,
      'hemato':
          read_Hemato,
  }
  import re
  pattern = re.compile('\w*')
  for name in data_meta.keys():
    assert pattern.match(name) and '_' not in name
  return data_meta


def get_dataset_summary(return_html=False):
  from sisua.data.utils import standardize_protein_name
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


def get_dataset(dataset_name, override=False,
                verbose=False) -> Tuple[SingleCellOMIC, SingleCellOMIC]:
  """ Check `get_dataset_meta` for more information

  Return
  ------
  dataset: `odin.fuel.dataset.Dataset` contains original data
  """
  data_meta = get_dataset_meta()
  # ====== special case: get all dataset ====== #
  dataset_name = str(dataset_name).lower().strip()
  if dataset_name not in data_meta:
    raise RuntimeError(
        'Cannot find dataset with name: "%s", all dataset include: %s' %
        (dataset_name, ",".join(list(data_meta.keys()))))
  ds = data_meta[dataset_name](override=override, verbose=verbose)
  validating_dataset(ds)
  # ******************** return ******************** #
  var = {'geneid': ds['X_col']}
  if 'X_col_name' in ds:
    var['genename'] = ds['X_col_name']
  x = SingleCellOMIC(X=ds['X'],
                     obs={'cellid': ds['X_row']},
                     var=var,
                     name=dataset_name)

  if 'y' in ds:
    var = {'protid': ds['y_col']}
    if 'y_col_name' in ds:
      var['protname'] = ds['y_col_name']
    y = SingleCellOMIC(X=ds['y'],
                       obs={'cellid': ds['X_row']},
                       var=var,
                       name=dataset_name)
  else:
    y = None
  return x, y


def get_scvi_dataset(dataset_name):
  """ Convert any SISUA dataset to relevant format for scVI models """
  from scvi.dataset import GeneExpressionDataset
  ds, gene, prot = get_dataset(dataset_name, override=False)
  X = np.concatenate((gene.X_train, gene.X_test), axis=0)
  labels = np.concatenate((prot.X_train, prot.X_test), axis=0)
  means, vars = get_library_size(X)
  is_multi_classes_labels = np.all(np.sum(labels, axis=1) != 1.)
  scvi = GeneExpressionDataset(X=X,
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
