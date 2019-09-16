from __future__ import absolute_import, division, print_function

import os
import shutil

import numpy as np
from six import string_types

from odin.fuel import Dataset
from odin.utils import as_tuple
from sisua.data.data_loader.pbmc8k import read_PBMC8k
from sisua.data.data_loader.pbmcecc import read_PBMCeec
from sisua.data.path import PREPROCESSED_BASE_DIR
from sisua.data.utils import save_to_dataset, standardize_protein_name


def read_PBMCcross_ecc_8k(subset,
                          return_ecc,
                          filtered_genes=False,
                          override=False,
                          verbose=False):
  """ This create a dataset with shared genes subset between
  PBMC-ecc and PBMC-8k

  It will select the most overlap subset when `filtered_genes=True`
  """
  preprocessed_path = os.path.join(
      PREPROCESSED_BASE_DIR,
      'PBMCcross_%s_%s_preprocessed' % ('ecc' if return_ecc else '8k', subset +
                                        ('' if filtered_genes else 'full')))
  if override and os.path.exists(preprocessed_path):
    shutil.rmtree(preprocessed_path)
  if not os.path.exists(preprocessed_path):
    os.mkdir(preprocessed_path)

  # ******************** preprocessed ******************** #
  if not os.path.exists(os.path.join(preprocessed_path, 'X')):
    pbmc8k_full = read_PBMC8k(subset=subset,
                              override=override,
                              filtered_genes=False,
                              verbose=verbose)
    pbmcecc_full = read_PBMCeec(subset=subset,
                                override=override,
                                filtered_genes=False,
                                verbose=verbose)

    all_genes = set(pbmc8k_full['X_col']) & set(pbmcecc_full['X_col'])

    if filtered_genes:
      pbmc8k = read_PBMC8k(subset=subset,
                           override=override,
                           filtered_genes=True,
                           verbose=verbose)
      pbmcecc = read_PBMCeec(subset=subset,
                             override=override,
                             filtered_genes=True,
                             verbose=verbose)
      s1 = set(pbmc8k['X_col']) & all_genes
      s2 = set(pbmcecc['X_col']) & all_genes
      all_genes = s2 if len(s2) > len(s1) else s1
    # the same order all the time
    all_genes = sorted(all_genes)

    pbmc = pbmcecc_full if return_ecc else pbmc8k_full
    X = pbmc['X']
    X_row = pbmc['X_row']
    X_col = pbmc['X_col']
    y = pbmc['y']
    y_col = pbmc['y_col']

    X_col_indices = {gene: i for i, gene in enumerate(X_col)}
    indices = np.array([X_col_indices[gene] for gene in all_genes])
    X = X[:, indices]
    X_col = X_col[indices]
    save_to_dataset(preprocessed_path,
                    X,
                    X_col,
                    y,
                    y_col,
                    rowname=X_row,
                    print_log=verbose)
  # ******************** return ******************** #
  ds = Dataset(preprocessed_path, read_only=True)
  return ds


# ===========================================================================
# Remove set of protein
# ===========================================================================
def read_PBMCcross_remove_protein(subset,
                                  return_ecc,
                                  filtered_genes=False,
                                  override=False,
                                  verbose=False,
                                  remove_protein=['CD4', 'CD8']):
  remove_protein = sorted(
      [i.lower() for i in as_tuple(remove_protein, t=string_types)])
  preprocessed_path = os.path.join(
      PREPROCESSED_BASE_DIR, 'PBMCcross_%s_%s_no%s_preprocessed' %
      ('ecc' if return_ecc else '8k', subset +
       ('' if filtered_genes else 'full'), ''.join(
           [i.lower() for i in remove_protein])))
  if override and os.path.exists(preprocessed_path):
    shutil.rmtree(preprocessed_path)
  if not os.path.exists(preprocessed_path):
    os.mkdir(preprocessed_path)

  # ******************** preprocessed ******************** #
  if not os.path.exists(os.path.join(preprocessed_path, 'X')):
    ds = read_PBMCcross_ecc_8k(subset,
                               return_ecc,
                               filtered_genes,
                               override=override,
                               verbose=verbose)
    X = ds['X'][:]
    X_row = ds['X_row']
    X_col = ds['X_col']
    y = ds['y']
    y_col = ds['y_col']

    remove_ids = [
        i for i, j in enumerate(y_col)
        if standardize_protein_name(j).lower() in remove_protein
    ]
    remain_ids = [i for i in range(len(y_col)) if i not in remove_ids]
    y_col = y_col[remain_ids]
    y = y[:, remain_ids]

    save_to_dataset(preprocessed_path,
                    X,
                    X_col,
                    y,
                    y_col,
                    rowname=X_row,
                    print_log=verbose)
  # ******************** return ******************** #
  ds = Dataset(preprocessed_path, read_only=True)
  return ds
