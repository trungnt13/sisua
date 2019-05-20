from __future__ import print_function, division, absolute_import
import os
import shutil

import numpy as np
from odin.fuel import Dataset

from sisua.data.data_loader.pbmcecc import read_PBMCeec
from sisua.data.data_loader.pbmc8k import read_PBMC8k
from sisua.data.path import PREPROCESSED_BASE_DIR
from sisua.data.utils import save_to_dataset


def read_PBMC_ecc_to_8k(subset, return_ecc, override=False, filtered_genes=False):
  """ This create a dataset with shared genes subset between
  PBMC-ecc and PBMC-8k

  It will select the most overlap subset when `filtered_genes=True`
  """
  preprocessed_path = os.path.join(
      PREPROCESSED_BASE_DIR,
      'PBMCcross_%s_%s_preprocessed' %
      ('ecc' if return_ecc else '8k',
       subset + ('' if filtered_genes else 'full')))
  if override and os.path.exists(preprocessed_path):
    shutil.rmtree(preprocessed_path)
  if not os.path.exists(preprocessed_path):
    os.mkdir(preprocessed_path)

  # ******************** preprocessed ******************** #
  if not os.path.exists(os.path.join(preprocessed_path, 'X')):
    pbmc8k_full = read_PBMC8k(
        subset=subset, override=override, filtered_genes=False)
    pbmcecc_full = read_PBMCeec(
        subset=subset, override=override, filtered_genes=False)

    all_genes = set(pbmc8k_full['X_col']) & set(pbmcecc_full['X_col'])

    if filtered_genes:
      pbmc8k = read_PBMC8k(
          subset=subset, override=override, filtered_genes=True)
      pbmcecc = read_PBMCeec(
          subset=subset, override=override, filtered_genes=True)
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
    save_to_dataset(preprocessed_path, X, X_col, y, y_col,
                    rowname=X_row)
  # ******************** return ******************** #
  ds = Dataset(preprocessed_path, read_only=True)
  return ds
