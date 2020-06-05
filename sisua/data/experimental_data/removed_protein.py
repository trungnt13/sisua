from __future__ import absolute_import, division, print_function

import os
import shutil

import numpy as np
from six import string_types

from odin.fuel import Dataset
from odin.utils import as_tuple
from sisua.data.data_loader.pbmc8k import read_PBMC8k
from sisua.data.data_loader.pbmcecc import read_PBMCeec
from sisua.data.path import DATA_DIR
from sisua.data.utils import save_to_dataset, standardize_protein_name


# ===========================================================================
# Remove set of protein
# ===========================================================================
def read_PBMC_crossdataset_remove_protein(subset,
                                  return_ecc,
                                  filtered_genes=False,
                                  override=False,
                                  verbose=False,
                                  remove_protein=['CD4', 'CD8']):
  remove_protein = sorted(
      [i.lower() for i in as_tuple(remove_protein, t=string_types)])
  preprocessed_path = os.path.join(
      DATA_DIR, 'PBMCcross_%s_%s_no%s_preprocessed' %
      ('ecc' if return_ecc else '8k', subset +
       ('' if filtered_genes else 'full'), ''.join(
           [i.lower() for i in remove_protein])))
  if override and os.path.exists(preprocessed_path):
    shutil.rmtree(preprocessed_path)
  if not os.path.exists(preprocessed_path):
    os.mkdir(preprocessed_path)

  # ******************** preprocessed ******************** #
  if not os.path.exists(os.path.join(preprocessed_path, 'X')):
    ds = read_PBMC_crossdataset_ecc_8k(subset,
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
