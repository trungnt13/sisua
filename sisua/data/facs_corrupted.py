from __future__ import print_function, division, absolute_import
import os
import shutil

import numpy as np
from odin.fuel import Dataset, MmapData
from odin.stats import describe, sparsity_percentage

from sisua.data.path import PREPROCESSED_BASE_DIR

_FACSC_PREPROCESSED = os.path.join(
    PREPROCESSED_BASE_DIR, 'FACSC_preprocessed')

def read_FACS_corrupted(override=False):
  from data.facs_gene_protein import read_FACS
  ds = read_FACS(n_protein=5, override=False)
  # ====== check the path ====== #
  preprocessed_path = _FACSC_PREPROCESSED
  if not os.path.exists(preprocessed_path):
    os.mkdir(preprocessed_path)
  elif override:
    shutil.rmtree(preprocessed_path)
    os.mkdir(preprocessed_path)
  # ====== start corrupting old dataset ====== #
  if len(os.listdir(preprocessed_path)) == 0:
    shutil.copy2(os.path.join(ds.path, 'y'),
                 os.path.join(preprocessed_path, 'y'))
    shutil.copy2(os.path.join(ds.path, 'y_col'),
                 os.path.join(preprocessed_path, 'y_col'))

    shutil.copy2(os.path.join(ds.path, 'X_col'),
                 os.path.join(preprocessed_path, 'X_col'))
    shutil.copy2(os.path.join(ds.path, 'X_row'),
                 os.path.join(preprocessed_path, 'X_row'))

    X = ds['X'][:]
    # this number are inferred from PBMC_CITEseq
    p_1 = np.logical_and(X > 0,
                         X < np.percentile(X[X != 0], q=70))
    p_2 = np.logical_and(X >= np.percentile(X[X != 0], q=70),
                         X < np.percentile(X[X != 0], q=87))
    X = np.where(p_1, 1, X)
    X = np.where(p_2, 2, X)
    # dropout to match sparsity of PBMC
    rand = np.random.RandomState(seed=52181208)
    mask = (rand.rand(*X.shape) >= 0.424).astype('float32')
    X = X * mask

    out = MmapData(os.path.join(preprocessed_path, 'X'),
                   dtype='float32', shape=(0, X.shape[1]),
                   read_only=False)
    out.append(X); out.flush(); out.close()

    # from data.pbmc_CITEseq import read_CITEseq_PBMC
    # ds1 = read_CITEseq_PBMC()
    # X1 = ds1['X'][:]
    # tmp = X[X != 0]
    # tmp1 = X1[X1 != 0]
    # for i in range(100):
    #   print(i, np.percentile(a=tmp, q=i), np.percentile(a=tmp1, q=i))
    # print(describe(X[X != 0]))
    # print(describe(X1[X1 != 0]))
    # print(sparsity_percentage(X))
    # print(sparsity_percentage(X1))
  # ====== return dataset ====== #
  ds = Dataset(path=preprocessed_path, read_only=True)
  return ds
