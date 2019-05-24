from __future__ import print_function, division, absolute_import

import os
import pickle
from six import string_types

import numpy as np
from odin.utils import ctext, as_tuple
from odin.fuel import MmapData, Dataset

# ===========================================================================
# Helpers
# ===========================================================================
def _check_data(X, X_col, y, y_col, rowname):
  assert X.min() >= 0, "Only support non-negative value for X"
  assert y.min() >= 0, "Only support non-negative value for y"

  assert X.ndim == 2 and y.ndim == 2, "Only support matrix for `X` and `y`"
  assert X.shape[0] == y.shape[0], \
  "Number of sample mismatch `X=%s` and `y=%s`" % (X.shape, y.shape)

  assert X_col.ndim == 1 and len(X_col) == X.shape[1]
  assert y_col.ndim == 1 and len(y_col) == y.shape[1]

  assert rowname.ndim == 1 and len(rowname) == X.shape[0] == y.shape[0]

def read_gzip_csv(path):
  import gzip
  with gzip.open(path, 'rb') as file_obj:
    data = []
    for line in file_obj:
      line = str(line, 'utf-8').strip()
      line = line.split(',')
      data.append(line)
    data = np.array(data)
    return data

# ===========================================================================
# Some normalization for meta-data
# ===========================================================================
_protein_name = {
    "PD-L1;CD274": "CD274",
    "PECAM;CD31": "CD31",
    "CD26;Adenosine": "CD26",
    "CD366;tim3": "CD366",
    "MHCII;HLA-DR": "MHCII",
    "IL7Ralpha;CD127": "CD127",
    "PD-1": "PD-1", # CD279
    "PD1": "PD1",
    "B220;CD45R": "CD45R",
    "Ox40;CD134": "CD134",
    "CD8a": "CD8",
    "CD8A": "CD8",
}
def standardize_protein_name(name):
  """ standardize """
  assert isinstance(name, string_types), "Protein name must be string types"
  name = name.replace('-TotalSeqB', '')
  name = name.strip()
  # regular expression could be used but it could brutally remove
  # a lot of things without our notice
  if name in _protein_name:
    name = _protein_name[name]
  return name

# ===========================================================================
# Utilities
# ===========================================================================
def remove_allzeros_columns(matrix, colname, print_log=True):
  """ Remove all zero columns from both the matrix and column name vector

  Return
  ------
  matrix : [n_samples, n_genes]
  column_name : [n_genes]
  """
  assert matrix.ndim == 2
  orig_shape = matrix.shape
  # at least > 1 for train, test splitting
  nonzero_col = np.sum(matrix, axis=0) > 1
  matrix = matrix[:, nonzero_col]
  colname = colname[nonzero_col]
  if print_log:
    print("Filtering %d all-zero columns from data: %s -> %s ..." %
      (len(nonzero_col) - np.sum(nonzero_col),
       str(orig_shape),
       str(matrix.shape)))
  return matrix, colname

def validating_dataset(path):
  if isinstance(path, Dataset):
    ds = path
  elif isinstance(path, string_types):
    ds = Dataset(path, read_only=True)
  assert 'X' in ds, \
  '`X` (n_samples, n_genes) must be stored at path: %s' % ds.path
  assert 'X_col' in ds, \
  '`X_col` (n_genes,) must be stored at path: %s' % ds.path
  assert 'X_row' in ds, \
  '`X_row` (n_samples,) must be stored at path: %s' % ds.path

  assert 'y' in ds, \
  '`y` (n_samples, n_protein) must be stored at path: %s' % ds.path
  assert 'y_col' in ds, \
  '`y_col` (n_protein,) must be stored at path: %s' % ds.path
  X, X_col, y, y_col, rowname = \
  ds['X'], ds['X_col'], ds['y'], ds['y_col'], ds['X_row']
  _check_data(X, X_col, y, y_col, rowname)

def save_to_dataset(path, X, X_col, y, y_col,
                    rowname, print_log=True):
  """
  path : output folder path
  X : (n_samples, n_genes) gene expression matrix
  X_col : (n_genes,) name of each gene
  y : (n_samples, n_proteins) protein marker level matrix
  y_col : (n_proteins) name of each protein
  rowname : (n_samples,) name of cells (i.e. the sample)
  print_log : bool (default: True)
  """
  _check_data(X, X_col, y, y_col, rowname)
  assert os.path.isdir(path), "'%s' must be path to a folder" % path
  # save data
  if print_log:
    print("Saving data to %s ..." % ctext(path, 'cyan'))
  out = MmapData(os.path.join(path, 'X'),
                 dtype='float32', shape=(0, X.shape[1]),
                 read_only=False)
  out.append(X)
  out.flush()
  out.close()
  with open(os.path.join(path, 'y'), 'wb') as f:
    pickle.dump(y, f)
  # save the meta info
  with open(os.path.join(path, 'X_col'), 'wb') as f:
    pickle.dump(X_col, f)
  with open(os.path.join(path, 'y_col'), 'wb') as f:
    pickle.dump(y_col, f)
  # row name for both X and y
  with open(os.path.join(path, 'X_row'), 'wb') as f:
    pickle.dump(rowname, f)
