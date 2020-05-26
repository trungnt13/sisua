from __future__ import absolute_import, division, print_function

import base64
import gzip
import os
import pickle
import shutil
import tarfile
import warnings
import zipfile
from contextlib import contextmanager
from copy import deepcopy
from urllib.request import urlretrieve

import numpy as np
from scipy import sparse
from six import string_types

from bigarray import MmapArrayWriter
from odin.fuel import Dataset
from odin.utils import as_tuple, ctext
from odin.utils.crypto import md5_checksum, md5_folder

__all__ = [
    'apply_artificial_corruption',
    'get_library_size',
    'read_compressed',
    'standardize_protein_name',
    'get_gene_id2name',
    'validating_dataset',
    'save_to_dataset',
]


# ===========================================================================
# For reading compressed files
# ===========================================================================
def validate_data_dir(path_dir, md5):
  if not os.path.exists(path_dir):
    os.makedirs(path_dir)
  elif md5_folder(path_dir) != md5:
    shutil.rmtree(path_dir)
    print(f"MD5 preprocessed at {path_dir} mismatch, remove and override!")
    os.makedirs(path_dir)
  return path_dir


def download_file(url, filename, override, md5=None):
  f""" Download file and check the MD5 (if provided) """
  if md5 is None:
    md5 = r""
  from tqdm import tqdm
  if os.path.exists(filename) and os.path.isfile(filename):
    if override:
      os.remove(filename)
    if len(md5) > 0:
      if md5 == md5_checksum(filename):
        return filename
      else:
        print(f"MD5 of file {filename} mismatch, re-download the file.")
        os.remove(filename)
    else:  # no MD5 provide just ignore the download if file exist
      return filename
  prog = [None]

  def progress(blocknum, blocksize, total):
    if prog[0] is None:
      prog[0] = tqdm(desc=f"Download {os.path.basename(filename)}",
                     total=int(total / 1024. / 1024.) + 1,
                     unit="MB")
    prog[0].update(int(blocknum * blocksize / 1024. / 1024.) - prog[0].n)

  urlretrieve(url=url, filename=filename, reporthook=progress)
  prog[0].clear()
  prog[0].close()
  print(f"File '{filename}' md5:{md5_checksum(filename)}")
  return filename


def read_r_matrix(matrix):
  r"""Convert (and transpose) a dgCMatrix from R to a csr_matrix in python

  Author:
    Isaac Virshup (ivirshup)
      https://github.com/theislab/anndata2ri/issues/8#issuecomment-482925685
  """
  import rpy2.robjects as ro
  from rpy2.robjects import pandas2ri
  from rpy2.robjects.conversion import localconverter
  from scipy.sparse import csr_matrix
  rclass = list(matrix.rclass)[0]
  if rclass == 'dgCMatrix':
    ro.r['options'](warn=-1)
    ro.r("library(Matrix)")
    with localconverter(ro.default_converter + pandas2ri.converter):
      X = csr_matrix((matrix.slots["x"], matrix.slots["i"], matrix.slots["p"]),
                     shape=tuple(ro.r("dim")(matrix))[::-1]).T
  elif rclass == 'matrix':
    X = np.array(matrix)
  elif isinstance(matrix, ro.vectors.DataFrame):
    pandas2ri.activate()
    X = ro.conversion.rpy2py(matrix)
  else:
    raise NotImplementedError(
        f"No support for R matrix of class {rclass}-{type(matrix)}")
  return X


@contextmanager
def read_compressed(in_file,
                    md5_download=None,
                    override=False,
                    verbose=False) -> dict:
  r"""
  Return:
    a Dictionary : mapping from name of files to extracted buffer
  """
  if md5_download is None:
    md5_download = ''
  assert os.path.isfile(in_file)
  ext = os.path.splitext(in_file.lower())[-1]
  extracted_name = {}
  opened_files = []
  ### validate md5
  if len(md5_download) > 0 and md5_download != md5_checksum(in_file):
    raise RuntimeError(f"MD5 file {in_file} mismatch")
  ### '.tar' file
  if tarfile.is_tarfile(in_file):
    if ext == '.tar':
      mode = 'r'
    elif ext == '.tar.gz':
      mode = 'r:gz'
    else:
      raise NotImplementedError(f"No support for decompressing {in_file}")
    f = tarfile.open(in_file, mode=mode)
    opened_files.append(f)
    for info in f.getmembers():
      name = info.name
      data = f.extractfile(info)
      if os.path.splitext(name)[-1] == '.gz':
        data = gzip.GzipFile(mode='rb', fileobj=data)
        name = name[:-3]
      extracted_name[name] = data
  ### zip file
  elif zipfile.is_zipfile(in_file):
    raise NotImplementedError(in_file)
  ### '.gz' file
  elif ext == '.gz':
    name = os.path.basename(in_file)[:-3]
    f = gzip.open(in_file, mode='rb')
    extracted_name[name[:-3]] = f
    opened_files.append(f)
  ### '.gz' file
  else:
    raise NotImplementedError(in_file)
  ### return a dictionary of file name to path
  yield extracted_name
  for f in opened_files:
    try:
      f.close()
    except:
      pass


# ===========================================================================
# Data preprocessing
# ===========================================================================
def apply_artificial_corruption(x,
                                dropout=0.0,
                                distribution='binomial',
                                retain_rate=0.2,
                                copy=False,
                                seed=8):
  r"""
    x : (n_samples, n_features)
    dropout : scalar (0.0 - 1.0), how many entries (in percent) be selected for
      corruption.
    retain_rate : scalar (0.0 - 1.0), how much percent of counts retained
      their original values.
    distribution : {'uniform', 'binomial'}
  """
  distribution = str(distribution).lower()
  dropout = float(dropout)
  assert 0 <= dropout < 1, \
  "dropout value must be >= 0 and < 1, given: %f" % dropout
  rand = np.random.RandomState(seed=seed)

  if not (0. < dropout < 1. or 0. < retain_rate < 1.):
    return x
  # ====== applying corruption ====== #
  # Original code from scVI, to provide a comparable result,
  # please acknowledge the author of scVI if you are using this
  # code for corrupting the data
  # https://github.com/YosefLab/scVI/blob/2357dde15351450e452efa426c516c60a2d5ee96/scvi/dataset/dataset.py#L83
  # the test data won't be corrupted
  corrupted_x = deepcopy(x) if copy else x
  # multiply the entry n with a Ber(0.9) random variable.
  if distribution == "uniform":
    i, j = np.nonzero(x)
    ix = rand.choice(range(len(i)),
                     size=int(np.floor(dropout * len(i))),
                     replace=False)
    i, j = i[ix], j[ix]
    # multiply the entry n with a Bin(n, 0.9) random variable.
    corrupted = np.multiply(
        x[i, j], \
        rand.binomial(n=np.ones(len(ix), dtype=np.int32), p=retain_rate))
  elif distribution == "binomial":
    # array representing the indices of a grid
    # [0, 0, 0, ..., 1, 1, 1, ...]
    # [0, 1, 2, ..., 0, 1, 2, ...]
    # i, j = (k.ravel() for k in np.indices(x.shape))
    i, j = np.nonzero(x)  # we prefer only nonzero here
    ix = rand.choice(range(len(i)),
                     size=int(np.floor(dropout * len(i))),
                     replace=False)
    i, j = i[ix], j[ix]
    # only 20% expression captured
    corrupted = rand.binomial(n=(x[i, j]).astype(np.int32), p=retain_rate)
  else:
    raise ValueError(
        "Only support 2 corruption distribution: 'uniform' and 'binomial', "
        "but given: '%s'" % distribution)
  # applying the corrupted values
  if isinstance(corrupted_x, sparse.base.spmatrix):
    corrupted = type(corrupted_x)(corrupted)
  corrupted_x[i, j] = corrupted
  return corrupted_x


def get_library_size(X, return_log_count=False):
  r""" Copyright scVI authors
  https://github.com/YosefLab/scVI/blob/master/README.rst

  Original Code:
  https://github.com/YosefLab/scVI/blob/9d9a525df810c47ce482ef7b554f25fcc6482c2d/scvi/dataset/dataset.py#L288

  size factor of X in log-space

  Arguments:
    X : matrix
      single-cell data matrix (n_samples, n_features)
    return_log_count : bool (default=False)
      if True, return the log-count library size

  Return:
    local_mean (n_samples, 1)
    local_var (n_samples, 1)
  """
  assert X.ndim == 2, "Only support 2-D matrix"
  total_counts = X.sum(axis=1)
  if not np.all(total_counts >= 0):
    warnings.warn(
        "Some cell contains negative-count, this results NaN log counts!")
  log_counts = np.log(total_counts + 1e-8)
  local_mean = (np.mean(log_counts) * np.ones(
      (X.shape[0], 1))).astype(np.float32)
  local_var = (np.var(log_counts) * np.ones((X.shape[0], 1))).astype(np.float32)
  if not return_log_count:
    return local_mean, local_var
  if log_counts.ndim < 2:
    log_counts = np.expand_dims(log_counts, axis=-1)
  return log_counts, local_mean, local_var


# ===========================================================================
# Helpers
# ===========================================================================
def _check_data(X, X_col, y, y_col, rowname):
  if not sparse.issparse(X):
    assert np.min(X) >= 0, "Only support non-negative value for X"
  assert X_col.ndim == 1 and len(X_col) == X.shape[1]
  assert rowname.ndim == 1 and len(rowname) == X.shape[0]

  if y is not None and len(y.shape) > 0 and y.shape[1] != 0:
    if not sparse.issparse(y):
      assert np.min(y) >= 0, "Only support non-negative value for y"

    assert X.ndim == 2 and y.ndim == 2, "Only support matrix for `X` and `y`"
    assert X.shape[0] == y.shape[0], \
    "Number of sample mismatch `X=%s` and `y=%s`" % (X.shape, y.shape)
    assert y_col.ndim == 1 and len(y_col) == y.shape[1]


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
    "PD-1": "PD-1",  # CD279
    "PD1": "PD1",
    "B220;CD45R": "CD45R",
    "Ox40;CD134": "CD134",
    "CD8a": "CD8",
    "CD8A": "CD8",
    "CD4 T cells": "CD4",
    "CD8 T cells": "CD8",
}


def standardize_protein_name(name):
  r""" standardize """
  if isinstance(name, (tuple, list, np.ndarray)):
    return [standardize_protein_name(i) for i in name]
  assert isinstance(name, string_types), "Protein name must be string types"
  for i in ('-', '_'):
    for j in ('TotalSeqB', 'control', 'TotalSeqC', 'TotalSeqA'):
      name = name.replace(f'{i}{j}', '')
  name = name.strip()
  # regular expression could be used but it could brutally remove
  # a lot of things without our notice
  if name in _protein_name:
    name = _protein_name[name]
  return name


# ===========================================================================
# Gene identifier processing
# ===========================================================================
def get_gene_id2name():
  r""" Return the mapping from gene identifier to gene symbol (i.e. name)
  for PBMC 8k data
  """
  from odin.utils import get_file
  from sisua.data.path import DOWNLOAD_DIR
  url = base64.decodebytes(
      b'aHR0cHM6Ly9haS1kYXRhc2V0cy5zMy5hbWF6b25hd3MuY29tL2dlbmVfaWQybmFtZS5wa2w=\n'
  )
  url = str(url, 'utf-8')
  get_file('gene_id2name.pkl', url, DOWNLOAD_DIR)
  with open(os.path.join(DOWNLOAD_DIR, 'gene_id2name.pkl'), 'rb') as f:
    return pickle.load(f)


# ===========================================================================
# Utilities
# ===========================================================================
def is_categorical_dtype(X):
  if not isinstance(X.dtype, np.number):
    return True
  return np.all(X.astype(np.int64) == X)


def is_binary_dtype(X):
  r""" return True if the data is binary values, i.e. 0 or 1 """
  return sorted(np.unique(X.astype(np.float32))) == [0., 1.]


def remove_allzeros_columns(matrix, colname, print_log=True):
  r""" Remove all zero columns from both the matrix and column name vector

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
          (len(nonzero_col) - np.sum(nonzero_col), str(orig_shape),
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

  if 'y' in ds:
    assert 'y' in ds, \
    '`y` (n_samples, n_protein) must be stored at path: %s' % ds.path
    assert 'y_col' in ds, \
    '`y_col` (n_protein,) must be stored at path: %s' % ds.path
    y, y_col = ds['y'], ds['y_col']
  else:
    y, y_col = None, None

  X, X_col, rowname = ds['X'], ds['X_col'], ds['X_row']
  _check_data(X, X_col, y, y_col, rowname)


def save_to_dataset(path,
                    X,
                    X_col=None,
                    y=None,
                    y_col=None,
                    rowname=None,
                    print_log=True):
  r"""
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
  # saving sparse matrix
  if sparse.issparse(X):
    with open(os.path.join(path, 'X'), 'wb') as f:
      pickle.dump(X, f)
  else:
    with MmapArrayWriter(path=os.path.join(path, 'X'),
                         dtype='float32',
                         shape=(0, X.shape[1]),
                         remove_exist=True) as out:
      out.write(X)
  # save the meta info (X features)
  if X_col is not None:
    with open(os.path.join(path, 'X_col'), 'wb') as f:
      pickle.dump(X_col, f)
  # saving the label data (can be continous or discrete or binary)
  if y is not None and len(y.shape) > 0 and y.shape[1] != 0:
    if sparse.issparse(y):
      with open(os.path.join(path, 'y'), 'wb') as f:
        pickle.dump(y, f)
    else:
      with MmapArrayWriter(path=os.path.join(path, 'y'),
                           dtype='float32',
                           shape=(0, y.shape[1]),
                           remove_exist=True) as out:
        out.write(y)
    with open(os.path.join(path, 'y_col'), 'wb') as f:
      pickle.dump(y_col, f)
  # row name for both X and y
  if rowname is not None:
    with open(os.path.join(path, 'X_row'), 'wb') as f:
      pickle.dump(rowname, f)
