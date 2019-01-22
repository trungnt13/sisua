import os
import math
import pickle
from six import string_types

import numpy as np

from odin.utils import ctext
from odin.fuel import MmapData, Dataset
from odin.stats import sparsity_percentage

def filtering_experiment_path(ds_name_or_path, incl_keywords, excl_keywords,
                              return_dataset=False, print_log=False):
  """

  Parameters
  ----------
  ds_name_or_path : string
      direct path to experiments folder or name of the dataset

  incl_keywords : string
      list of keywords for including the experiments (connect by ',')

  excl_keywords : string
      list of keywords for excluding the experiments (connect by ',')

  Return
  ------
  list of absolute path to all satisfied experiments

  Note
  ----
  only finished experiments are select, i.e. the experiment folder contain
  2 files 'config.pkl' and 'model.pkl'

  """
  from sisua.data import EXP_DIR, get_dataset
  # ====== get the exp path ====== #
  ds_name_or_path = str(ds_name_or_path)
  if os.path.isdir(ds_name_or_path):
    exp_path = ds_name_or_path
    (ds, ds_name_or_path, _, gene_ds, prot_ds) = get_dataset(
        os.path.dirname(ds_name_or_path))
  else:
    (ds, ds_name_or_path, _, gene_ds, prot_ds) = get_dataset(ds_name_or_path)
    exp_path = os.path.join(EXP_DIR, ds_name_or_path)
  assert os.path.isdir(exp_path), exp_path
  # ====== Extract all experiments ====== #
  all_exp = [os.path.join(exp_path, i)
             for i in os.listdir(exp_path)]
  # ====== start filtering ====== #
  incl_keywords = [i for i in str(incl_keywords).split(',') if len(i) > 0]
  excl_keywords = [i for i in str(excl_keywords).split(',') if len(i) > 0]

  all_exp = [i for i in all_exp
             if all(j in os.path.basename(i).split('_')
                    for j in incl_keywords)]

  all_exp = [i for i in all_exp
             if all(j not in os.path.basename(i).split('_')
                    for j in excl_keywords)]

  # ====== check if experiments finished ====== #
  all_exp = [i for i in all_exp
             if os.path.exists(os.path.join(i, 'config.pkl')) and
             os.path.exists(os.path.join(i, 'model.pkl'))]
  all_exp = sorted(all_exp,
                   key=lambda x: os.path.basename(x))
  # ====== logging ====== #
  if bool(print_log):
    print(ctext("Found following experiments:", 'lightyellow'))
    for i in all_exp:
      print('*', os.path.basename(i))

  if return_dataset:
    return all_exp, ds_name_or_path, gene_ds, prot_ds
  return all_exp

def anything2image(x):
  if x.ndim == 1:
    img_dim = int(math.ceil(math.sqrt(x.shape[0])))
    z = np.zeros(shape=img_dim * img_dim, dtype=x.dtype)
    z[:x.shape[0]] = x
    x = z.reshape(img_dim, img_dim)
  elif x.ndim == 2:
    pass
  elif x.ndim == 3:
    pass
  else:
    raise ValueError("No support for image with %d dimensions" % x.ndim)
  return x

# ===========================================================================
# For thresholding based on validation dataset
# ===========================================================================
def apply_threshold(x, threshold):
  """
  x = 0 if x < threshold
  elif threshold <= x < 1, x = 1
  otherwise, x
  """
  x = np.where(x < threshold, 0, x)
  x = np.where(np.logical_and(0 < x, x < 1), 1, x).astype('int32')
  return x

def thresholding_by_sparsity_matching(T, W, *applying_data):
  """
  T : original count
  W : reconstructed count
  """
  T = T.astype('int32')
  W = W[0] if isinstance(W, (tuple, list)) else W
  if W.ndim == 3:
    W = W[0]
  assert W.ndim == 2

  best_threshold = None
  nonzeros = T != 0
  # this is just matching the sparsity
  for threshold in np.linspace(0, 1, num=100, endpoint=True)[::-1]:
    tmp = W >= threshold
    if np.sum(tmp) >= np.sum(nonzeros):
      best_threshold = threshold
      break
  # ====== applying the threshold ====== #
  new_data = []
  for data in applying_data:
    if data is None:
      new_data.append(None)
    else:
      if isinstance(data, tuple):
        data = list(data)
      if isinstance(data, list) or data.ndim == 3:
        data[0] = apply_threshold(data[0], threshold=best_threshold)
      else:
        data = apply_threshold(data, threshold=best_threshold)
      new_data.append(data)
  return best_threshold, tuple(new_data)

# ===========================================================================
# For data preprocessing
# ===========================================================================
def remove_allzeros_columns(matrix, colname, print_log=True):
  assert matrix.ndim == 2
  orig_shape = matrix.shape
  nonzero_col = np.sum(matrix, axis=0) != 0
  matrix = matrix[:, nonzero_col]
  colname = colname[nonzero_col]
  if print_log:
    print("Filtering %d all-zero columns from data: %s -> %s ..." %
      (len(nonzero_col) - np.sum(nonzero_col),
       str(orig_shape),
       str(matrix.shape)))
  return matrix, colname

def _check_data(X, X_col, y, y_col, rowname):
  assert X.min() >= 0, "Only support non-negative value for X"
  assert y.min() >= 0, "Only support non-negative value for y"

  assert X.ndim == 2 and y.ndim == 2, "Only support matrix for `X` and `y`"
  assert X.shape[0] == y.shape[0], \
  "Number of sample mismatch `X=%s` and `y=%s`" % (X.shape, y.shape)

  assert X_col.ndim == 1 and len(X_col) == X.shape[1]
  assert y_col.ndim == 1 and len(y_col) == y.shape[1]

  assert rowname.ndim == 1 and len(rowname) == X.shape[0] == y.shape[0]

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

def save_data_to_dataset(path, X, X_col, y, y_col,
                         rowname, print_log=True):
  _check_data(X, X_col, y, y_col, rowname)
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
