import os
import re
import math
import pickle
from six import string_types
from collections import OrderedDict, defaultdict

import numpy as np

from odin.utils import ctext, as_tuple
from odin.fuel import MmapData, Dataset
from odin.stats import sparsity_percentage

def filtering_experiment_path(ds_name, incl_keywords, excl_keywords,
                              fn_filter=None,
                              return_dataset=False, print_log=False,
                              exp_path=''):
  """

  Parameters
  ----------
  ds_name : string
      direct path to experiments folder or name of the dataset

  incl_keywords : string
      list of keywords for including the experiments (connect by ',')

  excl_keywords : string
      list of keywords for excluding the experiments (connect by ',')

  exp_path : string
      optional, if not given, use SISUA_EXP

  Return
  ------
  dictionary
  corruption_config -> list of absolute path to all satisfied experiments

  Note
  ----
  only finished experiments are select, i.e. the experiment folder contain
  2 files 'config.pkl' and 'model.pkl'

  """
  from sisua.data import EXP_DIR, get_dataset
  ds_name = str(ds_name)
  if exp_path is None:
    exp_path = ''
  exp_path = str(exp_path)
  if len(exp_path) == 0:
    exp_path = EXP_DIR
  assert os.path.isdir(exp_path), exp_path
  # ====== check the keywords ====== #
  if incl_keywords is None:
    incl_keywords = []
  if excl_keywords is None:
    excl_keywords = []
  if fn_filter is None:
    fn_filter = lambda keywords: True
  # ====== get the exp path ====== #
  if ds_name is None or return_dataset:
    (ds, gene_ds, prot_ds) = get_dataset(ds_name)
    ds_name = ds.name
  exp_path = os.path.join(exp_path, ds_name)
  assert os.path.exists(exp_path), "Experiment path '%s' must exists" % exp_path
  # ====== Extract all experiments ====== #
  all_exp = []
  for name in os.listdir(exp_path):
    path = os.path.join(exp_path, name)
    # check if experiments finished
    if os.path.exists(os.path.join(path, 'model.pkl')):
      all_exp.append(path)
  all_exp = sorted(all_exp)
  # ====== start filtering ====== #
  if isinstance(incl_keywords, string_types):
    incl_keywords = [i for i in str(incl_keywords).split(',') if len(i) > 0]
  elif isinstance(incl_keywords, (tuple, list)):
    incl_keywords = as_tuple(incl_keywords, t=str)
  else:
    raise ValueError("No support for incl_keywords type: %s" % str(type(incl_keywords)))

  if isinstance(excl_keywords, string_types):
    excl_keywords = [i for i in str(excl_keywords).split(',') if len(i) > 0]
  elif isinstance(excl_keywords, (tuple, list)):
    excl_keywords = as_tuple(excl_keywords, t=str)
  else:
    raise ValueError("No support for excl_keywords type: %s" % str(type(excl_keywords)))

  all_exp = [i for i in all_exp
             if all(any(j in keyword
                        for keyword in os.path.basename(i).split('_'))
                    for j in incl_keywords)]
  all_exp = [i for i in all_exp
             if all(all(j not in keyword
                        for keyword in os.path.basename(i).split('_'))
                    for j in excl_keywords)]

  # filter function
  all_exp = [i for i in all_exp
             if fn_filter(os.path.basename(i).split('_'))]
  # ====== logging ====== #
  if bool(print_log):
    print(ctext("Found following experiments:", 'lightyellow'))
    for name, paths in all_exp.items():
      print("*", ctext(name, 'yellow'))
      for i in paths:
        print('  ', os.path.basename(i))

  if return_dataset:
    return all_exp, ds, gene_ds, prot_ds
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
