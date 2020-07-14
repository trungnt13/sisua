from __future__ import absolute_import, division, print_function

import base64
import os
import pickle
import shutil

import numpy as np

from odin.fuel import Dataset
from odin.utils import get_file
from sisua.data.path import DATA_DIR, DOWNLOAD_DIR
from sisua.data.single_cell_dataset import SingleCellOMIC
from sisua.data.utils import (download_file, remove_allzeros_columns,
                              save_to_dataset)

_URL_LYMPHOID = b'aHR0cHM6Ly9zMy5hbWF6b25hd3MuY29tL2FpLWRhdGFzZXRzL3BibWNlY2NfbHkubnB6\n'
_URL_MYELOID = None
_URL_FULL = None


def read_PBMCeec(subset='ly',
                 override=False,
                 verbose=True,
                 filtered_genes=True) -> SingleCellOMIC:
  subset = str(subset).strip().lower()
  if subset not in ('ly', 'my', 'full'):
    raise ValueError(
        "subset can only be 'ly'-lymphoid and 'my'-myeloid or 'full'")
  if subset in ('my', 'full'):
    raise NotImplementedError("No support for subset: %s - PBMCecc" % subset)
  download_path = os.path.join(DOWNLOAD_DIR, "PBMCecc_%s_original" % subset)
  if not os.path.exists(download_path):
    os.mkdir(download_path)
  preprocessed_path = os.path.join(
      DATA_DIR,
      f"PBMCecc_{subset}_{'filtered' if filtered_genes else 'all'}_preprocessed"
  )
  if override and os.path.exists(preprocessed_path):
    shutil.rmtree(preprocessed_path)
    if verbose:
      print(f"Override preprocessed data at path {preprocessed_path}")
  if not os.path.exists(preprocessed_path):
    os.mkdir(preprocessed_path)
  # ******************** preprocessed ******************** #
  if not os.path.exists(os.path.join(preprocessed_path, 'X')):
    # ====== full ====== #
    if subset == 'full':
      raise NotImplementedError
    # ====== ly and my ====== #
    else:
      url = str(
          base64.decodebytes(_URL_LYMPHOID if subset == 'ly' else _URL_MYELOID),
          'utf-8')
      base_name = os.path.basename(url)
      path = os.path.join(download_path, base_name)
      download_file(filename=path, url=url, override=False)
      # ====== extract the data ====== #
      data = np.load(path)
      X_row = data['X_row']
      y = data['y']
      y_col = data['y_col']
      if filtered_genes:
        X = data['X_var']
        X_col = data['X_var_col']
      else:
        X = data['X_full']
        X_col = data['X_full_col']
      cell_types = np.array(['ly'] * X.shape[0])
    # ====== save everything ====== #
    X, X_col = remove_allzeros_columns(matrix=X,
                                       colname=X_col,
                                       print_log=verbose)
    assert X.shape == (len(X_row), len(X_col))
    assert len(X) == len(y)
    assert y.shape[1] == len(y_col)
    with open(os.path.join(preprocessed_path, 'cell_types'), 'wb') as f:
      pickle.dump(cell_types, f)
    save_to_dataset(preprocessed_path,
                    X,
                    X_col,
                    y,
                    y_col,
                    rowname=X_row,
                    print_log=verbose)
  # ******************** read preprocessed data ******************** #
  ds = Dataset(preprocessed_path, read_only=True)
  sco = SingleCellOMIC(X=ds['X'],
                       cell_id=ds['X_row'],
                       gene_id=ds['X_col'],
                       omic='transcriptomic',
                       name=f"ecc{subset}{'' if filtered_genes else 'all'}")
  sco.add_omic('proteomic', X=ds['y'], var_names=ds['y_col'])
  progenitor = ds['cell_types']
  sco.add_omic(
      'progenitor',
      X=np.array([(1, 0) if i == 'my' else (0, 1) for i in progenitor],
                 dtype=np.float32),
      var_names=np.array(['myeloid', 'lymphoid']),
  )
  return sco
