from __future__ import print_function, division, absolute_import

import os
import shutil
import base64
import pickle

import numpy as np

from odin.utils import get_file
from odin.fuel import Dataset

from sisua.data.path import DOWNLOAD_DIR, PREPROCESSED_BASE_DIR
from sisua.data.utils import save_to_dataset, remove_allzeros_columns

_URL_LYMPHOID = b'aHR0cHM6Ly9zMy5hbWF6b25hd3MuY29tL2FpLWRhdGFzZXRzL3BibWNlY2NfbHkubnB6\n'
_URL_MYELOID = None
_URL_FULL = None

def read_PBMCeec(subset, override=False, filtered_genes=False):
  subset = str(subset).strip().lower()
  if subset not in ('ly', 'my', 'full'):
    raise ValueError("subset can only be 'ly'-lymphoid and 'my'-myeloid or 'full'")
  if subset in ('my', 'full'):
    raise NotImplementedError("No support for subset: %s - PBMCecc" % subset)

  download_path = os.path.join(DOWNLOAD_DIR, "PBMCecc_%s_original" % subset)
  if not os.path.exists(download_path):
    os.mkdir(download_path)

  preprocessed_path = os.path.join(
      PREPROCESSED_BASE_DIR,
      'PBMCecc_%s_preprocessed' % (subset + ('' if filtered_genes else 'full')))

  if override and os.path.exists(preprocessed_path):
    shutil.rmtree(preprocessed_path)
  if not os.path.exists(preprocessed_path):
    os.mkdir(preprocessed_path)

  # ******************** preprocessed ******************** #
  if not os.path.exists(os.path.join(preprocessed_path, 'X')):
    # ====== full ====== #
    if subset == 'full':
      pass
    # ====== ly and my ====== #
    else:
      url = str(base64.decodebytes(
          _URL_LYMPHOID if subset == 'ly' else _URL_MYELOID), 'utf-8')
      base_name = os.path.basename(url)
      get_file(fname=base_name, origin=url, outdir=download_path)
      # ====== extract the data ====== #
      data = np.load(os.path.join(download_path, base_name))
      X_row = data['X_row']
      y = data['y']
      y_col = data['y_col']
      if filtered_genes:
        X = data['X_var']
        X_col = data['X_var_col']
      else:
        X = data['X_full']
        X_col = data['X_full_col']
      cell_types = None

    # ====== save everything ====== #
    X, X_col = remove_allzeros_columns(matrix=X, colname=X_col,
                                       print_log=True)
    assert X.shape == (len(X_row), len(X_col))
    assert len(X) == len(y)
    assert y.shape[1] == len(y_col)

    if cell_types is not None:
      with open(os.path.join(preprocessed_path, 'cell_types'), 'wb') as f:
        pickle.dump(cell_types, f)

    save_to_dataset(preprocessed_path, X, X_col, y, y_col,
                    rowname=X_row)

  # ******************** read preprocessed data ******************** #
  ds = Dataset(preprocessed_path, read_only=True)
  return ds
