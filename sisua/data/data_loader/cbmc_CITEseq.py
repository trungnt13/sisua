from __future__ import print_function, division, absolute_import

import os
import base64
import shutil
import pickle
from io import BytesIO, StringIO

import numpy as np
import scipy as sp

from odin.fuel import MmapData, Dataset
from odin.utils import get_file, crypto, ctext, batching

from sisua.data.path import PREPROCESSED_BASE_DIR, DOWNLOAD_DIR
from sisua.data.utils import remove_allzeros_columns, save_to_dataset

_CITEseq_CBMC_PREPROCESSED = os.path.join(
    PREPROCESSED_BASE_DIR, 'CBMC_preprocessed')

_URL = b'aHR0cHM6Ly9zMy5hbWF6b25hd3MuY29tL2FpLWRhdGFzZXRzL0NCTUMuemlw\n'
_PASSWORD = 'uef-czi'

def read_CITEseq_CBMC(override=False):
  download_path = os.path.join(DOWNLOAD_DIR, "CBMC_original")
  if not os.path.exists(download_path):
    os.mkdir(download_path)

  preprocessed_path = _CITEseq_CBMC_PREPROCESSED
  if not os.path.exists(preprocessed_path):
    os.mkdir(preprocessed_path)
  elif override:
    shutil.rmtree(_CITEseq_CBMC_PREPROCESSED)
    os.mkdir(_CITEseq_CBMC_PREPROCESSED)
  # ******************** preprocessed data NOT found ******************** #
  if not os.path.exists(os.path.join(preprocessed_path, 'X')):
    X, X_row, X_col = [], None, None
    y, y_row, y_col = [], None, None
    # ====== download the data ====== #
    url = str(base64.decodebytes(_URL), 'utf-8')
    base_name = os.path.basename(url)
    get_file(fname=base_name, origin=url, outdir=download_path)
    zip_path = os.path.join(download_path, base_name)
    # ====== extract the data ====== #
    data_dict = {}
    for name, data in crypto.unzip_aes(zip_path, password=_PASSWORD,
                                       verbose=True):
      base_name = os.path.splitext(name)[0]
      if '.npz' in name:
        data = sp.sparse.load_npz(BytesIO(data)).todense()
      elif '.csv' in name:
        data = np.loadtxt(StringIO(str(data, 'utf-8')),
                          dtype=str, delimiter=',')
      else:
        raise RuntimeError("Unknown format: %s" % name)
      data_dict[base_name] = data
    # ====== post-processing ====== #
    X = data_dict['X'].astype('float32')
    X_row, X_col = data_dict['X_row'], data_dict['X_col']
    X, X_col = remove_allzeros_columns(matrix=X, colname=X_col)
    assert len(X_row) == X.shape[0] and len(X_col) == X.shape[1]

    y = data_dict['y'].astype('float32')
    y_row, y_col = data_dict['y_row'], data_dict['y_col']
    assert len(y_row) == y.shape[0] and len(y_col) == y.shape[1]

    assert np.all(X_row == y_row), \
    "Cell order mismatch between gene count and protein count"

    # save data
    print("Saving data to %s ..." %
      ctext(preprocessed_path, 'cyan'))

    save_to_dataset(preprocessed_path,
                    X, X_col, y, y_col,
                    rowname=X_row)
  # ====== read preprocessed data ====== #
  ds = Dataset(preprocessed_path, read_only=True)
  return ds
