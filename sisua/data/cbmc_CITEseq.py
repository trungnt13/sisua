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

_CITEseq_CBMC_PREPROCESSED = os.path.join(
    PREPROCESSED_BASE_DIR, 'CBMC_preprocessed')

_URL = b'aHR0cHM6Ly9zMy5hbWF6b25hd3MuY29tL2FpLWRhdGFzZXRzL0NCTUMuemlw\n'
_MD5 = '5f6c756e7bd938703c57fc27f83ce903c58d10fc1fb10531c76692' + \
       'f0766c16abf41bcbeebcc25c96b692febf45bbfc360fff0f8b5d1f' + \
       'e90d45e99c2c53af613d683e3c5d988bba71d858c5f10d06fc7c'
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
    assert len(X_row) == X.shape[0] and len(X_col) == X.shape[1]

    y = data_dict['y'].astype('float32')
    y_row, y_col = data_dict['y_row'], data_dict['y_col']
    assert len(y_row) == y.shape[0] and len(y_col) == y.shape[1]

    assert np.all(X_row == y_row), \
    "Cell order mismatch between gene count and protein count"

    # save data
    print("Saving data to %s ..." %
      ctext(preprocessed_path, 'cyan'))
    out = MmapData(os.path.join(preprocessed_path, 'X'),
                   dtype='float32', shape=(0, X.shape[1]),
                   read_only=False)
    for start, end in batching(batch_size=1024, n=X.shape[0]):
      x = X[start:end]
      out.append(x)
    out.flush()
    out.close()
    with open(os.path.join(preprocessed_path, 'y'), 'wb') as f:
      pickle.dump(y, f)
    # save the meta info
    with open(os.path.join(preprocessed_path, 'X_row'), 'wb') as f:
      pickle.dump(X_row, f)
    with open(os.path.join(preprocessed_path, 'X_col'), 'wb') as f:
      pickle.dump(X_col, f)
    with open(os.path.join(preprocessed_path, 'y_col'), 'wb') as f:
      pickle.dump(y_col, f)
  # ====== read preprocessed data ====== #
  ds = Dataset(preprocessed_path,
               read_only=True)
  # y_bin, and y_prob files are generated later
  if ds.get_md5_checksum(excluded_name=['y_bin', 'y_prob']) != _MD5:
    ds.close()
    shutil.rmtree(preprocessed_path)
    raise RuntimeError("Invalid MD5 checksum, removed dataset at: %s" %
                       preprocessed_path)
  return ds