import os
import shutil
import pickle
import base64
import zipfile
from io import BytesIO

import numpy as np

from odin.fuel import Dataset, MmapData
from odin.utils import ctext, get_file, batching, select_path
from odin.utils.crypto import decrypt_aes, md5_checksum

from sisua.data.path import PREPROCESSED_BASE_DIR, DOWNLOAD_DIR
from sisua.data.utils import remove_allzeros_columns, save_to_dataset

# ===========================================================================
# Const
# ===========================================================================
# top 5000 variable genes
_URL_5000 = b'aHR0cHM6Ly9zMy5hbWF6b25hd3MuY29tL2FpLWRhdGFzZXRzL0dTRTEwMDg2Nl9QQk1DLnJhd0Nv\ndW50RGF0YS41MDAwLmh2Zy5jc3Yuemlw\n'
_MD5_5000 = '46150f63e5a3c81d4f07445a759faa2b'

# raw Count Gene
_URL_FULL = b'aHR0cHM6Ly9zMy5hbWF6b25hd3MuY29tL2FpLWRhdGFzZXRzL0dTRTEwMDg2Nl9QQk1DLnJhd0Nv\ndW50RGF0YS5jc3Yuemlw\n'
_MD5_FULL = '7481cc9d20adef4d06fdb601d9d99e77'

# protein
_URL_PROTEIN = b'aHR0cHM6Ly9zMy5hbWF6b25hd3MuY29tL2FpLWRhdGFzZXRzL0dTRTEwMDg2Nl9QQk1DLnJhd0Nv\ndW50UHJvdGVpbi5jc3Yuemlw\n'
_MD5_PROTEIN = '7dc5f64c2916d864568f1b739679717e'

_CITEseq_PBMC_PREPROCESSED = select_path(
    os.path.join(PREPROCESSED_BASE_DIR, 'PBMC_citeseq_preprocessed'),
    create_new=True)
_5000_PBMC_PREPROCESSED = select_path(
    os.path.join(PREPROCESSED_BASE_DIR, 'PBMC_citeseq_5000_preprocessed'),
    create_new=True)

_PASSWORD = 'uef-czi'

# ===========================================================================
# Main
# ===========================================================================
def read_CITEseq_PBMC(override=False, version_5000genes=False):
  download_path = os.path.join(
      DOWNLOAD_DIR,
      "PBMC_%s_original" % ('5000' if version_5000genes else 'CITEseq'))
  if not os.path.exists(download_path):
    os.mkdir(download_path)

  preprocessed_path = (_5000_PBMC_PREPROCESSED if version_5000genes else
                       _CITEseq_PBMC_PREPROCESSED)
  if override:
    shutil.rmtree(preprocessed_path)
    os.mkdir(preprocessed_path)
  # ******************** preprocessed data NOT found ******************** #
  if not os.path.exists(os.path.join(preprocessed_path, 'X')):
    X, X_row, X_col = [], None, None
    y, y_row, y_col = [], None, None
    # ====== download the data ====== #
    download_files = {}
    for url, md5 in zip([_URL_5000 if version_5000genes else _URL_FULL, _URL_PROTEIN],
                        [_MD5_5000 if version_5000genes else _MD5_FULL, _MD5_PROTEIN]):
      url = str(base64.decodebytes(url), 'utf-8')
      base_name = os.path.basename(url)
      get_file(fname=base_name, origin=url, outdir=download_path)
      download_files[base_name] = (os.path.join(download_path, base_name), md5)
    # ====== extract the data ====== #
    n = set()
    for name, (path, md5) in sorted(download_files.items()):
      print("Extracting %s ..." % ctext(name, 'lightcyan'))
      binary_data = decrypt_aes(path, password=_PASSWORD)
      md5_ = md5_checksum(binary_data)
      assert md5_ == md5, "MD5 checksum mismatch for file: %s" % name
      with zipfile.ZipFile(file=BytesIO(binary_data), mode='r') as f:
        for name in f.namelist():
          data = str(f.read(name), 'utf8')
          for line in data.split('\n'):
            if len(line) == 0:
              continue
            line = line.strip().split(',')
            n.add(len(line))
            if 'Protein' in name:
              y.append(line)
            else:
              X.append(line)
    # ====== post-processing ====== #
    assert len(n) == 1, \
    "Number of samples inconsistent between raw count and protein count"
    print("Processing %s ..." % ctext("gene count", 'cyan'))
    X = np.array(X).T
    X_row, X_col = X[1:, 0], X[0, 1:]
    X = X[1:, 1:].astype('float32')

    # ====== filter mouse genes ====== #
    human_cols = [True if "HUMAN_" in i else False for i in X_col]
    print("Removing %s MOUSE genes ..." %
          ctext(np.sum(np.logical_not(human_cols)), 'cyan'))
    X = X[:, human_cols]
    X_col = np.array([i for i in X_col if "HUMAN_" in i])
    X, X_col = remove_allzeros_columns(matrix=X, colname=X_col,
                                       print_log=True)

    # ====== protein ====== #
    print("Processing %s ..." % ctext("protein count", 'cyan'))
    y = np.array(y).T
    y_row, y_col = y[1:, 0], y[0, 1:]
    y = y[1:, 1:].astype('float32')

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
