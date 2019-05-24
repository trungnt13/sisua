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
from sisua.data.utils import save_to_dataset, remove_allzeros_columns

# ===========================================================================
# Constants
# ===========================================================================
# Protein
_URL_LYMPHOID = b'aHR0cHM6Ly9zMy5hbWF6b25hd3MuY29tL2FpLWRhdGFzZXRzL3BibWM4a19seS5ucHo=\n'
_URL_MYELOID = b'aHR0cHM6Ly9zMy5hbWF6b25hd3MuY29tL2FpLWRhdGFzZXRzL3BibWM4a19teS5ucHo=\n'
_URL_PBMC8k = b'aHR0cHM6Ly9zMy5hbWF6b25hd3MuY29tL2FpLWRhdGFzZXRzL3BibWM4a19mdWxsLm5weg==\n'

# ===========================================================================
# Main
# ===========================================================================
def read_PBMC8k(subset, override=False, filtered_genes=False):
  subset = str(subset).strip().lower()
  if subset not in ('ly', 'my', 'full'):
    raise ValueError("subset can only be 'ly'-lymphoid and 'my'-myeloid or 'full'")

  download_path = os.path.join(DOWNLOAD_DIR, "PBMC8k_%s_original" % subset)
  if not os.path.exists(download_path):
    os.mkdir(download_path)

  preprocessed_path = os.path.join(
      PREPROCESSED_BASE_DIR,
      'PBMC8k_%s_preprocessed' % (subset + ('' if filtered_genes else 'full')))

  if override and os.path.exists(preprocessed_path):
    shutil.rmtree(preprocessed_path)
  if not os.path.exists(preprocessed_path):
    os.mkdir(preprocessed_path)

  # ******************** preprocessed ******************** #
  if not os.path.exists(os.path.join(preprocessed_path, 'X')):
    # ====== pbmc 8k ====== #
    if subset == 'full':
      ly = read_PBMC8k('ly', override=override, filtered_genes=filtered_genes)
      my = read_PBMC8k('my', override=override, filtered_genes=filtered_genes)

      url = str(base64.decodebytes(_URL_PBMC8k), 'utf-8')
      base_name = os.path.basename(url)
      get_file(fname=base_name, origin=url, outdir=download_path)

      data = np.load(os.path.join(download_path, base_name))
      X = data['X']
      X_row = data['X_row']
      X_col = data['X_col'].tolist()
      y = data['y']
      y_col = data['y_col'].tolist()

      all_genes = set(ly['X_col'].tolist() + my['X_col'].tolist())
      all_genes = sorted([X_col.index(i) for i in all_genes])

      all_proteins = set(ly['y_col'].tolist() + my['y_col'].tolist())
      all_proteins = sorted([y_col.index(i) for i in all_proteins])

      X = X[:, all_genes]
      y = y[:, all_proteins]
      X_col = np.array(X_col)[all_genes]
      y_col = np.array(y_col)[all_proteins]
      cell_types = np.array(
          ['ly' if i in ly['X_row'] else 'my'
           for i in X_row])
    # ====== pbmc ly and my ====== #
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
        X = data['X_filt']
        X_col = data['X_filt_col']
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
