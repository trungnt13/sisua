import base64
import os
import pickle
import shutil
from io import BytesIO

import numpy as np

from odin.fuel import Dataset
from odin.utils import batching, ctext, get_file, select_path
from odin.utils.crypto import decrypt_aes, md5_checksum
from sisua.data.path import DATA_DIR, DOWNLOAD_DIR
from sisua.data.single_cell_dataset import SingleCellOMIC
from sisua.data.utils import (download_file, remove_allzeros_columns,
                              save_to_dataset)

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
def read_PBMC8k(subset='full',
                override=False,
                verbose=True,
                filtered_genes=True,
                return_arrays=False) -> SingleCellOMIC:
  subset = str(subset).strip().lower()
  if subset not in ('ly', 'my', 'full'):
    raise ValueError(
        "subset can only be 'ly'-lymphoid and 'my'-myeloid or 'full'")
  # prepare the path
  download_path = os.path.join(DOWNLOAD_DIR, f"PBMC8k_{subset}_original")
  if not os.path.exists(download_path):
    os.mkdir(download_path)
  preprocessed_path = os.path.join(
      DATA_DIR,
      f"PBMC8k_{subset}_{'filtered' if filtered_genes else 'all'}_preprocessed")
  if override and os.path.exists(preprocessed_path):
    shutil.rmtree(preprocessed_path)
  if not os.path.exists(preprocessed_path):
    os.mkdir(preprocessed_path)
  # ******************** preprocessed ******************** #
  if len(os.listdir(preprocessed_path)) == 0:
    # ====== pbmc 8k ====== #
    if subset == 'full':
      ly = read_PBMC8k('ly', filtered_genes=filtered_genes, return_arrays=True)
      my = read_PBMC8k('my', filtered_genes=filtered_genes, return_arrays=True)
      url = str(base64.decodebytes(_URL_PBMC8k), 'utf-8')
      base_name = os.path.basename(url)
      path = os.path.join(download_path, base_name)
      download_file(filename=path, url=url, override=False)
      # load data
      data = np.load(path)
      X = data['X']
      X_row = data['X_row']
      X_col = data['X_col'].tolist()
      y = data['y']
      y_col = data['y_col'].tolist()
      # merge all genes from my and ly subset
      all_genes = set(ly['X_col'].tolist() + my['X_col'].tolist())
      all_genes = sorted([X_col.index(i) for i in all_genes])
      # same for protein
      all_proteins = set(ly['y_col'].tolist() + my['y_col'].tolist())
      all_proteins = sorted([y_col.index(i) for i in all_proteins])
      #
      X = X[:, all_genes]
      y = y[:, all_proteins]
      X_col = np.array(X_col)[all_genes]
      y_col = np.array(y_col)[all_proteins]
      cell_types = np.array(['ly' if i in ly['X_row'] else 'my' for i in X_row])
    # ====== pbmc ly and my ====== #
    else:
      url = str(
          base64.decodebytes(_URL_LYMPHOID if subset == 'ly' else _URL_MYELOID),
          'utf-8')
      base_name = os.path.basename(url)
      path = os.path.join(download_path, base_name)
      download_file(filename=path, url=url, override=False)
      # extract the data
      data = np.load(path)
      X_row = data['X_row']
      y = data['y']
      y_col = data['y_col']
      if filtered_genes:
        X = data['X_filt']
        X_col = data['X_filt_col']
      else:
        X = data['X_full']
        X_col = data['X_full_col']
      cell_types = np.array([subset] * X.shape[0])
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
  if return_arrays:
    return ds
  sco = SingleCellOMIC(X=ds['X'],
                       cell_id=ds['X_row'],
                       gene_id=ds['X_col'],
                       omic='transcriptomic',
                       name=f"8k{subset}{'' if filtered_genes else 'all'}")
  sco.add_omic('proteomic', X=ds['y'], var_names=ds['y_col'])
  progenitor = ds['cell_types']
  sco.add_omic(
      'progenitor',
      X=np.array([(1, 0) if i == 'my' else (0, 1) for i in progenitor],
                 dtype=np.float32),
      var_names=np.array(['myeloid', 'lymphoid']),
  )
  return sco
