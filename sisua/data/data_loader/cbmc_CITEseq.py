from __future__ import absolute_import, division, print_function

import base64
import os
import pickle
import shutil
from io import BytesIO, StringIO

import numpy as np
import scanpy as sc
import scipy as sp

from odin.fuel import Dataset
from odin.utils import batching, crypto
from sisua.data.path import DATA_DIR, DOWNLOAD_DIR
from sisua.data.single_cell_dataset import SingleCellOMIC
from sisua.data.utils import (download_file, remove_allzeros_columns,
                              save_to_dataset)

_CITEseq_CBMC_PREPROCESSED = os.path.join(DATA_DIR, 'CBMC_preprocessed')

_URL = b'aHR0cHM6Ly9zMy5hbWF6b25hd3MuY29tL2FpLWRhdGFzZXRzL0NCTUMuemlw\n'
_PASSWORD = 'uef-czi'


def read_CITEseq_CBMC(filtered_genes=True, override=False, verbose=True):
  download_path = os.path.join(DOWNLOAD_DIR, "CBMC_original")
  if not os.path.exists(download_path):
    os.mkdir(download_path)
  preprocessed_path = _CITEseq_CBMC_PREPROCESSED
  if not os.path.exists(preprocessed_path):
    os.mkdir(preprocessed_path)
  elif override:
    if verbose:
      print("Overriding path: %s" % _CITEseq_CBMC_PREPROCESSED)
    shutil.rmtree(_CITEseq_CBMC_PREPROCESSED)
    os.mkdir(_CITEseq_CBMC_PREPROCESSED)
  # ******************** preprocessed data NOT found ******************** #
  if not os.path.exists(os.path.join(preprocessed_path, 'X')):
    X, X_row, X_col = [], None, None
    y, y_row, y_col = [], None, None
    # ====== download the data ====== #
    url = str(base64.decodebytes(_URL), 'utf-8')
    base_name = os.path.basename(url)
    zip_path = os.path.join(download_path, base_name)
    download_file(filename=zip_path,
                  url=url,
                  override=False,
                  md5=r"beb76d01a67707c61c21bfb188e1b69f")
    # ====== extract the data ====== #
    data_dict = {}
    for name, data in crypto.unzip_aes(zip_path,
                                       password=_PASSWORD,
                                       verbose=False):
      base_name = os.path.splitext(name)[0]
      if '.npz' in name:
        data = sp.sparse.load_npz(BytesIO(data)).todense()
      elif '.csv' in name:
        data = np.loadtxt(StringIO(str(data, 'utf-8')),
                          dtype=str,
                          delimiter=',')
      else:
        raise RuntimeError("Unknown format: %s" % name)
      data_dict[base_name] = data
    # ====== post-processing ====== #
    X = np.array(data_dict['X'].astype('float32'))
    X_row, X_col = data_dict['X_row'], data_dict['X_col']
    X, X_col = remove_allzeros_columns(matrix=X, colname=X_col)
    assert len(X_row) == X.shape[0] and len(X_col) == X.shape[1]
    y = data_dict['y'].astype('float32')
    y_row, y_col = data_dict['y_row'], data_dict['y_col']
    assert len(y_row) == y.shape[0] and len(y_col) == y.shape[1]
    assert np.all(X_row == y_row), \
    "Cell order mismatch between gene count and protein count"
    # save data
    if verbose:
      print(f"Saving data to {preprocessed_path} ...")
    save_to_dataset(preprocessed_path,
                    X,
                    X_col,
                    y,
                    y_col,
                    rowname=X_row,
                    print_log=verbose)
    sco = SingleCellOMIC(X, cell_id=X_row, gene_id=X_col)
    sc.pp.filter_cells(sco, min_genes=200)
    sc.pp.filter_genes(sco, min_cells=3)
    sc.pp.normalize_total(sco, target_sum=1e4)
    result = sc.pp.filter_genes_dispersion(sco.X,
                                           min_mean=0.0125,
                                           max_mean=3,
                                           min_disp=0.5,
                                           log=False,
                                           n_top_genes=2000)
    sco._inplace_subset_var(result.gene_subset)
    with open(os.path.join(preprocessed_path, 'top_genes'), 'wb') as f:
      pickle.dump(set(sco.var_names.values), f)
    del sco
  # ====== read preprocessed data ====== #
  ds = Dataset(preprocessed_path, read_only=True)
  sco = SingleCellOMIC(
      X=ds['X'],
      cell_id=ds['X_row'],
      gene_id=ds['X_col'],
      omic='transcriptomic',
      name=f"cbmcCITEseq{'_filtered' if filtered_genes else ''}",
  ).add_omic('proteomic', ds['y'], ds['y_col'])
  if filtered_genes:
    with open(os.path.join(preprocessed_path, 'top_genes'), 'rb') as f:
      top_genes = pickle.load(f)
    sco._inplace_subset_var([i in top_genes for i in sco.var_names])
  return sco
