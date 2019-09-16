# /data1/czi_data/GSE75478_velten
from __future__ import absolute_import, division, print_function

import base64
import gzip
import os
import pickle
import shutil
from io import BytesIO, StringIO

import numpy as np
import scipy as sp

from odin.fuel import Dataset
from odin.utils import crypto, ctext, get_file
from sisua.data.path import DOWNLOAD_DIR, PREPROCESSED_BASE_DIR
from sisua.data.utils import remove_allzeros_columns, save_to_dataset

_URL = b'aHR0cHM6Ly9zMy5hbWF6b25hd3MuY29tL2FpLWRhdGFzZXRzL0tJX0ZBQ1NfJWRwcm90ZWluLnpp\ncA==\n'
_FACS_PREPROCESSED = os.path.join(PREPROCESSED_BASE_DIR, 'FACS%d_preprocessed')

_PASSWORD = 'uef-czi'


def read_FACS(n_protein, override=False, verbose=False):
  download_path = os.path.join(DOWNLOAD_DIR, "FACS_original")
  if not os.path.exists(download_path):
    os.mkdir(download_path)

  n_protein = int(n_protein)
  assert n_protein in (2, 5)

  preprocessed_path = _FACS_PREPROCESSED % n_protein
  if not os.path.exists(preprocessed_path):
    os.mkdir(preprocessed_path)
  elif override:
    shutil.rmtree(preprocessed_path)
    os.mkdir(preprocessed_path)
  # ******************** preprocessed data NOT found ******************** #
  if not os.path.exists(os.path.join(preprocessed_path, 'X')):
    X, X_row, X_col = [], None, None
    y, y_row, y_col = [], None, None
    # ====== download the data ====== #
    url = str(base64.decodebytes(_URL), 'utf-8') % n_protein
    base_name = os.path.basename(url)
    get_file(fname=base_name, origin=url, outdir=download_path, verbose=verbose)
    zip_path = os.path.join(download_path, base_name)
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
      if verbose:
        print('%-12s' % base_name, ':', data.shape)
    # ====== post-processing ====== #
    X = data_dict['X'].astype('float32')
    X = np.array(X)
    X_row, X_col = data_dict['X_row'], data_dict['X_col']
    assert len(X_row) == X.shape[0] and len(X_col) == X.shape[1]

    y = data_dict['y'].astype('float32')
    y_row, y_col = data_dict['y_row'], data_dict['y_col']
    assert len(y_row) == y.shape[0] and len(y_col) == y.shape[1]

    assert np.all(X_row == y_row), \
    "Cell order mismatch between gene count and protein count"

    # ====== filter zero columns ====== #
    X, X_col = remove_allzeros_columns(matrix=X,
                                       colname=X_col,
                                       print_log=verbose)

    save_to_dataset(path=preprocessed_path,
                    X=X,
                    X_col=X_col,
                    y=y,
                    y_col=y_col,
                    rowname=X_row,
                    print_log=verbose)
  # ******************** read preprocessed data ******************** #
  ds = Dataset(preprocessed_path, read_only=True)
  return ds


# ===========================================================================
# Full FACS data
# ===========================================================================
def read_full_FACS(override=False, verbose=False):
  """ https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE75478
  This is the full FACS data of 2 individuals with 7 protein markers
  """
  download_path = os.path.join(DOWNLOAD_DIR, "FACS_full")
  if not os.path.exists(download_path):
    os.mkdir(download_path)
  # ====== download the data ====== #
  file_url = [
      ('GSE75478_transcriptomics_facs_indeces_filtered_I1.csv.gz',
       'https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE75478&format=file&file=GSE75478%5Ftranscriptomics%5Ffacs%5Findeces%5Ffiltered%5FI1%2Ecsv%2Egz'
      ),
      ('GSE75478_transcriptomics_facs_indeces_filtered_I2.csv.gz',
       'https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE75478&format=file&file=GSE75478%5Ftranscriptomics%5Ffacs%5Findeces%5Ffiltered%5FI2%2Ecsv%2Egz'
      ),
      ('GSE75478_transcriptomics_raw_filtered_I1.csv.gz',
       'https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE75478&format=file&file=GSE75478%5Ftranscriptomics%5Fraw%5Ffiltered%5FI1%2Ecsv%2Egz'
      ),
      ('GSE75478_transcriptomics_raw_filtered_I2.csv.gz',
       'https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE75478&format=file&file=GSE75478%5Ftranscriptomics%5Fraw%5Ffiltered%5FI2%2Ecsv%2Egz'
      ),
  ]
  for name, url in file_url:
    if not os.path.exists(os.path.join(download_path, name)):
      get_file(fname=name, origin=url, outdir=download_path, verbose=verbose)
  # ====== extract the data ====== #
  preprocessed_path = _FACS_PREPROCESSED % 7
  if not os.path.exists(preprocessed_path):
    os.mkdir(preprocessed_path)
  elif override:
    shutil.rmtree(preprocessed_path)
    os.mkdir(preprocessed_path)
  # ******************** preprocessed data NOT found ******************** #
  if not os.path.exists(os.path.join(preprocessed_path, 'X')):
    data_map = {}
    for name, _ in file_url:
      zip_path = os.path.join(download_path, name)
      with gzip.open(zip_path, 'rb') as f:
        data_map[name.split('.')[0]] = np.array(
            [str(line, 'utf-8').strip().split(',') for line in f]).T

    i1 = data_map['GSE75478_transcriptomics_raw_filtered_I1']
    f1 = data_map['GSE75478_transcriptomics_facs_indeces_filtered_I1']

    i2 = data_map['GSE75478_transcriptomics_raw_filtered_I2']
    f2 = data_map['GSE75478_transcriptomics_facs_indeces_filtered_I2']
    # Matching duplicated row in `i` and `f`
    row_name = set(i1[1:, 0]) & set(f1[1:, 0])
    i1 = i1[[True] + [True if i in row_name else False for i in i1[1:, 0]], :]
    f1 = f1[[True] + [True if i in row_name else False for i in f1[1:, 0]], :]
    assert np.all(i1[:, 0] == f1[:, 0])

    row_name = set(i2[1:, 0]) & set(f2[1:, 0])
    i2 = i2[[True] + [True if i in row_name else False for i in i2[1:, 0]], :]
    f2 = f2[[True] + [True if i in row_name else False for i in f2[1:, 0]], :]
    assert np.all(i2[:, 0] == f2[:, 0])

    # Matching the genes and protein among individuals
    gene_name = set(i1[0][1:]) & set(i2[0][1:])
    i1 = i1[:, [True] + [True if i in gene_name else False for i in i1[0][1:]]]
    i2 = i2[:, [True] + [True if i in gene_name else False for i in i2[0][1:]]]
    assert np.all(i1[0] == i2[0])
    gene = np.concatenate((i1, i2[1:]), axis=0)

    prot_name = set([i for i in set(f1[0][1:]) & set(f2[0][1:]) if '_cd' in i])
    prot_name = sorted(prot_name)
    f1 = f1[:, [0] + [f1[0].tolist().index(i) for i in prot_name]]
    f2 = f2[:, [0] + [f2[0].tolist().index(i) for i in prot_name]]
    assert np.all(f1[0] == f2[0])
    prot = np.concatenate((f1, f2[1:]), axis=0)

    # ====== save data to disk ====== #
    X = gene[1:, 1:].astype('float32')
    X_row = gene[1:, 0]
    X_col = gene[0, 1:]
    X_col = np.array([i.replace('"', '') for i in X_col])

    y = prot[1:, 1:].astype('float32')
    y_row = prot[1:, 0]
    y_col = np.array(
        [i.replace('"', '').split('_')[-1].upper() for i in prot[0, 1:]])

    assert np.all(X_row == y_row)
    X_row = np.array([i.replace('"', '') for i in X_row])

    # ====== the protein marker can be smaller than zero ====== #
    min_values = np.min(y, axis=0, keepdims=True)
    min_values = np.where(min_values > 0, 0, min_values)
    y = y + np.abs(min_values)
    # ====== filter zero columns ====== #
    X, X_col = remove_allzeros_columns(matrix=X,
                                       colname=X_col,
                                       print_log=verbose)
    save_to_dataset(path=preprocessed_path,
                    X=X,
                    X_col=X_col,
                    y=y,
                    y_col=y_col,
                    rowname=X_row,
                    print_log=verbose)
  # ******************** read preprocessed data ******************** #
  ds = Dataset(preprocessed_path, read_only=True)
  return ds
