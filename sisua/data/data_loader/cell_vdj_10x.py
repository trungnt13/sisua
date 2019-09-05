# For more information:
# https://support.10xgenomics.com/single-cell-gene-expression/software/pipelines/latest/output/matrices
from __future__ import absolute_import, division, print_function

import csv
import gzip
import os
import pickle
import shutil
import tarfile

import numpy as np
from scipy.io import mmread

from odin.fuel import Dataset
from odin.utils import batching, ctext, get_file, select_path
from sisua.data.path import DOWNLOAD_DIR, PREPROCESSED_BASE_DIR
from sisua.data.utils import remove_allzeros_columns, save_to_dataset

_URL = "http://cf.10xgenomics.com/samples/cell-vdj/3.0.2/vdj_v1_hs_aggregated_donor1/vdj_v1_hs_aggregated_donor1_filtered_feature_bc_matrix.tar.gz"


def read_CellVDJ(override=False):
  download_path = os.path.join(DOWNLOAD_DIR, "CellVDJ_original")
  if not os.path.exists(download_path):
    os.mkdir(download_path)

  preprocessed_path = os.path.join(PREPROCESSED_BASE_DIR,
                                   'CellVDJ_preprocessed')

  if override and os.path.exists(preprocessed_path):
    shutil.rmtree(preprocessed_path)
  if not os.path.exists(preprocessed_path):
    os.mkdir(preprocessed_path)

  # ******************** preprocessed ******************** #
  if not os.path.exists(os.path.join(preprocessed_path, 'X')):
    url = _URL
    base_name = os.path.basename(url)
    path = get_file(fname=base_name, origin=url, outdir=download_path)

    contents = {}
    with tarfile.open(path, mode="r:gz") as f:
      for info in f:
        if info.isfile():
          name = info.name
          data = f.extractfile(name)
          data = gzip.open(data, mode="rb")
          name = os.path.basename(name).split('.')[0]
          if name == 'barcodes':
            data = np.array([str(line, 'utf-8')[:-1] for line in data])
          elif name == 'features':
            _ = []
            for line in data:
              line = str(line, 'utf-8')[:-1].split('\t')
              _.append(line)
            data = np.array(_)
          # this will take some time, better print out something
          elif name == 'matrix':
            print("Reading big cell matrix ...")
            data = mmread(data)
          else:
            raise RuntimeError(
                "Unknown downloaded file '%s', something changed from 10xGenomics."
                % name)
          contents[name] = data

    # cell barcodes
    X_row = contents['barcodes']
    # feature (Id, Name, Type(antibody or gene-expression))
    X_col = contents['features']
    # matrix
    X = contents['matrix'].astype('int32').T.todense()
    assert X.shape[0] == X_row.shape[0] and X.shape[1] == X_col.shape[0]

    prot_ids = []
    gene_ids = []
    for idx, row in enumerate(X_col):
      if row[-1] == 'Antibody Capture':
        prot_ids.append(idx)
      elif row[-1] == 'Gene Expression':
        gene_ids.append(idx)
      else:
        raise ValueError("Unknown features: %s" % str(row))

    y = X[:, prot_ids]
    # Antibody ID, Antibody Name
    y_col = X_col[prot_ids][:, 0]
    y_col_name = X_col[prot_ids][:, 1]

    X = X[:, gene_ids]
    # Gene ID, Gene Name
    X_col_name = X_col[gene_ids][:, 1]
    X_col = X_col[gene_ids][:, 0]

    save_to_dataset(preprocessed_path,
                    X=X,
                    X_col=X_col,
                    y=y,
                    y_col=y_col,
                    rowname=X_row)
    with open(os.path.join(preprocessed_path, 'X_col_name'), 'wb') as f:
      pickle.dump(X_col_name, f)
    with open(os.path.join(preprocessed_path, 'y_col_name'), 'wb') as f:
      pickle.dump(y_col_name, f)
  # ====== read preprocessed data ====== #
  ds = Dataset(preprocessed_path, read_only=True)
  return ds
