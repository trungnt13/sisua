import os
import pickle
import shutil
from collections import defaultdict
from urllib.request import urlretrieve

import numpy as np
from scipy import sparse

from odin.utils import md5_checksum, md5_folder, one_hot
from sisua.data.const import MARKER_GENES, OMIC
from sisua.data.path import DATA_DIR, DOWNLOAD_DIR
from sisua.data.single_cell_dataset import SingleCellOMIC
from sisua.data.utils import read_compressed, validate_data_dir

_URL = [
    r"https://github.com/aertslab/cisTopic/raw/3394de3fb57ba5a4e6ab557c7e948e98289ded2c/data/counts_mel.RData",
    r"https://github.com/aertslab/cisTopic/raw/3394de3fb57ba5a4e6ab557c7e948e98289ded2c/data/cellData_mel.RData",
    # r"https://github.com/aertslab/cisTopic/raw/3394de3fb57ba5a4e6ab557c7e948e98289ded2c/data/dm3_CtxRegions.RData",
    # r"https://github.com/aertslab/cisTopic/raw/3394de3fb57ba5a4e6ab557c7e948e98289ded2c/data/dm6_CtxRegions.RData",
    # r"https://github.com/aertslab/cisTopic/raw/3394de3fb57ba5a4e6ab557c7e948e98289ded2c/data/hg19_CtxRegions.RData",
    # r"https://github.com/aertslab/cisTopic/raw/3394de3fb57ba5a4e6ab557c7e948e98289ded2c/data/mm9_CtxRegions.RData",
]


def read_melanoma_cisTopicData(filtered_genes=True, override=False, verbose=True):
  r""" melanoma ATAC data from (Bravo González-Blas, et al. 2019)

  Reference:
    Bravo González-Blas, C. et al. cisTopic: cis-regulatory topic modeling
      on single-cell ATAC-seq data. Nat Methods 16, 397–400 (2019).
    Verfaillie, A. et al. Decoding the regulatory landscape of melanoma
      reveals TEADS as regulators of the invasive cell state.
      Nat Commun 6, (2015).
  """
  download_dir = os.path.join(DOWNLOAD_DIR, 'cistopic')
  if not os.path.exists(download_dir):
    os.makedirs(download_dir)
  preprocessed_path = os.path.join(DATA_DIR, 'cistopic_preprocessed')
  if not os.path.exists(preprocessed_path):
    os.makedirs(preprocessed_path)
  ### downloading the data
  data = {}
  for url in _URL:
    fname = os.path.basename(url)
    fpath = os.path.join(download_dir, fname)
    if not os.path.exists(fpath):
      if verbose:
        print(f"Downloading file: {fname} ...")
      urlretrieve(url, filename=fpath)
    data[fname.split(".")[0]] = fpath
  ### preprocess data
  if len(os.listdir(preprocessed_path)) == 0:
    try:
      import rpy2.robjects as robjects
      from rpy2.robjects import pandas2ri
      from rpy2.robjects.conversion import localconverter
      robjects.r['options'](warn=-1)
      robjects.r("library(Matrix)")
      pandas2ri.activate()
    except ImportError:
      raise ImportError("Require package 'rpy2' for reading Rdata file.")
    for k, v in data.items():
      robjects.r['load'](v)
      x = robjects.r[k]
      outpath = os.path.join(preprocessed_path, k)
      if k == "counts_mel":
        with localconverter(robjects.default_converter + pandas2ri.converter):
          # dgCMatrix
          x = sparse.csr_matrix((x.slots["x"], x.slots["i"], x.slots["p"]),
                                shape=tuple(robjects.r("dim")(x))[::-1],
                                dtype=np.float32)
      else:
        x = robjects.conversion.rpy2py(x)
      with open(outpath, "wb") as f:
        pickle.dump(x, f)
      if verbose:
        print(f"Loaded file: {k} - {type(x)} - {x.shape}")
    pandas2ri.deactivate()
  ### load_data
  data = {}
  for name in os.listdir(preprocessed_path):
    with open(os.path.join(preprocessed_path, name), 'rb') as f:
      data[name] = pickle.load(f)
  ### sco
  # print(data["dm3_CtxRegions"])
  x = data['counts_mel']
  sco = SingleCellOMIC(X=x,
                       cell_id=data["cellData_mel"].index,
                       gene_id=[f"Region{i + 1}" for i in range(x.shape[1])],
                       omic=OMIC.atac)
  # celltype
  labels = []
  for i, j in zip(data["cellData_mel"]['cellLine'],
                  data["cellData_mel"]['LineType']):
    labels.append(i + '_' + j.split("-")[0])
  labels = np.array(labels)
  labels_name = {name: i for i, name in enumerate(sorted(set(labels)))}
  labels = np.array([labels_name[i] for i in labels])
  sco.add_omic(OMIC.celltype, one_hot(labels, len(labels_name)),
               list(labels_name.keys()))
  return sco
