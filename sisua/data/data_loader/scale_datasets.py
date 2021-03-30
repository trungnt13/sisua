import base64
import gzip
import os
import zipfile

import numpy as np
from scipy import sparse
from scipy.io import mmread

from odin.utils import one_hot
from odin.utils.crypto import md5_folder
from sisua.data.const import OMIC
from sisua.data.path import DATA_DIR, DOWNLOAD_DIR
from sisua.data.single_cell_dataset import SingleCellOMIC
from sisua.data.utils import download_file, read_compressed, validate_data_dir

# https://cloud.tsinghua.edu.cn/d/eb4371c556bc46ef8516/?p=%2F&mode=list
_URL = b'aHR0cHM6Ly9haS1kYXRhc2V0cy5zMy5hbWF6b25hd3MuY29tL3NjYWxlX2RhdGFzZXRzLnppcA==\n'
_MD5 = r"5fc7c52108220e30a04f033e355716c0"


def read_scale_dataset(dsname="leukemia",
                       filtered_genes=True,
                       override=False,
                       verbose=True) -> SingleCellOMIC:
  r""" Datasets provided by (Xiong et al. 2019), four datasets are supported:

    - 'breast_tumor'
    - 'forebrain'
    - 'leukemia'
    - 'insilico'

  Reference:
    Xiong, L. et al. SCALE method for single-cell ATAC-seq analysis via latent
      feature extraction. Nat Commun 10, 4576 (2019).

  """
  datasets = {'breast_tumor', 'forebrain', 'leukemia', 'insilico'}
  assert dsname in datasets, \
    f"Cannot find dataset with name {dsname}, available datasets are: {datasets}"
  download_path = os.path.join(DOWNLOAD_DIR, f"scale_dataset")
  preprocessed_path = os.path.join(DATA_DIR, f"scale_preprocessed")
  if not os.path.exists(download_path):
    os.makedirs(download_path)
  if not os.path.exists(preprocessed_path):
    os.makedirs(preprocessed_path)
  ### Download data
  url = str(base64.decodebytes(_URL), 'utf-8')
  path = os.path.join(download_path, os.path.basename(url))
  download_file(url, path, override=False, md5=_MD5)
  ### extract the data
  if len(os.listdir(preprocessed_path)) == 0:
    with zipfile.ZipFile(path, "r") as f:
      for info in f.filelist:
        name = os.path.basename(info.filename)
        if len(name) == 0:
          continue
        with open(os.path.join(preprocessed_path, name), 'wb') as fout:
          fout.write(f.read(info))
  ### load the data
  cell = np.load(os.path.join(preprocessed_path, f"{dsname}_cell"))
  labels = np.load(os.path.join(preprocessed_path, f"{dsname}_labels"))
  peak = np.load(os.path.join(preprocessed_path, f"{dsname}_peak"))
  x = sparse.load_npz(os.path.join(preprocessed_path, f"{dsname}_x"))
  sco = SingleCellOMIC(X=x,
                       cell_id=cell,
                       gene_id=peak,
                       omic=OMIC.atac,
                       name=dsname)
  ids = {key: i for i, key in enumerate(sorted(set(labels)))}
  sco.add_omic(OMIC.celltype,
               X=one_hot(np.array([ids[i] for i in labels]), len(ids)),
               var_names=list(ids.keys()))
  return sco
