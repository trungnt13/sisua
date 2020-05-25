import gzip
import os

import numpy as np
from scipy.io import mmread

from odin.utils.crypto import md5_folder
from sisua.data.path import DATA_DIR, DOWNLOAD_DIR
from sisua.data.single_cell_dataset import SingleCellOMIC
from sisua.data.utils import download_file, read_compressed, validate_data_dir

_URL = dict(
    matrix=
    (r"http://krishna.gs.washington.edu/content/members/ajh24/mouse_atlas_data_release/matrices/atac_matrix.binary.qc_filtered.mtx.gz",
     r"93ec3d4012290543523a70e90d54f98a"),
    cells=
    (r"http://krishna.gs.washington.edu/content/members/ajh24/mouse_atlas_data_release/matrices/atac_matrix.binary.qc_filtered.cells.txt",
     r"6ff0a57bd95b0d403d7441e3c0bca152"),
    genes=
    (r"http://krishna.gs.washington.edu/content/members/ajh24/mouse_atlas_data_release/matrices/atac_matrix.binary.qc_filtered.peaks.txt",
     r"00b354cef423f645087c49d6a829c98f"),
    cellmeta=
    (r"http://krishna.gs.washington.edu/content/members/ajh24/mouse_atlas_data_release/metadata/cell_metadata.txt",
     r"5e63f0830e940e153e7ab254c8787000"))

_MD5_PREPROCESSED = r""


def read_mouse_ATAC(filtered_genes=True,
                    override=False,
                    verbose=True) -> SingleCellOMIC:
  r""" sci-ATAC-seq, to profile genome-wide chromatin accessibility in
  ~100,000 single cells from 13 adult mouse

  Reference:
    Cusanovich DA, Hill AJ, Aghamirzaie D, Daza RM et al. "A Single-Cell Atlas
      of In Vivo Mammalian Chromatin Accessibility". Cell 2018 Aug
    http://atlas.gs.washington.edu/mouse-atac/data/#atac-matrices
    https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE111586
  """
  download_path = os.path.join(DOWNLOAD_DIR, "mouse_atac")
  preprocessed_path = os.path.join(DATA_DIR, "mouse_atac_preprocessed")
  if not os.path.exists(download_path):
    os.makedirs(download_path)
  ### Download data
  files = {}
  for name, (url, md5) in _URL.items():
    files[name] = download_file(url,
                                os.path.join(download_path, name),
                                override=False,
                                md5=md5)
  ### extract the data
  validate_data_dir(preprocessed_path, _MD5_PREPROCESSED)
  if len(os.listdir(preprocessed_path)) == 0:
    cells = np.genfromtxt(files['cells'], dtype=str)
    genes = np.genfromtxt(files['genes'], dtype=str)
    meta = np.genfromtxt(files['cellmeta'], delimiter='\t', dtype=str)
    # super sparse matrix (81173, 436206)
    with gzip.open(files['matrix'], 'rb') as f:
      matrix = mmread(f)
    print(matrix.shape)
    # print(meta)
  ### load the data
