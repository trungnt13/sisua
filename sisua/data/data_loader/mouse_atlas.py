import gzip
import os

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmread

from odin.utils import one_hot
from odin.utils.crypto import md5_folder
from sisua.data.const import OMIC
from sisua.data.path import DATA_DIR, DOWNLOAD_DIR
from sisua.data.single_cell_dataset import SingleCellOMIC
from sisua.data.utils import download_file, read_compressed, validate_data_dir

_URLs = dict(
    counts=
    (r"http://krishna.gs.washington.edu/content/members/ajh24/mouse_atlas_data_release/matrices/atac_matrix.binary.qc_filtered.mtx.gz",
     r"93ec3d4012290543523a70e90d54f98a"),
    cellids=
    (r"http://krishna.gs.washington.edu/content/members/ajh24/mouse_atlas_data_release/matrices/atac_matrix.binary.qc_filtered.cells.txt",
     r"6ff0a57bd95b0d403d7441e3c0bca152"),
    peakids=
    (r"http://krishna.gs.washington.edu/content/members/ajh24/mouse_atlas_data_release/matrices/atac_matrix.binary.qc_filtered.peaks.txt",
     r"00b354cef423f645087c49d6a829c98f"),
    metadata=
    (r"http://krishna.gs.washington.edu/content/members/ajh24/mouse_atlas_data_release/metadata/cell_metadata.txt",
     r"5e63f0830e940e153e7ab254c8787000"),
)


def read_mouse_ATLAS(filtered_genes=True,
                     override=False,
                     verbose=True) -> SingleCellOMIC:
  r""" sci-ATAC-seq, to profile genome-wide chromatin accessibility in ∼100,000
  single cells from 13 adult mouse tissues:

    - The regulatory landscape of adult mouse tissues mapped by single-cell
      chromatin assay
    - Characterization of 85 distinct chromatin patterns across 13 different
      tissues
    - Annotation of key regulators and regulatory sequences in diverse
      mammalian cell types
    - Dataset allows resolution of cell types underlying common human traits
      and diseases

  References:
    Cusanovich, D. A. et al. A Single-Cell Atlas of In Vivo Mammalian Chromatin
      Accessibility. Cell 174, 1309-1324.e18 (2018).
    Link https://atlas.gs.washington.edu/mouse-atac/
  """
  download_path = os.path.join(DOWNLOAD_DIR, f"mouse_atac")
  preprocessed_path = os.path.join(DATA_DIR, f"mouse_atac_preprocessed")
  if not os.path.exists(download_path):
    os.makedirs(download_path)
  if not os.path.exists(preprocessed_path):
    os.makedirs(preprocessed_path)
  ### Download data
  files = {}
  for name, (url, md5) in _URLs.items():
    filepath = os.path.join(download_path, os.path.basename(url))
    files[name] = download_file(url, filepath, override=False, md5=md5)
  ### save counts matrix
  path = os.path.join(preprocessed_path, 'counts')
  if not os.path.exists(path):
    print("Reading counts matrix ...")
    counts = mmread(files['counts'])
    counts: sparse.coo_matrix
    counts = counts.astype(np.unit8)
    with open(path, 'wb') as f:
      sparse.save_npz(f, counts, compressed=False)
  ### save metadata
  path = os.path.join(preprocessed_path, 'metadata')
  if not os.path.exists(path):
    with open(files['cellids'], 'r') as f:
      cell = np.array([i for i in f.read().split('\n') if len(i) > 0])
    with open(files['peakids'], 'r') as f:
      peak = np.array([i for i in f.read().split('\n') if len(i) > 0])
    metadata = pd.read_csv(files['metadata'], sep="\t")
    assert metadata.shape[0] == len(cell)
    tissue = metadata['tissue'].to_numpy()
    celltype = metadata['cell_label'].to_numpy()
    with open(path, 'wb') as f:
      np.savez(f, cell=cell, peak=peak, tissue=tissue, celltype=celltype)
  ### Read all data and create SCO
  counts = sparse.csr_matrix(
      sparse.load_npz(os.path.join(preprocessed_path, 'counts')))
  metadata = np.load(os.path.join(preprocessed_path, 'metadata'),
                     allow_pickle=True)
  cell = metadata['cell']
  peak = metadata['peak']
  tissue = metadata['tissue']
  celltype = metadata['celltype']
  # need to transpose here, counts matrix is [peaks, cells]
  sco = SingleCellOMIC(X=counts.T,
                       cell_id=cell,
                       gene_id=peak,
                       omic=OMIC.atac,
                       name="mouse_atlas")
  # add celltype
  labels = {name: i for i, name in enumerate(sorted(set(celltype)))}
  sco.add_omic(OMIC.celltype,
               X=one_hot(np.array([labels[i] for i in celltype]), len(labels)),
               var_names=list(labels.keys()))
  # add tissue type
  labels = {name: i for i, name in enumerate(sorted(set(tissue)))}
  sco.add_omic(OMIC.tissue,
               X=one_hot(np.array([labels[i] for i in tissue]), len(labels)),
               var_names=list(labels.keys()))
  return sco
