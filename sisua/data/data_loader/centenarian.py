# Reference:
# "Single-cell transcriptomics reveals expansion of cytotoxic CD4 T-cells in
# supercentenarians", Kosuke Hashimoto, Tsukasa Kouno, Tomokatsu Ikawa, el. at
# https://www.biorxiv.org/content/10.1101/643528v1
import gzip
import os
import pickle
import shutil
from urllib.request import urlretrieve

import numpy as np

from bigarray import MmapArrayWriter
from odin.fuel import Dataset
from odin.stats import describe
from odin.utils import batching, one_hot, select_path
from odin.utils.crypto import decrypt_aes, md5_checksum
from sisua.data.path import DATA_DIR, DOWNLOAD_DIR
from sisua.data.utils import remove_allzeros_columns, save_to_dataset

path = "/home/trung/bio_data/downloads/SuperCentenarian_original/01.UMI.txt.gz"

# ===========================================================================
# Constants
# ===========================================================================
# ## ---- Raw UMI counts ---- ##
# 01.UMI.txt.gz
#         row: Ensembl Gene ID (23384 genes)
#         col: Cell barcode (61202 cells)
# ## ---- Normalized expression ---- ##
# 02.UMI.lognorm.txt.gz
#         row: Ensembl Gene ID (23384 genes)
#         col: Cell barcode (61202 cells)
# ## ---- Table of Sample - Cell barcode ---- ##
# 03.Cell.Barcodes.txt.gz
#         Col1: Cell barcode (61202 cells)
#         Col2: Sample ID
#         Col3: Sample Type (SC and CT denotes supercentenarians and controls)
# ## ---- The second sequencing of SC1 ---- ##
# 04.SC1.tar
#         barcodes.tsv.gz: Cell barcode
#         features.tsv.gz: Ensemble Gene ID
#         matrix.mtx.gz: pression matrix in the Market Exchange Format
# ## ---- The second sequencing of SC2 ---- ##
# 05.SC2.tar
#         barcodes.tsv.gz: Cell barcode
#         features.tsv.gz: Ensemble Gene ID
#         matrix.mtx.gz: pression matrix in the Market Exchange Format
# Protein
_URL = [
    "http://gerg.gsc.riken.jp/SC2018/01.UMI.txt.gz",
    "http://gerg.gsc.riken.jp/SC2018/02.UMI.lognorm.txt.gz",
    "http://gerg.gsc.riken.jp/SC2018/03.Cell.Barcodes.txt.gz",
    "http://gerg.gsc.riken.jp/SC2018/04.SC1.tar",
    "http://gerg.gsc.riken.jp/SC2018/05.SC2.tar",
]


def read_gzip_csv(path):
  data = []
  cell_id = None
  gene_id = None
  with gzip.open(path, mode='rb') as f:
    for line in f:
      line = str(line, 'utf-8').strip().split('\t')
      data.append(line)
  cell_id = np.array(data[0])
  data = data[1:]
  gene_id = np.array([i[0] for i in data])
  data = np.array([i[1:] for i in data], dtype='float32').T
  return data, cell_id, gene_id


def download_file(url, outpath):
  from tqdm import tqdm
  prog = [None]

  def _progress(blocknum, blocksize, size):
    if prog[0] is None:
      prog[0] = tqdm(desc=f"Downloading {os.path.basename(url)}",
                     total=int(size / 1024.) + 1,
                     unit='kB')
    prog[0].update(int(blocknum * blocksize / 1024.) - prog[0].n)

  urlretrieve(url=url, filename=outpath, reporthook=_progress)
  prog[0].clear()
  prog[0].close()


# ===========================================================================
# Main
# ===========================================================================
def read_centenarian(override=False, verbose=False):
  r""" Data used in:

    "Single-cell transcriptomics reveals expansion of cytotoxic CD4 T-cells in
    supercentenarians" | bioRxiv [WWW Document], n.d.
      URL https://www.biorxiv.org/content/10.1101/643528v1 (accessed 5.21.20).

  """
  download_path = os.path.join(DOWNLOAD_DIR, "SuperCentenarian_original")
  if not os.path.exists(download_path):
    os.mkdir(download_path)
  preprocessed_path = os.path.join(DATA_DIR, 'SuperCentenarian_preprocessed')
  if override and os.path.exists(preprocessed_path):
    shutil.rmtree(preprocessed_path)
  if not os.path.exists(preprocessed_path):
    os.mkdir(preprocessed_path)
  # ******************** preprocessed ******************** #
  if not os.path.exists(os.path.join(preprocessed_path, 'X')):
    labels = download_file(
        outpath=os.path.join(download_path, os.path.basename(_URL[2])),
        url=_URL[2],
    )
    data = []
    with gzip.open(labels, mode='rb') as f:
      for line in f:
        line = str(line, 'utf-8').strip().split('\t')
        assert line[1][:2] == line[2]
        data.append(line)
    labels = np.array(data)
    y_col = sorted(set(labels[:, 1]))
    y = one_hot(np.array([y_col.index(i) for i in labels[:, 1]]),
                len(y_col)).astype('float32')
    y_col = np.array(y_col)
    #
    raw = download_file(
        outpath=os.path.join(download_path, os.path.basename(_URL[0])),
        url=_URL[0],
    )
    if verbose:
      print("Unzip and reading raw UMI ...")
    X_raw, cell_id1, gene_id1 = read_gzip_csv(raw)
    #
    norm = download_file(
        outpath=os.path.join(download_path, os.path.basename(_URL[1])),
        url=_URL[1],
    )
    if verbose:
      print("Unzip and reading log-norm UMI ...")
    X_norm, cell_id2, gene_id2 = read_gzip_csv(norm)
    #
    assert np.all(cell_id1 == cell_id2) and np.all(labels[:, 0] == cell_id1) and \
      np.all(gene_id1 == gene_id2)
    assert X_raw.shape[0] == X_norm.shape[0] == len(cell_id1) and \
      X_raw.shape[1] == X_norm.shape[1] == len(gene_id1)
    #
    if verbose:
      print(f"Saving data to {preprocessed_path} ...")
    save_to_dataset(preprocessed_path,
                    X=X_raw,
                    X_col=gene_id1,
                    y=y,
                    y_col=y_col,
                    rowname=cell_id1,
                    print_log=verbose)
    with MmapArrayWriter(os.path.join(preprocessed_path, 'X_log'),
                         shape=(0, X_norm.shape[1]),
                         dtype='float32',
                         remove_exist=True) as f:
      for s, e in batching(batch_size=2048, n=X_norm.shape[0]):
        f.write(X_norm[s:e])
  # ====== read preprocessed data ====== #
  ds = Dataset(preprocessed_path, read_only=True)
  return ds
