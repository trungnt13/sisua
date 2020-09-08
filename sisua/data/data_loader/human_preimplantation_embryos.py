import os
import pickle
import shutil
import zipfile
from functools import partial

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from six import string_types

from odin.utils import MPI, one_hot
from sisua.data.const import MARKER_GENES
from sisua.data.path import DATA_DIR, DOWNLOAD_DIR
from sisua.data.single_cell_dataset import OMIC, SingleCellOMIC
from sisua.data.utils import download_file, read_r_matrix, validate_data_dir

_URLs = [
    r"https://www.ebi.ac.uk/arrayexpress/files/E-MTAB-3929/E-MTAB-3929.processed.1.zip",
    r"https://www.ebi.ac.uk/arrayexpress/files/E-MTAB-3929/E-MTAB-3929.processed.2.zip",
    r"https://www.ebi.ac.uk/arrayexpress/files/E-MTAB-3929/E-MTAB-3929.processed.3.zip",
    r"https://www.ebi.ac.uk/arrayexpress/files/E-MTAB-3929/E-MTAB-3929.processed.4.zip",
]

_MD5s = [
    r"aecae7898f8313d326426720603133c0",
    r"a83b09ee9465e3a908dd6a691da63e69",
    r"d8fc86b50cae1f8ff0cb3ceb6ca73d40",
    r"ecf2bd8b0176c00e9c05fdebbf7a856a",
]


def read_human_embryos(filtered_genes=True,
                       override=False,
                       verbose=True) -> SingleCellOMIC:
  r""" Transcriptional map of human embryo development, including the sequenced
    transcriptomes of 1529 individual cells from 88 human preimplantation
    embryos. These data show that cells undergo an intermediate state of
    co-expression of lineage-specific genes, followed by a concurrent
    establishment of the trophectoderm, epiblast, and primitive endoderm
    lineages, which coincide with blastocyst formation.

  References:
    Petropoulos S, Edsgärd D, Reinius B, et al. Single-Cell RNA-Seq Reveals
      Lineage and X Chromosome Dynamics in Human Preimplantation Embryos.
      Cell. 2016 Sep

  Note:
    Gene expression levels (RefSeq annotations) were estimated in terms of
      reads per kilobase exon model and per million mapped reads (RPKM)
      using rpkmforgenes
    Genes were filtered, keeping 15633/26178 genes that
      * were expressed in at least 5 out of 1919 sequenced cells (RPKM >= 10).
        and
      * for which cells with expression came from at least two
        different embryos.
    Cells were quality-filtered based on 4 criteria, keeping 1529/1919 cells.
      * First, Spearman correlations, using the RPKM expression levels of
        all genes, for every possible pair of cells were calculated and a
        histogram of the maximum correlation obtained for each cell,
        corresponding to the most similar cell, was used to identify 305
        outlier cells with a maximum pair-wise correlations below 0.63.
      * Second, a histogram of the number of expressed genes per cell was
        used to identify 330 outlier cells with less than 5000 expressed
        genes.
      * Third, a histogram of the total transcriptional expression output
        from the sex chromosomes (RPKM sum) was used to identify 33 cells
        with indeterminable sex, or a called sex that was inconsistent with
        other cells of that embryo
      * Fourth, 13 outlier cells were identified using PCA and t-SNE
        dimensionality reduction.

  """
  download_dir = os.path.join(DOWNLOAD_DIR, 'human_embryos')
  if not os.path.exists(download_dir):
    os.makedirs(download_dir)
  preprocessed_path = os.path.join(DATA_DIR, 'human_embryos_preprocessed')
  if override:
    shutil.rmtree(preprocessed_path)
    if verbose:
      print(f"Override preprocessed data at {preprocessed_path}")
  if not os.path.exists(preprocessed_path):
    os.makedirs(preprocessed_path)
  ### download data
  files = []
  for url, md5 in zip(_URLs, _MD5s):
    path = download_file(url=url,
                         filename=os.path.join(download_dir,
                                               os.path.basename(url)),
                         override=False,
                         md5=md5)
    files.append(path)
  ### preprocessing
  if len(os.listdir(preprocessed_path)) == 0:
    data_map = {}
    for f in files:
      zipname = os.path.basename(f)
      with zipfile.ZipFile(f, mode="r") as f:
        for dat_file in f.filelist:
          filename = dat_file.filename
          dat = str(f.read(filename), 'utf-8')
          x = []
          for line in dat.split('\n'):
            if len(line) == 0:
              continue
            line = line.split('\t')
            x.append(line)
          x = np.asarray(x).T
          row_name = x[1:, 0]
          col_name = x[0, 1:]
          x = x[1:, 1:].astype(np.float32)
          x = sparse.coo_matrix(x)
          data_map[filename] = (x, row_name, col_name)
          print(f"Read: {zipname} - {filename}")
          print(f" * Matrix: {x.shape}")
          print(f" * Row   : {row_name.shape}-{row_name[:3]}")
          print(f" * Col   : {col_name.shape}-{col_name[:3]}")
    # save loaded data to disk
    for name, (x, row, col) in data_map.items():
      with open(os.path.join(preprocessed_path, f"{name}:x"), "wb") as f:
        sparse.save_npz(f, x)
      with open(os.path.join(preprocessed_path, f"{name}:row"), "wb") as f:
        np.save(f, row)
      with open(os.path.join(preprocessed_path, f"{name}:col"), "wb") as f:
        np.save(f, col)
    del data_map
  ### read the data
  # counts.txt (1529, 26178)
  # ercc.counts.txt (1529, 92)
  # rpkm.txt (1529, 26178)
  # ercc.rpkm.txt (1529, 92)
  data = {}
  genes_path = os.path.join(preprocessed_path, "filtered_genes")
  for path in os.listdir(preprocessed_path):
    if path == os.path.basename(genes_path):
      continue
    name, ftype = os.path.basename(path).split(':')
    with open(os.path.join(preprocessed_path, path), 'rb') as f:
      if ftype == 'x':
        x = sparse.load_npz(f).tocsr()
      else:
        x = np.load(f)
    data[f"{name}_{ftype}"] = x
  rpkm = data['rpkm.txt_x']
  counts = data['counts.txt_x']
  genes = data['counts.txt_col']
  cells = data['counts.txt_row']
  ### filter genes
  if not os.path.exists(genes_path):
    # filter genes by rpkm
    ids = np.asarray(np.sum(rpkm, axis=0) >= 10).ravel()
    rpkm = rpkm[:, ids]
    counts = counts[:, ids]
    genes = genes[ids]
    # filter genes by min 5 cells
    ids = np.asarray(np.sum(counts > 0, axis=0) >= 5).ravel()
    rpkm = rpkm[:, ids]
    counts = counts[:, ids]
    genes = genes[ids]
    # filter highly variable genes
    sco = SingleCellOMIC(X=counts, cell_id=cells, gene_id=genes)
    sco.normalize(omic=OMIC.transcriptomic, log1p=True)
    sco.filter_highly_variable_genes(n_top_genes=2000)
    filtered = sco.var_names.to_numpy()
    with open(genes_path, 'wb') as f:
      pickle.dump([genes, filtered], f)
    del sco
  else:
    with open(genes_path, 'rb') as f:
      ids, filtered = pickle.load(f)
    ids = set(ids)
    ids = np.asarray([i in ids for i in genes])
    rpkm = rpkm[:, ids]
    counts = counts[:, ids]
    genes = genes[ids]
  # last filtering
  if filtered_genes:
    filtered = set(filtered)
    ids = np.asarray([i in filtered for i in genes])
    rpkm = rpkm[:, ids]
    counts = counts[:, ids]
    genes = genes[ids]
  ### create the SingleCellOMIC
  sco = SingleCellOMIC(X=counts,
                       cell_id=cells,
                       gene_id=genes,
                       omic=OMIC.transcriptomic,
                       name="HumanEmbryos")
  sco.add_omic(omic=OMIC.rpkm, X=rpkm, var_names=genes)
  labels = ['.'.join(i.split('.')[:-2]) for i in sco.obs_names]
  labels = ['E7' if i == 'E7.4' else i for i in labels]
  labels_name = {j: i for i, j in enumerate(sorted(set(labels)))}
  labels = np.array([labels_name[i] for i in labels])
  sco.add_omic(omic=OMIC.celltype,
               X=one_hot(labels, len(labels_name)),
               var_names=list(labels_name.keys()))
  sco.add_omic(omic=OMIC.ercc,
               X=data['ercc.counts.txt_x'],
               var_names=data['ercc.counts.txt_col'])
  return sco
