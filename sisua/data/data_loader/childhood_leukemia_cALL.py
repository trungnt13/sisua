import os
import pickle
import shutil
from collections import defaultdict

import numpy as np
import scanpy as sc
from scipy.io import mmread

from bigarray import MmapArray, MmapArrayWriter
from odin.utils import md5_checksum, md5_folder, one_hot
from sisua.data.const import MARKER_GENES
from sisua.data.path import DATA_DIR
from sisua.data.single_cell_dataset import SingleCellOMIC
from sisua.data.utils import read_compressed, validate_data_dir

_URL = r"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE132509"
_MD5_DOWNLOAD = r"1f22e169d590def62e0992d19fe45ba7"
_MD5_PREPROCESSED = r"00a636e37204da4244415140ebe146e8"
_NAME = 'leukemia_bmmc'

__all__ = ['read_leukemia_BMMC']

# Read ETV6-RUNX1_1 (2776, 33694) int64 max:7163 min:0
# Read ETV6-RUNX1_2 (6274, 33694) int64 max:8759 min:0
# Read ETV6-RUNX1_3 (3862, 33694) int64 max:8296 min:0
# Read ETV6-RUNX1_4 (5069, 33694) int64 max:7634 min:0
# Read HHD_1 (3728, 33694) int64 max:6717 min:0
# Read HHD_2 (5013, 33694) int64 max:9092 min:0
# Read PBMMC_1 (1612, 33694) int64 max:6530 min:0
# Read PBMMC_2 (3105, 33694) int64 max:8900 min:0
# Read PBMMC_3 (2229, 33694) int64 max:6035 min:0
# Read PRE-T_1 (2959, 33694) int64 max:5940 min:0
# Read PRE-T_2 (2748, 33694) int64 max:9810 min:0


def _create_sco(X, rowname, colname, labels):
  sco = SingleCellOMIC(X, cell_id=rowname, gene_id=colname, name="cALL")
  mito = [i for i, gene in enumerate(sco.var_names) if 'MT' == gene[:2]]
  percent_mito = np.sum(sco.X[:, mito], axis=1) / np.sum(sco.X, axis=1)
  sco.obs['percent_mito'] = percent_mito
  # add another omic for labels
  if labels is not None:
    sco.obs['labels'] = labels
    labels = [i[:-2] for i in labels]
    var_names = np.asarray(sorted(np.unique(labels)))
    ids = {j: i for i, j in enumerate(var_names)}
    labels = one_hot(np.asarray([ids[i] for i in labels]), len(var_names))
    sco.add_omic('disease', labels, var_names)
  return sco


def read_leukemia_BMMC(path='~/bio_data/downloads/GSE132509_RAW.tar',
                       filtered_genes=True,
                       override=False,
                       verbose=True) -> SingleCellOMIC:
  r""" Childhood acute lymphoblastic leukemia

 	10X Genomics 3’ single cell RNA-seq:

    - 4 individuals: Pre-B t(12;21) [ETV6-RUNX1] acute lymphoblastic leukemia,
    - 2 individuals: Pre-B High hyper diploid [HHD] acute lymphoblastic leukemia,
    - 2 individuals: Pre-T acute lymphoblastic leukemia [PRE-T],
    - 3 individuals: Healthy pediatric bone marrow mononuclear cells [PBMMC].

  Reference:
    - https://github.com/sinnettlab/childhood_leukemia_single_cell_expression
    - Caron, M., St-Onge, P., Sontag, T., et al. 2019. "Single-cell analysis of
        childhood leukemia reveals a link between developmental states and
        ribosomal protein expression as a source of intra-individual
        heterogeneity" (preprint). Cancer Biology. https://doi.org/10.1101/683854
  """
  ### prepare path
  path = os.path.abspath(os.path.expanduser(path))
  assert os.path.exists(path) and os.path.isfile(path), \
    f"{path} doesn't exists, please go to {_URL} and download GSE132509 package"
  preprocessed_path = os.path.join(DATA_DIR, f"{_NAME}_preprocessed")
  validate_data_dir(preprocessed_path, _MD5_PREPROCESSED)
  ### extract file
  with read_compressed(in_file=path,
                       md5_download=_MD5_DOWNLOAD,
                       override=override,
                       verbose=verbose) as extract_files:
    ## extract file structure
    data_name = defaultdict(dict)
    for key, data in extract_files.items():
      # feat: 'barcodes', 'genes', 'matrix'
      name, feat, ext = key.split('.')
      name = name.split('_')
      name = '_'.join(name[1:])
      data_name[name][feat] = data
    ## preprocess the data
    if len(os.listdir(preprocessed_path)) == 0:
      data = []
      labels = []
      rowname = []
      for k, v in sorted(data_name.items()):
        barcodes = np.array([str(i, 'utf-8').strip() for i in v['barcodes']],
                            dtype=str)
        # gene_id, gene_name
        genes = np.array(
            [str(i, 'utf-8').strip().split('\t') for i in v['genes']],
            dtype=str)[:, 1]
        # transpose to cell-gene
        matrix = mmread(v['matrix']).T
        labels.append(np.array([k] * matrix.shape[0]))
        rowname.append(barcodes)
        # convert data to np.uint16, max value is 9810
        matrix = np.asarray(matrix.astype(np.uint16).todense())
        data.append(matrix)
        if verbose:
          print(f"Read {k} {type(matrix)} {matrix.shape} {matrix.dtype}",
                f"max:{np.max(matrix)} min:{np.min(matrix)}")
      # final data
      colname = genes
      rowname = np.concatenate(rowname, axis=0)
      labels = np.concatenate(labels, axis=0)
      data = np.concatenate(data, axis=0)
      assert labels.shape[0] == rowname.shape[0]
      assert data.shape[1] == colname.shape[0]
      assert data.shape[0] == rowname.shape[0]
      with MmapArrayWriter(os.path.join(preprocessed_path, 'X'),
                           shape=(0, data.shape[1]),
                           dtype=np.uint16,
                           remove_exist=True) as f:
        f.write(data)
      for name, x in [
          ('colname', colname),
          ('rowname', rowname),
          ('labels', labels),
      ]:
        with open(os.path.join(preprocessed_path, name), 'wb') as f:
          pickle.dump(x, f)
      # extract variables genes
      sco = _create_sco(data.astype(np.float32), rowname, colname, labels)
      ids = sco.obs['percent_mito'] <= 0.08
      sco = sco[ids]
      sc.pp.filter_cells(sco, min_genes=200)
      sc.pp.filter_genes(sco, min_cells=3)
      sc.pp.normalize_total(sco, target_sum=1e4)
      result = sc.pp.filter_genes_dispersion(sco.X,
                                             min_mean=0.0125,
                                             max_mean=3,
                                             min_disp=0.5,
                                             log=False,
                                             n_top_genes=2000)
      # make sure all marker genes are included
      gene_subset = result.gene_subset
      gene_indices = sco.get_var_indices()
      for gene in MARKER_GENES:
        idx = gene_indices.get(gene, None)
        if idx is not None:
          gene_subset[idx] = True
      sco._inplace_subset_var(gene_subset)
      if verbose:
        print(f"Filtered {len(sco.obs.index.values)} cells and "
              f"{len(sco.var_names.values)} genes.")
      with open(os.path.join(preprocessed_path, 'top_genes'), 'wb') as f:
        pickle.dump(sco.var_names.values, f)
      with open(os.path.join(preprocessed_path, 'top_cells'), 'wb') as f:
        pickle.dump(sco.obs.index.values, f)
      del sco
      # md5
      if verbose:
        print(f"Finish preprocessing: MD5='{md5_folder(preprocessed_path)}'")
  ### create the data set
  X = MmapArray(os.path.join(preprocessed_path, 'X')).astype(np.float32)
  colname = pickle.load(open(os.path.join(preprocessed_path, 'colname'), 'rb'))
  rowname = pickle.load(open(os.path.join(preprocessed_path, 'rowname'), 'rb'))
  labels = pickle.load(open(os.path.join(preprocessed_path, 'labels'), 'rb'))
  # top cells and genes
  cells = pickle.load(open(os.path.join(preprocessed_path, 'top_cells'), 'rb'))
  genes = pickle.load(open(os.path.join(preprocessed_path, 'top_genes'), 'rb'))
  # filter cells
  rowids = {j: i for i, j in enumerate(rowname)}
  ids = [rowids[i] for i in cells]
  X = X[ids]
  rowname = rowname[ids]
  labels = labels[ids]
  if filtered_genes:
    colids = {j: i for i, j in enumerate(colname)}
    ids = [colids[i] for i in genes]
    X = X[:, ids]
    colname = colname[ids]
  return _create_sco(X, rowname, colname, labels)
