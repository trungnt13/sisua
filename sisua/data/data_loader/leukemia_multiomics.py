import os
import pickle
import shutil
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

_URL = dict(
    atac=
    (r"https://jeffgranja.s3.amazonaws.com/MPAL-10x/Supplementary_Data/Healthy-Disease-Data/scATAC-All-Hematopoiesis-MPAL-191120.rds",
     r""),
    rna=
    (r"https://jeffgranja.s3.amazonaws.com/MPAL-10x/Supplementary_Data/Healthy-Disease-Data/scRNA-All-Hematopoiesis-MPAL-191120.rds",
     r"842087f34da9f9201dc47bb069fcebb2"),
    adt=
    (r"https://jeffgranja.s3.amazonaws.com/MPAL-10x/Supplementary_Data/Healthy-Disease-Data/scADT-All-Hematopoiesis-MPAL-191120.rds",
     r"6383ebe33caf774a45e50374452f0c32"),
)
_MD5_PREPROCESSED = r""


# ===========================================================================
# Helpers
# ===========================================================================
class SummarizedExperiment():

  def __init__(self, path):
    self.path = path
    self.cellnames = None
    self.genenames = None
    self.celldata = None
    self.genedata = None
    self.datatype = None
    self.X = None

  def validate(self):
    assert self.datatype == 'counts', "Only support counts data"
    assert (np.all(self.cellnames == self.celldata.index) and
            np.all(self.genenames == self.genedata.index) and
            self.X.shape == (len(self.cellnames), len(self.genenames)))

  @property
  def is_exists(self):
    return os.path.exists(self.path + '.X.npz') or \
      os.path.exists(self.path + '.X.npy')

  def _read_pickle(self, name):
    with open(self.path + '.' + name, 'rb') as f:
      return pickle.load(f)

  def load(self):
    if self.X is None:
      self.cellnames = self._read_pickle('cellnames')
      self.genenames = self._read_pickle('genenames')
      self.celldata = self._read_pickle('celldata')
      self.genedata = self._read_pickle('genedata')
      self.datatype = self._read_pickle('datatype')
      if os.path.exists(self.path + '.X.npz'):
        self.X = sparse.load_npz(self.path + '.X.npz')
      else:
        self.X = np.load(self.path + '.X.npy',
                         mmap_mode='r',
                         allow_pickle=False)
    return self

  def __repr__(self):
    return self.__str__()

  def __str__(self):
    return f"<SummarizedExperiment {self.path}>"


def _metadata(data):
  x = data.slots['listData']
  return pd.DataFrame({k: v for k, v in zip(x.names, x)},
                      index=data.slots['rownames'])


def _read_data(name_path, verbose, preprocessed_path):
  name, path = name_path
  outpath = os.path.join(preprocessed_path, name)
  exp = SummarizedExperiment(path=outpath)
  if exp.is_exists:
    return name, exp
  # load from scratch
  try:
    import rpy2.robjects as robjects
    import rpy2.robjects.packages as rpackages
    readRDS = robjects.r['readRDS']
    se = rpackages.importr("SummarizedExperiment")
  except ImportError:
    raise ImportError("Require package 'rpy2' for reading binary RDS file.")
  ext = os.path.splitext(path)[-1].lower()
  assert '.rds' == ext, "Only support reading RDS files"
  try:
    data = readRDS(path)
    rclass = list(data.rclass)[0]
  except Exception as e:
    print("Require 'SummarizedExperiment' package for reading RDS file.")
    import traceback
    traceback.print_exc()
    raise e
  # read SummarizedExperiment object
  assert 'SummarizedExperiment' in rclass, "Only support SummarizedExperiment"
  for k, v in [('cellnames', np.array(robjects.r.colnames(data))),
               ('genenames', np.array(robjects.r.rownames(data))),
               ('celldata', _metadata(se.colData(data))),
               ('genedata', _metadata(se.rowData(data))),
               ('datatype', se.assayNames(data)[0]),
               ('X', read_r_matrix(se.assay(data)).T)]:
    outpath = exp.path + '.' + k
    if k == 'X':
      if sparse.issparse(v):
        if np.max(v) <= 255:
          dtype = np.uint8
        elif np.max(v) < 65536:
          dtype = np.uint16
        else:
          raise RuntimeError("Only support uint16 or uint8")
        if name == 'atac':
          sparse.save_npz(outpath, v.astype(dtype), compressed=False)
        else:
          np.save(outpath, v.todense().astype(dtype), allow_pickle=False)
      else:
        np.save(outpath, v, allow_pickle=False)
    else:
      with open(outpath, 'wb') as f:
        pickle.dump(v, f)
  if verbose:
    print(f"Read {name} -> {outpath}")
  return name, exp


def _celltypes(y):
  labels = sorted(np.unique(y))
  index = {name: i for i, name in enumerate(labels)}
  y = one_hot(np.array([index[i] for i in y], dtype=np.int32),
              nb_classes=len(labels))
  return y, [i.replace("_Like", '').lower() for i in labels]


# ===========================================================================
# Main
# ===========================================================================
def read_leukemia_MixedPhenotypes(filtered_genes=True,
                                  omic='rna',
                                  ignore_na=True,
                                  override=False,
                                  verbose=True) -> SingleCellOMIC:
  r""" Integrates highly multiplexed protein quantification, transcriptome
  profiling, and chromatin accessibility analysis. Using this approach,
  we establish a normal epigenetic baseline for healthy blood development,
  which we then use to deconvolve aberrant molecular features within blood
  from mixed-phenotype acute leukemia (MPAL) patients.

  scATAC-seq and CITE-seq performed on healthy bone marrow, CD34+ bone marrow,
  peripheral blood, and MPAL donors

  References:
    Granja JM et al., 2019. "Single-cell multiomic analysis identifies
      regulatory  programs in mixed-phenotype acute leukemia".
      Nature Biotechnology.
    https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE139369
    https://github.com/GreenleafLab/MPAL-Single-Cell-2019
  """
  ### prepare the path
  download_dir = os.path.join(DOWNLOAD_DIR, 'mpal')
  if not os.path.exists(download_dir):
    os.makedirs(download_dir)
  preprocessed_path = os.path.join(DATA_DIR, 'mpal_preprocessed')
  if override:
    shutil.rmtree(preprocessed_path)
    if verbose:
      print(f"Override preprocessed data at {preprocessed_path}")
  if not os.path.exists(preprocessed_path):
    os.makedirs(preprocessed_path)
  ### download
  files = {}
  for name, (url, md5) in _URL.items():
    path = download_file(url=url,
                         filename=os.path.join(download_dir,
                                               os.path.basename(url)),
                         override=False,
                         md5=md5)
    files[name] = path
  ### read the files
  if omic == 'atac':
    del files['rna']
    del files['adt']
  elif omic == 'rna':
    del files['atac']
  else:
    raise NotImplementedError(f"No support for omic type: {omic}")
  all_data = {}
  for name, data in MPI(jobs=list(files.items()),
                        func=partial(_read_data,
                                     verbose=True,
                                     preprocessed_path=preprocessed_path),
                        batch=1,
                        ncpu=4):
    all_data[name] = data.load()
  ### load scRNA and ADT
  if omic == 'rna':
    rna = all_data['rna']
    adt = all_data['adt']
    cell_id = list(set(rna.celldata['Barcode']) & set(adt.celldata['Barcode']))
    #
    barcode2ids = {j: i for i, j in enumerate(rna.celldata['Barcode'])}
    ids = [barcode2ids[i] for i in cell_id]
    X_rna = rna.X[ids].astype(np.float32)
    classification = rna.celldata['ProjectClassification'][ids].values
    #
    barcode2ids = {j: i for i, j in enumerate(adt.celldata['Barcode'])}
    X_adt = adt.X[[barcode2ids[i] for i in cell_id]].astype(np.float32)
    #
    if filtered_genes:
      top_genes_path = os.path.join(preprocessed_path, 'top_genes')
      if os.path.exists(top_genes_path):
        with open(top_genes_path, 'rb') as f:
          top_genes = set(pickle.load(f))
        ids = [i for i, j in enumerate(rna.genenames) if j in top_genes]
        sco = SingleCellOMIC(X_rna[:, ids],
                             cell_id=cell_id,
                             gene_id=rna.genenames[ids],
                             omic=OMIC.transcriptomic,
                             name='mpalRNA')
      else:
        sco = SingleCellOMIC(X_rna,
                             cell_id=cell_id,
                             gene_id=rna.genenames,
                             omic=OMIC.transcriptomic,
                             name='mpalRNA')
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
        with open(top_genes_path, 'wb') as f:
          pickle.dump(sco.var_names.values, f)
    else:
      sco = SingleCellOMIC(X_rna,
                           cell_id=cell_id,
                           gene_id=rna.genenames,
                           omic=OMIC.transcriptomic,
                           name='mpalRNAall')
    # loading dataset
    if ignore_na:
      ids = np.logical_not(np.isnan(np.max(X_adt, axis=0)))
      sco.add_omic(OMIC.proteomic, X_adt[:, ids], adt.genenames[ids])
    else:
      sco.add_omic(OMIC.proteomic, X_adt, adt.genenames)
    y, labels = _celltypes(classification)
    sco.add_omic(OMIC.celltype, y, labels)
    exon = {i: j for i, j in rna.genedata[['gene_name', 'exonLength']].values}
    sco.var['exonlength'] = np.array([exon[i] for i in sco.var_names],
                                     dtype=np.float32)
  ### load ATAC
  else:
    atac = all_data['atac']
    sco = SingleCellOMIC(atac.X.astype(np.float32),
                         cell_id=atac.celldata['Barcode'],
                         gene_id=atac.genenames,
                         omic=OMIC.chromatin,
                         name='mpalATAC')
    y, labels = _celltypes(atac.celldata['ProjectClassification'].values)
    sco.add_omic(OMIC.celltype, y, labels)
    sco.obs['clusters'] = atac.celldata['Clusters'].values
    sco.var['score'] = atac.genedata['score'].values
  return sco
