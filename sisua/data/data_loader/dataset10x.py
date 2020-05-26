# Modfied code from scVI to SISUA for loading 10xGenomics datasets
#
# MIT License
# Copyright (c) 2018, Romain Lopez
# ...
#
# Full license: https://github.com/YosefLab/scVI/blob/master/LICENSE
# Original code: https://github.com/YosefLab/scVI/blob/master/scvi/dataset/dataset10X.py
#
# For more information about how to read data from 10xGenomic repos:
# https://support.10xgenomics.com/single-cell-gene-expression/software/pipelines/latest/output/matrices
from __future__ import absolute_import, division, print_function

import gzip
import os
import pickle
import shutil
import tarfile
from typing import Tuple
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.io import mmread
from scipy.sparse import csr_matrix, issparse
from tqdm import tqdm

from bigarray import MmapArray, MmapArrayWriter
from odin.fuel import Dataset
from odin.utils import MPI, batching, ctext, is_gzip_file, select_path
from sisua.data.const import MARKER_GENES, MARKER_REGIONS, OMIC
from sisua.data.path import DATA_DIR, DOWNLOAD_DIR
from sisua.data.single_cell_dataset import SingleCellOMIC
from sisua.data.utils import (download_file, remove_allzeros_columns,
                              save_to_dataset, standardize_protein_name)

__all__ = ['read_dataset10x']

all_datasets = {
    # https://support.10xgenomics.com/single-cell-atac/datasets
    'cell-atac': {
        "1.2.0": [
            "atac_hgmm_500_v1",
            "atac_hgmm_500_nextgem",
            "atac_hgmm_1k_v1",
            "atac_hgmm_1k_nextgem",
            "atac_hgmm_5k_v1",
            "atac_hgmm_5k_nextgem",
            "atac_hgmm_10k_v1",
            "atac_hgmm_10k_nextgem",
            "atac_pbmc_500_v1",
            "atac_pbmc_500_nextgem",
            "atac_pbmc_1k_v1",
            "atac_pbmc_1k_nextgem",
            "atac_pbmc_5k_v1",
            "atac_pbmc_5k_nextgem",
            "atac_pbmc_10k_v1",
            "atac_pbmc_10k_nextgem",
            "atac_v1_hgmm_500",
            "atac_v1_hgmm_1k",
            "atac_v1_hgmm_5k",
            "atac_v1_hgmm_10k",
            "atac_v1_pbmc_5k",
            "atac_v1_pbmc_10k",
            "atac_v1_E18_brain_fresh_5k",
            "atac_v1_E18_brain_cryo_5k",
            "atac_v1_E18_brain_flash_5k",
        ]
    },
    # https://support.10xgenomics.com/single-cell-vdj/datasets
    'cell-vdj': {
        "3.1.0": [
            "vdj_v1_hs_pbmc3_protein",
            "vdj_nextgem_hs_pbmc3_protein",
            "vdj_v1_hs_pbmc3",
            "vdj_nextgem_hs_pbmc3",
            "vdj_v1_mm_pbmc4",
            "vdj_nextgem_mm_pbmc4",
        ],
        "3.0.2": [
            "vdj_v1_hs_aggregated_donor1",  # * merja selected
            "vdj_v1_hs_aggregated_donor2",
            "vdj_v1_hs_aggregated_donor3",
            "vdj_v1_hs_aggregated_donor4",
        ]
    },
    'cell-exp': {
        "1.1.0": [
            "frozen_pbmc_donor_a", "frozen_pbmc_donor_b", "frozen_pbmc_donor_c",
            "fresh_68k_pbmc_donor_a", "cd14_monocytes", "b_cells", "cd34",
            "cd56_nk", "cd4_t_helper", "regulatory_t", "naive_t", "memory_t",
            "cytotoxic_t", "naive_cytotoxic"
        ],
        "2.1.0": ["pbmc8k", "pbmc4k", "t_3k", "t_4k", "neuron_9k"],
        "3.0.0": [
            "pbmc_1k_protein_v3",
            "pbmc_10k_protein_v3",
            "malt_10k_protein_v3",
            "pbmc_1k_v2",
            "pbmc_1k_v3",
            "pbmc_10k_v3",
            "hgmm_1k_v2",
            "hgmm_1k_v3",
            "hgmm_5k_v3",
            "hgmm_10k_v3",
            "neuron_1k_v2",
            "neuron_1k_v3",
            "neuron_10k_v3",
            "heart_1k_v2",
            "heart_1k_v3",
            "heart_10k_v3",
        ],
        "3.1.0": ["5k_pbmc_protein_v3", "5k_pbmc_protein_v3_nextgem"],
    }
}

group_to_url_skeleton = {
    "cell-atac": {
        "1.2.0":
            "http://cf.10xgenomics.com/samples/cell-atac/{}/{}/{}_{}_peak_bc_matrix.tar.gz",
    },
    "cell-vdj": {
        "3.1.0":
            "http://cf.10xgenomics.com/samples/cell-vdj/{}/{}/{}_{}_feature_bc_matrix.tar.gz",
        "3.0.2":
            "http://cf.10xgenomics.com/samples/cell-vdj/{}/{}/{}_{}_feature_bc_matrix.tar.gz"
    },
    "cell-exp": {
        "1.1.0":
            "http://cf.10xgenomics.com/samples/cell-exp/{}/{}/{}_{}_gene_bc_matrices.tar.gz",
        "2.1.0":
            "http://cf.10xgenomics.com/samples/cell-exp/{}/{}/{}_{}_gene_bc_matrices.tar.gz",
        "3.0.0":
            "http://cf.10xgenomics.com/samples/cell-exp/{}/{}/{}_{}_feature_bc_matrix.tar.gz",
        "3.1.0":
            "http://cf.10xgenomics.com/samples/cell-exp/{}/{}/{}_{}_feature_bc_matrix.tar.gz",
    }
}

_MD5 = {
    "cell-vdj*3.0.2*vdj_v1_hs_aggregated_donor1*filtered":
        r"2989b9f660f6acfb2f3a22066afae83d",
    "cell-vdj*3.0.2*vdj_v1_hs_aggregated_donor2*filtered":
        r"8a34c2b8b41016ac066ebd3e7623fe40",
    "cell-atac*1.2.0*atac_pbmc_10k_v1*filtered":
        r"e189271db6fba135dd0672423d7957bf",
    "cell-atac*1.2.0*atac_pbmc_500_v1*filtered":
        r"7b925598349958b1a3ea3ee5c637a760"
}
available_specification = ["filtered", "raw"]


# ===========================================================================
# Helpers
# ===========================================================================
def _read_tarinfo(path_name_size_verbose):
  path, name, size, verbose = path_name_size_verbose
  with tarfile.open(path, mode="r:gz") as f:
    if verbose:
      print(f"Extracting '{name}' size {size / 1024. / 1024.:.2f}(MB) ...")
    data = f.extractfile(name)
    if is_gzip_file(data):
      data = gzip.open(data, mode="rb")
    all_ext = name.strip().lower().split('.')
    name = os.path.basename(name).split('.')[0]
    # metadata
    if name == "peaks":
      data = np.array([str(line, 'utf-8').strip().split("\t") for line in data])
    elif name == 'barcodes':
      data = np.array([str(line, 'utf-8')[:-1] for line in data])
    elif name in ('features', 'genes'):
      if 'tsv' in all_ext:
        sep = "\t"
      elif 'csv' in all_ext:
        sep = ","
      elif 'txt' in all_ext:
        sep = " "
      else:
        raise RuntimeError(f"Unknown data format for file {name}, from {path}")
      data = np.array([str(line, 'utf-8')[:-1].split(sep) for line in data])
    # read the data matrix
    elif name == 'matrix':
      data = mmread(data)
      if data.dtype == np.int64:
        data = data.astype(np.int32)
      elif data.dtype == np.float64:
        data = data.astype(np.float32)
    else:
      raise RuntimeError(f"Unknown downloaded file {name}, from {path}")
  if verbose:
    print(f" {name}: {type(data)}{data.shape}-{data.dtype}")
  return name, data


# ===========================================================================
# Main dataset download
# ===========================================================================
def read_dataset10x(name,
                    filtered_cells=True,
                    filtered_genes=True,
                    override=False,
                    verbose=True) -> SingleCellOMIC:
  r""" Predefined procedure for download and preprocessing 10x dataset into
  `SingleCellOMIC` i.e. scanpy.AnnData object

  Reference:
    https://artyomovlab.wustl.edu/publications/supp_materials/4Oleg/2019_sc_ATAC_seq_DT1634_Denis/sc-atacseq-explorer-Denis-121119.html

  """
  ### prepare the URL
  name = str(name).lower().strip()
  spec = 'filtered' if filtered_cells else 'raw'
  flatten_datasets = [(exp, version, dsname) for exp, i in all_datasets.items()
                      for version, j in i.items() for dsname in j]
  found = []
  for exp, version, dsname in flatten_datasets:
    if name == dsname:
      found.append((exp, version, dsname))
  if not found:
    raise ValueError(f"Cannot find data with name {name}, "
                     f"all available datasets are: {flatten_datasets}")
  if len(found) > 1:
    raise RuntimeError(f"Found multiple datasets {found} with name='{name}'")
  exp, version, name = found[0]
  dataset_name = name + '_' + spec
  url = group_to_url_skeleton[exp][version].format(version, name, name, spec)
  ### prepare the output path
  filename = os.path.basename(url)
  # download path
  download_path = os.path.join(DOWNLOAD_DIR, exp, version)
  if not os.path.exists(download_path):
    os.makedirs(download_path)
  # preprocessing path
  preprocessed_path = os.path.join(DATA_DIR,
                                   f'10x_{exp}_{name}_{spec}_preprocessed')
  if override and os.path.exists(preprocessed_path):
    if verbose:
      print("Overriding path: %s" % preprocessed_path)
    shutil.rmtree(preprocessed_path)
  if not os.path.exists(preprocessed_path):
    os.mkdir(preprocessed_path)
  # ******************** preprocessed ******************** #
  if len(os.listdir(preprocessed_path)) == 0:
    if verbose:
      print("Dataset10X:")
      print(" Meta       :", found)
      print(" File       :", filename)
      print(" URL        :", url)
      print(" Download   :", download_path)
      print(" Preprocess :", preprocessed_path)
    ### download the tar file
    path = download_file(url=url,
                         filename=os.path.join(download_path, filename),
                         override=False,
                         md5=_MD5.get(f"{exp}*{version}*{name}*{spec}", None))
    if not tarfile.is_tarfile(path):
      raise RuntimeError("Expecting tarfile but received: %s" % path)
    contents = {}
    with tarfile.open(path, mode="r:gz") as f:
      all_files = [
          (path, info.name, info.size, verbose) for info in f if info.isfile()
      ]
    for name, data in MPI(jobs=all_files, func=_read_tarinfo, batch=1, ncpu=4):
      contents[name] = data
    # cell barcodes
    barcodes = contents['barcodes']
    ### cell-atac
    if exp == 'cell-atac':
      n_top_genes = 20000  # this is ad-hoc value
      X = contents['matrix'].T.todense()
      peaks = contents['peaks']
      X_peaks = peaks[:, 2].astype(np.float32) - peaks[:, 1].astype(np.float32)
      X_col_name = np.array([':'.join(i) for i in peaks])
      save_data = [('chromatin', X)]
      save_metadata = dict(main_omic='chromatin',
                           barcodes=barcodes,
                           chromatin_var=X_col_name)
      sco = SingleCellOMIC(X,
                           cell_id=barcodes,
                           gene_id=X_col_name,
                           omic=OMIC.chromatin,
                           name=name)
    ### cell-exp and cell-vdj
    elif exp in ('cell-exp', 'cell-vdj'):
      n_top_genes = 2000
      # feature (Id, Name, Type(antibody or gene-expression))
      X_col = contents['features'] if 'features' in contents else contents[
          'genes']
      # data matrix
      X = contents['matrix'].T
      if not isinstance(X, csr_matrix) and hasattr(X, 'tocsr'):
        X = X.tocsr()
      X = X.astype('float32')
      assert X.shape[0] == barcodes.shape[0] and X.shape[1] == X_col.shape[0]
      # antibody and gene are provided
      prot_ids = []
      gene_ids = []
      if X_col.shape[1] == 3:
        for idx, row in enumerate(X_col):
          if row[-1] == 'Antibody Capture':
            prot_ids.append(idx)
          elif row[-1] == 'Gene Expression':
            gene_ids.append(idx)
          else:
            raise ValueError(f"Unknown feature type:{row}")
      elif X_col.shape[1] == 2:
        gene_ids = slice(None, None)
      else:
        raise ValueError(f"No support for features matrix\n{X_col}")
      # Antibody ID, Antibody Name
      y = X[:, prot_ids]
      y_col = X_col[prot_ids][:, 0]  # the id
      y_col_name = X_col[prot_ids][:, 1]  # the name
      # Gene ID, Gene Name
      X = X[:, gene_ids].todense()
      X_col_name = X_col[gene_ids][:, 1]  # the name
      X_col = X_col[gene_ids][:, 0]  # the id
      assert np.min(X) >= 0 and np.max(X) < 65000, \
        f"Only support uint16 data type, given data with max={np.max(X)}"
      # data and metadata
      sco = SingleCellOMIC(X,
                           cell_id=barcodes,
                           gene_id=X_col_name,
                           omic=OMIC.transcriptomic,
                           name=name)
      save_data = [('transcriptomic', X), ('proteomic', y)]
      save_metadata = dict(main_omic='transcriptomic',
                           barcodes=barcodes,
                           transcriptomic_var=X_col_name,
                           proteomic_var=y_col_name)
    ### others
    else:
      raise NotImplementedError(f"No support for experiment: {exp}")
    ### save data and metadata
    for name, data in save_data:
      outpath = os.path.join(preprocessed_path, name)
      n_samples, n_features = data.shape
      if n_samples == 0 or n_features == 0:
        continue
      with MmapArrayWriter(outpath,
                           shape=(0, n_features),
                           dtype=np.uint16,
                           remove_exist=True) as f:
        if verbose:
          prog = tqdm(f"Saving {outpath}", total=n_samples, unit='samples')
        for s, e in batching(batch_size=5120, n=n_samples):
          x = data[s:e]
          if hasattr(x, 'todense'):
            x = x.todense()
          f.write(x)
          if verbose:
            prog.update(e - s)
        if verbose:
          prog.clear()
          prog.close()
    # save metadata
    outpath = os.path.join(preprocessed_path, 'metadata')
    with open(outpath, 'wb') as f:
      pickle.dump(save_metadata, f)
    if verbose:
      print(f"Saved metadata to path {outpath}")
    ### filter genes, follow 10x and use Cell Ranger recipe
    n_genes = sco.shape[1]
    sc.pp.recipe_zheng17(sco, n_top_genes=n_top_genes)
    if verbose:
      print(f"Filtering genes {n_genes} to {sco.shape[1]} variated genes.")
    with open(os.path.join(preprocessed_path, 'top_genes'), 'wb') as f:
      pickle.dump(sco.var_names.values, f)
  # ******************** load and return the dataset ******************** #
  omics = [
      name for name in os.listdir(preprocessed_path)
      if name not in ('metadata', 'top_genes') and '_' not in name
  ]
  with open(os.path.join(preprocessed_path, 'metadata'), 'rb') as f:
    metadata = pickle.load(f)
  with open(os.path.join(preprocessed_path, 'top_genes'), 'rb') as f:
    top_genes = pickle.load(f)
  data = {
      name: MmapArray(os.path.join(preprocessed_path, name)).astype(np.float32)
      for name in omics
  }
  main_omic = metadata['main_omic']
  X = data[main_omic]
  var_names = metadata[f'{main_omic}_var']
  if filtered_genes:
    var_ids = {j: i for i, j in enumerate(var_names)}
    ids = [var_ids[i] for i in top_genes]
    X = X[:, ids]
    var_names = var_names[ids]
  sco = SingleCellOMIC(X,
                       cell_id=metadata['barcodes'],
                       gene_id=var_names,
                       omic=main_omic,
                       name=dataset_name)
  for o in omics:
    if o != main_omic:
      sco.add_omic(omic=o,
                   X=data[o],
                   var_names=np.asarray(metadata[f'{o}_var']))
  return sco
