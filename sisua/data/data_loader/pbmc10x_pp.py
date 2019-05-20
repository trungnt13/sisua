# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

import os
import shutil
import pickle
import tarfile

import numpy as np
import scipy as sp
from scipy.io import mmread

from odin import fuel as F
from odin.utils import (crypto, get_file, ctext, mpi, batching,
                        select_path, one_hot)

from sisua.data.path import DOWNLOAD_DIR, PREPROCESSED_BASE_DIR

URLs = {
    "CD19+ B cells": "http://cf.10xgenomics.com/samples/cell-exp/1.1.0/b_cells/b_cells_filtered_gene_bc_matrices.tar.gz",
    "CD34+ cells": "http://cf.10xgenomics.com/samples/cell-exp/1.1.0/cd34/cd34_filtered_gene_bc_matrices.tar.gz",
    "CD4+ helper T cells": "http://cf.10xgenomics.com/samples/cell-exp/1.1.0/cd4_t_helper/cd4_t_helper_filtered_gene_bc_matrices.tar.gz",
    "CD4+/CD25+ regulatory T cells": "http://cf.10xgenomics.com/samples/cell-exp/1.1.0/regulatory_t/regulatory_t_filtered_gene_bc_matrices.tar.gz",
    "CD4+/CD45RA+/CD25- naïve T cells": "http://cf.10xgenomics.com/samples/cell-exp/1.1.0/naive_t/naive_t_filtered_gene_bc_matrices.tar.gz",
    "CD4+/CD45RO+ memory T cells": "http://cf.10xgenomics.com/samples/cell-exp/1.1.0/memory_t/memory_t_filtered_gene_bc_matrices.tar.gz",
    "CD56+ natural killer cells": "http://cf.10xgenomics.com/samples/cell-exp/1.1.0/cd56_nk/cd56_nk_filtered_gene_bc_matrices.tar.gz",
    "CD8+ cytotoxic T cells": "http://cf.10xgenomics.com/samples/cell-exp/1.1.0/cytotoxic_t/cytotoxic_t_filtered_gene_bc_matrices.tar.gz",
    "CD8+/CD45RA+ naïve cytotoxic T cells": "http://cf.10xgenomics.com/samples/cell-exp/1.1.0/naive_cytotoxic/naive_cytotoxic_filtered_gene_bc_matrices.tar.gz"
}

_10xPBMC_PREPROCESSED = select_path(
    os.path.join(PREPROCESSED_BASE_DIR, '10xPBMC_preprocessed'),
    create_new=True)

def _extract_data(name_path):
  class_name, path = name_path
  parent_paths = set()
  with tarfile.open(path, "r:gz") as tarball:
    for member in sorted(tarball, key=lambda member: member.name):
      if member.isfile():
        parent_path, filename = os.path.split(member.name)
        parent_paths.add(parent_path)
        if len(parent_paths) > 1:
          raise RuntimeError("Multiple parent directory")
        name, extension = os.path.splitext(filename)
        with tarball.extractfile(member) as data_file:
          if filename == "matrix.mtx":
            values = mmread(data_file)
          elif extension == ".tsv":
            names = np.array(data_file.read().splitlines())
            if name == "barcodes":
              example_names = names
            elif name == "genes":
              feature_names = names
  return (class_name,
          values.T,
          example_names.astype("U"),
          feature_names.astype("U"),
          os.path.split(parent_path)[-1]) # genome name


def read_10xPBMC_PP(override=False):
  download_path = os.path.join(DOWNLOAD_DIR, 'PBMC_10xPP_original')
  if not os.path.exists(download_path):
    os.mkdir(download_path)

  if override:
    shutil.rmtree(_10xPBMC_PREPROCESSED)
    os.mkdir(_10xPBMC_PREPROCESSED)
  # ====== download and extract data ====== #
  if len(os.listdir(_10xPBMC_PREPROCESSED)) == 0:
    # download data
    data_zip = {}
    for name, url in URLs.items():
      base_name = os.path.basename(url)
      out_path = os.path.join(download_path, base_name)
      if not os.path.exists(out_path):
        get_file(fname=base_name, origin=url, outdir=download_path)
      data_zip[name] = out_path
    # prepare extraction
    value_sets = {}
    example_name_sets = {}
    feature_name_sets = {}
    genome_names = {}
    # extracting data multi-processing
    print("Extracting data ...")
    for class_name, values, example_names, feature_names, genome in mpi.MPI(
        jobs=list(data_zip.items()), func=_extract_data,
        ncpu=None, batch=1):
      # store the data
      value_sets[class_name] = values
      example_name_sets[class_name] = example_names
      feature_name_sets[class_name] = feature_names
      genome_names[class_name] = genome
      # print some log
      print(ctext(class_name, 'lightyellow'))
      print('  Values:', ctext(value_sets[class_name].shape, 'cyan'),
            value_sets[class_name].dtype)
      print('  Rows  :', ctext(example_name_sets[class_name][:2], 'cyan'))
      print('  Cols  :', ctext(feature_name_sets[class_name][:2], 'cyan'))
      print('  Genome:', ctext(genome_names[class_name], 'cyan'))
    # ====== Combine data sets ====== #
    # Check for multiple genomes
    assert len(set(genome_names.values())) == 1, \
    "Multiple genomes found: %s" % str(genome_names)
    # Infer labels
    label_sets = {name: [name] * rows.shape[0]
                  for name, rows in example_name_sets.items()}

    # just order everything by class_name
    sorted_values = lambda d: [v for k, v in sorted(d.items())]
    values = sp.sparse.vstack(sorted_values(value_sets))
    example_names = np.concatenate(sorted_values(example_name_sets))
    labels = np.concatenate(sorted_values(label_sets))

    # Extract feature names and check for differences
    class_name, feature_names = feature_name_sets.popitem()
    for other_class_name, other_feature_names in feature_name_sets.items():
      if not all(feature_names == other_feature_names):
        raise ValueError(
            "The feature names for \"{}\" and \"{}\" do not match."
            .format(class_name, other_class_name))

    # save data to disk
    all_classes = np.array(sorted(np.unique(labels)),
                           dtype="U")
    cls_2_idx = {c: i for i, c in enumerate(all_classes)}
    print("All classes:", ctext(all_classes, 'lightcyan'))
    y = one_hot(np.array([cls_2_idx[l] for l in labels]),
                nb_classes=len(all_classes),
                dtype='float32')
    file_meta = {'X': values.todense().astype('float16'),
                 'X_col': feature_names,
                 'X_row': example_names,
                 'y': y,
                 'y_col': all_classes}
    for name, data in sorted(file_meta.items()):
      path = os.path.join(_10xPBMC_PREPROCESSED, name)
      print("Saving %s to %s ..." % (ctext(name, 'lightyellow'),
                                     ctext(path, 'cyan')))
      if name == 'X':
        out = F.MmapData(path=path, dtype=data.dtype,
                         shape=(0,) + data.shape[1:], read_only=False)
        for start, end in batching(batch_size=1024, n=data.shape[0]):
          x = data[start:end]
          out.append(x)
        out.flush()
        out.close()
      else:
        with open(path, 'wb') as f:
          pickle.dump(data, f)
  # ====== load and return dataset ====== #
  ds = F.Dataset(path=_10xPBMC_PREPROCESSED, read_only=True)
  return ds
