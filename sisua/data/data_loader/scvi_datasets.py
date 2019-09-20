from __future__ import absolute_import, division, print_function

import os
import pickle
import shutil

import numpy as np

from odin.fuel import Dataset
from odin.utils import ctext, one_hot, select_path
from sisua.data.path import DOWNLOAD_DIR, PREPROCESSED_BASE_DIR



def _save_data_to_path(preprocessed_path, X, y, gene_names, label_names,
                       cell_names, verbose):
  # save data
  if verbose:
    print("Saving data to %s ..." % ctext(preprocessed_path, 'cyan'))
  with open(os.path.join(preprocessed_path, 'X'), 'wb') as f:
    pickle.dump(X, f)
  with open(os.path.join(preprocessed_path, 'y'), 'wb') as f:
    pickle.dump(y, f)
  # save the meta info
  with open(os.path.join(preprocessed_path, 'X_row'), 'wb') as f:
    pickle.dump(cell_names, f)
  with open(os.path.join(preprocessed_path, 'X_col'), 'wb') as f:
    pickle.dump(gene_names, f)
  with open(os.path.join(preprocessed_path, 'y_col'), 'wb') as f:
    pickle.dump(label_names, f)


# ===========================================================================
# Common dataset
# ===========================================================================
def _read_scvi_dataset(name, clazz_name, override, verbose):
  preprocessed_path = select_path(os.path.join(PREPROCESSED_BASE_DIR,
                                               '%s_preprocessed' % name),
                                  create_new=True)
  if override:
    shutil.rmtree(preprocessed_path)
    os.mkdir(preprocessed_path)
  # ====== copy the dataset from scVI ====== #
  if not os.path.exists(os.path.join(preprocessed_path, 'X')):
    try:
      import scvi.dataset as scvi_dataset
    except ImportError:
      raise RuntimeError("Require `scVI` package for PBMC dataset")
    clazz = getattr(scvi_dataset, clazz_name)
    gene_dataset = clazz(save_path=DOWNLOAD_DIR)

    X = gene_dataset._X
    if hasattr(X, 'todense'):
      X = np.array(X.todense())

    gene_names = np.array(gene_dataset.gene_names)
    # convert gene identifier to gene symbol (i.e. name)
    if hasattr(gene_dataset, 'de_metadata'):
      from sisua.data.utils import get_gene_id2name
      meta = gene_dataset.de_metadata
      converter = {i: j for i, j in zip(meta.ENSG, meta.GS)}
      pbmc8kconverter = get_gene_id2name()
      gene_names = np.array([
          pbmc8kconverter[i] if i in pbmc8kconverter else converter[i]
          for i in gene_names
      ])
    assert len(gene_names) == X.shape[1]

    label_names = np.array(gene_dataset.cell_types)
    y = one_hot(gene_dataset.labels.ravel(), nb_classes=len(label_names))
    assert len(label_names) == y.shape[1]

    cell_names = np.array(['Cell#%d' % i for i in range(X.shape[0])])
    _save_data_to_path(preprocessed_path, X, y, gene_names, label_names,
                       cell_names, verbose)
  # ====== read preprocessed data ====== #
  ds = Dataset(preprocessed_path, read_only=True)
  return ds


def read_Cortex(override=False, verbose=False):
  return _read_scvi_dataset(name="CORTEX",
                            clazz_name='CortexDataset',
                            override=override,
                            verbose=verbose)


def read_PBMC(override=False, verbose=False):
  return _read_scvi_dataset(name="PBMC_scVI",
                            clazz_name='PbmcDataset',
                            override=override,
                            verbose=verbose)


def read_Retina(override=False, verbose=False):
  return _read_scvi_dataset(name="RETINA",
                            clazz_name='RetinaDataset',
                            override=override,
                            verbose=verbose)


# ===========================================================================
# HEMATO dataset
# ===========================================================================
def read_Hemato(override=False, verbose=False):
  preprocessed_path = select_path(os.path.join(PREPROCESSED_BASE_DIR,
                                               'HEMATO_preprocessed'),
                                  create_new=True)

  if override:
    shutil.rmtree(preprocessed_path)
    os.mkdir(preprocessed_path)
  # ====== copy the dataset from scVI ====== #
  if not os.path.exists(os.path.join(preprocessed_path, 'X')):
    try:
      from scvi.dataset import HematoDataset
    except ImportError:
      raise RuntimeError("Require `scVI` package for HEMATO dataset")

    gene_dataset = HematoDataset(
        save_path=os.path.join(DOWNLOAD_DIR, 'HEMATO/'))

    X = gene_dataset._X
    gene_names = np.array(gene_dataset.gene_names)
    assert len(gene_names) == X.shape[1]

    y = gene_dataset.meta.values[:, 1:]
    label_names = np.array(gene_dataset.cell_types_levels)
    assert len(label_names) == y.shape[1]

    cell_names = np.array(['Cell#%d' % i for i in range(X.shape[0])])

    _save_data_to_path(preprocessed_path, X, y, gene_names, label_names,
                       cell_names, verbose)

    # create a binary classes for testing
    label_names = np.array(["Erythroblasts", "Granulocytes"])
    min_y = np.min(gene_dataset.labels)
    max_y = np.max(gene_dataset.labels)
    y_val = 2 * (gene_dataset.labels - min_y) / (max_y - min_y) - 1
    y_bin = np.argmax(
        np.hstack((
            gene_dataset.meta.iloc[:, 1].values[:, None],  # Er
            gene_dataset.meta.iloc[:, 2].values[:, None])),  # Gr
        axis=-1)
    with open(os.path.join(preprocessed_path, 'labels_name'), 'wb') as f:
      pickle.dump(label_names, f)
    with open(os.path.join(preprocessed_path, 'labels_bin'), 'wb') as f:
      pickle.dump(y_bin, f)
    with open(os.path.join(preprocessed_path, 'labels_val'), 'wb') as f:
      pickle.dump(y_val, f)
  # ====== read preprocessed data ====== #
  ds = Dataset(preprocessed_path, read_only=True)
  return ds
