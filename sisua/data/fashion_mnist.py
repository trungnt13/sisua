from __future__ import print_function, division, absolute_import

import os
import shutil
import pickle

import numpy as np

from odin import fuel as F
from odin.utils import select_path, one_hot, ctext

from sisua.data.path import DOWNLOAD_DIR, PREPROCESSED_BASE_DIR

# ===========================================================================
# Helper
# ===========================================================================
def _check_override(path, override):
  if override:
    if os.path.exists(path):
      shutil.rmtree(path)
      os.mkdir(path)
  elif not os.path.exists(path):
    os.mkdir(path)

def _preprocessing_dataset(ds, outpath):
  X_train = ds['X_train'].astype('float32').reshape(-1, 784)
  X_test = ds['X_test'].astype('float32').reshape(-1, 784)

  y_train = ds['y_train'].astype('float32')
  y_test = ds['y_test'].astype('float32')

  labels = ds['labels']
  n_classes = len(labels)

  y_train = one_hot(y_train, nb_classes=n_classes)
  y_test = one_hot(y_test, nb_classes=n_classes)
  print("All classes:", ctext(', '.join(labels), 'cyan'))

  X_train = np.concatenate((X_train, X_test), axis=0)
  y_train = np.concatenate((y_train, y_test), axis=0)
  # ====== perform normalization ====== #
  example_names = np.array(["image {}".format(i + 1)
                            for i in range(X_train.shape[0])])
  feature_names = np.array(["pixel {}".format(j + 1)
                            for j in range(X_train.shape[1])])
  # ====== saving ====== #
  data_meta = {
      'X': X_train,
      'X_row': example_names,
      'X_col': feature_names,
      'y': y_train,
      'y_col': labels
  }
  for name, val in data_meta.items():
    path = os.path.join(outpath, name)
    with open(path, 'wb') as f:
      pickle.dump(val, f)


def _validate_dataset(path):
  ds = F.Dataset(path, read_only=True)
  return ds

# ===========================================================================
# Original FMNIST
# ===========================================================================
_FMNIST_PREPROCESSED = select_path(
    os.path.join(PREPROCESSED_BASE_DIR, 'FMNIST_preprocessed'),
create_new=True)

def read_fashion_MNIST(override=False):
  _check_override(_FMNIST_PREPROCESSED, override)
  # ******************** load the dataset for the first time ******************** #
  if not os.path.exists(os.path.join(_FMNIST_PREPROCESSED, 'X_train')):
    ds = F.FMNIST_original.load()
    _preprocessing_dataset(ds, outpath=_FMNIST_PREPROCESSED)
  # ====== load the dataset ====== #
  return _validate_dataset(_FMNIST_PREPROCESSED)

# ===========================================================================
# Special version of FMNIST
# 30% of the count dropped
# ===========================================================================
_FMNIST_DROP_PREPROCESSED = select_path(
    os.path.join(PREPROCESSED_BASE_DIR, 'FMNIST_drop_preprocessed'),
create_new=True)

def read_fashion_MNIST_drop(override=False):
  _check_override(_FMNIST_DROP_PREPROCESSED, override)
  # ******************** load the dataset for the first time ******************** #
  if not os.path.exists(os.path.join(_FMNIST_DROP_PREPROCESSED, 'X_train')):
    ds = F.FMNIST_dropout.load()
    _preprocessing_dataset(ds, outpath=_FMNIST_DROP_PREPROCESSED)
  # ====== load the dataset ====== #
  return _validate_dataset(_FMNIST_DROP_PREPROCESSED)

# ===========================================================================
# MNIST dropout
# ===========================================================================
_MNIST_DROP_PREPROCESSED = select_path(
    os.path.join(PREPROCESSED_BASE_DIR, 'MNIST_drop_preprocessed'),
create_new=True)

def read_MNIST_drop(override=False):
  _check_override(_MNIST_DROP_PREPROCESSED, override)
  # ******************** load the dataset for the first time ******************** #
  if not os.path.exists(os.path.join(_MNIST_DROP_PREPROCESSED, 'X_train')):
    ds = F.MNIST_dropout.load()
    _preprocessing_dataset(ds, outpath=_MNIST_DROP_PREPROCESSED)
  # ====== load the dataset ====== #
  return _validate_dataset(_MNIST_DROP_PREPROCESSED)
