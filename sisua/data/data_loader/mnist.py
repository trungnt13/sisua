from __future__ import absolute_import, division, print_function

import gzip
import os
import pickle
import shutil
import struct

import numpy as np

from odin import fuel as F
from odin.utils import ctext, get_file, one_hot, select_path
from sisua.data.path import DOWNLOAD_DIR, PREPROCESSED_BASE_DIR

_URLs = {
    "values": {
        "training":
            "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
        "test":
            "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz"
    },
    "labels": {
        "training":
            "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
        "test":
            "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"
    }
}

_MNITS_PREPROCESSED = select_path(os.path.join(PREPROCESSED_BASE_DIR,
                                               'MNIST_preprocessed'),
                                  create_new=True)


def read_MNIST(override=False, verbose=False):
  download_path = os.path.join(DOWNLOAD_DIR, 'MNIST_original')
  if not os.path.exists(download_path):
    os.mkdir(download_path)

  if override:
    shutil.rmtree(_MNITS_PREPROCESSED)
    os.mkdir(_MNITS_PREPROCESSED)

  if len(os.listdir(_MNITS_PREPROCESSED)) == 0:
    # ====== load the values ====== #
    values = {}
    for kind, url in _URLs['values'].items():
      base_name = os.path.basename(url)
      zip_path = os.path.join(download_path, base_name)
      # download zip file
      if not os.path.exists(zip_path):
        get_file(fname=base_name,
                 origin=url,
                 outdir=download_path,
                 verbose=verbose)
      # extract the file
      with gzip.open(zip_path, "rb") as values_stream:
        _, M, r, c = struct.unpack(">IIII", values_stream.read(16))
        values_buffer = values_stream.read(M * r * c)
        values_flat = np.frombuffer(values_buffer, dtype=np.uint8)
        values[kind] = values_flat.reshape(-1, r * c)
    N = r * c

    # ====== load the labels ====== #
    labels = {}
    for kind, url in _URLs["labels"].items():
      base_name = os.path.basename(url)
      zip_path = os.path.join(download_path, base_name)
      # download zip file
      if not os.path.exists(zip_path):
        get_file(fname=base_name,
                 origin=url,
                 outdir=download_path,
                 verbose=verbose)
      # extract the file
      with gzip.open(zip_path, "rb") as labels_stream:
        _, M = struct.unpack(">II", labels_stream.read(8))
        labels_buffer = labels_stream.read(M)
        labels[kind] = np.frombuffer(labels_buffer, dtype=np.int8)

    # ====== post processing ====== #
    X_train, X_test = values["training"].astype(
        'float32'), values["test"].astype('float32')
    y_train, y_test = labels["training"], labels["test"]
    X_train = np.concatenate((X_train, X_test), axis=0)
    y_train = np.concatenate((y_train, y_test), axis=0)

    all_classes = [i for i in sorted(np.unique(y_train))]
    cls_2_idx = {c: i for i, c in enumerate(all_classes)}
    y_train = one_hot(np.array([cls_2_idx[i] for i in y_train]),
                      nb_classes=len(all_classes),
                      dtype='float32')
    all_classes = np.array(['#%d' % i for i in all_classes], dtype="U")

    X_train_row = np.array(
        ["image {}".format(i + 1) for i in range(X_train.shape[0])])

    feature_names = np.array(
        ["pixel {}".format(j + 1) for j in range(X_train.shape[1])])
    # ====== saving ====== #
    data_meta = {
        'X': X_train,
        'X_row': X_train_row,
        'X_col': feature_names,
        'y': y_train,
        'y_col': all_classes
    }
    for name, val in data_meta.items():
      path = os.path.join(_MNITS_PREPROCESSED, name)
      if verbose:
        print("Saving '%s' to path: %s" % (name, path))
      with open(path, 'wb') as f:
        pickle.dump(val, f)
  # ====== load the dataset ====== #
  ds = F.Dataset(_MNITS_PREPROCESSED, read_only=True)
  return ds
