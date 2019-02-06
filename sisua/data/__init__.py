from __future__ import print_function, division, absolute_import

import os
import re
import copy
import pandas as pd
from numbers import Number
from six import string_types

import numpy as np
from odin.fuel import MmapData
from odin.utils import ctext, one_hot, cache_memory
from odin.utils.crypto import md5_checksum
from odin.stats import (train_valid_test_split, describe,
                        sparsity_percentage)

from sisua.data.path import EXP_DIR
from sisua.data.const import UNIVERSAL_RANDOM_SEED
from sisua.utils.others import validating_dataset

# ===========================================================================
# The dataset object
# ===========================================================================
class SingleCellDataset(object):
  """ SingleCellDataset """
  TRAIN_PERCENTAGE = 0.9

  def __init__(self, data, rowname=None, colname=None):
    super(SingleCellDataset, self).__init__()
    assert data.ndim == 2
    if rowname is None:
      rowname = ['Sample#%d' % i for i in range(data.shape[0])]
    if colname is None:
      colname = ['Feature#%d' % i for i in range(data.shape[1])]
    # ====== meta data ====== #
    self._col = np.array(colname)
    # ====== getting the data ====== #
    row = np.array(rowname)
    md5 = [None, None]
    # prepare the data
    if isinstance(data, MmapData):
      data = data[:]
    data = data.astype('float32')
    # split train, test
    rand = np.random.RandomState(seed=UNIVERSAL_RANDOM_SEED)
    ids = rand.permutation(data.shape[0])
    train_ids, test_ids = train_valid_test_split(
        x=ids,
        train=SingleCellDataset.TRAIN_PERCENTAGE,
        inc_test=False, seed=rand.randint(10e8))
    train_ids = np.array(train_ids)
    test_ids = np.array(test_ids)

    self._raw_data = (data[train_ids], data[test_ids])
    # store the indices
    self._indices = (train_ids, test_ids)
    self._n_train = len(train_ids)
    self._n_test = len(test_ids)

    # store the row
    train_row = row[train_ids]
    test_row = row[test_ids]
    self._row = (train_row, test_row)

    # check md5 checksum
    md5_row = ''.join([md5_checksum(i) for i in
                       [train_row, test_row]])
    md5_indices = ''.join([md5_checksum(i) for i in
                           [train_ids, test_ids]])
    if md5[0] is None:
      md5[0] = md5_row
    else:
      assert md5[0] == md5_row
    if md5[1] is None:
      md5[1] = md5_indices
    else:
      assert md5[1] == md5_indices
    self._md5 = tuple(md5)

    # ====== cell statistics ====== #
    train_raw, test_raw = self._raw_data
    self._cell_size = (np.sum(train_raw, axis=-1, keepdims=True),
                       np.sum(test_raw, axis=-1, keepdims=True))
    self._cell_max = max(np.max(c) for c in self._cell_size)
    self._cell_min = min(np.min(c) for c in self._cell_size)
    c = np.concatenate(self._cell_size, axis=0)
    self._cell_mean = np.mean(c)
    self._cell_std = np.std(c)
    self._cell_median = np.median(c)
    # ====== feature statistics ====== #
    x = np.sum(train_raw, axis=0, keepdims=True) + \
        np.sum(test_raw, axis=0, keepdims=True)
    self._feat_size = (np.repeat(x, repeats=self.n_train, axis=0),
                       np.repeat(x, repeats=self.n_test, axis=0))
    self._feat_mean = np.mean(x)
    self._feat_std = np.std(x)
    self._feat_max = np.max(x)
    self._feat_min = np.min(x)

  @property
  def shape(self):
    return (None, self._raw_data['raw'][0].shape[1])

  @property
  def md5(self):
    """ Return unique tuple of 3 MD5 checksums:
     * the row
     * the indices used to split dataset
    X and y should have the same row and indices checksum
    """
    return self._md5

  @property
  def n_train(self):
    return self._n_train

  @property
  def n_test(self):
    return self._n_test

  @property
  def col_name(self):
    return self._col

  @property
  def row_name(self):
    return self._row

  @property
  def indices(self):
    """ The array indices used for train, test splitting

    Return
    ------
    tuple : (train_indices, test_indices)
    """
    return self._indices

  # ******************** cell statistics ******************** #
  @property
  def cell_size(self):
    """ Tuple of three matrices:
    * train: (n_train, 1)
    * test : (n_test , 1)
    """
    return self._cell_size

  @property
  def cell_mean(self):
    return self._cell_mean

  @property
  def cell_std(self):
    return self._cell_std

  @property
  def cell_max(self):
    return self._cell_max

  @property
  def cell_median(self):
    return self._cell_median

  @property
  def cell_min(self):
    return self._cell_min

  # ******************** feature statistics ******************** #
  @property
  def feat_size(self):
    """ Tuple of three matrices (the gene count is
    repeated along 0 axis):
    * train: (n_train, feat_dim)
    * test : (n_test , feat_dim)
    """
    return self._feat_size

  @property
  def feat_dim(self):
    return self.shape[1]

  @property
  def feat_mean(self):
    return self._feat_mean

  @property
  def feat_std(self):
    return self._feat_std

  @property
  def feat_max(self):
    return self._feat_max

  @property
  def feat_min(self):
    return self._feat_min

  # ******************** helper ******************** #
  @cache_memory
  def _get_data_all(self, normalize_method='raw',
                    dropout=0, distribution="uniform"):
    # no dropout
    if dropout is None or dropout <= 0 or dropout >= 1:
      return self._raw_data

    train, test = self._raw_data
    rand = np.random.RandomState(seed=UNIVERSAL_RANDOM_SEED)
    # ====== applying corruption ====== #
    # Original code from scVI, to provide a comparable result,
    # please acknowledge the author of scVI if you are using this
    # code for corrupting the data
    # https://github.com/YosefLab/scVI/blob/2357dde15351450e452efa426c516c60a2d5ee96/scvi/dataset/dataset.py#L83
    # the test data won't be corrupted
    train = copy.deepcopy(train)
    if distribution == "uniform":  # multiply the entry n with a Ber(0.9) random variable.
      i, j = np.nonzero(train)
      ix = rand.choice(range(len(i)),
                       int(np.floor(dropout * len(i))),
                       replace=False)
      i, j = i[ix], j[ix]
      corrupted = np.multiply(
          train[i, j],
          rand.binomial(n=np.ones(len(ix), dtype=np.int32), p=0.9))
    elif distribution == "binomial":  # multiply the entry n with a Bin(n, 0.9) random variable.
      i, j = (k.ravel() for k in np.indices(self.X.shape))
      ix = rand.choice(range(len(i)),
                       int(np.floor(dropout * len(i))),
                       replace=False)
      i, j = i[ix], j[ix]
      # only 20% expression captured
      corrupted = rand.binomial(n=(train[i, j]).astype(np.int32), p=0.2)
    else:
      raise ValueError(
          "Only support 2 corruption distribution: 'uniform' and 'binomial', "
          "but given: '%s'" % distribution)

    train[i, j] = corrupted
    return (train, test)

  def __getitem__(self, key):
    return self.get_data(data_type=key, dropout=0)

  def get_data(self, data_type='all',
               dropout=0, distribution="uniform"):
    data_type = str(data_type).lower()
    assert data_type in ('all', 'train', 'test')
    new_data = self._get_data_all(dropout=dropout, distribution=distribution)
    # return
    if data_type == 'all':
      return np.concatenate(new_data, axis=0)
    if data_type == 'train':
      return new_data[0]
    if data_type == 'test':
      return new_data[1]

  # ====== shortcut ====== #
  @property
  def X(self):
    return self.get_data(data_type='all', dropout=0)

  @property
  def X_row(self):
    return np.concatenate(self.row_name)

  @property
  def X_col(self):
    return np.array(self.col_name)

  # ******************** logging ******************** #
  def __str__(self):
    s = "======== Data ========\n"

    s += ctext('Raw:', 'lightcyan') + ':' + '\n'
    for name, dat in zip(["train", "test"], self._raw_data):
      s += "  %-5s :" % name + ctext('%-15s' % str(dat.shape), 'cyan') + describe(dat, shorten=True) + '\n'
      s += "    Sparsity: %s\n" % ctext('%.2f' % sparsity_percentage(dat.astype('int32')), 'cyan')
      s += "    #ZeroCol: %s\n" % ctext(np.sum(np.sum(dat, axis=0) == 0), 'cyan')
      s += "    #ZeroRow: %s\n" % ctext(np.sum(np.sum(dat, axis=1) == 0), 'cyan')

    s += ctext('SizeFactor:', 'lightcyan') + ':' + '\n'
    s += '  mean:%f std:%f max:%f min:%f\n' % \
    (self.cell_mean, self.cell_std, self.cell_max, self.cell_min)
    train, test = self.cell_size
    s += "  train :" + ctext('%-15s' % str(train.shape), 'cyan') + describe(train, shorten=True) + '\n'
    s += "  test  :" + ctext('%-15s' % str(test.shape), 'cyan') + describe(test, shorten=True) + '\n'

    s += ctext('Feat:', 'lightcyan') + ':' + '\n'
    s += '  mean:%f std:%f max:%f min:%f\n' % \
    (self.feat_mean, self.feat_std, self.feat_max, self.feat_min)
    train, test = self.feat_size
    s += "  train :" + ctext('%-15s' % str(train.shape), 'cyan') + describe(train, shorten=True) + '\n'
    s += "  test  :" + ctext('%-15s' % str(test.shape), 'cyan') + describe(test, shorten=True)

    return s

# ===========================================================================
# Main methods
# ===========================================================================
def get_dataset(dataset_name, override=False):
  """ Supporting dataset:

  'pbmc_citeseq' :
  'pbmc_10x' :
  'pbmc' :
  'pbmc_5000' :
  'cbmc_citeseq' :
  'mnist' :
  'mnist_org' :
  'mnist_imp' :
  'fmnist' :
  'fmnist_org' :
  'fmnist_imp' :
  'facs_7' :
  'facs_5' :
  'facs_2' :
  'facs_corrupt' :
  'cortex' :
  'retina' :
  'hemato' :

  Return
  ------
  dataset: `odin.fuel.dataset.Dataset` contains original data
  cell_gene: instance of `sisua.data.SingleCellDataset` for cell-gene matrix
  cell_protein: instance of `sisua.data.SingleCellDataset` for cell-protein matrix
  """
  from sisua.data.pbmc_CITEseq import read_CITEseq_PBMC
  from sisua.data.pbmc10x_pp import read_10xPBMC_PP
  from sisua.data.cbmc_CITEseq import read_CITEseq_CBMC
  from sisua.data.mnist import read_MNIST
  from sisua.data.facs_gene_protein import read_FACS, read_full_FACS
  from sisua.data.facs_corrupted import read_FACS_corrupted
  from sisua.data.fashion_mnist import (read_fashion_MNIST, read_fashion_MNIST_drop,
                                        read_MNIST_drop)
  from sisua.data.scvi_datasets import (read_Cortex, read_Hemato, read_PBMC)

  data_meta = {
      'pbmc_citeseq': read_CITEseq_PBMC,
      'pbmc_10x': read_10xPBMC_PP,
      'pbmc': read_PBMC,
      'pbmc_5000': lambda override: read_CITEseq_PBMC(override,
                                                     version_5000genes=True),
      'cbmc_citeseq': read_CITEseq_CBMC,

      'mnist': read_MNIST,
      'mnist_org': read_MNIST,
      'mnist_imp': read_MNIST_drop,

      'fmnist': read_fashion_MNIST,
      'fmnist_org': read_fashion_MNIST,
      'fmnist_imp': read_fashion_MNIST_drop,

      'facs_7': lambda override: read_full_FACS(override=override),
      'facs_5': lambda override: read_FACS(n_protein=5, override=override),
      'facs_2': lambda override: read_FACS(n_protein=2, override=override),
      'facs_corrupt': read_FACS_corrupted,

      'cortex': read_Cortex,
      'retina': read_Cortex,
      'hemato': read_Hemato,
  }
  # ====== special case: get all dataset ====== #
  dataset_name = str(dataset_name).lower().strip()
  if dataset_name not in data_meta:
    raise RuntimeError(
        'Cannot find dataset with name: "%s", all dataset include: %s'
        % (dataset_name, ",".join(list(data_meta.keys()))))
  ds = data_meta[dataset_name](override=override)
  validating_dataset(ds)
  # ====== get particular dataset ====== #
  cell_gene = SingleCellDataset(
      data=ds['X'], rowname=ds['X_row'], colname=ds['X_col'])

  cell_prot = SingleCellDataset(
      data=ds['y'], rowname=ds['X_row'], colname=ds['y_col'])

  assert cell_gene.md5 == cell_prot.md5
  # ******************** return ******************** #
  setattr(ds, 'name', dataset_name)
  setattr(cell_gene, 'name', dataset_name)
  setattr(cell_prot, 'name', dataset_name)
  return ds, cell_gene, cell_prot

def get_dataframe(dataset_name, override=False):
  """ Return 2 tuples of the DataFrame instance of:

    * (train_gene, test_gene)
    * (train_prot, test_prot)
  """
  ds, gene_ds, prot_ds = get_dataset(dataset_name=dataset_name, override=override)
  train_gene = pd.DataFrame(data=gene_ds['train'],
                            index=gene_ds.row_name[0],
                            columns=gene_ds.col_name)
  test_gene = pd.DataFrame(data=gene_ds['test'],
                           index=gene_ds.row_name[1],
                           columns=gene_ds.col_name)

  train_prot = pd.DataFrame(data=prot_ds['train'],
                            index=prot_ds.row_name[0],
                            columns=prot_ds.col_name)
  test_prot = pd.DataFrame(data=prot_ds['test'],
                           index=prot_ds.row_name[1],
                           columns=prot_ds.col_name)
  return (train_gene, test_gene), (train_prot, test_prot)
