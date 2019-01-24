from __future__ import print_function, division, absolute_import

import os
import re
from numbers import Number
from six import string_types

import numpy as np
from odin.fuel import MmapData
from odin.maths import edit_distance
from odin.utils import ctext, one_hot, cache_memory
from odin.utils.crypto import md5_checksum
from odin.stats import (train_valid_test_split, describe,
                        sparsity_percentage)

from sisua.data.path import EXP_DIR
from sisua.data.const import UNIVERSAL_RANDOM_SEED, DROPOUT_TEST
from sisua.utils.others import validating_dataset

# ===========================================================================
# Helper
# ===========================================================================
_space_char = re.compile("\s")
_non_alphanumeric_char = re.compile("\W")

def _normalize_text(text, sorting=True):
  text = str(text).strip().lower()
  text = re.sub("\s\s+", " ", text)
  text = _space_char.sub(' ', text)
  text = _non_alphanumeric_char.sub(' ', text)
  if bool(sorting):
    text = ' '.join(sorted(text.split(' ')))
  return text

def _find_match_dataset(dataset_name, override):
  from .pbmc_CITEseq import read_CITEseq_PBMC
  from .pbmc10x_pp import read_10xPBMC_PP
  from .cbmc_CITEseq import read_CITEseq_CBMC
  from .mnist import read_MNIST
  from .facs_gene_protein import read_FACS, read_full_FACS
  from .facs_corrupted import read_FACS_corrupted
  from .fashion_mnist import (read_fashion_MNIST, read_fashion_MNIST_drop,
                              read_MNIST_drop)

  to_dsname = lambda s: re.sub("\s\s+", " ", s).lower().replace(' ', '_')

  data_meta = {
      'PBMC CITEseq': read_CITEseq_PBMC,
      'PBMC 10x pp': read_10xPBMC_PP,
      'CBMC CITEseq': read_CITEseq_CBMC,
      'MNIST original': read_MNIST,
      'MNIST dropout': read_MNIST_drop,
      'FMNIST original': read_fashion_MNIST,
      'FMNIST dropout': read_fashion_MNIST_drop,
      'FACS 7protein': lambda override: read_full_FACS(override=override),
      'FACS 5protein': lambda override: read_FACS(n_protein=5, override=override),
      'FACS 2protein': lambda override: read_FACS(n_protein=2, override=override),
      'FACS corrupt ': read_FACS_corrupted,
      'PBMC 5000': lambda override: read_CITEseq_PBMC(override,
                                                      version_5000genes=True)
  }
  # ====== special case: get all dataset ====== #
  if _normalize_text(dataset_name) == 'all':
    out_ds = {}
    for name, func in data_meta.items():
      try:
        ds = func(override=override)
        out_ds[to_dsname(name)] = ds
      except NotImplementedError:
        pass
    print("Found dataset:",
      ctext(', '.join(sorted(out_ds.keys())), 'lightyellow'))
    return out_ds
  # ====== get particular dataset ====== #
  dataset_name = _normalize_text(dataset_name)
  all_dataset = [name for name in sorted(data_meta.keys())
                 if dataset_name in name.lower()]
  if len(all_dataset) == 0:
    all_dataset = [name for name in sorted(data_meta.keys())]

  fuzzy_match = [min(
      edit_distance(dataset_name, _normalize_text(name, sorting=True)),
      edit_distance(dataset_name, _normalize_text(name, sorting=False))
  ) for name in all_dataset]
  match = all_dataset[np.argmin(fuzzy_match)]
  print("Found dataset:", ctext(match, 'lightyellow'))

  ds = data_meta[match](override=override)
  validating_dataset(ds)
  return ds, to_dsname(match)

# ===========================================================================
# The dataset object
# ===========================================================================
class SingleCellDataset(object):
  """ SingleCellDataset """
  TRAIN_PERCENTAGE = 0.9

  def __init__(self, data, rowname=None, colname=None,
               max_clip=None):
    super(SingleCellDataset, self).__init__()
    assert data.ndim == 2
    if rowname is None:
      rowname = ['Sample#%d' % i for i in range(data.shape[0])]
    if colname is None:
      colname = ['Feature#%d' % i for i in range(data.shape[1])]
    # ====== meta data ====== #
    self._col = np.array(colname)
    if max_clip is not None and max_clip <= 0:
      max_clip = None
    self.max_clip = max_clip
    # ====== getting the data ====== #
    row = np.array(rowname)
    md5 = [None, None]
    # prepare the data
    if isinstance(data, MmapData):
      data = data[:]
    data = data.astype('float32')
    if max_clip is not None:
      data = np.clip(data, a_min=0, a_max=float(max_clip))
    # mapping: 'method_name' -> (train, test)
    self._cache_data = {'raw': data}
    # split train, test
    rand = np.random.RandomState(seed=UNIVERSAL_RANDOM_SEED)
    ids = rand.permutation(data.shape[0])
    train_ids, test_ids = train_valid_test_split(
        x=ids,
        train=SingleCellDataset.TRAIN_PERCENTAGE,
        inc_test=False, seed=rand.randint(10e8))
    train_ids = np.array(train_ids)
    test_ids = np.array(test_ids)

    self._cache_data = {
        name: (dat[train_ids], dat[test_ids])
        for name, dat in self._cache_data.items()}

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
    train_raw, test_raw = self._cache_data['raw']
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
    return (None, self._cache_data['raw'][0].shape[1])

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
  def _get_data_all(self, normalize_method='raw', dropout=0):
    # no dropout
    if dropout is None or dropout <= 0 or dropout >= 1:
      return self._cache_data[normalize_method]

    # applying dropout
    raw_data = self._cache_data['raw']
    rand = np.random.RandomState(seed=UNIVERSAL_RANDOM_SEED)
    new_data = []
    for d in raw_data:
      mask = (rand.rand(*d.shape) >= dropout).astype('float32')
      d = d * mask
      new_data.append(d)
    return tuple(new_data)

  def __getitem__(self, key):
    return self.get_data(data_type=key, dropout=0)

  def get_data(self, data_type='all', dropout=0):
    data_type = str(data_type).lower()
    assert data_type in ('all', 'train', 'test')
    new_data = self._get_data_all(dropout=dropout)
    # return
    if data_type == 'all':
      return np.concatenate(new_data, axis=0)
    if data_type == 'train':
      return new_data[0]
    if data_type == 'test':
      return new_data[1]

  # ******************** logging ******************** #
  def __str__(self):
    s = "======== Data ========\n"
    s += 'Max clip  : %s\n' % ctext(self.max_clip, 'yellow')

    s += ctext('Raw:', 'lightcyan') + ':' + '\n'
    for name, dat in zip(["train", "test"], self._cache_data['raw']):
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
def get_dataset(dataset_name,
                xclip=None, yclip=None,
                override=False):
  """ Return

  dataset, normalized dataset name, labels, cell_gene, cell_protein
  """
  ds, ds_name = _find_match_dataset(dataset_name, override)
  print(ds)

  cell_gene = SingleCellDataset(
      data=ds['X'], rowname=ds['X_row'], colname=ds['X_col'],
      max_clip=xclip)

  cell_prot = SingleCellDataset(
      data=ds['y'], rowname=ds['X_row'], colname=ds['y_col'],
      max_clip=yclip)

  assert cell_gene.md5 == cell_prot.md5
  labels = cell_prot.col_name
  # ******************** return ******************** #
  return ds, ds_name, labels, cell_gene, cell_prot