from __future__ import print_function, division, absolute_import

import os
import copy

import numpy as np

from odin.fuel import MmapData
from odin import visual as vs
from odin.utils import ctext, cache_memory
from odin.utils.crypto import md5_checksum
from odin.stats import (train_valid_test_split, describe,
                        sparsity_percentage)

from sisua.data.const import UNIVERSAL_RANDOM_SEED
from sisua.utils.visualization import save_figures

# ===========================================================================
# Helper
# ===========================================================================
def apply_artificial_corruption(x, dropout, distribution):
  distribution = str(distribution).lower()
  dropout = float(dropout)
  assert 0 <= dropout < 1, \
  "dropout value must be >= 0 and < 1, given: %f" % dropout
  rand = np.random.RandomState(seed=UNIVERSAL_RANDOM_SEED)

  if dropout == 0:
    return x
  # ====== applying corruption ====== #
  # Original code from scVI, to provide a comparable result,
  # please acknowledge the author of scVI if you are using this
  # code for corrupting the data
  # https://github.com/YosefLab/scVI/blob/2357dde15351450e452efa426c516c60a2d5ee96/scvi/dataset/dataset.py#L83
  # the test data won't be corrupted
  corrupted_x = copy.deepcopy(x)

  # multiply the entry n with a Ber(0.9) random variable.
  if distribution == "uniform":
    i, j = np.nonzero(x)
    ix = rand.choice(range(len(i)),
                     size=int(np.floor(dropout * len(i))),
                     replace=False)
    i, j = i[ix], j[ix]
    corrupted = np.multiply(
        x[i, j],
        rand.binomial(n=np.ones(len(ix), dtype=np.int32), p=0.9))
  # multiply the entry n with a Bin(n, 0.9) random variable.
  elif distribution == "binomial":
    i, j = (k.ravel() for k in np.indices(x.shape))
    ix = rand.choice(range(len(i)),
                     size=int(np.floor(dropout * len(i))),
                     replace=False)
    i, j = i[ix], j[ix]
    # only 20% expression captured
    corrupted = rand.binomial(n=(x[i, j]).astype(np.int32), p=0.2)
  else:
    raise ValueError(
        "Only support 2 corruption distribution: 'uniform' and 'binomial', "
        "but given: '%s'" % distribution)

  corrupted_x[i, j] = corrupted
  return corrupted_x

def get_library_size(X, return_library_size=False):
  """ Copyright scVI authors
  https://github.com/YosefLab/scVI/blob/master/README.rst

  Original Code:
  https://github.com/YosefLab/scVI/blob/9d9a525df810c47ce482ef7b554f25fcc6482c2d/scvi/dataset/dataset.py#L288

  size factor of X in log-space

  Parameters
  ----------
  return_library_size : bool (default: False)
    if True, return the log-count library size

  Return
  ------
  local_mean (n_samples, 1)
  local_var (n_samples, 1)
  """
  assert X.ndim == 2, "Only support 2-D matrix"
  total_counts = X.sum(axis=1)
  assert np.all(total_counts > 0), "Some cell contains zero count!"
  log_counts = np.log(total_counts)
  local_mean = (np.mean(log_counts) * np.ones((X.shape[0], 1))).astype(np.float32)
  local_var = (np.var(log_counts) * np.ones((X.shape[0], 1))).astype(np.float32)
  if not return_library_size:
    return local_mean, local_var
  return np.expand_dims(log_counts, -1), local_mean, local_var


# ===========================================================================
# Main class
# ===========================================================================
class SingleCellDataset(object):
  """ SingleCellDataset """
  TRAIN_PERCENTAGE = 0.9

  def __init__(self, data, rowname=None, colname=None):
    super(SingleCellDataset, self).__init__()
    assert data.ndim == 2, "data must be a matrix [n_cells, n_features]"
    if rowname is None:
      rowname = ['Sample#%d' % i for i in range(data.shape[0])]
    if colname is None:
      colname = ['Feature#%d' % i for i in range(data.shape[1])]
    # ====== check zero and one columns ====== #
    s = data.sum(0)
    assert np.all(s > 1), \
    "All columns sum must be greater than 1 " + \
    "(i.e. non-zero and > 1 for train, test splitting)"
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
    # try splitting again until get all non-zeros columns
    # in both training an testing set
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

    # ====== library size modeling ====== #
    self._library_size = get_library_size(train_raw)

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
  def is_binary(self):
    X = self.get_data(data_type='all', dropout=0)
    return sorted(np.unique(X.astype('float32'))) == [0., 1.]

  @property
  def shape(self):
    return (None, self._raw_data[0].shape[1])

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

  @property
  def library_size(self):
    """ Return the mean and variance for library size
    modeling in log-space

    This is the only statistics from training set,
    only used for training

    NO cheating!
    """
    return self._library_size

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
  def _get_data_all(self, dropout=0, distribution="uniform"):
    # no dropout
    if dropout is None or dropout <= 0 or dropout >= 1:
      return self._raw_data
    train, test = self._raw_data
    return (
        apply_artificial_corruption(train, dropout=dropout, distribution=distribution),
        apply_artificial_corruption(test, dropout=dropout, distribution=distribution))

  def __getitem__(self, key):
    return self.get_data(data_type=key, dropout=0)

  def get_data(self, data_type='all', dropout=0, distribution="uniform"):
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
  def X_train(self):
    return self.get_data(data_type='train', dropout=0)

  @property
  def X_test(self):
    return self.get_data(data_type='test', dropout=0)

  @property
  def X_row(self):
    return np.concatenate(self.row_name)

  @property
  def X_col(self):
    return np.array(self.col_name)

  # ====== statistics ====== #
  @property
  def sparsity(self):
    return sparsity_percentage(self.get_data(data_type='all'))

  @property
  def sparsity_train(self):
    return sparsity_percentage(self.get_data(data_type='train'))

  @property
  def sparsity_test(self):
    return sparsity_percentage(self.get_data(data_type='test'))

  @property
  def n_cells_per_gene(self):
    """ Important statistics to evaluate the quality of genes
    Return a number of non-zeros cell for each gene
    """
    X = self.get_data(data_type='all')
    X = X != 0.
    return np.sum(X, axis=0)

  # ******************** plotting helper ******************** #
  def plot_percentile_histogram(self, n_hist, title=None, outlier=0.001):
    """ Data is chopped into multiple percentile (`n_hist`) and the
    histogram is plotted for each percentile.

    """
    from matplotlib import pyplot as plt
    arr = np.concatenate(
        [self.get_data(data_type='train'), self.get_data(data_type='test')],
        axis=0)

    n_percentiles = n_hist + 1
    n_col = 5
    n_row = int(np.ceil(n_hist / n_col))
    fig = vs.plot_figure(nrow=int(n_row * 1.5), ncol=20)
    percentile = np.linspace(start=np.min(arr),
                             stop=np.max(arr),
                             num=n_percentiles)
    n_samples = len(arr)
    for i, (p_min, p_max) in enumerate(zip(percentile, percentile[1:])):
      min_mask = arr >= p_min
      max_mask = arr <= p_max
      mask = np.logical_and(min_mask, max_mask)
      a = arr[mask]
      vs.plot_histogram(a, bins=120, ax=(n_row, n_col, i + 1),
                       fontsize=8,
                       color='red' if len(a) / n_samples < outlier else 'blue',
                       title=("[%s]" % title if i == 0 else "") +
                       "%d(samples)  Range:[%g, %g]" %
                       (len(a), p_min, p_max))
    plt.tight_layout()
    return fig

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
