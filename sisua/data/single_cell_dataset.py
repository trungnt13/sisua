from __future__ import absolute_import, division, print_function

import os
import warnings
from copy import deepcopy
from typing import Optional, Tuple

import numpy as np
import scanpy as sc
import scipy as sp
import tensorflow as tf
from anndata.core.alignedmapping import AxisArrays
from scipy.sparse import issparse
from six import string_types

from odin import visual as vs
from odin.fuel import MmapArrayWriter
from odin.stats import describe, sparsity_percentage, train_valid_test_split
from odin.utils import (as_tuple, batching, cache_memory, catch_warnings_ignore,
                        ctext)
from odin.utils.crypto import md5_checksum
from odin.visual import Visualizer, to_axis
from sisua.label_threshold import ProbabilisticEmbedding

# ===========================================================================
# Heuristic constants
# ===========================================================================
_DEFAULT_BATCH_SIZE = 4096


# ===========================================================================
# Helper
# ===========================================================================
def apply_artificial_corruption(x, dropout, distribution, copy=False, seed=8):
  """
  Parameters
  ----------
  x : (n_samples, n_features)
  dropout : scalar (0.0 - 1.0)
  distribution : {'uniform', 'binomial
  """
  distribution = str(distribution).lower()
  dropout = float(dropout)
  assert 0 <= dropout < 1, \
  "dropout value must be >= 0 and < 1, given: %f" % dropout
  rand = np.random.RandomState(seed=seed)

  if dropout == 0:
    return x
  # ====== applying corruption ====== #
  # Original code from scVI, to provide a comparable result,
  # please acknowledge the author of scVI if you are using this
  # code for corrupting the data
  # https://github.com/YosefLab/scVI/blob/2357dde15351450e452efa426c516c60a2d5ee96/scvi/dataset/dataset.py#L83
  # the test data won't be corrupted
  if copy:
    corrupted_x = deepcopy(x)
  else:
    corrupted_x = x

  # multiply the entry n with a Ber(0.9) random variable.
  if distribution == "uniform":
    i, j = np.nonzero(x)
    ix = rand.choice(range(len(i)),
                     size=int(np.floor(dropout * len(i))),
                     replace=False)
    i, j = i[ix], j[ix]
    corrupted = np.multiply(
        x[i, j], rand.binomial(n=np.ones(len(ix), dtype=np.int32), p=0.9))
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

  if isinstance(corrupted_x, sp.sparse.base.spmatrix):
    corrupted = type(corrupted_x)(corrupted)
  corrupted_x[i, j] = corrupted
  return corrupted_x


def get_library_size(X, return_log_count=False):
  """ Copyright scVI authors
  https://github.com/YosefLab/scVI/blob/master/README.rst

  Original Code:
  https://github.com/YosefLab/scVI/blob/9d9a525df810c47ce482ef7b554f25fcc6482c2d/scvi/dataset/dataset.py#L288

  size factor of X in log-space

  Parameters
  ----------
  X : matrix
    single-cell data matrix (n_samples, n_features)
  return_log_count : bool (default=False)
    if True, return the log-count library size

  Return
  ------
  local_mean (n_samples, 1)
  local_var (n_samples, 1)
  """
  assert X.ndim == 2, "Only support 2-D matrix"
  total_counts = X.sum(axis=1)
  assert np.all(total_counts >= 0), "Some cell contains negative-count!"
  log_counts = np.log(total_counts + 1e-8)
  local_mean = (np.mean(log_counts) * np.ones(
      (X.shape[0], 1))).astype(np.float32)
  local_var = (np.var(log_counts) * np.ones((X.shape[0], 1))).astype(np.float32)
  if not return_log_count:
    return local_mean, local_var
  return np.expand_dims(log_counts, -1), local_mean, local_var


# ===========================================================================
# OMICS
# ===========================================================================
class SingleCellOMIC(sc.AnnData, Visualizer):
  """ An annotated data matrix.

  Parameters
  ----------
  X
      A #observations × #variables data matrix. A view of the data is used
      if the data type matches, otherwise, a copy is made.
  obs
      Key-indexed one-dimensional observations annotation of length
      #observations.
  var
      Key-indexed one-dimensional variables annotation of length #variables.
  uns
      Key-index unstructured annotation.
  obsm
      Key-indexed multi-dimensional observations annotation of length
      #observations. If passing a :class:`~numpy.ndarray`,
      it needs to have a structured datatype.
  varm
      Key-indexed multi-dimensional variables annotation of length #variables.
      If passing a :class:`~numpy.ndarray`, it needs to have a structured
      datatype.
  dtype
      Data type used for storage.
  shape
      Shape tuple (#observations, #variables). Can only be provided
      if ``X`` is ``None``.
  filename
      Name of backing file. See :class:`anndata.h5py.File`.
  filemode
      Open mode of backing file. See :class:`anndata.h5py.File`.
  layers
      Dictionary with keys as layers' names and values as matrices of the
      same dimensions as X.
  """

  def __init__(self,
               X=None,
               obs=None,
               var=None,
               uns=None,
               obsm=None,
               varm=None,
               layers=None,
               raw=None,
               dtype='float32',
               shape=None,
               filename=None,
               filemode=None,
               asview=False,
               oidx=None,
               vidx=None,
               name="scOMICS"):
    if X is not None:
      if obs is None:
        obs = {'rowid': ['Row#%d' % i for i in range(X.shape[0])]}
      if var is None:
        var = {'colid': ['Col#%d' % i for i in range(X.shape[1])]}
    super(SingleCellOMIC, self).__init__(X=X,
                                         obs=obs,
                                         var=var,
                                         uns=uns,
                                         obsm=obsm,
                                         varm=varm,
                                         layers=layers,
                                         raw=raw,
                                         dtype=dtype,
                                         shape=shape,
                                         filename=filename,
                                         filemode=filemode,
                                         asview=asview,
                                         oidx=oidx,
                                         vidx=vidx)
    self._name = str(name)
    self._indices = np.arange(self.X.shape[0], dtype='int32')
    self._calculate_library_info()

  @property
  def X(self):
    with catch_warnings_ignore(FutureWarning):
      X = super(SingleCellOMIC, self).X
    if X.ndim == 1:
      X = np.expand_dims(X, axis=1)
    return X

  def assert_matching_cells(self, sco) -> 'SingleCellOMIC':
    assert isinstance(sco, SingleCellOMIC), \
      "sco must be instance of SingleCellOMIC"
    assert sco.shape[0] == self.shape[0], \
      "Number of cell mismatch %d and %d" % (self.shape[0], sco.shape[0])
    if 'cellid' in sco.obs and 'cellid' in self.obs:
      assert np.all(sco.obs['cellid'] == self.obs['cellid'])
    else:  # just check matching first column
      assert np.all(sco.obs.iloc[:, 0] == self.obs.iloc[:, 0])
    return self

  def _calculate_library_info(self):
    if sp.sparse.issparse(self.X):
      total_counts = np.expand_dims(np.sum(self.X, axis=1), axis=-1)
    else:
      total_counts = np.sum(self.X, axis=1, keepdims=True)

    log_counts, local_mean, local_var = get_library_size(self.X,
                                                         return_log_count=True)
    self._total_counts = total_counts
    self._log_counts = log_counts
    self._local_mean = local_mean
    self._local_var = local_var

  def as_obsm(self, obsm_name):
    assert obsm_name in self.obsm
    omics = SingleCellOMIC(self,
                           oidx=slice(None, None),
                           vidx=slice(None, None),
                           asview=True)
    X_org = omics._X
    omics._X = omics.obsm[obsm_name]
    # already applied obsm at least once
    if hasattr(self, '_obsm_name'):
      omics.obsm[self._obsm_name] = X_org
    else:
      omics.obsm['X_org'] = X_org
    # update the tracking information
    del omics.obsm[obsm_name]
    omics._obsm_name = obsm_name
    return omics

  def __getitem__(self, index):
    """Returns a sliced view of the object."""
    oidx, vidx = self._normalize_indices(index)
    omics = SingleCellOMIC(self, oidx=oidx, vidx=vidx, asview=True)
    # update observation indexing
    omics._obs = self._obs.iloc[oidx]
    omics._obsm = AxisArrays(omics,
                             0,
                             vals={i: j[oidx] for i, j in self._obsm.items()})
    omics._indices = self._indices[oidx]
    # update variable indexing
    omics._var = self._var.iloc[vidx]
    omics._varm = AxisArrays(omics,
                             1,
                             vals={i: j[vidx] for i, j in self._varm.items()})
    # other meta
    omics._n_obs = omics._obs.shape[0]
    omics._n_vars = omics._var.shape[0]
    omics._name = self._name + '_index'
    # library info
    omics._total_counts = self._total_counts[oidx]
    omics._log_counts = self._log_counts[oidx]
    omics._local_mean = self._local_mean[oidx]
    omics._local_var = self._local_var[oidx]
    return omics

  def apply_indices(self, indices, observation=True):
    """ Inplace indexing, this indexing algorithm also update
    `obs`, `obsm`, `var`, `varm` to complement with the new indices.

    Parameters
    ----------
    indices : array of `int` or `bool`
    observation : `bool` (default=True)
      if True, applying the indices to the observation (i.e. axis=0),
      otherwise, to the variable (i.e. axis=1)
    """
    indices = np.array(indices)
    itype = indices.dtype.type
    if not issubclass(itype, (np.bool, np.bool_, np.integer)):
      raise ValueError("indices type must be boolean or integer.")
    if observation:
      self._X = self._X[indices]
      self._obs = self._obs.iloc[indices]
      self._n_obs = self._X.shape[0]
      self._obsm = AxisArrays(
          self, 0, vals={i: j[indices] for i, j in self._obsm.items()})
      self._indices = self._indices[indices]
      self._total_counts = self._total_counts[indices]
      self._log_counts = self._log_counts[indices]
      self._local_mean = self._local_mean[indices]
      self._local_var = self._local_var[indices]
    else:
      self._X = self._X[:, indices]
      self._var = self._var.iloc[indices]
      self._n_vars = self._X.shape[1]
      self._varm = AxisArrays(
          self, 1, vals={i: j[indices] for i, j in self._varm.items()})
    return self

  def copy(self, X=None, filename=None, name=None):
    """Full copy, optionally on disk. (this code is copied from
    `AnnData`, modification to return `SingleCellOMIC` instance.
    """
    from anndata.core.views import DictView
    if not self.isbacked:
      omics = SingleCellOMIC(
          X if X is not None else
          (self._X.copy() if self._X is not None else None),
          self._obs.copy(),
          self._var.copy(),
          # deepcopy on DictView does not work and is unnecessary
          # as uns was copied already before
          self._uns.copy()
          if isinstance(self._uns, DictView) else deepcopy(self._uns),
          self._obsm.copy(),
          self._varm.copy(),
          raw=None if self._raw is None else self._raw.copy(),
          layers=dict(self.layers),
          dtype=self._X.dtype.name if self._X is not None else 'float32',
          name=self.name + '_copy' if name is None else name)
    else:
      if filename is None:
        raise ValueError(
            'To copy an SingleCellOMIC object in backed mode, '
            'pass a filename: `.copy(filename=\'myfilename.h5ad\')`.')
      if self.isview:
        self.write(filename)
      else:
        from shutil import copyfile
        copyfile(self.filename, filename)
      omics = SingleCellOMIC(filename=filename,
                             name=self.name if name is None else name)
    # other info related to SingleCellOMIC
    omics._indices = self.indices
    omics._total_counts = self._total_counts
    omics._log_counts = self._log_counts
    omics._local_mean = self._local_mean
    omics._local_var = self._local_var
    return omics

  @property
  def name(self):
    return self._name

  @property
  def is_binary(self):
    """ return True if the data is binary """
    return sorted(np.unique(self.X.astype('float32'))) == [0., 1.]

  @property
  def indices(self):
    """ Return the row indices had been used to created this data,
    helpful when using `SingleCellOMIC.split` to keep track the
    data partition """
    return self._indices

  @property
  def n_obs(self):
    """Number of observations."""
    return self._n_obs if self._X is None else self._X.shape[0]

  @property
  def n_vars(self):
    """Number of variables/features."""
    return self._n_vars if self._X is None else self._X.shape[1]

  @property
  def dtype(self):
    return self.X.dtype

  @property
  def library_size(self):
    """ Return the mean and variance for library size
    modeling in log-space """
    return self._local_mean, self._local_var

  @property
  def local_mean(self):
    return self._local_mean

  @property
  def local_var(self):
    return self._local_var

  @property
  def total_counts(self):
    return self._total_counts

  @property
  def log_counts(self):
    return self._log_counts

  # ******************** converter ******************** #
  def as_tensorflow_dataset(self, obs=(), obsm=(),
                            include_x=True) -> tf.data.Dataset:
    data = [self.X] if include_x else []
    for name in as_tuple(obs, t=string_types):
      data.append(self.obs[name].values)
    for name in as_tuple(obsm, t=string_types):
      data.append(self.obsm[name])
    assert len(data) > 0, "No data is given"
    ds = tf.data.Dataset.from_tensor_slices(data[0] if len(data) ==
                                            1 else tuple(data))
    return ds

  # ******************** transformation ******************** #
  def split(self, seed=8, train_percent=0.8) -> Tuple['SingleCellOMIC',
                                                      'SingleCellOMIC']:
    """ Spliting the data into training and test dataset

    Parameters
    ----------
    seed : `int` (default=8)
      the same seed will ensure the same partition of any `SingleCellOMIC`,
      as long as all the data has the same number of `SingleCellOMIC.nsamples`
    train_percent : `float` (default=0.8)
      the percent of data used for training, the rest is for testing
    add_name : `bool` (default=True)
      annotation the name of new data with 'train' and 'test'

    Returns
    -------
    train : `SingleCellOMIC`
    test : `SingleCellOMIC`

    Example
    -------
    >>> x_train, x_test = x.split()
    >>> y_train, y_test = y.split()
    >>> assert np.all(x_train.obs['cellid'] == y_train.obs['cellid'])
    >>> assert np.all(x_test.obs['cellid'] == y_test.obs['cellid'])
    >>> #
    >>> x_train_train, x_train_test = x_train.split()
    >>> assert np.all(x_train_train.obs['cellid'] ==
    >>>               y_train[x_train_train.indices].obs['cellid'])
    """
    assert 0 < train_percent < 1
    ids = np.random.RandomState(seed=seed).permutation(
        self.n_obs).astype('int32')
    ntrain = int(train_percent * self.n_obs)

    train_ids = ids[:ntrain]
    train = SingleCellOMIC(
        X=self.X[train_ids],
        obs=self.obs.iloc[train_ids],
        obsm={i: j[train_ids] for i, j in self.obsm.items()},
        var=self.var,
        varm=self.varm,
        uns=self.uns,  # it is tricky to split unstructed annotation
        name=self.name + '_train')
    train._indices = train_ids  # copy the indices, this is important

    test_ids = ids[ntrain:]
    test = SingleCellOMIC(
        X=self.X[test_ids],
        obs=self.obs.iloc[test_ids],
        obsm={i: j[test_ids] for i, j in self.obsm.items()},
        var=self.var,
        varm=self.varm,
        uns=self.uns,  # it is tricky to split unstructed annotation
        name=self.name + '_test')
    test._indices = test_ids  # copy the indices, this is important
    return train, test

  def corrupt(self,
              corruption_rate=0.25,
              corruption_dist='binomial',
              inplace=True,
              seed=8):
    """
    Parameters
    ----------
    corruption_rate : `float` (default=0.25)
    corruption_dist : {'binomial', 'uniform} (default='binomial')
    inplace : `bool` (default=True)
      Perform computation inplace or return new `SingleCellOMIC` with
      the corrupted data.
    seed : `int` (default=8)
        seed for the random state.

    """
    if corruption_rate <= 0:
      return self if inplace else self.copy()

    data = apply_artificial_corruption(self.X,
                                       corruption_rate,
                                       corruption_dist,
                                       copy=not inplace,
                                       seed=seed)
    name = '%s_%s%s' % (self.name, corruption_dist,
                        str(corruption_rate).split('.')[-1])
    if not inplace:
      omics = self.copy(X=data)
    else:
      omics = self
    omics._name = name
    omics._calculate_library_info()
    return omics

  def filter_highly_variable_genes(self,
                                   min_disp=0.5,
                                   max_disp=np.inf,
                                   min_mean=0.0125,
                                   max_mean=3,
                                   n_top_genes=1000,
                                   n_bins=20,
                                   flavor='cell_ranger',
                                   inplace=True):
    """ Annotate highly variable genes [Satija15]_ [Zheng17]_.

    Expects logarithmized data.

    Depending on `flavor`, this reproduces the R-implementations of Seurat
    [Satija15]_ and Cell Ranger [Zheng17]_.

    The normalized dispersion is obtained by scaling with the mean and standard
    deviation of the dispersions for genes falling into a given bin for mean
    expression of genes. This means that for each bin of mean expression, highly
    variable genes are selected.

    Parameters
    ----------
    min_disp : `float`, optional (default=0.5)
        If `n_top_genes` unequals `None`, this and all other cutoffs for
        the means and the normalized dispersions are ignored.
    max_disp : `float`, optional (default=`np.inf`)
        If `n_top_genes` unequals `None`, this and all other cutoffs for
        the means and the normalized dispersions are ignored.
    min_mean : `float`, optional (default=0.0125)
        If `n_top_genes` unequals `None`, this and all other cutoffs for
        the means and the normalized dispersions are ignored.
    max_mean : `float`, optional (default=3)
        If `n_top_genes` unequals `None`, this and all other cutoffs for
        the means and the normalized dispersions are ignored.
    n_top_genes : {`float`, int`, `None`}, optional (default=`None`)
        Number of highly-variable genes to keep., if the value is in (0, 1],
        intepret as percent of genes
    n_bins : `int`, optional (default: 20)
        Number of bins for binning the mean gene expression. Normalization is
        done with respect to each bin. If just a single gene falls into a bin,
        the normalized dispersion is artificially set to 1.
    flavor : `{'seurat', 'cell_ranger'}`, optional (default='seurat')
        Choose the flavor for computing normalized dispersion. In their default
        workflows, Seurat passes the cutoffs whereas Cell Ranger passes
        `n_top_genes`.
    inplace : `bool` (default=True)
        if False, copy the `SingleCellOMIC` and apply the vargene filter.

    Returns
    -------
    New `SingleCellOMIC` with filtered features if `applying_filter=True` else
    assign `SingleCellOMIC.highly_variable_features` with following attributes

    highly_variable : bool
        boolean indicator of highly-variable genes
    **means**
        means per gene
    **dispersions**
        dispersions per gene
    **dispersions_norm**
        normalized dispersions per gene

    Notes
    -----
    Proxy to `scanpy.pp.highly_variable_genes`
    It is recommended to do `log1p` normalization before if `flavor='seurat'`
    """
    flavor = str(flavor).lower()
    if n_top_genes is not None:
      if 0. < n_top_genes < 1.:
        n_top_genes = int(n_top_genes * self.n_vars)
      n_top_genes += 1
    # prepare the data
    # this function will take the exponential of X all the time,
    # so non-logarithmzed data might led to overflow
    omics = self if inplace else self.copy()
    sc.pp.highly_variable_genes(omics,
                                min_disp=min_disp,
                                max_disp=max_disp,
                                min_mean=min_mean,
                                max_mean=max_mean,
                                n_top_genes=n_top_genes,
                                flavor=flavor,
                                subset=True,
                                inplace=False)
    omics._name += '_vargene'
    omics._n_vars = omics._X.shape[1]
    # recalculate library info
    omics._calculate_library_info()
    return omics

  def filter_genes(self,
                   min_counts=None,
                   max_counts=None,
                   min_cells=None,
                   max_cells=None,
                   inplace=True):
    """ Filter features (columns) based on number of rows or counts.

    Keep columns that have at least ``[min_counts, max_counts]``
    or are expressed in at least ``[min_row_counts, max_row_counts]``

    Parameters
    ----------
    min_counts : {int, None} (default=None)
      Minimum number of counts required for a feature to pass filtering.
    max_counts : {int, None} (default=None)
      Maximum number of counts required for a feature to pass filtering.
    min_row_counts : {int, None} (default=None)
      Minimum number of rows expressed required for a feature to pass filtering.
    max_row_counts : {int, None} (default=None)
      Maximum number of rows expressed required for a feature to pass filtering.
    inplace : `bool` (default=True)
      if False, return new `SingleCellOMIC` with the filtered
      genes applied

    Returns
    -------
    if `applying_filter=False` annotates the `SingleCellOMIC`, otherwise,
    return new `SingleCellOMIC` with the new subset of genes

    gene_subset : `numpy.ndarray`
        Boolean index mask that does filtering. `True` means that the
        gene is kept. `False` means the gene is removed.
    number_per_gene : `numpy.ndarray`
        Depending on what was thresholded (`counts` or `cells`), the array
        stores `n_counts` or `n_cells` per gene.

    Note
    ----
    Proxy method to Scanpy preprocessing
    """
    omics = self if inplace else self.copy()
    sc.pp.filter_genes(omics,
                       min_counts=min_counts,
                       max_counts=max_counts,
                       min_cells=min_cells,
                       max_cells=max_cells,
                       inplace=True)
    omics._name += '_filtergene'
    omics._n_vars = omics._X.shape[1]
    # recalculate library info
    omics._calculate_library_info()
    return omics

  def filter_cells(self,
                   min_counts=None,
                   max_counts=None,
                   min_genes=None,
                   max_genes=None,
                   inplace=True):
    """ Filter examples (rows) based on number of features or counts.

    Keep rows that have at least ``[min_counts, max_counts]``
    or are expressed in at least ``[min_col_counts, max_col_counts]``

    Parameters
    ----------
    min_counts : {int, None} (default=None)
      Minimum number of counts required for a feature to pass filtering.
    max_counts : {int, None} (default=None)
      Maximum number of counts required for a feature to pass filtering.
    min_genes : {int, None} (default=None)
      Minimum number of rows expressed required for a feature to pass filtering.
    max_genes : {int, None} (default=None)
      Maximum number of rows expressed required for a feature to pass filtering.
    inplace : `bool` (default=True)
      if False, return new `SingleCellOMIC` with the filtered
      cells applied

    Returns
    -------
    if `applying_filter=False` annotates the `SingleCellOMIC`, otherwise,
    return new `SingleCellOMIC` with the new subset of cells

    cells_subset : numpy.ndarray
        Boolean index mask that does filtering. ``True`` means that the
        cell is kept. ``False`` means the cell is removed.
    number_per_cell : numpy.ndarray
        Depending on what was tresholded (``counts`` or ``genes``), the array stores
        ``n_counts`` or ``n_cells`` per gene.

    Note
    ----
    Proxy method to Scanpy preprocessing
    """
    # scanpy messed up here, the obs was not updated with the new indices
    cells_subset, number_per_cell = sc.pp.filter_cells(self,
                                                       min_counts=min_counts,
                                                       max_counts=max_counts,
                                                       min_genes=min_genes,
                                                       max_genes=max_genes,
                                                       inplace=False)
    omics = self if inplace else self.copy()
    omics.apply_indices(cells_subset, observation=True)
    omics._name += '_filtercell'
    return omics

  def probabilistic_embedding(self,
                              n_components_per_class=2,
                              positive_component=1,
                              log_norm=False,
                              clip_quartile=0.,
                              remove_zeros=True,
                              ci_threshold=-0.68,
                              seed=8,
                              pbe: Optional[ProbabilisticEmbedding] = None):
    """ Fit a GMM on each feature column to get the probability or binary
    representation of the features

    Parameters
    ----------
    pbe : {`sisua.ProbabilisticEmbedding`, `None`}
      pretrained instance of `ProbabilisticEmbedding` if given
    """
    # We turn-off default log_norm here since the data can be normalized
    # separately in advance.
    if self.shape[1] >= 100:
      warnings.warn("%d GMM will be trained!" % self.shape[1])

    if pbe is None:
      if 'pbe' not in self.uns:
        pbe = ProbabilisticEmbedding(
            n_components_per_class=n_components_per_class,
            positive_component=positive_component,
            log_norm=log_norm,
            clip_quartile=clip_quartile,
            remove_zeros=remove_zeros,
            ci_threshold=ci_threshold,
            random_state=seed)
        from sklearn.exceptions import ConvergenceWarning
        with catch_warnings_ignore(ConvergenceWarning):
          pbe.fit(self.X)
        self.uns['pbe'] = pbe
      else:
        pbe = self.uns['pbe']
    else:
      assert isinstance(pbe, ProbabilisticEmbedding), \
        'pbe, if given, must be instance of sisua.ProbabilisticEmbedding'
    self.obsm['X_prob'] = pbe.predict_proba(self._X)
    self.obsm['X_bin'] = pbe.predict(self._X)
    self._name += '_pbe'
    return self

  def pca(self, n_comps=50):
    """ Code from `scanpy.pp.pca`, there is bug when `chunked=True` so we
    modify it here. """
    from sklearn.decomposition import IncrementalPCA

    if self.n_vars < n_comps:
      n_comps = self.n_vars - 1

    X_pca = np.zeros((self.n_obs, n_comps), self.dtype)
    pca_ = IncrementalPCA(n_components=n_comps)

    for chunk, _, _ in self.chunked_X(_DEFAULT_BATCH_SIZE):
      chunk = chunk.toarray() if issparse(chunk) else chunk
      pca_.partial_fit(chunk)

    for chunk, start, end in self.chunked_X(_DEFAULT_BATCH_SIZE):
      chunk = chunk.toarray() if issparse(chunk) else chunk
      X_pca[start:end] = pca_.transform(chunk)

    self.obsm['X_pca'] = X_pca
    self.varm['PCs'] = pca_.components_.T
    self.uns['pca'] = {}
    self.uns['pca']['variance'] = pca_.explained_variance_
    self.uns['pca']['variance_ratio'] = pca_.explained_variance_ratio_
    return self

  def expm1(self, inplace=True):
    omics = self if inplace else self.copy()
    _expm1 = lambda x: (np.expm1(x.data, out=x.data)
                        if issparse(x) else np.expm1(x, out=x))
    for s, e in batching(n=self.n_obs, batch_size=_DEFAULT_BATCH_SIZE):
      omics._X[s:e] = _expm1(omics._X[s:e])
    omics._name += '_expm1'
    return omics

  def normalize(self,
                total_counts=False,
                log1p=False,
                scale=False,
                target_sum=None,
                exclude_highly_expressed=False,
                max_fraction=0.05,
                max_value=None,
                inplace=True):
    """
    If ``exclude_highly_expressed=True``, very highly expressed genes are
    excluded from the computation of the normalization factor (size factor)
    for each cell. This is meaningful as these can strongly influence
    the resulting normalized values for all other genes [1]_.

    Parameters
    ----------
    total_counts : bool (default=False)
      Normalize counts per cell.
    log1p : bool (default=False)
      Logarithmize the data matrix.
    scale : bool (default=False)
      Scale data to unit variance and zero mean.
    target_sum : {float, None} (default=None)
      If None, after normalization, each observation (cell) has a
      total count equal to the median of total counts for
      observations (cells) before normalization.
    exclude_highly_expressed : bool (default=False)
      Exclude (very) highly expressed genes for the computation of the
      normalization factor (size factor) for each cell. A gene is considered
      highly expressed, if it has more than ``max_fraction`` of the total counts
      in at least one cell. The not-excluded genes will sum up to
      ``target_sum``.
    max_fraction : bool (default=0.05)
      If ``exclude_highly_expressed=True``, consider cells as highly expressed
      that have more counts than ``max_fraction`` of the original total counts
      in at least one cell.
    max_value : `float` or `None`, optional (default=`None`)
        Clip (truncate) to this value after scaling. If `None`, do not clip.
    inplace : `bool` (default=True)
      if False, return new `SingleCellOMIC` with the filtered
      cells applied

    References
    ----------
    [1] Weinreb et al. (2016), SPRING: a kinetic interface for visualizing
    high dimensional single-cell expression data, bioRxiv.

    Note
    ----
    Proxy to `scanpy.pp.normalize_total`,  `scanpy.pp.log1p`
    and  `scanpy.pp.scale`
    """
    omics = self if inplace else self.copy()

    if total_counts:
      sc.pp.normalize_total(omics,
                            target_sum=target_sum,
                            exclude_highly_expressed=exclude_highly_expressed,
                            max_fraction=max_fraction,
                            inplace=True)
      # since the total counts is normalized, store the old library size
      omics._total_counts = self._total_counts
      omics._log_counts = self._log_counts
      omics._local_mean = self._local_mean
      omics._local_var = self._local_var
      omics._name += '_countnorm'

    if log1p:
      sc.pp.log1p(omics,
                  chunked=True,
                  chunk_size=_DEFAULT_BATCH_SIZE,
                  copy=False)
      omics._name += '_log1p'

    if scale:
      sc.pp.scale(omics, zero_center=True, max_value=max_value, copy=False)
      omics._name += '_scale'
    return omics

  # ====== statistics ====== #
  @property
  def sparsity(self):
    return sparsity_percentage(self.X)

  @property
  def counts_per_cell(self):
    """ Return total number of counts per cell. This method
    is scalable. """
    counts = 0
    for s, e in batching(batch_size=_DEFAULT_BATCH_SIZE, n=self.X.shape[1]):
      counts += np.sum(self.X[:, s:e], axis=1)
    return counts

  @property
  def counts_per_gene(self):
    """ Return total number of counts per gene. This method
    is scalable. """
    counts = 0
    for s, e in batching(batch_size=_DEFAULT_BATCH_SIZE, n=self.X.shape[0]):
      counts += np.sum(self.X[s:e], axis=0)
    return counts

  # ******************** metrics ******************** #
  def neighbors(self,
                n_neighbors=15,
                n_pcs=None,
                use_rep=None,
                knn=True,
                method='umap',
                metric='euclidean',
                seed=8):
    """\
    Compute a neighborhood graph of observations [McInnes18]_.

    The neighbor search efficiency of this heavily relies on UMAP [McInnes18]_,
    which also provides a method for estimating connectivities of data points -
    the connectivity of the manifold (`method=='umap'`). If `method=='gauss'`,
    connectivities are computed according to [Coifman05]_, in the adaption of
    [Haghverdi16]_.

    Parameters
    ----------
    n_neighbors : `int` (default=15)
        The size of local neighborhood (in terms of number of neighboring data
        points) used for manifold approximation. Larger values result in more
        global views of the manifold, while smaller values result in more local
        data being preserved. In general values should be in the range 2 to 100.
        If `knn` is `True`, number of nearest neighbors to be searched. If `knn`
        is `False`, a Gaussian kernel width is set to the distance of the
        `n_neighbors` neighbor.
    n_pcs : {`int`, `None`} (default=None)
        Use this many PCs. If n_pcs==0 use .X if use_rep is None.
    use_rep : {`None`, ‘X’} or any key for .obsm, optional (default=None)
        Use the indicated representation. If None, the representation is
        chosen automatically: for .n_vars < 50, .X is used, otherwise
        ‘X_pca’ is used. If ‘X_pca’ is not present, it’s computed with
        default parameters.
    knn : `bool` (default=True)
        If `True`, use a hard threshold to restrict the number of neighbors to
        `n_neighbors`, that is, consider a knn graph. Otherwise, use a Gaussian
        Kernel to assign low weights to neighbors more distant than the
        `n_neighbors` nearest neighbor.
    seed : `int` (default=8)
        A numpy random seed.
    method : {{'umap', 'gauss', `None`}}  (default: `'umap'`)
        Use 'umap' [McInnes18]_ or 'gauss' (Gauss kernel following [Coifman05]_
        with adaptive width [Haghverdi16]_) for computing connectivities.
    metric : {`str`, `callable`} (default='euclidean')
        A known metric’s name or a callable that returns a distance.

    Returns
    -------
    returns `SingleCellOMIC` with the following:

    **connectivities** : sparse matrix (`.uns['neighbors']`, dtype `float32`)
        Weighted adjacency matrix of the neighborhood graph of data
        points. Weights should be interpreted as connectivities.
    **distances** : sparse matrix (`.uns['neighbors']`, dtype `float32`)
        Instead of decaying weights, this stores distances for each pair of
        neighbors.
    """
    sc.pp.neighbors(self,
                    n_neighbors=n_neighbors,
                    knn=knn,
                    method=method,
                    metric=metric,
                    n_pcs=n_pcs,
                    use_rep=use_rep,
                    random_state=seed,
                    copy=False)
    return self

  def louvain(self,
              resolution=None,
              restrict_to=None,
              key_added='louvain',
              adjacency=None,
              flavor='vtraag',
              directed=True,
              use_weights=False,
              partition_type=None,
              partition_kwargs=None,
              seed=8):
    """Cluster cells into subgroups [Blondel08]_ [Levine15]_ [Traag17]_.

    Cluster cells using the Louvain algorithm [Blondel08]_ in the implementation
    of [Traag17]_. The Louvain algorithm has been proposed for single-cell
    analysis by [Levine15]_.

    This requires having ran :func:`~scanpy.pp.neighbors` or :func:`~scanpy.external.pp.bbknn` first,
    or explicitly passing a ``adjacency`` matrix.

    Parameters
    ----------
    resolution
        For the default flavor (``'vtraag'``), you can provide a resolution
        (higher resolution means finding more and smaller clusters),
        which defaults to 1.0. See “Time as a resolution parameter” in [Lambiotte09]_.
    restrict_to
        Restrict the clustering to the categories within the key for sample
        annotation, tuple needs to contain ``(obs_key, list_of_categories)``.
    key_added
        Key under which to add the cluster labels. (default: ``'louvain'``)
    adjacency
        Sparse adjacency matrix of the graph, defaults to
        ``adata.uns['neighbors']['connectivities']``.
    flavor : {``'vtraag'``, ``'igraph'``}
        Choose between to packages for computing the clustering.
        ``'vtraag'`` is much more powerful, and the default.
    directed
        Interpret the ``adjacency`` matrix as directed graph?
    use_weights
        Use weights from knn graph.
    partition_type
        Type of partition to use.
        Only a valid argument if ``flavor`` is ``'vtraag'``.
    partition_kwargs
        Key word arguments to pass to partitioning,
        if ``vtraag`` method is being used.
    seed
        Change the initialization of the optimization.

    Returns
    -------
    :obj:`None`
        By default (``copy=False``), updates ``adata`` with the following fields:

        ``adata.obs['louvain']`` (:class:`pandas.Series`, dtype ``category``)
            Array of dim (number of samples) that stores the subgroup id
            (``'0'``, ``'1'``, ...) for each cell.

    :class:`~anndata.AnnData`
        When ``copy=True`` is set, a copy of ``adata`` with those fields is returned.
    """
    sc.tl.louvain(self,
                  resolution=resolution,
                  random_state=seed,
                  restrict_to=restrict_to,
                  key_added=key_added,
                  adjacency=adjacency,
                  flavor=flavor,
                  directed=directed,
                  use_weights=use_weights,
                  partition_type=partition_type,
                  partition_kwargs=partition_kwargs,
                  copy=False)
    return self

  def calculate_qc_metrics(self,
                           expr_type='counts',
                           var_type='genes',
                           qc_vars=(),
                           percent_top=(50, 100, 200, 500),
                           layer=None,
                           use_raw=False,
                           parallel=None):
    """\
    Calculate quality control metrics.

    Calculates a number of qc metrics for an AnnData object, see section
    `Returns` for specifics. Largely based on `calculateQCMetrics` from scater
    [McCarthy17]_. Currently is most efficient on a sparse CSR or dense matrix.

    Parameters
    ----------
    expr_type : str
      Name of kind of values in X.
    var_type : str
      The kind of thing the variables are.
    qc_vars : Collection[str]
      Keys for boolean columns of .var which identify variables you could
      want to control for (e.g. “ERCC” or “mito”).
    percent_top : Collection[int]
      Which proportions of top genes to cover. If empty or None don’t
      calculate. Values are considered 1-indexed, percent_top=[50] finds
      cumulative proportion to the 50th most expressed gene.
    layer : str, None
      If provided, use adata.layers[layer] for expression values instead of
      adata.X.
    use_raw : bool
      If True, use adata.raw.X for expression values instead of adata.X.
    parallel : bool, None
      Whether to force parallelism. Otherwise usage of paralleism is based
      on compilation time and sample size heuristics.

    Returns
    -------
    updates `SingleCellOMIC`'s `obs` and `var`.

    Observation level metrics include:
    total_{var_type}_by_{expr_type}
      E.g. “total_genes_by_counts”. Number of genes with positive counts in
      a cell.
    total_{expr_type}
      E.g. “total_counts”. Total number of counts for a cell.
    pct_{expr_type}_in_top_{n}_{var_type} - for n in percent_top
      E.g. “pct_counts_in_top_50_genes”. Cumulative percentage of counts for
      50 most expressed genes in a cell.
    total_{expr_type}_{qc_var} - for qc_var in qc_vars
      E.g. “total_counts_mito”. Total number of counts for variabes in qc_vars.
    pct_{expr_type}_{qc_var} - for qc_var in qc_vars
      E.g. “pct_counts_mito”. Proportion of total counts for a cell which are
      mitochondrial.

    Variable level metrics include:
    total_{expr_type}
      E.g. “total_counts”. Sum of counts for a gene.
    mean_{expr_type}
      E.g. “mean counts”. Mean expression over all cells.
    n_cells_by_{expr_type}
      E.g. “n_cells_by_counts”. Number of cells this expression is measured in.
    pct_dropout_by_{expr_type}
      E.g. “pct_dropout_by_counts”. Percentage of cells this feature does not
      appear in.
    """
    sc.pp.calculate_qc_metrics(self,
                               expr_type=expr_type,
                               var_type=var_type,
                               qc_vars=qc_vars,
                               percent_top=percent_top,
                               layer=layer,
                               use_raw=use_raw,
                               parallel=parallel,
                               inplace=True)
    return self

  # ******************** plotting helper ******************** #
  def plot_percentile_histogram(self,
                                n_hist,
                                title=None,
                                outlier=0.001,
                                non_zeros=False,
                                fig=None):
    """ Data is chopped into multiple percentile (`n_hist`) and the
    histogram is plotted for each percentile.

    """
    from matplotlib import pyplot as plt
    arr = self.X
    if non_zeros:
      arr = arr[arr != 0]
    n_percentiles = n_hist + 1
    n_col = 5
    n_row = int(np.ceil(n_hist / n_col))
    if fig is None:
      fig = vs.plot_figure(nrow=int(n_row * 1.5), ncol=20)
    self.assert_figure(fig)
    percentile = np.linspace(start=np.min(arr),
                             stop=np.max(arr),
                             num=n_percentiles)
    n_samples = len(arr)
    for i, (p_min, p_max) in enumerate(zip(percentile, percentile[1:])):
      min_mask = arr >= p_min
      max_mask = arr <= p_max
      mask = np.logical_and(min_mask, max_mask)
      a = arr[mask]
      _, bins = vs.plot_histogram(
          a,
          bins=120,
          ax=(n_row, n_col, i + 1),
          fontsize=8,
          color='red' if len(a) / n_samples < outlier else 'blue',
          title=("[%s]" % title if i == 0 else "") +
          "%d(samples)  Range:[%g, %g]" % (len(a), p_min, p_max))
      plt.gca().set_xticks(np.linspace(np.min(bins), np.max(bins), num=8))
    plt.tight_layout()
    self.add_figure('percentile_%dhistogram' % n_hist, fig)
    return self

  def plot_heatmap(self, ax=None):
    raise NotImplementedError

  def plot_rank_features_stacked_violin(self, n_features=10, ax=None):
    raise NotImplementedError

  # ******************** logging and io ******************** #
  def save_to_mmaparray(self, path, dtype=None):
    """ This only save the data without row names and column names """
    with MmapArrayWriter(path=path,
                         shape=self.shape,
                         dtype=self.dtype if dtype is None else dtype,
                         remove_exist=True) as f:
      for s, e in batching(batch_size=_DEFAULT_BATCH_SIZE, n=self.n_obs):
        x = self.X[s:e]
        if dtype is not None:
          x = x.astype(dtype)
        f.write(x)

  def _get_str(self):
    all_nonzeros = []
    for s, e in batching(n=self.n_obs, batch_size=_DEFAULT_BATCH_SIZE):
      x = self.X[s:e]
      ids = np.nonzero(x)
      all_nonzeros.append(x[ids[0], ids[1]])
    all_nonzeros = np.concatenate(all_nonzeros)

    text = super(SingleCellOMIC, self).__repr__()
    text = text.replace('AnnData object', self.name)
    text += '\n    Sparsity: %.2f' % self.sparsity
    text += '\n    Nonzeros: %s' % describe(
        all_nonzeros, shorten=True, float_precision=2)
    text += '\n    Cell    : %s' % describe(
        self.counts_per_cell, shorten=True, float_precision=2)
    text += '\n    Gene    : %s' % describe(
        self.counts_per_gene, shorten=True, float_precision=2)
    text += '\n    TotalCount: %s' % describe(
        self.total_counts, shorten=True, float_precision=2)
    text += '\n    LogCount  : %s' % describe(
        self.log_counts, shorten=True, float_precision=2)
    text += '\n    LocalMean : %s' % describe(
        self.local_mean, shorten=True, float_precision=2)
    text += '\n    LocalVar  : %s' % describe(
        self.local_var, shorten=True, float_precision=2)
    return text

  def __repr__(self):
    return self._get_str()

  def __str__(self):
    return self._get_str()
