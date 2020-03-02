from __future__ import absolute_import, division, print_function

import inspect
import itertools
import os
import types
import warnings
from copy import deepcopy
from enum import Enum
from numbers import Number
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
import scipy as sp
import tensorflow as tf
from anndata._core.aligned_mapping import AxisArrays
from scipy.sparse import issparse
from six import string_types

from bigarray import MmapArrayWriter
from odin import visual as vs
from odin.search import diagonal_beam_search
from odin.stats import describe, sparsity_percentage, train_valid_test_split
from odin.utils import (IndexedList, as_tuple, batching, cache_memory,
                        catch_warnings_ignore, ctext, is_primitive)
from odin.utils.crypto import md5_checksum
from sisua.data._single_cell_visualizer import SingleCellVisualizer
from sisua.data.const import MARKER_GENES, OMIC
from sisua.data.utils import (apply_artificial_corruption, get_library_size,
                              is_binary_dtype, is_categorical_dtype,
                              standardize_protein_name)
from sisua.label_threshold import ProbabilisticEmbedding

# Heuristic constants
_BATCH_SIZE = 4096
# TODO: take into account obsp and varp


def get_all_omics(sco: sc.AnnData):
  assert isinstance(sco, sc.AnnData)
  if hasattr(sco, 'omics'):
    return sco.omics
  om = OMIC.transcriptomic
  all_omics = {o.name: o for o in OMIC}
  for k in sco.obsm.keys():
    if isinstance(k, OMIC):
      om |= k
    elif k in all_omics:
      om |= all_omics[k]
  return om


class SingleCellOMIC(sc.AnnData, SingleCellVisualizer):
  r""" An annotated data matrix.

  Arguments:
    X : a matrix of shape `[n_cells, n_rna]`, transcriptomics
    cell_name : 1-D array of cell identification.
    gene_name : 1-D array of gene/rna identification.
    dtype : specific dtype for `X`
    name : identity of the single-cell dataset
    kwargs: extra keyword arguments for `scanpy.AnnData`

  Attributes:
    pass

  Methods:
    pass
  """

  def __init__(self,
               X,
               cell_id=None,
               gene_id=None,
               dtype=None,
               name=None,
               **kwargs):
    # directly first time init from file
    if 'filename' in kwargs:
      X = None
      kwargs['dtype'] = dtype
    # init as view or copy of created SCO
    elif isinstance(X, sc.AnnData):
      self._omics = get_all_omics(X)
      self._history = IndexedList(X._history) if hasattr(X, '_history') else \
        IndexedList()
      asview = kwargs.get('asview', False)
      if asview:
        name = X._name + "_view" if name is None else str(name)
      else:
        name = X._name + "_copy" if name is None else str(name)
    # init as completely new dataset
    else:
      self._omics = OMIC.transcriptomic
      self._history = IndexedList()
      if cell_id is None:
        cell_id = ['Cell#%d' % i for i in range(X.shape[0])]
      if gene_id is None:
        gene_id = ['Gene#%d' % i for i in range(X.shape[1])]
      if dtype is None:
        dtype = X.dtype
      if name is None:
        name = "scOMICS"
      kwargs['dtype'] = dtype
      kwargs['obs'] = pd.DataFrame(index=cell_id)
      kwargs['var'] = pd.DataFrame(index=gene_id)
      kwargs['asview'] = False
    # init
    super(SingleCellOMIC, self).__init__(X, **kwargs)
    self._name = str(name)
    if OMIC.transcriptomic.name + '_var' not in self.uns:
      self.uns[OMIC.transcriptomic.name + '_var'] = self.var
    # The class is created for first time
    if not isinstance(X, sc.AnnData):
      self.obs['indices'] = np.arange(self.X.shape[0], dtype='int64')
      self._calculate_statistics(OMIC.transcriptomic)

  def _record(self, name: str, local: dict):
    method = getattr(self, name)
    specs = inspect.getfullargspec(method)
    assert inspect.ismethod(method)
    local = {
        k: v if is_primitive(v, inc_ndarray=False) else str(type(v)) \
          for k, v in local.items() \
            if not isinstance(v, SingleCellOMIC) and \
              (k in specs.args or specs.varkw is not None)
    }
    self._history[name] = local

  def add_omic(self, omic: OMIC, X: np.ndarray, var_names=None):
    self._record('add_omic', locals())
    omic = OMIC.parse(omic)
    assert X.shape[0] == self.X.shape[0], \
      "Number of samples of new omic type mismatch, given: %s, require: %s" % \
        (str(X.shape), self.X.shape[0])
    self.obsm[omic.name] = X
    # variable name
    if var_names is not None:
      var_names = np.array(var_names).ravel()
      assert len(var_names) == X.shape[1]
      if omic in (OMIC.proteomic | OMIC.celltype):
        var_names = standardize_protein_name(var_names)
    else:
      var_names = ['%s%d' % (omic.name, i) for i in range(X.shape[1])]
    self.uns[omic.name + '_var'] = pd.DataFrame(index=var_names)
    # update
    self._omics |= omic
    self._calculate_statistics(omic)
    return self

  # ******************** shape manipulation ******************** #
  def assert_matching_cells(self, sco):
    assert isinstance(sco, SingleCellOMIC), \
      "sco must be instance of SingleCellOMIC"
    assert sco.shape[0] == self.shape[0], \
      "Number of cell mismatch %d and %d" % (self.shape[0], sco.shape[0])
    if 'cellid' in sco.obs and 'cellid' in self.obs:
      assert np.all(sco.obs['cellid'] == self.obs['cellid'])
    else:  # just check matching first column
      assert np.all(sco.obs.iloc[:, 0] == self.obs.iloc[:, 0])
    return self

  def _calculate_statistics(self, omic=OMIC.transcriptomic):
    X = self.numpy(omic)
    # start processing
    if sp.sparse.issparse(X):
      total_counts = np.expand_dims(np.sum(X, axis=1), axis=-1)
    else:
      total_counts = np.sum(X, axis=1, keepdims=True)
    log_counts, local_mean, local_var = get_library_size(X,
                                                         return_log_count=True)
    self.obsm[omic.name + '_stats'] = np.hstack(
        [total_counts, log_counts, local_mean, local_var])

  def copy(self, filename=None):
    r""" Full copy, optionally on disk. (this code is copied from
    `AnnData`, modification to return `SingleCellOMIC` instance.
    """
    self._record('copy', locals())
    anndata = super().copy(filename)
    anndata._name = self.name
    sco = SingleCellOMIC(anndata, asview=False)
    return sco

  def __getitem__(self, index):
    """Returns a sliced view of the object."""
    oidx, vidx = self._normalize_indices(index)
    om = SingleCellOMIC(self, oidx=oidx, vidx=vidx, asview=True)
    om._n_obs, om._n_vars = om.X.shape
    om._X = None
    for key, X in itertools.chain(om.obsm.items(), om.obs.items()):
      assert X.shape[0] == om.n_obs, \
        "obsm of name:'%s' and shape:'%s', but the dataset has %d observations"\
          % (key, str(X.shape), om.n_obs)
    for key, X in itertools.chain(om.varm.items(), om.var.items()):
      assert X.shape[0] == om.n_vars, \
        "obsm of name:'%s' and shape:'%s', but the dataset has %d observations"\
          % (key, str(X.shape), om.n_vars)
    return om

  def apply_indices(self, indices, observation=True):
    r""" Inplace indexing, this indexing algorithm also update
    `obs`, `obsm`, `var`, `varm` to complement with the new indices.

    Arguments:
      indices : array of `int` or `bool`
      observation : `bool` (default=True)
        if True, applying the indices to the observation (i.e. axis=0),
        otherwise, to the variable (i.e. axis=1)
    """
    self._record('apply_indices', locals())
    indices = np.array(indices)
    itype = indices.dtype.type
    if not issubclass(itype, (np.bool, np.bool_, np.integer)):
      raise ValueError("indices type must be boolean or integer.")
    if observation:
      self._X = self._X[indices]
      self._n_obs = self._X.shape[0]
      self._obs = self._obs.iloc[indices]
      self._obsm = AxisArrays(
          self, 0, vals={i: j[indices] for i, j in self._obsm.items()})
    else:
      self._X = self._X[:, indices]
      self._n_vars = self._X.shape[1]
      self._var = self._var.iloc[indices]
      self._varm = AxisArrays(
          self, 1, vals={i: j[indices] for i, j in self._varm.items()})
    return self

  def split(self, train_percent=0.8, copy=True, seed=8):
    r""" Spliting the data into training and test dataset

    Arguments:
      train_percent : `float` (default=0.8)
        the percent of data used for training, the rest is for testing
      copy : a Boolean. if True, copy the data before splitting.
      seed : `int` (default=8)
        the same seed will ensure the same partition of any `SingleCellOMIC`,
        as long as all the data has the same number of `SingleCellOMIC.nsamples`

    Returns:
      train : `SingleCellOMIC`
      test : `SingleCellOMIC`

    Example:
    >>> x_train, x_test = x.split()
    >>> y_train, y_test = y.split()
    >>> assert np.all(x_train.obs['cellid'] == y_train.obs['cellid'])
    >>> assert np.all(x_test.obs['cellid'] == y_test.obs['cellid'])
    >>> #
    >>> x_train_train, x_train_test = x_train.split()
    >>> assert np.all(x_train_train.obs['cellid'] ==
    >>>               y_train[x_train_train.indices].obs['cellid'])
    """
    self._record('split', locals())
    assert 0 < train_percent < 1
    ids = np.random.RandomState(seed=seed).permutation(
        self.n_obs).astype('int32')
    ntrain = int(train_percent * self.n_obs)
    train_ids = ids[:ntrain]
    test_ids = ids[ntrain:]
    om = self.copy() if copy else self
    train = om[train_ids]
    test = om[test_ids]
    return train, test

  # ******************** properties ******************** #
  @property
  def history(self):
    r""" A dictionary recorded all methods and arguments have been called
    within this instance of `SingleCellDataset`,
    i.e. it provide a trace back of how data is preprocessed. """
    return self._history

  @property
  def indices(self):
    r""" Return the row indices had been used to created this data,
    helpful when using `SingleCellOMIC.split` to keep track the
    data partition """
    return self.obs['indices'].values

  @property
  def cell_id(self):
    return self.obs.index

  @property
  def gene_id(self):
    return self.var.index

  @property
  def marker_genes(self):
    marker_genes = set([i.lower() for i in MARKER_GENES])
    genes = [
        name for i, name in enumerate(self.gene_id)
        if name.lower() in marker_genes
    ]
    return genes

  def omic_var(self, omic):
    omic = OMIC.parse(omic)
    assert omic in self.omics
    if omic == OMIC.transcriptomic:
      return self.var
    return self.uns[omic.name + '_var']

  def numpy(self, omic=OMIC.transcriptomic):
    r""" Return observation ndarray in `obsm` or `obs`"""
    # obs
    if isinstance(omic, string_types) and \
      not isinstance(omic, OMIC) and \
        omic in self.obs:
      return self.obs[omic].values
    # obsm
    omic = OMIC.parse(omic)
    if omic not in self.omics:
      raise ValueError(
          "Cannot find OMIC type '%s', in the single-cell dataset of: %s" %
          (omic, self.omics))
    if omic == OMIC.transcriptomic:
      return self.X
    return self.obsm[omic.name]

  def labels(self, omic=OMIC.proteomic):
    omic = OMIC.parse(omic)
    return self.obs[self.labels_name(omic)]

  def labels_name(self, omic=OMIC.proteomic):
    omic = OMIC.parse(omic)
    return omic.name + '_labels'

  @property
  def omics(self):
    r"""Return all OMIC types stored in this single-cell dataset"""
    return self._omics

  @property
  def name(self):
    return self._name

  @property
  def is_log(self):
    if 'normalize' in self.history and \
      any(h['log1p'] for h in as_tuple(self.history['normalize'])):
      return True
    return False

  def is_binary(self, omic):
    r""" return True if the given OMIC type is binary """
    return is_binary_dtype(self.numpy(omic))

  def is_categorical(self, omic):
    r""" return True if the given OMIC type is binary """
    return is_categorical_dtype(self.numpy(omic))

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

  def statistics(self, omic=OMIC.transcriptomic):
    r""" Return a matrix of shape `[n_obs, 4]`.

    The columns are: 'total_counts', 'log_counts', 'local_mean', 'local_var'
    """
    return self.obsm[omic.name + '_stats']

  def library_size(self, omic=OMIC.transcriptomic):
    r""" Return the mean and variance for library size modeling in log-space """
    return self.local_mean(omic), self.local_var(omic)

  def total_counts(self, omic=OMIC.transcriptomic):
    return self.statistics(omic)[:, 0]

  def log_counts(self, omic=OMIC.transcriptomic):
    return self.statistics(omic)[:, 1]

  def local_mean(self, omic=OMIC.transcriptomic):
    return self.statistics(omic)[:, 2]

  def local_var(self, omic=OMIC.transcriptomic):
    return self.statistics(omic)[:, 3]

  def probability(self, omic=OMIC.proteomic):
    r""" Return the probability embedding of an OMIC """
    return self.probabilistic_embedding(omic=omic)[1]

  def binary(self, omic=OMIC.proteomic):
    r""" Return the binary embedding of an OMIC """
    return self.probabilistic_embedding(omic=omic)[2]

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
  def corrupt(self,
              dropout_rate=0.2,
              retain_rate=0.2,
              distribution='binomial',
              omic=OMIC.transcriptomic,
              inplace=True,
              seed=8):
    r"""
      dropout_rate : scalar (0.0 - 1.0), (default=0.25)
        how many entries (in percent) be selected for corruption.
      retain_rate : scalar (0.0 - 1.0), (default=0.2)
        how much percent of counts retained their original values.
      distribution : {'binomial', 'uniform} (default='binomial')
      omic : `sisua.data.OMIC`, which OMIC type will be corrupted
      inplace : `bool` (default=True). Perform computation inplace or return
        new `SingleCellOMIC` with the corrupted data.
      seed : `int` (default=8). Seed for the random state.
    """
    om = self if inplace else self.copy()
    om._record('corrupt', locals())
    if not (0. < retain_rate < 1. or 0. < dropout_rate < 1.):
      return om
    apply_artificial_corruption(om.numpy(omic),
                                dropout=dropout_rate,
                                retain_rate=retain_rate,
                                distribution=distribution,
                                copy=False,
                                seed=seed)
    om._calculate_statistics(omic)
    return om

  def filter_highly_variable_genes(self,
                                   min_disp=0.5,
                                   max_disp=np.inf,
                                   min_mean=0.0125,
                                   max_mean=3,
                                   n_top_genes=1000,
                                   n_bins=20,
                                   flavor='cell_ranger',
                                   inplace=True):
    r""" Annotate highly variable genes [Satija15]_ [Zheng17]_.

    Expects logarithmized data.

    Depending on `flavor`, this reproduces the R-implementations of Seurat
    [Satija15]_ and Cell Ranger [Zheng17]_.

    The normalized dispersion is obtained by scaling with the mean and standard
    deviation of the dispersions for genes falling into a given bin for mean
    expression of genes. This means that for each bin of mean expression, highly
    variable genes are selected.

    Arguments:
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

    Returns:
      New `SingleCellOMIC` with filtered features if `applying_filter=True`
        else assign `SingleCellOMIC.highly_variable_features` with following
        attributes.

      highly_variable : bool
          boolean indicator of highly-variable genes
      **means**
          means per gene
      **dispersions**
          dispersions per gene
      **dispersions_norm**
          normalized dispersions per gene

    Notes:
      Proxy to `scanpy.pp.highly_variable_genes`. It is recommended to do
      `log1p` normalization before if `flavor='seurat'`.
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
    omics._record('filter_highly_variable_genes', locals())
    sc.pp.highly_variable_genes(omics,
                                min_disp=min_disp,
                                max_disp=max_disp,
                                min_mean=min_mean,
                                max_mean=max_mean,
                                n_top_genes=n_top_genes,
                                n_bins=int(n_bins),
                                flavor=flavor,
                                subset=True,
                                inplace=False)
    omics._name += '_vargene'
    omics._n_vars = omics._X.shape[1]
    # recalculate library info
    omics._calculate_statistics()
    return omics

  def filter_genes(self,
                   min_counts=None,
                   max_counts=None,
                   min_cells=None,
                   max_cells=None,
                   inplace=True):
    r""" Filter features (columns) based on number of rows or counts.

    Keep columns that have at least ``[min_counts, max_counts]``
    or are expressed in at least ``[min_row_counts, max_row_counts]``

    Arguments:
      min_counts : {int, None} (default=None)
        Minimum number of counts required for a gene to pass filtering.
      max_counts : {int, None} (default=None)
        Maximum number of counts required for a gene to pass filtering.
      min_cells : {int, None} (default=None)
        Minimum number of cells expressed required for a feature to pass filtering.
      max_cells : {int, None} (default=None)
        Maximum number of cells expressed required for a feature to pass filtering.
      inplace : `bool` (default=True)
        if False, return new `SingleCellOMIC` with the filtered
        genes applied

    Returns:
      if `applying_filter=False` annotates the `SingleCellOMIC`, otherwise,
      return new `SingleCellOMIC` with the new subset of genes

      gene_subset : `numpy.ndarray`
          Boolean index mask that does filtering. `True` means that the
          gene is kept. `False` means the gene is removed.
      number_per_gene : `numpy.ndarray`
          Depending on what was thresholded (`counts` or `cells`), the array
          stores `n_counts` or `n_cells` per gene.

    Note:
      Proxy method to Scanpy preprocessing
    """
    omics = self if inplace else self.copy()
    omics._record('filter_genes', locals())
    sc.pp.filter_genes(omics,
                       min_counts=min_counts,
                       max_counts=max_counts,
                       min_cells=min_cells,
                       max_cells=max_cells,
                       inplace=True)
    omics._name += '_filtergene'
    omics._n_vars = omics._X.shape[1]
    # recalculate library info
    omics._calculate_statistics()
    return omics

  def filter_cells(self,
                   min_counts=None,
                   max_counts=None,
                   min_genes=None,
                   max_genes=None,
                   inplace=True):
    r""" Filter examples (rows) based on number of features or counts.

    Keep rows that have at least ``[min_counts, max_counts]``
    or are expressed in at least ``[min_col_counts, max_col_counts]``

    Arguments:
      min_counts : {int, None} (default=None)
        Minimum number of counts required for a cell to pass filtering.
      max_counts : {int, None} (default=None)
        Maximum number of counts required for a cell to pass filtering.
      min_genes : {int, None} (default=None)
        Minimum number of genes expressed required for a cell to pass filtering.
      max_genes : {int, None} (default=None)
        Maximum number of genes expressed required for a cell to pass filtering.
      inplace : `bool` (default=True)
        if False, return new `SingleCellOMIC` with the filtered
        cells applied

    Returns:
      if `applying_filter=False` annotates the `SingleCellOMIC`, otherwise,
      return new `SingleCellOMIC` with the new subset of cells

      cells_subset : numpy.ndarray
          Boolean index mask that does filtering. ``True`` means that the
          cell is kept. ``False`` means the cell is removed.
      number_per_cell : numpy.ndarray
          Depending on what was tresholded (``counts`` or ``genes``), the array stores
          ``n_counts`` or ``n_cells`` per gene.

    Note:
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
    omics._record('filter_cells', locals())
    omics.apply_indices(cells_subset, observation=True)
    omics._name += '_filtercell'
    # recalculate library info
    omics._calculate_statistics()
    return omics

  def probabilistic_embedding(self,
                              omic=OMIC.proteomic,
                              n_components_per_class=2,
                              positive_component=1,
                              log_norm=False,
                              clip_quartile=0.,
                              remove_zeros=True,
                              ci_threshold=-0.68,
                              seed=8,
                              pbe: Optional[ProbabilisticEmbedding] = None):
    r""" Fit a GMM on each feature column to get the probability or binary
    representation of the features

    Return:
      `ProbabilisticEmbedding` model
      np.ndarray : probabilities X
      np.ndarray : binary X

    Arguments:
      pbe : {`sisua.ProbabilisticEmbedding`, `None`}, optional pretrained
        instance of `ProbabilisticEmbedding`
    """
    self._record('probabilistic_embedding', locals())
    # We turn-off default log_norm here since the data can be normalized
    # separately in advance.
    omic = OMIC.parse(omic)
    X = self.numpy(omic)
    if X.shape[1] >= 100:
      warnings.warn("%d GMM will be trained!" % self.shape[1])
    name = omic.name
    pbe_name = '%s_pbe' % name
    prob_name = '%s_prob' % name
    bin_name = '%s_bin' % name
    label_name = self.labels_name(name)

    if is_binary_dtype(X):
      X_prob = X
      X_bin = X
      self.uns[pbe_name] = None
    else:
      if pbe is None:
        if pbe_name not in self.uns:
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
            pbe.fit(X)
          self.uns[pbe_name] = pbe
        else:
          pbe = self.uns[pbe_name]
      else:
        assert isinstance(pbe, ProbabilisticEmbedding), \
          'pbe, if given, must be instance of sisua.ProbabilisticEmbedding'
      # make prediction
      X_prob = np.clip(pbe.predict_proba(X), 0. + 1e-8, 1. - 1e-8)
      X_bin = pbe.predict(X)
    # store the data
    if prob_name not in self.obsm:
      self.obsm[prob_name] = X_prob
    if label_name not in self.obs and name + '_var' in self.uns:
      omic_id = self.omic_var(name).index
      labels = [omic_id[i] for i in np.argmax(self.obsm[prob_name], axis=1)]
      self.obs[label_name] = pd.Categorical(labels)
    if bin_name not in self.obsm:
      self.obsm[bin_name] = X_bin
    return pbe, self.obsm[prob_name], self.obsm[bin_name]

  def dimension_reduce(self,
                       omic=OMIC.transcriptomic,
                       n_components=100,
                       algo='pca',
                       random_state=1):
    r""" Perform dimension reduction on given OMIC data. """
    self._record('dimension_reduce', locals())
    from sklearn.decomposition import IncrementalPCA
    algo = str(algo).lower().strip()
    assert algo in ('pca', 'tsne', 'umap')
    omic = OMIC.parse(omic)
    name = '%s_%s' % (omic.name, algo)
    ## already transformed
    if name in self.obsm:
      return self.obsm[name] if n_components is None else \
        self.obsm[name][:, :int(n_components)]
    X = self.numpy(omic)
    ## train new PCA model
    if algo == 'pca':
      X_ = np.empty(shape=(X.shape[0], n_components), dtype=X.dtype)
      model = IncrementalPCA(n_components=n_components)
      # fitting
      for start, end in batching(_BATCH_SIZE, n=X.shape[0]):
        chunk = X[start:end]
        chunk = chunk.toarray() if issparse(chunk) else chunk
        model.partial_fit(chunk)
      # transforming
      for start, end in batching(_BATCH_SIZE, n=X.shape[0]):
        chunk = X[start:end]
        chunk = chunk.toarray() if issparse(chunk) else chunk
        X_[start:end] = model.transform(chunk)
    ## TSNE
    elif algo == 'tsne':
      from multiprocessing import cpu_count
      self.dimension_reduce(omic,
                            n_components=n_components,
                            algo='pca',
                            random_state=random_state)
      sc.tl.tsne(self,
                 n_pcs=n_components,
                 use_rep=omic.name + '_pca',
                 copy=False,
                 n_jobs=max(1, cpu_count() - 1),\
                 random_state=random_state)
      X_ = self.obsm['X_tsne']
      model = None
      del self.obsm['X_tsne']
    ## UMAP
    elif algo == 'umap':
      try:
        import cuml
        method = 'rapids'
      except ImportError:
        method = 'umap'
      nb = self.neighbors(omic, method='umap', random_state=random_state)
      self.uns['neighbors'] = nb
      with catch_warnings_ignore(UserWarning):
        sc.tl.umap(self, method=method, random_state=random_state, copy=False)
      X_ = self.obsm['X_umap']
      model = self.uns['umap']
      del self.obsm['X_umap']
      del self.uns['umap']
      del self.uns['neighbors']
    ## store and return the result
    self.obsm[name] = X_
    # the model could be None, in case of t-SNE
    self.uns[name] = model
    return self.obsm[name] if n_components is None else \
      self.obsm[name][:, :int(n_components)]

  def expm1(self, omic=OMIC.transcriptomic, inplace=True):
    om = self if inplace else self.copy()
    om._record('expm1', locals())
    _expm1 = lambda x: (np.expm1(x.data, out=x.data)
                        if issparse(x) else np.expm1(x, out=x))
    X = om.numpy(omic)
    for s, e in batching(n=self.n_obs, batch_size=_BATCH_SIZE):
      X[s:e] = _expm1(X[s:e])
    om._calculate_statistics(omic)
    return om

  def normalize(self,
                omic=OMIC.transcriptomic,
                total=False,
                log1p=False,
                scale=False,
                target_sum=None,
                exclude_highly_expressed=False,
                max_fraction=0.05,
                max_value=None,
                inplace=True):
    r""" If ``exclude_highly_expressed=True``, very highly expressed genes are
    excluded from the computation of the normalization factor (size factor)
    for each cell. This is meaningful as these can strongly influence
    the resulting normalized values for all other genes [1]_.

    Arguments:
      total : bool (default=False). Normalize counts per cell.
      log1p : bool (default=False). Logarithmize the data matrix.
      scale : bool (default=False). Scale data to unit variance and zero mean.
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

    References:
      Weinreb et al. (2016), SPRING: a kinetic interface for visualizing
        high dimensional single-cell expression data, bioRxiv.

    Note:
      Proxy to `scanpy.pp.normalize_total`,  `scanpy.pp.log1p` and
        `scanpy.pp.scale`
    """
    om = self if inplace else self.copy()
    om._record('normalize', locals())
    if omic != OMIC.transcriptomic:
      org_X = om._X
      om._X = om.numpy(omic)

    if total:
      sc.pp.normalize_total(om,
                            target_sum=target_sum,
                            exclude_highly_expressed=exclude_highly_expressed,
                            max_fraction=max_fraction,
                            inplace=True)
      # since the total counts is normalized, store the old library size
      om._name += '_total'

    if log1p:
      sc.pp.log1p(om, chunked=True, chunk_size=_BATCH_SIZE, copy=False)
      om._name += '_log1p'
      del om.uns['log1p']
    # scaling may result negative total counts
    if scale:
      sc.pp.scale(om, zero_center=True, max_value=max_value, copy=False)
      om._name += '_scale'

    if omic != OMIC.transcriptomic:
      om.obsm[omic.name] = om.X
      om._X = org_X
    om._calculate_statistics(omic)
    return om

  # ====== statistics ====== #
  def sparsity(self, omic=OMIC.transcriptomic):
    return sparsity_percentage(self.numpy(omic))

  def counts_per_cell(self, omic=OMIC.transcriptomic):
    r""" Return total number of counts per cell. This method
    is scalable. """
    counts = 0
    X = self.numpy(omic)
    for s, e in batching(batch_size=_BATCH_SIZE, n=X.shape[1]):
      counts += np.sum(X[:, s:e], axis=1)
    return counts

  def counts_per_gene(self, omic=OMIC.transcriptomic):
    r""" Return total number of counts per gene. This method
    is scalable. """
    counts = 0
    X = self.numpy(omic)
    for s, e in batching(batch_size=_BATCH_SIZE, n=X.shape[0]):
      counts += np.sum(X[s:e], axis=0)
    return counts

  # ******************** metrics ******************** #
  def neighbors(self,
                omic=OMIC.transcriptomic,
                n_neighbors=15,
                n_pcs=100,
                knn=True,
                method='umap',
                metric='euclidean',
                random_state=1):
    r"""\
    Compute a neighborhood graph of observations [McInnes18]_.

    The neighbor search efficiency of this heavily relies on UMAP [McInnes18]_,
    which also provides a method for estimating connectivities of data points -
    the connectivity of the manifold (`method=='umap'`). If `method=='gauss'`,
    connectivities are computed according to [Coifman05]_, in the adaption of
    [Haghverdi16]_.

    Arguments:
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
        if n_pcs==None, use obsm['X_pca'].
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
      method : {{'umap', 'gauss', `rapids`}}  (default: `'umap'`)
        for computing connectivities.
      metric : {`str`, `callable`} (default='euclidean')
        A known metric’s name or a callable that returns a distance.

    Returns:
      returns neighbors object with the following:

      **connectivities** : sparse matrix (`.uns['neighbors']`, dtype `float32`)
          Weighted adjacency matrix of the neighborhood graph of data
          points. Weights should be interpreted as connectivities.
      **distances** : sparse matrix (`.uns['neighbors']`, dtype `float32`)
          Instead of decaying weights, this stores distances for each pair of
          neighbors.
    """
    self._record('neighbors', locals())
    omic = OMIC.parse(omic)
    name = omic + '_neighbors'
    if name not in self.uns:
      self.dimension_reduce(omic, algo='pca', random_state=random_state)
      sc.pp.neighbors(self,
                      n_neighbors=n_neighbors,
                      knn=knn,
                      method=method,
                      metric=metric,
                      n_pcs=int(n_pcs),
                      use_rep=omic.name + '_pca',
                      random_state=random_state,
                      copy=False)
      neighbors = self.uns['neighbors']
      del self.uns['neighbors']
      self.uns[name] = neighbors
    return self.uns[name]

  def clustering(self,
                 omic=OMIC.transcriptomic,
                 n_clusters=OMIC.proteomic,
                 n_init='auto',
                 dimension_reduction=None,
                 matching_labels=True,
                 algo='kmeans',
                 random_state=1234):
    r""" k-Means clustering for given OMIC type

    Arguments:
      matching_labels : a Boolean. Matching OMIC var_names to appropriate
        clusters, only when `n_clusters` is string or OMIC type.
    """
    self._record('clustering', locals())
    ## clustering algorithm
    from sklearn.cluster import MiniBatchKMeans, AgglomerativeClustering
    from sklearn.neighbors import KNeighborsClassifier
    algo = str(algo).strip().lower()
    ## input data
    omic = OMIC.parse(omic)
    cluster_omic = None
    if isinstance(n_clusters, Number):
      n_clusters = int(n_clusters)
    elif isinstance(n_clusters, string_types):
      cluster_omic = OMIC.parse(n_clusters)
      n_clusters = self.numpy(cluster_omic).shape[1]
      if n_clusters > 20:
        warnings.warn("Found omic type: '%s' with %d clusters" %
                      (cluster_omic.name, n_clusters))
    n_clusters = int(n_clusters)
    n_init = int(n_init) if isinstance(n_init, Number) else int(n_clusters) * 2
    ## check if output already extracted
    output_name = '%s_%s%d' % (omic.name, algo, n_clusters)
    if output_name in self.obs:
      return self.obs[output_name]
    ## get appropriate input data
    if dimension_reduction is None:
      X = self.numpy(omic)
    else:
      X = self.dimension_reduce(omic,
                                n_components=None,
                                algo=dimension_reduction,
                                random_state=random_state)
    ## fit KMeans
    if algo == 'kmeans':  # KMeans
      model = MiniBatchKMeans(n_clusters=int(n_clusters),
                              max_iter=100,
                              n_init=int(n_init),
                              compute_labels=False,
                              batch_size=_BATCH_SIZE // 5,
                              random_state=random_state)
      # better suffering the batch
      for s, e in batching(_BATCH_SIZE, self.n_obs, seed=random_state):
        x = X[s:e]
        model.partial_fit(x)
      # make prediction
      labels = []
      for s, e in batching(_BATCH_SIZE, self.n_obs):
        x = X[s:e]
        labels.append(model.predict(x))
      labels = np.concatenate(labels, axis=0)
    ## fit KNN
    elif algo == 'knn':
      from sklearn.cluster import SpectralClustering
      neighbors = self.neighbors(omic)
      model = None
      labels = SpectralClustering(
          n_clusters=n_clusters,
          random_state=random_state,
          n_init=n_init,
          affinity='precomputed_nearest_neighbors',
          n_neighbors=neighbors['params']['n_neighbors'] - 1).fit_predict(
              neighbors['connectivities'])
    else:
      raise NotImplementedError(algo)
    ## correlation matrix
    if cluster_omic is not None and matching_labels:
      _, X, _ = self.probabilistic_embedding(cluster_omic)
      # omic-cluster correlation matrix
      corr = np.empty(shape=(X.shape[1], n_clusters), dtype=np.float32)
      for i, x in enumerate(X.T):
        for lab in range(n_clusters):
          mask = labels == lab
          corr[i, lab] = np.sum(x[mask])
      ids = diagonal_beam_search(corr)
      varnames = self.omic_var(cluster_omic).index
      labels_to_omic = {lab: name for lab, name, in zip(ids, varnames)}
      labels = np.array([labels_to_omic[i] for i in labels])
    ## saving data and model
    self.obs[output_name] = pd.Categorical(labels)
    # self.uns[output_name] = model
    return labels

  def louvain(self,
              omic=OMIC.transcriptomic,
              resolution=None,
              restrict_to=None,
              adjacency=None,
              flavor='vtraag',
              directed=True,
              use_weights=False,
              partition_type=None,
              partition_kwargs={},
              random_state=1):
    r"""Cluster cells into subgroups [Blondel08]_ [Levine15]_ [Traag17]_.

    Cluster cells using the Louvain algorithm [Blondel08]_ in the implementation
    of [Traag17]_. The Louvain algorithm has been proposed for single-cell
    analysis by [Levine15]_.

    This requires having ran :func:`~scanpy.pp.neighbors` or :func:`~scanpy.external.pp.bbknn` first,
    or explicitly passing a ``adjacency`` matrix.

    Arguments:
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
      random_state : Change the initialization of the optimization.
    """
    self._record('louvain', locals())
    try:
      import louvain
    except ImportError:
      raise ImportError("pip install louvain>=0.6 python-igraph")
    omic = OMIC.parse(omic)
    nb = self.neighbors(omic)
    self.uns['neighbors'] = nb
    sc.tl.louvain(self,
                  resolution=resolution,
                  random_state=random_state,
                  restrict_to=restrict_to,
                  key_added=omic.name + '_louvain',
                  adjacency=adjacency,
                  flavor=flavor,
                  directed=directed,
                  use_weights=use_weights,
                  partition_type=partition_type,
                  partition_kwargs=partition_kwargs,
                  copy=False)
    del self.uns['neighbors']
    model = self.uns['louvain']
    del self.uns['louvain']
    self.uns[omic.name + '_louvain'] = model
    return self.obs[omic.name + '_louvain'].values

  # ******************** Genes metrics and ranking ******************** #
  def top_genes(self, n_genes=100, return_indices=False):
    r""" The genes that, top highly variated, less dropout
    (i.e. smallest counts of zero-values), and appeared in most cells
    will be returned.

    Arguments:
      return_indices : a Boolean. If True, return the index of top genes,
        otherwise, return the genes' ID.
    """
    self.quality_metrics()
    highly_var = self.var['highly_variable']
    n_cells = self.var['n_cells_by_counts'][highly_var].values
    n_cells = (n_cells - np.min(n_cells)) / (np.max(n_cells) - np.min(n_cells))
    zero_counts = self.var['pct_dropout_by_counts'][highly_var].values
    zero_counts = (zero_counts - np.min(zero_counts)) / \
      (np.max(zero_counts) - np.min(zero_counts))
    # higher is better
    rating = (n_cells + (1. - zero_counts)) / 2
    ids = np.argsort(rating)[::-1]
    # indexing the genes
    genes = np.arange(self.n_vars, dtype=np.int64) \
      if return_indices else self.gene_id.values
    genes = genes[highly_var][ids][:n_genes]
    return genes

  def rank_genes_groups(self,
                        n_genes=100,
                        groupby=OMIC.proteomic,
                        clustering=None,
                        method='logreg',
                        corr_method='benjamini-hochberg'):
    r"""
    method :
      't-test_overestim_var' overestimates variance of each group,
      `'t-test'` uses t-test, `'wilcoxon'` uses Wilcoxon rank-sum,
      `'logreg'` uses logistic regression.
    corr_method :  p-value correction method.
      Used only for `'t-test'`, `'t-test_overestim_var'`, and `'wilcoxon'`.

    """
    self._record('rank_genes_groups', locals())
    x = self.numpy(groupby)
    if x.ndim > 1:
      omic = OMIC.parse(groupby)
      # clustering and community detection
      if clustering is not None:
        clustering = str(clustering).lower().strip()
        clustering_method = getattr(self, clustering, None)
        if clustering_method is not None:
          clustering_method(omic)
          groupby = omic.name + '_%s' % clustering
        else:
          raise ValueError("No support for clustering method: %s" % clustering)
      # just probabilistic embedding
      else:
        self.probabilistic_embedding(omic)
        groupby = self.labels_name(omic)
    elif isinstance(groupby, OMIC):
      groupby = groupby.name
    sc.tl.rank_genes_groups(self,
                            groupby=groupby,
                            n_genes=int(n_genes),
                            method=method,
                            corr_method=corr_method,
                            copy=False,
                            key_added='%s_rank' % groupby)
    return self

  def calculate_quality_metrics(self,
                                n_bins=20,
                                flavor='cell_ranger',
                                percent_top=100,
                                use_raw=False):
    r"""\
    Calculate quality control metrics for both the observations and variable.
    Highly variable genes (i.e. variables) also calculated.

    Arguments:
      n_bins
        Number of bins for binning the mean gene expression. Normalization is
        done with respect to each bin. If just a single gene falls into a bin,
        the normalized dispersion is artificially set to 1. You'll be informed
        about this if you set `settings.verbosity = 4`.
      flavor
        Choose the flavor for computing normalized dispersion. In their default
        workflows, Seurat passes the cutoffs whereas Cell Ranger passes
        `n_top_genes`.
      percent_top : a list of Integer. Which proportions of top genes to cover.
        If empty or None don’t calculate. Values are considered 1-indexed,
        percent_top=[50] finds cumulative proportion to the 50th most
        expressed gene.
      use_raw : a Boolean. If True, use adata.raw.X for expression values
        instead of adata.X.

    Observation level metrics include:
      "n_genes_by_counts". Number of genes with positive counts in a cell.
      "total_counts". Total number of counts for a cell.
      "pct_counts_in_top_50_genes". Cumulative percentage of counts for 50 most
        expressed genes in a cell.

    Variable level metrics include:
      "total_counts". Sum of counts for a gene.
      "mean_counts". Mean expression over all cells.
      "n_cells_by_counts". Number of cells this expression is measured in.
      "pct_dropout_by_counts". Percentage of cells this feature does not
        appear in.
      "highly_variable" : boolean indicator of highly-variable genes
      "dispersions" : dispersions per gene
      "dispersions_norm" : normalized dispersions per gene

    """
    self._record('calculate_quality_metrics', locals())
    sc.pp.calculate_qc_metrics(self,
                               percent_top=as_tuple(percent_top, t=int)
                               if percent_top is not None else None,
                               use_raw=use_raw,
                               inplace=True)
    # Expects logarithmized data.
    if not self.is_log:
      sc.pp.log1p(self)
    sc.pp.highly_variable_genes(self,
                                n_bins=int(n_bins),
                                flavor=flavor,
                                subset=False,
                                inplace=True)
    del self.var['means']
    if not self.is_log:
      X = self.X
      for s, e in batching(_BATCH_SIZE, n=self.n_obs):
        x = X[s:e]
        if sp.sparse.issparse(x):
          np.expm1(x.data, out=x.data)
        else:
          np.expm1(x, out=x)
    return self

  # ******************** logging and io ******************** #
  def save_to_mmaparray(self, path, dtype=None):
    """ This only save the data without row names and column names """
    self._record('save_to_mmaparray', locals())
    with MmapArrayWriter(path=path,
                         shape=self.shape,
                         dtype=self.dtype if dtype is None else dtype,
                         remove_exist=True) as f:
      for s, e in batching(batch_size=_BATCH_SIZE, n=self.n_obs):
        x = self.X[s:e]
        if dtype is not None:
          x = x.astype(dtype)
        f.write(x)

  def _get_str(self):
    text = super(SingleCellOMIC, self).__repr__()
    text = text.replace('AnnData object', self.name)
    pad = "\n     "

    for omic in self.omics:
      X = self.numpy(omic)
      all_nonzeros = []
      for s, e in batching(n=self.n_obs, batch_size=_BATCH_SIZE):
        x = X[s:e]
        ids = np.nonzero(x)
        all_nonzeros.append(x[ids[0], ids[1]])
      all_nonzeros = np.concatenate(all_nonzeros)

      text += pad[:-1] + "OMIC: '%s' - dtype: '%s'" % (
          omic.name, "binary" if self.is_binary(omic) else "continuous")
      text += pad + 'Sparsity  : %.2f' % self.sparsity(omic)
      text += pad + 'Nonzeros  : %s' % describe(
          all_nonzeros, shorten=True, float_precision=2)
      text += pad + 'Cell      : %s' % describe(
          self.counts_per_cell(omic), shorten=True, float_precision=2)
      text += pad + 'Gene      : %s' % describe(
          self.counts_per_gene(omic), shorten=True, float_precision=2)
      text += pad + 'LogCount  : %s' % describe(
          self.log_counts(omic), shorten=True, float_precision=2)
      text += pad + 'LocalMean : %s' % describe(
          self.local_mean(omic), shorten=True, float_precision=2)
      text += pad + 'LocalVar  : %s' % describe(
          self.local_var(omic), shorten=True, float_precision=2)
    return text

  def __repr__(self):
    return self._get_str()

  def __str__(self):
    return self._get_str()
