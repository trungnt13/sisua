from __future__ import absolute_import, division, print_function

import inspect
import itertools
import os
import warnings
from contextlib import contextmanager
from numbers import Number
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
import scipy as sp
import tensorflow as tf
from anndata._core.aligned_mapping import AxisArrays
from scipy.sparse import issparse
from scipy.stats import pearsonr, spearmanr
from six import string_types

from bigarray import MmapArrayWriter
from odin import visual as vs
from odin.search import diagonal_beam_search, diagonal_bruteforce_search
from odin.stats import describe, sparsity_percentage, train_valid_test_split
from odin.utils import (IndexedList, as_tuple, batching, cache_memory,
                        catch_warnings_ignore, ctext, is_primitive)
from odin.utils.crypto import md5_checksum
from sisua.data.const import MARKER_GENES, OMIC
from sisua.data.utils import (apply_artificial_corruption, get_library_size,
                              is_binary_dtype, is_categorical_dtype,
                              standardize_protein_name)
from sisua.label_threshold import ProbabilisticEmbedding

# Heuristic constants
BATCH_SIZE = 4096
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


class _OMICbase(sc.AnnData):

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
      name = X._name
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
    super().__init__(X, **kwargs)
    self._name = str(name)
    self._verbose = False
    # store transcriptomic
    if OMIC.transcriptomic.name + '_var' not in self.uns:
      self.uns[OMIC.transcriptomic.name + '_var'] = self.var
    if not kwargs.get('asview', False):
      self.obsm[OMIC.transcriptomic.name] = self._X
    # The class is created for first time
    if not isinstance(X, sc.AnnData):
      self.obs['indices'] = np.arange(self.X.shape[0], dtype='int64')
      self._calculate_statistics(OMIC.transcriptomic)

  def set_verbose(self, verbose):
    r""" If True, print out all method call and its arguments """
    self._verbose = bool(verbose)
    return self

  @property
  def verbose(self):
    return self._verbose

  @contextmanager
  def _swap_omic(self, omic):
    r""" Temporary change the main OMIC type to other than transcriptomic """
    omic = OMIC.parse(omic)
    # do nothing if transcriptomic (the default)
    if omic == OMIC.transcriptomic:
      yield self
    # swap then reset back to transcriptomic
    else:
      x = self.numpy(omic)
      var = self.omic_var(omic)
      self._X = x
      self._var = var
      self._n_vars = self._X.shape[1]
      yield self
      self._X = self.numpy(OMIC.transcriptomic)
      self._var = self.omic_var(OMIC.transcriptomic)
      self._n_vars = self._X.shape[1]

  @property
  def _current_omic_name(self):
    x = self.X
    name = OMIC.transcriptomic.name
    for omic in self.omics:
      x1 = self.numpy(omic)
      if x.shape == x1.shape and np.all(x[:BATCH_SIZE] == x1[:BATCH_SIZE]):
        name = omic.name
        break
    return name

  def _record_call(self, name: str, local: dict):
    method = getattr(self, name)
    specs = inspect.getfullargspec(method)
    assert inspect.ismethod(method)
    local = {
        k: v if is_primitive(v, inc_ndarray=False) else str(type(v)) \
          for k, v in local.items() \
            if not isinstance(v, _OMICbase) and \
              (k in specs.args or specs.varkw is not None)
    }
    self._history[name] = local
    if self.verbose:  # print out every method call and its arguments
      print("Method:", name)
      for k, v in local.items():
        print(" ", k, ':', v)

  def add_omic(self, omic: OMIC, X: np.ndarray, var_names=None):
    self._record_call('add_omic', locals())
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
    omic = OMIC.parse(omic)
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
    self._record_call('copy', locals())
    anndata = super().copy(filename)
    anndata._name = self.name
    sco = self.__class__(anndata, asview=False)
    return sco

  def __getitem__(self, index):
    """Returns a sliced view of the object."""
    oidx, vidx = self._normalize_indices(index)
    om = self.__class__(self, oidx=oidx, vidx=vidx, asview=True)
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
    self._record_call('apply_indices', locals())
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

  def split(self,
            train_percent=0.8,
            copy=True,
            seed=1) -> Tuple['_OMICbase', '_OMICbase']:
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
    self._record_call('split', locals())
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
    for om in list(omic):
      name = om.name + '_var'
      if name in self.uns:
        return self.uns[om.name + '_var']
    raise ValueError("OMIC not found, give: '%s', support: '%s'" %
                     (omic, self.omics))

  def omic_varnames(self, omic):
    return self.omic_var(omic).index.values

  def get_omic_data(self, omic):
    r""" Return observation ndarray in `obsm` or `obs` """
    return self.numpy(omic=omic)

  def numpy(self, omic=OMIC.transcriptomic):
    r""" Return observation ndarray in `obsm` or `obs` """
    # obs
    if isinstance(omic, string_types) and \
      not isinstance(omic, OMIC) and \
        omic in self.obs:
      return self.obs[omic].values
    # obsm
    omic = OMIC.parse(omic)
    for om in list(omic):
      if om in self.omics:
        return self.obsm[om.name]
    raise ValueError("OMIC not found, give: '%s', support: '%s'" %
                     (omic, self.omics))

  def labels(self, omic=OMIC.proteomic):
    omic = OMIC.parse(omic)
    for om in list(omic):
      name = self.labels_name(om)
      if name in self.obs:
        return self.obs[name]
    raise ValueError("OMIC not found, give: '%s', support: '%s'" %
                     (omic, self.omics))

  def labels_name(self, omic=OMIC.proteomic):
    omic = OMIC.parse(omic)
    return omic.name + '_labels'

  @property
  def omics(self):
    r"""Return all OMIC types stored in this single-cell dataset"""
    return self._omics

  @property
  def n_omics(self):
    r""" Return number of OMIC types stored in this dataset """
    return len(list(self._omics))

  @property
  def name(self):
    return self._name

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

  def stats(self, omic=OMIC.transcriptomic):
    r""" Return a matrix of shape `[n_obs, 4]`.

    The columns are: 'total_counts', 'log_counts', 'local_mean', 'local_var'
    """
    return self.obsm[omic.name + '_stats']

  def get_library_size(self, omic=OMIC.transcriptomic):
    r""" Return the mean and variance for library size modeling in log-space """
    return self.library_size(omic=omic)

  def library_size(self, omic=OMIC.transcriptomic):
    r""" Return the mean and variance for library size modeling in log-space """
    return self.local_mean(omic), self.local_var(omic)

  def total_counts(self, omic=OMIC.transcriptomic):
    return self.stats(omic)[:, 0:1]

  def log_counts(self, omic=OMIC.transcriptomic):
    return self.stats(omic)[:, 1:2]

  def local_mean(self, omic=OMIC.transcriptomic):
    return self.stats(omic)[:, 2:3]

  def local_var(self, omic=OMIC.transcriptomic):
    return self.stats(omic)[:, 3:4]

  def probability(self, omic=OMIC.proteomic):
    r""" Return the probability embedding of an OMIC """
    return self.probabilistic_embedding(omic=omic)[1]

  def binary(self, omic=OMIC.proteomic):
    r""" Return the binary embedding of an OMIC """
    return self.probabilistic_embedding(omic=omic)[2]

  # ====== statistics ====== #
  def sparsity(self, omic=OMIC.transcriptomic):
    return sparsity_percentage(self.numpy(omic))

  def counts_per_cell(self, omic=OMIC.transcriptomic):
    r""" Return total number of counts per cell. This method
    is scalable. """
    counts = 0
    X = self.numpy(omic)
    for s, e in batching(batch_size=BATCH_SIZE, n=X.shape[1]):
      counts += np.sum(X[:, s:e], axis=1)
    return counts

  def counts_per_gene(self, omic=OMIC.transcriptomic):
    r""" Return total number of counts per gene. This method
    is scalable. """
    counts = 0
    X = self.numpy(omic)
    for s, e in batching(batch_size=BATCH_SIZE, n=X.shape[0]):
      counts += np.sum(X[s:e], axis=0)
    return counts

  # ******************** logging and io ******************** #
  def create_dataset(self,
                     omics: OMIC = OMIC.transcriptomic,
                     labels_percent=0,
                     batch_size=64,
                     drop_remainder=False,
                     shuffle=1000,
                     cache='',
                     framework='tensorflow',
                     seed=1) -> tf.data.Dataset:
    r""" Create dataset for training using one or multiple OMIC data

    Arguments:
      omics : `OMIC` or list of `OMIC`. Specify all the OMIC types will be
        included in the dataset
      library_size : a Boolean or list of Boolean. If true, log mean and log
        var will be include, the length of the list is coordinated to the `omics`
      labels_percent : a Scalar [0., 1.]. If > 0, create a mask with given
        percent set to True.
    """
    framework = str(framework).lower().strip()
    assert framework in ('tf', 'pt', 'tensorflow', 'pytorch'), \
      f"Only support tensorflow or pytorch framework, given: {framework}"
    omics = [OMIC.parse(o) for o in tf.nest.flatten(omics)]
    inputs = [self.get_omic_data(o) for o in omics]
    # library size
    library = []
    for o in omics:
      library.append(np.concatenate(self.get_library_size(o), axis=-1))
    # create the dataset
    ds = [tf.data.Dataset.from_tensor_slices(i) for i in inputs] + \
      [tf.data.Dataset.from_tensor_slices(i) for i in library]
    if len(ds) > 0:
      ds = tf.data.Dataset.zip(tuple(ds))
    # for labels_percent
    labels_percent = np.clip(labels_percent, 0., 1.)
    if len(omics) == 1:
      labels_percent = 0.
    gen = tf.random.experimental.Generator.from_seed(seed=seed)

    def masking(*data):
      if labels_percent == 0.:
        mask = (False,)
      else:
        mask = gen.uniform(shape=(1,)) < labels_percent
      inputs = data[:len(omics)]
      library = data[len(omics):]
      return dict(inputs=inputs, library=library, mask=mask)

    ds = ds.map(masking, tf.data.experimental.AUTOTUNE)
    # post processing
    if cache is not None:
      ds = ds.cache(str(cache))
    # shuffle must be called after cache
    if shuffle is not None:
      ds = ds.shuffle(int(shuffle))
    ds = ds.batch(batch_size, drop_remainder)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    return ds

  def save_to_mmaparray(self, path, dtype=None):
    """ This only save the data without row names and column names """
    self._record_call('save_to_mmaparray', locals())
    with MmapArrayWriter(path=path,
                         shape=self.shape,
                         dtype=self.dtype if dtype is None else dtype,
                         remove_exist=True) as f:
      for s, e in batching(batch_size=BATCH_SIZE, n=self.n_obs):
        x = self.X[s:e]
        if dtype is not None:
          x = x.astype(dtype)
        f.write(x)

  def _get_str(self):
    text = super().__repr__()
    text = text.replace('AnnData object', self.name)
    pad = "\n     "

    for omic in self.omics:
      X = self.numpy(omic)
      all_nonzeros = []
      for s, e in batching(n=self.n_obs, batch_size=BATCH_SIZE):
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

    text += pad[:-1] + "History: %d methods" % len(self.history)
    for method, args in self.history:
      text += pad + '%s : %s' % (method, ', '.join(
          ['%s:%s' % (k, v) for k, v in args.items()]))
    return text

  def __repr__(self):
    return self._get_str()

  def __str__(self):
    return self._get_str()
