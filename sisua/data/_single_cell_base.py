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
  om = []
  all_omics = {o.name: o for o in OMIC}
  for k in sco.obsm.keys():
    if isinstance(k, OMIC):
      om.append(k)
    elif k in all_omics:
      om.append(all_omics[k])
  # merge
  o = om[0]
  for i in om:
    o |= i
  return o


class _OMICbase(sc.AnnData):

  def __init__(self,
               X,
               cell_id=None,
               gene_id=None,
               dtype=None,
               omic=OMIC.transcriptomic,
               name=None,
               **kwargs):
    omic = OMIC.parse(omic)
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
      if hasattr(X, '_current_omic'):
        omic = X._current_omic
    # init as completely new dataset
    else:
      self._omics = omic
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
    self._current_omic = omic
    # store given omic
    if omic.name + '_var' not in self.uns:
      self.uns[omic.name + '_var'] = self.var
    if not kwargs.get('asview', False):
      self.obsm[omic.name] = self._X
    # The class is created for first time
    if not isinstance(X, sc.AnnData):
      self.obs['indices'] = np.arange(self.X.shape[0], dtype='int64')
      self._calculate_statistics(omic)

  def set_verbose(self, verbose):
    r""" If True, print out all method call and its arguments """
    self._verbose = bool(verbose)
    return self

  @property
  def verbose(self):
    return self._verbose

  @contextmanager
  def _swap_omic(self, omic):
    r""" Temporary change the main OMIC type to other than the default
    transcriptomic """
    omic = OMIC.parse(omic)
    current_omic = self._current_omic
    # do nothing if transcriptomic (the default)
    if omic == current_omic:
      yield self
    # swap then reset back to transcriptomic
    else:
      x = self.numpy(omic)
      var = self.get_var(omic)
      self._X = x
      self._var = var
      self._n_vars = self._X.shape[1]
      yield self
      self._X = self.numpy(current_omic)
      self._var = self.get_var(current_omic)
      self._n_vars = self._X.shape[1]

  @property
  def current_omic(self) -> OMIC:
    return self._current_omic

  def _record(self, name: str, local: dict):
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
    assert isinstance(sco, _OMICbase), \
      "sco must be instance of SingleCellOMIC"
    assert sco.shape[0] == self.shape[0], \
      "Number of cell mismatch %d and %d" % (self.shape[0], sco.shape[0])
    if 'cellid' in sco.obs and 'cellid' in self.obs:
      assert np.all(sco.obs['cellid'] == self.obs['cellid'])
    else:  # just check matching first column
      assert np.all(sco.obs.iloc[:, 0] == self.obs.iloc[:, 0])
    return self

  def _calculate_statistics(self, omic=None):
    if omic is None:
      omic = self.current_omic
    else:
      omic = OMIC.parse(omic)
    X = self.numpy(omic)
    # start processing
    if sp.sparse.issparse(X):
      total_counts = np.sum(X, axis=1)
      if total_counts.ndim < 2:
        total_counts = np.expand_dims(total_counts, axis=-1)
    else:
      total_counts = np.sum(X, axis=1, keepdims=True)
    log_counts, local_mean, local_var = get_library_size(X,
                                                         return_log_count=True)
    self.obsm[omic.name + '_stats'] = np.hstack(
        [total_counts, log_counts, local_mean, local_var])

  def __getitem__(self, index):
    r"""Returns a sliced view of the object."""
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

  def get_var(self, omic) -> pd.DataFrame:
    omic = OMIC.parse(omic)
    for om in list(omic):
      name = om.name + '_var'
      if name in self.uns:
        return self.uns[om.name + '_var']
    raise ValueError("OMIC not found, give: '%s', support: '%s'" %
                     (omic, self.omics))

  def get_var_names(self, omic):
    return self.get_var(omic).index.values

  def get_shape(self, omic):
    return self.numpy(omic=omic).shape[1]

  def get_omic(self, omic):
    r""" Return observation ndarray in `obsm` or `obs` """
    return self.numpy(omic=omic)

  def numpy(self, omic=None):
    r""" Return observation ndarray in `obsm` or `obs` """
    if omic is None:
      omic = self._current_omic
    arr = None
    # obs
    if isinstance(omic, string_types) and \
      not isinstance(omic, OMIC) and \
        omic in self.obs:
      arr = self.obs[omic].values
    # obsm
    omic = OMIC.parse(omic)
    for om in list(omic):
      if om in self.omics:
        arr = self.obsm[om.name]
        break
    # not found
    if arr is None:
      raise ValueError("OMIC not found, give: '%s', support: '%s'" %
                       (omic, self.omics))
    return arr

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

  def stats(self, omic=None):
    r""" Return a matrix of shape `[n_obs, 4]`.

    The columns are: 'total_counts', 'log_counts', 'local_mean', 'local_var'
    """
    if omic is None:
      omic = self._current_omic
    return self.obsm[omic.name + '_stats']

  def get_library_size(self, omic=None):
    r""" Return the mean and variance for library size modeling in log-space """
    if omic is None:
      omic = self._current_omic
    return self.library_size(omic=omic)

  def library_size(self, omic=None):
    r""" Return the mean and variance for library size modeling in log-space """
    if omic is None:
      omic = self._current_omic
    return self.local_mean(omic), self.local_var(omic)

  def total_counts(self, omic=None):
    return self.stats(omic)[:, 0:1]

  def log_counts(self, omic=None):
    return self.stats(omic)[:, 1:2]

  def local_mean(self, omic=None):
    return self.stats(omic)[:, 2:3]

  def local_var(self, omic=None):
    return self.stats(omic)[:, 3:4]

  # ====== statistics ====== #
  def sparsity(self, omic=None):
    return sparsity_percentage(self.numpy(omic))

  def counts_per_cell(self, omic=None):
    r""" Return total number of counts per cell. This method
    is scalable. """
    counts = 0
    X = self.numpy(omic)
    for s, e in batching(batch_size=BATCH_SIZE, n=X.shape[1]):
      counts += np.sum(X[:, s:e], axis=1)
    return counts

  def counts_per_gene(self, omic=None):
    r""" Return total number of counts per gene. This method
    is scalable. """
    counts = 0
    X = self.numpy(omic)
    for s, e in batching(batch_size=BATCH_SIZE, n=X.shape[0]):
      counts += np.sum(X[s:e], axis=0)
    return counts

  # ******************** logging and io ******************** #
  def create_dataset(self,
                     omics: OMIC = None,
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
    if omics is None:
      omics = self.current_omic
    framework = str(framework).lower().strip()
    assert framework in ('tf', 'pt', 'tensorflow', 'pytorch'), \
      f"Only support tensorflow or pytorch framework, given: {framework}"
    if isinstance(omics, OMIC):
      omics = list(omics)
    omics = [OMIC.parse(o) for o in tf.nest.flatten(omics)]
    inputs = [self.get_omic(o) for o in omics]
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
        mask = False
      else:
        mask = gen.uniform(shape=(1,)) < labels_percent
      inputs = data[:len(omics)]
      library = data[len(omics):]
      return dict(inputs=inputs[0] if len(inputs) == 1 else inputs,
                  library=library[0] if len(library) == 1 else library,
                  mask=mask)

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

  def _get_str(self):
    text = super().__repr__()
    text = text.replace('AnnData object', self.name)
    pad = "\n     "
    for omic in self.omics:
      X = self.numpy(omic)
      text += pad[:-1] + \
        f" {'*' if omic == self.current_omic else ''}OMIC:{omic.name} " + \
        f"shape:{X.shape} dtype:{X.dtype} sparsity:{self.sparsity(omic):.2f}"
    text += pad[:-1] + "History: %d methods" % len(self.history)
    for idx, (method, args) in enumerate(self.history[::-1]):
      text += pad + '%d) %s : %s' % (idx, method, ', '.join(
          ['%s:%s' % (k, v) for k, v in args.items()]))
    return text

  def describe(self) -> str:
    text = f"SingleCellOMICs: {self.name}"
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
    return text

  def __repr__(self):
    return self._get_str()

  def __str__(self):
    return self._get_str()
