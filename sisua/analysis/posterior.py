from __future__ import absolute_import, division, print_function

import inspect
import os
import pickle
import shutil
import time
import warnings
from collections import OrderedDict, defaultdict
from functools import partial
from itertools import product, zip_longest
from typing import Optional, Union

import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from matplotlib import pyplot as plt
from scipy.stats import pearsonr, spearmanr
from six import string_types
from sklearn.feature_selection import (mutual_info_classif,
                                       mutual_info_regression)
from sklearn.linear_model import Lasso
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from odin import visual as vs
from odin.backend import log_norm
from odin.bay import distributions as tfd
from odin.bay import vi
from odin.bay.vi import Criticizer, discretizing
from odin.fuel import Dataset
from odin.stats import is_binary, is_discrete
from odin.utils import (MPI, as_tuple, cache_memory, catch_warnings_ignore,
                        clean_folder, flatten_list, md5_checksum)
from odin.visual import (Visualizer, plot_aspect, plot_confusion_matrix,
                         plot_figure, plot_frame, plot_save, plot_scatter,
                         to_axis2D)
from sisua.analysis.imputation_benchmarks import (correlation_scores,
                                                  imputation_mean_score,
                                                  imputation_score,
                                                  imputation_std_score)
from sisua.analysis.latent_benchmarks import (clustering_scores,
                                              plot_distance_heatmap,
                                              plot_latents_binary,
                                              plot_latents_protein_pairs,
                                              streamline_classifier)
from sisua.data import (MARKER_ADT_GENE, MARKER_ADTS, MARKER_ATAC, MARKER_GENES,
                        OMIC, PROTEIN_PAIR_NEGATIVE, PROTEIN_PAIR_POSITIVE,
                        SingleCellOMIC, apply_artificial_corruption,
                        get_dataset)
from sisua.data.path import EXP_DIR


# 'X' for input gene expression,
# 'T' for corrupted gene expression,
# 'V' for imputed gene expression,
# and 'W' for reconstructed gene expression
def _iter_2list(l1, l2):
  for i in l1:
    for j in l2:
      yield i, j


# ===========================================================================
# The Posterior
# ===========================================================================
class Posterior(Visualizer):
  r""" Posterior class for analysis

  Arguments:
    scm : {`sisua.models.SingleCellModel`, string}, a trained single-cell model
    gene : a `numpy.ndarray` with shape `[n_samples, n_genes]`
    protein : a `numpy.ndarray` with shape `[n_samples, n_protein]`
    batch_size : an Integer, batch size for the prediction tasks.
    sample_shape : an Integer, number of MCMC samples for evaluation
    verbose : a Boolean, turn on verbose

  Example:
  ```
  pos1 = Posterior(mae, x_test, y_test)
  print(pos1.scores_classifier(x_train, y_train))
  print(pos1.scores_llk())
  print(pos1.scores_imputation())
  print(pos1.scores_spearman())
  print(pos1.scores_pearson())
  print(pos1.scores_clustering())
  pos1.save_scores('/tmp/tmp1.txt')

  pos1.plot_protein_scatter().plot_protein_predicted_series()
  pos1.plot_classifier_F1(x_train, y_train)
  pos1.plot_learning_curves('loss_x')
  pos1.plot_learning_curves()
  pos1.get_correlation_marker_pairs('X')
  pos1.get_correlation_marker_pairs('V')
  pos1.get_correlation_marker_pairs('W')
  pos1.get_correlation_marker_pairs('T')
  pos1.plot_correlation_top_pairs().plot_correlation_bottom_pairs()
  pos1.plot_latents_binary_scatter().plot_latents_distance_heatmap(
  ).plot_latents_protein_pairs()
  pos1.save_figures('/tmp/tmp1.pdf')
  ```

  """

  TYPES = ('imputed', 'original', 'corrupted', 'reconstructed')

  def __init__(self,
               scm,
               sco: SingleCellOMIC,
               dropout_rate=0.2,
               retain_rate=0.2,
               corrupt_distribution='binomial',
               batch_size=8,
               sample_shape=10,
               random_state=1,
               reduce_latents=partial(tf.concat, axis=1),
               name=None,
               verbose=True):
    super(Posterior, self).__init__()
    if name is None:
      name = scm.name + '_' + sco.name
    self._name = str(name)
    self.verbose = int(verbose)
    self.sample_shape = int(sample_shape)
    self.batch_size = int(batch_size)
    self.rand = random_state \
      if isinstance(random_state, np.random.RandomState) else \
        np.random.RandomState(seed=random_state)
    # reduce latents function
    assert callable(reduce_latents), \
      ("reduce_latents must be callable to merge multiple latents "
       "into single one for evaluation")
    self.reduce_latents = reduce_latents
    # description of the variables
    self._n_latents = None
    self._n_outputs = None
    # single cell model
    from sisua.models.single_cell_model import SingleCellModel
    assert isinstance(scm, SingleCellModel), \
      f"scm must be instance of SingleCellModel, but given {type(scm)}"
    self._scm = scm
    if not self.scm.is_fitted:
      warnings.warn("The SingleCellModel is not fitted.")
    # multi-omics expression data
    assert isinstance(sco, SingleCellOMIC), \
      "sco must be instance of SingleCellOMIC, but given %s" % str(type(sco))
    assert sco.n_omics >= 2, \
      "SingleCellOMIC need at least 2 different OMIC types for evaluation."
    self.sco_original = sco
    self.sco_corrupted = sco.corrupt(dropout_rate=dropout_rate,
                                     retain_rate=retain_rate,
                                     distribution=corrupt_distribution,
                                     inplace=False,
                                     seed=self.rand.randint(1e8))
    self.input_omics = list(set(i.name for i in self.sco_corrupted.omics))
    self.output_omics = list()
    self.omics_data = {
        (om.name, 'original'): sco.get_omic(om) for om in sco.omics
    }
    # mapping from tuple of (omic.name, type) -> distribution/tensor
    self._dataset = None
    self._criticizers = dict()
    self._initialize()
    # create the default sco and criticizer for analysis
    for factor_omic in (OMIC.proteomic, OMIC.celltype, OMIC.disease,
                        OMIC.iproteomic, OMIC.icelltype, OMIC.idisease,
                        OMIC.progenitor, OMIC.iprogenitor):
      if factor_omic in self.dataset.omics:
        self.get_criticizer(factor_omic=factor_omic)

  def _initialize(self):
    scm = self.scm
    sco = self.sco_corrupted
    outputs, latents = scm.predict(
        sco.create_dataset(self.scm.output_layers[0].name,
                           batch_size=self.batch_size,
                           shuffle=0,
                           drop_remainder=False),
        sample_shape=self.sample_shape,
        verbose=self.verbose,
    )
    # infer output OMICs
    dim2omic = defaultdict(list)
    for om in self.input_omics:
      dim2omic[self.sco_original.get_dim(om)].append(om)
    for o in tf.nest.flatten(outputs):
      assert isinstance(o, tfd.Distribution), \
        f"SingleCellModel must output Distribution but return {o}"
      name = o.name
      try:
        om = OMIC.parse(name)
      except Exception:
        om = None
      if om is None:
        oms = dim2omic[o.event_shape[0]]
        if len(oms) > 1:
          raise RuntimeError(f"Cannot infer OMIC type for output {o}")
        om = oms[0]
      self.output_omics.append(om.name)
    # variables' description
    self._n_latents = len(tf.nest.flatten(latents))
    self._n_outputs = len(tf.nest.flatten(outputs))
    ## default inputs
    for om in self.input_omics:
      self.omics_data[(om, 'corrupted')] = sco.get_omic(om)
    # latent is the same for all
    self.omics_data[(OMIC.latent.name, 'corrupted')] = tf.nest.flatten(latents)
    # infer if the distribution is imputed
    for l, o in zip(scm.output_layers, tf.nest.flatten(outputs)):
      self.omics_data[(l.name, 'reconstructed')] = o
      is_independent = 0
      if isinstance(o, tfd.Independent):
        is_independent = o.reinterpreted_batch_ndims
        o = o.distribution
      if isinstance(o, tfd.ZeroInflated):
        o = o.count_distribution
      if is_independent > 0:
        o = tfd.Independent(o, reinterpreted_batch_ndims=is_independent)
      self.omics_data[(l.name, 'imputed')] = o
    ### create the SingleCellOMIC dataset for analysis
    sco = self.sco_original.copy()
    for om in self.input_omics:
      if (om, 'imputed') in self.omics_data:
        data_type = 'imputed'
      elif (om, 'reconstructed') in self.omics_data:
        data_type = 'reconstructed'
      else:
        continue
      data = self.omics_data[(om, data_type)]
      om_new = OMIC.parse(f'i{om}')
      # prepare the new data
      if isinstance(data, tfd.Distribution):
        data = data.mean().numpy()
        if data.ndim == 3:
          data = np.mean(data, axis=0)
      # find the variable's names
      if om in self.scm.metadata:
        var_names = self.scm.metadata[om]
      else:
        var_names = np.array([f'{om}{i}' for i in range(data.shape[1])])
      sco.add_omic(omic=om_new, X=data, var_names=var_names)
    # add the latents
    Zs = self.omics_data[('latent', 'corrupted')]
    if len(Zs) > 1:
      means = [z.mean() for z in Zs]
      Zs = self.reduce_latents(means)
    else:
      Zs = Zs[0].mean()
    with catch_warnings_ignore(UserWarning, RuntimeWarning):
      sco.add_omic(omic=OMIC.latent,
                   X=Zs.numpy(),
                   var_names=np.array([f'Z{i}' for i in range(Zs.shape[1])]))
    # store the extracted SingleCellOMIC dataset
    self._dataset = sco

  # ******************** properties and getter ******************** #
  def get_marker_pairs(self,
                       omic1='proteomic',
                       omic2=None,
                       var_names1=MARKER_ADTS,
                       var_names2=None,
                       threshold=None,
                       n=10,
                       most_correlated=False,
                       remove_duplicated=True):
    r""" Return the most differentiated (or correlated) pairs within a
    single OMIC (in case `omic2=None`) or between 2 different OMICs.
    """
    return self.dataset.get_marker_pairs(omic1, omic2, var_names1, var_names2,
                                         threshold, n, most_correlated,
                                         remove_duplicated)

  def get_data(self,
               omic,
               data_type='auto') -> Union[tfd.Distribution, np.ndarray]:
    r""" Return extract data, could be an array or a distribution.

    Arguments:
      omic : `OMIC`
      data_type : {'imputed', 'original', 'corrupted', 'reconstructed', 'auto'}
        if a list of string is provided, return the first instance found.
        'auto' - select the first occurrence of the omic
    """
    omic = OMIC.parse(omic).name
    if data_type == 'auto':
      for k, v in self.omics_data.items():
        if k == omic:
          if tf.is_tensor(v):
            v = v.numpy()
          return v
    else:
      for dtype in [str(i).lower().strip() for i in tf.nest.flatten(data_type)]:
        key = (omic, dtype)
        if key in self.omics_data:
          x = self.omics_data[key]
          if tf.is_tensor(x):
            x = x.numpy()
          return x
    # error
    raise ValueError(f"No data found for OMIC: {omic}-{data_type}, "
                     f"available data are: {list(self.omics_data.keys())}")

  def get_criticizer(self,
                     factor_omic='proteomic',
                     latent_indices=None,
                     n_bins=5,
                     strategy='quantile') -> Criticizer:
    r""" Create a probabilistic criticizer for evaluating the latent codes of
    variational models.

    Arguments:
      factor_omic : instance of OMIC.
        which OMIC type be used as factors (or labels).
      n_bins : int (default=8)
        The number of bins to produce discretized factors.
      strategy : {'uniform', 'quantile', 'kmeans', 'gmm'}, (default='quantile')
        Strategy used to define the widths of the bins.
        uniform - All bins in each feature have identical widths.
        quantile - All bins in each feature have the same number of points.
        kmeans - Values in each bin have the same nearest center of a 1D
          k-means cluster.
    """
    sco = self.dataset
    assert factor_omic in sco.omics, \
      f"factor_omic='{factor_omic}' not found, available are: {sco.omics}"
    factor_omic = OMIC.parse(factor_omic)
    if latent_indices is None:
      key = f"{factor_omic.name}"
    else:
      name = '_'.join(f'{i:d}' for i in latent_indices)
      key = f"{factor_omic.name}{name}"
    # create the Criticizer
    if key not in self._criticizers:
      # check the factors is valid
      factors = sco.numpy(factor_omic)
      factor_names = sco.get_var_names(factor_omic)
      kw = dict(n_bins=int(n_bins), strategy=None)
      # binary classes
      if np.all(np.sum(factors, axis=1) == 1):
        factors = np.argmax(factors, axis=1)[:, np.newaxis]
        factor_names = np.asarray([factor_omic.name])
      # continuous or discrete cases
      elif factor_omic in (OMIC.proteomic, OMIC.iproteomic, OMIC.pmhc,
                           OMIC.ipmhc):
        kw['strategy'] = strategy
      # categorical factors
      elif factor_omic in (OMIC.progenitor, OMIC.iprogenitor, OMIC.celltype,
                           OMIC.icelltype):
        pass
      # unknown factor
      else:
        warnings.warn(f"No support for discretization of OMIC: {factor_omic}",
                      RuntimeWarning)
        return
      # only valid factors with > 1 classes
      ids = [len(np.unique(i)) > 1 for i in factors.T]
      if not any(ids):  # no valid factor found
        warnings.warn(f"Not a valid factor: {factor_omic.name}", RuntimeWarning)
        return
      factors = factors[:, ids]
      factor_names = factor_names[ids]
      # create the criticizer
      crt = Criticizer(vae=self.scm,
                       latent_indices=latent_indices,
                       random_state=self.rand.randint(1e8))
      crt.factor_omic: OMIC = factor_omic
      with catch_warnings_ignore(UserWarning):
        latents = self.omics_data[('latent', 'corrupted')]
        crt.sample_batch(latents=latents,
                         factors=factors,
                         factor_names=factor_names,
                         **kw)
      self._criticizers[key] = crt
    return self._criticizers[key]

  @property
  def latents(self) -> tfd.Distribution:
    zs = self.get_data(OMIC.latent, data_type='corrupted')
    if len(zs) == 1:
      z = zs[0]
    else:
      z = tfd.CombinedDistribution(zs,
                                   validate_args=False,
                                   name=f"{self.name}_latents")
    return z

  @property
  def scm(self):
    from sisua.models.single_cell_model import SingleCellModel
    model: SingleCellModel = self._scm
    return model

  @property
  def dataset(self) -> SingleCellOMIC:
    return self._dataset

  @property
  def n_latents(self) -> int:
    return self._n_latents

  @property
  def n_outputs(self) -> int:
    return self._n_outputs

  @property
  def name(self) -> str:
    return self._name

  @property
  def is_semi_supervised(self):
    return self.scm.is_semi_supervised

  # ******************** helpers ******************** #
  def _train_data(self, x_train, y_train):
    """ return outputs, latents, y_train """
    if not isinstance(x_train, SingleCellOMIC):
      x_train = SingleCellOMIC(x_train)
    if not isinstance(y_train, SingleCellOMIC):
      y_train = SingleCellOMIC(y_train)
    # get binary values for y_train
    if not y_train.is_binary:
      if 'pbe' not in y_train.uns:
        y_train.probabilistic_embedding()
      y_train = y_train.obsm['X_bin']
    else:
      y_train = y_train.X
    # create corrupted data and get the outputs and latents
    x_train = x_train.corrupt(corruption_rate=self.scm.corruption_rate,
                              corruption_dist=self.scm.corruption_dist,
                              inplace=False)
    outputs, latents = self.scm.predict(x_train,
                                        sample_shape=self.sample_shape,
                                        batch_size=self.batch_size,
                                        enable_cache=False,
                                        verbose=self.verbose)
    # return, always a list, except the labels
    if not isinstance(outputs, (tuple, list)):
      outputs = [outputs]
    if not isinstance(latents, (tuple, list)):
      latents = [latents]
    return outputs, latents, y_train

  # ******************** Latent space analysis ******************** #
  def plot_scatter(self,
                   X='latent',
                   color_by='proteomic',
                   marker_by=None,
                   clustering='kmeans',
                   dimension_reduction='tsne',
                   max_scatter_points=-1,
                   **kwargs):
    r"""
    X : OMIC type for coordinates
    color_by : OMIC type for coloring
    marker_by : OMIC type for marker type
    clustering : {'kmeans', 'knn', 'tsne', 'pca', 'umap'}
    dimension_reduction : {'tsne', 'pca', 'umap'}
    """
    X = OMIC.parse(X)
    color_by = OMIC.parse(color_by)
    title = f"scatter_{X.name}_{color_by.name}_" + \
      f"{clustering.lower()}_{dimension_reduction.lower()}"
    fig = plt.figure(figsize=(8, 8))
    self.dataset.plot_scatter(X=X,
                              color_by=color_by,
                              marker_by=marker_by,
                              clustering=clustering,
                              dimension_reduction=dimension_reduction,
                              max_scatter_points=max_scatter_points,
                              ax=fig.gca(),
                              return_figure=True,
                              **kwargs)
    return self.add_figure(name=title, fig=fig)

  def plot_violins(self,
                   X='transcriptomic',
                   group_by='proteomic',
                   groups=None,
                   var_names=MARKER_GENES,
                   clustering='kmeans',
                   rank_vars=0,
                   **kwargs):
    r"""
    X : OMIC type for the violin
    group_by : OMIC type for grouping the violin plot
    var_names : name of variables in `X` will be shown
    clustering : {'kmeans', 'knn', 'tsne', 'pca', 'umap'}
    rank_vars : ranking variable in `X` for each group in `group_by`
    """
    X = OMIC.parse(X)
    group_by = OMIC.parse(group_by)
    title = f"violins_{X.name}_{group_by.name}{'_rank' if rank_vars > 0 else ''}"
    fig = self.dataset.plot_stacked_violins(X=X,
                                            group_by=group_by,
                                            groups=groups,
                                            var_names=var_names,
                                            clustering=clustering,
                                            rank_vars=rank_vars,
                                            return_figure=True,
                                            **kwargs)
    return self.add_figure(name=title, fig=fig)

  def plot_heatmap(self,
                   X='transcriptomic',
                   group_by='proteomic',
                   groups=None,
                   var_names=MARKER_GENES,
                   clustering='kmeans',
                   rank_vars=0,
                   **kwargs):
    r"""
    X : OMIC type for the violin
    group_by : OMIC type for grouping the violin plot
    var_names : name of variables in `X` will be shown
    clustering : {'kmeans', 'knn', 'tsne', 'pca', 'umap'}
    rank_vars : ranking variable in `X` for each group in `group_by`
    """
    X = OMIC.parse(X)
    group_by = OMIC.parse(group_by)
    title = f"heatmap_{X.name}_{group_by.name}{'_rank' if rank_vars > 0 else ''}"
    fig = self.dataset.plot_heatmap(X=X,
                                    group_by=group_by,
                                    groups=groups,
                                    var_names=var_names,
                                    clustering=clustering,
                                    rank_vars=rank_vars,
                                    return_figure=True,
                                    **kwargs)
    return self.add_figure(name=title, fig=fig)

  def plot_distance_heatmap(self,
                            X='transcriptomic',
                            group_by='proteomic',
                            clustering='kmeans',
                            **kwargs):
    r"""
    X : OMIC type for the violin
    group_by : OMIC type for grouping the violin plot
    clustering : {'kmeans', 'knn', 'tsne', 'pca', 'umap'}
    """
    X = OMIC.parse(X)
    group_by = OMIC.parse(group_by)
    title = f"distheatmap_{X.name}_{group_by.name}"
    fig = plt.figure(figsize=(8, 8))
    self.dataset.plot_distance_heatmap(X=X,
                                       group_by=group_by,
                                       clustering=clustering,
                                       ax=fig.gca(),
                                       return_figure=True,
                                       **kwargs)
    return self.add_figure(name=title, fig=fig)

  def plot_correlation_matrix(self,
                              omic1=OMIC.transcriptomic,
                              omic2=OMIC.proteomic,
                              var_names1=MARKER_ADT_GENE.values(),
                              var_names2=MARKER_ADT_GENE.keys(),
                              is_marker_pairs=True,
                              score_type='spearman'):
    r""" Heatmap correlation between omic1 (x-axis) and omic2 (y-axis)

    omic1 : OMIC type for x-axis (column)
    omic2 : OMIC type for y-axis (row)
    var_names1 : name of variables in `omic1` will be shown
    var_names2 : name of variables in `omic2` will be shown
    is_marker_pairs : coor-pairs in `var_names1` and `var_names2` are markers
    score_type : {'pearson', 'spearman', 'mutual_information'}
    """
    omic1 = OMIC.parse(omic1)
    omic2 = OMIC.parse(omic2)
    sco = self.dataset
    title = f"{score_type.lower()}_{omic1.name}_{omic2.name}"
    if score_type == 'pearson':
      fn = sco.plot_pearson_matrix
    elif score_type == 'spearman':
      fn = sco.plot_spearman_matrix
    elif score_type == 'mutual_information':
      fn = sco.plot_mutual_information
    else:
      raise NotImplementedError(
          f"No implementation for score_type={score_type}")
    fig = fn(omic1=omic1,
             omic2=omic2,
             var_names1=var_names1,
             var_names2=var_names2,
             is_marker_pairs=is_marker_pairs,
             return_figure=True)
    return self.add_figure(name=title, fig=fig)

  def plot_correlation_scatter(self,
                               omic1='transcriptomic',
                               omic2='proteomic',
                               var_names1=MARKER_ADT_GENE.values(),
                               var_names2=MARKER_ADT_GENE.keys(),
                               is_marker_pairs=True,
                               log1=True,
                               log2=True,
                               max_scatter_points=200,
                               top=3,
                               bottom=3):
    r""" Scatter correlation plot between 2 series (if `var_names1=None` plot
    the minimum and maximum correlation scores pairs)

    omic1 : OMIC type for x-axis (column)
    omic2 : OMIC type for y-axis (row)
    var_names1 : name of variables in `omic1` will be shown
    var_names2 : name of variables in `omic2` will be shown
    is_marker_pairs : coor-pairs in `var_names1` and `var_names2` are markers
    """
    omic1 = OMIC.parse(omic1)
    omic2 = OMIC.parse(omic2)
    title = f"corrscat_{omic1.name}_{omic2.name}"
    fig = self.dataset.plot_correlation_scatter(
        omic1=omic1,
        omic2=omic2,
        var_names1=var_names1,
        var_names2=var_names2,
        is_marker_pairs=is_marker_pairs,
        log1=log1,
        log2=log2,
        top=top,
        bottom=bottom,
        max_scatter_points=max_scatter_points,
        return_figure=True)
    return self.add_figure(name=title, fig=fig)

  def plot_series(self,
                  omic1=OMIC.transcriptomic,
                  omic2=OMIC.proteomic,
                  var_names1=MARKER_ADT_GENE.values(),
                  var_names2=MARKER_ADT_GENE.keys(),
                  log1=True,
                  log2=True):
    r"""
    Arguments:
      omic1 : OMIC type for y-axis
      omic2 : OMIC type for twinx-axis
      var_names1 : name of variables in `omic1` will be shown
      var_names2 : name of variables in `omic2` will be shown
      is_marker_pairs : coor-pairs in `var_names1` and `var_names2` are markers
    """
    omic1 = OMIC.parse(omic1)
    omic2 = OMIC.parse(omic2)
    title = f"series_{omic1.name}_{omic2.name}"
    fig = self.dataset.plot_series(omic1=omic1,
                                   omic2=omic2,
                                   var_names1=var_names1,
                                   var_names2=var_names2,
                                   log1=log1,
                                   log2=log2,
                                   return_figure=True)
    return self.add_figure(name=title, fig=fig)

  def plot_confusion_matrix(self, y_true='celltype', y_pred='icelltype'):
    r""" Confusion matrix for binary labels """
    y_true = OMIC.parse(y_true)
    y_pred = OMIC.parse(y_pred)
    name = f"true:{y_true.name}_pred:{y_pred.name}"
    x_true = self.dataset.get_omic(y_true)
    x_pred = self.dataset.get_omic(y_pred)
    if x_true.ndim > 1:
      x_true = np.argmax(x_true, axis=-1)
    if x_pred.ndim > 1:
      x_pred = np.argmax(x_pred, axis=-1)
    fig = plt.figure(figsize=(6, 6))
    plot_confusion_matrix(y_true=x_true,
                          y_pred=x_pred,
                          labels=self.dataset.get_var_names(y_true),
                          cmap="Blues",
                          ax=fig.gca(),
                          fontsize=12,
                          cbar=True,
                          title=name)
    return self.add_figure(name=f"cm_{name}", fig=fig)

  def plot_learning_curves(self, summary_steps=(100, 10), dpi=100):
    r""" Plotting the loss or metrics returned during training progress """
    fig = self.scm.plot_learning_curves(path=None,
                                        summary_steps=summary_steps,
                                        dpi=dpi,
                                        title=self.name)
    return self.add_figure('learning_curves', fig)

  def plot_disentanglement(self,
                           factor_omic='proteomic',
                           factor_names='auto',
                           n_bins_factors=10,
                           n_bins_latents=80,
                           corr_type='spearman',
                           show_all_latents=False,
                           latent_indices=None):
    r""" Ploting the histogram colored by the activation in each factor, i.e.
    disentanglement plotting

    Arguments:
      factor_omic : OMIC, which OMIC used as ground truth factor
      factor_names : {'auto', list of factors}, which subset of factors will be
        shown
      latent_indices : an Integer or list of Integers.
        Indicates which latent will be used for `Criticizer`
    """
    factor_omic = OMIC.parse(factor_omic)
    if isinstance(factor_names, string_types) and factor_names == 'auto':
      factor_names = factor_omic.markers
    # creating the criticizer
    crt = self.get_criticizer(factor_omic=factor_omic,
                              latent_indices=latent_indices)
    # filter relevant factor from the markers' list
    if factor_names is not None:
      var_names = crt.factor_names
      org = factor_names
      factor_names = list(
          filter(lambda x: x in var_names, tf.nest.flatten(factor_names)))
      assert len(factor_names) > 0, f"No matching factor names for {org}"
    # plotting
    fig = crt.plot_disentanglement(factor_names=factor_names,
                                   n_bins_factors=n_bins_factors,
                                   n_bins_codes=n_bins_latents,
                                   corr_type=corr_type,
                                   show_all_codes=show_all_latents,
                                   title=f"{factor_omic.name}",
                                   return_figure=True)
    name = factor_omic.name + str(latent_indices)
    return self.add_figure(f"disentanglement_{name}_{corr_type}", fig)

  def plot_disentanglement_scatter(self,
                                   factor_omic='proteomic',
                                   pairs=PROTEIN_PAIR_NEGATIVE,
                                   corr_matrix=None,
                                   n_pairs=10,
                                   latents_per_pair=5,
                                   magnify=2):
    r""" Select the most differentiated pairs of `factor_omic`, then,
    select the most correlated latent to each factor within the pair,
    use the latents' coordination for plotting the scatter points, and,
    use the `factor_omic` values for coloring the heatmap.

    Arguments:
      factor_omic : `OMIC`, the OMIC used as groundtruth factors
      pairs : list of `(factor_name1, factor_name2)` (optional)
        This determines which pairs will be plotted.
      corr_matrix : correlation matrix between `factor_omic` and the latents,
        dimension must be `[n_factor_omic, n_latents]`.
        This determines which latents dimension will be selected for plotting
        the pairs.
      magnify : a Scalar (default: 1)
        a constant for magnifying the color divergence of
        the `factor_omic`, the higher, small differences lead to stronger
        color divergence.

    Example:
    ```
    pairs = post.get_marker_pairs()
    corr = post.get_correlation_matrix('proteomic', 'latent')
    post.plot_disentanglement_scatter('proteomic',
                                      pairs=pairs,
                                      corr_matrix=corr,
                                      magnify=2)
    post.plot_disentanglement_scatter('iproteomic',
                                      pairs=pairs,
                                      corr_matrix=corr,
                                      magnify=2)
    ```
    """
    factor_omic = OMIC.parse(factor_omic)
    var_ids = self.dataset.get_var_indices(factor_omic)
    ### marker pairs
    if pairs is None:
      pairs = self.get_marker_pairs(omic1=factor_omic,
                                    omic2=None,
                                    most_correlated=False,
                                    remove_duplicated=True,
                                    n=int(n_pairs))
    else:
      pairs = [(name1, name2)
               for name1, name2 in pairs
               if name1 in var_ids and name2 in var_ids]
      assert len(pairs) > 0
    ### correlation matrix
    if corr_matrix is None:
      corr_matrix = self.get_correlation_matrix(factor_omic, 'latent',
                                                'average')
    shape = (self.dataset.get_dim(factor_omic),
             self.dataset.get_dim(OMIC.latent))
    assert corr_matrix.shape == shape, \
      (f"Correlation matrix must has shape {shape} but given matrix "
       f"with shape {corr_matrix.shape}")
    ### getting all latents for each pair
    latents_per_pair = int(latents_per_pair)
    omic2latent = {}
    for name1, name2 in pairs:
      latents = []
      seen = set()
      # sort in descending order
      for i1, i2 in _iter_2list(
          np.argsort(corr_matrix[var_ids[name1]])[::-1],
          np.argsort(corr_matrix[var_ids[name2]])[::-1]):
        if i1 != i2 and i1 not in seen and i2 not in seen:
          seen.add(i1)
          seen.add(i2)
          latents.append([i1, i2])
        if len(latents) == latents_per_pair:
          break
      omic2latent[(name1, name2)] = latents
    ### plotting
    ncol = 5
    X = self.dataset.get_omic(omic=factor_omic)
    Z = self.latents.mean().numpy()
    latent_names = self.dataset.get_var_names(OMIC.latent)
    norm = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))
    for (name1, name2), pairs in omic2latent.items():
      nrow = int(np.ceil(len(pairs) / ncol))
      fig = vs.plot_figure(nrow=nrow * 3.3, ncol=ncol * 4, dpi=80)
      # normalize the factor OMIC for color values
      x1 = X[:, var_ids[name1]]
      x2 = X[:, var_ids[name2]]
      x = norm(x1) - norm(x2)
      x = np.clip(x * magnify, -1., 1.)
      # get latents' coordination
      for idx, (i1, i2) in enumerate(pairs):
        z1 = Z[:, i1]
        z2 = Z[:, i2]
        ax = vs.plot_scatter(x=z1,
                             y=z2,
                             val=x,
                             ax=(nrow, ncol, idx + 1),
                             cbar=True,
                             cbar_ticks=[name1, 'others', name2],
                             cbar_labrotation=-60,
                             ticks_off=True,
                             fontsize=8,
                             max_n_points=2000,
                             size=16)
        # xticks
        v = max(np.abs(np.min(z1)), np.abs(np.max(z1)))
        ticks = np.linspace(-v, v, num=5)
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{i:.2g}" for i in ticks], fontsize=8)
        ax.set_xlabel(latent_names[i1], fontsize=10)
        # yticks
        v = max(np.abs(np.min(z2)), np.abs(np.max(z2)))
        ticks = np.linspace(-v, v, num=5)
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{i:.2g}" for i in ticks], fontsize=8)
        ax.set_ylabel(latent_names[i2], fontsize=10)
      # final title
      fig.suptitle(f"[{factor_omic.name}] {name1}-{name2}", fontsize=10)
      fig.tight_layout(rect=[0.0, 0.02, 1.0, 0.98])
      self.add_figure(name=f"scatter_{factor_omic.name}_{name1}_{name2}",
                      fig=fig)
    return self

  # ******************** metrics ******************** #
  def get_correlation_matrix(self,
                             omic1,
                             omic2=None,
                             corr_type='spearman') -> np.ndarray:
    r""" Correlation matrix of shape `[ndim_omic1, ndim_omic2]`

    Arguments:
      omic1 : First OMIC type.
      omic2 : Second OMIC type, if None, calculate the pair-wise correlation
      corr_type : {'spearman', 'pearson', 'lasso', 'average', 'mi'}
        spearman - rank correlation
        pearson - linear correlation
        average - average of spearman and pearson correlation
        lasso - L1 regression feature importance
        mi - mutual information
    """
    omic1 = OMIC.parse(omic1)
    omic2 = omic1 if omic2 is None else OMIC.parse(omic2)
    x1 = self.dataset.get_omic(omic1)
    x2 = self.dataset.get_omic(omic2)
    corr_type = str(corr_type).lower().strip()
    ###
    if corr_type in ('spearman', 'pearson'):
      fn = (lambda a, b: spearmanr(a, b, nan_policy='omit')[0]) \
        if corr_type == 'spearman' else \
          (lambda a, b: pearsonr(a, b)[0])
      mat = np.empty(shape=(x1.shape[1], x2.shape[1]), dtype=np.float64)
      for i1, a in enumerate(x1.T):
        for i2, b in enumerate(x2.T):
          mat[i1, i2] = fn(a, b)
    ###
    elif corr_type == 'lasso':
      lasso = Lasso(random_state=1, alpha=0.05, max_iter=2000)
      lasso.fit(x1, x2)
      # coef_ is [n_target, n_features], so we need transpose here
      mat = np.transpose(np.absolute(lasso.coef_))
    ###
    elif corr_type == 'average':
      mat = (self.get_correlation_matrix(omic1, omic2, corr_type='spearman') +
             self.get_correlation_matrix(omic1, omic2, corr_type='pearson')) / 2
    ###
    elif corr_type == 'mi':
      mat = np.empty(shape=(x1.shape[1], x2.shape[1]), dtype=np.float64)
      discrete_features = [is_discrete(i) for i in x1.T]
      discrete_targets = [is_discrete(i) for i in x2.T]
      for i, (discrete, target) in enumerate(zip(discrete_targets, x2.T)):
        if discrete:
          y = mutual_info_classif(X=x1,
                                  y=target,
                                  discrete_features=discrete_features,
                                  random_state=1)
        else:
          y = mutual_info_regression(X=x1,
                                     y=target,
                                     discrete_features=discrete_features,
                                     random_state=1)

        mat[:, i] = y
    else:
      raise ValueError("Support corr_type values are: 'spearman', 'pearson', "
                       "'lasso', 'average', 'mi'")
    return mat

  @cache_memory
  def cal_llk(self, omic='transcriptomic'):
    r""" Log-likelihood of a given OMIC type """
    omic = OMIC.parse(omic)
    name = omic.name
    x_org = self.sco_original.get_omic(omic)
    x_cor = self.sco_corrupted.get_omic(omic)
    y_rec = self.omics_data[(name, 'reconstructed')]
    y_imp = self.omics_data[(name, 'imputed')]
    n_samples = tf.constant(y_rec.batch_shape[0], dtype=tf.float32)
    with tf.device("/CPU:0"):
      reduce = lambda llk: tf.reduce_mean(
          tf.reduce_logsumexp(llk, axis=0) - tf.math.log(n_samples),
          axis=0,
      ).numpy()
      return {
          f"llk_{name}_imp_org": reduce(y_imp.log_prob(x_org)),
          f"llk_{name}_imp_cor": reduce(y_imp.log_prob(x_cor)),
          f"llk_{name}_rec_cor": reduce(y_rec.log_prob(x_cor)),
          f"llk_{name}_rec_org": reduce(y_rec.log_prob(x_org)),
      }

  @cache_memory
  def cal_marginal_llk(self):
    r""" calculate the marginal log-likelihood and the reconstruction
    (log-likelihood)

    Arguments:
      original : a Boolean
        use original dataset or artificially corrupted dataset

    Return:
      a Dictionary : mapping scores' name to scalar value (higher is better)
    """
    from tqdm import tqdm
    sco = self.sco_original
    prog = tqdm(sco.create_dataset(sco.omics,
                                   labels_percent=1.0,
                                   batch_size=2,
                                   drop_remainder=True,
                                   shuffle=0),
                desc="Marginal LLK",
                disable=not self.verbose)
    mllk = []
    llk = defaultdict(list)
    for Xs in prog:
      y = self.scm.marginal_log_prob(**Xs, sample_shape=100)
      mllk.append(y[0])
      for i, j in y[1].items():
        llk[i].append(j)
    prog.clear()
    prog.close()
    # aggregate the batches' results
    llk = {
        f"{i}_llk": np.mean(tf.concat(j, axis=0).numpy())
        for i, j in llk.items()
    }
    llk["marginal_llk"] = np.mean(tf.concat(mllk, axis=0).numpy())
    return {f"{i}": j for i, j in llk.items()}

  @cache_memory
  def cal_imputation_scores(self):
    r""" Imputation score, estimated by distance between original and
    imputed values (smaller is better).
    """
    omic = self.sco_original.omics[0].name
    X_org = self.omics_data[(omic, 'original')]
    X_crr = self.omics_data[(omic, 'corrupted')]
    imputed = self.omics_data[(omic, 'imputed')].mean().numpy()
    if imputed.ndim > 2:
      imputed = np.mean(imputed, axis=0)
    return {
        'imputation_med': imputation_score(X_org, imputed),
        'imputation_mean': imputation_mean_score(X_org, X_crr, imputed),
        'imputation_std': imputation_std_score(X_org, X_crr, imputed),
    }

  @cache_memory
  def _matrix_scores(self,
                     score_type,
                     omic1='itranscriptomic',
                     omic2='proteomic',
                     var_names1=MARKER_ADT_GENE.values(),
                     var_names2=MARKER_ADT_GENE.keys()):
    omic1 = OMIC.parse(omic1)
    omic2 = OMIC.parse(omic2)
    sco = self.dataset
    var1 = {name: i for i, name in enumerate(sco.get_var_names(omic1))}
    var2 = {name: i for i, name in enumerate(sco.get_var_names(omic2))}
    if score_type in {'spearman', 'pearson'}:
      cor = sco.get_correlation(omic1, omic2)
      matrix = np.empty(shape=(len(var1), len(var2)), dtype=np.float64)
      for i1, i2, p, s in cor:
        matrix[i1, i2] = p if score_type == 'pearson' else s
    elif score_type == 'mi':
      matrix = sco.get_mutual_information(omic1, omic2)
    elif score_type == 'importance':
      matrix = sco.get_importance_matrix(omic1, omic2)
    else:
      raise NotImplementedError(f"No support for score_type='{score_type}'")
    scores = {}
    for name1, name2 in zip(var_names1, var_names2):
      if name1 in var1 and name2 in var2:
        i1 = var1[name1]
        i2 = var2[name2]
        scores[f"{score_type}_{name1}_{name2}"] = matrix[i1, i2]
    return scores

  def cal_pearson(self,
                  omic1='itranscriptomic',
                  omic2='proteomic',
                  var_names1=MARKER_ADT_GENE.values(),
                  var_names2=MARKER_ADT_GENE.keys()):
    r""" Return a dictionary of Pearson correlation between variables' pair in
    `var_names1` and `var_names2` """
    return self._matrix_scores(score_type='pearson',
                               omic1=omic1,
                               omic2=omic2,
                               var_names1=var_names1,
                               var_names2=var_names2)

  def cal_spearman(self,
                   omic1='itranscriptomic',
                   omic2='proteomic',
                   var_names1=MARKER_ADT_GENE.values(),
                   var_names2=MARKER_ADT_GENE.keys()):
    r""" Return a dictionary of Spearman correlation between variables' pair in
    `var_names1` and `var_names2` """
    return self._matrix_scores(score_type='spearman',
                               omic1=omic1,
                               omic2=omic2,
                               var_names1=var_names1,
                               var_names2=var_names2)

  def cal_mutual_information(self,
                             omic1='itranscriptomic',
                             omic2='proteomic',
                             var_names1=MARKER_ADT_GENE.values(),
                             var_names2=MARKER_ADT_GENE.keys()):
    r""" Return a dictionary of Mutual Information between variables' pair in
    `var_names1` and `var_names2` """
    return self._matrix_scores(score_type='mi',
                               omic1=omic1,
                               omic2=omic2,
                               var_names1=var_names1,
                               var_names2=var_names2)

  def cal_importance(self,
                     omic1='itranscriptomic',
                     omic2='proteomic',
                     var_names1=MARKER_ADT_GENE.values(),
                     var_names2=MARKER_ADT_GENE.keys()):
    r""" Return a dictionary of Importance Score between variables' pair in
    `var_names1` and `var_names2`

    Note:
      This function take a lot of computation time.
    """
    return self._matrix_scores(score_type='importance',
                               omic1=omic1,
                               omic2=omic2,
                               var_names1=var_names1,
                               var_names2=var_names2)

  ######## Disentanglement metrics
  def cal_betavae(self, predict_factor=False):
    r""" BetaVAE score """
    scores = {}
    for key, crt in self._criticizers.items():
      crt: Criticizer
      if not predict_factor and crt.factor_omic.is_imputed:
        continue
      for name, s in crt.cal_betavae_score(n_samples=10000,
                                           verbose=self.verbose).items():
        scores[f"{key}_{name}"] = s
    return scores

  def cal_factorvae(self, predict_factor=False):
    r""" FactorVAE score """
    scores = {}
    for key, crt in self._criticizers.items():
      crt: Criticizer
      if not predict_factor and crt.factor_omic.is_imputed:
        continue
      for name, s in crt.cal_factorvae_score(n_samples=10000,
                                             verbose=self.verbose).items():
        scores[f"{key}_{name}"] = s
    return scores

  def cal_mig(self, predict_factor=False):
    r""" Mutual Information Gap """
    scores = {}
    for key, crt in self._criticizers.items():
      crt: Criticizer
      if not predict_factor and crt.factor_omic.is_imputed:
        continue
      for name, s in crt.cal_mutual_info_gap(mean=True).items():
        scores[f"{key}_{name}"] = s
    return scores

  def cal_dci(self, predict_factor=False):
    r""" Disentanglement, Completeness, Informativeness score"""
    scores = {}
    for key, crt in self._criticizers.items():
      crt: Criticizer
      if not predict_factor and crt.factor_omic.is_imputed:
        continue
      for name, s in crt.cal_dci_scores(mean=True).items():
        scores[f"{key}_{name}"] = s
    return scores

  def __str__(self):
    text = f"Posterior: {self.name}\n"
    text += f"Input  OMICs: {self.input_omics}\n"
    text += f"Output OMICs: {self.output_omics}\n"
    text += "Data:\n"
    for k, v in sorted(self.omics_data.items()):
      desc = [
          str(i).replace('tfp.distributions.', '') if isinstance(
              i, tfd.Distribution) else str([i.shape, i.dtype])
          for i in tf.nest.flatten(v)
      ]
      text += f" {'-'.join(k)}:{', '.join(desc)}\n"
    # dataset SCO
    for line in str(self._dataset).split('\n'):
      text += f" {line}\n"
    text += "Criticizers:\n"
    for key, crt in self._criticizers.items():
      text += f" factor:{key}\n"
    text += "Figures:\n"
    for i, _ in self.figures.items():
      text += f" {i}\n"
    return text[:-1]

  def __hash__(self):
    return id(self)
