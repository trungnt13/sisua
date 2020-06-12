from __future__ import absolute_import, division, print_function

import inspect
import os
import pickle
import shutil
import time
import warnings
from collections import OrderedDict, defaultdict
from itertools import product, zip_longest
from typing import Optional, Union

import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from matplotlib import pyplot as plt
from scipy.stats import pearsonr, spearmanr
from six import string_types
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from odin import visual as vs
from odin.backend import log_norm
from odin.bay import distributions as tfd
from odin.bay import vi
from odin.bay.vi import Criticizer
from odin.fuel import Dataset
from odin.stats import is_binary
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
                        OMIC, SingleCellOMIC, apply_artificial_corruption,
                        get_dataset)
from sisua.data.path import EXP_DIR
from sisua.models.single_cell_model import SingleCellModel


# 'X' for input gene expression,
# 'T' for corrupted gene expression,
# 'V' for imputed gene expression,
# and 'W' for reconstructed gene expression
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
               scm: SingleCellModel,
               sco: SingleCellOMIC,
               dropout_rate=0.2,
               retain_rate=0.2,
               corrupt_distribution='binomial',
               batch_size=8,
               sample_shape=10,
               random_state=1,
               reduce_latents=lambda *Zs: tf.concat(Zs, axis=1),
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
    # single cell model
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
    self.omics = set(i.name for i in self.sco_corrupted.omics)
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
        self.create_criticizer(factor_omic=factor_omic)

  @property
  def scm(self) -> SingleCellModel:
    return self._scm

  @property
  def dataset(self) -> SingleCellOMIC:
    return self._dataset

  def get_criticizer(self, factor_omic) -> Criticizer:
    factor_omic = OMIC.parse(factor_omic).name
    crt = []
    for k, v in self._criticizers.items():
      if factor_omic in k.split('_'):
        crt.append(v)
    return crt[0] if len(crt) == 1 else crt

  def get_data(self, omic, type) -> Union[tfd.Distribution, np.ndarray]:
    omic = OMIC.parse(omic).name
    type = str(type).lower().strip()
    key = (omic, type)
    if (omic, type) not in self.omics_data:
      raise ValueError(
          f"No data found for {key}, "
          f"available data include: {list(self.omics_data.keys())}")
    return self.omics_data[(omic, type)]

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
    for om in self.omics:
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
    # create the SingleCellOMIC dataset for analysis
    sco = self.sco_original.copy()
    for om in self.omics:
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
      Zs = self.reduce_latents([z.mean() for z in Zs])
    else:
      Zs = Zs[0].mean()
    with catch_warnings_ignore(UserWarning, RuntimeWarning):
      sco.add_omic(omic=OMIC.latent,
                   X=Zs.numpy(),
                   var_names=np.array([f'Z{i}' for i in range(Zs.shape[1])]))
    self._dataset = sco

  def create_criticizer(self,
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
    key = f"{latent_indices}_{factor_omic.name}"
    # create the Criticizer
    if key not in self._criticizers:
      # create the criticizer
      crt = Criticizer(vae=self.scm,
                       latent_indices=latent_indices,
                       random_state=self.rand.randint(1e8))
      with catch_warnings_ignore(UserWarning):
        crt.sample_batch(latents=self.omics_data[('latent', 'corrupted')],
                         factors=sco.get_omic(factor_omic),
                         factor_names=sco.get_var_names(factor_omic),
                         n_bins=int(n_bins),
                         strategy=strategy)
      self._criticizers[key] = crt
    return self._criticizers[key]

  # ******************** Basic matrices ******************** #
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
    title = f"scatter_{X.name}_{color_by.name}"
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

  def plot_learning_curves(self, summary_steps=[100, 10], dpi=100):
    r""" Plotting the loss or metrics returned during training progress """
    fig = self.scm.plot_learning_curves(path=None,
                                        summary_steps=summary_steps,
                                        dpi=dpi,
                                        title=self.name)
    self.add_figure('learning_curves', fig)
    return self

  def plot_latents_risk(self, sample_shape=100, seed=1):
    r""" R.I.S.K :
     - Representative : llk from GMM of protein
     - Informative : mutual information
     - Supportive : downstream task
     - Knowledge : disentanglement and biological relevant
    """
    # only use clean data here
    if self.protein.is_binary:
      return self
    qZ = self.latents_clean[0]
    factor = StandardScaler().fit_transform(self.y_true)
    n_latent, n_factor = qZ.event_shape[0], factor.shape[1]
    # ====== fit GMM on each protein ====== #
    gmm = []
    for f in factor.T:
      model = GaussianMixture(n_components=2,
                              covariance_type='full',
                              random_state=seed)
      model.fit(np.expand_dims(f, axis=-1))
      gmm.append(model)
    # ====== llk ====== #
    Z = qZ.sample(sample_shape, seed=seed).numpy()
    llk = np.empty(shape=(n_factor, n_latent), dtype=np.float64)
    for factor_idx in range(n_factor):
      for latent_idx in range(n_latent):
        factor = gmm[factor_idx]
        latent = Z[:, :, latent_idx].reshape(-1, 1)
        llk[factor_idx, latent_idx] = factor.score(latent)
    # ====== plotting ====== #
    fig = plt.figure(figsize=(n_factor / 1.5, n_latent / 1.5))
    ax = vs.plot_heatmap(llk,
                         cmap="Blues",
                         ax=None,
                         xticklabels=["Z%d" % i for i in range(n_latent)],
                         yticklabels=self.labels,
                         xlabel="Latent dimension",
                         ylabel="Protein",
                         cbar_title="Log-likelihood",
                         cbar=True,
                         fontsize=14,
                         annotation=True,
                         text_colors=dict(diag="black",
                                          minrow="green",
                                          maxrow="red",
                                          other="black"),
                         title="Latent presentativeness matrix")
    self.add_figure('latents_llk_mcmc%d_seed%d' % (sample_shape, seed),
                    ax.get_figure())
    # ====== mutual information ====== #
    return self

  # ******************** metrics ******************** #
  @cache_memory
  def cal_llk(self, omic='transcriptomic'):
    omic = OMIC.parse(omic)
    name = omic.name
    x_org = self.sco_original.get_omic(omic)
    x_crt = self.sco_corrupted.get_omic(omic)
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
          f"llk_{name}_imp_crt": reduce(y_imp.log_prob(x_crt)),
          f"llk_{name}_rec_crt": reduce(y_rec.log_prob(x_crt)),
          f"llk_{name}_rec_org": reduce(y_rec.log_prob(x_org)),
      }

  @cache_memory
  def cal_marginal_llk(self, original=True):
    r""" calculate the marginal log-likelihood and the reconstruction
    (log-likelihood)

    Arguments:
      original : a Boolean
        use original dataset or artificially corrupted dataset

    Return:
      a Dictionary : mapping scores' name to scalar value (higher is better)
    """
    from tqdm import tqdm
    sco = self.sco if original else self.sco_corrupt
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
    return {f"{'org' if original else 'crr'}_{i}": j for i, j in llk.items()}

  @cache_memory
  def cal_imputation_scores(self):
    r""" Imputation score, estimated by distance between original and
    imputed values (smaller is better).
    """
    omic = self.sco.omics[0].name
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
                     omic1='transcriptomic',
                     type1='imputed',
                     var_names1=MARKER_ADT_GENE.values(),
                     omic2='proteomic',
                     type2='original',
                     var_names2=MARKER_ADT_GENE.keys()):
    omic1 = OMIC.parse(omic1)
    omic2 = OMIC.parse(omic2)
    sco = self._create_sco({omic1.name: type1, omic2.name: type2})
    var1 = {name: i for i, name in enumerate(sco.get_var_names(omic1))}
    var2 = {name: i for i, name in enumerate(sco.get_var_names(omic2))}
    if score_type in {'spearman', 'pearson'}:
      cor = sco.get_correlation(omic1, omic2)
      matrix = np.empty(shape=(len(var1), len(var2)), dtype=np.float64)
      for i1, i2, p, s in cor:
        matrix[i1, i2] = p if score_type == 'pearson' else s
    elif score_type == 'mutual_info':
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
        scores[f"{name1}_{name2}"] = matrix[i1, i2]
    return scores

  def cal_pearson(self,
                  omic1='transcriptomic',
                  type1='imputed',
                  var_names1=MARKER_ADT_GENE.values(),
                  omic2='proteomic',
                  type2='original',
                  var_names2=MARKER_ADT_GENE.keys()):
    return self._matrix_scores(score_type='pearson',
                               omic1=omic1,
                               type1=type1,
                               var_names1=var_names1,
                               omic2=omic2,
                               type2=type2,
                               var_names2=var_names2)

  def cal_spearman(self,
                   omic1='transcriptomic',
                   type1='imputed',
                   var_names1=MARKER_ADT_GENE.values(),
                   omic2='proteomic',
                   type2='original',
                   var_names2=MARKER_ADT_GENE.keys()):
    return self._matrix_scores(score_type='spearman',
                               omic1=omic1,
                               type1=type1,
                               var_names1=var_names1,
                               omic2=omic2,
                               type2=type2,
                               var_names2=var_names2)

  def cal_mutual_information(self,
                             omic1='transcriptomic',
                             type1='imputed',
                             var_names1=MARKER_ADT_GENE.values(),
                             omic2='proteomic',
                             type2='original',
                             var_names2=MARKER_ADT_GENE.keys()):
    return self._matrix_scores(score_type='mutual_info',
                               omic1=omic1,
                               type1=type1,
                               var_names1=var_names1,
                               omic2=omic2,
                               type2=type2,
                               var_names2=var_names2)

  def cal_importance(self,
                     omic1='transcriptomic',
                     type1='imputed',
                     var_names1=MARKER_ADT_GENE.values(),
                     omic2='proteomic',
                     type2='original',
                     var_names2=MARKER_ADT_GENE.keys()):
    return self._matrix_scores(score_type='importance',
                               omic1=omic1,
                               type1=type1,
                               var_names1=var_names1,
                               omic2=omic2,
                               type2=type2,
                               var_names2=var_names2)

  def __str__(self):
    text = f"Posterior: {self.name}\n"
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
