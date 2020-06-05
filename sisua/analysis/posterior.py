from __future__ import absolute_import, division, print_function

import inspect
import os
import pickle
import shutil
import time
import warnings
from collections import OrderedDict, defaultdict
from itertools import product, zip_longest
from typing import Optional

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
from sisua.data.utils import standardize_protein_name
from sisua.models.single_cell_model import SingleCellModel
from sisua.utils import dimension_reduction, filtering_experiment_path


# 'X' for input gene expression,
# 'T' for corrupted gene expression,
# 'V' for imputed gene expression,
# and 'W' for reconstructed gene expression
# ===========================================================================
# Helpers
# ===========================================================================
def _omics_footprint(omics, available_omics):
  available_omics = [OMIC.parse(i).name for i in available_omics]
  omics = {OMIC.parse(i).name: j for i, j in omics.items()}
  omics = {
      name: omics[name] if name in omics else 'original'
      for name in available_omics
  }
  if OMIC.latent.name not in omics:
    omics[OMIC.latent.name] = 'original'
  assert all(k in Posterior.TYPES for k in omics.values()), \
    f"Invalid output types: {omics} only support: {Posterior.TYPES}"
  # create unique key
  omics = dict(OrderedDict(sorted(omics.items())))
  return omics


def _sco_shortname(footprint: dict):
  if isinstance(footprint, string_types):
    footprint = eval(footprint)
  assert isinstance(footprint, dict)
  footprint = sorted([f"{k[:3]}_{v[:3]}" for k, v in footprint.items()])
  return '_'.join(footprint)


# crt.sample_batch(latents=post.get_data('latent', 'corrupted'),
#                  factors=post.get_data('proteomic', 'original'),
#                  n_samples=10000)


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
    self.verbose = bool(verbose)
    self.sample_shape = int(sample_shape)
    self.batch_size = int(batch_size)
    self.rand = random_state \
      if isinstance(random_state, np.random.RandomState) else \
        np.random.RandomState(seed=random_state)
    assert callable(reduce_latents), \
      ("reduce_latents must be callable to merge multiple latents "
       "into single one for evaluation")
    self.reduce_latents = reduce_latents
    # single cell model
    if isinstance(scm, string_types):
      with open(scm, 'rb') as f:
        scm = pickle.load(f)
    assert isinstance(scm, SingleCellModel), \
      "scm must be instance of SingleCellModel, but given %s" % str(type(scm))
    self.scm = scm
    if not self.scm.is_fitted and verbose:
      warnings.warn("The SingleCellModel is not fitted.")
    # multi-omics expression data
    assert isinstance(sco, SingleCellOMIC), \
      "sco must be instance of SingleCellOMIC, but given %s" % str(type(sco))
    assert sco.n_omics >= 2, \
      "SingleCellOMIC need at least 2 different OMIC types for evaluation."
    self.sco_original = sco
    self.sco = sco.corrupt(dropout_rate=dropout_rate,
                           retain_rate=retain_rate,
                           distribution=corrupt_distribution,
                           inplace=False,
                           seed=self.rand.randint(1e8))
    self.omics = set(i.name for i in self.sco.omics)
    self.omics_data = {
        (om.name, 'original'): sco.get_omic(om) for om in sco.omics
    }
    # mapping from tuple of (omic.name, type) -> distribution/tensor
    self._dataset = None
    self._criticizers = dict()
    self._initialize()
    # create the default sco and criticizer for analysis
    sco = self._create_sco()
    factor_omic = None
    for om in ['proteomic', 'celltype', 'disease']:
      if om in sco.omics:
        factor_omic = om
        break
    if factor_omic is not None:
      self.create_criticizer(factor_omic=factor_omic)

  @property
  def dataset(self) -> SingleCellOMIC:
    return self._dataset

  def get_data(self, omic, type):
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
    sco = self.sco
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
    self.omics_data[(OMIC.latent.name, 'original')] = tf.nest.flatten(latents)
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

  def _create_sco(self) -> SingleCellOMIC:
    r""" Create a SingleCellOMIC dataset based on the output of
    SingleCellModel

    Arguments:
      omics_config : a Dictionary
    """
    if self._dataset is None:
      sco = self.sco_original.copy()
      for om in self.omics:
        if (om, 'imputed') in self.omics_data:
          data_type = 'imputed'
        elif (om, 'reconstructed') in self.omics_data:
          data_type = 'reconstructed'
        else:
          break
        data = self.omics_data[(om, data_type)]
        om_new = OMIC.parse(f'i{om}')
        # set the new data
        if isinstance(data, tfd.Distribution):
          data = data.mean().numpy()
          if data.ndim == 3:
            data = np.mean(data, axis=0)
        elif isinstance(data, (tuple, list)):
          data = [i.mean() for i in data]
          data = data[0] if len(data) == 1 else self.reduce_latents(data)
          if hasattr(data, 'numpy'):
            data = data.numpy()
        # set the omic values
        if om == OMIC.latent:
          var_names = np.array([f'Z{i}' for i in range(data.shape[1])])
        elif om in self.scm.metadata:
          var_names = self.scm.metadata[om]
        else:
          var_names = np.array([f'{om}{i}' for i in range(data.shape[1])])
        sco.add_omic(omic=om_new, X=data, var_names=var_names)
      self._dataset = sco
    return self._dataset

  def create_criticizer(self,
                        factor_omic='proteomic',
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
    key = factor_omic.name
    # create the Criticizer
    if key not in self._criticizers:
      # create the criticizer
      crt = Criticizer(vae=self.scm, random_state=self.rand.randint(1e8))
      crt.sample_batch(latents=self.omics_data[('latent', 'original')],
                       factors=sco.get_omic(factor_omic),
                       factors_name=sco.get_var_names(factor_omic),
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

  # ******************** helper ******************** #
  def _iter_sco(self, title):
    for cfg, sco in self._dataset.items():
      cfg = eval(cfg)
      key = '_'.join(
          [f"{i[:3]}_{j[:3]}" for i, j in cfg.items() if j != 'original'])
      fig = []
      yield key, cfg, sco, fig
      if len(fig) == 0:
        raise RuntimeError
      self.add_figure(f"{title}_{key}", fig[0])

  # ******************** Latent space analysis ******************** #
  def plot_scatter(self,
                   X='latent',
                   color_by='proteomic',
                   marker_by=None,
                   clustering='kmeans',
                   dimension_reduction='tsne',
                   max_scatter_points=-1,
                   **kwargs):
    X = OMIC.parse(X)
    color_by = OMIC.parse(color_by)
    title = f"scatter_{X.name}_{color_by.name}"
    for key, cfg, sco, fig in self._iter_sco(title):
      plt.figure(figsize=(8, 8))
      fig.append(
          sco.plot_scatter(X=X,
                           color_by=color_by,
                           marker_by=marker_by,
                           clustering=clustering,
                           dimension_reduction=dimension_reduction,
                           max_scatter_points=max_scatter_points,
                           ax=None,
                           return_figure=True,
                           **kwargs))
    return self

  def plot_violins(self,
                   X='transcriptomic',
                   group_by='proteomic',
                   groups=None,
                   var_names=MARKER_GENES,
                   clustering='kmeans',
                   rank_vars=0,
                   **kwargs):
    X = OMIC.parse(X)
    group_by = OMIC.parse(group_by)
    for key, cfg, sco, fig in self._iter_sco(
        f"violins_{X.name}_{group_by.name}{'_rank' if rank_vars > 0 else ''}"):
      fig.append(
          sco.plot_stacked_violins(X=X,
                                   group_by=group_by,
                                   groups=groups,
                                   var_names=var_names,
                                   clustering=clustering,
                                   rank_vars=rank_vars,
                                   return_figure=True,
                                   **kwargs))
    return self

  def plot_heatmap(self,
                   X='transcriptomic',
                   group_by='proteomic',
                   groups=None,
                   var_names=MARKER_GENES,
                   clustering='kmeans',
                   rank_vars=0,
                   **kwargs):
    X = OMIC.parse(X)
    group_by = OMIC.parse(group_by)
    for key, cfg, sco, fig in self._iter_sco(
        f"heatmap_{X.name}_{group_by.name}{'_rank' if rank_vars > 0 else ''}"):
      fig.append(
          sco.plot_heatmap(X=X,
                           group_by=group_by,
                           groups=groups,
                           var_names=var_names,
                           clustering=clustering,
                           rank_vars=rank_vars,
                           return_figure=True,
                           **kwargs))
    return self

  def plot_distance_heatmap(self,
                            X='transcriptomic',
                            group_by='proteomic',
                            clustering='kmeans',
                            **kwargs):
    r""" Distance Heatmap """
    X = OMIC.parse(X)
    group_by = OMIC.parse(group_by)
    title = f"distance_heatmap_{X.name}_{group_by.name}"
    for key, cfg, sco, fig in self._iter_sco(title):
      plt.figure(figsize=(8, 8))
      fig.append(
          sco.plot_distance_heatmap(X=X,
                                    group_by=group_by,
                                    clustering=clustering,
                                    title=key,
                                    ax=None,
                                    return_figure=True,
                                    **kwargs))
    return self

  def plot_correlation_matrix(self,
                              score_type,
                              omic1=OMIC.transcriptomic,
                              omic2=OMIC.proteomic,
                              var_names1=MARKER_ADT_GENE.values(),
                              var_names2=MARKER_ADT_GENE.keys(),
                              is_marker_pairs=True):
    r""" Heatmap correlation between omic1 (x-axis) and omic2 (y-axis) """
    omic1 = OMIC.parse(omic1)
    omic2 = OMIC.parse(omic2)
    title = f"{score_type.lower()}_{omic1.name}_{omic2.name}"
    for key, cfg, sco, fig in self._iter_sco(title):
      if score_type == 'pearson':
        fn = sco.plot_pearson_matrix
      elif score_type == 'spearman':
        fn = sco.plot_spearman_matrix
      elif score_type == 'mutual_information':
        fn = sco.plot_mutual_information
      else:
        raise NotImplementedError(
            f"No implementation for score_type={score_type}")
      fig.append(
          fn(omic1=omic1,
             omic2=omic2,
             var_names1=var_names1,
             var_names2=var_names2,
             is_marker_pairs=is_marker_pairs,
             return_figure=True))
    return self

  def plot_correlation_scatter(self,
                               omic1='transcriptomic',
                               omic2='proteomic',
                               var_names1=MARKER_ADT_GENE.values(),
                               var_names2=MARKER_ADT_GENE.keys(),
                               is_marker_pairs=True,
                               log_omic1=True,
                               log_omic2=True,
                               max_scatter_points=200,
                               top=3,
                               bottom=3):
    r""" Scatter correlation plot """
    omic1 = OMIC.parse(omic1)
    omic2 = OMIC.parse(omic2)
    title = f"corrscat_{omic1.name}_{omic2.name}"
    for key, cfg, sco, fig in self._iter_sco(title):
      fig.append(
          sco.plot_correlation_scatter(omic1=omic1,
                                       omic2=omic2,
                                       var_names1=var_names1,
                                       var_names2=var_names2,
                                       is_marker_pairs=is_marker_pairs,
                                       log_omic1=log_omic1,
                                       log_omic2=log_omic2,
                                       top=top,
                                       bottom=bottom,
                                       max_scatter_points=max_scatter_points,
                                       return_figure=True))
    return self

  def plot_latents_protein_pairs(self,
                                 legend=True,
                                 algo='tsne',
                                 all_pairs=False):
    r""" Using marker gene/protein to select mutual exclusive protein
    pairs for comparison """
    z, y = self.Z, self.y_true
    title = self.name
    fig = plot_latents_protein_pairs(Z=z,
                                     y=y,
                                     labels_name=self.labels,
                                     all_pairs=all_pairs,
                                     title=title,
                                     algo=algo,
                                     show_colorbar=bool(legend))
    if fig is not None:
      self.add_figure('latents_protein_pairs', fig)
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

  # ******************** Streamline classifier ******************** #
  def plot_classifier_F1(self,
                         x_train: Optional[SingleCellOMIC] = None,
                         y_train: Optional[SingleCellOMIC] = None,
                         plot_train_results=False,
                         mode='ovr'):
    r"""
    Arguments:
      mode : {'ovr', 'ovo'}
        ovr - one vs rest
        ovo - one vs one
      fig : Figure or tuple (`float`, `float`), optional (default=`None`)
        width, height in inches
    """
    if mode == 'ovo':
      raise NotImplementedError
    # ====== prepare the results ====== #
    if x_train is not None and y_train is not None:
      _, latents, y_train = self._train_data(x_train, y_train)
      Z_train = latents[0].mean().numpy()
    else:
      Z_train = self.Z
      y_train = self.y_bin
    Z_test, y_test = self.Z, self.y_bin
    # ====== train a plot the classifier ====== #
    (train, test), (fig_train, fig_test) = streamline_classifier(
        Z_train,
        y_train,
        Z_test,
        y_test,
        plot_train_results=plot_train_results,
        labels_name=self.labels,
        show_plot=True,
        return_figure=True)
    if plot_train_results:
      self.add_figure('streamline_f1_%s' % 'train', fig_train)
    self.add_figure('streamline_f1_%s' % 'test', fig_test)
    return self

  # ******************** Cellsize analysis ******************** #
  def plot_cellsize_series(self, log=False, show_corrupted=False, ax=None):
    import seaborn as sns
    sns.set()
    sco = self.sco_corrupt if show_corrupted else self.sco
    outputs = self.outputs_corrupt if show_corrupted else self.outputs_clean
    # original inputs
    x = self.sco.X
    w = np.mean(self.outputs_clean[0].mean(), axis=0)
    if log:
      x = np.log1p(x)
      w = np.log1p(w)
    x = np.sum(x, axis=1)
    w = np.sum(w, axis=1)
    # start plotting
    ids = np.argsort(x)
    ax = to_axis2D(ax)
    styles = dict(linewidth=1.8, alpha=0.8)
    # input data
    x = x[ids]
    w = w[ids]
    ax.plot(x, color='blue', label='Library', **styles)
    ax.plot(w, color='orange', label='Reconstruction', **styles)
    # fine-tune
    ax.legend()
    ax.set_xlabel("Cell in increasing order of library size")
    ax.set_ylabel("%s library size" % ('log' if log else 'raw'))
    self.add_figure('library_%s' % ('log' if log else 'raw'), ax.get_figure())
    return self

  # ******************** Simple analysis ******************** #
  def scores_clustering(self):
    return clustering_scores(latent=self.Z,
                             labels=self.y_true,
                             n_labels=len(self.labels))

  def scores_classifier(self,
                        x_train: Optional[SingleCellOMIC] = None,
                        y_train: Optional[SingleCellOMIC] = None):
    if x_train is not None and y_train is not None:
      _, latents, y_train = self._train_data(x_train, y_train)
      Z_train = latents[0].mean().numpy()
    else:
      Z_train = self.Z
      y_train = self.y_bin
    Z_test, y_test = self.Z, self.y_bin
    return streamline_classifier(Z_train,
                                 y_train,
                                 Z_test,
                                 y_test,
                                 labels_name=self.labels,
                                 return_figure=False,
                                 show_plot=False)

  # ******************** metrics ******************** #
  @cache_memory
  def cal_marginal_llk(self, original=True):
    r""" Calcualte the marginal log-likelihood and the reconstruction
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

  # ******************** Protein analysis ******************** #
  def plot_learning_curves(self, summary_steps=[100, 10], dpi=100):
    r""" Plotting the loss or metrics returned during training progress """
    fig = self.scm.plot_learning_curves(path=None,
                                        summary_steps=summary_steps,
                                        dpi=dpi,
                                        title=self.name)
    self.add_figure('learning_curves', fig)
    return self

  def plot_protein_predicted_series(self,
                                    fontsize=10,
                                    proteins=None,
                                    y_true_new=None,
                                    y_pred_new=None,
                                    labels_new=None):
    if not self.is_semi_supervised:
      return self
    if proteins is not None:
      proteins = [standardize_protein_name(i).lower() for i in proteins]

    y_pred, y_true = self.y_pred, self.y_true
    if not self.is_binary_classes:
      y_true = log_norm(y_true, axis=1).numpy()
    # ====== override provided values ====== #
    labels = self.labels if labels_new is None else labels_new
    if y_true_new is not None:
      y_true = y_true_new
    if y_pred_new is not None:
      y_pred = y_pred_new

    n_protein = len(labels if proteins is None else
                    [i for i in labels if i.lower() in proteins])
    if n_protein == 0:
      return self
    colors = sns.color_palette(n_colors=2)

    #######
    if self.is_binary_classes:
      fig = plt.figure(figsize=(12, 12))
      from sklearn.metrics import confusion_matrix
      y_true = np.argmax(y_true, axis=-1)
      y_pred = np.argmax(y_pred, axis=-1)
      plot_confusion_matrix(cm=confusion_matrix(y_true, y_pred),
                            labels=labels,
                            colorbar=True,
                            fontsize=fontsize)
    #######
    else:
      fig = plt.figure(figsize=(12, n_protein * 4))
      cidx = 0
      for (name, pred, true) in zip(labels, y_pred.T, y_true.T):
        if proteins is not None and name.lower() not in proteins:
          continue
        cidx += 1

        assert pred.shape == true.shape
        ids = np.argsort(true)
        # TODO: handle different type of normalization for protein levels
        # e.g. if self.ynorm == 'prob':
        ax = plt.subplot(n_protein, 1, cidx)

        ax.plot(true[ids],
                linewidth=2,
                color=colors[0],
                label="[True]%s" % name)
        ax.plot(true[ids][0],
                linestyle='--',
                alpha=0.88,
                linewidth=1.2,
                color=colors[1],
                label="[Pred]%s" % name)
        ax.set_ylabel("Log-normalized true protein level", color=colors[0])
        ax.set_xlabel("Cell in sorted order of protein level")
        ax.tick_params(axis='y', colors=colors[0], labelcolor=colors[0])
        ax.set_title(name, fontsize=fontsize)
        ax.legend()

        ax = ax.twinx()
        ax.plot(pred[ids],
                linestyle='--',
                alpha=0.88,
                linewidth=1.2,
                color=colors[1],
                label="[Pred]%s" % name)
        ax.set_ylabel("Predicted protein response", color=colors[1])
        ax.tick_params(axis='y', colors=colors[1], labelcolor=colors[1])

    plt.suptitle(self.name, fontsize=fontsize + 4)
    with catch_warnings_ignore(UserWarning):
      plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    self.add_figure('protein_series', fig)
    return self

  def plot_protein_scatter(self,
                           algo='tsne',
                           protein_name='CD4',
                           y_true_new=None,
                           y_pred_new=None,
                           labels_new=None):
    r""" Plot scatter points of latent space, imputed and original input space
    colored by original protein and predicted protein levels. """
    if not self.is_semi_supervised:
      return self
    # plot all protein in case of None
    if protein_name is None:
      for name in self.labels:
        self.plot_protein_scatter(algo=algo,
                                  protein_name=name,
                                  y_true_new=y_true_new,
                                  y_pred_new=y_pred_new,
                                  labels_new=labels_new)
      return self

    protein_name = standardize_protein_name(protein_name).strip().lower()
    fig_name = 'multispaces_scatter_%s' % protein_name

    labels = self.labels if labels_new is None else labels_new
    labels = [i.strip().lower() for i in labels]

    y_true = self.y_true if y_true_new is None else y_true_new
    y_pred = self.y_pred if y_pred_new is None else y_pred_new

    if protein_name not in labels:
      warnings.warn("Cannot find protein '%s', available protein are: %s" %
                    (protein_name, self.labels))
      return self

    idx = [i for i, j in enumerate(labels) if protein_name in j][0]
    y_true = y_true[:, idx]
    y_pred = y_pred[:, idx]
    X, Z, V = self.X, self.Z, self.I

    fig = plot_figure(nrow=13, ncol=10)
    assert isinstance(fig, plt.Figure), \
    "fig must be instance of matplotlib.Figure"

    x = dimension_reduction(Z, algo=algo)
    ax = plot_scatter(x, val=y_true, ax=321, grid=False, colorbar=True)
    ax.set_ylabel('Scatter of Latent Space')
    ax.set_xlabel('Colored by "Protein Original"')
    ax = plot_scatter(x, val=y_pred, ax=322, grid=False, colorbar=True)
    ax.set_xlabel('Colored by "Protein Predicted"')

    x = dimension_reduction(V, algo=algo)
    ax = plot_scatter(x, val=y_true, ax=323, grid=False, colorbar=True)
    ax.set_ylabel('Scatter of Imputed mRNA')
    ax.set_xlabel('Colored by "Protein Original"')
    ax = plot_scatter(x, val=y_pred, ax=324, grid=False, colorbar=True)
    ax.set_xlabel('Colored by "Protein Predicted"')

    x = dimension_reduction(X, algo=algo)
    ax = plot_scatter(x, val=y_true, ax=325, grid=False, colorbar=True)
    ax.set_ylabel('Scatter of Original mRNA')
    ax.set_xlabel('Colored by "Protein Original"')
    ax = plot_scatter(x, val=y_pred, ax=326, grid=False, colorbar=True)
    ax.set_xlabel('Colored by "Protein Predicted"')

    fig.suptitle(protein_name, fontsize=16)
    with catch_warnings_ignore(UserWarning):
      plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    self.add_figure(fig_name, fig)
    return self

  # ******************** just summary everything  ******************** #
  def full_report(self, path, x_train=None, y_train=None, dpi=80):
    r""" Generate a combined report

    Arguments:
      path : a String, path to a folder saving the report
      x_train, y_train : training data for downstream task
    """
    if not os.path.exists(path):
      os.mkdir(path)
    elif not os.path.isdir(path):
      raise ValueError("path to %s must be a folder" % path)
    clean_folder(path)

    score_path = os.path.join(path, 'scores.yaml')
    self.save_scores(score_path)
    if self.verbose:
      print("Saved scores at:", score_path)

    self.plot_learning_curves()
    if self.is_semi_supervised:
      self.plot_protein_scatter(protein_name=None)
      self.plot_protein_predicted_series()

    self.plot_correlation_top_pairs()
    self.plot_correlation_bottom_pairs()
    self.plot_correlation_marker_pairs(imputed=True)
    self.plot_correlation_marker_pairs(imputed=False)

    self.plot_classifier_F1(x_train, y_train, plot_train_results=True)
    self.plot_latents_binary_scatter()
    self.plot_latents_distance_heatmap()
    self.plot_latents_protein_pairs(all_pairs=True)
    self.save_figures(path,
                      dpi=int(dpi),
                      separate_files=True,
                      clear_figures=True,
                      verbose=self.verbose)
    return self

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
