from __future__ import absolute_import, division, print_function

import inspect
import os
import pickle
import shutil
import time
import warnings
from collections import OrderedDict, defaultdict
from itertools import product
from typing import Optional

import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from matplotlib import pyplot as plt
from six import string_types
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from odin import visual as vs
from odin.backend import log_norm
from odin.bay import vi
from odin.fuel import Dataset
from odin.utils import (as_tuple, cache_memory, catch_warnings_ignore,
                        clean_folder, ctext, flatten_list, md5_checksum)
from odin.utils.mpi import MPI
from odin.visual import (Visualizer, plot_aspect, plot_confusion_matrix,
                         plot_figure, plot_frame, plot_save, plot_scatter,
                         plot_scatter_heatmap, to_axis2D)
from sisua.analysis.imputation_benchmarks import (correlation_scores,
                                                  imputation_mean_score,
                                                  imputation_score,
                                                  imputation_std_score)
from sisua.analysis.latent_benchmarks import (clustering_scores,
                                              plot_distance_heatmap,
                                              plot_latents_binary,
                                              plot_latents_protein_pairs,
                                              streamline_classifier)
from sisua.analysis.sc_metrics import _preprocess_output_distribution
from sisua.data import SingleCellOMIC, apply_artificial_corruption, get_dataset
from sisua.data.path import EXP_DIR
from sisua.data.utils import standardize_protein_name
from sisua.models.base import SingleCellModel
from sisua.models.scvi import SCVI
from sisua.utils import dimension_reduction, filtering_experiment_path


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
    n_mcmc : an Integer, number of MCMC samples for evaluation
    verbose : a Boolean, turn on verbose

  Example
  -------
  >>> pos1 = Posterior(mae, x_test, y_test)
  >>> print(pos1.scores_classifier(x_train, y_train))
  >>> print(pos1.scores_llk())
  >>> print(pos1.scores_imputation())
  >>> print(pos1.scores_spearman())
  >>> print(pos1.scores_pearson())
  >>> print(pos1.scores_clustering())
  >>> pos1.save_scores('/tmp/tmp1.txt')
  ...
  >>> pos1.plot_protein_scatter().plot_protein_predicted_series()
  >>> pos1.plot_classifier_F1(x_train, y_train)
  >>> pos1.plot_learning_curves('loss_x')
  >>> pos1.plot_learning_curves()
  >>> pos1.get_correlation_marker_pairs('X')
  >>> pos1.get_correlation_marker_pairs('V')
  >>> pos1.get_correlation_marker_pairs('W')
  >>> pos1.get_correlation_marker_pairs('T')
  >>> pos1.plot_correlation_top_pairs().plot_correlation_bottom_pairs()
  >>> pos1.plot_latents_binary_scatter().plot_latents_distance_heatmap(
  >>> ).plot_latents_protein_pairs()
  >>> pos1.save_figures('/tmp/tmp1.pdf')

  """

  def __init__(self,
               scm: SingleCellModel,
               gene: SingleCellOMIC,
               protein: Optional[SingleCellOMIC] = None,
               batch_size=16,
               n_mcmc=10,
               verbose=True):
    super(Posterior, self).__init__()
    self.verbose = bool(verbose)
    self.n_mcmc = int(n_mcmc)
    self.batch_size = int(batch_size)

    if isinstance(scm, string_types):
      with open(scm, 'rb') as f:
        scm = pickle.load(f)
    assert isinstance(scm, SingleCellModel), \
      "scm must be instance of SingleCellModel but given %s" % str(type(scm))
    self.scm = scm
    # gene expression data
    assert isinstance(gene, SingleCellOMIC), \
      "gene must be instance of SingleCellOMIC"
    self.gene = gene
    # protein level data
    assert isinstance(protein, (SingleCellOMIC, type(None))), \
      "protein must be instance of SingleCellOMIC or None"
    if protein is not None:
      if 'pbe' not in protein.uns:
        protein.probabilistic_embedding()
    self.protein = protein
    # corrupted gene expression
    gene_corrupt = self.gene.corrupt(corruption_rate=self.scm.corruption_rate,
                                     corruption_dist=self.scm.corruption_dist,
                                     inplace=False)
    self.gene_corrupt = gene_corrupt
    # initialize all the prediction
    self._initialize()

  def _initialize(self):
    scm = self.scm
    gene = self.gene
    gene_corrupt = self.gene_corrupt
    protein = self.protein

    if self.verbose:
      print("Making prediction for clean data ...")
    outputs_clean, latents_clean = scm.predict(gene,
                                               apply_corruption=False,
                                               n_mcmc=self.n_mcmc,
                                               batch_size=self.batch_size,
                                               enable_cache=False,
                                               verbose=self.verbose)
    if not isinstance(outputs_clean, (tuple, list)):
      outputs_clean = [outputs_clean]
    if not isinstance(latents_clean, (tuple, list)):
      latents_clean = [latents_clean]

    if self.verbose:
      print("Making prediction for corrupted data ...")
    outputs_corrupt, latents_corrupt = scm.predict(gene_corrupt,
                                                   apply_corruption=False,
                                                   n_mcmc=self.n_mcmc,
                                                   batch_size=self.batch_size,
                                                   enable_cache=False,
                                                   verbose=self.verbose)
    if not isinstance(outputs_corrupt, (tuple, list)):
      outputs_corrupt = [outputs_corrupt]
    if not isinstance(latents_corrupt, (tuple, list)):
      latents_corrupt = [latents_corrupt]

    self.outputs_clean = outputs_clean
    self.latents_clean = latents_clean
    self.outputs_corrupt = outputs_corrupt
    self.latents_corrupt = latents_corrupt

  # ******************** Basic matrices ******************** #
  @property
  def gene_name(self):
    return self.gene.var['geneid'].values

  @property
  def protein_name(self):
    name = self.protein.var['protid'].values
    return [standardize_protein_name(i) for i in name]

  @property
  def gene_dim(self):
    return self.gene.shape[1]

  @property
  def prot_dim(self):
    return self.protein.shape[1]

  @property
  def X(self):
    r""" Original gene expression data without artificial corruption """
    return self.gene.X

  @property
  def T(self):
    r""" Target variables: artificial corrupted gene expression similar
    to the one used for fitting the model """
    return self.gene_corrupt.X

  @property
  def W(self):
    r""" The reconstruction of single-cell matrix (with un-imputed genes) """
    return tf.reduce_mean(self.outputs_corrupt[0].mean(), axis=0).numpy()

  @property
  def V(self):
    r""" The imputation of single-cell matrix """
    dist = _preprocess_output_distribution(self.outputs_corrupt[0])
    return tf.reduce_mean(dist.mean(), axis=0).numpy()

  @property
  def Z(self):
    """ Latent space of the autoencoder"""
    return self.latents_corrupt[0].mean().numpy()

  @property
  def y_true(self):
    r""" True value of protein marker level or cell type labels """
    return self.protein.X

  @property
  def y_true_binary(self):
    if self.protein.is_binary:
      return self.y_true
    if 'pbe' not in self.protein.uns:
      self.protein.probabilistic_embedding()
    return self.protein.obsm['X_bin']

  @property
  def y_true_probability(self):
    if self.protein.is_binary:
      return self.y_true
    if 'pbe' not in self.protein.uns:
      self.protein.probabilistic_embedding()
    return self.protein.obsm['X_prob']

  @property
  def y_prob(self):
    """ Return the probabilized value of `y_true` """
    if self.is_binary_classes:
      return self.y_true
    return self.protein.obsm['X_prob']

  @property
  def y_bin(self):
    """ Return the binarized value of `y_true` """
    if self.is_binary_classes:
      return self.y_true
    return self.protein.obsm['X_bin']

  @property
  def y_pred(self):
    """ Labels prediction from semi-supervised learning """
    if not self.is_semi_supervised:
      raise RuntimeError(
          "Not semi supervised model, cannot generate prediction for protein values"
      )
    return tf.reduce_mean(self.outputs_corrupt[1].mean(), axis=0).numpy()

  @property
  def name(self):
    i = self.scm.id
    ds_name = self.gene.name.split('_')[0]
    n_gene = '%dgene' % self.X.shape[1]
    n_prot = '%dprot' % (self.y_true.shape[1]
                         if self.protein is not None else 0)
    norm = 'log' if self.scm.log_norm else 'raw'
    corruption_dist = self.scm.corruption_dist
    corruption_rate = self.scm.corruption_rate
    if 'bi' in corruption_dist:
      corruption = 'bi' + str(int(corruption_rate * 100))
    else:
      corruption = 'un' + str(int(corruption_rate * 100))
    if self.is_semi_supervised:
      semi = 'semi'
    else:
      semi = 'unsp'
    return '_'.join([ds_name, i, norm, n_gene, n_prot, corruption, semi])

  @property
  def name_lines(self):
    """same as short_id but divided into 3 lines"""
    short_id = self.name.split('_')
    return '\n'.join(
        ['_'.join(short_id[:2]), '_'.join(short_id[2:-1]), short_id[-1]])

  @property
  def is_semi_supervised(self):
    return self.scm.is_semi_supervised

  @property
  def is_binary_classes(self):
    return np.all(np.sum(self.y_true, axis=-1) == 1.)

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
                                        n_mcmc=self.n_mcmc,
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
                                     labels_name=self.protein_name,
                                     all_pairs=all_pairs,
                                     title=title,
                                     algo=algo,
                                     show_colorbar=bool(legend))
    if fig is not None:
      self.add_figure('latents_protein_pairs', fig)
    return self

  def plot_latents_distance_heatmap(self, legend=True, ax=None, fig=(8, 8)):
    r""" Heatmap of the distance among latents vector from different classes
    """
    ax = to_axis2D(ax, fig)
    z, y = self.Z, self.y_true
    title = self.name
    if not self.is_binary_classes:
      y = self.y_true_probability
    plot_distance_heatmap(z,
                          labels=y,
                          labels_name=self.protein_name,
                          legend_enable=bool(legend),
                          ax=ax,
                          fontsize=8,
                          legend_ncol=2,
                          title=title)
    self.add_figure('latents_distance_heatmap', ax.get_figure())
    return self

  def plot_latents_risk(self, n_mcmc=100, seed=1):
    r""" R.I.S.K :
     - Representative : llk from GMM of protein
     - Informative : mutual information
     - Supportive : streamline task
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
    Z = qZ.sample(n_mcmc, seed=seed).numpy()
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
                         xticklabels=["Z#%d" % i for i in range(n_latent)],
                         yticklabels=self.protein_name,
                         xlabel="Latent dimension",
                         ylabel="Protein",
                         colorbar_title="Log-likelihood",
                         colorbar=True,
                         fontsize=14,
                         annotation=True,
                         text_colors=dict(diag="black",
                                          minrow="green",
                                          maxrow="red",
                                          other="black"),
                         title="Latent presentativeness matrix")
    self.add_figure('latents_llk_mcmc%d_seed%d' % (n_mcmc, seed),
                    ax.get_figure())
    # ====== mutual information ====== #

    return self

  def plot_latents_uncertainty_scatter(self, n_samples=2, seed=1):
    for qZ_clean, qZ_corrupted in zip(self.latents_clean, self.latents_corrupt):
      mean, samples = qZ_clean.mean(), qZ_clean.sample(n_samples, seed=seed)
      print(mean, samples)

  def plot_latents_binary_scatter(self,
                                  legend=True,
                                  algo='tsne',
                                  size=8,
                                  ax=None):
    r""" Scatter plot of dimension using binarized protein labels """
    ax = to_axis2D(ax, (8, 8))
    z, y = self.Z, self.y_true
    title = self.name_lines
    plot_latents_binary(Z=z,
                        y=y,
                        title=title,
                        show_legend=bool(legend),
                        size=8,
                        fontsize=8,
                        ax=ax,
                        labels_name=self.protein_name,
                        algo=algo,
                        enable_argmax=True,
                        enable_separated=False)
    self.add_figure('latents_scatter_%s' % str(algo).lower(), ax.get_figure())
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
        labels_name=self.protein_name,
        show_plot=True,
        return_figure=True)
    if plot_train_results:
      self.add_figure('streamline_f1_%s' % 'train', fig_train)
    self.add_figure('streamline_f1_%s' % 'test', fig_test)
    return self

  # ******************** Cellsize analysis ******************** #
  def plot_cellsize_series(self, fontsize=10, ax=None, fig=None):
    ax = to_axis2D(ax, fig)
    mean, std, x = self.L_test, self.Lstddev_test, self.X_test_org
    mean = mean.ravel()
    std = std.ravel()
    cell_size = np.sum(x, axis=-1)
    sorted_ids = np.argsort(cell_size)

    ax.plot(cell_size[sorted_ids], linewidth=1, label="Original")
    ax.plot(mean[sorted_ids],
            linestyle='--',
            alpha=0.66,
            linewidth=1,
            label='Prediction')

    ax.set_title('[%s]%s' % ('Test' if test else 'Train', self.name),
                 fontsize=fontsize)
    ax.set_ylabel('Cell Size')
    ax.set_xlabel('Cell in sorted order of size')
    ax.legend()

    self.add_figure('cellsize_%s' % ('test' if test else 'train'),
                    ax.get_figure())
    return self

  # ******************** Correlation analysis ******************** #
  def plot_correlation_top_pairs(self,
                                 data_type='V',
                                 proteins=None,
                                 n=8,
                                 fontsize=10):
    return self._plot_correlation_ranked_pairs(data_type=data_type,
                                               n=n,
                                               proteins=proteins,
                                               top=True,
                                               fontsize=fontsize)

  def plot_correlation_bottom_pairs(self,
                                    data_type='V',
                                    proteins=None,
                                    n=8,
                                    fontsize=10):
    return self._plot_correlation_ranked_pairs(data_type=data_type,
                                               n=n,
                                               proteins=proteins,
                                               top=False,
                                               fontsize=fontsize)

  def _plot_correlation_ranked_pairs(self,
                                     data_type='V',
                                     n=8,
                                     proteins=None,
                                     top=True,
                                     fontsize=14):
    r"""
    Arguments:
      data_type : {'X', 'T', 'V', 'W'}
        'X' for input gene expression,
        'T' for corrupted gene expression,
        'V' for imputed gene expression,
        and 'W' for reconstructed gene expression
      proteins : {None, 'marker', list of string}
    """
    correlations = self.get_correlation_all_pairs(data_type=data_type)
    y = self.y_true
    if data_type == 'V':
      ydata = self.V
      data_type_name = "Imputed"
    elif data_type == 'X':
      ydata = self.X
      data_type_name = "Original"
    elif data_type == 'T':
      ydata = self.T
      data_type_name = "Corrupted"
    elif data_type == 'W':
      ydata = self.W
      data_type_name = "Reconstructed"

    n = int(n)
    if isinstance(proteins,
                  string_types) and proteins.lower().strip() == 'marker':
      from sisua.data.const import MARKER_GENES
      proteins = [
          i for i in self.protein_name
          if standardize_protein_name(i) in MARKER_GENES
      ]
    elif proteins is None:
      proteins = self.protein_name
    proteins = as_tuple(proteins, t=string_types)

    labels = {
        standardize_protein_name(j).lower(): i
        for i, j in enumerate(self.protein_name)
    }
    prot_ids = []
    for i in proteins:
      i = standardize_protein_name(i).lower()
      if i in labels:
        prot_ids.append(labels[i])
    prot_ids = set(prot_ids)

    # mapping protein_id -> (gene, pearson, spearman)
    correlations_map = defaultdict(list)
    for gene_id, prot_id, pearson, spearman in correlations:
      if prot_id in prot_ids:
        correlations_map[prot_id].append((gene_id, pearson, spearman))
    correlations_map = {
        i: j[:n] if top else j[-n:][::-1] for i, j in correlations_map.items()
    }
    # ====== create figure ====== #
    nrow = len(correlations_map)
    if nrow == 0:  # no matching protein found
      return self
    ncol = n
    ax = to_axis2D(ax=None, fig=(ncol * 4, int(nrow * 3.6)))
    fig = ax.get_figure()
    # ====== plotting ====== #
    for i, (prot_idx, data) in enumerate(correlations_map.items()):
      prot = self.protein_name[prot_idx]
      for j, (gene_idx, pearson, spearman) in enumerate(data):
        ax = plt.subplot(nrow, ncol, i * ncol + j + 1)
        gene = self.gene_name[gene_idx]
        sns.scatterplot(x=y[:, prot_idx],
                        y=ydata[:, gene_idx],
                        ax=ax,
                        alpha=0.8,
                        linewidths=None)

        title = 'Pearson:%.2f Spearman:%.2f' % (pearson, spearman)
        ax.set_title(title, fontsize=fontsize)
        if j == 0:
          ax.set_ylabel('Protein:%s' % prot, fontsize=fontsize)
        ax.set_xlabel('Gene:%s' % gene, fontsize=fontsize)
    # ====== store the figure ====== #
    plt.suptitle('[data_type: %s] %s %d pairs' %
                 (data_type_name, 'Top' if top else 'Bottom', n),
                 fontsize=fontsize + 6)
    with catch_warnings_ignore(UserWarning):
      plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    self.add_figure(
        'correlation_%s_%s%d' %
        (data_type_name.lower(), 'top' if top else 'bottom', n), fig)
    return self

  def plot_correlation_marker_pairs(self,
                                    imputed=True,
                                    fontsize=12,
                                    proteins=None):
    r""" Plotting the correlation series between marker gene and protein,
    The original (uncorrupted data) is used for comparison to the
    imputation.

    Arguments:
      imputed : `bool` (default: True)
        if `True`, plot the imputed value (in case of zero-inflated distribution),
        otherwise, plot the reconstructed value
      proteins : {`None`, `str`, list of `str`} (default=`None`)
        a list of protein names to be included, if `None`, include all marker
        proteins.
    """
    from scipy.stats import pearsonr, spearmanr
    if proteins is not None:
      proteins = [
          standardize_protein_name(i).lower() for i in as_tuple(proteins, t=str)
      ]

    if imputed:
      v = self.V
      name = 'Imputed'
    else:
      v = self.W
      name = "Reconstructed"

    x, y = self.X, self.y_true
    original_series = correlation_scores(X=x,
                                         y=y,
                                         gene_name=self.gene_name,
                                         protein_name=self.protein_name,
                                         return_series=True)
    imputed_series = correlation_scores(X=v,
                                        y=y,
                                        gene_name=self.gene_name,
                                        protein_name=self.protein_name,
                                        return_series=True)
    # only select given protein
    if proteins is not None:
      original_series = {
          i: j
          for i, j in original_series.items()
          if i.split('/')[0].lower() in proteins
      }
      imputed_series = {
          i: j
          for i, j in original_series.items()
          if i.split('/')[0].lower() in proteins
      }

    assert len(original_series) == len(imputed_series)
    n_pair = len(imputed_series)
    if n_pair == 0:
      return self

    fig = plt.figure(figsize=(15, 5 * n_pair), constrained_layout=True)
    width = 4
    grids = fig.add_gridspec(n_pair, 2 * width)

    for idx, prot_gene in enumerate(sorted(imputed_series.keys())):
      prot_name, gene_name = prot_gene.split('/')
      imputed_gene, prot1 = imputed_series[prot_gene]
      original_gene, prot2 = original_series[prot_gene]
      assert np.all(prot1 == prot2)
      y = prot1
      name = 'Imputed' if imputed else 'Reconstructed'

      for j, (name, series) in enumerate(
          (("Original", original_gene), (name, imputed_gene))):
        ax = plt.subplot(grids[idx, width * j:(width * j + width - 1)])

        # plot the points
        ax.scatter(y, series, s=25, alpha=0.6, linewidths=0)
        plot_aspect('auto', 'box', ax)

        # annotations
        ax.set_title('[%s]Pearson:%.2f Spearman:%.2f' % (
            name,
            pearsonr(series, y)[0],
            spearmanr(series, y).correlation,
        ),
                     fontsize=fontsize)
        ax.set_xlabel('[Protein] %s' % prot_name, fontsize=fontsize)
        ax.set_ylabel('[Gene] %s' % gene_name, fontsize=fontsize)

        # box plot for the distribution
        ax = fig.add_subplot(grids[idx, (width * j + width - 1):(width * j +
                                                                 width)])
        ax.boxplot(series)
        ax.set_xticks(())
        ax.set_xlabel(name, fontsize=fontsize)
    # set title and save the figure
    plt.suptitle('[%s]%s' % (name, self.name), fontsize=fontsize + 2)
    with catch_warnings_ignore(UserWarning):
      plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    self.add_figure('correlation_marker_%s' % name.lower(), fig)
    return self

  # ******************** Simple analysis ******************** #
  def scores_llk(self):
    """ Return the 'llk' (log-likelihood) of the output distribution
    and the corrupted data used, also includes the 'llk_imputed' which
    is the log-likelihood of the denoised distribution and the original
    data without corruption.
    """
    llk = tf.reduce_mean(self.outputs_corrupt[0].log_prob(self.T)).numpy()
    llk_imputed = tf.reduce_mean(
        _preprocess_output_distribution(self.outputs_corrupt[0]).log_prob(
            self.X)).numpy()
    return {'llk': llk, 'llk_imputed': llk_imputed}

  def scores_imputation(self):
    """(train_score, train_score_mean, train_score_std),
       (test_score, test_score_mean, test_score_std)"""
    return {
        'all': imputation_score(self.X, self.V),
        'mean': imputation_mean_score(self.X, self.T, self.V),
        'std': imputation_std_score(self.X, self.T, self.V),
    }

  def scores_spearman(self):
    return self.get_correlation_marker_pairs(data_type='V',
                                             score_type='spearman')

  def scores_pearson(self):
    return self.get_correlation_marker_pairs(data_type='V',
                                             score_type='pearson')

  def scores_clustering(self):
    return clustering_scores(latent=self.Z,
                             labels=self.y_true,
                             n_labels=len(self.protein_name))

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
                                 labels_name=self.protein_name,
                                 return_figure=False,
                                 show_plot=False)

  def save_scores(self, path=None):
    r""" Saving all scores to a yaml file """
    import yaml
    all_scores = {"Model": self.name}
    classifier_train, classifier_test = self.scores_classifier()
    for name, scores in (
        ('llk', self.scores_llk()),
        ('pearson', self.scores_pearson()),
        ('spearman', self.scores_spearman()),
        ('classifier_train', classifier_train),
        ('classifier_test', classifier_test),
        ('cluster', self.scores_clustering()),
        ('imputation', self.scores_imputation()),
    ):
      all_scores[name] = {key: float(val) for key, val in scores.items()}
    if path is None:
      return all_scores
    with open(path, 'w') as f:
      yaml.dump(all_scores, f)
    return self

  # ******************** learning curves and metrics ******************** #
  @property
  def learning_curves(self):
    return {
        'train': self.train_history['loss'],
        'valid': self.valid_history['val_loss'],
    }

  @property
  def train_history(self):
    return self.scm.train_history

  @property
  def valid_history(self):
    return self.scm.valid_history

  def plot_learning_curves(self):
    r""" Plotting the loss or metrics returned during training progress """
    fig = self.scm.plot_learning_curves(title=self.name, return_figure=True)
    self.add_figure('learning_curves', fig)
    return self

  # ******************** Correlation scores ******************** #
  def get_correlation_marker_pairs(self, data_type='X', score_type='spearman'):
    """
    Parameters
    ----------
    data_type : {'X', 'V', 'W', 'T'}
      'X' for input gene expression,
      'T' for corrupted gene expression,
      'V' for imputed gene expression,
      and 'W' for reconstructed gene expression

    score_type : {'spearman', 'pearson'}
      spearman correlation for rank (monotonic) relationship, and pearson
      for linear correlation

    Return
    ------
    correlation : OrderedDict
      mapping from marker protein/gene name (string) to
      correlation score (scalar)
    """
    assert score_type in ('spearman', 'pearson')
    assert data_type in ('X', 'V', 'W', 'T')
    y = self.y_true
    X = getattr(self, data_type)
    corr = correlation_scores(X=X,
                              y=y,
                              gene_name=self.gene_name,
                              protein_name=self.protein_name,
                              return_series=False)
    score_idx = 0 if score_type == 'spearman' else 1
    return OrderedDict([(i, j[score_idx]) for i, j in corr.items()])

  @cache_memory
  def get_correlation_all_pairs(self, data_type='X'):
    """
    Parameters
    ----------
    data_type : {'X', 'T', 'V', 'W'}
      'X' for input gene expression,
      'T' for corrupted gene expression,
      'V' for imputed gene expression,
      and 'W' for reconstructed gene expression

    Return
    ------
    correlation : tuple of four scalars
      list of tuple contained 4 scalars
      (gene-idx, protein-idx, pearson, spearman)
      sorted in order from high to low correlation
    """
    assert data_type in ('X', 'V', 'W', 'T')

    from scipy.stats import pearsonr, spearmanr
    v, x, t, w, y = self.V, self.X, self.T, self.W, self.y_true

    n_proteins = y.shape[1]
    n_genes = x.shape[1]
    data = getattr(self, data_type)

    # ====== search for most correlated series ====== #
    def _corr(idx):
      for gene_idx, prot_idx in idx:
        g = data[:, gene_idx]
        p = y[:, prot_idx]
        with catch_warnings_ignore(RuntimeWarning):
          yield (gene_idx, prot_idx, pearsonr(g,
                                              p)[0], spearmanr(g,
                                                               p).correlation)

    jobs = list(product(range(n_genes), range(n_proteins)))

    # ====== multiprocessing ====== #
    mpi = MPI(jobs, func=_corr, ncpu=3, batch=len(jobs) // 3)
    all_correlations = sorted([i for i in mpi],
                              key=lambda scores:
                              (scores[-2] + scores[-1]) / 2)[::-1]
    return all_correlations

  # ******************** Protein analysis ******************** #
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
    labels = self.protein_name if labels_new is None else labels_new
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
      for name in self.protein_name:
        self.plot_protein_scatter(algo=algo,
                                  protein_name=name,
                                  y_true_new=y_true_new,
                                  y_pred_new=y_pred_new,
                                  labels_new=labels_new)
      return self

    protein_name = standardize_protein_name(protein_name).strip().lower()
    fig_name = 'multispaces_scatter_%s' % protein_name

    labels = self.protein_name if labels_new is None else labels_new
    labels = [i.strip().lower() for i in labels]

    y_true = self.y_true if y_true_new is None else y_true_new
    y_pred = self.y_pred if y_pred_new is None else y_pred_new

    if protein_name not in labels:
      warnings.warn("Cannot find protein '%s', available protein are: %s" %
                    (protein_name, self.protein_name))
      return self

    idx = [i for i, j in enumerate(labels) if protein_name in j][0]
    y_true = y_true[:, idx]
    y_pred = y_pred[:, idx]
    X, Z, V = self.X, self.Z, self.V

    fig = plot_figure(nrow=13, ncol=10)
    assert isinstance(fig, plt.Figure), \
    "fig must be instance of matplotlib.Figure"

    x = dimension_reduction(Z, algo=algo)
    ax = plot_scatter_heatmap(x, val=y_true, ax=321, grid=False, colorbar=True)
    ax.set_ylabel('Scatter of Latent Space')
    ax.set_xlabel('Colored by "Protein Original"')
    ax = plot_scatter_heatmap(x, val=y_pred, ax=322, grid=False, colorbar=True)
    ax.set_xlabel('Colored by "Protein Predicted"')

    x = dimension_reduction(V, algo=algo)
    ax = plot_scatter_heatmap(x, val=y_true, ax=323, grid=False, colorbar=True)
    ax.set_ylabel('Scatter of Imputed mRNA')
    ax.set_xlabel('Colored by "Protein Original"')
    ax = plot_scatter_heatmap(x, val=y_pred, ax=324, grid=False, colorbar=True)
    ax.set_xlabel('Colored by "Protein Predicted"')

    x = dimension_reduction(X, algo=algo)
    ax = plot_scatter_heatmap(x, val=y_true, ax=325, grid=False, colorbar=True)
    ax.set_ylabel('Scatter of Original mRNA')
    ax.set_xlabel('Colored by "Protein Original"')
    ax = plot_scatter_heatmap(x, val=y_pred, ax=326, grid=False, colorbar=True)
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
