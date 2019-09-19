from __future__ import absolute_import, division, print_function

import inspect
import os
import pickle
import shutil
import time
from collections import OrderedDict, defaultdict
from itertools import product
from typing import Optional

import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from matplotlib import pyplot as plt
from six import string_types

from odin.backend import log_norm
from odin.fuel import Dataset
from odin.ml import fast_pca, fast_tsne
from odin.utils import (as_tuple, cache_memory, catch_warnings_ignore, ctext,
                        flatten_list, md5_checksum)
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
                                              plot_latents_multiclasses,
                                              streamline_classifier)
from sisua.analysis.sc_metrics import _preprocess_output_distribution
from sisua.data import SingleCellOMIC, apply_artificial_corruption, get_dataset
from sisua.data.path import EXP_DIR
from sisua.data.utils import standardize_protein_name
from sisua.models.base import SingleCellModel
from sisua.models.scvi_models import SCVI
from sisua.utils import filtering_experiment_path


# ===========================================================================
# The Posterior
# ===========================================================================
class Posterior(Visualizer):
  """ Posterior

  Parameters
  ----------
  scm : {`sisua.models.SingleCellModel`, string}

  n_samples : int
    number of MCMC samples for evaluation

  verbose : bool
    turn on verbose

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
               n_samples=10,
               verbose=True):
    super(Posterior, self).__init__()
    self.verbose = bool(verbose)
    self.n_samples = int(n_samples)
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

    # NOTE: for now we haven't find the use of input clean yet
    # save some processing time here
    # outputs_clean, latents_clean = scm.predict(gene,
    #                                            apply_corruption=False,
    #                                            n_samples=self.n_samples,
    #                                            batch_size=self.batch_size,
    #                                            enable_cache=False,
    #                                            verbose=self.verbose)
    outputs_clean = []
    latents_clean = []
    if not isinstance(outputs_clean, (tuple, list)):
      outputs_clean = [outputs_clean]
    if not isinstance(latents_clean, (tuple, list)):
      latents_clean = [latents_clean]

    outputs_corrupt, latents_corrupt = scm.predict(gene_corrupt,
                                                   apply_corruption=False,
                                                   n_samples=self.n_samples,
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
    """ Original gene expression data without artificial corruption """
    return self.gene.X

  @property
  def T(self):
    """ The artificial corrupted gene expression similar to the one
    used for fitting the model """
    return self.gene_corrupt.X

  @property
  def W(self):
    return tf.reduce_mean(self.outputs_corrupt[0].mean(), axis=0).numpy()

  @property
  def V(self):
    dist = _preprocess_output_distribution(self.outputs_corrupt[0])
    return tf.reduce_mean(dist.mean(), axis=0).numpy()

  @property
  def Z(self):
    """ Latent space of the autoencoder"""
    return self.latents_corrupt[0].mean().numpy()

  @property
  def y_true(self):
    """ True value of protein marker level or cell type labels """
    return self.protein.X

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
                                        n_samples=self.n_samples,
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
  def plot_latents_protein_pairs(self, legend=True, pca=False):
    """ Using marker gene/protein to select mutual exclusive protein
    pairs for comparison """
    z, y = self.Z, self.y_true
    title = self.name
    fig = plot_latents_multiclasses(Z=z,
                                    y=y,
                                    labels_name=self.protein_name,
                                    title=title,
                                    use_PCA=bool(pca),
                                    show_colorbar=bool(legend))
    if fig is not None:
      self.add_figure('latents_protein_pairs', fig)
    return self

  def plot_latents_distance_heatmap(self, legend=True, ax=None, fig=(8, 8)):
    ax = to_axis2D(ax, fig)
    z, y = self.Z, self.y_true
    title = self.name
    if not self.is_binary_classes:
      from sisua.label_threshold import ProbabilisticEmbedding
      y = ProbabilisticEmbedding().fit_transform(y, return_probabilities=True)
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

  def plot_latents_binary_scatter(self,
                                  legend=True,
                                  pca=False,
                                  size=8,
                                  ax=None,
                                  fig=(8, 8)):
    """
    test : if True, plotting latent space of test set, otherwise, use training set
    """
    ax = to_axis2D(ax, fig)
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
                        use_PCA=pca,
                        enable_argmax=True,
                        enable_separated=False)
    self.add_figure('latents_scatter', ax.get_figure())
    return self

  # ******************** Streamline classifier ******************** #
  def plot_classifier_F1(self,
                         x_train: Optional[SingleCellOMIC] = None,
                         y_train: Optional[SingleCellOMIC] = None,
                         plot_train_results=False,
                         mode='ovr'):
    """
    ovr - one vs rest
    ovo - one vs one
    """
    if mode == 'ovo':
      raise NotImplementedError

    if x_train is not None and y_train is not None:
      _, latents, y_train = self._train_data(x_train, y_train)
      Z_train = latents[0].mean().numpy()
    else:
      Z_train = self.Z
      y_train = self.y_bin
    Z_test, y_test = self.Z, self.y_bin

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
                                 fontsize=10,
                                 fig=None):
    return self._plot_correlation_ranked_pairs(data_type=data_type,
                                               n=n,
                                               proteins=proteins,
                                               top=True,
                                               fontsize=fontsize,
                                               fig=fig)

  def plot_correlation_bottom_pairs(self,
                                    data_type='V',
                                    proteins=None,
                                    n=8,
                                    fontsize=10,
                                    fig=None):
    return self._plot_correlation_ranked_pairs(data_type=data_type,
                                               n=n,
                                               proteins=proteins,
                                               top=False,
                                               fontsize=fontsize,
                                               fig=fig)

  def _plot_correlation_ranked_pairs(self,
                                     data_type='V',
                                     n=8,
                                     proteins=None,
                                     top=True,
                                     fontsize=10,
                                     fig=None):
    """
    Parameters
    ----------
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
    if fig is None:
      fig = plot_figure(nrow=3 * nrow, ncol=4 * ncol)
    for i, (prot_idx, data) in enumerate(correlations_map.items()):
      prot = self.protein_name[prot_idx]
      for j, (gene_idx, pearson, spearman) in enumerate(data):
        ax = plt.subplot(nrow, ncol, i * ncol + j + 1)
        gene = self.gene_name[gene_idx]
        sns.scatterplot(x=y[:, prot_idx], y=ydata[:, gene_idx], ax=ax)

        title = 'Pearson:%.2f Spearman:%.2f' % (pearson, spearman)
        ax.set_title(title, fontsize=fontsize)
        if j == 0:
          ax.set_ylabel('Protein:%s' % prot, fontsize=fontsize + 2)
        ax.set_xlabel('Gene:%s' % gene)

    # ====== store the figure ====== #
    plt.suptitle('[data_type: %s] %s %d pairs' %
                 (data_type_name, 'Top' if top else 'Bottom', n),
                 fontsize=fontsize + 8)
    with catch_warnings_ignore(UserWarning):
      plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    self.add_figure(
        'correlation_%s_%s%d' %
        (data_type_name.lower(), 'top' if top else 'bottom', n), fig)
    return self

  def plot_correlation_marker_pairs(self,
                                    imputed=True,
                                    fontsize=10,
                                    proteins=None,
                                    fig=None):
    """ Plotting the correlation series between marker gene and protein,
    The original (uncorrupted data) is used for comparison to the
    imputation.

    Parameters
    ----------
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

    if fig is None:
      fig = plt.figure(figsize=(15, 5 * n_pair), constrained_layout=True)
    assert isinstance(fig, plt.Figure), \
    "fig must be instance of matplotlib.Figure"

    width = 4
    grids = fig.add_gridspec(n_pair, 2 * width)

    for idx, prot_gene in enumerate(sorted(imputed_series.keys())):
      prot_name, gene_name = prot_gene.split('/')
      imputed_gene, prot1 = imputed_series[prot_gene]
      original_gene, prot2 = original_series[prot_gene]
      assert np.all(prot1 == prot2)
      y = prot1

      for j, (name, series) in enumerate(
          (("Original", original_gene), ("Imputed", imputed_gene))):
        ax = fig.add_subplot(grids[idx, width * j:(width * j + width - 1)])

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

    plt.suptitle('[%s]%s' % (name, self.name), fontsize=fontsize + 2)
    with catch_warnings_ignore(UserWarning):
      plt.tight_layout(rect=[0, 0.03, 1, 0.96])
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
                                 train_results=False,
                                 labels_name=self.protein_name,
                                 show_plot=False)

  def save_scores(self, path=None):
    """ Saving all scores to a txt file """
    text = '==== %s ====\n' % self.name
    for name, scores in (
        ('llk', self.scores_llk()),
        ('pearson', self.scores_pearson()),
        ('spearman', self.scores_spearman()),
        ('classifier', self.scores_classifier()),
        ('cluster', self.scores_clustering()),
        ('imputation', self.scores_imputation()),
    ):
      text += '%s\n' % name
      for i, j in sorted(scores.items()):
        j = '%+.4f' % j
        j = j.replace('+', ' ')
        text += ' %12s :%s\n' % (i, j)
    if path is None:
      return text
    with open(path, 'w') as f:
      f.write(text)
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

  def plot_learning_curves(self,
                           metrics='loss',
                           ignore_case=True,
                           ignore_missing=True,
                           fig=(8, 8)):
    """ Plotting the loss or metrics returned during training progress

    Parameters
    ----------
    metrics : {`str`, list of `str`} (default='loss')
      a metric name or list of metric name for plotting
    ignore_case : `bool` (default=`False`)
      if `True` turn everything to lower case for matching the metric name.
    ignore_missing : `bool` (default=`True`)
      if `True`, ignore the missing metric which not found in the history of
      trained model.
    """
    metrics = as_tuple(metrics, t=str)
    if ignore_case:
      metrics = [m.lower() for m in metrics]
    ax = to_axis2D(None, fig)
    fig = ax.get_figure()

    train_history = self.train_history
    valid_history = self.valid_history

    _ = []
    train_keys = {
        i.lower() if ignore_case else i: i for i in train_history.keys()
    }
    valid_keys = {
        i.lower() if ignore_case else i: i for i in valid_history.keys()
    }
    for m in metrics:
      if not (m in train_keys and 'val_' + m in valid_keys):
        if not ignore_missing:
          raise ValueError(
              "Cannot find metric with name: %s; all given metrics are: %s" %
              (m, ', '.join(train_keys)))
      else:
        _.append(train_keys[m])

    metrics = tuple(_)
    n_metrics = len(metrics)
    if n_metrics == 0:  # no metric or loss found for plotting
      return self

    for idx, m in enumerate(metrics):
      ax = plt.subplot(1, n_metrics, idx + 1)

      train = train_history[m]
      valid = valid_history['val_' + m]

      line_styles = dict(linewidth=1.8)
      point_styles = dict(alpha=0.6, s=80, linewidths=0)

      ax.plot(train, label='train', color='blue', linestyle='-', **line_styles)
      ax.scatter(np.argmin(train), np.min(train), c='blue', **point_styles)

      ax.plot(valid,
              label='valid',
              color='orange',
              linestyle='--',
              **line_styles)
      ax.scatter(np.argmin(valid), np.min(valid), c='orange', **point_styles)

      # ax.set_ylabel("Loss")
      ax.set_xlabel("#Epoch")
      # convert all the xticks to integer and remove overlapping
      xticks = ax.get_xticks()
      ax.set_xticks(list(set([int(i) for i in xticks])))

      ax.legend()
      ax.set_title('[metric: %s]' % m, fontsize=14)

    fig.suptitle(self.name)
    with catch_warnings_ignore(UserWarning):
      plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    self.add_figure('learning_curves_%s' % m, ax.get_figure())
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
                           pca=False,
                           protein_name='CD4',
                           fig=None,
                           y_true_new=None,
                           y_pred_new=None,
                           labels_new=None):
    """ Plot comparison of protein in latent space, imputed and original
    input space """
    if not self.is_semi_supervised:
      return self

    fn_dim = fast_pca if pca else fast_tsne
    protein_name = standardize_protein_name(protein_name).strip().lower()
    fig_name = 'protein_scatter_%s' % protein_name

    labels = self.protein_name if labels_new is None else labels_new
    labels = [i.strip().lower() for i in labels]

    y_true = self.y_true if y_true_new is None else y_true_new
    y_pred = self.y_pred if y_pred_new is None else y_pred_new

    if protein_name not in labels:
      return self

    idx = [i for i, j in enumerate(labels) if protein_name in j][0]
    y_true = y_true[:, idx]
    y_pred = y_pred[:, idx]
    X, Z, V = self.X, self.Z, self.V

    if fig is None:
      fig = plot_figure(nrow=13, ncol=10)
    assert isinstance(fig, plt.Figure), \
    "fig must be instance of matplotlib.Figure"

    x = fn_dim(Z)
    ax = plot_scatter_heatmap(x, val=y_true, ax=321, grid=False, colorbar=True)
    ax.set_xlabel('Latent / ProteinOriginal')
    ax = plot_scatter_heatmap(x, val=y_pred, ax=322, grid=False, colorbar=True)
    ax.set_xlabel('Latent / ProteinPredicted')

    x = fn_dim(V)
    ax = plot_scatter_heatmap(x, val=y_true, ax=323, grid=False, colorbar=True)
    ax.set_xlabel('InputImputed / ProteinOriginal')
    ax = plot_scatter_heatmap(x, val=y_pred, ax=324, grid=False, colorbar=True)
    ax.set_xlabel('InputImputed / ProteinPredicted')

    x = fn_dim(X)
    ax = plot_scatter_heatmap(x, val=y_true, ax=325, grid=False, colorbar=True)
    ax.set_xlabel('InputOriginal / ProteinOriginal')
    ax = plot_scatter_heatmap(x, val=y_pred, ax=326, grid=False, colorbar=True)
    ax.set_xlabel('InputOriginal / ProteinPredicted')

    fig.suptitle(protein_name, fontsize=16)
    with catch_warnings_ignore(UserWarning):
      plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    self.add_figure(fig_name, fig)
    return self

  # ******************** just summary everything  ******************** #
  def full_report(self, path, x_train=None, y_train=None, override=True):
    """
    path : `str`
      path to a folder
    """
    if not os.path.exists(path):
      os.mkdir(path)
    elif not os.path.isdir(path):
      raise ValueError("path to %s must be a folder" % path)
    else:
      if self.verbose:
        print("Override analysis at path:", path)
      shutil.rmtree(path)
      os.mkdir(path)

    score_path = os.path.join(path, 'scores.txt')
    self.save_scores(score_path)
    if self.verbose:
      print("Saved scores at:", score_path)

    self.plot_learning_curves(metric='loss')
    if self.is_semi_supervised:
      self.plot_protein_scatter(protein_name='CD4')
      self.plot_protein_scatter(protein_name='CD8')
      self.plot_protein_scatter(protein_name='CD45RA')
      self.plot_protein_scatter(protein_name='CD45RO')
      self.plot_protein_predicted_series()

    self.plot_correlation_top_pairs().plot_correlation_bottom_pairs()
    self.plot_correlation_marker_pairs(
        imputed=True).plot_correlation_marker_pairs(imputed=False)

    self.plot_classifier_F1(x_train, y_train)
    self.plot_latents_binary_scatter()
    self.plot_latents_distance_heatmap()
    self.plot_latents_protein_pairs()

    self.save_figures(path,
                      dpi=120,
                      separate_files=True,
                      clear_figures=True,
                      verbose=self.verbose)
    return self
