from __future__ import absolute_import, division, print_function

import inspect
import os
import pickle
import time
from collections import OrderedDict, defaultdict
from itertools import product

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from six import string_types

from odin.fuel import Dataset
from odin.ml import fast_pca, fast_tsne
from odin.utils import (as_tuple, cache_memory, catch_warnings_ignore, ctext,
                        flatten_list, md5_checksum)
from odin.visual import (Visualizer, plot_aspect, plot_confusion_matrix,
                         plot_figure, plot_frame, plot_save, plot_scatter,
                         plot_scatter_heatmap, to_axis2D)
from sisua.analysis.imputation_benchmarks import (get_correlation_scores,
                                                  imputation_mean_score,
                                                  imputation_score,
                                                  imputation_std_score)
from sisua.analysis.latent_benchmarks import (clustering_scores,
                                              plot_distance_heatmap,
                                              plot_latents_binary,
                                              plot_latents_multiclasses,
                                              streamline_classifier)
from sisua.data import apply_artificial_corruption, get_dataset
from sisua.data.path import EXP_DIR
from sisua.data.utils import standardize_protein_name
from sisua.models.base import SingleCellModel
from sisua.utils import filtering_experiment_path
from sisua.utils.visualization import save_figures

# ===========================================================================
# Helper
# ===========================================================================
_LOADED_DATASET = {}


def _fast_load_dataset(name):
  if name not in _LOADED_DATASET:
    ds = get_dataset(dataset_name=name)
    _LOADED_DATASET[name] = ds
  return _LOADED_DATASET[name]


def get_all_posteriors(path_or_dataset,
                       incl=[],
                       excl=[],
                       fn_filter=None,
                       show_progress=True):
  if os.path.exists(path_or_dataset):
    exp_path = os.path.dirname(path_or_dataset)
    ds_name = os.path.basename(path_or_dataset)
  else:
    exp_path = ''
    ds_name = path_or_dataset
  all_exp = filtering_experiment_path(ds_name=ds_name,
                                      exp_path=exp_path,
                                      incl_keywords=incl,
                                      excl_keywords=excl,
                                      fn_filter=fn_filter,
                                      return_dataset=False,
                                      print_log=False)
  all_posteriors = []
  if show_progress:
    print("Loading %s posteriors for dataset '%s' at path '%s'..." %
          (ctext(len(all_exp), 'lightyellow'), ctext(
              ds_name, 'lightyellow'), ctext(exp_path, 'lightyellow')))
  with catch_warnings_ignore(Warning):
    for i in all_exp:
      start_time = time.time()
      pos = Posterior(i)
      all_posteriors.append(pos)
      if show_progress:
        print("  Loaded %s for %.2f(s)" %
              (ctext(pos.id, 'lightcyan'), time.time() - start_time))
  return all_posteriors


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

  """

  def __init__(self, scm: SingleCellModel, n_samples=100, verbose=False):
    super(Posterior, self).__init__()
    self.verbose = bool(verbose)
    self.n_samples = int(n_samples)
    if isinstance(scm, string_types):
      pass
    else:
      pass

  def save_scores(self, path):
    assert '.html' in path.lower(), "Only save scores to html file"
    text = ''
    for name, (train, test) in (
        ('scores', self.scores),
        ('pearson', self.scores_pearson),
        ('spearman', self.scores_spearman),
        ('classifier', self.scores_classifier),
        ('cluster', self.scores_clustering),
        ('imputation', self.scores_imputation),
    ):
      text += '<h4>Score type: "%s"<h4>' % name
      df = pd.DataFrame(data=[train, test], index=['train', 'test'])
      text += df.to_html(float_format='%.3f') + '\n'
    with open(path, 'w') as f:
      f.write(text)
    return self

  # ******************** Latent space analysis ******************** #
  def plot_latents_protein_pairs(self, test=True, legend=True, pca=False):
    """ Using marker gene/protein to select mutual exclusive protein
    pair for comparison

    """
    if test:
      z, y = self.Z_test, self.y_test
    else:
      z, y = self.Z_train, self.y_train
    title = ("[test]%s" if test else "[train]%s") % self.short_id

    fig = plot_latents_multiclasses(Z=z,
                                    y=y,
                                    labels_name=self.labels,
                                    title=title,
                                    use_PCA=bool(pca),
                                    show_colorbar=bool(legend))

    if fig is not None:
      self.add_figure('latents_pairs_%s' % ('test' if test else 'train'), fig)
    return self

  def plot_latents_distance_heatmap(self, test=True, legend=True, ax=None):
    ax = to_axis2D(ax)
    if test:
      z, y = self.Z_test, self.y_test
    else:
      z, y = self.Z_train, self.y_train
    title = ("[test]%s" if test else "[train]%s") % self.short_id

    if not self.is_binary_classes:
      from sisua.label_threshold import ProbabilisticEmbedding
      y = ProbabilisticEmbedding().fit_transform(y, return_probabilities=True)

    plot_distance_heatmap(z,
                          labels=y,
                          labels_name=self.labels,
                          legend_enable=bool(legend),
                          ax=ax,
                          fontsize=8,
                          legend_ncol=2,
                          title=title)

    self.add_figure(
        'latents_distance_heatmap_%s' % ('test' if test else 'train'),
        ax.get_figure())
    return self

  def plot_latents_binary_scatter(self,
                                  test=True,
                                  legend=True,
                                  pca=False,
                                  size=8,
                                  ax=None):
    """
    test : if True, plotting latent space of test set, otherwise, use training set
    """
    ax = to_axis2D(ax)
    if test:
      z, y = self.Z_test, self.y_test
    else:
      z, y = self.Z_train, self.y_train
    title = ("[test]%s" if test else "[train]%s") % self.short_id_lines
    plot_latents_binary(Z=z,
                        y=y,
                        title=title,
                        show_legend=bool(legend),
                        size=8,
                        fontsize=8,
                        ax=ax,
                        labels_name=self.labels,
                        use_PCA=pca,
                        enable_argmax=True,
                        enable_separated=False)
    self.add_figure('latents_scatter_%s' % ('test' if test else 'train'),
                    ax.get_figure())
    return self

  def plot_streamline_F1(self, mode='ovr'):
    """
    ovr - one vs rest
    ovo - one vs one
    """
    if mode == 'ovo':
      return self

    (train, test), (fig_train,
                    fig_test) = streamline_classifier(self.Z_train,
                                                      self.y_train,
                                                      self.Z_test,
                                                      self.y_test,
                                                      train_results=True,
                                                      labels_name=self.labels,
                                                      show_plot=True,
                                                      return_figure=True)
    if self.X_train.shape == self.X_test.shape and \
    np.allclose(self.X_train, self.X_test):
      self.add_figure('streamline_f1_%s' % 'test', fig_test)
    else:
      self.add_figure('streamline_f1_%s' % 'train', fig_train)
      self.add_figure('streamline_f1_%s' % 'test', fig_test)
    return self

  # ******************** Imputation analysis ******************** #
  def plot_cellsize_series(self, test=True, fontsize=10, ax=None):
    ax = to_axis2D(ax)
    if test:
      mean, std, x = self.L_test, self.Lstddev_test, self.X_test_org
    else:
      mean, std, x = self.L_train, self.Lstddev_train, self.X_train_org
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

    ax.set_title('[%s]%s' % ('Test' if test else 'Train', self.short_id),
                 fontsize=fontsize)
    ax.set_ylabel('Cell Size')
    ax.set_xlabel('Cell in sorted order of size')
    ax.legend()

    self.add_figure('cellsize_%s' % ('test' if test else 'train'),
                    ax.get_figure())
    return self

  def plot_correlation_top_pairs(self,
                                 test=True,
                                 data_type='V',
                                 n=8,
                                 proteins=None,
                                 top=True,
                                 fontsize=10,
                                 fig=None):
    """
    Parameters
    ----------

    proteins : {None, 'marker', list of string}

    """
    if test:
      v, x, w, y = self.V_test, self.X_test_org, self.W_test, self.y_test
    else:
      v, x, w, y = self.V_train, self.X_train_org, self.W_train, self.y_train
    correlations = self.get_correlation_all_pairs(data_type=data_type,
                                                  test=test)

    if data_type == 'V':
      ydata = v
      data_type_name = "Imputed"
    elif data_type == 'X':
      ydata = x
      data_type_name = "Original"
    elif data_type == 'W':
      ydata = w
      data_type_name = "Reconstructed"

    n = int(n)
    if isinstance(proteins,
                  string_types) and proteins.lower().strip() == 'marker':
      from sisua.data.const import MARKER_GENES
      proteins = [
          i for i in self.labels if standardize_protein_name(i) in MARKER_GENES
      ]
    elif proteins is None:
      proteins = self.labels
    proteins = as_tuple(proteins, t=string_types)

    labels = {
        standardize_protein_name(j).lower(): i
        for i, j in enumerate(self.labels)
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
        ax.set_ylabel('Protein:%s Gene:%s' % (prot, gene),
                      fontsize=fontsize + 2)

    # ====== store the figure ====== #
    plt.suptitle('[set: %s][data_type: %s]%s%d' %
                 ('test' if test else 'train', data_type_name,
                  'top' if top else 'bottom', n),
                 fontsize=fontsize + 2)
    with catch_warnings_ignore(UserWarning):
      plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    self.add_figure(
        'correlation_%s_%s%s%d' %
        ('test' if test else 'train', data_type, 'top' if top else 'bottom', n),
        plt.gcf())
    return self

  def plot_correlation_marker_pairs(self, test=True, fontsize=10, fig=None):
    from scipy.stats import pearsonr, spearmanr
    if test:
      v, x, y = self.V_test, self.X_test_org, self.y_test
    else:
      v, x, y = self.V_train, self.X_train_org, self.y_train
    original_series = get_correlation_scores(X=x,
                                             y=y,
                                             gene_name=self.gene_name,
                                             protein_name=self.labels,
                                             return_series=True)
    imputed_series = get_correlation_scores(X=v,
                                            y=y,
                                            gene_name=self.gene_name,
                                            protein_name=self.labels,
                                            return_series=True)
    assert len(original_series) == len(imputed_series)
    n_pair = len(imputed_series)

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

    data_type = ('test' if test else 'train')
    plt.suptitle('[%s]%s' % (data_type, self.short_id), fontsize=fontsize + 2)
    with catch_warnings_ignore(UserWarning):
      plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    self.add_figure('correlation_%s' % data_type, plt.gcf())
    return self

  # ******************** shortcut to Tensors ******************** #
  def get_cached_output(self, X, name, y=None, n_mcmc_samples=None):
    if not hasattr(self, '_cached_data'):
      self._cached_data = defaultdict(dict)
    assert X.ndim == 2, "Only support 2-D tensor, given X.shape=%s" % X.shape

    f_pred = None
    n_mcmc = (self.n_mcmc_samples
              if n_mcmc_samples is None else int(n_mcmc_samples))
    md5 = md5_checksum(X)
    name = str(name).strip().lower()
    is_semi = self.infer.is_semi_supervised

    if 'v' == name[0]:
      f_pred = self.infer.predict_V
    elif 'vstddev' == name[:7]:
      f_pred = self.infer.predict_Vstddev
    elif 'w' == name[0]:
      f_pred = self.infer.predict_W
    elif 'wstddev' == name[:7]:
      f_pred = self.infer.predict_Wstddev
    elif 'pi' == name[:2]:
      f_pred = self.infer.predict_PI
    elif 'l' == name[0]:
      f_pred = self.infer.predict_L
    elif 'lstddev' == name[:7]:
      f_pred = self.infer.predict_Lstddev
    elif 'z' == name[0]:
      f_pred = self.infer.predict_Z
    elif 'y' == name[0]:
      f_pred = self.infer.predict_y
    elif 'score' == name:
      f_pred = self.infer.score
    else:
      raise RuntimeError("No support for prediction value name: '%s'" % name)

    sign = inspect.getargspec(f_pred)
    kwargs = {}
    if 'n_mcmc_samples' in sign.args:
      kwargs['n_mcmc_samples'] = n_mcmc
    if is_semi and 'y' in sign.args and y is not None:
      kwargs['y'] = y

    output_id = str(n_mcmc) + md5 + name
    if output_id not in self._cached_data:
      self._cached_data[output_id] = f_pred(X, **kwargs)
    return self._cached_data[output_id]

  # ******************** Simple analysis ******************** #
  @property
  def scores(self):
    train = self.get_cached_output(X=self.X_train, name='score', y=self.y_train)
    test = self.get_cached_output(X=self.X_test, name='score', y=self.y_test)
    return train, test

  @property
  def scores_imputation(self):
    """(train_score, train_score_mean, train_score_std),
       (test_score, test_score_mean, test_score_std)"""
    return \
    {'all': imputation_score(self.X_train_org, self.V_train),
     'mean': imputation_mean_score(self.X_train_org, self.X_train, self.V_train),
     'std': imputation_std_score(self.X_train_org, self.X_train, self.V_train)},\
    {'all': imputation_score(self.X_test_org, self.V_test),
     'mean': imputation_mean_score(self.X_test_org, self.X_test, self.V_test),
     'std': imputation_std_score(self.X_test_org, self.X_test, self.V_test)}

  @property
  def scores_spearman(self):
    return \
    self.get_correlation_marker_pairs(
        data_type='V', score_type='spearman', is_original=False, test=False), \
    self.get_correlation_marker_pairs(
        data_type='V', score_type='spearman', is_original=False, test=True)

  @property
  def scores_pearson(self):
    return \
    self.get_correlation_marker_pairs(
        data_type='V', score_type='pearson', is_original=False, test=False), \
    self.get_correlation_marker_pairs(
        data_type='V', score_type='pearson', is_original=False, test=True)

  @property
  def scores_clustering(self):
    train = clustering_scores(latent=self.Z_train,
                              labels=self.y_train,
                              n_labels=len(self.labels))
    test = clustering_scores(latent=self.Z_test,
                             labels=self.y_test,
                             n_labels=len(self.labels))
    return train, test

  @property
  def scores_classifier(self):
    train, test = streamline_classifier(self.Z_train,
                                        self.y_train,
                                        self.Z_test,
                                        self.y_test,
                                        train_results=True,
                                        labels_name=self.labels,
                                        show_plot=False)
    return train, test

  # ******************** learning curves and metrics ******************** #
  @property
  def learning_curves(self):
    return {
        'train': self.infer.get_train_loss(),
        'valid': self.infer.get_valid_loss(),
    }

  @property
  def train_history(self):
    return self.infer.history['train']

  @property
  def valid_history(self):
    return self.infer.history['valid']

  def plot_learning_curves(self, ax=None):
    ax = to_axis2D(ax)
    losses = self.learning_curves
    train = losses['train']
    valid = losses['valid']
    line_styles = dict(linewidth=1.8)
    point_styles = dict(alpha=0.6, s=80, linewidths=0)

    ax.plot(train, label='train', color='blue', linestyle='-', **line_styles)
    ax.scatter(np.argmin(train), np.min(train), c='blue', **point_styles)

    ax.plot(valid, label='valid', color='orange', linestyle='--', **line_styles)
    ax.scatter(np.argmin(valid), np.min(valid), c='orange', **point_styles)

    ax.set_ylabel("Loss")
    ax.set_xlabel("#Epoch")

    ax.legend()
    ax.set_title(self.short_id, fontsize=10)
    self.add_figure('learning_curves', ax.get_figure())
    return self

  def plot_metrics(self, name, ax=None):
    """ name : list of metrics name for plotting

    NOTE
    ----
    The given name only need to be contained within the metric name
    """
    ax = to_axis2D(ax)
    line_styles = dict(linewidth=1.8)
    name = as_tuple(name, t=string_types)
    name = [i.strip().lower() for i in name]
    found_any_metric = False

    for n, series in self.train_history.items():
      if any(i in n for i in name):
        ax.plot(series, label='[train]%s' % n, linestyle='-', **line_styles)
        found_any_metric = True
    for n, series in self.valid_history.items():
      if any(i in n for i in name):
        ax.plot(series, label='[valid]%s' % n, linestyle='--', **line_styles)

    if found_any_metric:
      ax.set_ylabel("Metrics: %s" % ', '.join(name))
      ax.set_xlabel("#Epoch")
      ax.legend()
      ax.set_title(self.short_id, fontsize=10)
      self.add_figure('metrics', ax.get_figure())
    return self

  # ******************** Z ******************** #
  @property
  def Z_train_org(self):
    return self.get_cached_output(self.X_train_org, 'z')

  @property
  def Z_train(self):
    return self.get_cached_output(self.X_train, 'z')

  @property
  def Z_test_org(self):
    return self.get_cached_output(self.X_test_org, 'z')

  @property
  def Z_test(self):
    return self.get_cached_output(self.X_test, 'z')

  # ******************** y ******************** #
  @property
  def y_pred_train(self):
    return self.get_cached_output(self.X_train, 'y', self.y_train)

  @property
  def y_pred_test(self):
    return self.get_cached_output(self.X_test, 'y', self.y_test)

  # ******************** W ******************** #
  @property
  def W_train_org(self):
    return self.get_cached_output(self.X_train_org, 'w')

  @property
  def W_train(self):
    return self.get_cached_output(self.X_train, 'w')

  @property
  def W_test_org(self):
    return self.get_cached_output(self.X_test_org, 'w')

  @property
  def W_test(self):
    return self.get_cached_output(self.X_test, 'w')

  @property
  def Wstddev_train_org(self):
    return self.get_cached_output(self.X_train_org, 'wstddev')

  @property
  def Wstddev_train(self):
    return self.get_cached_output(self.X_train, 'wstddev')

  @property
  def Wstddev_test_org(self):
    return self.get_cached_output(self.X_test_org, 'wstddev')

  @property
  def Wstddev_test(self):
    return self.get_cached_output(self.X_test, 'wstddev')

  # ******************** V ******************** #
  @property
  def V_train_org(self):
    return self.get_cached_output(self.X_train_org, 'v')

  @property
  def V_train(self):
    return self.get_cached_output(self.X_train, 'v')

  @property
  def V_test_org(self):
    return self.get_cached_output(self.X_test_org, 'v')

  @property
  def V_test(self):
    return self.get_cached_output(self.X_test, 'v')

  @property
  def Vstddev_train_org(self):
    return self.get_cached_output(self.X_train_org, 'vstddev')

  @property
  def Vstddev_train(self):
    return self.get_cached_output(self.X_train, 'vstddev')

  @property
  def Vstddev_test_org(self):
    return self.get_cached_output(self.X_test_org, 'vstddev')

  @property
  def Vstddev_test(self):
    return self.get_cached_output(self.X_test, 'vstddev')

  # ******************** L ******************** #
  @property
  def L_train_org(self):
    return self.get_cached_output(self.X_train_org, 'l')

  @property
  def L_train(self):
    return self.get_cached_output(self.X_train, 'l')

  @property
  def L_test_org(self):
    return self.get_cached_output(self.X_test_org, 'l')

  @property
  def L_test(self):
    return self.get_cached_output(self.X_test, 'l')

  @property
  def Lstddev_train_org(self):
    return self.get_cached_output(self.X_train_org, 'lstddev')

  @property
  def Lstddev_train(self):
    return self.get_cached_output(self.X_train, 'lstddev')

  @property
  def Lstddev_test_org(self):
    return self.get_cached_output(self.X_test_org, 'lstddev')

  @property
  def Lstddev_test(self):
    return self.get_cached_output(self.X_test, 'lstddev')

  # ******************** L ******************** #
  @property
  def PI_train_org(self):
    return self.get_cached_output(self.X_train_org, 'pi')

  @property
  def PI_train(self):
    return self.get_cached_output(self.X_train, 'pi')

  @property
  def PI_test_org(self):
    return self.get_cached_output(self.X_test_org, 'pi')

  @property
  def PI_test(self):
    return self.get_cached_output(self.X_test, 'pi')

  # ******************** X, y ******************** #
  def preprocessX(self, X):
    # X_norm, T_norm, L, L_mean, L_var, y_norm
    outputs = self.infer._preprocess_inputs(X=X, y=None)
    return outputs[0]

  def preprocessY(self, y):
    # X_norm, T_norm, L, L_mean, L_var, y_norm
    X = np.ones(shape=(y.shape[0], self.gene_dim))
    outputs = self.infer._preprocess_inputs(X=X, y=y)
    return outputs[-1]

  @property
  def X_train_org(self):
    return self._X_train_org

  @property
  def X_train(self):
    return self._X_train

  @property
  def X_test_org(self):
    return self._X_test_org

  @property
  def X_test(self):
    return self._X_test

  @property
  def y_train(self):
    return self._y_train

  @property
  def y_test(self):
    return self._y_test

  # ******************** Correlation scores ******************** #
  @property
  def original_spearman(self):
    """ Correlation score between original gene expression and protein marker """
    return \
    self.get_correlation_marker_pairs(score_type='spearman', test=False, is_original=True), \
    self.get_correlation_marker_pairs(score_type='spearman', test=True, is_original=True)

  @property
  def imputed_spearman(self):
    """ Correlation score between imputed gene expression and protein marker """
    return \
    self.get_correlation_marker_pairs(score_type='spearman', test=False, is_original=False), \
    self.get_correlation_marker_pairs(score_type='spearman', test=True, is_original=False)

  @property
  def original_pearson(self):
    """ Correlation score between original gene expression and protein marker """
    return \
    self.get_correlation_marker_pairs(score_type='pearson', test=False, is_original=True), \
    self.get_correlation_marker_pairs(score_type='pearson', test=True, is_original=True)

  @property
  def imputed_pearson(self):
    """ Correlation score between imputed gene expression and protein marker """
    return \
    self.get_correlation_marker_pairs(score_type='pearson', test=False, is_original=False), \
    self.get_correlation_marker_pairs(score_type='pearson', test=True, is_original=False)

  def get_correlation_marker_pairs(self,
                                   data_type='X',
                                   score_type='spearman',
                                   test=True,
                                   is_original=False):
    """
    Parameters
    ----------
    data_type : {'X', 'V', 'W'}
      'X' for input gene expression, 'V' for imputed gene expression,
      and 'W' for reconstructed gene expression

    score_type : {'spearman', 'pearson'}
      spearman correlation for rank (monotonic) relationship, and pearson
      for linear correlation

    test : bool
      if True, calculate the score on test set

    is_original : bool
      in case `data_type='X'`, if False, using the artificial corrupted
      data during training for calculating the correlation

    Return
    ------
    correlation : OrderedDict
      mapping from marker protein/gene name (string) to
      correlation score (scalar)
    """
    assert score_type in ('spearman', 'pearson')
    assert data_type in ('X', 'V', 'W')
    y = self.y_test if test is True else self.y_train
    X = getattr(
        self, '%s_%s%s' %
        (data_type, 'test' if test else 'train', '_org' if is_original else ''))
    corr = get_correlation_scores(X=X,
                                  y=y,
                                  gene_name=self.gene_name,
                                  protein_name=self.labels,
                                  return_series=False)
    score_idx = 0 if score_type == 'spearman' else 1
    return OrderedDict([(i, j[score_idx]) for i, j in corr.items()])

  @cache_memory
  def get_correlation_all_pairs(self,
                                data_type='X',
                                test=True,
                                is_original=False):
    """
    Parameters
    ----------
    data_type : {'X', 'V', 'W'}
      'X' for input gene expression, 'V' for imputed gene expression,
      and 'W' for reconstructed gene expression

    test : bool
      if True, calculate the score on test set

    is_original : bool
      in case `data_type='X'`, if False, using the artificial corrupted
      data during training for calculating the correlation

    Return
    ------
    correlation : tuple of four scalars
      list of tuple contained 4 scalars
      (gene-idx, protein-idx, pearson, spearman)
    """
    assert data_type in ('X', 'V', 'W')

    from scipy.stats import pearsonr, spearmanr
    if test:
      v, x, w, y = self.V_test, self.X_test_org, self.W_test, self.y_test
    else:
      v, x, w, y = self.V_train, self.X_train_org, self.W_train, self.y_train
    n_proteins = y.shape[1]
    n_genes = x.shape[1]

    if data_type == 'X':
      data = x
    elif data_type == 'V':
      data = v
    elif data_type == 'W':
      data = w

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
    from odin.utils.mpi import MPI
    mpi = MPI(jobs, func=_corr, ncpu=3, batch=len(jobs) // 3)
    all_correlations = sorted([i for i in mpi],
                              key=lambda scores:
                              (scores[-2] + scores[-1]) / 2)[::-1]
    return all_correlations

  # ******************** Protein analysis ******************** #
  def plot_protein_predicted_series(self,
                                    test=True,
                                    fontsize=10,
                                    y_true_new=None,
                                    y_pred_new=None,
                                    labels_new=None):
    from odin.backend import log_norm

    if not self.is_semi_supervised:
      return self

    if test:
      y_pred, y_true = self.y_pred_test, self.y_test
    else:
      y_pred, y_true = self.y_pred_train, self.y_train
    if not self.is_binary_classes:
      y_true = log_norm(y_true, axis=1)
    labels = self.protein_name if labels_new is None else labels_new
    # ====== override provided values ====== #
    if y_true_new is not None:
      y_true = y_true_new
    if y_pred_new is not None:
      y_pred = y_pred_new

    n_protein = len(labels)
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
      for cidx, (name, pred, true) in enumerate(zip(labels, y_pred.T,
                                                    y_true.T)):
        assert pred.shape == true.shape
        ids = np.argsort(true)

        if self.ynorm == 'prob':
          pass
        else:
          pass

        ax = plt.subplot(n_protein, 1, cidx + 1)

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

    plt.suptitle('[%s]%s' % ('Test' if test else 'Train', self.short_id),
                 fontsize=fontsize)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    self.add_figure('protein_series_%s' % ('test' if test else 'train'), fig)
    return self

  def plot_protein_scatter(self,
                           test=True,
                           pca=False,
                           protein_name='CD4',
                           fig=None,
                           y_true_new=None,
                           y_pred_new=None,
                           labels_new=None):
    if not self.is_semi_supervised:
      return self

    fn_dim = fast_pca if pca else fast_tsne
    data_type = 'test' if test else 'train'
    protein_name = standardize_protein_name(protein_name).strip().lower()
    fig_name = 'protein_scatter_%s_%s' % (data_type, protein_name)

    labels = self.protein_name if labels_new is None else labels_new
    labels = [i.strip().lower() for i in labels]

    if y_true_new is None:
      y_true = self.y_test if test else self.y_train
    else:
      y_true = y_true_new
    if y_pred_new is None:
      y_pred = self.y_pred_test if test else self.y_pred_train
    else:
      y_pred = y_pred_new

    if protein_name in labels:
      idx = [i for i, j in enumerate(labels) if protein_name in j][0]
      y_true = y_true[:, idx]
      y_pred = y_pred[:, idx]

      X = self.X_test_org if test else self.X_train_org
      Z = self.Z_test if test else self.Z_train
      V = self.V_test if test else self.V_train

      if fig is None:
        fig = plot_figure(nrow=13, ncol=10)
      assert isinstance(fig, plt.Figure), \
      "fig must be instance of matplotlib.Figure"

      x = fn_dim(Z)
      ax = plot_scatter_heatmap(x,
                                val=y_true,
                                ax=321,
                                grid=False,
                                colorbar=True)
      ax.set_xlabel('Latent/ProteinOriginal')
      ax = plot_scatter_heatmap(x,
                                val=y_pred,
                                ax=322,
                                grid=False,
                                colorbar=True)
      ax.set_xlabel('Latent/ProteinPredicted')

      x = fn_dim(V)
      ax = plot_scatter_heatmap(x,
                                val=y_true,
                                ax=323,
                                grid=False,
                                colorbar=True)
      ax.set_xlabel('InputImputed/ProteinOriginal')
      ax = plot_scatter_heatmap(x,
                                val=y_pred,
                                ax=324,
                                grid=False,
                                colorbar=True)
      ax.set_xlabel('InputImputed/ProteinPredicted')

      x = fn_dim(X)
      ax = plot_scatter_heatmap(x,
                                val=y_true,
                                ax=325,
                                grid=False,
                                colorbar=True)
      ax.set_xlabel('InputOriginal/ProteinOriginal')
      ax = plot_scatter_heatmap(x,
                                val=y_pred,
                                ax=326,
                                grid=False,
                                colorbar=True)
      ax.set_xlabel('InputOriginal/ProteinPredicted')

      fig.suptitle('[%s]%s' % (data_type, protein_name))
      with catch_warnings_ignore(UserWarning):
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
      self.add_figure(fig_name, fig)
    return self

  # ******************** setter ******************** #
  def set_mcmc_samples(self, n):
    self._n_mcmc_samples = int(n)
    return self

  def set_labels(self,
                 y_train=None,
                 y_test=None,
                 y_train_pred=None,
                 y_test_pred=None,
                 labels=None):
    """ This can override the provided protein labels, helpful
    when running the trained model on different dataset for
    cross-dataset validation """
    raise NotImplementedError
    return self

  # ******************** properties ******************** #
  @property
  def figures(self):
    if not hasattr(self, '_figures'):
      self._figures = OrderedDict()
    assert all(isinstance(k, string_types) and
               isinstance(v, plt.Figure)
              for k, v in self._figures.items()), \
    "Invalid stored Figures"
    return self._figures

  @property
  def gene_name(self):
    return self._gene_name

  @property
  def protein_name(self):
    return self._prot_name

  @property
  def labels(self):
    return self.protein_name

  @property
  def id(self):
    return self.infer.id

  @property
  def short_id(self):
    i = self.infer.id.split('_')
    corruption = i[-1]
    if 'bi' == corruption[:2]:
      corruption = 'bi' + corruption[-2:]
    else:
      corruption = 'un' + corruption[-2:]
    if self.is_semi_supervised:
      semi = i[6]
    else:
      semi = 'unsupervised'
    return '_'.join(
        [corruption, '_'.join(i[:2]), '_'.join(i[2:5]).replace('_', ''), semi])

  @property
  def short_id_lines(self):
    """same as short_id but divided into 3 lines"""
    short_id = self.short_id.split('_')
    return '\n'.join(
        ['_'.join(short_id[:2]), '_'.join(short_id[2:-1]), short_id[-1]])

  @property
  def model_name(self):
    return self.infer._name

  @property
  def metric_names(self):
    return list(self.train_history.keys())

  @property
  def is_binary_classes(self):
    return np.all(np.sum(self.y_train, axis=-1) == 1.)

  @property
  def is_semi_supervised(self):
    return self.infer.is_semi_supervised

  @property
  def gene_dim(self):
    return self.infer.gene_dim

  @property
  def prot_dim(self):
    return self.infer.prot_dim

  @property
  def xnorm(self):
    return self.infer.xnorm

  @property
  def ynorm(self):
    return self.infer.ynorm

  @property
  def infer(self):
    return self._infer

  @property
  def training_log(self):
    return self._infer.training_log

  @property
  def n_mcmc_samples(self):
    return int(self._n_mcmc_samples)

  def __str__(self):
    s = 'Inference: %s\n' % str(self._infer.__class__)
    for k, v in sorted(self.infer.configs.items()):
      s += '  %-8s: %s\n' % (k, str(v))
    s += 'Dataset: %s\n' % self._ds.path
    for k, v in sorted(self._ds.items()):
      s += '  %-8s: %s\n' % (k, str(v.shape))
    s += '  is_binary_classes: %s\n' % self.is_binary_classes
    s += '#MCMC samples: %d\n' % self.n_mcmc_samples
    return s
