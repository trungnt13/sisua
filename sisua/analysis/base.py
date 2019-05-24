from __future__ import print_function, division, absolute_import
import os
import time
import pickle
import inspect
from six import string_types
from collections import defaultdict, OrderedDict

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from odin.fuel import Dataset
from odin.backend import log_norm
from odin.ml import fast_pca, fast_tsne
from odin.utils import (md5_checksum, as_tuple, flatten_list, catch_warnings_ignore,
                        cache_memory, ctext)
from odin.visual import (plot_save, plot_figure, to_axis2D, plot_aspect,
                         plot_scatter_heatmap, plot_scatter,
                         plot_confusion_matrix, plot_frame)

from sisua.data.path import EXP_DIR
from sisua.inference import Inference
from sisua.utils import filtering_experiment_path
from sisua.utils.visualization import save_figures
from sisua.data.utils import standardize_protein_name
from sisua.data import (get_dataset, apply_artificial_corruption)
from sisua.analysis.imputation_benchmarks import (
    get_correlation_scores,
    imputation_score, imputation_mean_score, imputation_std_score
)
from sisua.analysis.latent_benchmarks import (
    plot_distance_heatmap, plot_latents_binary, plot_latents_multiclasses,
    streamline_classifier, clustering_scores)

# ===========================================================================
# Helper
# ===========================================================================
_LOADED_DATASET = {}

def _fast_load_dataset(name):
  if name not in _LOADED_DATASET:
    ds = get_dataset(dataset_name=name)
    _LOADED_DATASET[name] = ds
  return _LOADED_DATASET[name]

def get_all_posteriors(path_or_dataset, incl=[], excl=[], fn_filter=None,
                       show_progress=True):
  if os.path.exists(path_or_dataset):
    exp_path = os.path.dirname(path_or_dataset)
    ds_name = os.path.basename(path_or_dataset)
  else:
    exp_path = ''
    ds_name = path_or_dataset
  all_exp = filtering_experiment_path(
      ds_name=ds_name,
      exp_path=exp_path,
      incl_keywords=incl, excl_keywords=excl,
      fn_filter=fn_filter,
      return_dataset=False,
      print_log=False)
  all_posteriors = []
  if show_progress:
    print("Loading %s posteriors for dataset '%s' at path '%s'..." %
      (ctext(len(all_exp), 'lightyellow'),
       ctext(ds_name, 'lightyellow'),
       ctext(exp_path, 'lightyellow')))
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
class Posterior(object):
  """ Posterior """

  def __init__(self, path_or_infer,
               ds_name=None, ds=None,
               n_mcmc_samples=100, verbose=False):
    super(Posterior, self).__init__()
    self.verbose = bool(verbose)
    # ====== load inference class from path ====== #
    if isinstance(path_or_infer, string_types):
      if not os.path.exists(path_or_infer):
        path_or_infer = os.path.join(EXP_DIR, path_or_infer)
      if os.path.isfile(path_or_infer):
        model_path = path_or_infer
      else:
        model_path = os.path.join(path_or_infer, 'model.pkl')
      with open(model_path, 'rb') as f:
        infer = pickle.load(f)
    elif isinstance(path_or_infer, Inference):
      infer = path_or_infer
    else:
      raise ValueError(
          "No support for `path_or_infer` type: %s" % str(type(path_or_infer)))
    # ====== validate infer ====== #
    assert isinstance(infer, Inference), \
    "`infer` must be instance of sisua.inference.Inference, " +\
    "but given: %s" % str(type(infer))
    self._infer = infer
    assert self._infer.is_fitted, \
    "Only fitted model can create a Posterior"
    # ====== dataset name ====== #
    if ds_name is None and ds is None:
      ds_name = self._infer.configs.get('dataset', None)
    if isinstance(ds_name, string_types):
      self._ds, gene_ds, prot_ds = _fast_load_dataset(ds_name)
      X_train_org = gene_ds.get_data(data_type='train')
      X_test_org = gene_ds.get_data(data_type='test')
      y_train = prot_ds.get_data(data_type='train')
      y_test = prot_ds.get_data(data_type='test')
      gene_name = gene_ds.col_name
      prot_name = prot_ds.col_name
    elif isinstance(ds, (Dataset, dict)):
      self._ds = ds
      X = ds.get('X', None)
      X_train_org = ds.get('X_train', None)
      X_test_org = ds.get('X_test', None)

      y = ds.get('y', None)
      y_train = ds.get('y_train', None)
      y_test = ds.get('y_test', None)

      if X_train_org is None or X_test_org is None:
        if X is None:
          raise ValueError(
              "X_train, X_test must be provide if full data X isn't given")
        X_train_org = X[:]
        X_test_org = X_train_org
        if y is not None:
          y_train = y
          y_test = y

      gene_name = np.array([str(i) for i in ds['X_col']])\
      if 'X_col' in ds else \
      ['Gene#%d' % i for i in range(X_train_org.shape[1])]
      if y_train is not None:
        prot_name = np.array([str(i) for i in ds['y_col']])\
        if 'y_col' in ds else \
        ['Prot#%d' % i for i in range(y_train.shape[1])]
      else:
        prot_name = None
    else:
      raise ValueError("ds_name must be string_types, " +
        "ds must be instance of odin.fuel.Dataset or dictionary")

    X_train = apply_artificial_corruption(X_train_org,
        dropout=self.infer.corruption_rate,
        distribution=self.infer.corruption_dist)
    X_test = apply_artificial_corruption(X_test_org,
        dropout=self.infer.corruption_rate,
        distribution=self.infer.corruption_dist)
    # validate the dimension
    assert (self.infer.gene_dim ==
            X_train.shape[1] == X_test.shape[1] ==
            len(gene_name)),\
    "Number of gene expression between trained inference and " + \
    "given dataset mismatch"

    # everything must be numpy array not memmap
    self._X_train = np.asarray(X_train)
    self._X_train_org = np.asarray(X_train_org)
    self._X_test = np.asarray(X_test)
    self._X_test_org = np.asarray(X_test_org)
    self._y_train = np.asarray(y_train)
    self._y_test = np.asarray(y_test)

    self._gene_name = gene_name
    self._prot_name = [standardize_protein_name(i)
                       for i in prot_name]
    # ====== MCMC ====== #
    self._n_mcmc_samples = int(n_mcmc_samples)

  def new_figure(self, nrow=8, ncol=8, name=None):
    fig = plot_figure(nrow=nrow, ncol=ncol)
    if name is not None:
      self.figures[str(name)] = fig
    return self

  def add_figure(self, name, fig):
    for k, v in self.figures.items():
      if v == fig:
        return
    self.figures[name] = fig
    return self

  def save_plots(self, path, dpi=None, separate_files=True):
    save_figures(self.figures, path, dpi, separate_files,
                clear_figures=True)
    return self

  def save_scores(self, path):
    assert '.html' in path.lower(), "Only save scores to html file"
    text = ''
    for name, (train, test) in (('scores', self.scores),
                                ('pearson', self.scores_pearson),
                                ('spearman', self.scores_spearman),
                                ('classifier', self.scores_classifier),
                                ('cluster', self.scores_clustering),
                                ('imputation', self.scores_imputation),):
      text += '<h4>Score type: "%s"<h4>' % name
      df = pd.DataFrame(data=[train, test], index=['train', 'test'])
      text += df.to_html(float_format='%.3f') + '\n'
    with open(path, 'w') as f:
      f.write(text)
    return self

  # ******************** Latent space analysis ******************** #
  def plot_latents_heatmap(self, test=True, legend=True, ax=None):
    ax = to_axis2D(ax)
    if test:
      z, y = self.Z_test, self.y_test
    else:
      z, y = self.Z_train, self.y_train
    title = ("[test]%s" if test else "[train]%s") % self.short_id

    if self.is_binary_classes:
      plot_distance_heatmap(
          z, labels=y, labels_name=self.labels,
          legend_enable=bool(legend),
          ax=ax, fontsize=8, legend_ncol=2,
          title=title)
    else:
      ax = plot_latents_multiclasses(
          Z=z, y=y, labels_name=self.labels,
          title=title, use_PCA=False,
          show_colorbar=bool(legend))

    self.add_figure('latents_heatmap_%s' % ('test' if test else 'train'),
                    ax.get_figure())
    return self

  def plot_latents_scatter(self, test=True, legend=True, pca=False,
                           size=8, ax=None):
    """
    test : if True, plotting latent space of test set, otherwise, use training set
    """
    ax = to_axis2D(ax)
    if test:
      z, y = self.Z_test, self.y_test
    else:
      z, y = self.Z_train, self.y_train
    title = ("[test]%s" if test else "[train]%s") % self.short_id_lines
    plot_latents_binary(
        Z=z, y=y, title=title,
        show_legend=bool(legend), size=8, fontsize=8,
        ax=ax, labels_name=self.labels, use_PCA=pca,
        enable_argmax=True, enable_separated=False)
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

    (train, test), (fig_train, fig_test) = streamline_classifier(
        self.Z_train, self.y_train, self.Z_test, self.y_test,
        train_results=True,
        labels_name=self.labels,
        show_plot=True, return_figure=True)
    if self.X_train.shape == self.X_test.shape and \
    np.allclose(self.X_train, self.X_test):
      self.add_figure('streamline_f1_%s' % 'test', fig_test)
    else:
      self.add_figure('streamline_f1_%s' % 'train', fig_train)
      self.add_figure('streamline_f1_%s' % 'test', fig_test)
    return self

  # ******************** Imputation analysis ******************** #
  def plot_cellsize_series(self, test=True, ax=None, fontsize=10):
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
    ax.plot(mean[sorted_ids], linestyle='--', alpha=0.66, linewidth=1,
            label='Prediction')

    ax.set_title('[%s]%s' % ('Test' if test else 'Train', self.short_id),
                 fontsize=fontsize)
    ax.set_ylabel('Cell Size')
    ax.set_xlabel('Cell in sorted order of size')
    ax.legend()

    self.add_figure('cellsize_%s' % ('test' if test else 'train'),
                    ax.get_figure())
    return self

  def plot_correlation_series(self, test=True, fontsize=10):
    from scipy.stats import pearsonr, spearmanr
    if test:
      v, x, y = self.V_test, self.X_test_org, self.y_test
    else:
      v, x, y = self.V_train, self.X_train_org, self.y_train
    original_series = get_correlation_scores(
        X=x, y=y, gene_name=self.gene_name, protein_name=self.labels,
        return_series=True)
    imputed_series = get_correlation_scores(
        X=v, y=y, gene_name=self.gene_name, protein_name=self.labels,
        return_series=True)
    assert len(original_series) == len(imputed_series)
    n_pair = len(imputed_series)

    # for prot_gene in original_series.keys():
    #   prot_name, gene_name = prot_gene.split('/')
    #   prot_name = '[Protein]%s' % prot_name
    #   gene_name = '[Gene]%s' % gene_name

    #   org, prot = original_series[prot_gene]
    #   imp, _ = imputed_series[prot_gene]
    #   assert np.all(prot == _)

    #   figures = []
    #   for name, gene in zip(("Original", "Imputed"), (org, imp)):
    #     fig = sns.jointplot(x=prot_name, y=gene_name,
    #                   data=pd.DataFrame({prot_name: prot, gene_name: gene}),
    #                   kind="scatter",
    #                   joint_kws=dict(s=25, alpha=0.6, linewidths=0),
    #                   marginal_kws=dict(bins=18, rug=True)
    #     )
    #     fig = fig.fig
    #     fig.get_axes()[1].set_title(
    #         '[%s]Pearson:%.2f Spearman:%.2f' %
    #         (name, pearsonr(gene, prot)[0], spearmanr(gene, prot).correlation,
    #     ), fontsize=fontsize)
    #     figures.append(fig)

    #   from odin import visual as V
    #   data = np.concatenate([V.fig2data(fig) for fig in figures], axis=0)
    #   fig = V.data2fig(data)
    #   V.plot_save()
    #   exit()
    # exit()

    fig = plt.figure(figsize=(15, 5 * n_pair),
                     constrained_layout=True)
    width = 4
    grids = fig.add_gridspec(n_pair, 2 * width)

    for idx, prot_gene in enumerate(sorted(imputed_series.keys())):
      prot_name, gene_name = prot_gene.split('/')
      imputed_gene, prot1 = imputed_series[prot_gene]
      original_gene, prot2 = original_series[prot_gene]
      assert np.all(prot1 == prot2)
      y = prot1

      for j, (name, series) in enumerate((("Original", original_gene),
                                          ("Imputed", imputed_gene))):
        ax = fig.add_subplot(
            grids[idx, width * j: (width * j + width - 1)])
        ax.scatter(y, series,
                   s=25, alpha=0.6, linewidths=0)
        plot_aspect('auto', 'box', ax)
        ax.set_title('[%s]Pearson:%.2f Spearman:%.2f' % (
            name,
            pearsonr(series, y)[0],
            spearmanr(series, y).correlation,
        ), fontsize=fontsize)
        ax.set_xlabel('[Protein] %s' % prot_name, fontsize=fontsize)
        ax.set_ylabel('[Gene] %s' % gene_name, fontsize=fontsize)

        ax = fig.add_subplot(
            grids[idx, (width * j + width - 1): (width * j + width)])
        ax.boxplot(series)
        ax.set_xticks(())
        ax.set_xlabel(name, fontsize=fontsize)

    data_type = ('test' if test else 'train')
    plt.suptitle('[%s]%s' % (data_type, self.short_id),
                 fontsize=fontsize + 2)
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
    n_mcmc = (self.n_mcmc_samples if n_mcmc_samples is None else
              int(n_mcmc_samples))
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
    self.get_correlation(
        data_type='V', score_type='spearman', is_original=False, test=False), \
    self.get_correlation(
        data_type='V', score_type='spearman', is_original=False, test=True)

  @property
  def scores_pearson(self):
    return \
    self.get_correlation(
        data_type='V', score_type='pearson', is_original=False, test=False), \
    self.get_correlation(
        data_type='V', score_type='pearson', is_original=False, test=True)

  @property
  def scores_clustering(self):
    train = clustering_scores(latent=self.Z_train, labels=self.y_train,
                              n_labels=len(self.labels))
    test = clustering_scores(latent=self.Z_test, labels=self.y_test,
                             n_labels=len(self.labels))
    return train, test

  @property
  def scores_classifier(self):
    train, test = streamline_classifier(
        self.Z_train, self.y_train, self.Z_test, self.y_test,
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

    ax.plot(train, label='train', color='blue',
            linestyle='-', **line_styles)
    ax.scatter(np.argmin(train), np.min(train),
               c='blue', **point_styles)

    ax.plot(valid, label='valid', color='orange',
            linestyle='--', **line_styles)
    ax.scatter(np.argmin(valid), np.min(valid),
               c='orange', **point_styles)

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
        ax.plot(series, label='[train]%s' % n,
                linestyle='-', **line_styles)
        found_any_metric = True
    for n, series in self.valid_history.items():
      if any(i in n for i in name):
        ax.plot(series, label='[valid]%s' % n,
                linestyle='--', **line_styles)

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
    self.get_correlation(score_type='spearman', test=False, is_original=True), \
    self.get_correlation(score_type='spearman', test=True, is_original=True)

  @property
  def imputed_spearman(self):
    """ Correlation score between imputed gene expression and protein marker """
    return \
    self.get_correlation(score_type='spearman', test=False, is_original=False), \
    self.get_correlation(score_type='spearman', test=True, is_original=False)

  @property
  def original_pearson(self):
    """ Correlation score between original gene expression and protein marker """
    return \
    self.get_correlation(score_type='pearson', test=False, is_original=True), \
    self.get_correlation(score_type='pearson', test=True, is_original=True)

  @property
  def imputed_pearson(self):
    """ Correlation score between imputed gene expression and protein marker """
    return \
    self.get_correlation(score_type='pearson', test=False, is_original=False), \
    self.get_correlation(score_type='pearson', test=True, is_original=False)

  def get_correlation(self, data_type='X', score_type='spearman',
                      test=True, is_original=False):
    assert score_type in ('spearman', 'pearson')
    y = self.y_test if test is True else self.y_train
    X = getattr(self, '%s_%s%s' % (data_type,
                                  'test' if test else 'train',
                                  '_org' if is_original else ''))
    corr = get_correlation_scores(
        X=X, y=y,
        gene_name=self.gene_name, protein_name=self.labels,
        return_series=False)
    score_idx = 0 if score_type == 'spearman' else 1
    return OrderedDict([(i, j[score_idx]) for i, j in corr.items()])

  # ******************** Protein analysis ******************** #
  def plot_protein_series(self, test=True, fontsize=10):
    if not self.is_semi_supervised:
      return self

    if test:
      y_pred, y_true = self.y_pred_test, self.y_test
    else:
      y_pred, y_true = self.y_pred_train, self.y_train
    if not self.is_binary_classes:
      y_true = log_norm(y_true, axis=1)
    # y_true = self.preprocessY(y_true)

    n_protein = len(self.protein_name)
    colors = sns.color_palette(n_colors=2)

    #######
    if self.is_binary_classes:
      fig = plt.figure(figsize=(12, 12))
      from sklearn.metrics import confusion_matrix
      y_true = np.argmax(y_true, axis=-1)
      y_pred = np.argmax(y_pred, axis=-1)
      plot_confusion_matrix(cm=confusion_matrix(y_true, y_pred),
        labels=self.protein_name, colorbar=True,
        fontsize=fontsize)
    #######
    else:
      fig = plt.figure(figsize=(12, n_protein * 4))
      for cidx, (name, pred, true) in enumerate(
          zip(self.protein_name, y_pred.T, y_true.T)):
        assert pred.shape == true.shape
        ids = np.argsort(true)

        if self.ynorm == 'prob':
          pass
        else:
          pass

        ax = plt.subplot(n_protein, 1, cidx + 1)

        ax.plot(true[ids], linewidth=2, color=colors[0],
                label="[True]%s" % name)
        ax.plot(true[ids][0],
                linestyle='--', alpha=0.88, linewidth=1.2,
                color=colors[1], label="[Pred]%s" % name)
        ax.set_ylabel("Log-normalized true protein level", color=colors[0])
        ax.set_xlabel("Cell in sorted order of protein level")
        ax.tick_params(axis='y', colors=colors[0], labelcolor=colors[0])
        ax.set_title(name, fontsize=fontsize)
        ax.legend()

        ax = ax.twinx()
        ax.plot(pred[ids], linestyle='--', alpha=0.88, linewidth=1.2,
                color=colors[1], label="[Pred]%s" % name)
        ax.set_ylabel("Predicted protein response", color=colors[1])
        ax.tick_params(axis='y', colors=colors[1], labelcolor=colors[1])

    plt.suptitle('[%s]%s' % ('Test' if test else 'Train', self.short_id),
                 fontsize=fontsize)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    self.add_figure('protein_series_%s' % ('test' if test else 'train'), fig)
    return self

  def plot_protein_scatter(self, test=True, protein_name='CD4'):
    if not self.is_semi_supervised:
      return self
    data_type = 'test' if test else 'train'
    protein_name = standardize_protein_name(protein_name)
    fig_name = 'protein_scatter_%s_%s' % (data_type, protein_name)

    if self.is_binary_classes:
      protein_name = ''
    elif protein_name in self.labels:
      idx = [i for i, j in enumerate(self.protein_name)
             if protein_name.strip().lower() in j.strip().lower()][0]

      y_true = self.y_test if test else self.y_train
      y_true = y_true[:, idx]

      y_pred = self.y_pred_test if test else self.y_pred_train
      y_pred = y_pred[:, idx]

      Z = self.Z_test if test else self.Z_train
      V = self.V_test if test else self.V_train

      fig = plot_figure(10, 10)

      x = fast_tsne(Z)
      ax = plot_scatter_heatmap(x, val=y_true, ax=221, grid=False, colorbar=True)
      ax.set_xlabel('Latent/Original')
      ax = plot_scatter_heatmap(x, val=y_pred, ax=222, grid=False, colorbar=True)
      ax.set_xlabel('Latent/Predicted')

      x = fast_tsne(V)
      ax = plot_scatter_heatmap(x, val=y_true, ax=223, grid=False, colorbar=True)
      ax.set_xlabel('Input/Original')
      ax = plot_scatter_heatmap(x, val=y_pred, ax=224, grid=False, colorbar=True)
      ax.set_xlabel('Input/Predicted')

      fig.suptitle('[%s]%s' % (data_type, protein_name))
      with catch_warnings_ignore(UserWarning):
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
      self.add_figure(fig_name, fig)
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

  def set_mcmc_samples(self, n):
    self._n_mcmc_samples = int(n)
    return self

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
    return '_'.join([corruption,
                     '_'.join(i[:2]),
                     '_'.join(i[2:5]).replace('_', ''),
                     semi])

  @property
  def short_id_lines(self):
    """same as short_id but divided into 3 lines"""
    short_id = self.short_id.split('_')
    return '\n'.join(['_'.join(short_id[:2]), '_'.join(short_id[2:-1]), short_id[-1]])

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
