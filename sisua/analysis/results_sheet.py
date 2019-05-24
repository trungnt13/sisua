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

from odin.ml import fast_pca, fast_tsne
from odin.backend import log_norm
from odin.utils import (md5_checksum, as_tuple, flatten_list, catch_warnings_ignore,
                        cache_memory, ctext)
from odin.visual import (plot_save, plot_figure, to_axis2D, plot_aspect,
                         plot_scatter_heatmap, plot_scatter,
                         plot_confusion_matrix, plot_frame)

from sisua.data import get_dataset
from sisua.data.path import EXP_DIR
from sisua.inference import Inference
from sisua.label_threshold import GMMThresholding
from sisua.utils import filtering_experiment_path
from sisua.data.utils import standardize_protein_name
from sisua.analysis.imputation_benchmarks import (
    get_correlation_scores,
    imputation_score, imputation_mean_score, imputation_std_score
)
from sisua.analysis.latent_benchmarks import (
    plot_distance_heatmap, plot_latents_binary, plot_latents_multiclasses,
    clustering_scores)
from sisua.analysis.base import Posterior
from sisua.utils.visualization import save_figures

# ===========================================================================
# Helpers
# ===========================================================================
def _get_score_fn(score_type):
  score_type = str(score_type).lower()
  if score_type == 'imputation':
    fn_score = lambda x: x.scores_imputation
  elif score_type == 'classifier':
    fn_score = lambda x: x.scores_classifier
  elif score_type == 'cluster':
    fn_score = lambda x: x.scores_clustering
  elif score_type == 'spearman':
    fn_score = lambda x: x.scores_spearman
  elif score_type == 'pearson':
    fn_score = lambda x: x.scores_pearson
  else:
    raise ValueError("No support for score type: '%s'" % score_type)
  return fn_score

def _extract_metrics(hist):
  hist = dict(hist)
  loss = hist['loss']

  def _get(prefix, suffix, allow_empty=False):
    name = prefix + '_' + suffix
    if name in hist:
      return hist[name]
    elif 'n' + name in hist:
      return [-i for i in hist['n' + name]]
    if allow_empty:
      if prefix in hist:
        return hist[prefix]
      elif 'n' + prefix in hist:
        return [-i for i in hist['n' + prefix]]
    return None

  llkx = _get('llk', 'x', allow_empty=True)
  llky = _get('llk', 'y')
  klqpx = _get('klqp', 'x', allow_empty=True)
  klqpy = _get('klqp', 'y')
  return loss, llkx, llky, klqpx, klqpy

def _normalize_metrics(metrics):
  output = {'LLK_x': np.nan,
            'LLK_y': np.nan,
            'KLqp_x': np.nan,
            'KLqp_y': np.nan}
  #
  for name in ('llk', 'llk_x', 'nllk', 'nllk_x'):
    sign = -1 if 'n' == name[0] else 1
    if name in metrics:
      output['LLK_x'] = sign * metrics[name]
  #
  for name in ('klqp', 'klqp_x'):
    if name in metrics:
      output['KLqp_x'] = metrics[name]
  #
  for name in ('llk_y', 'nllk_y'):
    sign = -1 if 'n' == name[0] else 1
    if name in metrics:
      output['LLK_y'] = sign * metrics[name]
  #
  output['KLqp_y'] = metrics.get('klqp_y', np.nan)
  return output

# ===========================================================================
# Main
# ===========================================================================
class ResultsSheet(object):
  """ ResultsSheet """
  ALL_SCORE_TYPES = ['imputation', 'classifier', 'cluster',
                     'spearman', 'pearson']

  def __init__(self, *posteriors, verbose=True):
    super(ResultsSheet, self).__init__()
    posteriors = flatten_list(posteriors)
    assert len(posteriors) > 0, "No posteriors for analysis!"
    assert all(isinstance(i, Posterior) for i in posteriors)
    self._posteriors = posteriors
    self.verbose = bool(verbose)

  @property
  def posteriors(self):
    return self._posteriors

  def _log(self, msg):
    if self.verbose:
      print(msg)
    return self

  def save_scores(self, outpath,
                  scores=['imputation', 'classifier', 'cluster',
                          'spearman', 'pearson']):
    """ Saving score to html table format """
    ext = os.path.splitext(outpath)[-1].lower()
    if ext == '':
      outpath = outpath + '.html'
    elif ext != '.html':
      raise ValueError(
          "Only support excel with extension: html; but given '%s'" % ext)

    scores = as_tuple(scores, t=string_types)
    assert all(i in ResultsSheet.ALL_SCORE_TYPES for i in scores), \
    "Support score type %s but given %s" % (
        str(ResultsSheet.ALL_SCORE_TYPES), str(scores))
    fn_score = [_get_score_fn(s) for s in scores]

    index = []
    data = defaultdict(list)
    # store the original scores of data for
    # spearman and pearson correlation
    original = {}
    corrupted = {}

    for pos in self.posteriors:
      posid = pos.short_id.split('_')
      systype = posid[-1]
      dsname = '_'.join(posid[:2])
      posid = '_'.join(posid[2:-1])
      index.append(posid)

      pos_time = time.time()
      self._log("Calculating score for '%s' ..." % ctext(pos.short_id, 'yellow'))

      for name, fn in zip(scores, fn_score):
        start_time = time.time()

        i, j = fn(pos)
        trn = {'0_train': dsname, '1_type': systype}
        tst = {'0_test': dsname, '1_type': systype}
        trn.update(i)
        tst.update(j)
        data[name + 'train'].append(trn)
        data[name + 'test'].append(tst)

        if name in ('spearman', 'pearson'):
          trn_org, tst_org = getattr(pos, 'original_%s' % name)
          trn_org['0_train'] = dsname; tst_org['0_test'] = dsname
          trn_org['1_type'] = ''; tst_org['1_type'] = ''
          original[name + 'train'] = trn_org
          original[name + 'test'] = tst_org

          trn_crr, tst_crr = getattr(pos, 'imputed_%s' % name)
          trn_crr['0_train'] = dsname; tst_crr['0_test'] = dsname
          trn_crr['1_type'] = ''; tst_crr['1_type'] = ''
          corrupted[name + 'train'] = trn_crr
          corrupted[name + 'test'] = tst_crr

        self._log("  %s: %.2f(s)" %
          (ctext(name, 'cyan'), time.time() - start_time))
      self._log(" Finished in %.2f(s)" % (time.time() - pos_time))

      # special case of the metrics
      trn = {'0_train': dsname, '1_type': systype}
      tst = {'0_test': dsname, '1_type': systype}
      i, j = pos.scores
      trn.update(_normalize_metrics(i))
      tst.update(_normalize_metrics(j))
      data['metricstrain'].append(trn)
      data['metricstest'].append(tst)

    text = ''
    for score_type in ['metrics'] + list(scores):
      idx = list(index)
      train = list(data[score_type + 'train'])
      test = list(data[score_type + 'test'])

      if score_type in ('spearman', 'pearson'):
        train.append(corrupted[score_type + 'train'])
        train.append(original[score_type + 'train'])

        test.append(corrupted[score_type + 'test'])
        test.append(original[score_type + 'test'])

        idx.append('Corrupted')
        idx.append('Original')

      text += '<h4>Score type: "%s"<h4>' % score_type
      df = pd.DataFrame(data=train, index=idx)
      text += df.to_html(float_format='%.3f') + '\n'
      df = pd.DataFrame(data=test, index=idx)
      text += df.to_html(float_format='%.3f')

    with open(outpath, 'w') as f:
      f.write(text)
    return self._log("Saved scores at path '%s'" % ctext(outpath, 'lightyellow'))

  # ******************** comparison series ******************** #
  def plot_comparison_f1(self, test=True, model_id=lambda m: m.name,
                         fig_width=12):
    assert callable(model_id), "model_id must be callable"
    start_time = time.time()
    score_type = 'classifier'
    data_type = 'test' if bool(test) else 'train'

    n_system = len(self)
    fn_score = _get_score_fn(score_type)
    scores_name = None
    scores = []
    for pos in self.posteriors:
      s = fn_score(pos)
      s = s[1] if bool(test) else s[0]
      if score_type == 'classifier':
        del s['F1weight']
        del s['F1micro']
        del s['F1macro']
      n_labels = len(s)
      s = sorted(s.items(), key=lambda x: x[0])
      scores_name = [i[0].replace('F1_', '') for i in s]
      scores.append((model_id(pos.infer), [i[1] for i in s]))

    colors = sns.color_palette(n_colors=n_labels)
    fig, subplots = plt.subplots(nrows=1, ncols=n_system,
                                 sharey=True, squeeze=True,
                                 figsize=(int(fig_width), 2))

    for idx, (name, f1) in enumerate(scores):
      assert len(scores_name) == len(f1)
      f1_weight = np.mean(f1)
      ax = subplots[idx]
      ax.grid(True, axis='both', which='both', linewidth=0.5, alpha=0.6)
      for i, (f, c) in enumerate(zip(f1, colors)):
        ax.scatter(i, f, color=c, s=22, marker='o', alpha=0.8)
        ax.text(i - 0.2, f + 24, '%.1f' % f, fontsize=10,
                rotation=75)

      ax.plot(np.arange(n_labels), f1, linewidth=1.0, linestyle='--')
      ax.plot(np.arange(n_labels), [f1_weight for i in range(n_labels)],
              linewidth=1.2, linestyle=':', color='black',
              label=r"$\overline{F1}$:%.1f" % f1_weight)
      ax.legend(fontsize=14, loc='lower left', handletextpad=0.1, frameon=False)

      ax.set_xticks(np.arange(n_labels))
      ax.set_xlabel(name, fontsize=12)

      ax.set_ylim(-8, 130)
      ax.set_yticks(np.linspace(0, 100, 5))

      ax.xaxis.set_ticklabels([])

      ax.tick_params(axis='x', length=0)
      ax.tick_params(axis='y', length=0, labelsize=8)

      plot_frame(ax, right=False, top=False, left=True if idx == 0 else False)

    plt.tight_layout(w_pad=0)
    self.add_figure("compare_%s_%s" % (score_type, data_type), fig)

    fig = plot_figure(nrow=1, ncol=4)
    for name, c in zip(scores_name, colors):
      plt.plot(0, 0, 'o', label=name, color=c)
    plt.axis('off')
    plt.legend(ncol=int(np.ceil(len(scores_name) / 2)),
          scatterpoints=1, scatteryoffsets=[0.375, 0.5, 0.3125],
          loc='upper center', bbox_to_anchor=(0.5, -0.01),
          handletextpad=0.1, labelspacing=0., columnspacing=0.4)
    self.add_figure(
        "compare_%s_%s_legend" % (score_type, data_type), fig)

    return self._log('plot_comparison_series[%s][%s] %s(s)' %
      (score_type, data_type,
       ctext(time.time() - start_time, 'lightyellow')))
    return self

  # ******************** bar plot ******************** #
  def _bar_box_line(self,
              title, ylabel, get_score, model_id, ax, ignore=[],
              using_bar=True):
    start_time = time.time()
    assert callable(model_id), "model_id must be callable"
    assert callable(get_score)

    data = []
    for pos in self.posteriors:
      name = model_id(pos.infer)
      train, test = get_score(pos)

      for i in ignore:
        del train[i]
        del test[i]

      for i, j in train.items():
        data.append({'Model': name, ylabel: j, 'Data': 'train'})
      for i, j in test.items():
        data.append({'Model': name, ylabel: j, 'Data': 'test'})
    df = pd.DataFrame(data)

    ax = to_axis2D(ax)
    # Bar plot
    if using_bar:
      sns.barplot(x='Model', y=ylabel, hue='Data', data=df, ax=ax)
    # Box plot
    else:
      sns.boxplot(x='Model', y=ylabel, hue='Data', data=df, ax=ax)

    ax.grid(axis='y', linewidth=1.2, alpha=0.5)
    ax.set_axisbelow(True)

    self.add_figure(title, ax.get_figure())
    return self._log('%s %s(s)' %
      (title, ctext(time.time() - start_time, 'lightyellow')))

  # ******************** series plot ******************** #

  # ******************** box plot ******************** #
  def boxplot_cluster(self, model_id=lambda m: m.name, ax=None):
    return self._bar_box_line(title="boxplot_cluster", ylabel="Pooled Cluster Metrics",
                        get_score=lambda x: x.scores_clustering,
                        model_id=model_id, ax=ax, using_bar=False)

  def boxplot_f1(self, model_id=lambda m: m.name, ax=None):
    return self._bar_box_line(title="boxplot_classifier", ylabel="F1 score",
                        get_score=lambda x: x.scores_classifier,
                        model_id=model_id, ax=ax, using_bar=False,
                        ignore=['F1weight', 'F1micro', 'F1macro'])

  def boxplot_pearson(self, model_id=lambda m: m.name, ax=None):
    return self._bar_box_line(title="boxplot_pearson", ylabel="Pearson correlation",
                        get_score=lambda x: x.scores_pearson,
                        model_id=model_id, ax=ax, using_bar=False)

  def boxplot_spearman(self, model_id=lambda m: m.name, ax=None):
    return self._bar_box_line(title="boxplot_spearman", ylabel="Spearman correlation",
                        get_score=lambda x: x.scores_spearman,
                        model_id=model_id, ax=ax, using_bar=False)

  # ******************** bar plot ******************** #
  def barplot_cluster(self, model_id=lambda m: m.name, ax=None):
    return self._bar_box_line(title="barplot_cluster", ylabel="Pooled Cluster Metrics",
                        get_score=lambda x: x.scores_clustering,
                        model_id=model_id, ax=ax, using_bar=False)

  def barplot_f1(self, model_id=lambda m: m.name, ax=None):
    return self._bar_box_line(title="barplot_classifier",
                        ylabel="F1 score",
                        get_score=lambda x: x.scores_classifier,
                        model_id=model_id, ax=ax,
                        ignore=['F1weight', 'F1micro', 'F1macro'])

  def barplot_pearson(self, model_id=lambda m: m.name, ax=None):
    return self._bar_box_line(title="barplot_pearson",
                        ylabel="Pearson correlation",
                        get_score=lambda x: x.scores_pearson,
                        model_id=model_id, ax=ax)

  def barplot_spearman(self, model_id=lambda m: m.name, ax=None):
    return self._bar_box_line(title="barplot_spearman",
                        ylabel="Spearman correlation",
                        get_score=lambda x: x.scores_spearman,
                        model_id=model_id, ax=ax)

  # ******************** series plot ******************** #
  def plot_correlation_series(self, test=True, fontsize=8):
    start_time = time.time()

    from scipy.stats import pearsonr, spearmanr
    n_system = len(self)
    data_type = 'test' if test else 'train'

    # OrderDict(name -> series)
    original_series = None
    imputed_series = []
    for pos in self:
      if test:
        v, x, y = pos.V_test, pos.X_test_org, pos.y_test
      else:
        v, x, y = pos.V_train, pos.X_train_org, pos.y_train
      if original_series is None:
        original_series = get_correlation_scores(
            X=x, y=y, gene_name=pos.gene_name, protein_name=pos.labels,
            return_series=True)
      imputed_series.append(
          get_correlation_scores(
              X=v, y=y, gene_name=pos.gene_name, protein_name=pos.labels,
          return_series=True))

    # ====== plotting ====== #
    n_pair = len(original_series)
    fig = plt.figure(figsize=(20, 5 * n_pair),
                     constrained_layout=True)
    width = 4
    grids = fig.add_gridspec(n_pair, (n_system + 1) * width)

    for row_idx, prot_gene in enumerate(original_series.keys()):
      prot_name, gene_name = prot_gene.split('/')
      original_gene, prot = original_series[prot_gene]

      # gather all series
      gene = [original_gene]
      system_name = ["Original"]
      for s, posetrior in zip(imputed_series, self.posteriors):
        i, j = s[prot_gene]
        assert np.all(prot == j)
        gene.append(i)
        system_name.append(posetrior.short_id_lines)

      # plotting each series
      for col_idx, (name, g) in enumerate(zip(system_name, gene)):
        ax = fig.add_subplot(
            grids[row_idx, width * col_idx: (width * col_idx + width)])
        ax.scatter(prot, g, s=25, alpha=0.6, linewidths=0)
        plot_aspect('auto', 'box', ax)

        title = data_type + ' - ' + prot_gene + ' - %s'if col_idx == 0 else "%s"
        title += '\nPearson:%.2f Spearman:%.2f'
        ax.set_title(title % (name,
                              pearsonr(g, prot)[0],
                              spearmanr(g, prot).correlation),
        fontsize=fontsize + (2 if col_idx == 0 else 0))
        if col_idx == 0:
          ax.set_xlabel('[Protein] %s' % prot_name, fontsize=fontsize)
          ax.set_ylabel('[Gene] %s' % gene_name, fontsize=fontsize)

        if np.mean(g) < 0.1:
          for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(6)
        # ax = fig.add_subplot(
        #     grids[row_idx, (width * col_idx + width - 1): (width * col_idx + width)])
        ax = ax.twiny()
        ax.boxplot(g)
        ax.set_xticks(())
        # ax.set_xlabel(gene_name, fontsize=fontsize)

    with catch_warnings_ignore(UserWarning):
      plt.tight_layout()
    self.add_figure('correlation_%s' % data_type, fig)
    return self._log('plot_correlation_series[%s] %s(s)' %
      (data_type, ctext(time.time() - start_time, 'lightyellow')))

  def plot_imputation_scatter(self, test=True, pca=False, color_by_library=True):
    start_time = time.time()
    n_system = len(self) + 2 # add the original and the corrupted
    data_type = 'test' if test else 'train'

    if n_system <= 5:
      nrow = 1
      ncol = n_system
    else:
      nrow = 2
      ncol = int(np.ceil(n_system / 2))

    X_org = self.posteriors[0].X_test_org if test else self.posteriors[0].X_train_org
    X_crr = self.posteriors[0].X_test if test else self.posteriors[0].X_train
    y = self.posteriors[0].y_test if test else self.posteriors[0].y_train
    labels = self.posteriors[0].labels
    is_binary_classes = self.posteriors[0].is_binary_classes
    allV = [X_org, X_crr] + [pos.V_test if test else pos.V_train
                             for pos in self.posteriors]
    assert X_org.shape == X_crr.shape and all(v.shape == X_org.shape for v in allV)
    all_names = ["[%s]Original" % data_type,
                 "[%s]Corrupted" % data_type] + [i.short_id_lines
                                                 for i in self.posteriors]

    # log-normalize everything
    if len(X_org) > 5000:
      np.random.seed(5218)
      ids = np.random.permutation(X_org.shape[0])[:5000]
      allV = [v[ids] for v in allV]
      y = y[ids]

    if is_binary_classes:
      y = np.argmax(y, axis=-1)
    else:
      y = GMMThresholding().fit_transform(y)
      y = np.argmax(y, axis=-1)

    allV = [log_norm(v) for v in allV]

    fig = plt.figure(figsize=(min(20, 5 * ncol) + 2, nrow * 5))
    for idx, (name, v) in enumerate(zip(all_names, allV)):
      ax = plt.subplot(nrow, ncol, idx + 1)
      n = np.sum(v, axis=-1)
      v = fast_pca(v, n_components=2) if pca else fast_tsne(v, n_components=2)
      with catch_warnings_ignore(Warning):
        if color_by_library:
          plot_scatter_heatmap(x=v, val=n, ax=ax, size=8,
                               legend_enable=False,
                               grid=False, title=name)
        else:
          plot_scatter(x=v,
                       color=[labels[i] for i in y],
                       marker=[labels[i] for i in y],
                       ax=ax, size=8,
                       legend_enable=True if idx == 0 else False,
                       grid=False, title=name)

    with catch_warnings_ignore(Warning):
      plt.tight_layout()
    self.add_figure(
        'imputation_scatter_%s_%s' % ('lib' if color_by_library else 'cell',
                                      data_type), fig)
    return self._log('plot_imputation_scatter[%s] %s(s)' %
      (data_type, ctext(time.time() - start_time, 'lightyellow')))

  def plot_latents_scatter(self, test=True, pca=False):
    start_time = time.time()
    data_type = 'test' if test else 'train'

    if len(self) <= 4:
      nrow = 1
      ncol = len(self)
    else:
      nrow = 2
      ncol = int(np.ceil(len(self) / 2))

    fig = plt.figure(figsize=(min(20, 5 * ncol) + 2, nrow * 5))

    for idx, pos in enumerate(self):
      ax = plt.subplot(nrow, ncol, idx + 1)
      with catch_warnings_ignore(Warning):
        pos.plot_latents_scatter(test=test, ax=ax,
                                 legend=True if idx == 0 else False,
                                 pca=pca)

    with catch_warnings_ignore(Warning):
      plt.tight_layout()
    self.add_figure(
        'latents_scatter_%s' % data_type, fig)
    return self._log('plot_latents_scatter[%s] %s(s)' %
      (data_type, ctext(time.time() - start_time, 'lightyellow')))

  def plot_learning_curves(self):
    start_time = time.time()

    fig = plt.figure(figsize=(20, len(self) * 4))
    n_metrics = 5

    for row_idx, pos in enumerate(self):
      row_idx = row_idx * n_metrics
      train = _extract_metrics(pos.train_history)
      valid = _extract_metrics(pos.valid_history)

      for col_idx, (name, i, j) in enumerate(
      zip(['loss', 'LLK_x', 'LLK_y', 'KLqp_x', 'KLqp_y'], train, valid)):
        col_idx += 1
        plt.subplot(len(self), n_metrics, row_idx + col_idx)

        if col_idx == 1:
          plt.title(pos.short_id_lines, fontsize=8, fontstyle='italic')
        else:
          plt.title(name)

        if i is None or j is None:
          plt.plot(0, 0); plt.xticks(()); plt.yticks(())
        else:
          plt.plot(i, linewidth=2.5, label='train:%.2f' % (np.max(i) if 'LLK' in name else np.min(i)),
                   linestyle='-')
          plt.plot(j, linewidth=2.5, label='valid:%.2f' % (np.max(j) if 'LLK' in name else np.min(j)),
                   linestyle='--', alpha=0.8)
          plt.legend()

    plt.tight_layout()
    self.add_figure('learning_curves', fig)
    return self._log('plot_learning_curves %s(s)' %
      ctext(time.time() - start_time, 'lightyellow'))

  def plot_scores(self, score_type='imputation', width=0.2):
    """
    score_type : 'imputation', 'cluster', 'spearman', 'pearson', 'classifier'
    """
    start_time = time.time()

    fn_score = _get_score_fn(score_type)
    scores = [fn_score(i) for i in self.posteriors]
    train_scores = [i[0] for i in scores]
    test_scores = [i[1] for i in scores]

    fmt = '%.4f' if score_type in ('imputation', 'cluster') \
    else '%.2f'

    n_scores = len(train_scores[0])
    n_system = len(self.posteriors)
    scores_name = sorted(list(train_scores[0].keys()))
    colors = sns.color_palette(n_colors=n_system)

    main_text_size = 10
    score_text_size = 6 if n_scores > 5 else 10

    fig = plt.figure(figsize=(18, 8))
    for idx, (name, scores) in enumerate(
        (('train', train_scores), ('test', test_scores))):
      ax = plt.subplot(2, 1, idx + 1)

      handlers = None
      start_pos = 0
      xticks = []
      for sc_idx, sc_name in enumerate(scores_name):
        sc = [i[sc_name] for i in scores]

        xids = start_pos + width * np.arange(len(sc))
        xticks.append(xids[0] + (xids[-1] - xids[0]) / 2)
        start_pos = xids[-1] + width * 2
        handlers = plt.bar(xids, sc, width=width, color=colors)
        # draw text on top of each bar
        for x, y in zip(xids, sc):
          plt.text(x, y, fmt % y,
                   fontsize=score_text_size, horizontalalignment='center')

      if idx == 0:
        plt.xticks(xticks, scores_name,
                   rotation=0, fontsize=main_text_size)
      else:
        plt.xticks(())

      plt.title('[%s]Comparison chart for %s scores' % (name, score_type),
                fontsize=main_text_size + 2)
      if idx == 1 and handlers is not None:
        ax.legend(handlers, [i.short_id for i in self.posteriors],
          loc='upper center', bbox_to_anchor=(0.5, -0.01), ncol=1,
          columnspacing=0.4, labelspacing=0.,
          fontsize=main_text_size, handletextpad=0.1)

    plt.tight_layout()
    self.add_figure('%s_compare' % score_type, fig)
    return self._log('plot_scores[%s] %s(s)' %
      (ctext(score_type, 'lightyellow'),
       ctext(time.time() - start_time, 'lightyellow')))

  # ******************** figure helper ******************** #
  def save_plots(self, path, dpi=None, separate_files=True):
    save_figures(self.figures, path, dpi, separate_files,
                clear_figures=True)
    return self

  @property
  def figures(self):
    if not hasattr(self, '_figures'):
      self._figures = OrderedDict()
    assert all(isinstance(k, string_types) and
               isinstance(v, plt.Figure)
              for k, v in self._figures.items()), \
    "Invalid stored Figures"
    return self._figures

  def add_figure(self, name, fig):
    for k, v in self.figures.items():
      if v == fig:
        return
    self.figures[name] = fig
    return self

  # ******************** properties ******************** #
  def summary(self):
    return self.__str__()

  def __str__(self):
    s = "#Posteriors: %s\n" % ctext(len(self), 'lightyellow')
    for i in self:
      s += " - %s\n" % ctext(i.id, 'cyan')
    return s

  def __len__(self):
    return len(self._posteriors)

  def __getitem__(self, key):
    if isinstance(key, string_types):
      return [i for i in self if key in i.id.split('_')][0]
    elif callable(key):
      return [i for i in self if key(i)][0]
    return self.posteriors.__getitem__(key)

  def __iter__(self):
    return self.posteriors.__iter__()
