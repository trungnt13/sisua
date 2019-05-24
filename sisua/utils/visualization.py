from __future__ import print_function, division, absolute_import

import os
from collections import OrderedDict

import numpy as np
import tensorflow as tf
import pandas as pd
import seaborn as sbn

from odin.utils import ctext, catch_warnings_ignore, one_hot
from odin import backend as K
from odin.utils import async_mpi
from odin.ml import fast_tsne, fast_pca
from odin.visual import (generate_random_colors, generate_random_marker,
                         plot_gridSubplot, plot_gridSpec, subplot,
                         plot_scatter, plot_scatter_layers, plot_figure,
                         plot_scatter_heatmap, plot_confusion_matrix,
                         plot_histogram, plot_save,
                         plot_series_statistics)

from sklearn.metrics import (confusion_matrix, f1_score,
                             r2_score, explained_variance_score,
                             mean_squared_error, mean_absolute_error)

from sisua.utils.others import anything2image

# ===========================================================================
# Some helper
# ===========================================================================
def _clip_count_sum(x):
  return np.clip(x,
                 a_min=np.min(x),
                 a_max=np.percentile(x, q=95))

def _clipping_quartile(x, alpha=1.5):
  # result wider range threshold, keep more data points,
  # lower LLK
  x = x.astype('int32')
  Q1 = np.percentile(x, q=25)
  Q3 = np.percentile(x, q=75)
  IQR = Q3 - Q1
  x = x[x < Q3 + alpha * IQR]
  x = x[x > Q1 - alpha * IQR]
  return x

def downsample_data(*X):
  y = [None] * len(X)
  _ = list(set(x.shape[0] for x in X
               if x is not None))
  assert len(_) == 1, "Inconsistent shape[0] for X and y"
  num_samples = _[0]
  _RAND = np.random.RandomState(seed=52181208)
  # ====== Downsample if the data is huge ====== #
  if num_samples > 8000:
    print("[Warning] Given: %s; downsample to 8000 samples" %
      ctext(', '.join([str(x.shape) for x in X
                      if x is not None]), 'cyan'))
    ids = _RAND.choice(a=np.arange(0, num_samples),
                       size=8000, replace=False)
    for i, x in enumerate(X):
      if x is not None:
        x = x[ids]
      y[i] = x
  else:
    y = X
  return tuple(y)

def show_image(x, is_probability=False):
  from matplotlib import pyplot as plt
  from odin import backend as K
  x = anything2image(x)

  if x.shape[0] > 32:
    x = tf.nn.max_pool(np.reshape(x, (1, x.shape[0], x.shape[1], 1)),
                       ksize=(1, 4, 4, 1),
                       strides=(1, 4, 4, 1),
                       padding="SAME")
    x = np.squeeze(K.eval(x))
  ax = plt.gca()
  plt.imshow(x, interpolation='nearest', cmap=plt.cm.Greys_r,
             vmin=0. if is_probability else None,
             vmax=1. if is_probability else None)
  plt.xticks(()); ax.set_xticklabels([])
  plt.yticks(()); ax.set_yticklabels([])
  ax.set_aspect(aspect='auto')

def fast_scatter(x, y, labels, title,
                 azim=None, elev=None, ax=None,
                 enable_legend=False,
                 size=18, fontsize=12):
  y = np.squeeze(y)
  if y.ndim == 1:
    pass
  elif y.ndim == 2: # provided one-hot vectors
    y = np.argmax(y, axis=-1)
  else:
    raise ValueError("No support for `y` shape: %s" % str(y.shape))
  # ====== get colors and legends ====== #
  if labels is not None:
    y = [labels[int(i)] for i in y]
    num_classes = len(labels)
  else:
    num_classes = len(np.unique(y))
  # int(np.ceil(num_classes / 2)) if num_classes <= 20 else num_classes // 5
  plot_scatter(x=x, color=y, marker=y,
      size=size, azim=azim, elev=elev,
      legend_enable=enable_legend,
      legend_ncol=3,
      fontsize=fontsize, ax=ax, title=title)

# ===========================================================================
# IO
# ===========================================================================
def save_figures(figures, path, dpi=None,
                 separate_files=True, clear_figures=True):
  if os.path.isfile(path) or '.pdf' == path[-4:].lower():
    separate_files = False
    assert '.pdf' == path[-4:].lower(), \
    "If a file is given, it must be PDF file"
  assert isinstance(figures, dict), \
  "figures must be dictionary mapping from figure name to matplotlib.Figure"
  n_figures = len(figures)
  # ====== saving PDF file ====== #
  if not separate_files:
    if dpi is None:
      dpi = 48

    if '.pdf' not in path:
      path = path + '.pdf'
    from matplotlib.backends.backend_pdf import PdfPages
    pp = PdfPages(path)
    for key, fig in figures.items():
      try:
        fig.savefig(pp, dpi=dpi, format='pdf', bbox_inches="tight")
      except Exception as e:
        print("Error:", key)
        print(" ", e)
    pp.close()
  # ====== saving PNG file ====== #
  else:
    if dpi is None:
      dpi = 160

    if not os.path.exists(path):
      os.mkdir(path)
    assert os.path.isdir(path), "'%s' must be path to a folder" % path
    kwargs = dict(dpi=dpi, bbox_inches="tight")
    for key, fig in figures.items():
      out_path = os.path.join(path, key + '.png')
      try:
        fig.savefig(out_path, **kwargs)
      except Exception as e:
        print("Error:", ctext(out_path, 'red'))
        print(" ", e)
  # ====== clear figures ====== #
  print("%s figures save to path: %s" %
    (ctext(n_figures, 'lightcyan'), ctext(path, 'lightyellow')))
  if clear_figures:
    figures.clear()

# ===========================================================================
# Evaluating the reconstruction
# ===========================================================================
def plot_evaluate_reconstruction(X, W, y_raw, y_prob,
                                 X_row, X_col, labels, title,
                                 pi=None,
                                 enable_image=True,
                                 enable_tsne=True,
                                 enable_sparsity=True):
  """
  pi : zero-inflated rate (imputation rate), or dropout probabilities
  """
  print("Evaluate: [Reconstruction]", ctext(title, 'lightyellow'))
  from matplotlib import pyplot as plt
  fontsize = 12

  W_stdev_total, W_stdev_explained = None, None
  if isinstance(W, (tuple, list)):
    if len(W) == 1:
      W = W[0]
    elif len(W) == 3:
      W, W_stdev_total, W_stdev_explained = W
    else:
      raise RuntimeError()
  elif W.ndim == 3:
    W, W_stdev_total, W_stdev_explained = W[0], W[1], W[2]
  # convert the prediction to integer
  # W = W.astype('int32')

  assert (X.shape[0] == W.shape[0] == y_raw.shape[0]) and \
  (X.shape == W.shape) and \
  (y_raw.shape == y_prob.shape)

  X, X_row, W, y_raw, y_prob, pi = downsample_data(X, X_row, W, y_raw, y_prob, pi)
  y_argmax = np.argmax(y_prob, axis=-1)

  # ====== prepare count-sum ====== #
  X_log = K.log_norm(X, axis=1)
  W_log = K.log_norm(W, axis=1)

  X_gene_countsum = np.sum(X, axis=0)
  X_cell_countsum = np.sum(X, axis=1)
  X_gene_nzeros = np.sum(X == 0, axis=0)
  X_cell_nzeros = np.sum(X == 0, axis=1)

  gene_sort = np.argsort(X_gene_countsum)
  cell_sort = np.argsort(X_cell_countsum)

  W_gene_countsum = np.sum(W, axis=0)
  W_cell_countsum = np.sum(W, axis=1)
  W_gene_nzeros = np.sum(W == 0, axis=0)
  W_cell_nzeros = np.sum(W == 0, axis=1)

  X_col_sorted = X_col[gene_sort] if X_col is not None else None
  X_row_sorted = X_row[cell_sort] if X_row is not None else None

  if pi is not None:
    pi_cell_countsum = np.mean(pi, axis=1)
    pi_gene_countsum = np.mean(pi, axis=0)

  # ====== Compare image ====== #
  if enable_image:
    _RAND = np.random.RandomState(seed=52181208)
    n_img = 12
    n_img_row = min(3, X.shape[0] // n_img)
    n_row_per_row = 2 if pi is None else 3
    plot_figure(nrow=n_img_row * 4, ncol=18)
    count = 1
    all_ids = _RAND.choice(np.arange(0, X.shape[0]),
                       size=n_img * n_img_row,
                       replace=False)

    for img_row in range(n_img_row):
      ids = all_ids[img_row * n_img: (img_row + 1) * n_img]

      # plot original images
      for _, i in enumerate(ids):
        ax = plt.subplot(n_row_per_row * n_img_row, n_img, count)
        show_image(X[i])
        if _ == 0:
          plt.ylabel("Original")
        if X_row is not None:
          ax.set_title(X_row[i], fontsize=8)
        count += 1
      # plot reconstructed images
      for _, i in enumerate(ids):
        plt.subplot(n_row_per_row * n_img_row, n_img, count)
        show_image(W[i])
        if _ == 0:
          plt.ylabel("Reconstructed")
        count += 1
      # plot zero-inflated rate
      if pi is not None:
        for _, i in enumerate(ids):
          plt.subplot(n_row_per_row * n_img_row, n_img, count)
          show_image(pi[i], is_probability=True)
          if _ == 0:
            plt.ylabel("$p_{zero-inflated}$")
          count += 1
    plt.tight_layout()
  # ====== compare the T-SNE plot ====== #
  if enable_tsne:
    def pca_and_tsne(x, w):
      x_pca, w_pca = fast_pca(x, w, n_components=512,
                              random_state=52181208)
      x_tsne = fast_tsne(x_pca, n_components=2, random_state=52181208)
      w_tsne = fast_tsne(w_pca, n_components=2, random_state=52181208)
      return x_pca[:, :2], x_tsne, w_pca[:, :2], w_tsne
    # transforming the data
    (X_cell_pca, X_cell_tsne,
     W_cell_pca, W_cell_tsne) = pca_and_tsne(X_log, W_log)
    (X_gene_pca, X_gene_tsne,
     W_gene_pca, W_gene_tsne) = pca_and_tsne(X_log.T, W_log.T)
    # prepare the figure
    n_plot = 3 + 2 # 3 for cells, 2 for genes
    if pi is not None:
      n_plot += 2 # 2 more row for pi
    plot_figure(nrow=n_plot * 5, ncol=18)
    # Cells
    fast_scatter(x=X_cell_pca, y=y_argmax, labels=labels,
                 title="[PCA]Original Cell Data",
                 ax=(n_plot, 2, 1), enable_legend=False)
    fast_scatter(x=W_cell_pca, y=y_argmax, labels=labels,
                 title="[PCA]Reconstructed Cell Data",
                 ax=(n_plot, 2, 2), enable_legend=False)

    fast_scatter(x=X_cell_tsne, y=y_argmax, labels=labels,
                 title="[t-SNE]Original Cell Data",
                 ax=(n_plot, 2, 3), enable_legend=True)
    fast_scatter(x=W_cell_tsne, y=y_argmax, labels=labels,
                 title="[t-SNE]Reconstructed Cell Data",
                 ax=(n_plot, 2, 4), enable_legend=False)

    fast_log = lambda x: K.log_norm(x, axis=0)

    plot_scatter_heatmap(x=X_cell_tsne, val=fast_log(X_cell_countsum),
                         title="[t-SNE]Original Cell Data + Original Cell Countsum",
                         ax=(n_plot, 2, 5), colorbar=True)
    plot_scatter_heatmap(x=X_cell_tsne, val=fast_log(W_cell_countsum),
                         title="[t-SNE]Original Cell Data + Reconstructed Cell Countsum",
                         ax=(n_plot, 2, 6), colorbar=True)
    # Genes
    plot_scatter_heatmap(x=X_gene_pca, val=fast_log(X_gene_countsum),
                         title="[PCA]Original Gene Data + Original Gene Countsum",
                         ax=(n_plot, 2, 7), colorbar=True)
    plot_scatter_heatmap(x=W_gene_pca, val=fast_log(X_gene_countsum),
                         title="[PCA]Reconstructed Gene Data + Original Gene Countsum",
                         ax=(n_plot, 2, 8), colorbar=True)

    plot_scatter_heatmap(x=X_gene_tsne, val=fast_log(X_gene_countsum),
                         title="[t-SNE]Original Gene Data + Original Gene Countsum",
                         ax=(n_plot, 2, 9), colorbar=True)
    plot_scatter_heatmap(x=X_gene_tsne, val=fast_log(W_gene_countsum),
                         title="[t-SNE]Original Gene Data + Reconstructed Gene Countsum",
                         ax=(n_plot, 2, 10), colorbar=True)
    # zero-inflation rate
    if pi is not None:
      plot_scatter_heatmap(x=X_cell_tsne, val=X_cell_countsum,
                           title="[t-SNE]Original Cell Data + Original Cell Countsum",
                           ax=(n_plot, 2, 11), colorbar=True)
      plot_scatter_heatmap(x=X_cell_tsne, val=pi_cell_countsum,
                           title="[t-SNE]Original Cell Data + Zero-inflated rate",
                           ax=(n_plot, 2, 12), colorbar=True)

      plot_scatter_heatmap(x=X_gene_tsne, val=X_gene_countsum,
                           title="[t-SNE]Original Gene Data + Original Gene Countsum",
                           ax=(n_plot, 2, 13), colorbar=True)
      plot_scatter_heatmap(x=X_gene_tsne, val=pi_gene_countsum,
                           title="[t-SNE]Original Gene Data + Zero-inflated rate",
                           ax=(n_plot, 2, 14), colorbar=True)
    plt.tight_layout()
  # ******************** sparsity ******************** #
  if enable_sparsity:
    plot_figure(nrow=8, ncol=8)
    # ====== sparsity ====== #
    z = (X.ravel() == 0).astype('int32')
    z_res = (W.ravel() == 0).astype('int32')
    plot_confusion_matrix(ax=None,
        cm=confusion_matrix(y_true=z, y_pred=z_res, labels=(0, 1)),
        labels=('Not Zero', 'Zero'), colorbar=True, fontsize=fontsize + 4,
        title="Sparsity")

# ===========================================================================
# Streamline classifier
# ===========================================================================
def plot_evaluate_classifier(y_pred, y_true, labels, title,
                             show_plot=True, return_figure=False):
  """ Return a dictionary of scores
  {
      F1micro=f1_micro * 100,
      F1macro=f1_macro * 100,
      F1weight=f1_weight * 100,
      F1_[classname]=...
  }
  """
  from matplotlib import pyplot as plt
  fontsize = 12
  num_classes = len(labels)
  nrow = int(np.ceil(num_classes / 5))
  ncol = int(np.ceil(num_classes / nrow))

  if y_pred.ndim == 1:
    y_pred = one_hot(y_pred, nb_classes=num_classes)
  if y_true.ndim == 1:
    y_true = one_hot(y_true, nb_classes=num_classes)

  if show_plot:
    fig = plot_figure(nrow=4 * nrow + 2, ncol=4 * ncol)

  f1_classes = []
  for i, (name, pred, true) in enumerate(zip(labels, y_pred.T, y_true.T)):
    f1_classes.append(f1_score(true, pred))
    if show_plot:
      plot_confusion_matrix(confusion_matrix(y_true=true, y_pred=pred),
                            labels=[0, 1],
                            fontsize=fontsize,
                            ax=(nrow, ncol, i + 1),
                            title=name + '\n')

  f1_micro = f1_score(y_true=y_true.ravel(), y_pred=y_pred.ravel())
  f1_macro = np.mean(f1_classes)
  f1_weight = f1_score(y_true=y_true, y_pred=y_pred, average='weighted')

  if show_plot:
    plt.suptitle('%s\nF1-micro:%.2f  F1-macro:%.2f  F1-weight:%.2f' %
                 (title, f1_micro * 100, f1_macro * 100, f1_weight * 100))
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])

  results = dict(
      F1micro=f1_micro * 100,
      F1macro=f1_macro * 100,
      F1weight=f1_weight * 100,
  )
  for name, f1 in zip(labels, f1_classes):
    results['F1_' + name] = f1 * 100

  if show_plot and return_figure:
    return results, fig
  return results

def plot_evaluate_regressor(y_pred, y_true, labels, title):
  from matplotlib import pyplot as plt
  num_classes = len(labels)
  nbins = 120
  fontsize = 8
  y_pred = np.round(y_pred).astype('int32')
  y_true = y_true.astype('int32')
  # ====== basic scores ====== #
  r2 = r2_score(y_true, y_pred)
  var = explained_variance_score(y_true, y_pred)
  mse = mean_squared_error(y_true, y_pred)
  mae = mean_absolute_error(y_true, y_pred)

  # ====== helper ====== #
  def plot_hist(hist, ax, name):
    count, bins = plot_histogram(true, bins=nbins, ax=ax, title=name, fontsize=fontsize)
    plt.xlim((np.min(bins), np.max(bins)))
    plt.xticks(np.linspace(start=np.min(bins), stop=np.max(bins), num=8, dtype='int32'),
               fontsize=6)
    plt.yticks(np.linspace(start=np.min(count), stop=np.max(count), num=8, dtype='int32'),
               fontsize=6)

  # ====== raw count prediction ====== #
  plot_figure(nrow=4, ncol=num_classes * 2)
  for i in range(num_classes):
    name = labels[i]
    r2_ = r2_score(y_true=y_true[:, i],
                   y_pred=y_pred[:, i])
    pred = _clipping_quartile(y_pred[:, i])
    true = _clipping_quartile(y_true[:, i])
    plot_hist(hist=true, ax=(2, num_classes, i + 1),
              name='[%s] R2: %.6f' % (name, r2_))
    if i == 0:
      plt.ylabel('True')
    plot_hist(hist=pred, ax=(2, num_classes, num_classes + i + 1), name=None)
    if i == 0:
      plt.ylabel('Pred')
  # set the title
  plt.suptitle('[%s]  R2: %.6f   ExpVAR: %.6f   MSE: %.6f   MAE: %.6f' %
    (str(title), r2, var, mse, mae), fontsize=fontsize + 1)

# ===========================================================================
# Monitoring the training
# ===========================================================================
def _show_zero_inflated_pi(p, ax, handles, indices):
  ax = ax.twinx()
  _, = ax.plot(p[indices],
               marker='o', linestyle='None',
               markersize=0.8, color='blue',
               alpha=0.3, label="$p_{dropout}$")
  ax.set_ylabel("Zero-inflation rate", fontsize=8)
  ax.tick_params(axis='both', labelsize=8)
  handles.append(_)

@async_mpi
def plot_monitoring_epoch(X, X_drop, y, Z, Z_drop,
                          W_outputs, W_drop_outputs,
                          pi, pi_drop,
                          row_name, dropout_percentage,
                          curr_epoch, ds_name, labels, save_dir):
  # Order of W_outputs: [W, W_stdev_total, W_stdev_explained]
  from matplotlib import pyplot as plt
  if y.ndim == 2:
    y = np.argmax(y, axis=-1)
  y = np.array([labels[i] for i in y])
  dropout_percentage_text = '%g%%' % (dropout_percentage * 100)

  Z_pca = fast_pca(Z, n_components=2, random_state=5218)
  Z_pca_drop = fast_pca(Z_drop, n_components=2, random_state=5218)
  if W_outputs is not None:
    X_pca, X_pca_drop, W_pca, W_pca_drop = fast_pca(
        X, X_drop, W_outputs[0], W_drop_outputs[0],
        n_components=2, random_state=5218)
   # ====== downsampling ====== #
  rand = np.random.RandomState(seed=5218)
  n_test_samples = len(y)
  ids = np.arange(n_test_samples, dtype='int32')
  if n_test_samples > 8000:
    ids = rand.choice(ids, size=8000, replace=False)
  # ====== scatter configuration ====== #
  config = dict(size=6, labels=None)
  y = y[ids]

  X = X[ids]
  X_drop = X_drop[ids]
  Z_pca = Z_pca[ids]
  X_pca = X_pca[ids]
  W_pca = W_pca[ids]

  W_outputs = [w[ids] for w in W_outputs]
  W_drop_outputs = [w[ids] for w in W_drop_outputs]
  Z_pca_drop = Z_pca_drop[ids]
  X_pca_drop = X_pca_drop[ids]
  W_pca_drop = W_pca_drop[ids]

  if pi is not None:
    pi = pi[ids]
    pi_drop = pi_drop[ids]
  # ====== plotting NO reconstruction ====== #
  if W_outputs is None:
    plot_figure(nrow=8, ncol=20)
    fast_scatter(x=Z_pca, y=y,
                 title="[PCA] Test data latent space",
                 enable_legend=True, ax=(1, 2, 1), **config)
    fast_scatter(x=Z_pca_drop, y=y,
                 title="[PCA][Dropped:%s] Test data latent space" % dropout_percentage_text,
                 ax=(1, 2, 2), **config)
  # ====== plotting WITH reconstruction ====== #
  else:
    plot_figure(nrow=16, ncol=20)
    # original test data WITHOUT dropout
    fast_scatter(x=X_pca, y=y, title="[PCA][Test Data] Original",
      ax=(2, 3, 1), **config)
    fast_scatter(x=W_pca, y=y, title="Reconstructed",
      ax=(2, 3, 2), **config)
    fast_scatter(x=Z_pca, y=y, title="Latent space",
      ax=(2, 3, 3), **config)
    # original test data WITH dropout
    fast_scatter(x=X_pca_drop, y=y,
      title="[PCA][Dropped:%s][Test Data] Original" % dropout_percentage_text,
      ax=(2, 3, 4), **config)
    fast_scatter(x=W_pca_drop, y=y, title="Reconstructed",
      ax=(2, 3, 5), enable_legend=True, **config)
    fast_scatter(x=Z_pca_drop, y=y, title="Latent space",
      ax=(2, 3, 6), **config)
  plot_save(os.path.join(save_dir, 'latent_epoch%d.png') % curr_epoch,
            dpi=180, clear_all=True, log=True)
  # ====== plot count-sum ====== #
  if W_outputs is not None:
    X_countsum = _clip_count_sum(np.sum(X, axis=-1))
    W_countsum = _clip_count_sum(np.sum(W_outputs[0], axis=-1))
    X_drop_countsum = _clip_count_sum(np.sum(X_drop, axis=-1))
    W_drop_countsum = _clip_count_sum(np.sum(W_drop_outputs[0], axis=-1))
    series_config = [dict(xscale='linear', yscale='linear', sort_by=None),
                     dict(xscale='linear', yscale='linear', sort_by='expected')]

    if pi is not None:
      pi_sum = np.mean(pi, axis=-1)
      pi_drop_sum = np.mean(pi_drop, axis=-1)
    # plot the reconstruction count sum
    plot_figure(nrow=3 * 5 + 8, ncol=18)
    with plot_gridSpec(nrow=3 * (2 if pi is None else 3) + 4 * 3 + 1,
                       ncol=6,
                       wspace=1.0, hspace=0.8) as grid:
      kws = dict(colorbar=True, fontsize=10, size=10, marker=y, n_samples=1200)
      # without dropout
      ax = subplot(grid[:3, 0:3])
      plot_scatter_heatmap(x=X_pca, val=X_countsum, ax=ax,
                           legend_enable=False, title='Original data (Count-sum)',
                           **kws)
      ax = subplot(grid[:3, 3:6])
      plot_scatter_heatmap(x=W_pca, val=W_countsum, ax=ax,
                           legend_enable=False, title='Reconstruction (Count-sum)',
                           **kws)
      # with dropout
      ax = subplot(grid[3:6, 0:3])
      plot_scatter_heatmap(x=X_pca_drop, val=X_drop_countsum, ax=ax,
                           legend_enable=True if pi is None else False,
                           legend_ncol=len(labels),
                           title='[Dropped:%s]Original data (Count-sum)' % dropout_percentage_text,
                           **kws)
      ax = subplot(grid[3:6, 3:6])
      plot_scatter_heatmap(x=W_pca_drop, val=W_drop_countsum, ax=ax,
                           legend_enable=False,
                           title='[Dropped:%s]Reconstruction (Count-sum)' % dropout_percentage_text,
                           **kws)
      row_start = 6
      # zero-inflated pi
      if pi is not None:
        ax = subplot(grid[6:9, 0:3])
        plot_scatter_heatmap(x=X_pca, val=pi_sum, ax=ax,
                             legend_enable=True, legend_ncol=len(labels),
                             title='Zero-inflated probabilities',
                             **kws)
        ax = subplot(grid[6:9, 3:6])
        plot_scatter_heatmap(x=X_pca, val=pi_drop_sum, ax=ax,
                             legend_enable=False,
                             title='[Dropped:%s]Zero-inflated probabilities' % dropout_percentage_text,
                             **kws)
        row_start += 3

      # plot the count-sum series
      def plot_count_sum_series(x, w, p, row_start, tit):
        if len(w) != 3: # no statistics provided
          return
        expected, stdev_total, stdev_explained = w
        count_sum_observed = np.sum(x, axis=0)
        count_sum_expected = np.sum(expected, axis=0)
        count_sum_stdev_total = np.sum(stdev_total, axis=0)
        count_sum_stdev_explained = np.sum(stdev_explained, axis=0)
        if p is not None:
          p_sum = np.mean(p, axis=0)
        for i, kws in enumerate(series_config):
          ax = subplot(grid[row_start:row_start + 3, (i * 3):(i * 3 + 3)])
          ax, handles, indices = plot_series_statistics(
              count_sum_observed, count_sum_expected,
              explained_stdev=count_sum_stdev_explained,
              total_stdev=count_sum_stdev_total,
              fontsize=8, ax=ax, legend_enable=False,
              title=tit if i == 0 else None,
              despine=True if p is None else False,
              return_handles=True, return_indices=True,
              **kws)
          if p is not None:
            _show_zero_inflated_pi(p_sum, ax, handles, indices)
          plt.legend(handles=handles, loc='best', markerscale=4, fontsize=8)
      # add one row extra padding
      row_start += 1
      plot_count_sum_series(x=X, w=W_outputs, p=pi,
                            row_start=row_start,
                            tit="Count-sum X_original - W_original")
      row_start += 1
      plot_count_sum_series(x=X_drop, w=W_drop_outputs, p=pi_drop,
                            row_start=row_start + 3,
                            tit="[Dropped:%s]Count-sum X_drop - W_dropout" % dropout_percentage_text)
      row_start += 1
      plot_count_sum_series(x=X, w=W_drop_outputs, p=pi_drop,
                            row_start=row_start + 6,
                            tit="[Dropped:%s]Count-sum X_original - W_dropout" % dropout_percentage_text)
    plot_save(os.path.join(save_dir, 'countsum_epoch%d.png') % curr_epoch,
              dpi=180, clear_all=True, log=True)
  # ====== plot series of samples ====== #
  if W_outputs is not None and len(W_outputs) == 3:
    # NOTe: turn off pi here
    pi = None

    n_visual_samples = 8
    plot_figure(nrow=3 * n_visual_samples + 8, ncol=25)
    col_width = 5
    with plot_gridSpec(nrow=3 * n_visual_samples,
                       ncol=4 * col_width,
                       wspace=5.0, hspace=1.0) as grid:
      curr_grid_index = 0
      for i in rand.permutation(len(X))[:n_visual_samples]:
        observed = X[i]
        expected, stdev_explained, stdev_total = [w[i] for w in W_outputs]
        expected_drop, stdev_explained_drop, stdev_total_drop = [w[i] for w in W_drop_outputs]
        if pi is not None:
          p_zi = pi[i]
          p_zi_drop = pi_drop[i]
        # compare to W_original
        for j, kws in enumerate(series_config):
          ax = subplot(grid[curr_grid_index:curr_grid_index + 3,
                            (j * col_width):(j * col_width + col_width)])
          ax, handles, indices = plot_series_statistics(
              observed, expected,
              explained_stdev=stdev_explained,
              total_stdev=stdev_total,
              fontsize=8, legend_enable=False,
              despine=True if pi is None else False,
              title=("'%s' X_original - W_original" % row_name[i])
              if j == 0 else None,
              return_handles=True, return_indices=True,
              **kws)
          if pi is not None:
            _show_zero_inflated_pi(p_zi, ax, handles, indices)
          plt.legend(handles=handles, loc='best', markerscale=4, fontsize=8)
        # compare to W_dropout
        for j, kws in enumerate(series_config):
          col_start = col_width * 2
          ax = subplot(grid[curr_grid_index:curr_grid_index + 3,
                            (col_start + j * col_width):(col_start + j * col_width + col_width)])
          ax, handles, indices = plot_series_statistics(
              observed, expected_drop,
              explained_stdev=stdev_explained_drop,
              total_stdev=stdev_total_drop,
              fontsize=8, legend_enable=False,
              despine=True if pi is None else False,
              title=("[Dropped:%s]'%s' X_original - W_dropout" %
              (dropout_percentage_text, row_name[i])) if j == 0 else None,
              return_handles=True, return_indices=True,
              **kws)
          if pi is not None:
            _show_zero_inflated_pi(p_zi_drop, ax, handles, indices)
          plt.legend(handles=handles, loc='best', markerscale=4, fontsize=8)
        curr_grid_index += 3
    plot_save(os.path.join(save_dir, 'samples_epoch%d.png') % curr_epoch,
              dpi=180, clear_all=True, log=True)
  # ====== special case for mnist ====== #
  if 'mnist' in ds_name and W_outputs is not None:
    plot_figure(nrow=3, ncol=18)
    n_images = 32
    ids = rand.choice(np.arange(X.shape[0], dtype='int32'),
                      size=n_images, replace=False)
    meta_data = [
        ("Org", X[ids]),
        ("Rec", W_outputs[0][ids]),
        ("OrgDropout", X_drop[ids]),
        ("RecDropout", W_drop_outputs[0][ids])
    ]
    count = 1
    for name, data in meta_data:
      for i in range(n_images):
        x = data[i].reshape(28, 28)
        plt.subplot(4, n_images, count); show_image(x)
        if i == 0:
          plt.ylabel(name, fontsize=8)
        count += 1
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    plot_save(os.path.join(save_dir, 'image_epoch%d.png') % curr_epoch,
              dpi=180, clear_all=True, log=True)
