from __future__ import print_function, division, absolute_import

from collections import OrderedDict

import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from matplotlib import gridspec

from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                             silhouette_score, adjusted_mutual_info_score)
from sklearn.utils.linear_assignment_ import linear_assignment

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from odin import backend as K
from odin.utils import (ctext, catch_warnings_ignore)
from odin.visual import (plot_figure, plot_scatter_heatmap,
                         to_axis, plot_colorbar)

from sisua.data.const import UNIVERSAL_RANDOM_SEED
from sisua.utils.visualization import (plot_evaluate_classifier,
  downsample_data, fast_tsne, fast_scatter, fast_pca)

# ===========================================================================
# Metrics
# ===========================================================================
def multi_label_adj_Rindex(label_bin_true, label_pred):
  assert label_bin_true.ndim == 2
  assert label_bin_true.shape[1] == len(np.unique(label_pred))
  n_classes = label_bin_true.shape[1]
  with catch_warnings_ignore(Warning):
    scores = []
    for y in label_bin_true.T:
      y = y.astype('int32')
      s = max(adjusted_rand_score(labels_true=y,
                                  labels_pred=(label_pred == i).astype('int32'))
              for i in range(n_classes))
      scores.append(s)
  return scores

def unsupervised_clustering_accuracy(y, y_pred):
  """
  Unsupervised Clustering Accuracy
  Author: scVI
  https://github.com/YosefLab/scVI/blob/a585f7d096f04ab0d50cadfdf8c2c9f78d907c19/scvi/inference/posterior.py#L637
  """
  assert len(y_pred) == len(y)
  u = np.unique(np.concatenate((y, y_pred)))
  n_clusters = len(u)
  mapping = dict(zip(u, range(n_clusters)))
  reward_matrix = np.zeros((n_clusters, n_clusters), dtype=np.int64)
  for y_pred_, y_ in zip(y_pred, y):
    if y_ in mapping:
      reward_matrix[mapping[y_pred_], mapping[y_]] += 1
  cost_matrix = reward_matrix.max() - reward_matrix
  ind = linear_assignment(cost_matrix)
  return sum([reward_matrix[i, j] for i, j in ind]) * 1.0 / y_pred.size, ind


def clustering_scores(latent, labels, n_labels=None,
                      prediction_algorithm='both'):
  """ Clustering Scores:

   * silhouette_score (higher is better, best is 1, worst is -1)
   * adjusted_rand_score (higher is better)
   * normalized_mutual_info_score (higher is better)
   * unsupervised_clustering_accuracy (higher is better)

  note: remember the order of returned value

  Parameters
  ----------
  prediction_algorithm : {'knn', 'gmm', 'both'}
  """
  if prediction_algorithm == 'knn':
    km = KMeans(n_labels, n_init=200, random_state=5218)
    labels_pred = km.fit_predict(latent)
  elif prediction_algorithm == 'gmm':
    gmm = GaussianMixture(n_labels, random_state=5218)
    gmm.fit(latent)
    labels_pred = gmm.predict(latent)
  elif prediction_algorithm == 'both':
    score1 = clustering_scores(latent, labels, n_labels=n_labels,
                               prediction_algorithm='knn')
    score2 = clustering_scores(latent, labels, n_labels=n_labels,
                               prediction_algorithm='gmm')
    return (score1 + score2) / 2
  else:
    raise ValueError(
        "Not support for prediction_algorithm: '%s'" % prediction_algorithm)
  #
  asw_score = silhouette_score(latent, labels)
  ari_score = adjusted_rand_score(labels, labels_pred)
  nmi_score = normalized_mutual_info_score(labels, labels_pred)
  uca_score = unsupervised_clustering_accuracy(labels, labels_pred)[0]
  return np.array((asw_score, ari_score, nmi_score, uca_score))


# ===========================================================================
# Visualization
# ===========================================================================
def streamline_classifier(Z_train, y_train, Z_test, y_test,
                          labels, title,
                          show_plot=True):
  """ Return a dictionary of scores
  {
      F1micro=f1_micro * 100,
      F1macro=f1_macro * 100,
      F1weight=f1_weight * 100,
      F1_[classname]=...
  }
  """
  with catch_warnings_ignore(FutureWarning):
    with catch_warnings_ignore(RuntimeWarning):
      # kernel : 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'
      classifier = OneVsRestClassifier(
          SVC(kernel='linear', random_state=UNIVERSAL_RANDOM_SEED))
      classifier.fit(X=Z_train, y=y_train)
      #
      results = plot_evaluate_classifier(
          y_pred=classifier.predict(Z_test), y_true=y_test,
          labels=labels,
          title=title, show_plot=bool(show_plot))
  return OrderedDict(sorted(results.items(),
                            key=lambda x: x[0]))

def plot_distance_heatmap(X, labels, labels_name=None, lognorm=True,
                  colormap='hot', ax=None,
                  legend_enable=True,
                  legend_loc='upper center', legend_ncol=3, legend_colspace=0.2,
                  fontsize=10, show_colorbar=True, title=None):
  """
  Example
  -------
  labels : (n_samples, n_classes) or (n_samples, 1) or (n_samples,)
    list of classes index, in case of binary classification,
    the list can be float value represent confidence value for
    positive class.

  labels_name : (n_classes,)
    list of classes' name, this will be used to determine
    number of classes

  # visualize_distance(latent_scVI, labels, "scVI")
  """
  from matplotlib.lines import Line2D
  X = K.length_norm(X, axis=-1, epsilon=np.finfo(X.dtype).eps)

  ax = to_axis(ax)
  n_samples, n_dim = X.shape

  # processing labels
  labels = np.array(labels)
  if labels.ndim == 2:
    if labels.shape[1] == 1:
      labels = labels.ravel()
    else:
      labels = np.argmax(labels, axis=-1)
  elif labels.ndim > 2:
    raise ValueError("Only support 1-D or 2-D labels")

  labels_int = labels.astype('int32')
  # float values label (normalize -1 to 1) or binary classification
  if not np.all(labels_int == labels) or \
  (labels_name is not None and len(labels_name) == 2) or \
  (len(np.unique(labels)) == 2):
    min_val = np.min(labels)
    max_val = np.max(labels)
    labels = 2 * (labels - min_val) / (max_val - min_val) - 1
    label_colormap = 'bwr'
  # integer values label and multiple classes classification
  else:
    labels = labels_int
    label_colormap = 'Dark2'

  # ====== sorting label and X ====== #
  order_X = np.vstack([x for _, x in sorted(zip(labels, X),
                      key=lambda pair: pair[0])])
  order_label = np.vstack([y for y, x in sorted(zip(labels, X),
                           key=lambda pair: pair[0])])
  distance = sp.spatial.distance_matrix(order_X, order_X)
  if bool(lognorm):
    distance = np.log1p(distance)
  min_non_zero = np.min(distance[np.nonzero(distance)])
  distance = np.clip(distance, a_min=min_non_zero, a_max=np.max(distance))

  # ====== convert data to image ====== #
  cm = plt.get_cmap(colormap)
  distance_img = cm(distance)
  # diagonal black line (i.e. zero distance)
  for i in range(n_samples):
    distance_img[i, i] = (0, 0, 0, 1)

  cm = plt.get_cmap(label_colormap)
  width = max(int(0.032 * n_samples), 8)
  horz_bar = np.repeat(cm(order_label.T), repeats=width, axis=0)
  vert_bar = np.repeat(cm(order_label), repeats=width, axis=1)

  final_img = np.zeros(shape=(n_samples + width, n_samples + width, distance_img.shape[2]),
                       dtype=distance_img.dtype)
  final_img[width:, width:] = distance_img
  final_img[:width, width:] = horz_bar
  final_img[width:, :width] = vert_bar
  assert np.sum(final_img[:width, :width]) == 0, \
  "Something wrong with my spacial coordination when writing this code!"
  # ====== plotting ====== #
  ax.imshow(final_img)
  ax.axis('off')
  # ====== legend ====== #
  if labels_name is not None and bool(legend_enable):
    cm = plt.get_cmap(label_colormap)
    labels_name = np.asarray(labels_name)
    if len(labels_name) == 2: # binary (easy peasy)
      all_colors = np.array((cm(np.min(labels)), cm(np.max(labels))))
    else: # multiple classes
      all_colors = cm(np.unique(labels))
    assert len(all_colors) == len(labels_name)
    legend_elements = [Line2D([0], [0], marker='o', color=color, label=name,
                              linewidth=0, linestyle=None, lw=0,
                              markerfacecolor=color, markersize=fontsize // 2)
                       for color, name in zip(all_colors, labels_name)]
    ax.legend(handles=legend_elements, markerscale=1.,
      scatterpoints=1, scatteryoffsets=[0.375, 0.5, 0.3125],
      loc=legend_loc, bbox_to_anchor=(0.5, -0.01), ncol=int(legend_ncol),
      columnspacing=float(legend_colspace), labelspacing=0.,
      fontsize=fontsize - 1, handletextpad=0.1)
  # ====== final configurations ====== #
  if title is not None:
    ax.set_title(str(title), fontsize=fontsize)
  if show_colorbar:
    plot_colorbar(colormap,
                  vmin=np.min(distance), vmax=np.max(distance),
                  ax=ax, orientation='vertical')
  return ax

def plot_latents(X, y, labels_name, title=None,
                 elev=None, azim=None, use_PCA=False,
                 ax=None, show_legend=True,
                 enable_argmax=True,
                 enable_separated=True):
  from matplotlib import pyplot as plt
  if title is None:
    title = ''
  title = '[%s]%s' % ("PCA" if use_PCA else "t-SNE", title)
  # ====== Downsample if the data is huge ====== #
  X, y = downsample_data(X, y)
  # ====== checking inputs ====== #
  assert X.ndim == 2, X.shape
  assert X.shape[0] == y.shape[0]
  num_classes = len(labels_name)
  # ====== preprocessing ====== #
  if X.shape[1] > 3:
    if not use_PCA:
      X = fast_tsne(X, n_components=2, perplexity=30.0,
                    learning_rate=200, n_iter=1000,
                    random_state=52181208, n_jobs=8)
    else:
      X = fast_pca(X, n_components=2, random_state=52181208)
  # ====== clustering metrics ====== #
  asw, ari, nmi, uca = clustering_scores(
      latent=X, labels=np.argmax(y, axis=-1) if y.ndim == 2 else y,
      n_labels=num_classes)
  title += '\nASW:%.2f ARI:%.2f NMI:%.2f UCA:%.2f' % (asw, ari, nmi, uca)
  # ====== plotting ====== #
  if enable_argmax:
    y_argmax = np.argmax(y, axis=-1)
    if ax is None:
      plot_figure(nrow=12, ncol=13)
    fast_scatter(x=X, y=y_argmax, labels=labels_name, ax=ax,
                 size=24 if len(X) > 500 else 32,
                 title=title,
                 enable_legend=bool(show_legend))
    plt.grid(False)
  # ====== plot each protein ====== #
  if enable_separated:
    colormap = 'Reds' # bwr
    ncol = 5 if num_classes <= 20 else 9
    nrow = int(np.ceil(num_classes / ncol))
    fig = plot_figure(nrow=4 * nrow, ncol=20)
    for i, lab in enumerate(labels_name):
      val = K.log_norm(y[:, i], axis=0)
      plot_scatter_heatmap(
          x=X[:, 0], y=X[:, 1],
          val=val / np.sum(val),
          ax=(nrow, ncol, i + 1),
          colormap=colormap, size=8, alpha=0.8,
          fontsize=8, grid=False,
          title=lab)
    # big title
    plt.suptitle(title)
    # show the colorbar
    import matplotlib as mpl
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cmap = mpl.cm.get_cmap(name=colormap)
    norm = mpl.colors.Normalize(vmin=0., vmax=1.)
    cb1 = mpl.colorbar.ColorbarBase(cbar_ax, cmap=cmap,
                                    norm=norm,
                                    orientation='vertical')
    cb1.set_label('Protein markers level')
