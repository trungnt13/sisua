from __future__ import print_function, division, absolute_import
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import matplotlib.mlab as mlab

import os
os.environ['ODIN'] = 'gpu,float32,seed=5218'
from collections import defaultdict
import pickle
import warnings

import numpy as np
from scipy import stats
import pandas as pd
import seaborn as sbn

from odin import nnet as N, backend as K, training as T, visual as V
from odin.visual import (print_dist, merge_text_graph, plot_confusion_matrix,
                         plot_figure, plot_histogram_layers, plot_save,
                         generate_random_colors, plot_histogram)
from odin.utils import (unique_labels, ctext, auto_logging, batching, UnitTimer,
                        ArgController, get_script_path, mpi, Progbar,
                        catch_warnings_ignore)
from odin.stats import describe

with warnings.catch_warnings():
  warnings.filterwarnings('ignore', category=ImportWarning)
  from sklearn.mixture import GaussianMixture
  from sklearn.neighbors import KernelDensity
  from sklearn.base import DensityMixin, BaseEstimator

from sisua.data import get_dataset, EXP_DIR
from sisua.data.utils import standardize_protein_name

# ===========================================================================
# LabelThresholding class
# ===========================================================================
class GMMThresholding(BaseEstimator, DensityMixin):

  """ GMMThresholding

  Parameters
  ----------

  """

  def __init__(self, n_components_per_class=2,
               input_normalization='log', quartile_alpha=1.,
               ci_threshold=-0.68, random_state=5218,
               verbose=False):
    super(GMMThresholding, self).__init__()
    self.n_components_per_class = int(n_components_per_class)
    self.input_normalization = str(input_normalization).lower()
    assert self.input_normalization in ('log', 'raw'), \
    "Only support 'raw' or 'log'-normalization method"

    ci_threshold = float(ci_threshold)
    assert 0 <= np.abs(ci_threshold) <= 1
    self.ci_threshold = ci_threshold

    self.quartile_alpha = float(quartile_alpha)
    self.verbose = bool(verbose)
    self.random_state = random_state
    self._model = []

  def fit(self, X):
    num_classes = X.shape[1]
    for i in range(num_classes):
      x_org = X[:, i]
      assert np.all(x_org >= 0), "Only support non-negative values"
      # ====== normalizing ====== #
      x_low, x_high, x_sum = None, None, None
      if self.input_normalization == 'raw':
        x_train, x_low, x_high = clipping_quartile(x_org,
          alpha=self.quartile_alpha)
      elif self.input_normalization == 'log':
        x_sum = np.sum(x_org)
        x_train = log_norm(x_org, x_sum)
      # ====== GMM ====== #
      gmm = GaussianMixture(n_components=self.n_components_per_class,
                            covariance_type='diag',
                            init_params='kmeans',
                            n_init=8, max_iter=120,
                            random_state=self.random_state)
      gmm.fit(x_train[:, np.newaxis])
      # ====== save GMM ====== #
      means_ = gmm.means_.ravel()
      order = np.argsort(means_)
      self._model.append((x_low, x_high, x_sum, order, gmm))

  def fit_transform(self, X):
    self.fit(X)
    return self.predict_proba(X)

  def _predict(self, X, threshold):
    assert X.shape[1] == len(self._model), "Number classes mis-match"
    y = []
    for i, (x_low, x_high, x_sum,
            order, gmm) in enumerate(self._model):
      x_org = X[:, i]
      if self.input_normalization == 'raw':
        x_test = np.clip(x_org,
                         a_min=x_low, a_max=x_high)
      elif self.input_normalization == 'log':
        x_test = log_norm(x_org, x_sum)

      # binary thresholding
      if threshold is not None:
        ci = stats.norm.interval(np.abs(threshold),
            loc=gmm.means_[order[-1]],
            scale=np.sqrt(1 / gmm.precisions_[order[-1]]))
        x_out = (x_test >= (ci[0] if threshold < 0 else ci[1])).astype('float32')
        x_out = x_out[:, np.newaxis]
      # probablizing
      else:
        probas = gmm.predict_proba(x_test[:, np.newaxis]).T[order][-1]
        x_out = probas[:, np.newaxis]
      y.append(x_out)
    return np.concatenate(y, axis=1)

  def predict(self, X):
    return self._predict(X, threshold=self.ci_threshold)

  def predict_proba(self, X):
    return self._predict(X, threshold=None)

  def score_samples(self, X):
    """Compute the weighted log probabilities for each sample.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        List of n_features-dimensional data points. Each row
        corresponds to a single data point.

    Returns
    -------
    log_prob : array, shape (n_samples,)
        Log probabilities of each data point in X.
    """
    raise NotImplementedError

  def score(self, X, y=None):
    """Compute the per-sample average log-likelihood of the given data X.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_dimensions)
        List of n_features-dimensional data points. Each row
        corresponds to a single data point.

    Returns
    -------
    log_likelihood : float
        Log likelihood of the Gaussian mixture given X.
    """
    return self.score_samples(X).mean()

# ===========================================================================
# Different methods for thresholding
# ===========================================================================
def clipping_percentile(x, low=5, high=95, return_indices=False):
  x = x.astype('int32')
  if return_indices:
    high = np.percentile(x, q=high)
    low = np.percentile(x, q=low)
    return x < low, x > high
  else:
    x = x[x < np.percentile(x, q=high)]
    x = x[x > np.percentile(x, q=low)]
    return x

def clipping_quartile(x, alpha=1.5):
  # result wider range threshold, keep more data points,
  # lower LLK
  x = x.astype('int32')
  Q1 = np.percentile(x, q=25)
  Q3 = np.percentile(x, q=75)
  IQR = Q3 - Q1

  high = Q3 + alpha * IQR
  low = Q1 - alpha * IQR
  return x[np.logical_and(Q1 - alpha * IQR <= x,
                          x <= Q3 + alpha * IQR)], \
  low, high

def log_norm(x, x_sum=None, scale_factor=10000):
  x = x.astype('float64')
  if x_sum is None:
    x_sum = np.sum(x)
  return np.log1p(
      x /
      (x_sum + np.finfo(x.dtype).eps) *
      scale_factor)

def gmm_thresholding(prot, method, threshold, prot_names,
                     num_dist=2, visualization=False):
  num_prot = len(prot_names)
  num_bins = 200
  quart_alpha = 1.
  prot_binary = []
  prot_probas = []
  nrow, ncol = 9, num_prot
  if visualization:
    V.plot_figure(nrow=nrow * 2, ncol=int(ncol * 2), dpi=180)

  for i in range(num_prot):
    # ====== preprocessing ====== #
    name = prot_names[i]
    x_org = prot[:, i]
    # ====== GMM ====== #
    # data normalization
    if method == 'raw':
      x_train, x_low, x_high = clipping_quartile(x_org, alpha=quart_alpha)
    elif method == 'log':
      x_sum = np.sum(x_org)
      x_train = log_norm(x_org, x_sum)
    # fit the GMM
    gmm = GaussianMixture(n_components=num_dist,
                          covariance_type='diag',
                          init_params='kmeans',
                          n_init=8, max_iter=120,
                          random_state=5218)
    gmm.fit(x_train[:, np.newaxis])
    # ====== sort mean and precision in increasing order ====== #
    means_ = gmm.means_.ravel()
    precision_ = gmm.precisions_.ravel()
    idx = np.argsort(means_)
    means_ = means_[idx]
    precision_ = precision_[idx]
    # ====== making prediction ====== #
    if method == 'raw':
      x_test = np.clip(x_org,
                       a_min=x_low, a_max=x_high)
    elif method == 'log':
      x_test = log_norm(x_org, x_sum)
    # probabilities
    probas = gmm.predict_proba(x_test[:, np.newaxis]).T[idx][1]
    # raw probability value
    x_probas = probas
    prot_probas.append(x_probas[:, np.newaxis])
    # binary
    ci = stats.norm.interval(threshold,
      loc=means_[1], scale=np.sqrt(1 / precision_)[1])
    x_binary = (x_test > ci[0]).astype('float32')
    prot_binary.append(x_binary[:, np.newaxis])
    # ====== skip visualization ====== #
    if not visualization:
      continue

    # ====== ploting hist ====== #
    def draw_hist(ax, show_title=True, show_yticks=True):
      count, bins = plot_histogram(
          x=x_train, bins=num_bins, ax=ax,
          normalize=False, kde=False, range_0_1=False,
          covariance_factor=0.25, centerlize=False,
          fontsize=8, title=name if show_title else None)
      plt.xlim((np.min(x_train), np.max(x_train)))
      plt.xticks(np.linspace(start=np.min(x_train), stop=np.max(x_train),
                             num=5, dtype='int32'),
                 fontsize=6)
      if show_yticks:
        plt.yticks(np.linspace(start=np.min(count), stop=np.max(count),
                               num=8, dtype='int32'),
                   fontsize=5)
      else:
        plt.yticks([], [])
      return count, bins
    # ====== ploting histogram ====== #
    count, bins = draw_hist(ax=(nrow, ncol, i + 1),
                            show_title=True,
                            show_yticks=True)

    _X = np.linspace(start=np.min(x_train), stop=np.max(x_train),
                     num=800)
    _Y = []
    for m, p in zip(means_, precision_):
      with catch_warnings_ignore(Warning):
        _ = mlab.normpdf(_X, m, np.sqrt(1 / p))
      _Y.append(_)
    _min = np.min(_Y)
    _max = np.max(_Y)
    for _y in _Y:
      _y = (_y - _min) / (_max - _min) # normalize to 0 - 1
      plt.plot(_X, _y * np.max(count))
    if i == 0:
      plt.ylabel("Hist + PDF")
    # ====== plot GMM probabilities ====== #
    probas = gmm.predict_proba(_X[:, np.newaxis]).T[idx]
    plt.subplot(nrow, ncol, ncol + i + 1)
    for p in probas:
      plt.plot(p, linestyle='--', linewidth=1)
    plt.plot(np.mean(probas, axis=0), linestyle='-', linewidth=1.1)
    plt.xlim((0, len(p)))
    plt.xticks(np.linspace(start=0, stop=len(p), num=5),
               np.linspace(start=np.min(x_train), stop=np.max(x_train),
                           num=5, dtype='int32'),
               fontsize=6)
    plt.yticks(fontsize=5)
    if i == 0:
      plt.ylabel("Response")
    # ====== plot GMM pdf ====== #
    scores = gmm.score_samples(_X[:, np.newaxis])
    scores = np.max(scores) - scores
    plt.subplot(nrow, ncol, ncol * 2 + i + 1)
    plt.plot(scores)
    plt.xlim((0, len(p)))
    plt.xticks(np.linspace(start=0, stop=len(p), num=5),
               np.linspace(start=np.min(x_train), stop=np.max(x_train),
                           num=5, dtype='int32'),
               fontsize=6)
    plt.yticks([], [])
    if i == 0:
      plt.ylabel("Uncertainty")

    # ====== response thresholding ====== #
    def response_threshold(ax, threshold):
      count, bins = draw_hist(ax=ax, show_title=False, show_yticks=False)
      bins = bins[:-1]
      probas = gmm.predict_proba(bins[:, np.newaxis]).T[idx]
      y1 = probas[1]
      plt.fill_between(bins[y1 >= threshold],
                       y1=0, y2=np.max(count), facecolor='red', alpha=0.3)
      if i == 0:
        plt.ylabel("Response >= %.2f" % threshold)

    response_threshold(ax=(nrow, ncol, ncol * 3 + i + 1), threshold=0.5)
    response_threshold(ax=(nrow, ncol, ncol * 4 + i + 1), threshold=0.25)
    response_threshold(ax=(nrow, ncol, ncol * 5 + i + 1), threshold=0.1)

    # ====== confident interval ====== #
    def threshold_confident_interval(ax, threshold):
      count, bins = draw_hist(ax=ax, show_title=False, show_yticks=False)
      m = means_[1]
      s = np.sqrt(1 / precision_)[1]
      plt.fill_between(bins[bins >= stats.norm.interval(threshold, loc=m, scale=s)[0]],
                       y1=0, y2=np.max(count), facecolor='orange', alpha=0.4)
      if i == 0:
        plt.ylabel("%.2f CI" % threshold)

    threshold_confident_interval(ax=(nrow, ncol, ncol * 6 + i + 1), threshold=0.)
    threshold_confident_interval(ax=(nrow, ncol, ncol * 7 + i + 1), threshold=0.68)
    threshold_confident_interval(ax=(nrow, ncol, ncol * 8 + i + 1), threshold=0.95)
  # ====== return ====== #
  return np.concatenate(prot_binary, axis=1), \
  np.concatenate(prot_probas, axis=1)

# ===========================================================================
# Testing
# ===========================================================================
def visualize_boxplot(prot, prot_names, method):
  num_prot = len(prot_names)
  V.plot_figure(nrow=3, ncol=num_prot + 2)
  for i, (x, name) in enumerate(zip(prot.T, prot_names)):
    plt.subplot(1, num_prot, i + 1)
    plt.boxplot(x if method == 'raw' else K.log_norm(x, axis=0),
                whis=1.5, labels=[name],
                flierprops={'marker': '.', 'markersize': 3},
                showmeans=True, meanline=True)
  plt.tight_layout()

def visualize_gmm(prot, prot_names, method):
  num_prot = len(prot_names)
  n_row = int(np.ceil(num_prot / 2))
  prog = Progbar(target=4 * num_prot, print_report=False, print_summary=False,
                 name='GMM')
  for num_dist in (1, 2, 3, 4):
    plot_figure(nrow=n_row * 2.5, ncol=12)
    llk = []
    bic = []
    for i in range(num_prot):
      # ====== preprocessing ====== #
      x_org = prot[:, i]
      # x = clipping_percentile(x, low=5, high=95)
      if method == 'raw':
        x_norm, x_low, x_high = clipping_quartile(x_org, alpha=1.)
      elif method == 'log':
        x_norm = log_norm(x_org)
      x_norm = x_norm[:, np.newaxis]
      # ====== GMM ====== #
      gmm = GaussianMixture(n_components=num_dist, covariance_type='diag',
                            init_params='kmeans', n_init=8, max_iter=100)
      gmm.fit(x_norm)
      # ====== plotting ====== #
      name = prot_names[i]
      count, bins = plot_histogram(x=x_norm, bins=100,
                                   ax=(n_row, 2, i + 1),
                                   normalize=False, kde=False, range_0_1=False,
                                   covariance_factor=0.25, centerlize=False,
                                   fontsize=8, title=name)
      # ====== plot GMM ====== #
      x_ = np.linspace(np.min(bins), np.max(bins), 1200)
      y_ = gmm.predict_proba(x_[:, np.newaxis]) * np.max(count)
      for i in range(num_dist):
        plt.plot(x_, y_[:, i], linestyle='--', linewidth=1.2)

      # pdf
      y_ = np.exp(gmm.score_samples(x_[:, np.newaxis]))
      y_ = (y_ - np.min(y_)) / (np.max(y_) - np.min(y_)) * np.max(count)
      plt.plot(x_, y_, color='red', linestyle='-', linewidth=1.6)

      # threshold
      threshold = gmm.means_.min() - 0.5 * np.sqrt(1.0 / gmm.precisions_[gmm.means_.argmin(), 0])
      ids = np.where(x_ > threshold, True, False)
      plt.fill_between(x_[ids], y1=0, y2=y_[ids], facecolor='red', alpha=0.3)

      # score
      score_llk = gmm.score(x_norm)
      llk.append(score_llk)
      score_bic = gmm.bic(x_norm)
      bic.append(score_bic)
      plt.title('%s Data:%s LLK:%.2f BIC:%.2f' %
                (name, str(x_norm.shape), score_llk, score_bic),
                fontsize=10)
      # ====== add progress ====== #
      prog.add(1)
    # ====== print some log ====== #
    plt.suptitle("#GMM:%d LLK:%.2f BIC:%.2f" % (num_dist, np.mean(llk), np.mean(bic)),
                 fontsize=12)
    plt.tight_layout()

def visualize_kde(prot, prot_names):
  num_prot = len(prot_names)
  n_row = int(np.ceil(num_prot / 2))
  prog = Progbar(target=6 * num_prot, print_report=False, print_summary=False,
                 name='KDE')
  for kernel in ('gaussian', 'tophat', 'epanechnikov', 'exponential', 'linear', 'cosine'):
    plot_figure(nrow=n_row * 2.5, ncol=12)
    llk = defaultdict(list)
    for i in range(num_prot):
      prog.add(1)
      # ====== preprocessing ====== #
      x = prot[:, i]
      # x = clipping_percentile(x, low=5, high=95)
      x = clipping_quartile(x, alpha=1.5)
      mean = np.mean(x)
      std = np.std(x)
      y = (x - mean) / std
      y = y[:, np.newaxis]
      # ====== ploting ====== #
      name = prot_names[i]
      count, bins = plot_histogram(x=x, bins=100, ax=(n_row, 2, i + 1),
                                   normalize=False, kde=False, range_0_1=False,
                                   covariance_factor=0.25, centerlize=False,
                                   fontsize=8, title=name)
      # ====== plot GMM ====== #
      x_ = np.linspace(np.min(bins), np.max(bins), 1200)
      x_normed = (x_[:, np.newaxis] - mean) / std
      for bandwidth in (0.25, 0.75, 1.):
        kde = KernelDensity(kernel=kernel, bandwidth=bandwidth)
        kde.fit(y)
        y_ = np.exp(kde.score_samples(x_normed))
        y_ = (y_ - np.min(y_)) / (np.max(y_) - np.min(y_)) * np.max(count)
        plt.plot(x_, y_, linestyle='-', linewidth=1.2, label='%.2f' % bandwidth)
        # score
        score_llk = kde.score(x_normed)
        llk[bandwidth].append(score_llk)
    # ====== print some log ====== #
    llk_text = '  '.join(['%.2f:%.2f' % (i, np.mean(j))
                         for i, j in sorted(llk.items(), key=lambda x: x[0])])
    plt.figlegend(loc='lower center', ncol=3)
    plt.suptitle("[%s] LLK:%s" % (kernel, llk_text), fontsize=12)
    plt.tight_layout()

# ===========================================================================
# Main
# ===========================================================================
if __name__ == '__main__':
  args = ArgController(
  ).add("dataset", "Name of the dataset"
  ).add("-norm", "method for normalizing: raw, log", 'log', ('log', 'raw')
  ).parse()

  ds, gene_ds, prot_ds = get_dataset(args.dataset, override=False)
  ds_name = ds.name
  print("Start label thresholding for:", ctext(ds_name, 'lightyellow'))
  path = ds.path
  # ====== protein.count ====== #
  y_prot_names = np.array([standardize_protein_name(i) for i in ds['y_col']])
  y_prot = ds['y']
  print("Protein labels:", ctext(y_prot_names, 'cyan'))
  print("Protein matrix:", ctext(y_prot.shape, 'cyan'))
  # Show some distribution for fun
  num_protein = len(y_prot_names)
  # ====== already binarized ====== #
  if len(np.unique(y_prot)) == 2:
    print(ctext("y is already binarized!", 'lightyellow'))
    path_binary = os.path.join(path, 'y_bin')
    print("Save binary:", ctext(path_binary, 'cyan'))
    with open(path_binary, 'wb') as f:
      pickle.dump(y_prot, f)
    exit()
  # ====== visualizing ====== #
  # visualize_boxplot(prot=y_prot, prot_names=y_prot_names,
  #                   method=args.norm)
  visualize_gmm(prot=y_prot, prot_names=y_prot_names,
                method=args.norm)
  # visualize_kde(prot=y_prot, prot_names=y_prot_names)
  # ====== save all figure ====== #
  y_bin, y_prob = gmm_thresholding(prot=y_prot,
                                   prot_names=y_prot_names,
                                   method=args.norm,
                                   threshold=0.68, num_dist=2,
                                   visualization=True)
  print("Thresholded values:")
  print("  Original     :", ctext(y_prot.shape, 'cyan'), ctext(describe(y_prot, shorten=True), 'lightcyan'))
  print("  Binarized    :", ctext(y_bin.shape, 'cyan'), ctext(describe(y_bin, shorten=True), 'lightcyan'))
  print("  Probabilities:", ctext(y_prob.shape, 'cyan'), ctext(describe(y_prob, shorten=True), 'lightcyan'))

  test = GMMThresholding(n_components_per_class=2,
                         input_normalization=args.norm,
                         ci_threshold=-0.68)
  test.fit(y_prot)
  y_bin, y_prob = test.predict(y_prot), test.predict_proba(y_prot)
  print("Thresholded values:")
  print("  Original     :", ctext(y_prot.shape, 'cyan'), ctext(describe(y_prot, shorten=True), 'lightcyan'))
  print("  Binarized    :", ctext(y_bin.shape, 'cyan'), ctext(describe(y_bin, shorten=True), 'lightcyan'))
  print("  Probabilities:", ctext(y_prob.shape, 'cyan'), ctext(describe(y_prob, shorten=True), 'lightcyan'))
  # ====== Visualize labels distribution after thresholding ====== #
  V.plot_figure(nrow=4, ncol=12)
  normalize_to_01 = lambda x: x / np.sum(x)
  dist_raw = normalize_to_01(np.sum(y_prot, axis=0))
  dist_bin = normalize_to_01(np.sum(y_bin, axis=0))
  dist_prob = normalize_to_01(np.sum(y_prob, axis=0))

  x = np.arange(num_protein)
  bar1 = plt.bar(x, dist_raw, width=0.2, color='r')
  bar2 = plt.bar(x + 0.2, dist_bin, width=0.2, color='g')
  bar3 = plt.bar(x + 0.4, dist_prob, width=0.2, color='b')
  ax = plt.gca()
  ax.set_xticks(x + 0.3)
  ax.set_xticklabels(y_prot_names)
  plt.legend([bar1, bar2, bar3],
             ['Original', 'Binarized', 'Probabilized'])

  # ====== Visualize results ====== #
  V.plot_figure(5, ncol=18)
  factor = 10

  plt.subplot(2, 1, 1)
  tmp = y_bin[:1000]
  tmp = np.concatenate([np.vstack([y[np.newaxis, :]] * factor) for y in tmp.T], axis=0)
  plt.imshow(tmp,
             cmap=plt.cm.get_cmap('binary'), interpolation='None')
  plt.xticks([], [])
  plt.yticks(np.arange(0, tmp.shape[0], step=factor) + factor // 2,
             y_prot_names)

  plt.subplot(2, 1, 2)
  tmp = y_prob[:1000]
  tmp = np.concatenate([np.vstack([y[np.newaxis, :]] * factor) for y in tmp.T], axis=0)
  plt.imshow(tmp,
             cmap=plt.cm.get_cmap('Oranges'), interpolation='None')
  plt.xticks([], [])
  plt.yticks(np.arange(0, tmp.shape[0], step=factor) + factor // 2,
             y_prot_names)

  plt.suptitle("Comparing binarized(Top) and probabilized(Bottom) labels")
  # ====== saving data ====== #
  path_binary = os.path.join(path, 'y_bin')
  print("Save binary:", ctext(path_binary, 'cyan'))
  with open(path_binary, 'wb') as f:
    pickle.dump(y_bin, f)

  path_prob = os.path.join(path, 'y_prob')
  print("Save probabilities:", ctext(path_prob, 'cyan'))
  with open(path_prob, 'wb') as f:
    pickle.dump(y_prob, f)

  plot_save(
      os.path.join(EXP_DIR, '%s_%s_label_threshold.pdf' %
        (ds_name, args.norm)),
      dpi=180)
