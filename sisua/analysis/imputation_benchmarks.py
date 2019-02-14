from __future__ import print_function, division, absolute_import

from six import string_types
from collections import defaultdict, OrderedDict

import numpy as np
from scipy.stats import kde, entropy, spearmanr

import pandas as pd
from matplotlib import pyplot as plt
import seaborn
from sklearn.neighbors import KernelDensity

from odin.visual import to_axis, plot_figure
from odin.utils import catch_warnings_ignore, cache_memory, as_tuple
from odin import backend as K, visual

from sisua.data import get_dataset
from sisua.inference import Inference
from sisua.data.const import MARKER_GENES

# ===========================================================================
# Helpers
# ===========================================================================
def get_imputed_indices(x_org, x_imp):
  """ Return the indices of the cells which are imputed"""
  ids = []
  for i, (xo, xi) in enumerate(zip(x_org, x_imp)):
    if np.sum(xo) != np.sum(xi):
      ids.append(i)
  return np.array(ids)

class ProteinGeneAnalysis(object):
  """
  Parameters
  ----------
  data : dict
      mapping from
      'train'|'test' -> {'prot1/gene1':[x_org, x_drop, v, y],
                         'prot2/gene2':[x_org, x_drop, v, y], ...}
  """

  def __init__(self, data, infer):
    super(ProteinGeneAnalysis, self).__init__()
    self._data = data
    self._dataset = None

    protein_name = []
    gene_name = []
    try:
      for name, _ in self._data.items():
        for prot_gene, (x_org, x_drop, v, y) in _.items():
          prot, gene = prot_gene.split('/')
          if prot not in protein_name:
            protein_name.append(prot)
            gene_name.append(gene)
      self._protein_name = tuple(protein_name)
      self._gene_name = tuple(gene_name)
    except Exception:
      raise RuntimeError("Invalid data format!")
    self._name = '_'.join([infer.config['model_name'],
                           infer.config['xnorm'],
                           infer.config['ynorm'],
                           '%.2d' % (infer.supervised_percent * 100)])
    self._config = dict(infer.config)

  @property
  def dataset(self):
    return self._dataset

  @property
  def gene_ds(self):
    return get_dataset(self.dataset, override=False)[1]

  @property
  def prot_ds(self):
    return get_dataset(self.dataset, override=False)[2]

  @property
  def name(self):
    return self._name

  @property
  def protein_name(self):
    return self._protein_name

  @property
  def gene_name(self):
    return self._gene_name

  @property
  def spearman_train(self):
    return self.get_spearman_correlation(return_pvalue=False)[0]

  @property
  def spearman_test(self):
    return self.get_spearman_correlation(return_pvalue=False)[1]

  def __getitem__(self, key):
    if key in self.protein_name or key in self.gene_name:
      if key in self.protein_name:
        idx = self.protein_name.index(key)
        prot = key
        gene = self.gene_name[idx]
      else:
        idx = self.gene_name.index(key)
        prot = self.protein_name[idx]
        gene = key
      train = self._data['train'][prot + '/' + gene]
      test = self._data['test'][prot + '/' + gene]
      return train, test
    return self._config[key]

  @cache_memory
  def get_spearman_correlation(self, return_pvalue=False, substitute_nan=-1):
    """ Return: train results, test results
    if `return_pvalue=True`
      [(prot/gene, (org_corr, org_pvalue), (drop_corr, drop_pvalue), (imp_corr, imp_pvalue)),
        ...]
    else
      [(prot/gene, org_corr, drop_corr, imp_corr), ...]

    Note
    ----
    might return NaN correlation when stddev = 0
    """
    results = []
    _number = lambda x: x if substitute_nan is None else \
    (substitute_nan if np.isnan(x) else x)

    for name, _ in self._data.items():
      tmp = []
      for prot_gene, (x_org, x_drop, v, y) in _.items():
        with catch_warnings_ignore(RuntimeWarning):
          s1 = spearmanr(x_org, y)
          s2 = spearmanr(x_drop, y)
          s3 = spearmanr(v, y)
        if return_pvalue:
          tmp.append((prot_gene,
                      (_number(s1.correlation), _number(s1.pvalue)), # original
                      (_number(s2.correlation), _number(s2.pvalue)), # corrupted
                      (_number(s3.correlation), _number(s3.pvalue)), # imputed
                      ))
        else:
          tmp.append((prot_gene,
                      _number(s1.correlation),  # original
                      _number(s2.correlation),  # corrupted
                      _number(s3.correlation),  # imputed
                      ))
      results.append(tmp)
    return results[0], results[1]

  def plot_clusters(self):
    pass

  def plot_curves(self, protein=None, max_clip=3):
    import seaborn
    from matplotlib import pyplot as plt

    if protein is None:
      all_protein = self.protein_name
    else:
      all_protein = as_tuple(protein, t=string_types)
      assert all(p in self.protein_name for p in all_protein)
    assert len(all_protein) > 0
    n_protein = len(all_protein)

    norm01 = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))
    norm01 = lambda x: x

    train = {}
    test = {}
    for name, _ in self._data.items():
      curr = train if name == 'train' else test
      for prot_gene, (x_org, x_drop, v, y) in _.items():
        prot, gene = prot_gene.split('/')
        if prot in all_protein:
          scale = np.mean([i / j for i, j in zip(x_org, y)])
          curr[prot] = (x_org,
                        y * scale,
                        v)

    plot_figure(nrow=int(n_protein * 4), ncol=10)
    kde = dict(linewidth=1, kernel='gau', shade=False, gridsize=500)

    for i, name in enumerate(all_protein):
      i = i * 2 + 1
      trn = train[name]
      tst = test[name]

      plt.subplot(n_protein, 2, i)
      seaborn.kdeplot(trn[0], color='b', label='Original', **kde)
      seaborn.kdeplot(trn[1], color='g', label='Protein', **kde)
      seaborn.kdeplot(trn[2], color='r', label='Imputed', **kde)
      plt.xticks(); plt.xlabel('Relative Expression')
      plt.legend()
      plt.title("[Train]%s" % name)

      plt.subplot(n_protein, 2, i + 1)
      seaborn.kdeplot(tst[0], color='b', label='Original', **kde)
      seaborn.kdeplot(tst[1], color='g', label='Protein', **kde)
      seaborn.kdeplot(tst[2], color='r', label='Imputed', **kde)
      plt.xticks(); plt.xlabel('Relative Expression')
      plt.legend()
      plt.title("[Test]%s" % name)

    plt.tight_layout()
    plot_save()
    exit()

  @staticmethod
  def plot_boxplot(results, ax=None, is_train=False,
                   inc_original=True, inc_corrupted=False,
                   title=None):
    # TODO: show mean values here
    assert isinstance(results, (tuple, list)), \
    "results must be tuple or list"
    assert len(results) > 1, \
    "More than 1 system must be given"
    assert all(isinstance(i, ProteinGeneAnalysis) for i in results),\
    "All member in results must be instance of ProteinGeneAnalysis"

    ax = to_axis(ax)
    to_spearman = lambda r: r.spearman_train if bool(is_train) else r.spearman_test

    org = [r[1] for r in to_spearman(results[0])]
    drop = [r[2] for r in to_spearman(results[0])]

    name_list = [r['model_name'].split('_')[0] for r in results]
    results = [[i[3] for i in to_spearman(r)] for r in results]

    if inc_corrupted:
      name_list = ['Corrupted'] + name_list
      results = [drop] + results
    if inc_original:
      name_list = ['Original'] + name_list
      results = [org] + results

    assert len(results) == len(name_list)
    ax.boxplot(results, showfliers=False)
    # ax.set_ylim(-0.25, max_val + 0.1)
    ax.set_xticklabels(name_list, rotation=-30, fontsize=8)
    if title is not None:
      ax.set_title(str(title))
    return ax

  @staticmethod
  def get_marker_genes_analysis(infer=None, dataset_name=None,
                                V_train=None, V_test=None, protein_markers=None):
    """
    Return None if no gene-marker pair found
    """
    if dataset_name is None:
      dataset_name = infer.config['dataset']
    _, gene, prot = get_dataset(dataset_name)
    results = defaultdict(OrderedDict)

    ximpu = infer.config['ximpu']
    cdist = infer.config['cdist']
    protein_markers = prot.col_name.tolist()

    # mapping from protein index to gene index
    marker_indices = {}
    for prot_name, gene_name in MARKER_GENES.items():
      if prot_name in protein_markers:
        index = [i
                 for i, name in enumerate(gene.col_name)
                 if gene_name == name]
        if len(index) == 0: # still not found anything
          index = [i
                   for i, name in enumerate(gene.col_name)
                   if gene_name == name.split('_')[-1]]
        if len(index) == 1: # found
          marker_indices[protein_markers.index(prot_name)] = index[0]

    if len(marker_indices) > 0:
      X_train = gene.get_data('train', dropout=ximpu, distribution=cdist)
      y_train = prot.get_data('train', dropout=0)

      X_test = gene.get_data('test', dropout=ximpu, distribution=cdist)
      y_test = prot.get_data('test', dropout=0)

      X_train_org = gene.get_data('train', dropout=0)
      X_test_org = gene.get_data('test', dropout=0)

      V_train, V_test = infer.predict_V(X_train), infer.predict_V(X_test)
      if V_train is None:
        V_train, V_test = infer.predict_W(X_train), infer.predict_W(X_test)
      V_train, V_test = V_train[0], V_test[0]

      for name, Xorg, Xdrop, Vorg, Yorg in [
              ('train', X_train_org, X_train, V_train, y_train),
              ('test', X_test_org, X_test, V_test, y_test)]:
        for prot_idx, gene_idx in marker_indices.items():
          prot_name = prot.col_name[prot_idx]
          gene_name = gene.col_name[gene_idx]

          x_drop = Xdrop[:, gene_idx].ravel()
          x_org = Xorg[:, gene_idx].ravel()
          v = Vorg[:, gene_idx].ravel()
          y = Yorg[:, prot_idx].ravel()
          results[name][prot_name + '/' + gene_name] = (x_org, x_drop, v, y)
    # check return
    if len(results) > 0:
      results = ProteinGeneAnalysis(data=results, infer=infer)
      results._dataset = dataset_name
      return results
    return None

# ===========================================================================
# Metrics
# ===========================================================================
def imputation_score(original, imputed):
  # Median of medians for all distances
  assert original.shape == imputed.shape
  nonzeros = np.nonzero(original)
  d = np.abs(original - imputed) # [nonzeros]
  return np.median(d)

def imputation_mean_score(original, corrupted, imputed):
  # Mean of medians for each cell imputation score
  assert original.shape == corrupted.shape == imputed.shape
  imputation_cells = []
  for cell_org, cell_crt, cell_imp in zip(original, corrupted, imputed):
    if np.sum(cell_org) != np.sum(cell_crt):
      imputation_cells.append(
          np.median(np.abs(cell_org - cell_imp)))
  return np.mean(imputation_cells) if len(imputation_cells) > 0 else 0

def imputation_std_score(original, corrupted, imputed):
  # Standard deviation of medians for each cell imputation score
  assert original.shape == corrupted.shape == imputed.shape
  imputation_cells = []
  for cell_org, cell_crt, cell_imp in zip(original, corrupted, imputed):
    if np.sum(cell_org) != np.sum(cell_crt):
      imputation_cells.append(
          np.median(np.abs(cell_org - cell_imp)))
  return np.std(imputation_cells) if len(imputation_cells) > 0 else 0

# ===========================================================================
# Imputation analysis
# ===========================================================================
def plot_imputation_series(original, imputed, title="Imputation"):
  original = K.log_norm(original, axis=0)
  imputed = K.log_norm(imputed, axis=0)
  max_val = max(np.max(original),
                np.max(imputed))

  with catch_warnings_ignore(FutureWarning):
    grid = seaborn.pairplot(data=pd.DataFrame({'Original Value': original,
                                               'Imputed Value': imputed}),
        height=4, aspect=1,
        kind='reg',
        diag_kws={'bins': 180},
        plot_kws={'scatter_kws': dict(s=2, alpha=0.6),
                  'line_kws': dict(color='red', alpha=0.8),
                  'color': 'g'})
    ids = np.linspace(0, max_val)
    grid.axes[0, 1].set_xlim((0, max_val))
    grid.axes[0, 1].set_ylim((0, max_val))
    grid.axes[0, 1].plot(ids, ids, linestyle='--', linewidth=1, color='black')

    grid.axes[1, 0].set_xlim((0, max_val))
    grid.axes[1, 0].set_ylim((0, max_val))
    grid.axes[1, 0].plot(ids, ids, linestyle='--', linewidth=1, color='black')

def plot_marker_genes(Z, y, labels_name, title=None,
                      ax=None, show_legend=True,):
  from matplotlib import pyplot as plt
  if title is None:
    title = ''
  title = '[%s]%s' % ("PCA" if use_PCA else "t-SNE", title)
  # ====== Downsample if the data is huge ====== #
  Z, y = downsample_data(Z, y)
  # ====== checking inputs ====== #
  assert Z.ndim == 2, Z.shape
  assert Z.shape[0] == y.shape[0]
  num_classes = len(labels_name)
  # ====== preprocessing ====== #
  if Z.shape[1] > 3:
    if not use_PCA:
      Z = fast_tsne(Z, n_components=2, perplexity=30.0,
                    learning_rate=200, n_iter=1000,
                    random_state=52181208, n_jobs=8)
    else:
      Z = fast_pca(Z, n_components=2, random_state=52181208)
  protein_list = [i for i in labels_name if i in MARKER_GENES]
  print(protein_list)
  exit()

def plot_imputation(original, imputed, corrupted=None,
                    kde_kernel='gaussian', ax=None,
                    title="Imputation"):
  """ Original code: scVI
      Modified by: SISUA

  kde_kernel : string (default: 'linear')
    'gaussian', 'tophat', 'epanechnikov', 'exponential', 'linear',
    'cosine'

  """
  y = imputed
  x = original
  assert imputed.shape == original.shape
  if corrupted is not None:
    assert original.shape == corrupted.shape

  # this modification focus only on missing data points
  if corrupted is not None:
    mask = np.where(original != corrupted, True, False)
    x = x[mask]
    y = y[mask]

  ymax = 25 # increasing ymax for more data points
  #
  mask = x < ymax
  x = x[mask]
  y = y[mask]
  #
  mask = y < ymax
  x = x[mask]
  y = y[mask]
  #
  l_minimum = np.minimum(x.shape[0], y.shape[0])
  x = x[:l_minimum]
  y = y[:l_minimum]

  data = np.vstack([x, y])

  axes = visual.to_axis(ax)
  axes.set_xlim([0, ymax])
  axes.set_ylim([0, ymax])

  nbins = 80

  # Evaluate a gaussian kde on a regular grid of nbins x nbins over data extents
  xi, yi = np.mgrid[0:ymax:nbins * 1j, 0:ymax:nbins * 1j]
  # ====== scipy ====== #
  k_ = kde.gaussian_kde(data)
  zi = k_(np.vstack([xi.flatten(), yi.flatten()]))
  # ====== sklearn ====== #
  # k_ = KernelDensity(kernel=str(kde_kernel))
  # k_.fit(data.T)
  # zi = k_.score_samples(
  #     np.vstack([xi.flatten(), yi.flatten()]).T).T

  plt.title(title, fontsize=12)
  plt.ylabel("Imputed counts")
  plt.xlabel('Original counts')

  plt.pcolormesh(yi, xi, zi.reshape(xi.shape), cmap="Reds")

  a, _, _, _ = np.linalg.lstsq(y[:, np.newaxis], x, rcond=-1)
  linspace = np.linspace(0, ymax)
  plt.plot(linspace, a * linspace, color='black')

  plt.plot(linspace, linspace, color='black', linestyle=":")
