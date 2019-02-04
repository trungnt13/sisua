from __future__ import print_function, division, absolute_import

import numpy as np
from scipy.stats import kde, entropy
import pandas as pd

from matplotlib import pyplot as plt
import seaborn

from odin.utils import catch_warnings_ignore
from odin import backend as K, visual

from sklearn.neighbors import KernelDensity

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

# ===========================================================================
# Metrics
# ===========================================================================
def imputation_score(original, imputed):
  # Median of medians for all distances
  assert original.shape == imputed.shape
  return np.median(np.abs(original - imputed))

def imputation_mean_score(original, corrupted, imputed):
  # Mean of medians for each cell
  assert original.shape == corrupted.shape == imputed.shape
  imputation_cells = []
  for cell_org, cell_crt, cell_imp in zip(original, corrupted, imputed):
    if np.sum(cell_org) != np.sum(cell_crt):
      imputation_cells.append(
          np.median(np.abs(cell_org - cell_imp)))
  return np.mean(imputation_cells) if len(imputation_cells) > 0 else 0

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
