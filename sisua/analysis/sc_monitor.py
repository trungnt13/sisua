from __future__ import absolute_import, division, print_function

import os
import string
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from numbers import Number
from typing import List, Union

import numpy as np
import seaborn as sns
import tensorflow as tf
from matplotlib import pyplot as plt
from six import add_metaclass, string_types
from tensorflow.python.keras.callbacks import Callback
from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python.distributions import Distribution

from odin.bay.distributions import ZeroInflated
from odin.visual import Visualizer
from sisua.analysis.latent_benchmarks import (plot_distance_heatmap,
                                              plot_latents_binary)
from sisua.analysis.sc_metrics import (SingleCellMetric,
                                       _preprocess_output_distribution,
                                       _to_binary)
from sisua.data import SingleCellOMIC
from sisua.data.utils import standardize_protein_name
from sisua.models import SingleCellModel
from sisua.models.base import _to_sc_omics

__all__ = ['SingleCellMonitor', 'LearningCurves', 'ScatterPlot', 'HeatmapPlot']


# ===========================================================================
# Base class
# ===========================================================================
@add_metaclass(ABCMeta)
class SingleCellMonitor(SingleCellMetric, Visualizer):

  def __init__(self, path, dpi=80, **kwargs):
    super(SingleCellMonitor, self).__init__(**kwargs)
    self._path = str(path)
    self.dpi = int(dpi)

  @property
  def path(self):
    return self._path

  def call(self, y_true: List[SingleCellOMIC], y_crpt: List[SingleCellOMIC],
           y_pred: List[Distribution], latents: List[Distribution], extras):
    self.plot(y_true, y_crpt, y_pred, latents, self.model.history.history,
              extras)
    if len(self.get_figures()) > 0:
      base, ext = os.path.splitext(self.path)
      path = base + '.%d' % (self._last_epoch + 1) + ext
      self.save_figures(path=path,
                        dpi=self.dpi,
                        separate_files=False,
                        clear_figures=True)

  @abstractmethod
  def plot(self, y_true: List[SingleCellOMIC], y_crpt: List[SingleCellOMIC],
           y_pred: List[Distribution], latents: List[Distribution], history,
           extras):
    raise NotImplementedError


class LearningCurves(SingleCellMonitor):
  """ Additional key in the loss dictionary could be given by `extras`,
  otherwise, plotting 'loss' and 'val_loss' by default.
  """

  def plot(self, y_true: List[SingleCellOMIC], y_crpt: List[SingleCellOMIC],
           y_pred: List[Distribution], latents: List[Distribution], history,
           extras):
    if extras is None:
      key = ['loss', 'val_loss']
    else:
      if isinstance(extras, string_types):
        extras = [extras]
      key = [str(i).lower() for i in extras]

    if len(key) == 0:
      return
    key = sorted([i for i in key if i in history])
    if len(key) == 0:
      return

    fig = plt.figure(figsize=(16, 4))
    for k in key:
      val = history[k]
      if k + '_epoch' in history:
        ids = history[k + '_epoch']
      else:
        ids = np.arange(len(val), dtype='int32')
      plt.plot(ids, val, label=k)
      plt.xticks(ids)
    plt.legend()
    plt.grid(True)
    self.add_figure('_'.join(key), fig)


class ScatterPlot(SingleCellMonitor):

  def __init__(self, path, data='latent', use_pca=True, dpi=80, **kwargs):
    super(ScatterPlot, self).__init__(path=path, dpi=dpi, **kwargs)
    self.use_pca = bool(use_pca)
    data = str(data.lower())
    assert data in ("latent", "imputation"), \
      "Only support three data types: latent, imputation and corrupt"
    self.data = data

  def _preprocess(self, y_true, y_pred, latents, extras):
    y_true = y_true[0]
    y_pred = [_preprocess_output_distribution(i) for i in y_pred]
    assert isinstance(extras, SingleCellOMIC), \
      "protein data must be provided as extras in form of SingleCellOMIC"
    protein = extras[y_true.indices]
    y_true.assert_matching_cells(protein)
    labels = _to_binary(protein)
    labels_name = [standardize_protein_name(i) for i in protein.var.iloc[:, 0]]

    if self.data == 'latent':
      data = latents
    elif self.data == 'imputation':
      data = y_pred
    else:
      raise NotImplementedError()
    return y_true, data, labels, labels_name

  def plot(self, y_true: List[SingleCellOMIC], y_crpt: List[SingleCellOMIC],
           y_pred: List[Distribution], latents: List[Distribution], history,
           extras):
    y_true, data, labels, labels_name = self._preprocess(
        y_true, y_pred, latents, extras)

    n = len(data)
    for idx, x in enumerate(data):
      distribution = x.__class__.__name__ \
        if not isinstance(x, tfd.Independent) else \
          x.distribution.__class__.__name__
      x = x.mean().numpy()
      if x.ndim == 3:  # in case MCMC sampling is used
        x = np.squeeze(x, axis=0)
      fig = plt.figure(figsize=(8, 8))
      plot_latents_binary(Z=x,
                          y=labels,
                          labels_name=labels_name,
                          title="[#%d]%s" % (idx, distribution),
                          use_PCA=self.use_pca,
                          ax=None,
                          show_legend=True,
                          size=12,
                          fontsize=12,
                          show_scores=False)
      self.add_figure("%s%d" % (self.data, idx), fig)


class HeatmapPlot(ScatterPlot):

  def plot(self, y_true: List[SingleCellOMIC], y_crpt: List[SingleCellOMIC],
           y_pred: List[Distribution], latents: List[Distribution], history,
           extras):
    y_true, data, labels, labels_name = self._preprocess(
        y_true, y_pred, latents, extras)

    n = len(data)
    for idx, x in enumerate(data):
      distribution = x.__class__.__name__ \
        if not isinstance(x, tfd.Independent) else \
          x.distribution.__class__.__name__
      x = x.mean().numpy()
      if x.ndim == 3:  # in case MCMC sampling is used
        x = np.squeeze(x, axis=0)
      fig = plt.figure(figsize=(8, 8))
      plot_distance_heatmap(X=x,
                            labels=labels,
                            labels_name=labels_name,
                            lognorm=True,
                            colormap='hot',
                            ax=None,
                            legend_enable=True,
                            legend_loc='upper center',
                            legend_ncol=3,
                            legend_colspace=0.2,
                            fontsize=10,
                            show_colorbar=True,
                            title="[#%d]%s" % (idx, distribution))
      self.add_figure("%s%d" % (self.data, idx), fig)
