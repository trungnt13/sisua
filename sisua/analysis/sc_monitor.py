from __future__ import absolute_import, division, print_function

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
from sisua.analysis.sc_metrics import (SingleCellMetric,
                                       _preprocess_output_distribution)
from sisua.data import SingleCellOMICS
from sisua.models import SingleCellModel
from sisua.models.base import _to_sc_omics, _to_semisupervised_inputs

__all__ = ['SingleCellMonitor', 'LearningCurves', 'LatentScatter']


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

  def call(self, y_true: List[SingleCellOMICS], y_crpt: List[SingleCellOMICS],
           y_pred: List[Distribution], latents: List[Distribution], extras):
    self.plot(y_true, y_crpt, y_pred, latents, self.model.history.history,
              extras)
    if len(self.get_figures()) > 0:
      self.save_figures(path=self.path,
                        dpi=self.dpi,
                        separate_files=False,
                        clear_figures=True)

  @abstractmethod
  def plot(self, y_true: List[SingleCellOMICS], y_crpt: List[SingleCellOMICS],
           y_pred: List[Distribution], latents: List[Distribution], history,
           extras):
    raise NotImplementedError


class LearningCurves(SingleCellMonitor):
  """ Additional key in the loss dictionary could be given by `extras` """

  def plot(self, y_true: List[SingleCellOMICS], y_crpt: List[SingleCellOMICS],
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

    fig = plt.figure(figsize=(8, 3))
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


class LatentScatter(SingleCellMonitor):

  def plot(self, y_true: List[SingleCellOMICS], y_crpt: List[SingleCellOMICS],
           y_pred: List[Distribution], latents: List[Distribution], history,
           extras):
    assert isinstance(extras, SingleCellOMICS), \
      "protein data must be provided as extras in form of SingleCellOMICS"

    labels = protein.X
    if 'X_prob' in protein.obsm:
      labels = protein.obsm['X_prob']
    elif 'X_bin' in protein.obsm:
      labels = protein.obsm['X_bin']
    if labels.ndim == 2:
      labels = np.argmax(labels, axis=1)
    elif labels.ndim > 2:
      raise RuntimeError("protein labels has %d dimensions, no support" %
                         labels.ndim)
    exit()
