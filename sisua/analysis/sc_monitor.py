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

  def __init__(self,
               path='/tmp/{model:s}_{name:s}.pdf',
               dpi=80,
               name=None,
               **kwargs):
    super(SingleCellMonitor, self).__init__(**kwargs)
    self._name = name
    self._org_path = path
    self._path = None
    self.extras = extras
    self.freq = int(freq)
    self.dpi = int(dpi)
    self.inputs = None
    assert self.freq > 0

  @property
  def path(self):
    return self._path

  @property
  def name(self):
    return self.__class__.__name__.lower() if self._name is None else self._name

  def set_model(self, model: SingleCellModel):
    kw = {'name': self.name, 'model': model.__class__.id}
    fmt = {}
    for (_, key, spec, _) in string.Formatter().parse(self._org_path):
      if spec is not None:
        fmt[key] = ''
    if len(fmt) > 0:
      fmt = {i: kw[i] if i in kw else j for i, j in fmt.items()}
      self._path = self._org_path.format(**fmt)
    return super(SingleCellMonitor, self).set_model(model)

  def call(self, y_true: List[SingleCellOMICS], y_crpt: List[SingleCellOMICS],
           y_pred: List[Distribution], latents: List[Distribution], extras):
    self.plot(y_true, y_crpt, y_pred, latents, history, extras)
    if len(self.get_figures()) > 0:
      self.save_figures(path=self.path, dpi=self.dpi, separate_files=False)

  @abstractmethod
  def plot(self, y_true: List[SingleCellOMICS], y_crpt: List[SingleCellOMICS],
           y_pred: List[Distribution], latents: List[Distribution], history,
           extras):
    raise NotImplementedError


class LearningCurves(SingleCellMonitor):

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

    fig = plt.figure()
    for i in key:
      plt.plot(history[i], label=i)
    plt.legend()
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
