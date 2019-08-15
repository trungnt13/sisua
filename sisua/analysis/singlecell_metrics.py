from __future__ import absolute_import, division, print_function

from abc import ABCMeta, abstractmethod
from typing import List, Union

import numpy as np
import tensorflow as tf
from six import add_metaclass
from tensorflow.python.keras.callbacks import Callback
from tensorflow_probability.python.distributions import Distribution

from sisua.data import SingleCellOMICS
from sisua.models import SingleCellModel
from sisua.models.base import _to_sc_omics, _to_semisupervised_inputs

__all__ = ['SingleCellMetric', 'NegativeLogLikelihood']


@add_metaclass(ABCMeta)
class SingleCellMetric(Callback):

  def __init__(self,
               inputs: Union[SingleCellOMICS, List[SingleCellOMICS], np.
                             ndarray, List[np.ndarray], None] = None,
               extras=None,
               n_samples=1,
               batch_size=128,
               verbose=0,
               name=None,
               **kwargs):
    super(SingleCellMetric, self).__init__(**kwargs)
    self.n_samples = n_samples
    self.batch_size = batch_size
    self.inputs = inputs
    self.extras = extras
    self.verbose = verbose
    self._name = name

  @property
  def name(self):
    return self.__class__.__name__.lower() if self._name is None else self._name

  def set_model(self, model: SingleCellModel):
    assert isinstance(
        model, SingleCellModel), "This callback only support SingleCellModel"
    self.model = model
    return self

  @abstractmethod
  def call(self, y_true: List[Distribution], y_pred: List[SingleCellOMICS],
           latents: List[Distribution], extras, corruption_rate,
           corruption_dist):
    raise NotImplementedError

  def __call__(self, inputs=None, n_samples=None):
    if inputs is None:
      inputs = self.inputs
    if n_samples is None:
      n_samples = self.n_samples
    model = self.model

    outputs, latents = model.predict(inputs,
                                     n_samples=self.n_samples,
                                     batch_size=self.batch_size,
                                     verbose=self.verbose)
    if not isinstance(outputs, (tuple, list)):
      outputs = [outputs]
    if not isinstance(latents, (tuple, list)):
      latents = [latents]

    if model.is_semi_supervised:
      x, y, _ = _to_semisupervised_inputs(inputs, False)
      y_true = [x] + y
    else:
      x = inputs
      y = []
      y_true = [x] if not isinstance(x, (tuple, list)) else x

    metrics = self.call(y_true=[_to_sc_omics(i) for i in y_true],
                        y_pred=outputs,
                        latents=latents,
                        corruption_rate=model.corruption_rate,
                        corruption_dist=model.corruption_dist,
                        extras=self.extras)
    if metrics is None:
      metrics = {}
    elif tf.is_tensor(metrics) or isinstance(metrics, np.ndarray):
      metrics = {self.name: metrics}
    assert isinstance(metrics, dict), \
      "Return metrics must be a dictionary mapping metric name to scalar value"
    return metrics

  def on_epoch_end(self, epoch, logs=None):
    """Called at the end of an epoch.

    Subclasses should override for any actions to run. This function should only
    be called during TRAIN mode.

    Arguments:
        epoch: integer, index of epoch.
        logs: dict, metric results for this training epoch, and for the
          validation epoch if validation is performed. Validation result keys
          are prefixed with `val_`.
    """
    metrics = self()
    logs.update(metrics)


class NegativeLogLikelihood(SingleCellMetric):
  """ Log likelihood metric """

  def call(self, y_true: List[Distribution], y_pred: List[SingleCellOMICS],
           latents: List[Distribution], extras, corruption_rate,
           corruption_dist):
    llk = 0
    for t, p in zip(y_true, y_pred):
      llk += tf.reduce_mean(p.log_prob(t.X))
    return -llk
