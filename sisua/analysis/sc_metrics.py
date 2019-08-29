from __future__ import absolute_import, division, print_function

from abc import ABCMeta, abstractmethod
from collections import defaultdict
from numbers import Number
from typing import List, Union

import numpy as np
import tensorflow as tf
from six import add_metaclass
from tensorflow.python.keras.callbacks import Callback
from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python.distributions import Distribution

from odin.bay.distributions import ZeroInflated
from odin.utils import catch_warnings_ignore
from sisua.analysis.imputation_benchmarks import (correlation_scores,
                                                  imputation_mean_score,
                                                  imputation_score,
                                                  imputation_std_score)
from sisua.analysis.latent_benchmarks import clustering_scores
from sisua.data import SingleCellOMIC
from sisua.models import SingleCellModel
from sisua.models.base import _to_sc_omics

__all__ = [
    'SingleCellMetric', 'NegativeLogLikelihood', 'ImputationError',
    'CorrelationScores', 'ClusteringScores'
]


def _preprocess_output_distribution(y_pred):
  """ In case of zero inflated distribution, extract the underlying count
  distribution """
  if isinstance(y_pred, tfd.Independent) and \
    isinstance(y_pred.distribution, ZeroInflated):
    y_pred = tfd.Independent(
        y_pred.distribution.count_distribution,
        reinterpreted_batch_ndims=y_pred.reinterpreted_batch_ndims)
  return y_pred


def _to_binary(protein):
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
  return labels


_CORRUPTED_INPUTS = {}


# ===========================================================================
# Base class
# ===========================================================================
@add_metaclass(ABCMeta)
class SingleCellMetric(Callback):
  """ Single cell metrics for evaluating the imputation and latent space
  during training

  Parameters
  ----------
  inputs : {`SingleCellOMIC`, `numpy.ndarray`}
  extras : None
    extras object (e.g. protein) used for calculating the metric
  n_samples : `int` (default=`1`)
    number of MCMC samples for prediction
  batch_size : `int` (default=`64`)
  freq : `int` (default=`3`)
    frequency of evaluating the metric, some metrics are very computational
    intensive and could slow down the training progress significantly
  """

  def __init__(self,
               inputs: Union[SingleCellOMIC, List[SingleCellOMIC], np.
                             ndarray, List[np.ndarray], None] = None,
               extras=None,
               n_samples=1,
               batch_size=64,
               freq=3,
               name=None,
               **kwargs):
    super(SingleCellMetric, self).__init__(**kwargs)
    self.n_samples = n_samples
    self.batch_size = batch_size
    self.inputs = inputs
    self.extras = extras
    self.freq = int(freq)
    self._name = name
    # store the last epoch that the metric was calculated
    self._last_epoch = 0
    assert self.freq > 0

  @property
  def name(self):
    return self.__class__.__name__.lower() if self._name is None else self._name

  def set_model(self, model: SingleCellModel):
    assert isinstance(
        model, SingleCellModel), "This callback only support SingleCellModel"
    self.model = model
    return self

  @abstractmethod
  def call(self, y_true: List[SingleCellOMIC], y_crpt: List[SingleCellOMIC],
           y_pred: List[Distribution], latents: List[Distribution], extras):
    raise NotImplementedError

  def __call__(self, inputs=None, n_samples=None):
    if inputs is None:
      inputs = self.inputs
    if n_samples is None:
      n_samples = self.n_samples
    model = self.model

    if not isinstance(inputs, (tuple, list)):
      inputs = [inputs]
    inputs = [_to_sc_omics(i) for i in inputs]
    if model.corruption_rate is not None:
      corruption_text = str(model.corruption_dist) + str(model.corruption_rate)
      inputs_corrupt = [
          (data.corrupt(corruption_rate=model.corruption_rate,
                        corruption_dist=model.corruption_dist,
                        inplace=False) \
           if str(id(data)) + corruption_text not in _CORRUPTED_INPUTS else
           _CORRUPTED_INPUTS[str(id(data)) + corruption_text]) \
             if idx == 0 else data
          for idx, data in enumerate(inputs)
      ]
      _CORRUPTED_INPUTS[str(id(inputs[0])) +
                        corruption_text] = inputs_corrupt[0]
    else:
      inputs_corrupt = inputs

    outputs, latents = model.predict(inputs_corrupt,
                                     n_samples=self.n_samples,
                                     batch_size=self.batch_size,
                                     verbose=0,
                                     apply_corruption=False)
    if not isinstance(outputs, (tuple, list)):
      outputs = [outputs]
    if not isinstance(latents, (tuple, list)):
      latents = [latents]

    metrics = self.call(y_true=inputs,
                        y_pred=outputs,
                        y_crpt=inputs_corrupt,
                        latents=latents,
                        extras=self.extras)
    if metrics is None:
      metrics = {}
    elif tf.is_tensor(metrics) or \
      isinstance(metrics, np.ndarray) or \
        isinstance(metrics, Number):
      metrics = {self.name: metrics}
    assert isinstance(metrics, dict), \
      "Return metrics must be a dictionary mapping metric name to scalar value"
    metrics = {
        i: j.numpy() if tf.is_tensor(j) else j for i, j in metrics.items()
    }
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
    if epoch % self.freq == 0 and logs is not None:
      self._last_epoch = epoch
      # calculating the metric
      try:
        metrics = self()
      except Exception as e:
        print("Error:", e)
        metrics = {}
      # update the log
      for key, val in metrics.items():
        logs[key] = val
        logs[key + '_epoch'] = epoch

  def on_train_end(self, logs=None):
    if self.model.epochs != self._last_epoch:
      self._last_epoch = self.model.epochs
      # calculating the metric
      try:
        metrics = self()
      except Exception as e:
        print("Error:", e)
        metrics = {}
      # update the log
      history = self.model.history.history
      for key, val in metrics.items():
        if key in history:
          history[key].append(val)
          history[key + '_epoch'].append(self._last_epoch)


# ===========================================================================
# Losses
# ===========================================================================
class NegativeLogLikelihood(SingleCellMetric):
  """ Log likelihood metric

  Parameters
  ----------
  inputs : {`SingleCellOMIC`, `numpy.ndarray`}
  extras : None
    extras object (e.g. protein) used for calculating the metric
  n_samples : `int` (default=`1`)
    number of MCMC samples for prediction
  batch_size : `int` (default=`64`)
  freq : `int` (default=`3`)
    frequency of evaluating the metric, some metrics are very computational
    intensive and could slow down the training progress significantly

  Returns
  -------
  dict:
    'nllk%d' for each tuple of input and output
  """

  def call(self, y_true: List[SingleCellOMIC], y_crpt: List[SingleCellOMIC],
           y_pred: List[Distribution], latents: List[Distribution], extras):
    nllk = {}
    for idx, (t, p) in enumerate(zip(y_true, y_pred)):
      nllk['nllk%d' % idx] = -tf.reduce_mean(p.log_prob(t.X))
    return nllk


class ImputationError(SingleCellMetric):
  """ Imputation error

  Parameters
  ----------
  inputs : {`SingleCellOMIC`, `numpy.ndarray`}
  extras : None
    extras object (e.g. protein) used for calculating the metric
  n_samples : `int` (default=`1`)
    number of MCMC samples for prediction
  batch_size : `int` (default=`64`)
  freq : `int` (default=`3`)
    frequency of evaluating the metric, some metrics are very computational
    intensive and could slow down the training progress significantly

  Return
  ------
  dict :
    'imp_med'
    'imp_mean'
  """

  def call(self, y_true: List[SingleCellOMIC], y_crpt: List[SingleCellOMIC],
           y_pred: List[Distribution], latents: List[Distribution], extras):
    # only care about the first data input
    y_true = y_true[0]
    y_crpt = y_crpt[0]
    y_pred = y_pred[0]

    y_pred = _preprocess_output_distribution(y_pred)
    y_pred = y_pred.mean()
    if y_pred.shape.ndims == 3:
      y_pred = tf.reduce_mean(y_pred, axis=0)
    return {
        'imp_med':
            imputation_score(original=y_true.X, imputed=y_pred),
        'imp_mean':
            imputation_mean_score(original=y_true.X,
                                  corrupted=y_crpt.X,
                                  imputed=y_pred)
    }


class CorrelationScores(SingleCellMetric):
  """ (1 - correlation_coefficients) to represent the loss

  Parameters
  ----------
  inputs : {`SingleCellOMIC`, `numpy.ndarray`}
  extras : {`SingleCellOMIC`, `numpy.ndarray`}
    the protein array
  n_samples : `int` (default=`1`)
    number of MCMC samples for prediction
  batch_size : `int` (default=`64`)
  freq : `int` (default=`3`)
    frequency of evaluating the metric, some metrics are very computational
    intensive and could slow down the training progress significantly

  Returns
  -------
  dict :
    'pearson_mean': np.mean(pearson),
    'spearman_mean': np.mean(spearman),
    'pearson_med': np.median(pearson),
    'spearman_med': np.median(spearman),

  Example
  -------
  >>> CorrelationScores(extras=y_train, freq=1)

  """

  def call(self, y_true: List[SingleCellOMIC], y_crpt: List[SingleCellOMIC],
           y_pred: List[Distribution], latents: List[Distribution], extras):
    y_true = y_true[0]
    y_crpt = y_crpt[0]
    y_pred = y_pred[0]
    assert isinstance(extras, SingleCellOMIC), \
      "protein data must be provided as extras in form of SingleCellOMIC"
    protein = extras[y_true.indices]
    y_true.assert_matching_cells(protein)

    y_pred = _preprocess_output_distribution(y_pred)
    y_pred = y_pred.mean()
    if y_pred.shape.ndims == 3:
      y_pred = tf.reduce_mean(y_pred, axis=0)

    scores = correlation_scores(X=y_pred,
                                y=protein.X,
                                gene_name=y_true.var['geneid'],
                                protein_name=protein.var['protid'],
                                return_series=False)
    if len(scores) == 0:
      return {}
    spearman = []
    pearson = []
    for _, (s, p) in scores.items():
      spearman.append(-s)
      pearson.append(-p)
    return {
        'pearson_mean': np.mean(pearson),
        'spearman_mean': np.mean(spearman),
        'pearson_med': np.median(pearson),
        'spearman_med': np.median(spearman),
    }


class ClusteringScores(SingleCellMetric):
  """
  Parameters
  ----------
  inputs : {`SingleCellOMIC`, `numpy.ndarray`}
  extras : {`SingleCellOMIC`, `numpy.ndarray`}
    the protein array
  n_samples : `int` (default=`1`)
    number of MCMC samples for prediction
  batch_size : `int` (default=`64`)
  freq : `int` (default=`3`)
    frequency of evaluating the metric, some metrics are very computational
    intensive and could slow down the training progress significantly

  Returns
  -------
  dict :
    silhouette_score (higher is better, best is 1, worst is -1)
    adjusted_rand_score (higher is better)
    normalized_mutual_info_score (higher is better)
    unsupervised_clustering_accuracy (higher is better)

  Example
  -------
  >>> ClusteringScores(extras=y_train, freq=1)
  """

  def call(self, y_true: List[SingleCellOMIC], y_crpt: List[SingleCellOMIC],
           y_pred: List[Distribution], latents: List[Distribution], extras):
    y_true = y_true[0]
    y_crpt = y_crpt[0]
    y_pred = y_pred[0]
    assert isinstance(extras, SingleCellOMIC), \
      "protein data must be provided as extras in form of SingleCellOMIC"
    protein = extras[y_true.indices]
    y_true.assert_matching_cells(protein)
    labels = _to_binary(protein)

    scores = {}
    scores_avg = defaultdict(list)
    # support multiple latents also
    for idx, z in enumerate(latents):
      for key, val in clustering_scores(latent=z.mean().numpy(),
                                        labels=labels,
                                        n_labels=protein.var.shape[0]).items():
        # since all score higher is better, we want them as loss value
        val = -val
        scores['%s_%d' % (key, idx)] = val
        scores_avg[key].append(val)
    # average scores
    scores.update({i: np.mean(j) for i, j in scores_avg.items()})
    return scores
