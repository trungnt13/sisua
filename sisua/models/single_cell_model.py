from __future__ import absolute_import, division, print_function

import inspect
import multiprocessing as mpi
import os
import string
from abc import ABCMeta, abstractmethod, abstractproperty
from collections import OrderedDict, defaultdict
from functools import partial
from typing import Iterable, List, Text, Union

import numpy as np
import tensorflow as tf
from six import add_metaclass, string_types
from tensorflow import keras
from tensorflow.python.data import Dataset
from tensorflow.python.keras.callbacks import (Callback, CallbackList,
                                               LambdaCallback,
                                               LearningRateScheduler,
                                               ModelCheckpoint)
from tensorflow.python.platform import tf_logging as logging
from tqdm import tqdm

from odin.backend import interpolation
from odin.backend.keras_callbacks import EarlyStopping
from odin.backend.keras_helpers import layer2text
from odin.bay import RandomVariable
from odin.bay.distributions import concat_distribution
from odin.bay.vi import VariationalAutoencoder
from odin.networks import NetworkConfig
from odin.utils import (cache_memory, catch_warnings_ignore, classproperty,
                        is_primitive)
from odin.visual import Visualizer
from sisua.data import OMIC, SingleCellOMIC

__all__ = [
  'SingleCellModel'
]

# ===========================================================================
# SingleCell model
# ===========================================================================
class SingleCellModel(VariationalAutoencoder, Visualizer):
  r"""
  Note:
    It is recommend to call `tensorflow.random.set_seed` for reproducible
    results.
  """

  def __init__(
      self,
      outputs: RandomVariable,
      latents: RandomVariable = RandomVariable(10, 'diag', True, 'Latents'),
      encoder: NetworkConfig = NetworkConfig(units=[64, 64], network='dense'),
      decoder: NetworkConfig = NetworkConfig(units=[64, 64], network='dense'),
      kl_interpolate=interpolation.const(vmax=1),
      analytic=True,
      log_norm=True,
      name=None,
  ):
    super().__init__(outputs=outputs,
                     latents=latents,
                     encoder=encoder,
                     decoder=decoder,
                     name=name)
    # parameters for ELBO
    self._analytic = bool(analytic)
    assert isinstance(kl_interpolate, interpolation.Interpolation), \
      'kl_interpolate must be instance of odin.backend.interpolation.Interpolation,' + \
        'but given type: %s' % str(type(kl_interpolate))
    if kl_interpolate.norm is None and kl_interpolate != interpolation.const:
      raise ValueError("interpolation normalization constant (i.e. 'norm') "
                       "must be provided.")
    self._kl_interpolate = kl_interpolate
    self._log_norm = bool(log_norm)
    self._dataset = None

  @property
  def fitted_dataset(self):
    r""" Return the name of the last SingleCellOMIC dataset fitted on """
    if not self.is_fitted:
      return None
    return self._fit_history[-1]['inputs']

  @property
  def corruption_rate(self):
    if len(self.fit_history) == 0:
      return 0.
    return max(i['corruption_rate'] for i in self.fit_history)

  @property
  def corruption_dist(self):
    if len(self.fit_history) == 0:
      return 'binomial'
    return list(set(i['corruption_dist'] for i in self.fit_history))[-1]

  @property
  def log_norm(self):
    return self._log_norm

  @property
  def is_zero_inflated(self):
    return self.posteriors[0].is_zero_inflated

  @classproperty
  def is_multiple_outputs(self):
    r""" Return true if __init__ contains both 'rna_dim' and 'adt_dim'
    as arguments """
    args = inspect.getfullargspec(self.__init__).args
    return 'rna_dim' in args and 'adt_dim' in args

  @property
  def epochs(self):
    return int(self._epochs.numpy())

  @property
  def analytic(self):
    return self._analytic

  @property
  def kl_weight(self):
    return self._kl_interpolate(self.step)

  def encode(self,
             inputs,
             library=None,
             training=None,
             mask=None,
             sample_shape=()):
    return super().encode(inputs=inputs,
                          training=training,
                          mask=mask,
                          sample_shape=sample_shape)

  def decode(self, latents, training=None, mask=None, sample_shape=()):
    return super().decode(latents=latents,
                          training=training,
                          mask=mask,
                          sample_shape=sample_shape)

  def test(self,
           inputs,
           library=None,
           mask=None,
           sample_shape=(),
           training=None):
    # target : [batch_size, n_genes], target for reconstruction of gene-expression
    #   matrix (can be different from `x`)
    # mask : [batch_size, 1] binary mask for semi-supervised training
    inputs, lmean, lvar, target, y, mask = self.prepare_inputs(inputs)
    # postprocessing the returns
    qZ = self.encode(inputs,
                     lmean,
                     lvar,
                     y,
                     training=training,
                     sample_shape=sample_shape)
    pX = self.decode(qZ, training=training)
    # Add Log-likelihood, don't forget to apply mask for semi-supervised loss
    # we have to use the log_prob function from DenseDistribution class, not
    # the one in pX
    llk_x = self.posteriors[0].log_prob(target)
    llk_y = tf.convert_to_tensor(0, dtype=inputs.dtype)
    track_llky = []
    for i_true, m, post in zip(y, mask, self.posteriors[1:]):
      llk = post.log_prob(i_true) * m
      track_llky.append((llk, post.name.lower()))
      llk_y += llk
    # calculating the KL
    kl = tf.convert_to_tensor(0., dtype=inputs.dtype)
    track_kl = []
    for idx, qz in enumerate(tf.nest.flatten(qZ)):
      div = qz.KL_divergence(analytic=self.analytic, sample_shape=self._kl_mcmc)
      # add mcmc dimension if used close-form KL
      if tf.rank(div) > 0 and self.analytic:
        div = tf.expand_dims(div, axis=0)
        if sample_shape > 1:
          div = tf.tile(div, [sample_shape] + [1] * (div.shape.ndims - 1))
      track_kl.append((div, 'z%s' % qz.event_shape[0]))
      kl += div
    # Final ELBO
    beta = self.kl_weight if training else 1.
    self.add_metric(beta, 'mean', 'beta')
    elbo = (llk_x + llk_y) - tf.expand_dims(kl, axis=-1) * beta
    elbo = tf.reduce_logsumexp(elbo, axis=0)
    loss = tf.reduce_mean(-elbo)
    # add loss and metrics
    self.add_loss(
        tf.cond(training,
                true_fn=lambda: loss,
                false_fn=lambda: tf.stop_gradient(loss)))
    self.add_metric(llk_x, 'mean', "llk_%s" % self.posteriors[0].name.lower())
    for llk_val, name in track_llky:
      self.add_metric(llk_val, 'mean', "llk_%s" % name)
    for kl_val, name in track_kl:
      self.add_metric(kl_val, 'mean', "klqp_%s" % name)
    # post-processing the return
    if not self.is_semi_supervised and isinstance(pX, (tuple, list)):
      pX = pX[0]
    return pX, qZ

  def predict(self,
              inputs,
              sample_shape=1,
              batch_size=64,
              apply_corruption=False,
              verbose=1):
    r"""
    Arguments:
      apply_corruption : `bool` (default=`False`)
        if `True` applying corruption on data before prediction to match the
        condition during fitting.

    Return:
      X : `Distribution` or tuple of `Distribution`
        output distribution, multiple distribution is return in case of
        multiple outputs
      Z : `Distribution` or tuple of `Distribution`
        latent distribution, multiple distribution is return in case of
        multiple latents
    """
    inputs = _to_sco(inputs, self.omic_outputs)
    ## applying corruption for testing
    if apply_corruption and self.corruption_rate is not None:
      inputs = inputs.corrupt(dropout_rate=self.corruption_rate,
                              distribution=self.corruption_dist,
                              inplace=False)
    n = inputs.n_obs
    data = _to_tfdata(sco=inputs,
                      rvs=self.omic_outputs,
                      mask=None,
                      batch_size=batch_size,
                      epochs=1,
                      training=False)
    ## making predictions
    X, Z = [], []
    for inputs in tqdm(data,
                       desc="Predicting",
                       total=int(np.ceil(n / batch_size)),
                       disable=not bool(verbose)):
      # the _to_tfddata will return (x, y) tuple for `fit` methods,
      # y=None and we only need x here.
      processed = self.prepare_inputs(inputs[0], without_target=True)
      z = self.encode(*processed,
                      training=False,
                      sample_shape=int(sample_shape))
      x = self.decode(z, training=False)
      # post-processing the return
      if not self.is_semi_supervised and isinstance(x, (tuple, list)):
        x = x[0]
      X.append(x)
      Z.append(z)
    # merging the batch distributions
    if isinstance(x, (tuple, list)):
      merging_axis = 0 if x[0].batch_shape.ndims == 1 else 1
    else:
      merging_axis = 0 if x.batch_shape.ndims == 1 else 1
    # multiple outputs
    if isinstance(X[0], (tuple, list)):
      X = tuple([
          concat_distribution([x[idx] for x in X], \
                              axis=merging_axis,
                              name=self.posteriors[idx].name)
          for idx in range(len(X[0]))
      ])
    # single output
    else:
      X = concat_distribution(X,
                              axis=merging_axis,
                              name=self.posteriors[0].name)
    # multiple latents
    if isinstance(Z[0], (tuple, list)):
      Z = tuple([
          concat_distribution([z[idx]
                               for z in Z], axis=0)
          for idx in range(len(Z[0]))
      ])
    else:
      Z = concat_distribution(Z, axis=0)
    return X, Z

  def fit(
      self,
      train: Union[SingleCellOMIC, Dataset],
      valid: Union[SingleCellOMIC, Dataset] = None,
      valid_freq=500,
      valid_interval=0,
      optimizer='adam',
      learning_rate=1e-3,
      clipnorm=None,
      epochs=-1,
      max_iter=1000,
      sample_shape=(),  # for ELBO
      analytic=False,  # for ELBO
      callback=None,
      compile_graph=True,
      autograph=False,
      logging_interval=2,
      skip_fitted=False,
      log_path=None,
      earlystop_threshold=0.001,
      earlystop_progress_length=0,
      earlystop_patience=-1,
      earlystop_min_epoch=-np.inf,
      terminate_on_nan=True,
      checkpoint=None,
      allow_rollback=False,
      allow_none_gradients=False):
    r""" This fit function is the combination of both
    `Model.compile` and `Model.fit` """
    assert isinstance(train, (SingleCellOMIC, Dataset))
    if valid is not None:
      assert isinstance(valid, (SingleCellOMIC, Dataset))
    ## preprocessing the data
    if isinstance(train, SingleCellOMIC):
      self._dataset = train.name
      train = train.create_dataset()
    if isinstance(valid, SingleCellOMIC):
      self._dataset = valid.name
      valid = valid.create_dataset()
    ## call fit
    kw = locals()
    del kw['self']
    args = inspect.getfullargspec(super().fit).args
    for k in list(kw.keys()):
      if k not in args:
        del kw[k]
    return super().fit(**kw)

  @classproperty
  def id(cls):
    class_name = cls.__name__
    name = ''
    for i in class_name:
      if i.isupper():
        name += i
    return name.lower()
