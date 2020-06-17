from __future__ import absolute_import, division, print_function

import inspect
import multiprocessing as mpi
import os
import pickle
import string
from abc import ABCMeta, abstractmethod, abstractproperty
from collections import OrderedDict, defaultdict
from functools import partial
from numbers import Number
from typing import Iterable, List, Text, Union

import numpy as np
import tensorflow as tf
from six import add_metaclass, string_types
from tensorflow import keras
from tensorflow.python.data import Dataset
from tensorflow.python.data.ops.dataset_ops import DatasetV2
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
from odin.bay.vi import BetaVAE
from odin.networks import NetworkConfig
from odin.utils import (cache_memory, catch_warnings_ignore, classproperty,
                        is_primitive)
from odin.visual import Visualizer
from sisua.analysis.posterior import Posterior
from sisua.data import OMIC, SingleCellOMIC, get_dataset

__all__ = [
    'SingleCellModel', 'NetworkConfig', 'RandomVariable', 'interpolation'
]


def _to_data(x, batch_size=64) -> Dataset:
  if isinstance(x, SingleCellOMIC):
    inputs = x.create_dataset(batch_size=batch_size)
  elif isinstance(x, DatasetV2):
    inputs = x
  # given numpy ndarrays
  else:
    x = tf.nest.flatten(x)
    inputs = SingleCellOMIC(x[0])
    if len(x) > 1:
      omics = list(OMIC)
      # we don't know what is the omic of data anyway, random assigning it
      for arr, om_random in zip(x[1:], omics[1:]):
        inputs.add_omic(omic=om_random, X=arr)
    inputs = inputs.create_dataset(inputs.omics,
                                   batch_size=batch_size,
                                   drop_remainder=True)
  return inputs


# ===========================================================================
# SingleCell model
# ===========================================================================
class SingleCellModel(BetaVAE, Visualizer):
  r"""
  Note:
    It is recommend to call `tensorflow.random.set_seed` for reproducible
    results.
  """

  def __init__(
      self,
      outputs: RandomVariable,
      latents: RandomVariable = RandomVariable(10, 'diag', True, 'Latents'),
      encoder: NetworkConfig = NetworkConfig([64, 64],
                                             batchnorm=True,
                                             input_dropout=0.3),
      decoder: NetworkConfig = NetworkConfig([64, 64], batchnorm=True),
      analytic=True,
      log_norm=True,
      beta=1.0,
      name=None,
      **kwargs,
  ):
    super().__init__(outputs=outputs,
                     latents=latents,
                     encoder=encoder,
                     decoder=decoder,
                     beta=beta,
                     name=name,
                     reduce_latent=kwargs.pop('reduce_latent', 'concat'),
                     input_shape=kwargs.pop('input_shape', None),
                     step=kwargs.pop('step', 0.),
                     path=kwargs.pop('path', None))
    self._analytic = bool(analytic)
    self._log_norm = bool(log_norm)
    self.dataset = None
    self.metadata = dict()
    self._n_inputs = max(len(l.inputs) for l in tf.nest.flatten(self.encoder))

  def set_metadata(self, sco: SingleCellOMIC):
    assert isinstance(sco, SingleCellOMIC), \
      f"sco must be instance of SingleCellOMIC but given: {type(sco)}"
    self.dataset = sco.name
    for om in sco.omics:
      self.metadata[om.name] = sco.get_var_names(om)
    return self

  @property
  def log_norm(self):
    return self._log_norm

  @property
  def is_zero_inflated(self):
    return self.posteriors[0].is_zero_inflated

  def encode(self,
             inputs,
             library=None,
             training=None,
             mask=None,
             sample_shape=(),
             **kwargs):
    if self.log_norm:
      if tf.is_tensor(inputs):
        inputs = tf.math.log1p(inputs)
      else:
        inputs = tf.nest.flatten(inputs)
        inputs[0] = tf.math.log1p(inputs[0])
    # just limit the number of inputs
    if isinstance(inputs, (tuple, list)):
      inputs = inputs[:self._n_inputs]
    return super().encode(inputs=inputs,
                          training=training,
                          mask=mask,
                          sample_shape=sample_shape,
                          **kwargs)

  def decode(self,
             latents,
             training=None,
             mask=None,
             sample_shape=(),
             **kwargs):
    return super().decode(latents=latents,
                          training=training,
                          mask=mask,
                          sample_shape=sample_shape,
                          **kwargs)

  def predict(self,
              inputs,
              sample_shape=(),
              batch_size=32,
              verbose=True,
              device="GPU"):
    r""" Predict on minibatches then return a single distribution by
    concatenation

    Return:
      X : `Distribution` or tuple of `Distribution`
        output distribution, multiple distribution is return in case of
        multiple outputs
      Z : `Distribution` or tuple of `Distribution`
        latent distribution, multiple distribution is return in case of
        multiple latents
    """
    assert device in ("CPU", "GPU"), \
      f"Only support device CPU or GPU, but given: {device}"
    inputs = _to_data(inputs, batch_size=batch_size)
    ## making predictions
    X, Z = [], []
    prog = tqdm(inputs, desc="Predicting", disable=not bool(verbose))
    with tf.device(f"/{device}:0"):
      for data in prog:
        pX_Z, qZ_X = self(**data, training=False, sample_shape=sample_shape)
        X.append(pX_Z)
        Z.append(qZ_X)
      prog.clear()
      prog.close()
    # merging the batch distributions
    if isinstance(pX_Z, (tuple, list)):
      merging_axis = 0 if pX_Z[0].batch_shape.ndims == 1 else 1
    else:
      merging_axis = 0 if pX_Z.batch_shape.ndims == 1 else 1
    with tf.device("/CPU:0"):
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

  def fit(self,
          train: Union[SingleCellOMIC, DatasetV2],
          valid: Union[SingleCellOMIC, DatasetV2] = None,
          metadata: SingleCellOMIC = None,
          analytic=None,
          **kwargs):
    r""" This fit function is the combination of both
    `Model.compile` and `Model.fit` """
    if analytic is None:
      analytic = self._analytic
    ## preprocessing the data
    if isinstance(train, SingleCellOMIC):
      self.set_metadata(train)
    elif isinstance(valid, SingleCellOMIC):
      self.set_metadata(valid)
    elif isinstance(metadata, SingleCellOMIC):
      self.set_metadata(metadata)
    if self.dataset is None or len(self.metadata) == 0:
      raise RuntimeError(
          "First time call `fit`, set the 'metadata' argument to a "
          "SingleCellOMIC dataset to keep the dataset name and OMICs' "
          "variables description.")
    batch_size = kwargs.pop('batch_size', 64)
    train = _to_data(train, batch_size=batch_size)
    if valid is not None:
      valid = _to_data(valid, batch_size=batch_size)
    return super().fit(train=train, valid=valid, analytic=analytic, **kwargs)

  @classproperty
  def id(cls):
    class_name = cls.__name__
    name = ''
    for i in class_name:
      if i.isupper():
        name += i
    return name.lower()

  def create_posterior(self,
                       dropout_rate=0.2,
                       retain_rate=0.2,
                       corrupt_distribution='binomial',
                       batch_size=8,
                       sample_shape=10,
                       reduce_latents=lambda *Zs: tf.concat(Zs, axis=1),
                       verbose=True,
                       train_percent=0.8,
                       random_state=1) -> Posterior:
    if not self.is_fitted:
      raise RuntimeError("fit() must be called before creating Posterior.")
    if isinstance(train_percent, Number):
      if not self.dataset is None:
        raise ValueError("set_metadata() to track the fitted dataset.")
      ds = get_dataset(self.dataset)
      _, test = ds.split(train_percent=train_percent, seed=random_state)
    elif isinstance(train_percent, SingleCellOMIC):
      test = train_percent
    else:
      raise ValueError(
          "train_percent can be a number or SingleCellOMIC "
          f"but given {type(train_percent)}"
      )
    return Posterior(scm=self,
                     sco=test,
                     dropout_rate=dropout_rate,
                     retain_rate=retain_rate,
                     corrupt_distribution=corrupt_distribution,
                     batch_size=batch_size,
                     sample_shape=sample_shape,
                     reduce_latents=reduce_latents,
                     verbose=verbose,
                     name=f"{self.id}_{self.dataset}",
                     random_state=random_state)

  def load_weights(self, filepath, raise_notfound=False, verbose=False):
    r""" Load all the saved weights in tensorflow format at given path """
    super().load_weights(filepath, raise_notfound, verbose)
    metamodel_path = f"{filepath}.metamodel"
    if os.path.exists(metamodel_path):
      with open(metamodel_path, 'rb') as f:
        class_name, dataset, metadata, kwargs = pickle.load(f)
      assert class_name == self.__class__.__name__
      self.dataset = dataset
      self.metadata = metadata
    return self

  def save_weights(self, filepath, overwrite=True):
    r""" Just copy this function here to fix the `save_format` to 'tf'
    Since saving 'h5' will drop certain variables.
    """
    super().save_weights(filepath, overwrite)
    class_name = self.__class__.__name__
    dataset = self.dataset
    metadata = self.metadata
    kwargs = dict(self.init_args)
    with open(f'{filepath}.metamodel', 'wb') as f:
      pickle.dump([class_name, dataset, metadata, kwargs], f)
    return self
