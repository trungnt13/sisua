from __future__ import absolute_import, division, print_function

from abc import ABCMeta, abstractmethod, abstractproperty
from functools import partial
from typing import Iterable, List, Text, Union

import numpy as np
import tensorflow as tf
from six import add_metaclass, string_types
from tensorflow.python.keras import Model
from tensorflow.python.keras.callbacks import (Callback, CallbackList,
                                               LambdaCallback)

from sisua.data import SingleCellOMICS


# ===========================================================================
# Helper
# ===========================================================================
def _to_sc_omics(x):
  if isinstance(x, SingleCellOMICS):
    return x
  return SingleCellOMICS(x)


# ===========================================================================
# SingleCell model
# ===========================================================================
@add_metaclass(ABCMeta)
class SingleCellModel(Model):
  """
  """

  def __init__(self,
               kl_analytic=True,
               kl_weight=1.,
               kl_warmup=400,
               supervised=0.8,
               seed=8,
               name=None):
    super(SingleCellModel, self).__init__(name=name)
    self._n_epoch = tf.Variable(0, trainable=False, dtype='float32')
    # parameters for ELBO
    self._kl_analytic = bool(kl_analytic)
    self._kl_weight = tf.convert_to_tensor(kl_weight, dtype='float32')
    self._kl_warmup = tf.convert_to_tensor(kl_warmup, dtype='float32')
    self._seed = int(seed)
    self._is_fitting = False

  @abstractproperty
  def is_semi_supervised(self):
    raise NotImplementedError

  @abstractmethod
  def call(self, inputs, training=None, n_samples=1):
    raise NotImplementedError

  @property
  def is_fitting(self):
    return self._is_fitting

  @property
  def n_epoch(self):
    return int(tf.cast(self._n_epoch, 'int32'))

  @property
  def kl_analytic(self):
    return self._kl_analytic

  @property
  def kl_warmup(self):
    return self._kl_warmup

  @property
  def kl_weight(self):
    warmup_weight = tf.minimum(
        tf.maximum(self._n_epoch, 1.) / self.kl_warmup, 1.)
    return warmup_weight * self._kl_weight

  @property
  def seed(self):
    return self._seed

  def _to_semisupervised_inputs(self, inputs):
    """ Preprocessing the inputs for semi-supervised training
    """
    # objective mask is provided
    if not isinstance(inputs, Iterable):
      inputs = [inputs]
    n = tf.shape(inputs[0])[0]
    for i in inputs:
      tf.assert_equal(tf.shape(i)[0], n)

    # mask is given at the end during training
    if self.is_fitting:
      x = inputs[0]
      y = inputs[1:-1]
      masks = [inputs[-1]] * len(y)
    # no mask is provided
    else:
      x = inputs[0]
      y = inputs[1:]
      masks = [1.] * len(y)
    return x, y, masks

  def fit(self,
          inputs: Union[SingleCellOMICS, Iterable[SingleCellOMICS]],
          optimizer: Union[Text, tf.optimizers.Optimizer] = 'adam',
          learning_rate=1e-4,
          n_mcmc_samples=1,
          supervised_percent=0.8,
          corruption_rate=0.25,
          corruption_dist='binomial',
          batch_size=64,
          epochs=1,
          callbacks=None,
          validation_split=0.1,
          validation_freq=1,
          shuffle=True,
          verbose=1):
    """ This fit function is the combination of both
    `Model.compile` and `Model.fit` """
    assert 0.0 <= supervised_percent <= 1.0
    if validation_split <= 0:
      raise ValueError("validation_split must > 0")
    # prepare optimizer
    if isinstance(optimizer, string_types):
      optimizer = tf.optimizers.get(optimizer)

    if isinstance(optimizer, tf.optimizers.Optimizer):
      pass
    elif issubclass(optimizer, tf.optimizers.Optimizer):
      optimizer = optimizer(learning_rate=learning_rate)

    # prepare input data
    if not isinstance(inputs, Iterable):
      inputs = [inputs]
    inputs = [_to_sc_omics(i) for i in inputs]

    # corrupting the data
    if corruption_rate > 0:
      inputs = [
          i.corrupt(corruption_rate=corruption_rate,
                    corruption_dist=corruption_dist) for i in inputs
      ]
    train, valid = [], []
    for i in inputs:
      tr, va = i.split(seed=self.seed, train_percent=1 - validation_split)
      train.append(tr)
      valid.append(va)

    # generate training mask for semi-supervised learning
    rand = np.random.RandomState(seed=self.seed)
    n = train[0].shape[0]
    train_mask = np.zeros(shape=(n, 1), dtype='float32')
    train_mask[rand.permutation(n)[:int(supervised_percent * n)]] = 1
    valid_mask = np.ones(shape=(valid[0].shape[0], 1), dtype='float32')

    # calculate the steps
    assert len(set(i.shape[0] for i in train)) == 1
    assert len(set(i.shape[0] for i in valid)) == 1
    steps_per_epoch = int(np.ceil(train[0].shape[0] / batch_size))
    validation_steps = int(np.ceil(valid[0].shape[0] / batch_size))

    # create tensorflow dataset, a bit ugly with many if-then-else
    # but it works!
    def to_tfdata(sco, mask):
      all_data = [i.X for i in sco]
      if self.is_semi_supervised:
        all_data += [mask]
      # NOTE: from_tensor_slices accept tuple but not list
      ds = tf.data.Dataset.from_tensor_slices(all_data[0] if len(all_data) ==
                                              1 else tuple(all_data))
      if shuffle:
        ds = ds.shuffle(1000)
      ds = ds.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
      ds = ds.repeat(epochs)
      # just return a tuple with 1 element to trick keras
      if len(all_data) == 1:
        ds = ds.map(lambda arg: (arg,))
      else:
        ds = ds.map(lambda *args: (args,))
      return ds

    train = to_tfdata(train, train_mask)
    valid = to_tfdata(valid, valid_mask)

    # prepare callback
    update_epoch = LambdaCallback(
        on_epoch_end=lambda *args, **kwargs: self._n_epoch.assign_add(1))
    if callbacks is None:
      callbacks = [update_epoch]
    elif isinstance(callbacks, Callback):
      callbacks = [callbacks, update_epoch]
    else:
      callbacks = list(callbacks)
      callbacks.append(update_epoch)

    # start training loop
    org_fn = self.call
    self.call = partial(self.call, n_samples=n_mcmc_samples)
    self._is_fitting = True

    # compile and fit
    if not self._is_compiled:
      super(SingleCellModel, self).compile(optimizer)
    super(SingleCellModel, self).fit(x=train,
                                     validation_data=valid,
                                     validation_freq=validation_freq,
                                     callbacks=callbacks,
                                     initial_epoch=self.n_epoch,
                                     steps_per_epoch=steps_per_epoch,
                                     validation_steps=validation_steps,
                                     epochs=epochs,
                                     verbose=verbose)

    # reset to original state
    self.call = org_fn
    self._is_fitting = False
