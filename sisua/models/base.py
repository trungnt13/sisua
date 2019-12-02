from __future__ import absolute_import, division, print_function

import inspect
import multiprocessing as mpi
import os
import string
import types
from abc import ABCMeta, abstractmethod, abstractproperty
from collections import OrderedDict, defaultdict
from functools import partial
from typing import Iterable, List, Text, Union

import numpy as np
import tensorflow as tf
from six import add_metaclass, string_types
from tensorflow import keras
from tensorflow.python.keras.callbacks import (Callback, CallbackList,
                                               LambdaCallback,
                                               LearningRateScheduler,
                                               ModelCheckpoint)
from tensorflow.python.platform import tf_logging as logging
from tensorflow_probability.python import distributions as tfd
from tqdm import tqdm

from odin.backend.interpolation import Interpolation
from odin.backend.keras_callbacks import EarlyStopping
from odin.bay import kl_divergence
from odin.bay.distribution_alias import parse_distribution
from odin.bay.distribution_layers import (CategoricalLayer,
                                          OneHotCategoricalLayer,
                                          VectorDeterministicLayer)
from odin.bay.distributions import concat_distribution
from odin.networks import DenseDistribution, MixtureDensityNetwork
from odin.utils import cache_memory, classproperty
from sisua.data import SingleCellOMIC


# ===========================================================================
# Helper
# ===========================================================================
def _to_sc_omics(x):
  if isinstance(x, SingleCellOMIC):
    return x
  return SingleCellOMIC(x)


def _to_tfdata(sco: SingleCellOMIC, mask: Union[np.ndarray, None],
               is_semi_supervised, batch_size, shuffle, epochs):
  all_data = []
  for i, j in enumerate(sco):
    if i == 0:  # main data, gene expression
      all_data += [j.X, j.local_mean, j.local_var]
    else:  # semi-supervised data
      all_data.append(j.X)
  # adding mask at the end if semi-supervised
  if is_semi_supervised and mask is not None:
    all_data += [mask]
  # NOTE: from_tensor_slices accept tuple but not list
  ds = tf.data.Dataset.from_tensor_slices(tuple(all_data))
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


_CACHE_PREDICT = defaultdict(dict)
_MAXIMUM_CACHE_SIZE = 2


# ===========================================================================
# Input description
# ===========================================================================
class KLinterpolation:

  def __init__(self,
               interpolate=Interpolation.exp10In,
               min_value=0.,
               max_value=10.,
               epoch_warmup=50):
    self.interpolate = interpolate
    self.min_value = min_value
    self.max_value = max_value
    self.epoch_warmup = epoch_warmup

  def __call__(self, epoch):
    epoch = tf.maximum(tf.cast(epoch, 'float32'), 1.)
    alpha = tf.minimum(epoch / self.epoch_warmup, 1.)
    return self.interpolate(alpha, self.min_value, self.max_value)

  def __repr__(self):
    return self.__str__()

  def __str__(self):
    return "<KL interpolate:%d min:%.2f max:%.2f warmup:%d>" % \
      (str(self.interpolate), self.min_value, self.max_value, self.epoch_warmup)


class OmicOutput:
  r""" Description of OMIC as output for models
  Arguments:
    dim : Integer, number of features (all OMIC data must be 2-D)
    posterior : alias for posterior distribution
    name : identity of the OMIC
    kwargs : keyword arguments for initializing the `DistributionLambda`
      of the posterior.
  """

  def __init__(self, dim, posterior='nb', name=None, **kwargs):
    if isinstance(dim, SingleCellOMIC):
      dim = dim.shape[1]
    self.name = name
    self.dim = int(dim)
    self.posterior = str(posterior)
    self.kwargs = kwargs

  def __str__(self):
    return "<OMIC dim:%d name:%s posterior:%s zi:%s>" % \
      (self.dim, str(self.name), self.posterior, self.is_zero_inflated)

  def __repr__(self):
    return self.__str__()

  @property
  def is_zero_inflated(self):
    return 'zi' == self.posterior[:2].lower()

  @property
  def input_shape(self):
    return (self.dim,)

  def create_posterior(self):
    # ====== simple loss function from tensorflow ====== #
    if self.posterior in dir(tf.losses):
      distribution = VectorDeterministicLayer
      activation = 'relu'
      fn = tf.losses.get(str(self.posterior))
      llk_fn = lambda self, y_true: tf.expand_dims(
          -fn(y_true, self.posterior.mean()), axis=-1)
    # ====== probabilistic loss ====== #
    else:
      if self.posterior in ('mdn', 'mixdiag', 'mixfull'):
        distribution = None
      else:
        distribution = parse_distribution(self.posterior)[0]
      activation = 'linear'
      llk_fn = lambda self, y_true: tf.expand_dims(
          self.posterior.log_prob(y_true), axis=-1)
    # ====== create distribution layers ====== #
    kwargs = dict(self.kwargs)
    activation = kwargs.pop('activation', activation)
    if self.posterior in ('mdn', 'mixdiag', 'mixfull'):
      kwargs.pop('covariance', None)
      layer = MixtureDensityNetwork(self.dim,
                                    activation=activation,
                                    covariance=dict(
                                        mdn='none',
                                        mixdiag='diag',
                                        mixfull='full')[self.posterior],
                                    name=self.name,
                                    **kwargs)
    else:
      layer = DenseDistribution(self.dim,
                                posterior=distribution,
                                activation=activation,
                                posterior_kwargs=kwargs,
                                name=self.name)
    layer.log_prob = types.MethodType(llk_fn, layer)
    return layer


# ===========================================================================
# SingleCell model
# ===========================================================================
class SingleCellModel(keras.Model):
  r"""
  Note:
    It is recommend to call `tensorflow.random.set_seed` for reproducible
    results.
  """

  def __init__(self,
               outputs: [OmicOutput, Iterable[OmicOutput]],
               analytic=True,
               kl_interpolate=KLinterpolation(),
               log_norm=True,
               seed=8,
               name=None):
    super(SingleCellModel, self).__init__(name=name)
    self._epoch = self.add_weight(
        name='epochs',
        shape=(),
        initializer=tf.initializers.Constant(value=0.),
        trainable=False,
        dtype='float32')
    # parameters for ELBO
    self._analytic = bool(analytic)
    assert isinstance(kl_interpolate, KLinterpolation)
    self._kl_interpolate = kl_interpolate
    # parameters for fitting
    self._seed = int(seed)
    self._is_fitting = False
    self._corruption_rate = 0
    self._corruption_dist = 'binomial'
    self._log_norm = bool(log_norm)
    # parsing the inputs
    outputs = [
        o if isinstance(o, OmicOutput) else OmicOutput(o)
        for o in tf.nest.flatten(outputs)
    ]
    assert all(isinstance(i, OmicOutput) for i in outputs), \
      "Inputs must be instance of OmicOutput but give: %s" % \
        ','.join([str(type(i)) for i in outputs])
    # check if is semi-supervised
    self._is_semi_supervised = True if len(outputs) > 1 else False
    self._omic_outputs = outputs
    # create all the output distributions
    # we must set the attribute directly so the Model will manage
    # the output layer, once all the output layers are initialized
    # the number of outputs will match `n_outputs`
    for idx, omic in enumerate(self._omic_outputs):
      name = 'output_%d' % idx
      post = omic.create_posterior()
      setattr(self, name, post)

  @property
  def posteriors(self):
    return [getattr(self, 'output_%d' % i) for i in range(self.n_outputs)]

  @property
  def omic_outputs(self):
    return list(self._omic_outputs)

  @property
  def n_outputs(self):
    return len(self._omic_outputs)

  @property
  def log_norm(self):
    return self._log_norm

  @property
  def is_zero_inflated(self):
    return self._omic_outputs[0].is_zero_inflated

  @property
  def is_semi_supervised(self):
    return self._is_semi_supervised

  @property
  def corruption_rate(self):
    return self._corruption_rate

  @property
  def corruption_dist(self):
    return self._corruption_dist

  @property
  def epoch(self):
    return int(self._epoch.numpy())

  @property
  def analytic(self):
    return self._analytic

  @property
  def kl_weight(self):
    return self._kl_interpolate(self._epoch)

  @property
  def seed(self):
    return self._seed

  @abstractmethod
  def _call(self, x, lmean, lvar, t, y, mask, training=None, n_mcmc=None):
    r"""
    Arguments:
      x : [batch_size, n_genes]
        input gene-expression matrix (multiple inputs can be given)
      lmean : [batch_size]
        mean of library size
      lvar : [batch_size]
        variance of library size
      t : [batch_size, n_genes]
        target for reconstruction of gene-expression matrix (can be different
        from `x`)
      y : [batch_size, n_protein]
        input for semi-supervised learning
      mask : [batch_size, 1]
        binary mask for semi-supervised training
      training : `bool`
        flag mark training progress
      n_mcmc : {`None`, `int`}
        number of MCMC samples
    """
    raise NotImplementedError

  def call(self, inputs, training=None, n_mcmc=1):
    # check arguments
    if not isinstance(inputs, (tuple, list)):
      inputs = [inputs]
    else:
      inputs = list(inputs)

    # eager mode, the Model is called directly
    # lmean and lvar is not given
    if (len(inputs) >= 3 and
        (inputs[1].shape[1] != 1 or inputs[2].shape[1] != 1)) or \
      len(inputs) < 3:
      if isinstance(inputs[0], SingleCellOMIC):
        inputs = [inputs[0].numpy(), inputs[0].local_mean, inputs[0].local_var
                 ] + inputs[1:]
      else:
        inputs = [
            inputs[0],
            tf.zeros(shape=(inputs[0].shape[0], 1)),
            tf.zeros(shape=(inputs[0].shape[0], 1))
        ] + inputs[1:]

    x, lmean, lvar = inputs[:3]
    # Preprocessing the inputs for semi-supervised training
    if self._is_fitting:
      y = list(inputs[3:-1])  # y1, y2, ...
      mask = [inputs[-1]] * len(y)  # same mask for all y
    # Eager call
    else:
      y = [
          i.numpy() if isinstance(i, SingleCellOMIC) else i for i in inputs[3:]
      ]  # y1, y2, ...
      mask = [1.] * len(y)
    # log normalization the input
    t = x
    if self.log_norm:
      x = tf.math.log1p(x)
    # postprocessing the returns
    pX, qZ = self._call(x, lmean, lvar, t, y, mask, training, n_mcmc)
    # Add Log-likelihood, don't forget to apply mask for semi-supervised loss
    # we have to use the log_prob function from DenseDistribution class, not
    # the one in pX
    llk_x = self.posteriors[0].log_prob(t)
    llk_y = tf.convert_to_tensor(0, dtype=x.dtype)
    track_llky = []
    for i_true, m, post in zip(y, mask, self.posteriors[1:]):
      llk = post.log_prob(i_true) * m
      track_llky.append((llk, post.name.lower()))
      llk_y += llk
    # calculating the KL
    kl = tf.convert_to_tensor(0, dtype=x.dtype)
    track_kl = []
    for idx, qz in enumerate(tf.nest.flatten(qZ)):
      div = qz.KL_divergence(analytic=self.analytic)
      if self.analytic:  # add mcmc dimension if used close-form KL
        div = tf.expand_dims(div, axis=0)
        if n_mcmc > 1:
          div = tf.tile(div, [n_mcmc] + [1] * (div.shape.ndims - 1))
      track_kl.append((div, 'z%s' % qz.event_shape[0]))
      kl += div
    # Final ELBO
    beta = self.kl_weight
    self.add_metric(beta, 'mean', 'beta')
    elbo = (llk_x + llk_y) - kl * beta
    elbo = tf.reduce_logsumexp(elbo, axis=0)
    loss = tf.reduce_mean(-elbo)
    if training:
      self.add_loss(loss)
    self.add_metric(llk_x, 'mean', "llk_%s" % self.posteriors[0].name)
    for llk_val, name in track_llky:
      self.add_metric(llk_val, 'mean', "llk_%s" % name)
    for kl_val, name in track_kl:
      self.add_metric(kl_val, 'mean', "klqp_%s" % name)
    # post-processing the return
    if not self.is_semi_supervised and isinstance(pX, (tuple, list)):
      pX = pX[0]
    return pX, qZ

  def evaluate(self, inputs, n_mcmc=1, batch_size=128, verbose=1):
    raise Exception(
        "This method is not support, please use sisua.analysis.Posterior")

  def predict(self,
              inputs,
              n_mcmc=1,
              batch_size=64,
              apply_corruption=False,
              enable_cache=True,
              verbose=1):
    r"""
    Arguments:
      apply_corruption : `bool` (default=`False`)
        if `True` applying corruption on data before prediction to match the
        condition during fitting.
      enable_cache : `bool` (default=`True`)
        if `True` store the "footprint" of the input arguments to return the
        cached outputs

    Return:
      X : `Distribution` or tuple of `Distribution`
        output distribution, multiple distribution is return in case of
        multiple outputs
      Z : `Distribution` or tuple of `Distribution`
        latent distribution, multiple distribution is return in case of
        multiple latents
    """
    if not isinstance(inputs, (tuple, list)):
      inputs = [inputs]
    inputs = [_to_sc_omics(i) for i in inputs]
    assert len(inputs) == 1, \
      "During prediction phase, only the mRNA gene expression is provided, " +\
        "this is strict regulation for all models!"

    # checking the cache, this mechanism will significantly improve speed
    # during monitoring of fitting process
    self_id = id(self)
    footprint = ''.join([str(id(i.X)) for i in inputs]) + \
      str(n_mcmc) + str(apply_corruption) + str(self.epoch)
    if enable_cache and footprint in _CACHE_PREDICT[id(self)]:
      return _CACHE_PREDICT[self_id][footprint]

    # applying corruption for testing
    if apply_corruption and self.corruption_rate is not None:
      inputs = [
          data.corrupt(corruption_rate=self.corruption_rate,
                       corruption_dist=self.corruption_dist,
                       inplace=False) if idx == 0 else data
          for idx, data in enumerate(inputs)
      ]
    n = inputs[0].shape[0]
    data = _to_tfdata(inputs,
                      None,
                      self.is_semi_supervised,
                      batch_size,
                      shuffle=False,
                      epochs=1)

    kw = {'n_mcmc': int(n_mcmc)}
    if 'n_mcmc' not in self._call_fn_args:
      del kw['n_mcmc']

    X, Z = [], []
    for inputs in tqdm(data,
                       desc="Predicting",
                       total=int(np.ceil(n / batch_size)),
                       disable=not bool(verbose)):
      # the _to_tfddata will return (x, y) tuple for `fit` methods,
      # y=None and we only need x here.
      x, z = self(inputs[0], training=False, **kw)
      X.append(x)
      Z.append(z)

    if isinstance(x, (tuple, list)):
      merging_axis = 0 if x[0].batch_shape.ndims == 1 else 1
    else:
      merging_axis = 0 if x.batch_shape.ndims == 1 else 1
    # multiple outputs
    if isinstance(X[0], (tuple, list)):
      X = tuple([
          concat_distribution([x[idx]
                               for x in X], axis=merging_axis)
          for idx in range(len(X[0]))
      ])
    # single output
    else:
      X = concat_distribution(X, axis=merging_axis)

    # multiple latents
    if isinstance(Z[0], (tuple, list)):
      Z = tuple([
          concat_distribution([z[idx]
                               for z in Z], axis=0)
          for idx in range(len(Z[0]))
      ])
    else:
      Z = concat_distribution(Z, axis=0)
    # cache and return
    if enable_cache:
      _CACHE_PREDICT[self_id][footprint] = (X, Z)
      # LIFO
      if len(_CACHE_PREDICT[self_id]) > _MAXIMUM_CACHE_SIZE:
        key = list(_CACHE_PREDICT[self_id].keys())[0]
        del _CACHE_PREDICT[self_id][key]
    return X, Z

  def fit(
      self,
      inputs: Union[SingleCellOMIC, Iterable[SingleCellOMIC]],
      optimizer: Union[Text, tf.optimizers.Optimizer] = 'adam',
      learning_rate=1e-4,
      clipnorm=100,
      n_mcmc=1,
      semi_percent=0.8,
      semi_weight=10,
      corruption_rate=0.25,
      corruption_dist='binomial',
      batch_size=64,
      epochs=2,
      callbacks=None,
      validation_split=0.1,
      validation_freq=1,
      min_delta=0.5,  # for early stopping
      min_epoch=80,  # for early stopping
      patience=25,  # for early stopping
      allow_rollback=True,  # for early stopping
      terminate_on_nan=True,  # for early stopping
      checkpoint=None,
      shuffle=True,
      verbose=1):
    r""" This fit function is the combination of both
    `Model.compile` and `Model.fit` """
    # check signature of `call` function
    # inputs, training=None, n_mcmc=1
    specs = inspect.getfullargspec(self.call)
    if specs.args != ['self', 'inputs', 'training', 'n_mcmc']:
      raise ValueError(
          "call method must have following arguments %s; bug given %s" %
          (['self', 'inputs', 'training', 'n_mcmc'], specs.args))

    # check arguments
    assert 0.0 <= semi_percent <= 1.0
    if validation_split <= 0:
      raise ValueError("validation_split must > 0")

    # store the corruption rate for later use
    self._corruption_dist = corruption_dist
    self._corruption_rate = corruption_rate

    # prepare input data
    if not isinstance(inputs, (tuple, list)):
      inputs = [inputs]
    inputs = [_to_sc_omics(i) for i in inputs]

    # corrupting the data (only the main data, the semi supervised data
    # remain clean)
    if corruption_rate > 0:
      inputs = [
          data.corrupt(corruption_rate=corruption_rate,
                       corruption_dist=corruption_dist,
                       inplace=False) if idx == 0 else data
          for idx, data in enumerate(inputs)
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
    train_mask[rand.permutation(n)[:int(semi_percent * n)]] = 1
    train_mask = train_mask * semi_weight
    valid_mask = np.ones(shape=(valid[0].shape[0], 1), dtype='float32')

    # calculate the steps
    assert len(set(i.shape[0] for i in train)) == 1
    assert len(set(i.shape[0] for i in valid)) == 1
    steps_per_epoch = int(np.ceil(train[0].shape[0] / batch_size))
    validation_steps = int(np.ceil(valid[0].shape[0] / batch_size))

    # convert to tensorflow Dataset
    train_data = _to_tfdata(train, train_mask, self.is_semi_supervised,
                            batch_size, shuffle, epochs)
    valid_data = _to_tfdata(valid, valid_mask, self.is_semi_supervised,
                            batch_size, shuffle, epochs)

    # prepare callback
    update_epoch = LambdaCallback(
        on_epoch_end=lambda *args, **kwargs: self._epoch.assign_add(1))
    if callbacks is None:
      callbacks = [update_epoch]
    elif isinstance(callbacks, Callback):
      callbacks = [callbacks, update_epoch]
    else:
      callbacks = list(callbacks)
      callbacks.append(update_epoch)
    # add early stopping
    if patience >= 0:
      callbacks.append(
          EarlyStopping(monitor='val_loss',
                        min_delta=min_delta,
                        min_epoch=min_epoch,
                        patience=patience,
                        verbose=verbose,
                        mode='min',
                        baseline=None,
                        terminate_on_nan=bool(terminate_on_nan),
                        restore_best_weights=bool(allow_rollback)))
    # checkpoint
    if checkpoint is not None:
      callbacks.append(
          ModelCheckpoint(filepath=str(checkpoint),
                          monitor='val_loss',
                          verbose=verbose,
                          save_weights_only=True,
                          save_freq='epoch',
                          save_best_only=True,
                          load_weights_on_restart=False,
                          mode='min'))
    # automatically set inputs as validation set for missing inputs
    # metrical callbacks
    from sisua.analysis.sc_metrics import SingleCellMetric
    from sisua.analysis.sc_monitor import SingleCellMonitor
    for cb in callbacks:
      if isinstance(cb, (SingleCellMetric, SingleCellMonitor)) and \
        cb.inputs is None:
        # only mRNA expression is needed during evaluation (prediction)
        cb.inputs = valid[0]
    # reorganize so the SingleCellMonitor is at the end, hence, all the
    # metrics are computed before the monitor start plotting
    cb_others = [
        cb for cb in callbacks if not isinstance(cb, SingleCellMonitor)
    ]
    cb_monitor = [cb for cb in callbacks if isinstance(cb, SingleCellMonitor)]
    callbacks = cb_others + cb_monitor

    # start training loop
    org_fn = self.call
    if 'n_mcmc' in self._call_fn_args:
      self.call = partial(self.call, n_mcmc=n_mcmc)
    self._is_fitting = True

    # prepare the logging
    curr_log = logging.get_verbosity()
    logging.set_verbosity(logging.ERROR)
    # compiling and setting the optimizer
    if not self._is_compiled and self.optimizer is None:
      # prepare optimizer
      if isinstance(optimizer, string_types):
        config = dict(learning_rate=learning_rate)
        if clipnorm is not None:
          config['clipnorm'] = clipnorm
        optimizer = tf.optimizers.get({
            'class_name': optimizer,
            'config': config
        })
      elif isinstance(optimizer, tf.optimizers.Optimizer):
        pass
      elif isinstance(optimizer, type) and issubclass(optimizer,
                                                      tf.optimizers.Optimizer):
        optimizer = optimizer(learning_rate=learning_rate) \
          if clipnorm is None else \
          optimizer(learning_rate=learning_rate, clipnorm=clipnorm)
      else:
        raise ValueError("No support for optimizer: %s" % str(optimizer))
      super(SingleCellModel, self).compile(optimizer,
                                           experimental_run_tf_function=False)
    # fitting
    super(SingleCellModel, self).fit(x=train_data,
                                     validation_data=valid_data,
                                     validation_freq=validation_freq,
                                     callbacks=callbacks,
                                     initial_epoch=self.epoch,
                                     steps_per_epoch=steps_per_epoch,
                                     validation_steps=validation_steps,
                                     epochs=self.epoch + epochs,
                                     verbose=verbose)
    logging.set_verbosity(curr_log)

    # reset to original state
    self.call = org_fn
    self._is_fitting = False

  def __repr__(self):
    return self.__str__()

  def __str__(self):
    return "<[%s]%s fitted:%s epoch:%s semi:%s>" % (
        self.__class__.__name__, self.name, self.epoch > 0, self.epoch,
        self.is_semi_supervised)

  @classproperty
  def id(cls):
    class_name = cls.__name__
    name = ''
    for i in class_name:
      if i.isupper():
        name += i
    return name.lower()
