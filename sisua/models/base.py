from __future__ import absolute_import, division, print_function

import inspect
import multiprocessing as mpi
import os
import string
from abc import ABCMeta, abstractmethod, abstractproperty
from collections import OrderedDict, defaultdict
from functools import partial
from typing import Iterable, List, Text, Union

import dill
import numpy as np
import tensorflow as tf
from six import add_metaclass, string_types
from tensorflow.python.keras.callbacks import (Callback, CallbackList,
                                               LambdaCallback)
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.layers import Layer
from tensorflow.python.platform import tf_logging as logging
from tensorflow_probability.python import distributions as tfd
from tqdm import tqdm

from odin.bay.distributions import stack_distributions
from odin.networks import AdvanceModel, DeterministicDense, DistributionDense
from odin.utils import cache_memory, classproperty
from sisua.data import SingleCellOMICS
from sisua.models import latents as sisua_latents
from sisua.models import networks as sisua_networks

try:
  from hyperopt import hp, fmin, Trials, STATUS_OK, STATUS_FAIL
  from hyperopt.tpe import suggest as tpe_suggest
  from hyperopt.rand import suggest as rand_suggest
  from hyperopt.pyll import scope
except ImportError as e:
  raise RuntimeError(
      "Cannot import hyperopt for hyper-parameters tuning, error: %s" % str(e))


# ===========================================================================
# Helper
# ===========================================================================
def _to_sc_omics(x):
  if isinstance(x, SingleCellOMICS):
    return x
  return SingleCellOMICS(x)


def _to_tfdata(sco: SingleCellOMICS, mask: Union[np.ndarray, None],
               is_semi_supervised, batch_size, shuffle, epochs):
  all_data = [i.X for i in sco]
  if is_semi_supervised and mask is not None:
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


def _to_semisupervised_inputs(inputs, is_fitting):
  """ Preprocessing the inputs for semi-supervised training """
  # objective mask is provided
  assert isinstance(inputs, (tuple, list))

  # mask is given at the end during training
  if is_fitting:
    x = inputs[0]
    y = list(inputs[1:-1])
    masks = [inputs[-1]] * len(y)
  # no mask is provided
  else:
    x = inputs[0]
    y = list(inputs[1:])
    masks = [1.] * len(y)
  return x, y, masks


_CACHE_PREDICT = defaultdict(dict)
_MAXIMUM_CACHE_SIZE = 2


# ===========================================================================
# SingleCell model
# ===========================================================================
@add_metaclass(ABCMeta)
class SingleCellModel(AdvanceModel):
  """

  Note
  ----
  It is recommend to call `tensorflow.random.set_seed` for reproducible results
  """

  def __init__(self,
               parameters,
               kl_analytic=True,
               kl_weight=1.,
               kl_warmup=200,
               seed=8,
               name=None):
    if name is None:
      name = self.__class__.__name__
    parameters.update(locals())
    super(SingleCellModel, self).__init__(parameters=parameters, name=name)
    self._epochs = tf.Variable(0,
                               trainable=False,
                               dtype='float32',
                               name="epoch")
    # parameters for ELBO
    self._kl_analytic = bool(kl_analytic)
    self._kl_weight = tf.convert_to_tensor(kl_weight, dtype='float32')
    self._kl_warmup = tf.convert_to_tensor(kl_warmup, dtype='float32')
    self._seed = int(seed)
    self._is_fitting = False
    self._corruption_rate = None
    self._corruption_dist = None

  @property
  def epoch_history(self):
    return self.history.history

  @property
  def custom_objects(self):
    return [sisua_latents, sisua_networks]

  @abstractproperty
  def is_semi_supervised(self):
    raise NotImplementedError

  @abstractmethod
  def _call(self, x, y, masks, training=None, n_samples=None):
    raise NotImplementedError

  @property
  def corruption_rate(self):
    return self._corruption_rate

  @property
  def corruption_dist(self):
    return self._corruption_dist

  @property
  def epochs(self):
    return int(self._epochs.numpy())

  @property
  def kl_analytic(self):
    return self._kl_analytic

  @property
  def kl_warmup(self):
    return self._kl_warmup

  @property
  def kl_weight(self):
    warmup_weight = tf.minimum(
        tf.maximum(self._epochs, 1.) / self.kl_warmup, 1.)
    return warmup_weight * self._kl_weight

  @property
  def seed(self):
    return self._seed

  @property
  def parameters(self):
    return self._parameters

  def call(self, inputs, training=None, n_samples=None):
    # check arguments
    if n_samples is None:
      n_samples = 1

    if isinstance(inputs, (tuple, list)) and not self.is_semi_supervised:
      raise RuntimeError(
          "Multiple inputs is given for non semi-supervised model")

    if self.is_semi_supervised:
      x, y, masks = _to_semisupervised_inputs(inputs, self._is_fitting)
    else:
      x = inputs
      y = []
      masks = []
    return self._call(x, y, masks, training, n_samples)

  def evaluate(self, inputs, n_samples=1, batch_size=128, verbose=1):
    raise Exception(
        "This method is not support, please use sisua.analysis.Posterior")

  def predict(self,
              inputs,
              n_samples=1,
              batch_size=128,
              apply_corruption=False,
              verbose=1):
    """
    Parameters
    ----------
    apply_corruption : `bool` (default=`False`)
      if `True` applying corruption on data before prediction to match the
      condition during fitting.

    Return
    ------
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
    # checking the cache, this mechanism will significantly improve speed
    # during monitoring of fitting process
    self_id = id(self)
    footprint = ''.join([str(id(i.X)) for i in inputs]) + \
      str(n_samples) + str(apply_corruption) + str(self.epochs)
    if footprint in _CACHE_PREDICT[id(self)]:
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

    kw = {'n_samples': int(n_samples)}
    if 'n_samples' not in self._call_fn_args:
      del kw['n_samples']

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
          stack_distributions([x[idx]
                               for x in X], axis=merging_axis)
          for idx in range(len(X[0]))
      ])
    # single output
    else:
      X = stack_distributions(X, axis=merging_axis)

    # multiple latents
    if isinstance(Z[0], (tuple, list)):
      Z = tuple([
          stack_distributions([z[idx]
                               for z in Z])
          for idx in range(len(Z[0]), axis=0)
      ])
    else:
      Z = stack_distributions(Z, axis=0)
    # cache and return
    _CACHE_PREDICT[self_id][footprint] = (X, Z)
    # LIFO
    if len(_CACHE_PREDICT[self_id]) > _MAXIMUM_CACHE_SIZE:
      key = list(_CACHE_PREDICT[self_id].keys())[0]
      del _CACHE_PREDICT[self_id][key]
    return X, Z

  def fit(self,
          inputs: Union[SingleCellOMICS, Iterable[SingleCellOMICS]],
          optimizer: Union[Text, tf.optimizers.Optimizer] = 'adam',
          learning_rate=1e-4,
          n_samples=1,
          semi_percent=0.8,
          semi_weight=25,
          corruption_rate=0.25,
          corruption_dist='binomial',
          batch_size=64,
          epochs=2,
          callbacks=None,
          validation_split=0.1,
          validation_freq=1,
          shuffle=True,
          verbose=1):
    """ This fit function is the combination of both
    `Model.compile` and `Model.fit` """
    # check signature of `call` function
    # inputs, training=None, n_samples=1
    specs = inspect.getfullargspec(self.call)
    if specs.args != ['self', 'inputs', 'training', 'n_samples']:
      raise ValueError(
          "call method must have following arguments %s; bug given %s" %
          (['self', 'inputs', 'training', 'n_samples'], specs.args))

    # check arguments
    assert 0.0 <= semi_percent <= 1.0
    if validation_split <= 0:
      raise ValueError("validation_split must > 0")
    # prepare optimizer
    if isinstance(optimizer, string_types):
      optimizer = tf.optimizers.get(optimizer)

    if isinstance(optimizer, tf.optimizers.Optimizer):
      pass
    elif issubclass(optimizer, tf.optimizers.Optimizer):
      optimizer = optimizer(learning_rate=learning_rate)

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

    # create tensorflow dataset, a bit ugly with many if-then-else
    # but it works!
    # def to_tfdata(sco, mask):
    #   all_data = [i.X for i in sco]
    #   if self.is_semi_supervised:
    #     all_data += [mask]
    #   # NOTE: from_tensor_slices accept tuple but not list
    #   ds = tf.data.Dataset.from_tensor_slices(all_data[0] if len(all_data) ==
    #                                           1 else tuple(all_data))
    #   if shuffle:
    #     ds = ds.shuffle(1000)
    #   ds = ds.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    #   ds = ds.repeat(epochs)
    #   # just return a tuple with 1 element to trick keras
    #   if len(all_data) == 1:
    #     ds = ds.map(lambda arg: (arg,))
    #   else:
    #     ds = ds.map(lambda *args: (args,))
    #   return ds

    train_data = _to_tfdata(train, train_mask, self.is_semi_supervised,
                            batch_size, shuffle, epochs)
    valid_data = _to_tfdata(valid, valid_mask, self.is_semi_supervised,
                            batch_size, shuffle, epochs)

    # prepare callback
    update_epoch = LambdaCallback(
        on_epoch_end=lambda *args, **kwargs: self._epochs.assign_add(1))
    if callbacks is None:
      callbacks = [update_epoch]
    elif isinstance(callbacks, Callback):
      callbacks = [callbacks, update_epoch]
    else:
      callbacks = list(callbacks)
      callbacks.append(update_epoch)
    # automatically set inputs as validation set for missing inputs
    # metrical callbacks
    from sisua.analysis.sc_metrics import SingleCellMetric
    from sisua.analysis.sc_monitor import SingleCellMonitor
    for cb in callbacks:
      if isinstance(cb, (SingleCellMetric, SingleCellMonitor)) and \
        cb.inputs is None:
        cb.inputs = valid
    # reorganize so the SingleCellMonitor is at the end, hence, all the
    # metrics are computed before the monitor start plotting
    cb_others = [
        cb for cb in callbacks if not isinstance(cb, SingleCellMonitor)
    ]
    cb_monitor = [cb for cb in callbacks if isinstance(cb, SingleCellMonitor)]
    callbacks = cb_others + cb_monitor

    # start training loop
    org_fn = self.call
    if 'n_samples' in self._call_fn_args:
      self.call = partial(self.call, n_samples=n_samples)
    self._is_fitting = True

    # compile and fit
    curr_log = logging.get_verbosity()
    logging.set_verbosity(logging.ERROR)
    if not self._is_compiled:
      super(SingleCellModel, self).compile(optimizer)
    super(SingleCellModel, self).fit(x=train_data,
                                     validation_data=valid_data,
                                     validation_freq=validation_freq,
                                     callbacks=callbacks,
                                     initial_epoch=self.epochs,
                                     steps_per_epoch=steps_per_epoch,
                                     validation_steps=validation_steps,
                                     epochs=epochs,
                                     verbose=verbose)
    logging.set_verbosity(curr_log)

    # reset to original state
    self.call = org_fn
    self._is_fitting = False

  def __repr__(self):
    return self.__str__()

  def __str__(self):
    return "<[%s]%s fitted:%s epoch:%s semi:%s>" % (
        self.__class__.__name__, self.name, self._is_compiled, self.epochs,
        self.is_semi_supervised)

  @classproperty
  def id(cls):
    class_name = cls.__name__
    name = ''
    for i in class_name:
      if i.isupper():
        name += i
    return name.lower()

  @classmethod
  def fit_hyper(
      cls,
      inputs: Union[SingleCellOMICS, Iterable[SingleCellOMICS]],
      params: dict = {
          'nlayers': scope.int(hp.quniform('nlayers', 1, 4, 1)),
          'hdim': scope.int(hp.quniform('hdim', 32, 512, 1)),
          'zdim': scope.int(hp.quniform('zdim', 32, 512, 1)),
      },
      loss_name: Text = 'val_loss',
      max_evals: int = 100,
      kwargs: dict = {},
      fit_kwargs: dict = {
          'epochs': 64,
          'batch_size': 128
      },
      algorithm: Text = 'bayes',
      seed: int = 8,
      save_path: Text = '/tmp/{model:s}_{data:s}_{loss:s}_{params:s}.hp',
      override: bool = True):
    """ Hyper-parameters optimization for given SingleCellModel
    Parameters
    ----------
    kwargs : `dict`
      keyword arguments for model construction
    fit_kwargs : `dict`
      keyword arguments for `fit` method

    Example
    -------
    >>> callbacks = [
    >>>     # NegativeLogLikelihood(),
    >>>     # ImputationError(),
    >>>     CorrelationScores(extras=y_train)
    >>> ]
    >>> i, j = DeepCountAutoencoder.fit_hyper(x_train,
    >>>                                       kwargs={'loss': 'zinb'},
    >>>                                       fit_kwargs={
    >>>                                           'callbacks': callbacks,
    >>>                                           'epochs': 64,
    >>>                                           'batch_size': 128
    >>>                                       },
    >>>                                       loss_name='pearson_mean',
    >>>                                       algorithm='bayes',
    >>>                                       max_evals=100,
    >>>                                       seed=8)
    """
    algorithm = str(algorithm.lower())
    assert algorithm in ('rand', 'grid', 'bayes'), \
      "Only support 3 algorithm: rand, grid and bayes; given %s" % algorithm
    # force to turn of keras verbose, it is a big mess to show
    # 2 progress bar at once
    fit_kwargs.update({'verbose': 0})

    # remove unncessary params
    args = inspect.getfullargspec(cls.__init__)
    params = {i: j for i, j in params.items() if i in args.args}

    # processing save_path
    fmt = {}
    for (_, key, spec, _) in string.Formatter().parse(save_path):
      if spec is not None:
        fmt[key] = None
    if isinstance(inputs, (tuple, list)):
      dsname = inputs[0].name if hasattr(inputs[0],
                                         'name') else 'x%d' % len(inputs)
    else:
      dsname = inputs.name if hasattr(inputs, 'name') else 'x'
    kw = {
        'model': cls.id,
        'data': dsname.replace('_', ''),
        'loss': loss_name.replace('val_', '').replace('_', ''),
        'params': '_'.join(sorted([i.replace('_', '') for i in params.keys()]))
    }
    kw = {i: j for i, j in kw.items() if i in fmt}
    save_path = save_path.format(**kw)
    if os.path.exists(save_path) and not override:
      raise RuntimeError("Cannot override path: %s" % save_path)

    def fit_and_evaluate(*args):
      kw = args[0]
      kw.update(kwargs)
      obj = cls(**kw)
      obj.fit(inputs, **fit_kwargs)
      history = obj.history.history
      loss = history[loss_name]
      # first epoch doesn't count
      return {
          'loss': np.mean(loss[1:]),
          'loss_variance': np.var(loss[1:]),
          'history': obj.history.history,
          'status': STATUS_OK,
      }

    trials = Trials()
    results = fmin(fit_and_evaluate,
                   space=params,
                   algo=tpe_suggest if algorithm == 'bayes' else rand_suggest,
                   max_evals=int(max_evals),
                   trials=trials,
                   rstate=np.random.RandomState(seed))
    history = []
    for t in trials:
      r = t['result']
      if r['status'] == STATUS_OK:
        history.append({
            'loss': r['loss'],
            'loss_variance': r['loss_variance'],
            'params': {i: j[0] for i, j in t['misc']['vals'].items()},
            'history': r['history']
        })
    with open(save_path, 'wb') as f:
      print("Saving hyperopt results to: %s" % save_path)
      dill.dump((results, history), f)
    return results, history
