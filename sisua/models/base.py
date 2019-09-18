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
                                               LambdaCallback,
                                               LearningRateScheduler,
                                               ModelCheckpoint)
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.layers import Layer
from tensorflow.python.platform import tf_logging as logging
from tensorflow_probability.python import distributions as tfd
from tqdm import tqdm

from odin.backend.keras_callbacks import EarlyStopping
from odin.bay.distribution_alias import parse_distribution
from odin.bay.distribution_layers import VectorDeterministicLayer
from odin.bay.distributions import concat_distribution
from odin.networks import AdvanceModel
from odin.utils import cache_memory, classproperty
from sisua.data import SingleCellOMIC
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


def _get_loss_and_distribution(loss):
  # note: loss function always return keepdims=True
  loss = str(loss).lower()
  # ====== simple loss function from tensorflow ====== #
  if loss in dir(tf.losses):
    output_layer = VectorDeterministicLayer
    activation = 'relu'
    loss_fn = lambda y_true, y_pred: tf.expand_dims(tf.losses.get(str(loss))
                                                    (y_true, y_pred),
                                                    axis=-1)
  # ====== probabilistic loss ====== #
  else:
    output_layer = parse_distribution(loss)[0]
    activation = 'linear'
    loss_fn = lambda y_true, y_pred: tf.expand_dims(-y_pred.log_prob(y_true),
                                                    axis=-1)
  # post-processing the loss function to be more universal
  return output_layer, activation, loss_fn


_CACHE_PREDICT = defaultdict(dict)
_MAXIMUM_CACHE_SIZE = 2


# ===========================================================================
# SingleCell model
# ===========================================================================
@add_metaclass(ABCMeta)
class SingleCellModel(AdvanceModel):
  """

  I/O
  ---
  >>> import dill
  >>> c, w = model.get_config(), model.get_weights()
  >>> with open('/tmp/model', 'wb') as f:
  >>>   dill.dump((c, w), f)
  ...
  >>> with open('/tmp/model', 'rb') as f:
  >>>   (c, w) = dill.load(f)
  >>> model = SingleCellModel.from_config(c)
  >>> model.set_weights(w)


  Note
  ----
  It is recommend to call `tensorflow.random.set_seed` for reproducible results
  """

  def __init__(self,
               units,
               xdist,
               dispersion,
               parameters,
               kl_analytic=True,
               kl_weight=1.,
               kl_warmup=50,
               log_norm=True,
               seed=8,
               name=None):
    if name is None:
      name = self.__class__.__name__
    parameters.update(locals())
    super(SingleCellModel, self).__init__(parameters=parameters, name=name)
    self._epochs = self.add_weight(
        name='epochs',
        shape=(),
        initializer=tf.initializers.Constant(value=0.),
        trainable=False,
        dtype='float32')
    # parameters for ELBO
    self._kl_analytic = bool(kl_analytic)
    self._kl_weight = tf.convert_to_tensor(kl_weight, dtype='float32')
    self._kl_warmup = tf.convert_to_tensor(kl_warmup, dtype='float32')
    self._seed = int(seed)
    self._is_fitting = False
    self._corruption_rate = 0
    self._corruption_dist = 'binomial'
    self._log_norm = bool(log_norm)
    # dispersion
    dispersion = str(dispersion).lower()
    assert dispersion in ('full', 'share', 'single')
    self.dispersion = dispersion
    # parsing the units
    if not isinstance(units, (tuple, list)):
      units = [units]
    units = [int(i) for i in units]
    self._units = units
    # parsing the input distribution
    if not isinstance(xdist, (tuple, list)):
      xdist = [xdist]
    # check length of units is matching xdist,
    # i.e. consistent number of outputs
    assert len(units) == len(xdist), \
      "Given %d output(s) in 'units' which differnt from  " % len(units) + \
        "%d distribution(s) given in 'xdist'" % len(xdist)
    # check if mRNA output is zero inflated
    if 'zi' == xdist[0][:2].lower():
      self._is_zero_inflated = True
    else:
      self._is_zero_inflated = False
    # check if is semi-supervised
    self._is_semi_supervised = True if len(xdist) > 1 else False
    # for initializing the output layers
    self.xdist = []
    self.xloss = []
    self.xactiv = []
    for i in xdist:
      fn_dist, activation, fn_loss = _get_loss_and_distribution(i)
      self.xloss.append(fn_loss)
      self.xdist.append(fn_dist)
      self.xactiv.append(activation)

  @property
  def units(self):
    """ Always return a list of units for all outputs """
    return tuple(self._units)

  @property
  def n_outputs(self):
    return len(self.xdist)

  @property
  def log_norm(self):
    return self._log_norm

  @property
  def custom_objects(self):
    return [sisua_latents, sisua_networks]

  @property
  def is_zero_inflated(self):
    return self._is_zero_inflated

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

  @abstractmethod
  def _call(self, x, lmean, lvar, t, y, masks, training=None, n_samples=None):
    """
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
    masks : [batch_size, 1]
      binary mask for semi-supervised training
    training : `bool`
      flag mark training progress
    n_samples : {`None`, `int`}
      number of MCMC samples
    """
    raise NotImplementedError

  def call(self, inputs, training=None, n_samples=None):
    # check arguments
    if n_samples is None:
      n_samples = 1
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
        inputs = [inputs[0].X, inputs[0].local_mean, inputs[0].local_var
                 ] + inputs[1:]
      else:
        inputs = [
            inputs,
            tf.zeros(shape=(inputs.shape[0], 1)),
            tf.zeros(shape=(inputs.shape[0], 1))
        ] + inputs[1:]

    # Preprocessing the inputs for semi-supervised training
    x, lmean, lvar = inputs[:3]
    if self._is_fitting:
      y = list(inputs[3:-1])  # y1, y2, ...
      masks = [inputs[-1]] * len(y)  # same mask for all y
    else:
      y = list(inputs[3:])  # y1, y2, ...
      masks = []

    # check log normalization
    t = x
    if self.log_norm:
      x = tf.math.log1p(x)

    # postprocessing the returns
    outputs, latents = self._call(x, lmean, lvar, t, y, masks, training,
                                  n_samples)
    if not self.is_semi_supervised and isinstance(outputs, (tuple, list)):
      outputs = outputs[0]
    return outputs, latents

  def evaluate(self, inputs, n_samples=1, batch_size=128, verbose=1):
    raise Exception(
        "This method is not support, please use sisua.analysis.Posterior")

  def predict(self,
              inputs,
              n_samples=1,
              batch_size=64,
              apply_corruption=False,
              enable_cache=True,
              verbose=1):
    """
    Parameters
    ----------
    apply_corruption : `bool` (default=`False`)
      if `True` applying corruption on data before prediction to match the
      condition during fitting.
    enable_cache : `bool` (default=`True`)
      if `True` store the "footprint" of the input arguments to return the
      cached outputs

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
    assert len(inputs) == 1, \
      "During prediction phase, only the mRNA gene expression is provided, " +\
        "this is strict regulation for all models!"

    # checking the cache, this mechanism will significantly improve speed
    # during monitoring of fitting process
    self_id = id(self)
    footprint = ''.join([str(id(i.X)) for i in inputs]) + \
      str(n_samples) + str(apply_corruption) + str(self.epochs)
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
      n_samples=1,
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
      patience=25,  # for early stopping
      allow_rollback=True,  # for early stopping
      terminate_on_nan=True,  # for early stopping
      checkpoint=None,
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
        on_epoch_end=lambda *args, **kwargs: self._epochs.assign_add(1))
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
    if 'n_samples' in self._call_fn_args:
      self.call = partial(self.call, n_samples=n_samples)
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
                                     initial_epoch=self.epochs,
                                     steps_per_epoch=steps_per_epoch,
                                     validation_steps=validation_steps,
                                     epochs=self.epochs + epochs,
                                     verbose=verbose)
    logging.set_verbosity(curr_log)

    # reset to original state
    self.call = org_fn
    self._is_fitting = False

  def __repr__(self):
    return self.__str__()

  def __str__(self):
    return "<[%s]%s fitted:%s epoch:%s semi:%s>" % (
        self.__class__.__name__, self.name, self.epochs > 0, self.epochs,
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
      inputs: Union[SingleCellOMIC, Iterable[SingleCellOMIC]],
      params: dict = {
          'nlayers': scope.int(hp.choice('nlayers', [1, 2, 3, 4])),
          'hdim': scope.int(hp.choice('hdim', [32, 64, 128, 256, 512])),
          'zdim': scope.int(hp.choice('zdim', [32, 64, 128, 256, 512])),
      },
      loss_name: Text = 'val_loss',
      max_evals: int = 100,
      model_kwargs: dict = {},
      fit_kwargs: dict = {
          'epochs': 64,
          'batch_size': 128
      },
      algorithm: Text = 'bayes',
      seed: int = 8,
      save_path: Text = '/tmp/{model:s}_{data:s}_{loss:s}_{params:s}.hp',
      override: bool = True,
      verbose: bool = False):
    """ Hyper-parameters optimization for given SingleCellModel
    Parameters
    ----------
    model_kwargs : `dict`
      keyword arguments for model construction
    fit_kwargs : `dict`
      keyword arguments for `fit` method

    Example
    -------
    >>> callbacks = [
    >>>     NegativeLogLikelihood(),
    >>>     ImputationError(),
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
    if isinstance(loss_name, string_types):
      loss_name = [loss_name]
    loss_name = [str(i) for i in loss_name]

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
        'model':
            cls.id,
        'data':
            dsname.replace('_', ''),
        'loss':
            '_'.join(
                [i.replace('val_', '').replace('_', '') for i in loss_name]),
        'params':
            '_'.join(sorted([i.replace('_', '') for i in params.keys()]))
    }
    kw = {i: j for i, j in kw.items() if i in fmt}
    save_path = save_path.format(**kw)
    if os.path.exists(save_path) and not override:
      raise RuntimeError("Cannot override path: %s" % save_path)

    # ====== verbose mode ====== #
    if verbose:
      print(" ======== Tunning: %s ======== " % cls.__name__)
      print("Save path:", save_path)
      print("Model config:", model_kwargs)
      print("Fit config  :", fit_kwargs)
      print("Loss name   :", loss_name)
      print("Algorithm   :", algorithm)
      print("Max evals   :", max_evals)
      print("Search space:")
      for i, j in params.items():
        print("  ", i, j)

    def fit_and_evaluate(*args):
      kw = args[0]
      kw.update(model_kwargs)
      obj = cls(**kw)
      obj.fit(inputs, **fit_kwargs)
      history = obj.history.history
      all_loss = [history[name] for name in loss_name]
      # get min, variance and status, if NaN set to Inf
      loss = 0
      loss_variance = 0
      is_nan = False
      for l in all_loss:
        if np.any(np.isnan(l)):
          is_nan = True
          loss = np.inf
          loss_variance = np.inf
          break
        else:  # first epoch doesn't count
          loss += np.min(l[1:])
          loss_variance += np.var(l[1:])
      loss = loss / len(all_loss)
      loss_variance = loss_variance / len(all_loss)
      return {
          'loss': loss,
          'loss_variance': loss_variance,
          'history': history,
          'status': STATUS_FAIL if is_nan else STATUS_OK,
      }

    def hyperopt_run():
      trials = Trials()
      results = fmin(fit_and_evaluate,
                     space=params,
                     algo=tpe_suggest if algorithm == 'bayes' else rand_suggest,
                     max_evals=int(max_evals),
                     trials=trials,
                     rstate=np.random.RandomState(seed),
                     verbose=verbose)
      history = []
      for t in trials:
        r = t['result']
        history.append({
            'loss': r['loss'],
            'loss_variance': r['loss_variance'],
            'params': {i: j[0] for i, j in t['misc']['vals'].items()},
            'history': r['history'],
            'status': r['status'],
        })
      with open(save_path, 'wb') as f:
        print("Saving hyperopt results to: %s" % save_path)
        dill.dump((results, history), f)

    p = mpi.Process(target=hyperopt_run)
    p.start()
    p.join()

    try:
      with open(save_path, 'rb') as f:
        results, history = dill.load(f)
    except FileNotFoundError:
      results, history = {}, {}

    if verbose:
      print("Best:", results)
    return results, history
