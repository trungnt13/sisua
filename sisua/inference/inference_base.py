from __future__ import print_function, division, absolute_import
import os
import re
import io
import copy
import types
import pickle
import inspect
import warnings
from itertools import chain
from six import string_types
from functools import partial
from collections import OrderedDict, defaultdict

import numpy as np
from sklearn.base import BaseEstimator

import tensorflow as tf
from tensorflow.python.keras import Input
from tensorflow.python.keras.callbacks import LambdaCallback
from tensorflow.python.keras.layers import (Lambda as KerasLambda,
                                            Add as KerasAdd,
                                            Concatenate as KerasConcatenate)
from tensorflow.python.util.nest import flatten


from odin.fuel import Dataset
from odin.ml import fast_pca, fast_tsne
from odin.stats import train_valid_test_split, describe
from odin.utils import (get_script_path, ctext, batching, one_hot, uuid,
                        as_list, get_module_from_path, stdio,
                        get_formatted_datetime)
from odin import nnet as N, backend as K, training
from odin.backend.keras_helpers import (has_keras_meta, to_keras_objective,
                                        copy_keras_metadata)

import sisua
from sisua import is_verbose
from sisua.data import get_library_size, apply_artificial_corruption
from sisua.data.const import UNIVERSAL_RANDOM_SEED
from sisua.data import get_dataset, SingleCellOMIC
from sisua.utils import LearningCurves, plot_monitoring_epoch
from sisua.label_threshold import ProbabilisticEmbedding

from ._consts import PREDICTION_BATCH_SIZE

# The following tensors are allowed to return
possible_outputs = (
    'Z', 'Z_sample', # latent space
    'loss', # a tensor for training
    'metr', # metrics for monitoring
    # library size
    'L', 'L_sample', 'L_stddev',
    # W is predicted corrupted count (with dropout)
    'W', 'W_sample', 'W_stddev',
    # V is the cleaned count (without dropout)
    'V', 'V_sample', 'V_stddev',
    'y', 'y_sample', # predict values of protein markers
    'pi' # the zero inflated rate
)

def _signature_error(args):
  print("Function must contain following arguments:")
  print('  X      : [n_sample, n_gene]    input gene expression')
  print('  T      : [n_sample, n_gene]    target for the reconstruction')
  print('  L      : [n_sample, 1]         library size')
  print('  L_mean : [n_sample, 1]         mean of library size')
  print('  L_var  : [n_sample, 1]         variance of library')
  print('  mask   : [n_sample, 1]         mask for supervised training')
  print('  y      : [n_sample, n_protein] protein markers as labels')
  print('  nsample: []                    a scalar, number of MCMC sample')
  print('  nepoch : []                    a scalar, number of trained epoch')
  print('  configs: dict                  dictionary for the configuration')
  raise RuntimeError("But given: %s" % args)

def _normalize_tensor_name(name):
  name = name.split(':')[0].split('/')[-1].lower()
  name = re.sub(r"_\d+$", '', name) # e.g. 'loss_1'
  return name

# ===========================================================================
# Inference class
# ===========================================================================
class Inference(BaseEstimator):
  """ Performing inference given pre-defined model

  Notation
  --------
  X : [n_samples, n_genes]
      the gene expression matrix for single cell data, each row is
      one cell

  T : [n_samples, n_genes]
      the target variables for the reconstruction objective, each row
      is a corresponding cell data for `X`

  C : [n_samples, 1]
      the size factor (i.e. count-sum, library size) for each cell
      in each row

  y : [n_samples, n_proteins]
      the protein markers level for each single cell, each row is
      one cell

  Z : [n_samples, zdim]
      the latent space, mean of the latent distribution

  Parameters
  ----------
  model_name : string
      name of the model defined in `sisua.models`

  dispersion : {'gene', 'gene-cell'}
      'gene' - dispersion parameter is constant per gene across cells
      'gene-cell' - dispersion parameter can differ for every gene in every cell

  hdim : number of hidden units for each hidden layer

  zdim : number of latent dimension

  nlayer : number of layers for both encoder and decoder

  xnorm : {'raw', 'log', 'bin', 'prob'}
  tnorm : {'raw', 'log', 'bin', 'prob'}
  ynorm : {'raw', 'log', 'bin', 'prob'}
      different strategies for data normalization
      raw - keep the raw data (i.e. count or expression level)
      log - log-normalization
      bin - using `sisua.label_threshold.ProbabilisticEmbedding` to convert
            data to binary format
      prob - using `sisua.label_threshold.ProbabilisticEmbedding` to convert
             data to probability values

  xdist : the distribution name for X
  ydist : the distribution name for y
      support distribution include:
      'bernoulli' - Bernoulli distribution for binary data
      'zibernoulli' - Zero-inflated Bernoulli distribution for binary data
      'normal' - normal (or Gaussian) distribution
      'nb' - negative binomial for raw count data
      'zinb' - Zero-inflated negative binomial
      'poisson' - Poisson distribution for rare count event
      'zipoisson' - Zero-inflated Poisson
      'beta' - Beta distribution

  zdist : the distribution name for Z

  batchnorm : enable batch-normalization

  xdrop : dropout level for input X (from 0. to 1.)
  edrop : dropout level for encoder output (from 0. to 1.)
  zdrop : dropout level for latent space (from 0. to 1.)
  ddrop : dropout level for decoder output (from 0. to 1.)
      Dropout is enable if the value is greater than 0

  analytic : bool (default: True)
      using analytic KL-divergence

  kl_weight : scalar
      fixed weight for KL-divergence on latent

  warmup : integer (> 0)
      number of warmup epoch for KL-divergence

  extra_module_path : {string, None}
      path to folder contain the model script that you defined
      yourself, the .py file should has 'models_' prefix
      (e.g. 'models_vae.py')

  """

  def __init__(self, gene_dim, prot_dim=None,
               model='vae', dispersion='gene-cell',
               xnorm='log', tnorm='raw', ynorm='prob',
               xclip=0, yclip=0,
               xdist='zinb', ydist='bernoulli', zdist='normal',
               xdrop=0.3, edrop=0, zdrop=0, ddrop=0,
               hdim=128, zdim=32, nlayer=2,
               batchnorm=True, analytic=True,
               kl_weight=1., warmup=400, y_weight=10.,
               extra_module_path=None,
               **kwargs):
    super(Inference, self).__init__()
    assert int(warmup) > 0, "Number of warmup epoch must > 0"
    # ====== dispersion value ====== #
    assert dispersion in ('gene', 'gene-cell'), \
    "Only support 'gene' or 'gene-cell' dispersion"
    configs = dict(locals())
    # ====== modify the config ====== #
    primary_path = [os.path.join(os.path.dirname(inspect.getfile(sisua)), 'models')]
    extra_module_path = ([] if extra_module_path is None else
                         as_list(extra_module_path, t=string_types))
    configs['module_name'] = str(model)
    configs['module_path'] = primary_path + extra_module_path
    # remove unnecessary
    del configs['model']
    del configs['extra_module_path']
    del configs['self']
    del configs['kwargs']
    del configs['__class__']
    configs.update(kwargs)
    # select objective for multitask model
    # ce : categorical cross entropy
    # mse : mean squared error
    # ll : log loss
    # sg : multiple sigmoid loss
    configs['rec_loss'] = 'mse'
    if ynorm == 'raw' or ynorm == 'log':
      configs['cls_loss'] = 'mse'
    else:
      configs['cls_loss'] = 'sg'
    # ====== given the model name, searching for it ====== #
    if isinstance(model, string_types):
      all_error_while_loading = {}
      found = []
      for p in primary_path + extra_module_path:
        m, e = get_module_from_path(
            identifier=model,
            path=p,
            prefix='models_',
            return_error=True)
        all_error_while_loading[p] = e
        found += m
      if len(found) == 0:
        print("The following errors happened while loading script:")
        for i, j in all_error_while_loading.items():
          print('Path:', i)
          print(str(j))
        raise RuntimeError("No model found!")
      elif len(found) > 1:
        raise RuntimeError("Duplicated model found: %s" % str(model))
      self._model = found[0]
      if isinstance(self._model, N.NNOp):
        configs['is_keras_model'] = False
      else:
        self._model = self._model()
        configs['is_keras_model'] = True
    # error no support
    else:
      raise ValueError("No support for `model` with type: %s" % str(type(model)))

    # ====== basics ====== #
    self._training_log = ''
    self._history = {'train': defaultdict(list),
                     'valid': defaultdict(list)}
    self._configs = configs

    self._gmm_threshold = {}
    self._gene_dim = int(gene_dim)
    self._prot_dim = 0 if prot_dim is None else max(int(prot_dim), 0)
    self._is_fitted = False

    self._ps = 0
    self._trained_n_epoch = 0
    self._batch_size = 0
    self._learning_rate = 0
    self._n_mcmc_train = 0
    self._corruption_rate = None
    self._corruption_dist = None
    # ====== validate the model ====== #
    if self.is_keras_model:
      self._name = self._model.__class__.__name__
    else:
      self._name = self._model.name
    configs['model_name'] = self._name
    if hasattr(self._model, '__call__'):
      # X, T, L, L_mean, L_var, mask, y, nsample, nepoch, kwargs
      sign = inspect.getargspec(self._model.__call__
                                if self.is_keras_model else
                                self._model._func._func)
      func_args = sign.args
      if 'self' == func_args[0]:
        func_args = func_args[1:]
      require_args = ['X', 'T',
                      'L', 'L_mean', 'L_var',
                      'mask', 'y',
                      'nsample', 'nepoch', 'configs']
      assert func_args == require_args, _signature_error(sign.args)
    else:
      raise ValueError(
          "Design model must be callable! Found object: %s" % self._model)
    # ====== init ====== #
    self._reset()
    self._initialize_placeholders_and_outputs()

  def _reset(self):
    self.input_plhs = None
    self.n_mcmc_samples_var = None
    self.n_epochs_var = None
    self._outputs = defaultdict(dict)
    self._other_outputs = defaultdict(dict)
    self._pred_functions = {}
    self.score_functions = {}

  # ******************** pickling ******************** #
  def get_states(self):
    # clean the configs
    for k, v in list(self._configs.items()):
      if isinstance(v, types.ModuleType):
        del self._configs[k]
    if 'self' in self._configs:
      del self._configs['self']
    if 'kwargs' in self._configs:
      del self._configs['kwargs']
    states = (self._name,
              self._configs, self._history, self._training_log,
              self._gmm_threshold, self._gene_dim, self._prot_dim,
              self._is_fitted,
              self._ps,
              self._trained_n_epoch,
              self._batch_size,
              self._learning_rate,
              self._n_mcmc_train,
              self._corruption_rate, self._corruption_dist)
    return states

  def set_states(self, states):
    (self._name,
     self._configs, self._history, self._training_log,
     self._gmm_threshold, self._gene_dim, self._prot_dim,
     self._is_fitted,
     self._ps,
     self._trained_n_epoch,
     self._batch_size,
     self._learning_rate,
     self._n_mcmc_train,
     self._corruption_rate, self._corruption_dist) = states

  def __getstate__(self):
    if not self.is_keras_model:
      model = N.serialize(nnops=self._model, path=None,
                          save_variables=True,
                          binary_output=True)
    else:
      model = self._model.get_weights()
    return self.get_states(), model

  def __setstate__(self, states):
    states, model = states
    # set the states
    self.set_states(states)
    # reload the model
    if not self.is_keras_model:
      self._model = N.deserialize(model, force_restore_vars=True)
    else: # for keras, we have to search for the function again
      found = []
      for p in self.configs['module_path']:
        m, e = get_module_from_path(
            identifier=self.configs['module_name'],
            path=p,
            prefix='models_',
            return_error=True)
        found += m
      self._model = found[0](model)
    # make sure everything initialized
    self._reset()
    self._initialize_placeholders_and_outputs()

  # ******************** initialization ******************** #
  def _initialize_placeholders_and_outputs(self):
    if self.input_plhs is not None and \
    len(self._outputs) > 0:
      return

    with tf.variable_scope(self.name + '_placeholders'):
      X_plh = Input(shape=(self.gene_dim,), dtype='float32', name='Gene_input')
      T_plh = Input(shape=(self.gene_dim,), dtype='float32', name="Gene_target")
      # mask for supervised training
      if self.prot_dim > 0:
        mask_plh = Input(shape=(), dtype='float32', name='mask')
        y_plh = Input(shape=(self.prot_dim,), dtype='float32', name='Protein')
      else:
        mask_plh = None
        y_plh = None
      # size factor
      L_plh = Input(shape=(1,), dtype='float32', name='CellSize')
      L_mean_plh = Input(shape=(1,), dtype='float32', name="CellSize_mean")
      L_var_plh = Input(shape=(1,), dtype='float32', name="CellSize_variance")
      # number of sample for MCMC
      self.n_mcmc_samples_var = K.variable(1, dtype='int32',
                                           name='n_mcmc_samples',
                                           initialize=True)
      self.n_epochs_var = K.variable(self._trained_n_epoch, dtype='float32',
                                     name='n_epochs',
                                     initialize=True)
    # all input in specific order
    self.input_plhs = (X_plh, T_plh,
                       L_plh, L_mean_plh, L_var_plh,
                       mask_plh, y_plh)
    if is_verbose():
      print(ctext("Input placeholders:", 'lightyellow'))
      for i in self.input_plhs:
        print(" *", ctext(i, 'cyan'))
    # ====== Keras ====== #
    input_dicts = {
        'X': X_plh, 'T': T_plh,
        'L': L_plh, 'L_mean': L_mean_plh, 'L_var': L_var_plh,
        'mask': mask_plh, 'y': y_plh,
        'nsample': self.n_mcmc_samples_var,
        'nepoch': self.n_epochs_var,
        'configs': self.configs
    }
    if self.is_keras_model:
      input_dicts['configs']['training'] = True
      outputs_train = self._model(**input_dicts)
      input_dicts['configs']['training'] = False
      outputs_test = self._model(**input_dicts)
    # ====== ODIN ====== #
    else:
      all_nnops = N.get_all_nnops(scope=self.name)
      for op in all_nnops:
        if isinstance(op, N.Container):
          op.set_debug(is_verbose())
      # applying to get the output
      outputs_train = self._model(**input_dicts)
      outputs_test = outputs_train
    # ====== compulsory outputs ====== #
    assert sorted(list(outputs_train.keys())) == sorted(list(outputs_test.keys())),\
    "Output for training and testing must be the same set of Tensor"
    assert 'Z' in outputs_train and 'Z' in outputs_test, \
    "Latent space must be in the outputs"
    assert 'loss' in outputs_train and 'loss' in outputs_test, \
    "Loss must be given in the outputs for training"
    if self.is_semi_supervised:
      assert 'y' in outputs_test, \
      "Require output `y` if model is semi-supervised"

    for output_idx, (output_type, outputs) in enumerate(
        zip(('train', 'test'), (outputs_train, outputs_test))):
      # ====== Custom outputs ====== #
      for k, v in outputs.items():
        if k not in possible_outputs:
          self._other_outputs[output_type][k] = v
      # ====== latent space output ====== #
      Z = outputs['Z']
      Z_sample = outputs.get('Z_sample', None)
      assert Z.get_shape().as_list()[-1] == int(self.zdim), \
          "Expect %d latent dimension but return Z with shape: %s" % \
          (int(self.zdim), Z.get_shape().as_list())
      # ====== Loss and metrics ====== #
      # make sure the loss is named loss
      loss = copy_keras_metadata(outputs['loss'],
                                 tf.identity(outputs['loss'], name='loss'))
      if loss.shape.ndims != 0:
        raise RuntimeError("loss must be scalar, but given: %s" % str(loss))
      metr = outputs.get('metr', [])
      if K.is_tensor(metr, inc_distribution=False):
        metr = [metr]
      assert isinstance(metr, list)
      # all metrics must be scalar
      for m in metr:
        if m.shape.ndims != 0:
          raise RuntimeError(
              "Returned metrics must be scalar, but given: %s" % str(m))
      # ====== library size ====== #
      L_expected = outputs.get('L', None)
      L_sample = outputs.get('L_sample', None)
      L_stddev = outputs.get('L_stddev', None)
      # ====== reconstructed ====== #
      W_expected = outputs.get('W', None)
      W_sample = outputs.get('W_sample', None)
      W_stddev = outputs.get('W_stddev', None)
      # ====== imputed ====== #
      V_expected = outputs.get('V', None)
      V_sample = outputs.get('V_sample', None)
      V_stddev = outputs.get('V_stddev', None)
      # ====== zero-inflated pi ====== #
      pi = outputs.get('pi', None)
      # ====== predicted labels ====== #
      y = outputs.get('y', None)
      y_sample = outputs.get('y_sample', None)
      if is_verbose():
        print(ctext("'%s' outputs:" % output_type, 'lightyellow'))
        print("   Latent           :", ctext(Z, 'cyan'))
        print("   Latent (sample)  :", ctext(Z_sample, 'cyan'))
        print("   zero-inflated PI :", ctext(pi, 'cyan'))
        print(ctext("  Labels:", 'lightyellow'))
        print("   Prediction       :", ctext(y, 'cyan'))
        print("   Sample           :", ctext(y_sample, 'cyan'))
        print(ctext("  Library size:", 'lightyellow'))
        print("   L expected       :", ctext(L_expected, 'cyan'))
        print("   L sample         :", ctext(L_sample, 'cyan'))
        print("   L stddev         :", ctext(L_stddev, 'cyan'))
        print(ctext("  Reconstruction:", 'lightyellow'))
        print("   W expected       :", ctext(W_expected, 'cyan'))
        print("   W sample         :", ctext(W_sample, 'cyan'))
        print("   W stddev         :", ctext(W_stddev, 'cyan'))
        print(ctext("  Imputed:", 'lightyellow'))
        print("   V expected       :", ctext(V_expected, 'cyan'))
        print("   V sample         :", ctext(V_sample, 'cyan'))
        print("   V stddev         :", ctext(V_stddev, 'cyan'))
        print(ctext("  Training info:", 'lightyellow'))
        print("   Loss    :", ctext(loss, 'cyan'))
        print("   Metrics :", ctext(metr, 'cyan'))
      self._outputs[output_type] = {
          'z': Z, 'z_sample': Z_sample,
          'pi': pi,
          'y': y, 'y_sample': y_sample,
          'l': L_expected, 'l_sample': L_sample, 'l_stddev': L_stddev,
          'w': W_expected, 'w_sample': W_sample, 'w_stddev': W_stddev,
          'v': V_expected, 'v_sample': V_sample, 'v_stddev': V_stddev,
          'loss': loss,
          'metr': metr
      }
    # initialize everything
    K.initialize_all_variables()

  def _initialize_predict_functions(self):
    if len(self._pred_functions) > 0:
      return self._pred_functions
    # ====== inputs ====== #
    inputs = list(self.input_plhs)[:-2]
    metrics = sorted(
        set(as_list(self._outputs['test']['loss']) +
            as_list(self._outputs['test']['metr'])),
        key=lambda x: x.name)
    self._metrics_name = [_normalize_tensor_name(i.name)
                          for i in metrics]

    def output_iter():
      for name, tensor in chain(self._outputs['test'].items(),
                                self._other_outputs['test'].items()):
        if tensor is None or \
        (isinstance(tensor, (tuple, list)) and len(tensor) == 0):
          continue
        if name in ('loss', 'metr', 'y'):
          continue
        yield name, tensor

    # ====== for Keras ====== #
    if self.is_keras_model:
      from tensorflow.python.keras import Model
      # unsupervised
      for name, tensor in output_iter():
        m = Model(inputs=inputs, outputs=tensor,
                  name="predict_%s" % name)
        if '_sample' in name: # special case for sample
          # NOTE: no batch support (careful memory)
          f = m.predict_on_batch
        else: # other output
          f = partial(m.predict, batch_size=PREDICTION_BATCH_SIZE, verbose=0)
        self._pred_functions[name] = f
      # metrics
      self._pred_functions['metrics'] = partial(
          Model(inputs=[i for i in self.input_plhs if i is not None],
                outputs=metrics,
                name="predict_metrics").predict,
          batch_size=PREDICTION_BATCH_SIZE, verbose=0)
      # semi-supervised
      if self.is_semi_supervised:
        # it is notable that prediction of y must not require
        # y as input data, we want to predict y not denoising y
        m = Model(inputs=inputs,
                  outputs=self._outputs['test']['y'],
                  name="predict_y")
        self._pred_functions['y'] = partial(
            m.predict, batch_size=PREDICTION_BATCH_SIZE, verbose=0)
      return self._pred_functions

    # ====== for ODIN ====== #
    def create_func(ins_, out_, is_scoring=False):
      if isinstance(out_, (tuple, list)):
        out_ = [o for o in out_ if o is not None]
        if len(out_) == 0:
          out_ = None
      if out_ is not None:
        fn = K.function(
            inputs=ins_, outputs=out_, training=False)

        # this function will iterate over mini-batch for prediction
        # make sure the order is preserved, so no shuffling
        def _batch_prediction(X):
          assert len(set(x.shape[0] for x in X)) == 1
          n = X[0].shape[0]
          all_y = []
          for start, end in batching(batch_size=PREDICTION_BATCH_SIZE,
                                     n=n, seed=None):
            x = [i[start:end] for i in X]
            y = fn(*x)
            all_y.append(y)
          # concatenate the results
          if not is_scoring:
            if isinstance(y, (tuple, list)):
              return [np.concatenate([j[i] for j in all_y], axis=0)
                      for i in range(len(y))]
            else:
              return np.concatenate(all_y, axis=0)
          # scoring mode (return scalar)
          else:
            return np.mean(np.array(all_y), axis=0)
        return _batch_prediction
      return None

    # unsupervised
    for name, tensor in output_iter():
      self._pred_functions[name] = create_func(ins_=inputs, out_=tensor)
    # metrics
    self._pred_functions['metrics'] = create_func(
        ins_=[i for i in self.input_plhs if i is not None],
        out_=metrics,
        is_scoring=True)
    # semi-supervised
    if self.is_semi_supervised:
      # it is notable that prediction of y must not require
      # y as input data, we want to predict y not denoising y
      self._pred_functions['y'] = create_func(
          ins_=inputs,
          out_=self._outputs['test']['y'])
    return self._pred_functions

  # ******************** helper ******************** #
  def normalize(self, x, method, data_name):
    x = np.atleast_2d(x)
    if method == 'raw':
      pass
    elif method == 'log':
      x = K.log_norm(x, axis=1, scale_factor=10000)
    elif method == 'bin' or method == 'prob':
      is_binary_classes = sorted(np.unique(x.astype('float32'))) == [0., 1.]
      if not is_binary_classes:
        if data_name not in self._gmm_threshold:
          gmm = ProbabilisticEmbedding()
          gmm.fit(x)
          self._gmm_threshold[data_name] = gmm
        else:
          gmm = self._gmm_threshold[data_name]
        x = gmm.predict(x) if method == 'bin' else gmm.predict_proba(x)
      else: # already binarized or probabilized
        pass
    else:
      raise NotImplementedError
    return x

  def _preprocess_inputs(self, X, y):
    """
    Return
    ------
    X_norm, T_norm, L, L_mean, L_var, y_norm

    Note
    ----
    y_norm could be None
    """
    assert X.shape[1] == self.gene_dim, \
    "This model require %d gene expression, but given X.shape=%s" % \
    (self.gene_dim, str(X.shape))

    if self.is_semi_supervised and y is not None:
      assert y.shape[1] == self.prot_dim,\
      "This model require %d protein markers, but given y.shape=%s" % \
      (self.prot_dim, str(y.shape))
    # ====== gene expression ====== #
    if X.ndim == 1:
      X = np.expand_dims(X, axis=-1)
    if self.xclip > 0:
      X = np.clip(X, a_min=0, a_max=float(self.xclip))
    # ====== library size ====== #
    L, L_mean, L_var = get_library_size(X, return_library_size=True)
    X_norm = self.normalize(X, method=self.xnorm, data_name='X')
    T_norm = self.normalize(X, method=self.tnorm, data_name='X')
    assert self._gene_dim == X_norm.shape[1], "Number of genes mismatch"
    # ====== protein ====== #
    y_norm = None
    if self.is_semi_supervised and y is not None:
      if y.ndim == 1:
        y = np.expand_dims(y, axis=-1)
      if self.yclip > 0:
        y = np.clip(y, a_min=0, a_max=float(self.yclip))

      y_norm = self.normalize(y, method=self.ynorm, data_name='y')
      assert X_norm.shape[0] == y_norm.shape[0]
    # ====== return ====== #
    return X_norm, T_norm, L, L_mean, L_var, y_norm

  # ******************** fitting ******************** #
  def fit(self, X, y=None,
          supervised_percent=0.8, validation_percent=0.1,
          n_mcmc_samples=1,
          corruption_rate=0.25, corruption_dist='binomial',
          batch_size=64, n_epoch=120, learning_rate=1e-4,
          monitoring=False, fig_dir=None,
          detail_logging=False):
    """

    Parameters
    ----------
    X : [n_samples, n_genes]
        single-cell gene expression matrix

    y : [n_samples, n_protein]
        single-cell protein marker matrix, if not given,
        assume to be the same as `X`, and might not be used
        by the model

    supervised_percent : float (0. to 1.)
        percent of training data used for supervised objective

    validation_percent : float (0. to 1.)
        percent of training data used for validation at the
        end of each epoch, if 0. then no validation is performed

    n_mcmc_samples : int (> 0)
        number of MCMC samples for training

    batch_size : int (default: 64)
        batch size for training

    n_epoch : int (default: 120)
        number of epoch for training

    learning_rate : float (default: 0.0001)
        learning rate for Adam

    monitoring : bool
        enable monitoring each epoch

    fig_dir : {None, string}
        path for saving the training summary figure,
        if not given, then no training summary is stored

    detail_logging : bool (default: False)
        if True, print loss of the monitoring metrics
        during training, otherwise, only so the current
        epoch progress

    """
    training_log = io.StringIO()
    stdio(training_log)

    if self._corruption_rate is None:
      self._corruption_rate = corruption_rate
      self._corruption_dist = corruption_dist
    else:
      assert (self._corruption_rate == corruption_rate and
              self._corruption_dist == corruption_dist),\
      ("Model have been fitted with following corruption configurations: %s, %s;"
       "but given: %s, %s" % (self._corruption_rate, self._corruption_dist,
                              corruption_rate, corruption_dist))
    n_samples = X.shape[0]
    self._ps = supervised_percent
    self._batch_size = batch_size
    self._learning_rate = learning_rate
    self._n_mcmc_train = n_mcmc_samples
    # apply artificial corruption for training
    X = apply_artificial_corruption(
        X, dropout=corruption_rate, distribution=corruption_dist)
    # ====== check semi-supervised system ====== #
    if self.is_semi_supervised and y is None:
      raise ValueError(
          "`y` must be not None for fitting semi-supervised model")
    X, T, L, L_mean, L_var, y = self._preprocess_inputs(X, y)
    # ====== splitting train valid ====== #
    (X_train, T_train,
     L_train, Lmean_train, Lvar_train,
     y_train) = None, None, None, None, None, None
    (X_valid, T_valid,
     L_valid, Lmean_valid, Lvar_valid,
     y_valid) = None, None, None, None, None, None

    if validation_percent > 0:
      train_ids, valid_ids = train_valid_test_split(
          x=np.arange(n_samples, dtype='int32'),
          train=1 - validation_percent,
          inc_test=False,
          seed=UNIVERSAL_RANDOM_SEED)
      X_train = X[train_ids]
      X_valid = X[valid_ids]

      T_train = T[train_ids]
      T_valid = T[valid_ids]

      L_train = L[train_ids]
      L_valid = L[valid_ids]
      Lmean_train = L_mean[train_ids]
      Lmean_valid = L_mean[valid_ids]
      Lvar_train = L_var[train_ids]
      Lvar_valid = L_var[valid_ids]

      if y is not None:
        y_train = y[train_ids]
        y_valid = y[valid_ids]
    else: # no validation set given
      X_train = X
      T_train = T
      L_train = L
      Lmean_train = L_mean
      Lvar_train = L_var
      if y is not None:
        y_train = y

    n_train = len(X_train)
    n_valid = 0 if X_valid is None else len(X_valid)
    # ====== supervised mask ====== #
    # Generate a mask of 0, 1
    # : 0 mean unsupervised sample,
    # : 1 mean supervised sample
    rand = np.random.RandomState(seed=UNIVERSAL_RANDOM_SEED)
    m_train = np.zeros(shape=n_train, dtype='float32')
    supervised_indices = rand.choice(
        np.arange(0, n_train),
        size=int(supervised_percent * n_train),
        replace=False)
    m_train[supervised_indices] = 1
    assert int(np.sum(m_train)) == int(supervised_percent * n_train)
    # we also need mask for validation but take into acocunt all
    # validation sample to evaluate supervised tasks
    m_valid = np.ones(shape=(n_valid,), dtype='float32')
    # ====== prepare the placeholders ====== #
    inputs = [i for i in self.input_plhs
              if i is not None]
    train_data = (X_train, T_train,
                  L_train, Lmean_train, Lvar_train,
                  m_train, y_train)
    valid_data = None if n_valid is 0 else\
    (X_valid, T_valid,
     L_valid, Lmean_valid, Lvar_valid,
     m_valid, y_valid)
    if not self.is_semi_supervised:
      train_data = train_data[:-2]
      if valid_data is not None:
        valid_data = valid_data[:-2]
    # ====== set value of the variable ====== #
    assert int(n_mcmc_samples) >= 1, \
    "Number of MCMC samples must be >= 1"
    K.set_value(self.n_mcmc_samples_var, int(n_mcmc_samples))
    # ====== training for keras ====== #
    if self.is_keras_model:
      from tensorflow.python.keras import Model
      from tensorflow.python.keras.optimizers import Adam
      if self.is_semi_supervised:
        model = Model(inputs=inputs,
          outputs=KerasConcatenate(axis=-1)([self._outputs['train']['w'],
                                             self._outputs['train']['y']]))
      else:
        model = Model(inputs=inputs,
                      outputs=self._outputs['train']['w'])
      optz = Adam(lr=learning_rate)

      if is_verbose():
        print(ctext("Trainable variables:", 'lightyellow'))
        for v in sorted(model.trainable_variables, key=lambda x: x.name):
          print(" ", v.name, v.shape)
        print(ctext("None-trainable variables:", 'lightyellow'))
        for v in sorted(model.non_trainable_variables, key=lambda x: x.name):
          print(" ", v.name, v.shape)

      model.compile(optz,
          loss=to_keras_objective(self._outputs['train']['loss']),
          metrics=[to_keras_objective(i)
                 for i in self._outputs['train']['metr']])
      hist = model.fit(x=train_data, y=None,
          batch_size=batch_size, epochs=n_epoch,
          verbose=1 if detail_logging else 0,
          callbacks=[LambdaCallback(
              on_epoch_begin=lambda *args, **kwargs:
              K.set_value(self.n_epochs_var, self.n_epochs_var + 1))],
          validation_data=(valid_data, None),
          shuffle=True)
      # extract the history
      for k, v in hist.history.items():
        name = k
        hist_type = 'train'
        if 'val_' == k[:4]:
          name = k[4:]
          hist_type = 'valid'
        # normalize all the tensor name
        self._history[hist_type][_normalize_tensor_name(name)] += list(v)
    # ====== training for O.D.I.N ====== #
    else:
      # create the optimizer
      optz = K.optimizers.Adam(lr=float(learning_rate), clipnorm=None,
                               name=self.name)
      updates = optz.minimize(self._outputs['train']['loss'],
                              verbose=is_verbose())
      global_norm = optz.norm
      K.initialize_all_variables()
      # ====== training functions ====== #
      # for training
      tmp = [self._outputs['train']['loss'], global_norm]
      for i in self._outputs['train']['metr']:
        if i not in tmp:
          tmp.append(i)
      f_train = K.function(inputs=inputs, outputs=tmp, training=True,
                           updates=updates)
      # for scoring
      tmp = [self._outputs['train']['loss']]
      for i in self._outputs['train']['metr']:
        if i not in tmp:
          tmp.append(i)
      f_score = K.function(inputs=inputs, outputs=tmp, training=False,
                           batch_size=None)
      # overlapping metrics for summary
      overlap_metrics = sorted(set([self._outputs['train']['loss']] +
                                   self._outputs['train']['metr']),
                               key=lambda x: x.name)
      if is_verbose():
        print(ctext("Metrics for monitoring:", 'lightyellow'))
        for m in overlap_metrics:
          print("  %s:" % ctext(m.name, 'cyan'), m)
      # ====== create training loop ====== #
      trainer = training.MainLoop(batch_size=int(batch_size),
                                  seed=UNIVERSAL_RANDOM_SEED, shuffle_level=2,
                                  allow_rollback=False,
                                  verbose=2 if bool(detail_logging) else 3,
                                  labels=None)
      trainer.set_checkpoint(obj=self._model)
      # ====== callbacks ====== #
      all_callbacks = [
          training.LambdaCallback(
              fn=lambda:K.set_value(self.n_epochs_var, self.n_epochs_var + 1),
              task_name='train',
              signal=training.TaskSignal.EpochStart),
          training.NaNDetector(),
          training.CheckpointGeneralization('valid',
                                      output_name=self._outputs['train']['loss']),
          # epoch summaries
          None if fig_dir is None else LearningCurves(save_path=fig_dir),
          # monitoring latent spaces
          training.LambdaCallback(fn=self.monitoring_epoch, task_name='train',
                                  signal=training.TaskSignal.EpochEnd)
          if bool(monitoring) else None,
      ]
      trainer.set_callbacks(all_callbacks)
      # ====== training task ====== #
      trainer.set_train_task(func=f_train,
                             data=train_data,
                             epoch=int(n_epoch),
                             name='train')
      if X_valid is not None:
        trainer.set_valid_task(func=f_score,
                               data=valid_data,
                               freq=training.Timer(percentage=1.0),
                               name='valid')
      # NOTE: this line is important
      trainer.run()
      # ====== store the history ====== #
      hist = trainer.history
      for task_name, task_results in hist.items():
        for epoch_id, epoch_results in task_results.items():
          for tensor_name, tensor_batch in epoch_results.items():
            tensor_name = _normalize_tensor_name(tensor_name)
            self._history[task_name][tensor_name].append(np.mean(tensor_batch))
    # ====== end training update states ====== #
    self._trained_n_epoch += n_epoch
    self._is_fitted = True
    stdio()
    training_log.seek(0)
    self._training_log += '\n%s\n' % get_formatted_datetime(only_number=False)
    self._training_log += training_log.read()
    training_log.close()

  # ******************** history ******************** #
  def get_train_loss(self):
    return self._history['train']['loss']

  def get_valid_loss(self):
    return self._history['valid']['loss']

  def get_train_history(self, name):
    return self._history['train'][name]

  def get_valid_history(self, name):
    return self._history['valid'][name]

  # ******************** plotting utils ******************** #
  def plot_tsne(self, X, y=None,
                labels=None, labels_name=None,
                n_samples=1000, n_mcmc_samples=100, output_type='Z',
                show_pca=False, ax=None, title=None, seed=5218):
    """
    Parameters
    ----------
    output_type : string
        Z : latent space
        W : reconstructed gene expression
        V : denoised gene expression
        PI : Zero-inflated rate
    """
    X, T, C, y = self._preprocess_inputs(X, y)
    if labels is None:
      labels = np.ones(shape=(X.shape[0], 1))
    if labels.ndim == 1:
      labels = np.expand_dims(labels, axis=-1)
    if np.max(labels) > 1 and labels.shape[1] == 1:
      labels = one_hot(y=labels.ravel(),
        nb_classes=np.max(labels) + 1
        if labels_name is None else
        len(labels_name))
    assert len(labels) == X.shape[0], \
    "Number of samples mismatch between X and labels"

    n_classes = labels.shape[1]
    if labels_name is None:
      labels_name = np.array(['#%d' % i for i in range(n_classes)])
    # ====== downsampling ====== #
    rand = np.random.RandomState(seed=seed)
    if n_samples is not None:
      ids = rand.permutation(X.shape[0])
      ids = rand.choice(ids, size=n_samples, replace=False)
      X = X[ids]
      T = T[ids]
      C = C[ids]
      y = y[ids]
      if labels is not None:
        labels = labels[ids]
    # ====== make prediction ====== #
    f = self._initialize_predict_functions()
    mask = np.ones(shape=(X.shape[0],))
    Z = f[str(output_type).lower()](X, T, C, mask, y)
    if isinstance(Z, (tuple, list)):
      Z = Z[0]
    if Z is None:
      raise RuntimeError(
          "Model has no support for output type: '%s'" % output_type)
    # ====== apply pca and tsne ====== #
    is_single_label = np.allclose(np.max(labels, axis=1), 1)
    Z_pca = fast_pca(Z, n_components=min(512, Z.shape[1]), algo='pca',
                     random_state=rand.randint(10e8))
    if not bool(show_pca):
      Z_tsne = fast_tsne(Z_pca, n_components=2,
                         random_state=rand.randint(10e8))
    else:
      Z_tsne = None
    # ====== plot ====== #
    from odin.visual import (plot_scatter, plot_figure, plot_scatter_heatmap)
    if is_single_label:
      plot_scatter(x=Z_pca[:, :2] if bool(show_pca) else Z_tsne,
                   color=[labels_name[i] for i in np.argmax(labels, axis=-1)],
                   size=6, ax=ax, grid=False,
                   legend_enable=True, legend_ncol=4, fontsize=8,
                   title=title)
    else:
      colormap = 'Reds' # bwr
      ncol = 5 if n_classes <= 20 else 9
      nrow = int(np.ceil(n_classes / ncol))
      fig = plot_figure(nrow=4 * nrow, ncol=20)
      for i, name in enumerate(labels_name):
        val = K.log_norm(labels[:, i], axis=0)
        plot_scatter_heatmap(
            x=Z_pca[:, :2] if bool(show_pca) else Z_tsne,
            val=val / np.sum(val),
            ax=(nrow, ncol, i + 1),
            colormap=colormap, size=8, alpha=0.8,
            fontsize=8, grid=False,
            title='[%s]%s' % ("PCA" if show_pca else "t-SNE", name))

      import matplotlib as mpl
      cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
      cmap = mpl.cm.get_cmap(name=colormap)
      norm = mpl.colors.Normalize(vmin=0., vmax=1.)
      cb1 = mpl.colorbar.ColorbarBase(cbar_ax, cmap=cmap,
                                      norm=norm,
                                      orientation='vertical')
      cb1.set_label('Protein markers level')

  # ******************** predicting ******************** #
  def _make_prediction(self, pred_type, X, y=None, n_mcmc_samples=100):
    K.set_value(self.n_mcmc_samples_var, n_mcmc_samples)
    X_norm, T_norm, L, L_mean, L_var, y_norm = self._preprocess_inputs(X, y)
    mask = np.ones(shape=(X.shape[0],))
    f = self._initialize_predict_functions()
    key = str(pred_type).lower()
    if key not in f:
      raise RuntimeError("No prediction for '%s'" % pred_type)
    f = f[key]
    if y is None:
      return f((X_norm, T_norm, L, L_mean, L_var))
    else:
      return f((X_norm, T_norm, L, L_mean, L_var, mask, y))

  # ******************** scoring ******************** #
  def score(self, X, y=None, n_mcmc_samples=100, return_mean=True):
    """Compute the per-sample average of recorded metrics of the given data X.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_dimensions)
        List of n_features-dimensional data points. Each row
        corresponds to a single data point.

    Returns
    -------
    dictionary : metric_name -> values
    """
    if not self.is_semi_supervised:
      y = None
    scores = self._make_prediction(
        'metrics', X=X, y=y, n_mcmc_samples=n_mcmc_samples)
    if self.is_keras_model:
      scores = [np.mean(i) for i in scores]
    return {_normalize_tensor_name(name): s
            for name, s in zip(self._metrics_name, scores)}

  # ******************** sampling function ******************** #
  def _sample(self, name, X, n_mcmc_samples):
    output = []
    for s, e in batching(batch_size=PREDICTION_BATCH_SIZE, n=X.shape[0]):
      x = X[s:e]
      output.append(
          self._make_prediction(pred_type=name, X=x, n_mcmc_samples=n_mcmc_samples))
    return np.concatenate(output, axis=1)

  def sample_Z(self, X, n_mcmc_samples=1):
    return self._sample('Z_sample', X, n_mcmc_samples)

  def sample_y(self, X, n_mcmc_samples=1):
    assert self.is_semi_supervised,\
    "Only semi-supervised model has this prediction!"
    return self._sample('y_sample', X, n_mcmc_samples)

  def sample_W(self, X, n_mcmc_samples=1):
    return self._sample('W_sample', X, n_mcmc_samples)

  def sample_V(self, X, n_mcmc_samples=1):
    return self._sample('V_sample', X, n_mcmc_samples)

  def sample_L(self, X, n_mcmc_samples=1):
    try:
      return self._sample('L_sample', X, n_mcmc_samples)
    except RuntimeError:
      return np.sum(self.sample_V(X, n_mcmc_samples), axis=-1, keepdims=True)

  # ******************** predicting ******************** #
  def predict_Z(self, X):
    """ Return mean of the latent posterior
    (i.e. mean of Normal distribution) """
    return self._make_prediction('Z', X, n_mcmc_samples=1)

  def predict_y(self, X, n_mcmc_samples=100):
    """ Predicting the protein marker """
    # and y is not None
    assert self.is_semi_supervised,\
    "Only semi-supervised model has this prediction!"
    return self._make_prediction(
        pred_type='y', X=X, y=None, n_mcmc_samples=n_mcmc_samples)

  def predict_L(self, X, n_mcmc_samples=100):
    """ Return library size prediction """
    try:
      return self._make_prediction('L', X, n_mcmc_samples=n_mcmc_samples)
    except RuntimeError:
      return np.sum(self._make_prediction('V', X, n_mcmc_samples=n_mcmc_samples),
                    axis=-1, keepdims=True)

  def predict_Lstddev(self, X, n_mcmc_samples=100):
    """ Return library size stddev prediction """
    try:
      return self._make_prediction('L_stddev', X, n_mcmc_samples=n_mcmc_samples)
    except RuntimeError:
      return np.sum(self._make_prediction('V_stddev', X, n_mcmc_samples=n_mcmc_samples),
                    axis=-1, keepdims=True)

  def predict_W(self, X, n_mcmc_samples=100):
    """ Return mean of the reconstructed value """
    return self._make_prediction('W', X, n_mcmc_samples=n_mcmc_samples)

  def predict_Wstddev(self, X, n_mcmc_samples=100):
    """ Return stddev of reconstructed value
    if not a variational model, then V_stddev is None
    """
    return self._make_prediction('W_stddev', X, n_mcmc_samples=n_mcmc_samples)

  def predict_V(self, X, n_mcmc_samples=100):
    """ Return mean of the denoised value """
    return self._make_prediction('V', X, n_mcmc_samples=n_mcmc_samples)

  def predict_Vstddev(self, X, n_mcmc_samples=100):
    """ Return stddev of denoised value
    if not a variational model, then V_stddev is None
    """
    return self._make_prediction('V_stddev', X, n_mcmc_samples=n_mcmc_samples)

  def predict_PI(self, X, n_mcmc_samples=100):
    """ Return a matrix (n_sample, n_gene) of Zero-inflated
    rate

    if not a zero-inflated model, then return None
    """
    return self._make_prediction('PI', X, n_mcmc_samples=n_mcmc_samples)

  def predict(self, X, n_mcmc_samples=100):
    """ Return denoised values, same as `predict_V` """
    return self.predict_V(X, n_mcmc_samples)

  # ******************** properties ******************** #
  @property
  def id(self):
    """ Unique ID for this inference class """
    assert self.is_fitted, "Only fitted Inference has ID"
    name = self.configs['module_name'] if 'module_name' in self.configs else \
    self.name
    name = name.strip().lower()

    kl_weight_text = '%.2f' % self.kl_weight
    kl_weight_text = kl_weight_text.split('.')
    if int(kl_weight_text[1]) == 0:
      kl_weight_text[1] = ''
    kl_weight_text = kl_weight_text[0] + kl_weight_text[1][:2]

    y_weight = '%.2g' % self.configs['y_weight']
    y_weight = y_weight.replace('.', '')

    model_id = '_'.join([
        name,
        'gene' if self.dispersion == 'gene' else 'genecell',
        'X%s%d%s' % (self.xnorm, int(max(self.xclip, 0)), self.xdist),
        'Y%s%d%s' % (self.ynorm, int(max(self.yclip, 0)), self.ydist),
        'T%s' % self.tnorm,
        'Z' + str(self.zdist),
        '%.3dspvs' % (self.supervised_percent * 100) + y_weight,
        'net%.2d%.3d%.3d' % (self.nlayer, self.hdim, self.zdim),
        'drop%.2d%.2d%.2d%.2d' % (self.xdrop * 100,
                                  self.edrop * 100,
                                  self.zdrop * 100,
                                  self.ddrop * 100),
        '%skl%d' % (kl_weight_text, int(self.warmup)),
        'alytcT' if bool(self.analytic) else 'alytcF',
        'bnormT' if bool(self.batchnorm) else 'bnormF',
        '%s%.2d' % (self.corruption_dist, self.corruption_rate * 100)
    ])
    return model_id

  @property
  def short_id(self):
    """ Shorter id, remove all the training configuration """
    return '_'.join(self.id.split('_')[:-5])

  @property
  def kl_weight(self):
    return float(self._configs['kl_weight'])

  @property
  def warmup(self):
    """ Number of warmup epoch for KL divergence """
    return float(self._configs['warmup'])

  @property
  def name(self):
    return self._name

  @property
  def nlayer(self):
    return self._configs['nlayer']

  @property
  def hdim(self):
    """ Number of hidden units for intermediate layers """
    return self._configs['hdim']

  @property
  def zdim(self):
    """ Number of hidden units for the bottleneck layer """
    return self._configs['zdim']

  @property
  def inputs(self):
    return [i for i in self.input_plhs if i is not None]

  @property
  def outputs(self):
    o = {}
    o['train'] = dict(self._outputs['train'])
    o['test'] = dict(self._outputs['test'])
    o['train'].update(self._other_outputs['train'])
    o['test'].update(self._other_outputs['test'])
    return o

  @property
  def corruption_rate(self):
    return self._corruption_rate

  @property
  def corruption_dist(self):
    return self._corruption_dist

  @property
  def configs(self):
    return dict(self._configs)

  @property
  def history(self):
    return copy.deepcopy(self._history)

  @property
  def is_keras_model(self):
    return self._configs['is_keras_model']

  @property
  def dispersion(self):
    return self._configs['dispersion']

  @property
  def is_semi_supervised(self):
    return self._prot_dim > 0

  @property
  def supervised_percent(self):
    return self._ps

  @property
  def xclip(self):
    return self._configs['xclip']

  @property
  def xdrop(self):
    return self._configs['xdrop']

  @property
  def edrop(self):
    return self._configs['edrop']

  @property
  def zdrop(self):
    return self._configs['zdrop']

  @property
  def ddrop(self):
    return self._configs['ddrop']

  @property
  def yclip(self):
    return self._configs['yclip']

  @property
  def xnorm(self):
    return self._configs['xnorm']

  @property
  def xdist(self):
    return self._configs['xdist']

  @property
  def ydist(self):
    return self._configs['ydist']

  @property
  def zdist(self):
    return self._configs['zdist']

  @property
  def tnorm(self):
    return self._configs['tnorm']

  @property
  def ynorm(self):
    return self._configs['ynorm']

  @property
  def y_weight(self):
    return self._configs['y_weight']

  @property
  def is_fitted(self):
    return self._is_fitted

  @property
  def gene_dim(self):
    """ Return number of genes used for training """
    return self._gene_dim

  @property
  def prot_dim(self):
    """ Return number of protein markers used for training """
    return self._prot_dim

  @property
  def analytic(self):
    return self._configs['analytic']

  @property
  def batchnorm(self):
    return self._configs['batchnorm']

  @property
  def training_log(self):
    return self._training_log

# ===========================================================================
# Shortcut
# ===========================================================================
class InferenceSCVAE(Inference):
  def __init__(self, gene_dim, dispersion='gene-cell',
               xnorm='log', tnorm='raw', ynorm='prob',
               xclip=0, yclip=0,
               xdist='zinb', ydist='bernoulli', zdist='normal',
               xdrop=0.3, edrop=0, zdrop=0, ddrop=0,
               hdim=128, zdim=32, nlayer=2,
               batchnorm=True, analytic=True,
               kl_weight=1., warmup=400, y_weight=10.,
               extra_module_path=None,
               **kwargs):
    super(InferenceSCVAE, self).__init__(gene_dim=gene_dim, prot_dim=None,
               model='vae', dispersion=dispersion,
               xnorm=xnorm, tnorm=tnorm, ynorm=ynorm,
               xclip=xclip, yclip=yclip,
               xdist=xdist, ydist=ydist, zdist=zdist,
               xdrop=xdrop, edrop=edrop, zdrop=zdrop, ddrop=ddrop,
               hdim=hdim, zdim=zdim, nlayer=nlayer,
               batchnorm=batchnorm, analytic=analytic,
               kl_weight=kl_weight, warmup=warmup, y_weight=y_weight,
               extra_module_path=extra_module_path,
               **kwargs)

class InferenceSISUA(Inference):
  def __init__(self, gene_dim, prot_dim,
               dispersion='gene-cell',
               xnorm='log', tnorm='raw', ynorm='prob',
               xclip=0, yclip=0,
               xdist='zinb', ydist='bernoulli', zdist='normal',
               xdrop=0.3, edrop=0, zdrop=0, ddrop=0,
               hdim=128, zdim=32, nlayer=2,
               batchnorm=True, analytic=True,
               kl_weight=1., warmup=400, y_weight=10.,
               extra_module_path=None,
               **kwargs):
    super(InferenceSISUA, self).__init__(gene_dim=gene_dim, prot_dim=prot_dim,
               model='movae', dispersion=dispersion,
               xnorm=xnorm, tnorm=tnorm, ynorm=ynorm,
               xclip=xclip, yclip=yclip,
               xdist=xdist, ydist=ydist, zdist=zdist,
               xdrop=xdrop, edrop=edrop, zdrop=zdrop, ddrop=ddrop,
               hdim=hdim, zdim=zdim, nlayer=nlayer,
               batchnorm=batchnorm, analytic=analytic,
               kl_weight=kl_weight, warmup=warmup, y_weight=y_weight,
               extra_module_path=extra_module_path,
               **kwargs)
