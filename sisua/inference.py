from __future__ import print_function, division, absolute_import
import os
import inspect
from six import string_types
from collections import OrderedDict, defaultdict

import numpy as np
from sklearn.base import BaseEstimator
import tensorflow as tf

from odin.fuel import Dataset
from odin.ml import fast_pca, fast_tsne
from odin.stats import train_valid_test_split, describe
from odin.utils.cache_utils import cache_memory
from odin.utils import (get_script_path, ctext, batching, one_hot, uuid,
                        as_list)
from odin import nnet as N, backend as K, training

import sisua
from sisua import is_verbose
from sisua.data.const import UNIVERSAL_RANDOM_SEED
from sisua.data import get_dataset, SingleCellDataset
from sisua.utils.others import validating_dataset
from sisua.utils import LearningCurves, plot_monitoring_epoch
from sisua.label_threshold import GMMThresholding

possible_outputs = ('Z',
                    'loss', 'metr',
                    # W is predicted corrupted count (with dropout)
                    'W', 'W_stdev_explained', 'W_stdev_total',
                    # V is the cleaned count (without dropout)
                    'V', 'V_stdev_explained', 'V_stdev_total',
                    'y_', 'pi')

def _normalize_tensor_name(name):
  name = name.split(':')[0].split('/')[-1].lower()
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

  hdim : number of hidden units for each hidden layer

  zdim : number of latent dimension

  nlayer : number of layers for both encoder and decoder

  xnorm : {'raw', 'log', 'bin', 'prob'}
  tnorm : {'raw', 'log', 'bin', 'prob'}
  ynorm : {'raw', 'log', 'bin', 'prob'}
      different strategies for data normalization
      raw - keep the raw data (i.e. count or expression level)
      log - log-normalization
      bin - using `sisua.label_threshold.GMMThresholding` to convert
            data to binary format
      prob - using `sisua.label_threshold.GMMThresholding` to convert
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

  count_sum : enable using count-sum features in latent space

  analytic : bool (default: True)
      using analytic KL-divergence

  iw : bool (default: True)
      using important weight sampling to avoid under- or over-flowing

  constraint : bool (default: False)
      enable constrained distribution

  cellsize_normalize_factor : float
      general factor for normalizing all the cell size

  extra_module_path : {string, None}
      path to folder contain the model script that you defined
      yourself, the .py file should has 'models_' prefix
      (e.g. 'models_vae.py')
  """

  def __init__(self, model_name='vae',
               xnorm='log', tnorm='raw', ynorm='prob',
               xclip=0, yclip=0,
               xdist='zinb', ydist='bernoulli', zdist='normal',
               xdrop=0.3, edrop=0, zdrop=0, ddrop=0,
               hdim=128, zdim=32, nlayer=2,
               batchnorm=True, count_sum=False,
               analytic=True, iw=True, constraint=False,
               cellsize_normalize_factor=1, extra_module_path=None,
               **kwargs):
    super(Inference, self).__init__()
    # ====== store the config ====== #
    configs = dict(locals())
    del configs['self']
    del configs['kwargs']
    del configs['__class__']
    configs.update(kwargs)
    # ====== search for module ====== #
    primary_path = [os.path.join(os.path.dirname(inspect.getfile(sisua)), 'models')]
    extra_module_path = [] if extra_module_path is None else \
    as_list(extra_module_path, t=string_types)
    configs['extra_module_path'] = extra_module_path

    self._model = N.Lambda.search(
        name=model_name,
        path=primary_path + extra_module_path,
        prefix='models_')

    self._name = self._model.name
    configs['model_name'] = self._name
    # ====== basics ====== #
    self._history = {}
    self._config = configs

    self._gmm_threshold = {}
    self._gene_dim = None
    self._prot_dim = None
    self._is_fitted = False

    self._ps = 0
    self._trained_n_epoch = 0
    self._batch_size = 0
    self._learning_rate = 0
    self._n_mcmc_train = 0
    # ====== init ====== #
    self._init_and_reset()

  def _init_and_reset(self):
    self.input_plhs = None
    self.n_mcmc_samples_plh = None
    self.outputs = None
    self.pred_functions = {}
    self.score_functions = {}

  def set_cell_info(self, gene_dim, prot_dim=None):
    """ This function allows making prediction without fitting

    Parameters
    ----------
    gene_dim : int
        number of genes in given dataset

    prot_dim : {int, None}
        number of protein in given dataset, if `None`,
        assumed the same number as `gene_dim`
    """
    self._gene_dim = int(gene_dim)
    if self._prot_dim is None and prot_dim is None:
      self._prot_dim = self._gene_dim
    elif prot_dim is not None:
      self._prot_dim = int(prot_dim)
    return self

  # ******************** pickling ******************** #
  def __getstate__(self):
    self._initialize_placeholders_and_outputs()
    model = N.serialize(nnops=self._model, path=None,
                        save_variables=True,
                        binary_output=True)
    return (self._name, model, self._config,
            self._gmm_threshold, self._gene_dim, self._prot_dim,
            self._is_fitted,
            self._ps,
            self._trained_n_epoch,
            self._batch_size,
            self._learning_rate,
            self._n_mcmc_train)

  def __setstate__(self, states):
    (self._name, model, self._config,
     self._gmm_threshold, self._gene_dim, self._prot_dim,
     self._is_fitted,
     self._ps,
     self._trained_n_epoch,
     self._batch_size,
     self._learning_rate,
     self._n_mcmc_train) = states
    # reload the model
    self._model = N.deserialize(model, force_restore_vars=True)
    # make sure everything initialized
    self._init_and_reset()

  # ******************** properties ******************** #
  @property
  def name(self):
    return self._name

  @property
  def config(self):
    return dict(self._config)

  @property
  def supervised_percent(self):
    return self._ps

  @property
  def xclip(self):
    return self._config['xclip']

  @property
  def yclip(self):
    return self._config['yclip']

  @property
  def xnorm(self):
    return self._config['xnorm']

  @property
  def tnorm(self):
    return self._config['tnorm']

  @property
  def ynorm(self):
    return self._config['ynorm']

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
  def zdim(self):
    return self._config['zdim']

  @property
  def hdim(self):
    return self._config['hdim']

  @property
  def n_layers(self):
    return self._config['nlayer']

  # ******************** initialization ******************** #
  def _initialize_placeholders_and_outputs(self):
    assert self._gene_dim is not None and self._prot_dim is not None, \
    "Unknown dimension of gene and protein matrix"
    if self.input_plhs is not None and self.outputs is not None:
      return

    with tf.variable_scope(self.name + '_placeholders'):
      X_plh = K.placeholder(
          shape=(None, self.gene_dim), dtype='float32', name='X_input')
      T_plh = K.placeholder(
          shape=(None, self.gene_dim), dtype='float32', name="T_target")
      y_plh = K.placeholder(
          shape=(None, self.prot_dim), dtype='float32', name='y_protein')

      # mask for supervised training
      mask_plh = K.placeholder(shape=(None,), dtype='float32', name='mask')
      # size factor
      C_plh = K.placeholder(shape=(None, 1), dtype='float32', name='CellSize')

      # number of sample for MCMC
      self.n_mcmc_samples_plh = K.placeholder(shape=(), dtype='int32',
                                              name='n_mcmc_samples')
    # all input in specific order
    self.input_plhs = (X_plh, T_plh, C_plh, mask_plh, y_plh)

    if is_verbose():
      print(ctext("Input placeholders:", 'lightyellow'))
      print(" *", ctext(X_plh, 'cyan'))
      print(" *", ctext(T_plh, 'cyan'))
      print(" *", ctext(C_plh, 'cyan'))
      print(" *", ctext(y_plh, 'cyan'))
      print(" *", ctext(mask_plh, 'cyan'))
      print(" *", ctext(self.n_mcmc_samples_plh, 'cyan'))

    # ====== applying the model ====== #
    kw = dict(self._config)
    kw['xnorm'] = self.xnorm
    kw['tnorm'] = self.tnorm
    kw['ynorm'] = self.ynorm
    # select objective for multitask model
    # ce : categorical cross entropy
    # mse : mean squared error
    # ll : log loss
    # sg : multiple sigmoid loss
    kw['rec_loss'] = 'mse'
    if self.ynorm == 'raw' or self.ynorm == 'log':
      kw['cls_loss'] = 'mse'
    else:
      kw['cls_loss'] = 'sg'
    outputs = self._model(*(self.input_plhs + (self.n_mcmc_samples_plh, kw)))
    # ====== compulsory outputs ====== #
    assert 'Z' in outputs, "Latent space must be in the outputs"
    assert 'loss' in outputs, "Loss must be given in the outputs for training"
    assert all(i in possible_outputs for i in outputs.keys()),\
        'Outputs must contain one of the following: %s' % ', '.join(possible_outputs)

    # latent space output
    Z = outputs['Z']
    assert Z.get_shape().as_list()[-1] == int(self.zdim), \
        "Expect %d latent dimension but return Z with shape: %s" % \
        (int(self.zdim), Z.get_shape().as_list())

    # Loss and metrics
    loss = outputs['loss']
    metr = outputs.get('metr', [])
    if K.is_tensor(metr, inc_distribution=False):
      metr = [metr]
    assert isinstance(metr, list)

    # ====== reconstructed ====== #
    W_expected = outputs.get('W', None)
    W_stdev_total = outputs.get('W_stdev_total', None)
    W_stdev_explained = outputs.get('W_stdev_explained', None)

    # ====== imputed ====== #
    V_expected = outputs.get('V', None)
    V_stdev_total = outputs.get('V_stdev_total', None)
    V_stdev_explained = outputs.get('V_stdev_explained', None)

    # ====== zero-inflated pi ====== #
    pi = outputs.get('pi', None)

    # ====== predicted labels ====== #
    y_ = outputs['y_'] if 'y_' in outputs else None

    if is_verbose():
      print(ctext("Network outputs:", 'lightyellow'))
      print("   Latent           :", ctext(Z, 'cyan'))
      print("   zero-inflated PI :", ctext(pi, 'cyan'))
      print("   Label Prediction :", ctext(y_, 'cyan'))

      print(ctext("Reconstruction:", 'lightyellow'))
      print("   W expected       :", ctext(W_expected, 'cyan'))
      print("   W stdev Explained:", ctext(W_stdev_explained, 'cyan'))
      print("   W stdev Total    :", ctext(W_stdev_total, 'cyan'))

      print(ctext("Imputed:", 'lightyellow'))
      print("   V expected       :", ctext(V_expected, 'cyan'))
      print("   V stdev Explained:", ctext(V_stdev_explained, 'cyan'))
      print("   V stdev Total    :", ctext(V_stdev_total, 'cyan'))

      print(ctext("Training info:", 'lightyellow'))
      print("   Loss    :", ctext(loss, 'cyan'))
      print("   Metrics :", ctext(metr, 'cyan'))

    self.outputs = dict(
        z=Z,
        pi=pi,
        y_=y_,

        w=(W_expected, W_stdev_total, W_stdev_explained),
        v=(V_expected, V_stdev_total, V_stdev_explained),

        loss=loss,
        metr=metr
    )

    # initialize everything
    K.initialize_all_variables()

  def _initialize_score_functions(self, n_mcmc_samples):
    self._initialize_placeholders_and_outputs()
    if n_mcmc_samples not in self.score_functions:
      scores = [self.outputs['loss']] + list(self.outputs['metr'])
      scores = sorted(set(scores), key=lambda x: x.name)
      f_score = K.function(inputs=self.input_plhs,
                           outputs=scores,
                           training=False,
                           defaults={self.n_mcmc_samples_plh: int(n_mcmc_samples)},
                           batch_size=max(2, 4000 / int(n_mcmc_samples)))
      self.score_functions[n_mcmc_samples] = f_score
    return self.score_functions[n_mcmc_samples]

  def _initialize_predict_functions(self, n_mcmc_samples):
    self._initialize_placeholders_and_outputs()
    n_mcmc_samples = int(n_mcmc_samples)
    if n_mcmc_samples in self.pred_functions:
      return self.pred_functions[n_mcmc_samples]

    # ====== prediction functions ====== #
    def create_func(out_):
      if isinstance(out_, (tuple, list)):
        out_ = [o for o in out_ if o is not None]
        if len(out_) == 0:
          out_ = None
      if out_ is not None:
        fn = K.function(
            inputs=self.input_plhs,
            outputs=out_,
            training=False,
            defaults={self.n_mcmc_samples_plh: int(n_mcmc_samples)})

        # this function will iterate over mini-batch for prediction
        # make sure the order is preserved, so no shuffling
        def _batch_prediction(*X):
          assert len(set(x.shape[0] for x in X)) == 1
          n = X[0].shape[0]
          all_y = []
          for start, end in batching(batch_size=32, n=n, seed=None):
            x = [i[start:end] for i in X]
            y = fn(*x)
            all_y.append(y)
          if isinstance(y, (tuple, list)):
            return [np.concatenate([j[i] for j in all_y], axis=0)
                    for i in range(len(y))]
          else:
            return np.concatenate(all_y, axis=0)
        return _batch_prediction
      return None

    # latent space
    f_z = create_func(out_=self.outputs['z'])
    # reconstruction, return order
    # W, stdev_total (analytic), stdev_explained
    # auto-ignore None
    f_w = create_func(out_=self.outputs['w'])
    f_v = create_func(out_=self.outputs['v'])
    # label
    f_y = create_func(out_=self.outputs['y_'])
    # pi
    f_pi = create_func(out_=self.outputs['pi'])
    self.pred_functions[n_mcmc_samples] = dict(
        z=f_z,
        w=lambda *args, **kwargs: None if f_w is None else f_w(*args, **kwargs),
        v=lambda *args, **kwargs: None if f_v is None else f_v(*args, **kwargs),
        y=lambda *args, **kwargs: None if f_y is None else f_y(*args, **kwargs),
        pi=lambda *args, **kwargs: None if f_pi is None else f_pi(*args, **kwargs)
    )
    return self.pred_functions[n_mcmc_samples]

  # ******************** helper ******************** #
  def monitoring_epoch(task):
    curr_epoch = task.curr_epoch
    if curr_epoch > 5 and curr_epoch % 5 != 0:
      return
    # ====== prepare latent space ====== #
    Z_test = self.f_z(*test_tuple)
    Z_test_drop = self.f_z(*testdrop_tuple)
    if np.any(np.isnan(Z_test)):
      return
    # ====== prepare the reconstruction ====== #
    if f_w is not None:
      # order: [W, W_stdev_total, W_stdev_explained]
      W_test = f_w(*test_tuple)
      W_test_drop = f_w(*testdrop_tuple)
    else:
      W_test = None
      W_test_drop = None
    # ====== zero-inflated PI ====== #
    if f_pi is not None:
      pi_test = f_pi(*test_tuple)
      pi_test_drop = f_pi(*testdrop_tuple)
    else:
      pi_test = None
      pi_test_drop = None
    # ====== plotting ====== #
    plot_monitoring_epoch(
        X=X_test, X_drop=X_test_drop, y=y_test_prob,
        Z=Z_test, Z_drop=Z_test_drop,
        W_outputs=W_test, W_drop_outputs=W_test_drop,
        pi=pi_test, pi_drop=pi_test_drop,
        row_name=row_test, dropout_percentage=DROPOUT_TEST,
        curr_epoch=curr_epoch, ds_name=ds_name, labels=labels,
        save_dir=MODEL_DIR)

  def normalize(self, x, method, data_name):
    x = np.atleast_2d(x)
    if method == 'raw':
      pass
    elif method == 'log':
      x = K.log_norm(x, axis=1, scale_factor=10000)
    elif method == 'bin' or method == 'prob':
      if data_name not in self._gmm_threshold:
        gmm = GMMThresholding()
        gmm.fit(x)
        self._gmm_threshold[data_name] = gmm
      else:
        gmm = self._gmm_threshold[data_name]
      x = gmm.predict(x) if method == 'bin' else gmm.predict_proba(x)
    else:
      raise NotImplementedError
    return x

  def _preprocess_inputs(self, X, y):
    if X.ndim == 1:
      X = np.expand_dims(X, axis=-1)
    if self.xclip > 0:
      X = np.clip(X, a_min=0, a_max=float(self.xclip))

    if y is not None:
      if y.ndim == 1:
        y = np.expand_dims(y, axis=-1)
      if self.yclip > 0:
        y = np.clip(y, a_min=0, a_max=float(self.yclip))

    C = np.sum(X, axis=-1, keepdims=True)
    X_norm = self.normalize(X, method=self.xnorm, data_name='X')
    T_norm = self.normalize(X, method=self.tnorm, data_name='X')
    if y is not None:
      y_norm = self.normalize(y, method=self.ynorm, data_name='y')
    else:
      y_norm = X_norm
    assert X_norm.shape[0] == y_norm.shape[0]
    # ====== check matching dimension ====== #
    if self._gene_dim is None:
      self._gene_dim = X_norm.shape[1]
    else:
      assert self._gene_dim == X_norm.shape[1], "Number of genes mismatch"

    if self._prot_dim is None:
      self._prot_dim = 0 if y_norm is None else y_norm.shape[1]
    else:
      assert self._prot_dim == y_norm.shape[1], "Number of protein mismatch"

    return X_norm, T_norm, C, y_norm

  # ******************** fitting ******************** #
  def fit(self, X, y=None,
          supervised_percent=0.8, validation_percent=0.1,
          n_mcmc_samples=1,
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
    X, T, C, y = self._preprocess_inputs(X, y)
    n_samples = X.shape[0]
    self._ps = supervised_percent
    self._trained_n_epoch += n_epoch
    self._batch_size = batch_size
    self._learning_rate = learning_rate
    self._n_mcmc_train = n_mcmc_samples
    # ====== initializing ====== #
    self._initialize_placeholders_and_outputs()

    # ====== splitting train valid ====== #
    X_train, T_train, C_train, y_train = None, None, None, None
    X_valid, T_valid, C_valid, y_valid = None, None, None, None

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

      C_train = C[train_ids]
      C_valid = C[valid_ids]

      if y is not None:
        y_train = y[train_ids]
        y_valid = y[valid_ids]
    else: # no validation set given
      X_train = X
      T_train = T
      C_train = C
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

    if is_verbose():
      print(ctext("Training data:", 'lightyellow'))
      print("X train:", ctext(X_train.shape, 'cyan'),
            describe(X_train, shorten=True))
      print("T train:", ctext(T_train.shape, 'cyan'),
            describe(T_train, shorten=True))
      print("C train:", ctext(C_train.shape, 'cyan'),
            describe(C_train, shorten=True))
      print("y train:", ctext(None if y is None else y_train.shape, 'cyan'),
            describe(y_train, shorten=True))
      print("m train:", ctext(m_train.shape, 'cyan'), np.sum(m_train))

      if X_valid is not None:
        print(ctext("Validation data:", 'lightyellow'))
        print("X valid:", ctext(X_valid.shape, 'cyan'),
              describe(X_valid, shorten=True))
        print("T valid:", ctext(T_valid.shape, 'cyan'),
              describe(T_valid, shorten=True))
        print("C valid:", ctext(C_valid.shape, 'cyan'),
              describe(C_valid, shorten=True))
        print("y valid:", ctext(None if y is None else y_valid.shape, 'cyan'),
              describe(y_valid, shorten=True))
        print("m valid:", ctext(m_valid.shape, 'cyan'), np.sum(m_valid))

    # ====== create the optimizer ====== #
    optz = K.optimizers.Adam(lr=float(learning_rate),
                             clipnorm=None,
                             name=self.name)
    updates = optz.minimize(self.outputs['loss'], verbose=is_verbose())
    global_norm = optz.norm
    K.initialize_all_variables()

    # ====== training functions ====== #
    # for training
    tmp = [self.outputs['loss'], global_norm]
    for i in self.outputs['metr']:
      if i not in tmp:
        tmp.append(i)
    f_train = K.function(inputs=self.input_plhs, outputs=tmp,
                         updates=updates, training=True,
                         defaults={self.n_mcmc_samples_plh: int(n_mcmc_samples)})
    # for scoring
    tmp = [self.outputs['loss']]
    for i in self.outputs['metr']:
      if i not in tmp:
        tmp.append(i)
    f_score = K.function(inputs=self.input_plhs, outputs=tmp,
                         training=False,
                         defaults={self.n_mcmc_samples_plh: int(n_mcmc_samples)},
                         batch_size=None)
    # overlapping metrics for summary
    overlap_metrics = sorted(set([self.outputs['loss']] + self.outputs['metr']),
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
        training.NaNDetector(),
        training.CheckpointGeneralization('valid',
                                          output_name=self.outputs['loss']),
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
                           data=(X_train, T_train, C_train, m_train, y_train),
                           epoch=int(n_epoch),
                           name='train')
    if X_valid is not None:
      trainer.set_valid_task(func=f_score,
                             data=(X_valid, T_valid, C_valid, m_valid, y_valid),
                             freq=training.Timer(percentage=1.0),
                             name='valid')
    # NOTE: this line is important
    trainer.run()
    # ====== store the history ====== #
    hist = trainer.history
    for task_name, task_results in hist.items():
      self._history[task_name] = {}
      for epoch_id, epoch_results in task_results.items():
        self._history[task_name][epoch_id] = {}
        for tensor_name, tensor_batch in epoch_results.items():
          tensor_name = _normalize_tensor_name(tensor_name)
          self._history[task_name][epoch_id][tensor_name] = tensor_batch
    # ====== end training ====== #
    self._is_fitted = True

  # ******************** for fast evaluation ******************** #
  @property
  def train_loss(self):
    loss_name = _normalize_tensor_name(self.outputs['loss'].name)
    for name, values in self.train_history.items():
      if loss_name == name:
        return values
    return []

  @property
  def valid_loss(self):
    loss_name = _normalize_tensor_name(self.outputs['loss'].name)
    for name, values in self.valid_history.items():
      if loss_name == name:
        return values
    return []

  @property
  def train_history(self):
    """ Return the epoch results history during training """
    hist = defaultdict(list)
    if not self._is_fitted:
      return hist
    for epoch_id, epoch_results in sorted(
        self._history['train'].items(), key=lambda x: x[0]):
      for tensor_name, batch_results in epoch_results.items():
        hist[tensor_name].append(np.mean(batch_results))
    return hist

  @property
  def valid_history(self):
    """ Return the epoch results history of validating during training """
    if 'valid' not in self._history:
      return {}

    hist = defaultdict(list)
    if not self._is_fitted:
      return hist
    for epoch_id, epoch_results in sorted(
        self._history['valid'].items(), key=lambda x: x[0]):
      for tensor_name, batch_results in epoch_results.items():
        hist[tensor_name].append(np.mean(batch_results))
    return hist

  # ******************** plotting utils ******************** #
  def plot_learning_curves(self, save_path=None):
    assert self._is_fitted, "Model hasn't fitted!"
    from sisua.utils.training_utils import plot_learning_curves
    hist = self._history
    records = defaultdict(lambda: defaultdict(list))

    for task_name, task_results in hist.items():
      for epoch_id, epoch_results in task_results.items():
        for tensor_name, tensor_batch in epoch_results.items():
          records[tensor_name][task_name].append(
              (np.mean(tensor_batch), np.std(tensor_batch)))

    plot_learning_curves(records)
    if save_path is not None:
      from odin.visual import plot_save
      plot_save(save_path, dpi=180)

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
    f = self._initialize_predict_functions(n_mcmc_samples)
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
    from odin.visual import (plot_scatter, plot_save,
                             plot_figure, plot_scatter_heatmap)
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

  # ******************** scoring ******************** #
  @cache_memory
  def score(self, X, y=None,
            n_mcmc_samples=100):
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
    X, T, C, y = self._preprocess_inputs(X, y)
    f = self._initialize_score_functions(n_mcmc_samples)
    mask = np.ones(shape=(X.shape[0],))
    scores = f(X, T, C, mask, y)
    return OrderedDict([(tensor.name, np.mean(val))
                       for tensor, val in zip(f.outputs, scores)])

  def log_likelihood(self, X, y=None, n_mcmc_samples=100):
    """ Compute the log-likelihood of p(x|z); i.e. the
    reconstruction loss """
    scores = self.score(X, y, n_mcmc_samples)
    for name, value in scores.items():
      if 'nllk:0' == name[-6:].lower():
        return -value
    for name, value in scores.items():
      if 'nllk_x' in name.lower():
        return -value
    raise RuntimeError("Cannot find reconstruction log-likelihood in the output "
                       "metrics, we have: %s" % str(list(scores.items())))

  def marginal_log_likelihood(self, X, y=None, n_mcmc_samples=100):
    """
    Computes a biased estimator for log p(x), which is the marginal log likelihood.
    Despite its bias, the estimator still converges to the real value
    of log p(x) when n_samples_mc (for Monte Carlo) goes to infinity
    (a fairly high value like 100 should be enough)
    Due to the Monte Carlo sampling, this method is not as computationally efficient
    as computing only the reconstruction loss
    """
    scores = self.score(X, y, n_mcmc_samples)
    for name, value in scores.items():
      if self.outputs['loss'].name == name:
        return -value
    raise RuntimeError("Cannot find reconstruction log-likelihood in the output "
                       "metrics, we have: %s" % str(list(scores.items())))

  # ******************** predicting ******************** #
  def _make_prediction(self, pred_type, X, y=None,
                       n_mcmc_samples=100):
    X, T, C, y = self._preprocess_inputs(X, y)
    f = self._initialize_predict_functions(n_mcmc_samples)
    mask = np.ones(shape=(X.shape[0],))
    return f[str(pred_type).lower()](X, T, C, mask, y)

  def predict_Z(self, X, y=None):
    """ Return mean of the latent posterior
    (i.e. mean of Normal distribution) """
    return self._make_prediction('Z', X, y, n_mcmc_samples=100)

  def predict_W(self, X, y=None, n_mcmc_samples=100):
    """ Return a tuple of
    (W_expected, W_stdev_total, W_stdev_explained)

    if not a variational model,
    then W_stdev_total and W_stdev_explained are None
    """
    return self._make_prediction('W', X, y, n_mcmc_samples)

  def predict_V(self, X, y=None, n_mcmc_samples=100):
    """ Return a tuple of
    (V_expected, V_stdev_total, V_stdev_explained)

    if not a zero-inflated model then return (None, None, None)

    if not a variational model,
    then V_stdev_total and V_stdev_explained are None
    """
    return self._make_prediction('V', X, y, n_mcmc_samples)

  def predict_PI(self, X, y=None, n_mcmc_samples=100):
    """ Return a matrix (n_sample, n_gene) of Zero-inflated
    rate

    if not a zero-inflated model, then return None
    """
    return self._make_prediction('PI', X, y, n_mcmc_samples)
