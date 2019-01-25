from __future__ import print_function, division, absolute_import
import os
import inspect
from collections import OrderedDict

import numpy as np
from sklearn.base import BaseEstimator

from odin.fuel import Dataset
from odin.stats import train_valid_test_split, describe
from odin.utils import get_script_path, ctext, batching
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

  config : dict
      The following attribute are allowed
      hdim - number of hidden units for each hidden layer
      zdim - number of latent dimension
      nlayer - number of layers for both encoder and decoder
      ps - percents of data used for supervised learning (from 0. to 1.)
      ximpu - percents of X will be corrupted for imputation experiment
      yimpu - percents of y will be corrupted for imputation experiment
      xdist - the distribution name for X
      ydist - the distribution name for y
      zdist - the distribution name for Z
      batchnorm - enable batch-normalization
      xdrop - dropout level for input X (from 0. to 1.)
      edrop - dropout level for encoder output (from 0. to 1.)
      zdrop - dropout level for latent space (from 0. to 1.)
      ddrop - dropout level for decoder output (from 0. to 1.)
      count_sum - enable using count-sum features in latent space
      analytic - using analytic KL-divergence
      constraint - enable constrained distribution
      iw - using important weight sampling to avoid under or over flowing

  cellsize_normalize_factor : float
  """

  def __init__(self, model_name, model_config={},
               xnorm='raw', tnorm='raw', ynorm='prob',
               cellsize_normalize_factor=1):
    super(Inference, self).__init__()
    self._name = model_name
    self._model = N.Lambda.search(
        name=model_name,
        path=os.path.join(os.path.dirname(inspect.getfile(sisua)), 'models'),
        prefix='models_')

    assert isinstance(model_config, dict)
    self._config = dict(
        hdim=model_config.get('hdim', 256),
        zdim=model_config.get('zdim', 64),
        nlayer=model_config.get('nlayer', 2),

        ps=model_config.get('ps', 0.8),

        ximpu=model_config.get('ximpu', 0),
        yimpu=model_config.get('yimpu', 0),

        xdist=model_config.get('xdist', 'zinb'),
        ydist=model_config.get('ydist', 'bernoulli'),
        zdist=model_config.get('zdist', 'normal'),

        batchnorm=model_config.get('batchnorm', True),

        xdrop=model_config.get('xdrop', 0.3),
        edrop=model_config.get('edrop', 0),
        zdrop=model_config.get('zdrop', 0),
        ddrop=model_config.get('ddrop', 0),

        count_sum=model_config.get('count_sum', False),
        analytic=model_config.get('analytic', True),
        constraint=model_config.get('constraint', False),
        iw=model_config.get('iw', True),

        count_coeff=float(cellsize_normalize_factor)
    )

    self.input_plhs = None
    self.n_mcmc_samples_plh = None
    self.outputs = None
    self.pred_functions = {}
    self.score_functions = {}

    self._gmm_threshold = {}
    self._gene_dim = model_config.get('gene_dim', None)
    self._prot_dim = model_config.get('prot_dim', None)

    self._is_fitted = False

    self._xnorm = model_config.get('xnorm', xnorm)
    self._tnorm = model_config.get('tnorm', tnorm)
    self._ynorm = model_config.get('ynorm', ynorm)
    assert self._xnorm is not None and\
    self._tnorm is not None and\
    self._ynorm is not None, \
    "xnorm, tnorm and ynorm must be provided in `model_config` or"\
    " explicitly in the arguments"

  # ******************** pickling ******************** #
  def __getstate__(self):
    self._initialize_placeholders_and_outputs()
    model = N.serialize(nnops=self._model, path=None,
                        save_variables=True,
                        binary_output=True)
    return (self._name, model, self._config, self._gmm_threshold,
            self._gene_dim, self._prot_dim,
            self._is_fitted,
            self._xnorm, self._tnorm, self._ynorm)

  def __setstate__(self, states):
    (self._name, model, self._config, self._gmm_threshold,
     self._gene_dim, self._prot_dim,
     self._is_fitted,
     self._xnorm, self._tnorm, self._ynorm) = states
    # reload the model
    self._model = N.deserialize(model, force_restore_vars=True)
    # make sure everything initialized
    self.input_plhs = None
    self.n_mcmc_samples_plh = None
    self.outputs = None
    self.pred_functions = {}
    self.score_functions = {}

  # ******************** properties ******************** #
  @property
  def name(self):
    return self._name

  @property
  def xnorm(self):
    return self._xnorm

  @property
  def tnorm(self):
    return self._tnorm

  @property
  def ynorm(self):
    return self._ynorm

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

    X_plh = K.placeholder(shape=(None, self.gene_dim), dtype='float32', name='X_input')
    T_plh = K.placeholder(shape=(None, self.gene_dim), dtype='float32', name="T_target")
    y_plh = K.placeholder(shape=(None, self.prot_dim), dtype='float32', name='y_protein')

    # mask for supervised training
    mask_plh = K.placeholder(shape=(None,), dtype='float32', name='mask')
    # size factor
    C_plh = K.placeholder(shape=(None, 1), dtype='float32', name='CellSize')

    # all input in specific order
    self.input_plhs = (X_plh, T_plh, C_plh, mask_plh, y_plh)
    # number of sample for MCMC
    self.n_mcmc_samples_plh = K.placeholder(shape=(), dtype='int32',
                                            name='n_mcmc_samples')

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
    kw['xnorm'] = self._xnorm
    kw['tnorm'] = self._tnorm
    kw['ynorm'] = self._ynorm
    # select objective for multitask model
    # ce - categorical cross entropy
    # mse - mean squared error
    # ll - log loss
    # sg - multiple sigmoid loss
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
          monitoring=False, fig_dir=None):
    X, T, C, y = self._preprocess_inputs(X, y)
    n_samples = X.shape[0]
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

    n_train = len(X_train)
    n_valid = 0 if X_valid is None else len(X_valid)

    # ====== supervised mask ====== #
    # Generate a mask of 0, 1
    # - 0 mean unsupervised sample,
    # - 1 mean supervised sample
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
    updates = optz.minimize(self.outputs['loss'], verbose=True)
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
                                allow_rollback=False, verbose=2,
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
    # ====== end training ====== #
    self._is_fitted = True

  # ******************** scoring ******************** #
  def score(self, X, y=None,
            n_mcmc_samples=100):
    """Compute the per-sample average log-likelihood of the given data X.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_dimensions)
        List of n_features-dimensional data points. Each row
        corresponds to a single data point.

    Returns
    -------
    log_likelihood : float
        Log likelihood of the Gaussian mixture given X.
    """
    X, T, C, y = self._preprocess_inputs(X, y)
    f = self._initialize_score_functions(n_mcmc_samples)
    mask = np.ones(shape=(X.shape[0],))
    scores = f(X, T, C, mask, y)
    return OrderedDict([(tensor.name, np.mean(val))
                       for tensor, val in zip(f.outputs, scores)])

  # ******************** predicting ******************** #
  def _make_prediction(self, pred_type, X, y=None,
                       n_mcmc_samples=100):
    X, T, C, y = self._preprocess_inputs(X, y)
    f = self._initialize_predict_functions(n_mcmc_samples)
    mask = np.ones(shape=(X.shape[0],))
    return f[str(pred_type).lower()](X, T, C, mask, y)

  def predict_Z(self, X, y=None, n_mcmc_samples=100):
    return self._make_prediction('Z', X, y, n_mcmc_samples)

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
