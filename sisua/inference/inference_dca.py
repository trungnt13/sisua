from __future__ import print_function, division, absolute_import

import io
import os
from collections import defaultdict

import numpy as np

from odin.utils import stdio, get_formatted_datetime

from .inference_base import Inference
from ._consts import PREDICTION_BATCH_SIZE
from sisua.data import apply_artificial_corruption

# ===========================================================================
# Helper for DCA
# ===========================================================================
def _no_zero_genes(X):
  # check if any genes is all zeros (i.e. zero columns)
  X = X.astype('float32')
  zero_genes = np.sum(X, axis=0, keepdims=True)
  if np.sum(zero_genes == 0) > 0:
    X = X + np.finfo(X.dtype).eps
  return X

def dca_normalize(X, log=True):
  from dca.api import normalize
  from anndata import AnnData
  from odin.utils import catch_warnings_error
  # this is the default configuration
  log1p = bool(log)
  X = _no_zero_genes(X)
  with catch_warnings_error(Warning):
    adata = normalize(AnnData(X),
                      filter_min_counts=False,
                      size_factors=True,
                      normalize_input=True,
                      logtrans_input=log1p)
  return adata.X, adata.obs.size_factors.values

def dca_train(
    X, network, log, learning_rate,
    epochs=300, reduce_lr=10,
    early_stop=15, batch_size=32,
    validation_split=0.1, verbose=True):
  import keras.optimizers as opt
  from keras.callbacks import (EarlyStopping, ReduceLROnPlateau)
  from dca.api import normalize
  from anndata import AnnData
  adata = normalize(AnnData(X),
                    filter_min_counts=False,
                    size_factors=True,
                    normalize_input=True,
                    logtrans_input=log)

  model = network.model
  loss = network.loss

  optimizer = opt.__dict__['adam'](lr=learning_rate, clipvalue=5.)
  model.compile(loss=loss, optimizer=optimizer)

  # Callbacks
  callbacks = []
  if reduce_lr:
    lr_cb = ReduceLROnPlateau(monitor='val_loss', patience=reduce_lr, verbose=verbose)
    callbacks.append(lr_cb)
  if early_stop:
    es_cb = EarlyStopping(monitor='val_loss', patience=early_stop, verbose=verbose)
    callbacks.append(es_cb)

  inputs = {'count': adata.X, 'size_factors': adata.obs.size_factors}
  output = adata.raw.X

  loss = model.fit(inputs, output,
                   epochs=epochs,
                   batch_size=batch_size,
                   shuffle=True,
                   callbacks=callbacks,
                   validation_split=validation_split,
                   verbose=verbose)
  return loss

# ===========================================================================
# Main
# ===========================================================================
class InferenceDCA(Inference):
  """ InferenceDCA """

  def __init__(self, gene_dim,
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
    try:
      import dca as _dca_
      from odin.autoconfig import get_session
      from keras import backend
      backend.set_session(get_session())
    except ImportError as e:
      raise RuntimeError("pip install dca")
    # ====== validate the arguments ====== #
    model = kwargs.get('model', 'vae')
    prot_dim = None
    if model not in ('vae', 'dca'):
      raise RuntimeError("InferenceSCVI only support 'vae' model")
    if xdist not in ('zinb', 'nb'):
      raise ValueError("Only support 'zinb' and 'nb' for `xdist`")
    if xnorm not in ('log', 'raw'):
      raise ValueError(
          "Only support log variational or raw count variational for scVI")
    if tnorm not in ('raw',):
      raise ValueError(
          "Only support raw count target value for scVI")
    self._name = 'dca'
    # ====== others ====== #
    self._history = {'train': defaultdict(list),
                     'valid': defaultdict(list)}
    self._training_log = ''
    self._configs = dict(locals())
    self._configs.update(kwargs)

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
    # ====== initialize ====== #
    self._initialize_model()

  def _initialize_model(self, weights=None):
    #
    if self.xdist == 'zinb':
      if self.dispersion == 'gene':
        from dca.network import ZINBConstantDispAutoencoder as Autoencoder
      elif self.dispersion == 'gene-cell':
        from dca.network import ZINBAutoencoder as Autoencoder
      else:
        raise ValueError("No support for dispersion: %s" % self.dispersion)
    #
    elif self.xdist == 'nb':
      if self.dispersion == 'gene':
        from dca.network import NBConstantDispAutoencoder as Autoencoder
      elif self.dispersion == 'gene-cell':
        from dca.network import NBAutoencoder as Autoencoder
      else:
        raise ValueError("No support for dispersion: %s" % self.dispersion)
    #
    else:
      raise ValueError("No support for xdist: %s" % self.xdist)

    self._ae = Autoencoder(
        input_size=self.gene_dim,
        output_size=None,
        hidden_size=(self.hdim,) * self.nlayer + (self.zdim,) + (self.hdim,) * self.nlayer,
        input_dropout=self.xdrop,
        hidden_dropout=self.zdrop,
        batchnorm=True,
        activation='relu')
    self._ae.build()
    if weights is not None:
      self._ae.model.set_weights(weights)

  # ******************** pickling ******************** #
  def __getstate__(self):
    weights = self._ae.model.get_weights()
    states = self.get_states()
    return states, weights

  def __setstate__(self, states):
    states, weights = states
    self.set_states(states)
    self._initialize_model(weights)

  # ******************** fitting ******************** #
  def fit(self, X, y=None,
          supervised_percent=0.8, validation_percent=0.1,
          n_mcmc_samples=1,
          corruption_rate=0.25, corruption_dist='binomial',
          batch_size=64, n_epoch=120, learning_rate=1e-4,
          monitoring=False, fig_dir=None,
          detail_logging=False):
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
    assert n_mcmc_samples == 1, \
    "Only support 1 MCMC sample during training for scVI"

    self._ps = supervised_percent
    self._batch_size = batch_size
    self._learning_rate = learning_rate
    self._n_mcmc_train = n_mcmc_samples
    X = apply_artificial_corruption(
        X, dropout=corruption_rate, distribution=corruption_dist)
    X = _no_zero_genes(X)
    # ====== training ====== #
    hist = dca_train(
        X=X, network=self._ae, log=self.xnorm == 'log',
        learning_rate=learning_rate, epochs=n_epoch, batch_size=batch_size,
        validation_split=validation_percent, verbose=detail_logging)
    # ====== save history ====== #
    hist = hist.history
    self._history['train']['loss'] += hist["loss"]
    self._history['valid']['loss'] += hist["val_loss"]
    # ====== update states ====== #
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
    # only VAE model for scVI
    # X, _, L, _, _, _ = self._preprocess_inputs(X, y=None)
    # L = np.exp(L)

    X, L = dca_normalize(X, log=self.xnorm == 'log')
    loss = self._ae.model.evaluate({
        'count': X,
        'size_factors': L}, y=X, batch_size=PREDICTION_BATCH_SIZE,
        verbose=0)
    return {'loss': loss}

  # ******************** predicting ******************** #
  def predict_Z(self, X):
    """ Return mean of the latent posterior
    (i.e. mean of Normal distribution) """
    X, L = dca_normalize(X, log=self.xnorm == 'log')
    return self._ae.encoder.predict({
        'count': X,
        'size_factors': L}, batch_size=PREDICTION_BATCH_SIZE)

  def predict_y(self, X, y, n_mcmc_samples=100):
    """ Predicting the protein marker """
    assert self.is_semi_supervised and y is not None,\
    "Only semi-supervised model has this prediction, and `y` must not be None"

  def predict_L(self, X, n_mcmc_samples=100):
    """ Return library size prediction """
    v = self.predict_V(X, n_mcmc_samples=n_mcmc_samples)
    return np.sum(v, axis=1, keepdims=True)

  def predict_Lstddev(self, X, n_mcmc_samples=100):
    """ Return library size stddev prediction """
    v_stddev = self.predict_Vstddev(X, n_mcmc_samples)
    return np.sum(v_stddev, axis=1, keepdims=True)

  def predict_W(self, X, n_mcmc_samples=100):
    """ Return mean of the reconstructed value """
    v = self.predict_V(X, n_mcmc_samples)
    if 'pi' not in self._ae.extra_models:
      return v
    pi = self.predict_PI(X, n_mcmc_samples)
    return (1 - pi) * v

  def predict_Wstddev(self, X, n_mcmc_samples=100):
    """ Return stddev of reconstructed value
    if not a variational model, then V_stddev is None
    """
    v_stddev = self.predict_Vstddev(X, n_mcmc_samples)
    if 'pi' not in self._ae.extra_models:
      return v_stddev
    v = self.predict_V(X, n_mcmc_samples)
    pi = self.predict_PI(X, n_mcmc_samples)
    w_mean = (1 - pi) * v
    return np.sqrt(
        (1 - pi) * (v_stddev + v**2) - w_mean**2)

  def predict_V(self, X, n_mcmc_samples = 100):
    """ Return mean of the denoised value """
    X, L = dca_normalize(X, log=self.xnorm == 'log')
    return self._ae.model.predict({
        'count': X,
        'size_factors': L}, batch_size=PREDICTION_BATCH_SIZE)

  def predict_Vstddev(self, X, n_mcmc_samples = 100):
    """ Return stddev of denoised value
    if not a variational model, then V_stddev is None
    """
    X, L = dca_normalize(X, log=self.xnorm == 'log')
    return self._ae.extra_models['dispersion'].predict({
        'count': X,
        'size_factors': L}, batch_size=PREDICTION_BATCH_SIZE)

  def predict_PI(self, X, n_mcmc_samples=100):
    """ Return a matrix (n_sample, n_gene) of Zero-inflated
    rate

    if not a zero-inflated model, then return None
    """
    if 'pi' not in self._ae.extra_models:
      raise RuntimeError("This is not zero-inflated model.")
    X, L = dca_normalize(X, log=self.xnorm == 'log')
    return self._ae.extra_models['pi'].predict({
        'count': X,
        'size_factors': L}, batch_size=PREDICTION_BATCH_SIZE)
