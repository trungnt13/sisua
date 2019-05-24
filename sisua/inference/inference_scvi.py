from __future__ import print_function, division, absolute_import

import io
from io import BytesIO
from collections import defaultdict

import numpy as np
from tqdm import tqdm

from odin.utils import batching, stdio, get_formatted_datetime

from .inference_base import Inference
from ._consts import PREDICTION_BATCH_SIZE
from sisua.data import get_library_size, apply_artificial_corruption
from sisua.data.const import UNIVERSAL_RANDOM_SEED

# ===========================================================================
# scVI Helpers
# ===========================================================================
try:
  import torch
  torch_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  torch.manual_seed(UNIVERSAL_RANDOM_SEED)
except ImportError:
  torch_device = None

def _create_gene_dataset(X, local_means=None, local_vars=None):
  from scvi.dataset import GeneExpressionDataset
  if local_means is None or local_vars is None:
    local_means, local_vars = get_library_size(X)
  col_names = np.array(['Gene#%d' % i for i in range(X.shape[1])])
  gene_dataset = GeneExpressionDataset(
      X=X,
      local_means=local_means,
      local_vars=local_vars,
      batch_indices=np.zeros(shape=(X.shape[0], 1)),
      labels=None,
      gene_names=col_names,
      cell_types=None)
  gene_dataset.labels = X
  gene_dataset.cell_types = col_names
  return gene_dataset

def _create_data_iter(X, log_variational):
  import torch
  # tqdm(,ncols=80, desc="Predicting")
  for start, end in batching(batch_size=PREDICTION_BATCH_SIZE // 3,
                             n=X.shape[0]):
    x = _to_tensor(X[start:end])
    if bool(log_variational):
      x = torch.log(1 + x)
    yield x

def _to_tensor(x):
  import torch
  return torch.from_numpy(x.astype('float32')).to(torch_device)

def _to_array(x):
  if hasattr(x, 'todense'):
    return np.array(x.todense())
  if hasattr(x, 'cpu'):
    return x.data.cpu().numpy()
  return x

def _use_cuda():
  import torch
  return torch.cuda.is_available()

# ===========================================================================
# Main
# ===========================================================================
class InferenceSCVI(Inference):
  """ InferenceSCVI """

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
      import scvi
    except ImportError as e:
      raise RuntimeError(
          "scVI package at https://github.com/YosefLab/scVI is required")
    model = kwargs.get('model', 'vae')
    prot_dim = None
    if model not in ('vae', 'scvi'):
      raise RuntimeError("InferenceSCVI only support 'vae' model")
    if xdist not in ('zinb', 'nb'):
      raise ValueError("Only support 'zinb' and 'nb' for `xdist`")
    # ====== main model ====== #
    if xnorm not in ('log', 'raw'):
      raise ValueError(
          "Only support log variational or raw count variational for scVI")
    if tnorm not in ('raw',):
      raise ValueError(
          "Only support raw count target value for scVI")
    self._name = 'scvi'
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
    self._initialize_model()

  # ******************** pickling ******************** #
  def _initialize_model(self, weights=None):
    from scvi.models import VAE
    self._vae = VAE(n_input=self.gene_dim, n_batch=0, n_labels=0,
                    n_hidden=self.hdim, n_latent=self.zdim, n_layers=self.nlayer,
                    dropout_rate=self.xdrop,
                    dispersion=self.dispersion,
                    log_variational=True if self.xnorm == 'log' else False,
                    reconstruction_loss=self.xdist)
    if weights is not None:
      data = BytesIO(weights)
      self._vae.load_state_dict(torch.load(data))
      self._vae.to(torch_device)

  def __getstate__(self):
    import torch
    states = self.get_states()
    weights = BytesIO()
    torch.save(self._vae.state_dict(), weights)
    weights.seek(0)
    weights = weights.read()
    return states, weights

  def __setstate__(self, states):
    states, weights = states
    self.set_states(states)
    self._initialize_model(weights=weights)

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
    # ====== check input dims ====== #
    assert X.shape[1] == self.gene_dim,\
    "Number of gene expression mismatch: %d != %d" % (self.gene_dim, X.shape[1])
    if validation_percent == 0:
      raise RuntimeError("scVI doesn't support 0% validation percent!")
    # ====== training ====== #
    from scvi.inference import UnsupervisedTrainer
    gene_dataset = _create_gene_dataset(X)
    # batchnorm bug in scVI, if return batch is 1, then
    # it raise Exception, just modify the batch size so
    # it never return any batch with size 1 for Cortex dataset
    # wouldn't make a significant different
    if 'dataset' in self.configs and self.configs['dataset'] == 'cortex':
      batch_size = batch_size - 1
    trainer = UnsupervisedTrainer(self._vae, gene_dataset,
        use_cuda=_use_cuda(),
        train_size=1 - validation_percent,
        frequency=1, benchmark=False,
        data_loader_kwargs=dict(batch_size=batch_size),
        metrics_to_monitor=None,
        verbose=detail_logging)
    # scVI will do the normalization by itself
    trainer.train(n_epochs=n_epoch, lr=learning_rate)
    self._scvi_trainer = trainer
    # ====== save history ====== #
    self._history['train']['loss'] += trainer.history["ll_train_set"][1:]
    self._history['valid']['loss'] += trainer.history["ll_test_set"][1:]
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
    self._vae.eval()
    from scvi.inference import Posterior
    pos = Posterior(model=self._vae, gene_dataset=_create_gene_dataset(X),
                    shuffle=False, use_cuda=_use_cuda(),
                    data_loader_kwargs=dict(batch_size=PREDICTION_BATCH_SIZE // 3))
    ll = pos.ll(verbose=False)
    marginal_ll = pos.marginal_ll(verbose=False, n_mc_samples=n_mcmc_samples)
    return {'loss': ll, 'llk': marginal_ll}

  # ******************** predicting ******************** #
  def predict_Z(self, X):
    """ Return mean of the latent posterior
    (i.e. mean of Normal distribution) """
    self._vae.eval()
    from scvi.inference import Posterior
    pos = Posterior(model=self._vae, gene_dataset=_create_gene_dataset(X),
                    shuffle=False, use_cuda=_use_cuda(),
                    data_loader_kwargs={'batch_size': 64})
    return pos.get_latent(sample=False)[0]

  def predict_y(self, X, y, n_mcmc_samples=100):
    """ Predicting the protein marker """
    assert self.is_semi_supervised and y is not None,\
    "Only semi-supervised model has this prediction, and `y` must not be None"

  def predict_L(self, X, n_mcmc_samples=100):
    """ Return library size prediction """
    self._vae.eval()
    L = []
    for x in _create_data_iter(X, log_variational=self._vae.log_variational):
      ql_m, ql_v, library = self._vae.l_encoder(x)
      L.append(_to_array(ql_m))
    return np.concatenate(L, axis=0)

  def predict_Lstddev(self, X, n_mcmc_samples=100):
    """ Return library size stddev prediction """
    self._vae.eval()
    Lstddev = []
    for x in _create_data_iter(X, log_variational=self._vae.log_variational):
      ql_m, ql_v, library = self._vae.l_encoder(x)
      Lstddev.append(_to_array(ql_v))
    return np.concatenate(Lstddev, axis=0)

  def predict_W(self, X, n_mcmc_samples=100):
    """ Return mean of the reconstructed value """
    self._vae.eval()
    W = []
    for x in _create_data_iter(X, log_variational=False):
      # scVI do log normalization within .inference()
      (px_scale, px_r, px_rate, px_dropout,
       qz_m, qz_v, z,
       ql_m, ql_v, library) = self._vae.inference(
          x, batch_index=None, y=None, n_samples=n_mcmc_samples)
      w = (1 - px_dropout) * px_rate
      w = w.mean(0)
      W.append(_to_array(w))
    return np.concatenate(W, axis=0)

  def predict_Wstddev(self, X, n_mcmc_samples=100):
    """ Return stddev of reconstructed value
    if not a variational model, then V_stddev is None
    """
    self._vae.eval()
    import torch
    Wstddev = []
    for x in _create_data_iter(X, log_variational=False):
      # scVI do log normalization within .inference()
      (px_scale, px_r, px_rate, px_dropout,
       qz_m, qz_v, z,
       ql_m, ql_v, library) = self._vae.inference(
          x, batch_index=None, y=None, n_samples=n_mcmc_samples)
      w_mean = (1 - px_dropout) * px_rate
      # (1 - pi) * (d.var + d.mean^2) - [(1 - pi) * d.mean]^2
      w_var = (1 - px_dropout) * (px_r + px_rate**2) - w_mean**2
      w_std = torch.sqrt(w_var).mean(0)
      Wstddev.append(_to_array(w_std))
    return np.concatenate(Wstddev, axis=0)

  def predict_V(self, X, n_mcmc_samples=100):
    """ Return mean of the denoised value """
    self._vae.eval()
    V = []
    for x in _create_data_iter(X, log_variational=False):
      # scVI do log normalization within .inference()
      out = self._vae.get_sample_rate(x, n_samples=n_mcmc_samples).mean(0)
      V.append(_to_array(out))
    return np.concatenate(V, axis=0)

  def predict_Vstddev(self, X, n_mcmc_samples=100):
    """ Return stddev of denoised value
    if not a variational model, then V_stddev is None
    """
    self._vae.eval()
    Vstddev = []
    for x in _create_data_iter(X, log_variational=False):
      n = x.shape[0]
      # px_r
      out = self._vae.inference(
          x, batch_index=None, y=None, n_samples=n_mcmc_samples)[1]
      if self.dispersion == 'gene-cell':
        out = np.mean(_to_array(out), axis=0)
      else:
        out = np.expand_dims(_to_array(out), axis=0)
        out = np.repeat(out, repeats=n, axis=0)
      Vstddev.append(out)
    return np.concatenate(Vstddev, axis=0)

  def predict_PI(self, X, n_mcmc_samples=100):
    """ Return a matrix (n_sample, n_gene) of Zero-inflated
    rate

    if not a zero-inflated model, then return None
    """
    self._vae.eval()
    import torch
    op = torch.nn.Sigmoid()
    PI = []
    for x in _create_data_iter(X, log_variational=False):
      # px_dropout (logits)
      out = self._vae.inference(
          x, batch_index=None, y=None, n_samples=n_mcmc_samples)[3]
      out = op(out.mean(0))
      out = _to_array(out)
      PI.append(out)
    return np.concatenate(PI, axis=0)
