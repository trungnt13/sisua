from __future__ import print_function, division, absolute_import

from six import add_metaclass
from abc import ABCMeta, abstractmethod

import tensorflow as tf
from tensorflow.python.keras import Model, Sequential
from tensorflow.python.keras.layers import (
  Layer, Dense, Dropout, Lambda, BatchNormalization, Activation)
from tensorflow_probability.python.distributions import (
  Distribution, Normal, NegativeBinomial, Poisson, MultivariateNormalDiag)

from odin.bay.distribution_layers import (
  GaussianLayer, DistributionLambda, NegativeBinomialLayer, PoissonLayer,
  MultivariateNormalLayer)
from odin.bay.distribution_util_layers import (Moments, Sampling)
from odin.bay.helpers import kl_divergence

__all__ = [
  'DistributionLayer',
  'NormalDiagLatent',
  'DenseNetwork',
]

# ===========================================================================
# Latent space
# ===========================================================================
class DistributionLayer(Model):

  """
  Parameters
  ----------
  n_units : int
    number of output units.

  posterior : subclass of tensorflow_probability.DistributionLambda
    posterior distribution, the class is given for later
    initialization

  prior : None or tensorflow_probability.Distribution
    prior distribution, used for calculating KL divergence later.

  """

  def __init__(self, n_units,
               posterior=GaussianLayer, prior=None,
               use_bias=True, name=None):
    if name is None:
      name = "DistributionLayer"
    super(DistributionLayer, self).__init__(name=name)
    assert isinstance(posterior, DistributionLambda) or\
       (isinstance(posterior, type) and issubclass(posterior, DistributionLambda)),\
         "posterior must be instance or subclass of DistributionLambda"
    self._n_units = int(n_units)
    layers = [
      Dense(posterior.params_size(self.n_units),
            activation='linear', use_bias=bool(use_bias)),
      posterior if isinstance(posterior, DistributionLambda) else
      posterior(self.n_units),
    ]
    if isinstance(posterior, DistributionLambda):
      distribution_type = type(posterior)
    else:
      distribution_type = posterior
    self._distribution = Sequential(
      layers, name="%s%s" % (name, distribution_type.__name__))
    self._last_distribution = None
    # check the prior, this could be given later
    assert prior is None or isinstance(prior, Distribution), \
      "prior can be None or instance of tensorflow_probability.Distribution"
    self.prior = prior

  @property
  def n_units(self):
    return self._n_units

  def mean(self, x):
    dist = self._distribution(x)
    y = Moments(variance=False)(dist)
    setattr(y, '_distribution', dist)
    self._last_distribution = y._distribution
    return y

  def variance(self, x):
    dist = self._distribution(x)
    y = Moments(mean=False)(dist)
    setattr(y, '_distribution', dist)
    self._last_distribution = y._distribution
    return y

  def stddev(self, x):
    return Lambda(tf.math.sqrt)(self.variance(x))

  def sample(self, x, n_samples=None):
    if n_samples is None or n_samples <= 0:
      n_samples = 1
    dist = self._distribution(x)
    y = Sampling(n_samples=n_samples)(dist)
    setattr(y, '_distribution', dist)
    self._last_distribution = y._distribution
    return y

  def call(self, x, n_samples=1, training=None):
    return self.sample(x, n_samples=n_samples)

  def kl_divergence(self, prior=None, analytic_kl=True, n_samples=1):
    """
    Parameters
    ---------
    prior : instance of tensorflow_probability.Distribution
      prior distribution of the latent

    analytic_kl : bool
      using closed form solution for calculating divergence,
      otherwise, sampling with MCMC
    """
    if prior is None:
      prior = self.prior
    assert isinstance(prior, Distribution), "prior is not given!"
    if self._last_distribution is None:
      raise RuntimeError(
        "DistributionLayer must be called to create the distribution before "
        "calculating the kl-divergence.")
    kl = kl_divergence(q=self._last_distribution, p=prior,
                       use_analytic_kl=bool(analytic_kl),
                       q_sample=int(n_samples),
                       auto_remove_independent=True)
    return kl

  def log_prob(self, x):
    assert self.n_units == x.shape[-1], \
      "Number of features mismatch, n_units=%d  input_shape=%s" % \
        (self.n_units, str(x.shape))
    if self._last_distribution is None:
      raise RuntimeError(
        "DistributionLayer must be called to create the distribution before "
        "calculating the log-likelihood.")
    dist = self._last_distribution
    return dist.log_prob(x)

# ===========================================================================
# Latent space
# ===========================================================================
class NormalDiagLatent(DistributionLayer):
  def __init__(self, n_units, use_bias=True, name=None):
    super(NormalDiagLatent, self).__init__(
      n_units=n_units,
      posterior=MultivariateNormalLayer(
        event_size=n_units, covariance_type='diag', softplus_scale=True),
      prior=MultivariateNormalDiag(
        loc=tf.zeros(shape=n_units), scale_identity_multiplier=1),
      use_bias=use_bias,
      name="LatentSpace"
    )

# ===========================================================================
# Basic networks
# ===========================================================================
class DenseNetwork(Model):

  def __init__(self, n_units=128, n_layers=2,
               activation='relu', batchnorm=True,
               input_dropout=0., output_dropout=0,
               seed=8, name=None):
    super(DenseNetwork, self).__init__(name=name)
    layers = []
    if 0. < input_dropout < 1.:
      layers.append(Dropout(input_dropout, seed=seed))
    for i in range(int(n_layers)):
      layers.append(Dense(n_units,
                          activation='linear' if batchnorm else activation,
                          use_bias=False if batchnorm else True,
                          name="DenseLayer%d" % i))
      if batchnorm:
        layers.append(BatchNormalization())
        layers.append(Activation(activation))
    if 0. < output_dropout < 1.:
      layers.append(Dropout(output_dropout, seed=seed))
    self._network = Sequential(layers)

  def call(self, x, training=None):
    return self._network(x, training=training)

# ===========================================================================
# SingleCell model
# ===========================================================================
_SUPPORT_MODE = {'sample', 'mean', 'stddev', 'all'}

@add_metaclass(ABCMeta)
class SingleCellModel(Model):

  def __init__(self, **kwargs):
    super(SingleCellModel, self).__init__(**kwargs)

  @abstractmethod
  def get_losses_and_metrics(self, inputs, n_mcmc_samples=1):
    """ Return training loss and dictionary of metrics """
    raise NotImplementedError

  def fit(self, inputs, optimizer='adam',
          n_mcmc_samples=1, supervised_percent=0.8,
          corruption_rate=0.25, corruption_dist='binomial',
          batch_size=None, epochs=1, callbacks=None,
          validation_split=0., validation_freq=1,
          shuffle=True, seed=8,
          verbose=1,
          **kwargs):
    """ This fit function is the combination of both
    `Model.compile` and `Model.fit` """
    pass

  def get_latents(self, inputs, n_mcmc_samples=1):
    pass

  def get_outputs(self, inputs, n_mcmc_samples=1):
    raise NotImplementedError