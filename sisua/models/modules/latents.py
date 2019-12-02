from __future__ import absolute_import, division, print_function

from typing import Type

import tensorflow as tf
from tensorflow_probability.python.distributions import (Independent, LogNormal,
                                                         MultivariateNormalDiag,
                                                         Normal)

from odin.bay.distribution_layers import (LogNormalLayer,
                                          MultivariateNormalLayer, NormalLayer)
from odin.networks import DenseDistribution, MixtureDensityNetwork


class NormalDiagLatent(DenseDistribution):

  def __init__(self, units, **kwargs):
    super().__init__(units,
                     posterior=MultivariateNormalLayer,
                     posterior_kwargs=dict(covariance='diag',
                                           scale_activation='softplus1'),
                     prior=MultivariateNormalDiag(loc=tf.zeros(shape=units),
                                                  scale_identity_multiplier=1),
                     **kwargs)


class NormalLatent(DenseDistribution):

  def __init__(self, units, **kwargs):
    super().__init__(units,
                     posterior=NormalLayer,
                     posterior_kwargs=dict(scale_activation='softplus1'),
                     prior=Independent(
                         Normal(loc=tf.zeros(shape=units),
                                scale=tf.ones(shape=units)), 1),
                     **kwargs)


class LogNormalLatent(DenseDistribution):

  def __init__(self, units, **kwargs):
    super().__init__(units=units,
                     posterior=LogNormalLayer,
                     posterior_kwargs=dict(scale_activation='softplus1'),
                     prior=Independent(
                         LogNormal(loc=tf.zeros(shape=units),
                                   scale=tf.ones(shape=units)), 1),
                     **kwargs)


class MixtureNormalLatent(MixtureDensityNetwork):

  def __init__(self, units, n_components=8, **kwargs):
    kwargs['covariance'] = 'none'
    kwargs['n_components'] = int(n_components)
    super().__init__(units, **kwargs)
    self.set_prior()


class MixtureDiagLatent(MixtureDensityNetwork):

  def __init__(self, units, n_components=8, **kwargs):
    kwargs['covariance'] = 'diag'
    kwargs['n_components'] = int(n_components)
    super().__init__(units, **kwargs)
    self.set_prior()


class MixtureFullLatent(MixtureDensityNetwork):

  def __init__(self, units, n_components=8, **kwargs):
    kwargs['covariance'] = 'full'
    kwargs['n_components'] = int(n_components)
    super().__init__(units, **kwargs)
    self.set_prior()


class DirichletLatent(DenseDistribution):
  pass


# ===========================================================================
# Helpers
# ===========================================================================
_latent_map = dict(normal=NormalLatent,
                   diag=NormalDiagLatent,
                   mdn=MixtureNormalLatent,
                   mixdiag=MixtureDiagLatent,
                   mixfull=MixtureFullLatent,
                   diri=DirichletLatent,
                   lognormal=LogNormalLatent)


def get_latent(distribution_name, units, **kwargs):
  distribution_name = str(distribution_name).lower()
  units = kwargs.pop('units', units)
  if distribution_name not in _latent_map:
    raise ValueError("Only support following latent: %s; but given: '%s'" %
                     (', '.join(_latent_map.keys()), distribution_name))
  return _latent_map[distribution_name](units, **kwargs)
