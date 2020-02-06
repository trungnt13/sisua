from __future__ import absolute_import, division, print_function

import inspect
from typing import List

import tensorflow as tf

from odin.networks import Identity
from sisua.models.base import SingleCellModel
from sisua.models.utils import NetworkConfig, RandomVariable


class VariationalAutoEncoder(SingleCellModel):
  r""" Variational Auto Encoder """

  def __init__(self,
               outputs: List[RandomVariable],
               latent_dim=10,
               network=NetworkConfig(),
               **kwargs):
    latents = kwargs.pop('latents', None)
    if latents is None:
      latents = RandomVariable(latent_dim, 'diag', 'latent'),
    super().__init__(outputs, latents, network, **kwargs)

  def encode(self, x, lmean=None, lvar=None, y=None, training=None, n_mcmc=1):
    # applying encoding
    e = self.encoder(x, training=training)
    # latent distribution
    qZ = self.latents[0](e, training=training, n_mcmc=n_mcmc)
    return qZ

  def decode(self, z, training=None):
    # decoding the latent
    d = self.decoder(z, training=training)
    pX = [p(d, training=training) for p in self.posteriors]
    return pX


class SISUA(VariationalAutoEncoder):
  r""" SemI-SUpervised Autoencoder

  Arguments:
    rna_dim : Integer, number of input dimension for scRNA-seq.
    adt_dim : Integer, number of input dimension for ADT.
    is_adt_probability : Boolean, if True, use `Bernoulli` for modeling the ADT,
      otherwise, use `NegativeBinomial`.
    alternative_nb : Boolean, if True, use mean-dispersion parameterization
      for negative binomial distribution.
  """

  def __init__(self,
               rna_dim=None,
               adt_dim=None,
               latent_dim=10,
               is_adt_probability=False,
               alternative_nb=False,
               **kwargs):
    outputs = kwargs.pop('outputs', None)
    if outputs is None:
      rna = RandomVariable(dim=rna_dim,
                           posterior='zinbd' if alternative_nb else 'zinb',
                           name='RNA')
      adt = RandomVariable(dim=adt_dim,
                           posterior='onehot' if is_adt_probability else
                           ('nbd' if alternative_nb else 'nb'),
                           name='ADT')
      outputs = [rna, adt]
    super().__init__(outputs, latent_dim=latent_dim, **kwargs)


class MISA(VariationalAutoEncoder):
  r""" MIxture Semi-supervised Autoencoder

  Arguments:
    rna_dim : Integer, number of input dimension for scRNA-seq.
    adt_dim : Integer, number of input dimension for ADT.
    mixture_gaussian : Boolean, if True, use GMM for modeling the ADT,
      otherwise, use mixture of `NegativeBinomial`.
    alternative_nb : Boolean, if True, use mean-dispersion parameterization
      for negative binomial distribution.
  """

  def __init__(self,
               rna_dim=None,
               adt_dim=None,
               latent_dim=10,
               mixture_gaussian=False,
               alternative_nb=False,
               **kwargs):
    outputs = kwargs.pop('outputs', None)
    if outputs is None:
      rna = RandomVariable(dim=rna_dim,
                           posterior='zinbd' if alternative_nb else 'zinb',
                           name='RNA')
      kw = dict(n_components=2)
      if not mixture_gaussian:
        kw['alternative'] = alternative_nb
      adt = RandomVariable(
          dim=adt_dim,
          posterior='mixgaussian' if mixture_gaussian else 'mixnb',
          name='ADT',
          **kw)
      outputs = [rna, adt]
    super().__init__(outputs, latent_dim=latent_dim, **kwargs)
