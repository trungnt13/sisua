from __future__ import absolute_import, division, print_function

import inspect
from typing import List

import tensorflow as tf

from odin.networks import Identity
from sisua.models.base import OmicOutput, SingleCellModel
from sisua.models.modules import create_encoder_decoder, get_latent


class VariationalAutoEncoder(SingleCellModel):
  r""" Variational Auto Encoder """

  def __init__(self,
               outputs: List[OmicOutput],
               zdim=32,
               zdist='diag',
               hdim=64,
               nlayers=2,
               xdrop=0.3,
               edrop=0,
               zdrop=0,
               ddrop=0,
               batchnorm=True,
               linear_decoder=False,
               pyramid=False,
               use_conv=False,
               kernel=5,
               stride=2,
               **kwargs):
    super().__init__(outputs, **kwargs)
    self.encoder, self.decoder = create_encoder_decoder(
        input_dim=self.omic_outputs[0].dim, seed=self.seed, **locals())
    self.latent = get_latent(zdist, zdim)

  def _call(self, x, lmean, lvar, t, y, mask, training, n_mcmc):
    # applying encoding
    e = self.encoder(x, training=training)
    # latent distribution
    qZ = self.latent(e, training=training, n_mcmc=n_mcmc)
    # decoding the latent
    d = self.decoder(qZ, training=training)
    pX = [p(d, training=training) for p in self.posteriors]
    return pX, qZ


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
               rna_dim,
               adt_dim,
               is_adt_probability=False,
               alternative_nb=False,
               zdim=32,
               zdist='diag',
               hdim=128,
               nlayers=2,
               xdrop=0.3,
               edrop=0,
               zdrop=0,
               ddrop=0,
               batchnorm=True,
               linear_decoder=False,
               pyramid=False,
               use_conv=False,
               kernel=5,
               stride=2,
               **kwargs):
    rna = OmicOutput(dim=rna_dim,
                     posterior='zinbd' if alternative_nb else 'zinb',
                     name='RNA')
    adt = OmicOutput(dim=adt_dim,
                     posterior='onehot' if is_adt_probability else
                     ('nbd' if alternative_nb else 'nb'),
                     name='ADT')
    super().__init__(outputs=[rna, adt],
                     log_norm=True,
                     zdim=zdim,
                     zdist=zdist,
                     hdim=hdim,
                     nlayers=nlayers,
                     xdrop=xdrop,
                     edrop=edrop,
                     zdrop=zdrop,
                     ddrop=ddrop,
                     batchnorm=batchnorm,
                     linear_decoder=linear_decoder,
                     pyramid=pyramid,
                     use_conv=use_conv,
                     kernel=kernel,
                     stride=stride,
                     **kwargs)


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
               rna_dim,
               adt_dim,
               mixture_gaussian=False,
               alternative_nb=False,
               zdim=32,
               zdist='diag',
               hdim=128,
               nlayers=2,
               xdrop=0.3,
               edrop=0,
               zdrop=0,
               ddrop=0,
               batchnorm=True,
               linear_decoder=False,
               pyramid=False,
               use_conv=False,
               kernel=5,
               stride=2,
               **kwargs):
    rna = OmicOutput(dim=rna_dim,
                     posterior='zinbd' if alternative_nb else 'zinb',
                     name='RNA')
    kw = dict(n_components=2)
    if not mixture_gaussian:
      kw['alternative'] = alternative_nb
    adt = OmicOutput(dim=adt_dim,
                     posterior='mixgaussian' if mixture_gaussian else 'mixnb',
                     name='ADT',
                     **kw)
    super().__init__(outputs=[rna, adt],
                     log_norm=True,
                     zdim=zdim,
                     zdist=zdist,
                     hdim=hdim,
                     nlayers=nlayers,
                     xdrop=xdrop,
                     edrop=edrop,
                     zdrop=zdrop,
                     ddrop=ddrop,
                     batchnorm=batchnorm,
                     linear_decoder=linear_decoder,
                     pyramid=pyramid,
                     use_conv=use_conv,
                     kernel=kernel,
                     stride=stride,
                     **kwargs)
