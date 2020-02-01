from __future__ import absolute_import, division, print_function

import inspect

import tensorflow as tf

from odin.networks import Identity
from sisua.models.base import SingleCellModel
from sisua.models.utils import NetworkConfig, RandomVariable



class SCALE(SingleCellModel):
  r""" Tensorflow implementation of SCALE

   Author: Lei Xiong - https://github.com/jsxlei
   License: https://github.com/jsxlei/SCALE/blob/master/LICENSE

   Reference:
    Xiong, L., Xu, K., Tian, K., et al., 2019. SCALE method for single-cell
      ATAC-seq analysis via latent feature extraction. Nature Communications.
      https://www.nature.com/articles/s41467-019-12630-7
  """

  def __init__(self,
               outputs,
               latent_dim=10,
               latent_component=8,
               network=NetworkConfig(),
               **kwargs):
    kwargs['analytic'] = False
    latents = kwargs.pop('latents', None)
    if latents is None:
      latents = RandomVariable(latent_dim,
                               'mixdiag',
                               n_components=int(latent_component))
    super().__init__(outputs, latents, network, **kwargs)

  def encode(self, x, lmean, lvar, y, training, n_mcmc):
    # applying encoding
    e = self.encoder(x, training=training)
    # latent distribution
    qZ = self.latents[0](e, training=training, n_mcmc=n_mcmc)
    return qZ

  def decode(self, z, training):
    # decoding the latent
    d = self.decoder(z, training=training)
    pX = [p(d, training=training) for p in self.posteriors]
    return pX
