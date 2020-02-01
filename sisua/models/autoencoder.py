# Tensorflow re-implementaiton of: https://github.com/theislab/dca
# License: https://github.com/theislab/dca/blob/master/LICENSE.txt
# Modification by Trung Ngo 2019
from __future__ import absolute_import, division, print_function

from typing import List

import tensorflow as tf
from tensorflow.python import keras

from odin.bay.layers import DenseDeterministic, DenseDistribution
from odin.networks import Identity, Parallel
from sisua.models.base import SingleCellModel
from sisua.models.utils import NetworkConfig, RandomVariable


class DeepCountAutoencoder(SingleCellModel):
  r""" Deep Count Autoencoder """

  def __init__(self,
               outputs: List[RandomVariable],
               latent_dim=10,
               network=NetworkConfig(),
               **kwargs):
    # force a deterministic latent space:
    latents = kwargs.pop('latents', None)
    if latents is None:
      latents = RandomVariable(latent_dim, 'relu', 'latent'),
    super().__init__(outputs, latents, network, **kwargs)

  def encode(self, x, lmean, lvar, y, training, n_mcmc):
    e = self.encoder(x, training=training)
    qZ = self.latents[0](e, training=training, n_mcmc=n_mcmc)
    return qZ

  def decode(self, z, training):
    # the first dimension always the MCMC sample dimension
    d = self.decoder(z, training=training)
    pX = [p(d, training=training) for p in self.posteriors]
    return pX
