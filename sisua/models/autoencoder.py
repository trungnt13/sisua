# Tensorflow re-implementaiton of: https://github.com/theislab/dca
# License: https://github.com/theislab/dca/blob/master/LICENSE.txt
# Modification by Trung Ngo 2019
from __future__ import absolute_import, division, print_function

from typing import Iterable

import tensorflow as tf
from tensorflow.python.keras.layers import Dense, Layer

from odin.networks import (DenseDeterministic, DenseDistribution, Identity,
                           Parallel)
from sisua.models.base import SingleCellModel
from sisua.models.modules import create_encoder_decoder


class DeepCountAutoencoder(SingleCellModel):
  r""" Deep Count Autoencoder """

  def __init__(self,
               outputs,
               hdim=64,
               zdim=32,
               latent_bias=False,
               nlayers=2,
               xdrop=0.3,
               edrop=0,
               zdrop=0,
               ddrop=0,
               batchnorm=True,
               linear_decoder=False,
               pyramid=False,
               use_conv=False,
               **kwargs):
    super().__init__(outputs, **kwargs)
    self.encoder, self.decoder = create_encoder_decoder(
        input_dim=self.omic_outputs[0].dim, seed=self.seed, **locals())
    self.latent_layer = DenseDeterministic(zdim,
                                           use_bias=bool(latent_bias),
                                           activation='linear',
                                           name='Latent')

  def _call(self, x, lmean, lvar, t, y, mask, training, n_mcmc):
    e = self.encoder(x, training=training)
    qZ = self.latent_layer(e, n_mcmc=n_mcmc)
    # the first dimension always the MCMC sample dimension
    d = self.decoder(qZ.sample(n_mcmc), training=training)
    pX = [p(d, training=training) for p in self.posteriors]
    return pX, qZ
