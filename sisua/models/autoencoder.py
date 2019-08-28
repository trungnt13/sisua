from __future__ import absolute_import, division, print_function

from typing import Iterable

import tensorflow as tf
from tensorflow.python.keras.layers import Dense, Layer

from odin.bay.distribution_layers import (NegativeBinomialDispLayer,
                                          NegativeBinomialLayer, PoissonLayer,
                                          ZINegativeBinomialDispLayer,
                                          ZINegativeBinomialLayer,
                                          ZIPoissonLayer)
from odin.bay.helpers import Statistic
from odin.networks import (DenseDeterministic, DenseDistribution, Identity,
                           Parallel)
from sisua.models.base import SingleCellModel
from sisua.models.networks import DenseNetwork


class DeepCountAutoencoder(SingleCellModel):
  """ Deep Count Autoencoder

  """

  def __init__(self,
               dispersion='full',
               xdist='zinb',
               xdrop=0.3,
               edrop=0,
               zdrop=0,
               ddrop=0,
               hdim=128,
               zdim=32,
               biased_latent=False,
               nlayers=2,
               batchnorm=True,
               linear_decoder=False,
               **kwargs):
    super(DeepCountAutoencoder, self).__init__(xdist=xdist,
                                               dispersion=dispersion,
                                               parameters=locals(),
                                               **kwargs)
    self.encoder = DenseNetwork(n_units=hdim,
                                nlayers=nlayers,
                                activation='relu',
                                batchnorm=batchnorm,
                                input_dropout=xdrop,
                                output_dropout=edrop,
                                seed=self.seed,
                                name='Encoder')
    if linear_decoder:
      self.decoder = Identity(name="Decoder")
    else:
      self.decoder = DenseNetwork(n_units=hdim,
                                  nlayers=nlayers,
                                  activation='relu',
                                  batchnorm=batchnorm,
                                  input_dropout=zdrop,
                                  output_dropout=ddrop,
                                  seed=self.seed,
                                  name='Decoder')
    self.latent_layer = DenseDeterministic(zdim,
                                           use_bias=bool(biased_latent),
                                           activation='linear',
                                           name='Latent')

  def _call(self, x, lmean, lvar, t, y, masks, training, n_samples):
    e = self.encoder(x, training=training)
    qZ = self.latent_layer(e)
    # the first dimension always the MCMC sample dimension
    d = self.decoder(qZ.sample(1), training=training)
    pX = [dist(d, mode=Statistic.DIST) for dist in self.output_layers]
    # calculating the losses
    loss_x = self.xloss[0](t, pX[0])
    # don't forget to apply mask for semi-supervised loss
    loss_y = 0
    for i_true, m, i_pred, fn_loss in zip(y, masks, pX[1:], self.xloss[1:]):
      loss_y += fn_loss(i_true, i_pred) * m
    loss = tf.reduce_mean(loss_x + loss_y)

    if training:
      self.add_loss(loss)
    self.add_metric(loss_x, 'mean', "loss_x")
    if self.is_semi_supervised:
      self.add_metric(loss_y, 'mean', "loss_y")

    return pX, qZ
