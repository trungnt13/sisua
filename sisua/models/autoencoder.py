# Tensorflow re-implementaiton of: https://github.com/theislab/dca
# License: https://github.com/theislab/dca/blob/master/LICENSE.txt
# Modification by Trung Ngo 2019
from __future__ import absolute_import, division, print_function

from typing import Iterable

import tensorflow as tf
from tensorflow.python.keras.layers import Dense, Layer

from odin.bay.helpers import Statistic
from odin.networks import (DenseDeterministic, DenseDistribution, Identity,
                           Parallel)
from sisua.models.base import SingleCellModel
from sisua.models.networks import DenseNetwork


class DeepCountAutoencoder(SingleCellModel):
  """ Deep Count Autoencoder

  """

  def __init__(self,
               units,
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
    super(DeepCountAutoencoder, self).__init__(units=units,
                                               xdist=xdist,
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
    # initialize output_layers distribution
    # we must set the attribute directly so the Model will manage
    # the output layer, once all the output layers are initialized
    # the number of outputs will match `n_outputs`
    for idx, (units, posterior,
              activation) in enumerate(zip(self.units, self.xdist,
                                           self.xactiv)):
      name = 'output_layer%d' % idx
      post = DenseDistribution(
          units=units,
          posterior=posterior,
          activation=activation,
          posterior_kwargs=dict(dispersion=self.dispersion))
      setattr(self, name, post)

  def _call(self, x, lmean, lvar, t, y, masks, training, n_samples):
    e = self.encoder(x, training=training)
    qZ = self.latent_layer(e)
    # the first dimension always the MCMC sample dimension
    d = self.decoder(qZ.sample(1), training=training)

    pX = [
        getattr(self, 'output_layer%d' % i)(d, mode=Statistic.DIST)
        for i in range(self.n_outputs)
    ]

    # calculating the losses
    loss_x = self.xloss[0](t, pX[0])
    # don't forget to apply mask for semi-supervised loss
    loss_y = tf.convert_to_tensor(0, dtype=x.dtype)
    for i_true, m, i_pred, fn_loss in zip(y, masks, pX[1:], self.xloss[1:]):
      loss_y += fn_loss(i_true, i_pred) * m
    loss = tf.reduce_mean(loss_x + loss_y)

    if training:
      self.add_loss(loss)
    self.add_metric(loss_x, 'mean', "loss_x")
    if self.is_semi_supervised:
      self.add_metric(loss_y, 'mean', "loss_y")

    return pX, qZ
