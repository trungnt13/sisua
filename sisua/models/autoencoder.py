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
from odin.networks import (DeterministicDense, DistributionDense, Identity,
                           Parallel)
from sisua.models.base import SingleCellModel
from sisua.models.networks import DenseNetwork


def _get_loss(loss, dispersion=None):
  loss = str(loss).lower()
  is_probabilistic_loss = True
  # ====== Poisson ====== #
  if loss == 'poisson':
    output_layer = lambda units: DistributionDense(units,
                                                   posterior=PoissonLayer)
  elif loss == 'zipoisson':
    output_layer = lambda units: DistributionDense(units,
                                                   posterior=ZIPoissonLayer)
  # ====== NB ====== #
  elif loss == 'nb':
    output_layer = lambda units: DistributionDense(
        units,
        posterior=lambda units: NegativeBinomialLayer(units,
                                                      dispersion=dispersion))
  elif loss == 'zinb':
    output_layer = lambda units: DistributionDense(
        units,
        posterior=lambda units: ZINegativeBinomialLayer(units,
                                                        dispersion=dispersion))
  # ====== alternate parameterization for NB ====== #
  elif loss == 'nbd':
    output_layer = lambda units: DistributionDense(
        units,
        posterior=lambda units: NegativeBinomialDispLayer(
            units, dispersion=dispersion))
  elif loss == 'zinbd':
    output_layer = lambda units: DistributionDense(
        units,
        posterior=lambda units: ZINegativeBinomialDispLayer(
            units, dispersion=dispersion))
  # ====== deterministic loss function ====== #
  else:
    is_probabilistic_loss = False
    output_layer = lambda units: DeterministicDense(units, activation='relu')
    loss_fn = tf.losses.get(str(loss))
  # post-processing the loss function to be more universal
  if is_probabilistic_loss:
    loss_fn = lambda y_true, y_pred: -y_pred.log_prob(y_true)
  return loss_fn, output_layer, is_probabilistic_loss


class DeepCountAutoencoder(SingleCellModel):
  """ Deep Count Autoencoder

  """

  def __init__(self,
               dispersion='full',
               loss='zinb',
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
    super(DeepCountAutoencoder, self).__init__(parameters=locals(), **kwargs)
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
    self.latent_layer = DeterministicDense(zdim,
                                           use_bias=bool(biased_latent),
                                           activation='linear',
                                           name='Latent')
    # loss funciton
    if not isinstance(loss, (tuple, list)):
      loss = [loss]
    self.loss_info = [_get_loss(i, dispersion=dispersion) for i in loss]
    self.output_layer = None

  @property
  def is_semi_supervised(self):
    return len(self.loss_info) > 1

  def _call(self, x, t, y, masks, training, n_samples):
    e = self.encoder(x, training=training)
    z = self.latent_layer(e)
    d = self.decoder(z, training=training)

    if self.output_layer is None:
      self.output_layer = Parallel([
          layer(i.shape[1]) for i, (_, layer, _) in zip([x] + y, self.loss_info)
      ],
                                   name="Outputs")
    pred = self.output_layer(d, mode=Statistic.DIST)

    # NOTE: TensorArray could be used to remove the redundancy of
    # loss computation during prediction, however, Tensorflow Autograph
    # cannot track the gradient here.
    # loss = tf.TensorArray(dtype='float32', size=1, dynamic_size=False)
    # loss.write(0, tf.reduce_mean(tf.losses.mean_squared_error(inputs, pred)))
    # self.add_loss(lambda: loss.read(0))

    # calculating the losses
    loss_x = self.loss_info[0][0](t, pred[0])
    # don't forget to apply mask for semi-supervised loss
    loss_y = 0
    for (fn, _, _), i_true, i_pred, m in zip(self.loss_info[1:], y, pred[1:],
                                             masks):
      loss_y += fn(i_true, i_pred) * m
    loss = tf.reduce_mean(loss_x + loss_y)

    if training:
      self.add_loss(loss)

    self.add_metric(loss_x, 'mean', "loss_x")
    if self.is_semi_supervised:
      self.add_metric(loss_y, 'mean', "loss_y")

    return pred if self.is_semi_supervised else pred[0], z
