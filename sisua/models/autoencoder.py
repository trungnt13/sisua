from __future__ import absolute_import, division, print_function

from typing import Iterable

import tensorflow as tf
from tensorflow.python.keras.layers import Dense, Layer

from odin.backend import DeterministicDense, Identity, Parallel
from odin.bay.distribution_layers import (NegativeBinomialLayer, PoissonLayer,
                                          ZeroInflatedNegativeBinomialLayer,
                                          ZeroInflatedPoissonLayer)
from odin.bay.helpers import Statistic
from odin.bay.models import DistributionLayer
from sisua.models.base import SingleCellModel
from sisua.models.networks import DenseNetwork


def _get_loss(loss):
  loss = str(loss).lower()
  is_probabilistic_loss = True
  if loss == 'poisson':
    output_layer = lambda units: DistributionLayer(units,
                                                   posterior=PoissonLayer)
  elif loss == 'zipoisson':
    output_layer = lambda units: DistributionLayer(units,
                                                   posterior=PoissonLayer)
  elif loss == 'nb':
    output_layer = lambda units: DistributionLayer(units,
                                                   posterior=PoissonLayer)
  elif loss == 'zinb':
    output_layer = lambda units: DistributionLayer(units,
                                                   posterior=PoissonLayer)
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
               loss='mse',
               xdrop=0.3,
               edrop=0,
               zdrop=0,
               ddrop=0,
               hdim=128,
               zdim=32,
               biased_latent=False,
               n_layers=2,
               batchnorm=True,
               linear_decoder=False,
               **kwargs):
    super(DeepCountAutoencoder, self).__init__(**kwargs)
    self.encoder = DenseNetwork(n_units=hdim,
                                n_layers=n_layers,
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
                                  n_layers=n_layers,
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
    self.loss_fn, self.output_layer, self._is_probabilistic_loss = _get_loss(
        loss)

  @property
  def is_probabilistic_loss(self):
    return self._is_probabilistic_loss

  @property
  def is_semi_supervised(self):
    return False

  def _apply_network(self, inputs, training, n_samples):
    # initialization
    units = inputs.shape[1]
    if not isinstance(self.output_layer, Layer):
      self.output_layer = self.output_layer(units)

    # applying the layers
    e = self.encoder(inputs, training=training)
    z = self.latent_layer(e)
    d = self.decoder(z, training=training)

    if self.is_probabilistic_loss:
      pred = self.output_layer(d, mode=Statistic.DIST)
    else:
      pred = self.output_layer(d)
    return e, z, d, pred

  def call(self, inputs, training=None, n_samples=None):
    e, z, d, pred = self._apply_network(inputs, training, n_samples)

    # NOTE: TensorArray could be used to remove the redundancy of
    # loss computation during prediction, however, Tensorflow Autograph
    # cannot track the gradient here.
    # loss = tf.TensorArray(dtype='float32', size=1, dynamic_size=False)
    # loss.write(0, tf.reduce_mean(tf.losses.mean_squared_error(inputs, pred)))
    # self.add_loss(lambda: loss.read(0))

    # calculating the losses
    loss = self.loss_fn(inputs, pred)
    loss = tf.reduce_mean(loss)

    if training:
      self.add_loss(lambda: loss)

    return pred, z


class MultitaskAutoEncoder(DeepCountAutoencoder):
  """
  """

  def __init__(self,
               loss='mse',
               loss_y='mse',
               xdrop=0.3,
               edrop=0,
               zdrop=0,
               ddrop=0,
               hdim=128,
               zdim=32,
               biased_latent=False,
               n_layers=2,
               batchnorm=True,
               linear_decoder=False,
               **kwargs):
    kw = dict(locals())
    del kw['self']
    del kw['kwargs']
    del kw['loss_y']
    del kw['__class__']
    kw.update(kwargs)
    super(MultitaskAutoEncoder, self).__init__(**kw)
    if not isinstance(loss_y, (tuple, list)):
      loss_y = [loss_y]
    self.loss_y = [_get_loss(i) for i in loss_y]
    self.output_layer_y = None

  @property
  def is_semi_supervised(self):
    return True

  def call(self, inputs, training=None, n_samples=None):
    x, y, masks = self._to_semisupervised_inputs(inputs)
    if len(y) != len(self.loss_y):
      raise RuntimeError(
          "given %d inputs for semi-supervised, but %d values for loss function"
          % (len(y), len(self.loss_y)))

    # initialization
    e, z, d, pred_x = self._apply_network(x, training, n_samples)
    if self.output_layer_y is None:
      self.output_layer_y = Parallel(
          [j[1](i.shape[1]) for i, j in zip(y, self.loss_y)],
          name="ParallelOutputs")
    pred_y = self.output_layer_y(d, mode=Statistic.DIST)

    # calculating the losses
    loss_x = self.loss_fn(x, pred_x)
    # don't forget to apply mask for semi-supervised loss
    loss_y = 0
    for (fn, _, _), i_true, i_pred, m in zip(self.loss_y, y, pred_y, masks):
      loss_y += fn(i_true, i_pred) * m
    loss = tf.reduce_mean(loss_x + loss_y)
    if training:
      self.add_loss(lambda: loss)

    self.add_metric(loss_x, 'mean', "Xloss")
    self.add_metric(loss_y, 'mean', "Yloss")

    outputs = (pred_x,) + tuple(pred_y)
    return outputs, z
