from __future__ import absolute_import, division, print_function

from typing import Iterable

import tensorflow as tf
from tensorflow.python.keras.layers import Dense

from odin.backend import Parallel
from sisua.models.base import SingleCellModel
from sisua.models.networks import DenseNetwork


class Autoencoder(SingleCellModel):
  """
  """

  def __init__(self,
               xdrop=0.3,
               edrop=0,
               zdrop=0,
               ddrop=0,
               hdim=128,
               zdim=32,
               biased_latent=False,
               n_layers=2,
               batchnorm=True,
               name='Autoencoder',
               **kwargs):
    super(Autoencoder, self).__init__(name=name, **kwargs)
    self.encoder = DenseNetwork(n_units=hdim,
                                n_layers=n_layers,
                                activation='relu',
                                batchnorm=batchnorm,
                                input_dropout=xdrop,
                                output_dropout=edrop,
                                seed=self.seed,
                                name='Encoder')
    self.decoder = DenseNetwork(n_units=hdim,
                                n_layers=n_layers,
                                activation='relu',
                                batchnorm=batchnorm,
                                input_dropout=zdrop,
                                output_dropout=ddrop,
                                seed=self.seed,
                                name='Decoder')
    self.latent_layer = Dense(zdim,
                              use_bias=bool(biased_latent),
                              activation='linear')
    self.output_layer = None

  @property
  def is_semi_supervised(self):
    return False

  def call(self, inputs, training=None, n_samples=None):
    # initialization
    if self.output_layer is None:
      n_units = inputs.shape[1]
      self.output_layer = Dense(n_units, activation='relu')

    # applying the layers
    e = self.encoder(inputs, training=training)
    z = self.latent_layer(e)
    d = self.decoder(z, training=training)
    pred = self.output_layer(d)

    # NOTE: TensorArray could be used to remove the redundancy of
    # loss computation during prediction, however, Tensorflow Autograph
    # cannot track the gradient here.
    # loss = tf.TensorArray(dtype='float32', size=1, dynamic_size=False)
    # loss.write(0, tf.reduce_mean(tf.losses.mean_squared_error(inputs, pred)))
    # self.add_loss(lambda: loss.read(0))

    # calculating the losses
    loss = tf.losses.mean_squared_error(inputs, pred)
    loss = tf.reduce_mean(loss)
    if training:
      self.add_loss(lambda: loss)

    return pred, z


class MultitaskEncoder(Autoencoder):
  """
  """

  @property
  def is_semi_supervised(self):
    return True

  def call(self, inputs, training=None, n_samples=None):
    x, y, masks = self._to_semisupervised_inputs(inputs)

    # initialization
    if self.output_layer is None:
      self.output_layer = Parallel(
          [Dense(i.shape[1], activation='relu') for i in [x] + y],
          name="ParallelOutputs")

    # applying the layers
    e = self.encoder(inputs[0], training=training)
    z = self.latent_layer(e)
    d = self.decoder(z, training=training)
    outputs = self.output_layer(d)
    x_pred = outputs[0]
    y_pred = outputs[1:]

    # calculating the losses
    loss = tf.losses.mean_squared_error(x, x_pred)
    for i, i_, m in zip(y, y_pred, masks):
      loss += tf.losses.mean_squared_error(i, i_) * m
    loss = tf.reduce_mean(loss)
    if training:
      self.add_loss(lambda: loss)

    return outputs, z
