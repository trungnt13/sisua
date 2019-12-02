from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow.python.keras import Model, Sequential
from tensorflow.python.keras.layers import (Activation, BatchNormalization,
                                            Conv1D, Dense, Dropout, Flatten,
                                            Input, Lambda)

from odin import backend as bk
from odin.networks import Conv1DTranspose

__all__ = ['DenseNetwork', 'ConvNetwork']


class DenseNetwork(Sequential):
  r""" Multi-layer fully connected neural network """

  def __init__(self,
               units=128,
               nlayers=2,
               activation='relu',
               batchnorm=True,
               input_dropout=0.,
               output_dropout=0.,
               layer_dropout=0.,
               pyramid=False,
               decoding=False,
               seed=8,
               input_dim=None,
               name=None):
    nlayers = int(nlayers)
    layers = []
    if pyramid:
      assert units % (2**nlayers) == 0
    if input_dim is not None:
      layers.append(Input(shape=(int(input_dim),)))
    if 0. < input_dropout < 1.:
      layers.append(Dropout(input_dropout, seed=seed))
    for i in range(nlayers):
      n = (units // 2**(nlayers - (i + 1)) \
            if decoding else \
              units // 2**i) \
        if pyramid else units
      layers.append(
          Dense(n,
                activation='linear',
                use_bias=False if batchnorm else True,
                name="DenseLayer%d" % i))
      if batchnorm:
        layers.append(BatchNormalization())
      layers.append(Activation(activation))
      if layer_dropout > 0 and i != nlayers - 1:
        layers.append(Dropout(rate=layer_dropout))
    if 0. < output_dropout < 1.:
      layers.append(Dropout(output_dropout, seed=seed))
    super().__init__(layers=layers, name=name)


def _deconv_reshape(x, units):
  shape = tf.shape(x)
  if len(x.shape) > 2:  # remove the MCMC dimensions
    x = tf.reshape(x, (-1, x.shape[-1]))
  x = tf.reshape(x, (-1, x.shape[-1] // units, units))
  return x


class ConvNetwork(Sequential):
  r""" Multi-layer convolutional neural network """

  def __init__(self,
               units=64,
               kernel=5,
               nlayers=2,
               activation='relu',
               batchnorm=True,
               input_dropout=0.,
               output_dropout=0.,
               layer_dropout=0.,
               pyramid=False,
               decoding=False,
               seed=8,
               input_dim=None,
               name=None):
    nlayers = int(nlayers)
    layers = []
    conv_kw = dict(
        kernel_size=int(kernel),
        strides=2,
        padding='same',
        data_format='channels_last',
        dilation_rate=1,
        activation='linear',
        use_bias=False if batchnorm else True,
    )
    if input_dim is not None:
      if decoding:
        layers.append(
            Dense(units=int(input_dim), use_bias=False, activation='linear'))
      else:
        layers.append(Input(shape=(int(input_dim),)))
    if decoding:
      layers.append(Lambda(_deconv_reshape, arguments={'units': units}))
    else:
      layers.append(Lambda(tf.expand_dims, arguments={'axis': -1}))
    if 0. < input_dropout < 1.:
      layers.append(Dropout(input_dropout, seed=seed))
    if pyramid:
      assert units % (2**nlayers) == 0
    for i in range(nlayers):
      n = (units // 2**(i + 1) \
            if decoding else \
              units // 2**(nlayers - (i + 1))) \
        if pyramid else units
      layers.append(
            Conv1DTranspose(1 if i == (nlayers -1) else n, **conv_kw) \
            if decoding else \
            Conv1D(n, **conv_kw))
      if batchnorm:
        layers.append(BatchNormalization())
      layers.append(Activation(activation))
      if layer_dropout > 0 and i != nlayers - 1:
        layers.append(Dropout(rate=layer_dropout))
    # flatten to 2D
    layers.append(Flatten())
    if 0. < output_dropout < 1.:
      layers.append(Dropout(output_dropout, seed=seed))
    super().__init__(layers=layers, name=name)
    self.decoding = decoding

  def call(self, inputs, training=None, mask=None):
    outputs = super().call(inputs, training, mask)
    if self.decoding and len(inputs.shape) > 2:
      outputs = tf.reshape(outputs, (inputs.shape[0], -1, outputs.shape[-1]))
    return outputs
