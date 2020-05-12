from __future__ import absolute_import, division, print_function

from typing import List

import numpy as np
import tensorflow as tf
from tensorflow.python import keras
from tensorflow_probability.python.distributions import Independent, Normal

from odin.bay.layers import (DenseDistribution, NegativeBinomialDispLayer,
                             ZINegativeBinomialDispLayer)
from odin.networks import Identity
from sisua.models.single_cell_model import (NetworkConfig, RandomVariable,
                                            SingleCellModel)


class SCVI(SingleCellModel):
  r""" Re-implementation of single cell variational inference (scVI) in
  Tensorflow

  Arguments:
    clip_library : `float` (default=`10000`)
      clipping the maximum library size to prevent overflow in exponential,
      e.g. if L=10 then the maximum library value is softplus(10)=~10

  References:
    Romain Lopez (2018). https://github.com/YosefLab/scVI/tree/master/scvi.

  """

  def __init__(
      self,
      outputs,
      latents=RandomVariable(10, 'diag', True, "Latents"),
      library=RandomVariable(1, 'normal', True, "Library"),
      encoder=NetworkConfig([64, 64],
                            batchnorm=True,
                            dropout=0.1,
                            name='Encoder'),
      encoder_l=NetworkConfig([64],
                              batchnorm=True,
                              dropout=0.1,
                              name='EncoderL'),
      clip_library=1e4,
      **kwargs,
  ):
    outputs = tf.nest.flatten(outputs)
    assert outputs[0].posterior in ('zinbd', 'nbd'), \
      "scVI only support transcriptomic distribution: 'zinbd' or 'nbd', " + \
        "but given: %s" % str(outputs)
    outputs[0].projection = False
    super().__init__(outputs,
                     latents=[latents, library],
                     encoder=[encoder, encoder_l],
                     reduce_latent=lambda Zs: Zs[0],
                     **kwargs)
    ### prepare the library
    self.clip_library = float(clip_library)
    n_dims = self.posteriors[0].event_shape[0]
    # decoder outputs
    decoder_output = self.decoder.outputs[0]
    # mean gamma (logits value, applying softmax later)
    self.px_scale = keras.layers.Dense(units=n_dims,
                                       activation='linear',
                                       name="MeanScale")
    self.px_scale(decoder_output)  # build the layer
    # dropout logits value
    if self.is_zero_inflated:
      self.px_dropout = keras.layers.Dense(n_dims,
                                           activation='linear',
                                           name="DropoutLogits")
      self.px_dropout(decoder_output)  # build the layer
    else:
      self.px_dropout = Identity(name="DropoutLogits")
    # dispersion (NOTE: while this is different implementation, it ensures the
    # same method as scVI, i.e. cell-gene, gene dispersion)
    self.px_r = keras.layers.Dense(n_dims,
                                   activation='linear',
                                   name='Dispersion')
    self.px_r(decoder_output)  # build the layer
    # since we feed the params directly, the DenseDistribution parameters won't
    # be used
    self.posteriors[0].trainable = False

  def encode(self,
             inputs,
             library=None,
             training=None,
             mask=None,
             sample_shape=()):
    qZ_X = super().encode(inputs=inputs,
                          library=library,
                          training=training,
                          mask=mask,
                          sample_shape=sample_shape)
    if library is not None:
      mean, var = tf.split(tf.nest.flatten(library)[0], 2, axis=1)
      pL = Independent(Normal(loc=mean, scale=tf.math.sqrt(var)), 1)
    else:
      pL = None
    qZ_X[-1].KL_divergence.prior = pL
    return qZ_X

  def decode(self, latents, training=None, mask=None, sample_shape=()):
    qZ, qL = latents
    Z_samples = qZ
    # clipping L value to avoid overflow, softplus(12) = 12
    L_samples = tf.clip_by_value(qL, 0., self.clip_library)
    # decoding the latent
    d = self.decoder(Z_samples, training=training)
    # ====== parameterizing the distribution ====== #
    # mean parameterizations
    px_scale = tf.nn.softmax(self.px_scale(d), axis=1)
    px_scale = tf.clip_by_value(px_scale, 1e-7, 1. - 1e-7)
    # NOTE: scVI use exp but we use softplus here
    px_rate = tf.exp(L_samples) * px_scale
    # dispersion parameterizations
    px_r = self.px_r(d)
    # NOTE: scVI use exp but we use softplus here
    px_r = tf.exp(px_r)
    # dropout for zero inflation
    px_dropout = self.px_dropout(d)
    # mRNA expression distribution
    # this order is the same as how the parameters are splited in distribution
    # layer
    if self.is_zero_inflated:
      params = tf.concat((px_rate, px_r, px_dropout), axis=-1)
    else:
      params = tf.concat((px_rate, px_r), axis=-1)
    pX = self.posteriors[0](params, training=training, projection=False)
    # for semi-supervised learning
    pY = [p(d, training=training) for p in self.posteriors[1:]]
    return [pX] + pY
