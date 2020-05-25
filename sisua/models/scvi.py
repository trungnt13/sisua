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

__all__ = ['SCVI']


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
      clip_library=1e3,
      **kwargs,
  ):
    outputs = tf.nest.flatten(outputs)
    assert outputs[0].posterior in ('zinbd', 'nbd'), \
      "scVI only support transcriptomic distribution: 'zinbd' or 'nbd', " + \
        "but given: %s" % str(outputs)
    outputs[0].projection = False
    kwargs['reduce_latent'] = lambda Zs: Zs[0]
    self.dispersion = outputs[0].kwargs.get('dispersion', 'full')
    self.inflation = outputs[0].kwargs.get('inflation', 'full')
    super().__init__(outputs,
                     latents=[latents, library],
                     encoder=[encoder, encoder_l],
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
    if self.is_zero_inflated and self.inflation == 'full':
      self.px_dropout = keras.layers.Dense(n_dims,
                                           activation='linear',
                                           name="DropoutLogits")
      self.px_dropout(decoder_output)  # build the layer
    # dispersion (NOTE: while this is different implementation, it ensures the
    # same method as scVI, i.e. cell-gene, gene dispersion)
    if self.dispersion == 'full':
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
             sample_shape=(),
             **kwargs):
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

  def decode(self,
             latents,
             training=None,
             mask=None,
             sample_shape=(),
             **kwargs):
    qZ, qL = latents
    Z_samples = qZ
    # clipping L value to avoid overflow, softplus(12) = 12
    L_samples = tf.clip_by_value(qL, 0., self.clip_library)
    if sample_shape:
      output_shape = tf.concat(
          [tf.nest.flatten(sample_shape),
           tf.convert_to_tensor(qZ.batch_shape)],
          axis=0)
      Z_samples = tf.reshape(qZ, (-1, qZ.shape[-1]))
      L_samples = tf.reshape(tf.clip_by_value(qL, 0., self.clip_library),
                             (-1, 1))
    # decoding the latent
    d = self.decoder(Z_samples, training=training)
    # ====== parameterizing the distribution ====== #
    # mean parameterizations
    px_scale = tf.nn.softmax(self.px_scale(d), axis=1)
    px_scale = tf.clip_by_value(px_scale, 1e-7, 1. - 1e-7)
    # NOTE: tried to use softplus1 here but it doesn't work, the model
    # reconstruction loss is extremely high!
    px_rate = tf.exp(L_samples) * px_scale
    # dispersion parameterizations
    if self.dispersion == 'full':
      px_r = self.px_r(d)
      px_r = tf.exp(px_r)
    # recover the sample shape
    if sample_shape:
      shape = tf.concat(
          [output_shape, tf.convert_to_tensor(px_rate.shape[1:])], axis=0)
      px_rate = tf.reshape(px_rate, shape)
      if self.dispersion == 'full':
        shape = tf.concat(
            [output_shape, tf.convert_to_tensor(px_r.shape[1:])], axis=0)
        px_r = tf.reshape(px_r, shape)
    # mRNA expression distribution
    # this order is the same as how the parameters are splited in distribution
    # layer
    params = [px_rate, px_r] if self.dispersion == 'full' else [px_rate]
    if self.is_zero_inflated and self.inflation == 'full':
      # dropout for zero inflation
      px_dropout = self.px_dropout(d)
      if sample_shape:
        shape = tf.concat(
            [output_shape,
             tf.convert_to_tensor(px_dropout.shape[1:])], axis=0)
        px_dropout = tf.reshape(px_dropout, shape)
      params.append(px_dropout)
      params = tf.concat(params, axis=-1)
    else:
      params = tf.concat(params, axis=-1) if len(params) > 1 else params[0]
    pX = self.posteriors[0](params, training=training, projection=False)
    # for semi-supervised learning
    if sample_shape:
      shape = tf.concat(
          [output_shape, tf.convert_to_tensor(d.shape[1:])], axis=0)
      d = tf.reshape(d, shape)
    pY = [p(d, training=training) for p in self.posteriors[1:]]
    return [pX] + pY


class TotalVI(SingleCellModel):
  pass
