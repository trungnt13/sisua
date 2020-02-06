from __future__ import absolute_import, division, print_function

from typing import List

import numpy as np
import tensorflow as tf
from tensorflow.python import keras
from tensorflow_probability.python.distributions import Independent, Normal

from odin.bay.layers import (DenseDistribution, NegativeBinomialDispLayer,
                             ZINegativeBinomialDispLayer)
from odin.networks import DenseNetwork, Identity
from sisua.models.base import SingleCellModel
from sisua.models.utils import NetworkConfig, RandomVariable


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

  def __init__(self,
               outputs: List[RandomVariable],
               latent_dim=10,
               network=NetworkConfig(),
               clip_library=1e4,
               **kwargs):
    latents = kwargs.pop('latents', None)
    if latents is None:
      latents = [
          RandomVariable(latent_dim, 'diag', 'latent'),
          RandomVariable(1, 'gaus', 'library')
      ]
    super().__init__(outputs=outputs,
                     latents=latents,
                     network=network,
                     **kwargs)
    self.encoder_l = DenseNetwork(
        units=tf.nest.flatten(self.network_config.hidden_dim)[0],
        activation=self.network_config.activation,
        batchnorm=self.network_config.batchnorm,
        input_dropout=self.network_config.input_dropout,
        output_dropout=self.network_config.encoder_dropout,
        name='EncoderL')
    self.latent = self.latents[0]
    self.library = self.latents[1]
    self.clip_library = float(clip_library)
    n_dims = self.posteriors[0].event_shape[0]
    # mean gamma (logits value, applying softmax later)
    self.px_scale = keras.layers.Dense(units=n_dims,
                                       activation='linear',
                                       name="MeanScale")
    # dropout logits value
    if self.is_zero_inflated:
      self.px_dropout = keras.layers.Dense(n_dims,
                                           activation='linear',
                                           name="DropoutLogits")
    else:
      self.px_dropout = Identity(name="DropoutLogits")
    # dispersion (NOTE: while this is different implementation, it ensures the
    # same method as scVI, i.e. cell-gene, gene dispersion)
    self.px_r = keras.layers.Dense(n_dims,
                                   activation='linear',
                                   name='Dispersion')
    # since we feed the params directly, the DenseDistribution parameters won't
    # be used
    self.posteriors[0].trainable = False

  def encode(self, x, lmean=None, lvar=None, y=None, training=None, n_mcmc=1):
    # applying encoding
    e_z = self.encoder(x, training=training)
    e_l = self.encoder_l(x, training=training)
    # latent space
    qZ = self.latent(e_z, training=training, n_mcmc=n_mcmc)
    # library space
    if lmean is None or lvar is None:
      pL = None
    else:
      pL = Independent(Normal(loc=lmean, scale=tf.math.sqrt(lvar)), 1)
    qL = self.library(e_l, training=training, n_mcmc=n_mcmc, prior=pL)
    return qZ, qL

  def decode(self, latents, training=None):
    qZ, qL = latents
    Z_samples = qZ
    # clipping L value to avoid overflow, softplus(12) = 12
    L_samples = tf.clip_by_value(qL, 0., self.clip_library)
    # decoding the latent
    d = self.decoder(Z_samples, training=training)
    # ====== parameterizing the distribution ====== #
    # mean parameterizations
    px_scale = tf.nn.softmax(self.px_scale(d), axis=1)
    px_scale = tf.clip_by_value(px_scale, 1e-8, 1. - 1e-8)
    # NOTE: scVI use exp but we use softplus here
    px_rate = tf.nn.softplus(L_samples) * px_scale
    # dispersion parameterizations
    px_r = self.px_r(d)
    # NOTE: scVI use exp but we use softplus here
    px_r = tf.nn.softplus(px_r)
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
