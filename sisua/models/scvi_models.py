# Tensorflow re-implementaiton of: https://github.com/YosefLab/scVI/tree/master/scvi
# MIT License
# Copyright (c) 2018, Romain Lopez
# Modification by Trung Ngo 2019
from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers import Dense
from tensorflow_probability.python.distributions import Normal

from odin.bay.distribution_layers import (NegativeBinomialDispLayer,
                                          ZINegativeBinomialDispLayer)
from odin.bay.helpers import Statistic
from odin.networks import DenseDistribution, Identity
from sisua.models.base import SingleCellModel
from sisua.models.latents import get_latent
from sisua.models.networks import DenseNetwork


class SCVI(SingleCellModel):
  """ Re-implementation of single cell variational inference in tensorflow

  Parameters
  ----------
  dispersion : {'full', 'share', 'single'}
    'full' is 'gene-cell' mode in scVI - dispersion can differe for every gene
    in every cell
    'share' is 'gene' mode in scVI - dispersion is constant per gene across
    cells
    'single' - single value dispersion for all genes and cells (only in SISUA)
  clip_library : `float` (default=`10`)
    clipping the maximum library size to prevent overflow in exponential,
    e.g. if L=10 then the maximum library value is exp(10)=~22000
  """

  def __init__(self,
               units,
               dispersion='full',
               xdist='zinbd',
               zdist='normal',
               ldist='normal',
               xdrop=0.3,
               edrop=0,
               zdrop=0,
               ddrop=0,
               hdim=128,
               zdim=32,
               nlayers=2,
               clip_library=12,
               batchnorm=True,
               linear_decoder=False,
               **kwargs):
    super(SCVI, self).__init__(units=units,
                               xdist=xdist,
                               dispersion=dispersion,
                               parameters=locals(),
                               **kwargs)
    assert self.xdist[0] in (NegativeBinomialDispLayer,
                             ZINegativeBinomialDispLayer), \
      "scVI only suppoert 'nbd' and 'zinbd' as output distribution, but given:"+\
        "%s" % str(self.xdist[0])
    # initialize the autoencoder
    self.encoder_z = DenseNetwork(n_units=hdim,
                                  nlayers=nlayers,
                                  activation='relu',
                                  batchnorm=batchnorm,
                                  input_dropout=xdrop,
                                  output_dropout=edrop,
                                  seed=self.seed,
                                  name='EncoderZ')
    self.encoder_l = DenseNetwork(n_units=1,
                                  nlayers=1,
                                  activation='relu',
                                  batchnorm=batchnorm,
                                  input_dropout=xdrop,
                                  output_dropout=edrop,
                                  seed=self.seed,
                                  name='EncoderL')

    if linear_decoder:
      self.decoder = Identity(name='LinearDecoder')
    else:
      self.decoder = DenseNetwork(n_units=hdim,
                                  nlayers=nlayers,
                                  activation='relu',
                                  batchnorm=batchnorm,
                                  input_dropout=zdrop,
                                  output_dropout=ddrop,
                                  seed=self.seed,
                                  name='Decoder')
    self.latent = get_latent(zdist)(units=zdim, name=self.name + 'Latent')
    self.library = get_latent(ldist)(units=1, name=self.name + 'Library')
    self.clip_library = float(clip_library)
    n_dims = self.units[0]
    # mean gamma (logits value, applying softmax later)
    self.px_scale = Dense(units=n_dims, activation='linear', name="MeanScale")
    # dropout logits value
    if self.is_zero_inflated:
      self.px_dropout = Dense(n_dims, activation='linear', name="DropoutLogits")
    else:
      self.px_dropout = Identity(name="DropoutLogits")
    # dispersion
    if self.dispersion == 'full':
      self.px_r = Dense(n_dims, activation='linear', name='Dispersion')
    elif self.dispersion == 'share':
      self.px_r = self.add_weight(name='Dispersion',
                                  shape=(1, n_dims),
                                  initializer=tf.initializers.RandomNormal,
                                  trainable=True)
    else:
      self.px_r = self.add_weight(name='Dispersion',
                                  shape=(),
                                  initializer=tf.initializers.RandomNormal,
                                  trainable=True)
    # mRNA distribution
    self.pX = self.xdist[0](n_dims, given_log_mean=False, given_log_disp=False)
    # semi-supervised distributions
    for idx, (units, posterior, activation) in enumerate(
        zip(self.units[1:], self.xdist[1:], self.xactiv[1:])):
      post = DenseDistribution(
          units=units,
          posterior=posterior,
          activation=activation,
          posterior_kwargs=dict(dispersion=self.dispersion))
      setattr(self, 'output_layer%d' % idx, post)

  def _call(self, x, lmean, lvar, t, y, masks, training, n_samples):
    # applying encoding
    e_z = self.encoder_z(x, training=training)
    e_l = self.encoder_l(x, training=training)

    # latent spaces
    qZ = self.latent(e_z, mode=Statistic.DIST)
    qL = self.library(e_l, mode=Statistic.DIST)
    Z_samples = qZ.sample(n_samples)
    # clipping L value to avoid overflow, exp(12) = ~160000
    L_samples = qL.sample(n_samples)
    L_samples = tf.clip_by_value(L_samples, 0, self.clip_library)
    # decoding the latent
    d = self.decoder(Z_samples, training=training)

    # mean parameterizations
    px_scale = tf.nn.softmax(self.px_scale(d), axis=1)
    px_scale = tf.clip_by_value(px_scale, 0 + 1e-8, 1 - 1e-8)
    px_rate = tf.math.exp(L_samples) * px_scale
    # dispersion parameterizations
    if self.dispersion == 'full':
      px_r = self.px_r(d)
    else:
      px_r = self.px_r
    px_r = tf.math.exp(px_r)
    # dropout for zero inflation
    px_dropout = self.px_dropout(d)

    # mRNA expression distribution
    # this order is the same as how the parameters are splited in
    # distribution layer
    if self.is_zero_inflated:
      pX = self.pX(tf.concat((px_rate, px_r, px_dropout), axis=-1))
    else:
      pX = self.pX(tf.concat((px_rate, px_r), axis=-1))

    # for semi-supervised learning
    pY = []
    for idx in range(self.n_outputs - 1):
      name = 'output_layer%d' % idx
      post = getattr(self, name)
      pY.append(post(d, mode=Statistic.DIST))

    # log like-lihood
    llk_x = tf.expand_dims(pX.log_prob(t), -1)
    # semi-supervised NLLK
    loss_y = tf.convert_to_tensor(0, dtype=x.dtype)
    for i_true, m, i_pred, fn_loss in zip(y, masks, pY, self.xloss[1:]):
      loss_y += fn_loss(i_true, i_pred) * m

    # KL for both latent and library
    kl_z = self.latent.kl_divergence(analytic_kl=self.kl_analytic,
                                     n_samples=n_samples)
    kl_l = self.library.kl_divergence(analytic_kl=self.kl_analytic,
                                      n_samples=n_samples,
                                      prior=Normal(loc=lmean,
                                                   scale=tf.math.sqrt(lvar)))
    # for semi-superivsed, we have the negative log-likelihood,
    # hence the minus sign
    elbo = llk_x - loss_y - kl_l - kl_z * self.kl_weight
    elbo = tf.reduce_logsumexp(elbo, axis=0)
    loss = tf.reduce_mean(-elbo)
    if training:
      self.add_loss(loss)

    # NOTE: add_metric should not be in control if-then-else
    self.add_metric(tf.reduce_max(L_samples), aggregation='mean', name="Lmax")
    self.add_metric(tf.reduce_mean(kl_z), aggregation='mean', name="KLqpZ")
    self.add_metric(tf.reduce_mean(kl_l), aggregation='mean', name="KLqpL")
    self.add_metric(tf.reduce_mean(-llk_x), aggregation='mean', name="nllk_x")
    if self.is_semi_supervised:
      self.add_metric(tf.reduce_mean(loss_y), aggregation='mean', name="nllk_y")
    return [pX] + pY, (qZ, qL)
