from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers import Dense
from tensorflow_probability.python.distributions import Normal

from odin.bay.distribution_layers import (NegativeBinomialDispLayer,
                                          ZINegativeBinomialDispLayer)
from odin.bay.helpers import Statistic
from odin.networks import Identity
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
    super(SCVI, self).__init__(xdist=xdist,
                               dispersion=dispersion,
                               parameters=locals(),
                               **kwargs)
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
    # mean gamma
    self.px_scale = None
    # dispersion
    self.px_r = None
    # dropout
    self.px_dropout = None
    self.clip_library = float(clip_library)

  def _initialize_layers(self, n_dims):
    if self.px_scale is not None:
      return
    self.px_scale = Dense(units=n_dims, activation='linear', name="MeanScale")

    if self.xdist == ZINegativeBinomialDispLayer:
      self.px_dropout = Dense(n_dims, activation='linear', name="Dropout")
    else:
      self.px_dropout = Identity(name="NoDropout")

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

  def _call(self, x, lmean, lvar, t, y, masks, training, n_samples):
    n_dims = x.shape[1]
    self._initialize_layers(n_dims)

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
    px_rate = tf.math.exp(L_samples) * px_scale
    # dispersion parameterizations
    px_r = self.px_r(d) if self.dispersion == 'full' else self.px_r
    px_r = tf.math.exp(px_r)
    # dropout for zero inflation
    px_dropout = self.px_dropout(d)

    exit()
    pX = self.xdist(event_shape=n_dims,
                    given_log_mean=False,
                    given_log_disp=False)
    if self.xdist == NegativeBinomialDispLayer:
      pX = pX((px_rate, px_r))
    else:
      pX = pX((px_rate, px_r, px_dropout))

    llk_x = tf.expand_dims(pX.log_prob(t), -1)
    kl_z = self.latent.kl_divergence(analytic_kl=self.kl_analytic,
                                     n_samples=n_samples)
    kl_l = self.library.kl_divergence(analytic_kl=self.kl_analytic,
                                      n_samples=n_samples,
                                      prior=Normal(loc=lmean,
                                                   scale=tf.math.sqrt(lvar)))
    elbo = llk_x - kl_l - kl_z * self.kl_weight
    elbo = tf.reduce_logsumexp(elbo, axis=0)
    loss = tf.reduce_mean(-elbo)
    if training:
      self.add_loss(loss)

    # NOTE: add_metric should not be in control if-then-else
    self.add_metric(tf.reduce_max(L_samples), aggregation='mean', name="Lmax")
    self.add_metric(tf.reduce_mean(kl_z), aggregation='mean', name="KLqpZ")
    self.add_metric(tf.reduce_mean(kl_l), aggregation='mean', name="KLqpL")
    self.add_metric(tf.reduce_mean(-llk_x), aggregation='mean', name="nllk_x")
    return pX, (qZ, qL)
