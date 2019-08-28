from __future__ import absolute_import, division, print_function

import inspect

import tensorflow as tf
from tensorflow.python.keras.layers import Layer

from odin.bay import Statistic, kl_divergence, print_dist
from odin.networks import DenseDistribution, Identity
from sisua.models.base import SingleCellModel
from sisua.models.latents import get_latent
from sisua.models.networks import DenseNetwork


class VariationalAutoEncoder(SingleCellModel):
  """ Variational Auto Encoder
  """

  def __init__(self,
               dispersion='full',
               xdist='zinb',
               zdist='normal',
               xdrop=0.3,
               edrop=0,
               zdrop=0,
               ddrop=0,
               hdim=128,
               zdim=32,
               nlayers=2,
               batchnorm=True,
               linear_decoder=False,
               **kwargs):
    super(VariationalAutoEncoder, self).__init__(xdist=xdist,
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

  def _call(self, x, lmean, lvar, t, y, masks, training, n_samples):
    # applying encoding
    e = self.encoder(x, training=training)
    # latent distribution
    qZ = self.latent(e,
                     training=training,
                     n_samples=n_samples,
                     mode=Statistic.DIST)
    # decoding the latent
    d = self.decoder(qZ.sample(n_samples), training=training)
    # calculating the KL
    kl = self.latent.kl_divergence(analytic_kl=self.kl_analytic,
                                   n_samples=n_samples)

    # unsupervised output distribution
    # note: loss is negative log-likelihood
    pX = [dist(d, mode=Statistic.DIST) for dist in self.output_layers]
    loss_x = self.xloss[0](t, pX[0])
    # semi-supervised output distribution
    loss_y = 0
    for i_true, m, i_pred, fn_loss in zip(y, masks, pX[1:], self.xloss[1:]):
      loss_y += fn_loss(i_true, i_pred) * m

    # Final ELBO
    elbo = -loss_x - loss_y - kl * self.kl_weight
    elbo = tf.reduce_logsumexp(elbo, axis=0)
    loss = tf.reduce_mean(-elbo)

    if training:
      self.add_loss(loss)
    # NOTE: add_metric should not be in control if-then-else
    self.add_metric(tf.reduce_mean(kl), aggregation='mean', name="KLqp")
    self.add_metric(tf.reduce_mean(loss_x), aggregation='mean', name="nllk_x")
    if self.is_semi_supervised:
      self.add_metric(tf.reduce_mean(loss_y), aggregation='mean', name="nllk_y")

    return pX, qZ
