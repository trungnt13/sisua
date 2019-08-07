from __future__ import absolute_import, division, print_function

import tensorflow as tf

from odin.bay import Statistic, kl_divergence, print_dist
from odin.bay.distribution_alias import parse_distribution
from odin.bay.distribution_util_layers import DistributionLayer
from sisua.models.base import SingleCellModel
from sisua.models.latents import get_latent
from sisua.models.networks import DenseNetwork


class VariationalAutoencoder(SingleCellModel):
  """
  """

  def __init__(self,
               dispersion='gene-cell',
               xdist='zinb',
               zdist='normal',
               xdrop=0.3,
               edrop=0,
               zdrop=0,
               ddrop=0,
               hdim=128,
               zdim=32,
               n_layers=2,
               batchnorm=True,
               name='VariationalAutoencoder',
               **kwargs):
    super(VariationalAutoencoder, self).__init__(name=name, **kwargs)
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
    self.latent = get_latent(zdist)(n_units=zdim, name=name + 'Latent')
    self.output_dist = parse_distribution(xdist)[0]

  @property
  def is_semi_supervised(self):
    return False

  def call(self, inputs, training=None, n_samples=1):
    # check arguments
    if n_samples is None:
      n_samples = 1

    # initialization
    if not isinstance(self.output_dist, DistributionLayer):
      n_units = inputs.shape[1]
      self.output_dist = DistributionLayer(n_units,
                                           posterior=self.output_dist,
                                           use_bias=True,
                                           name=self.name + 'Output')

    # applying the layers
    e = self.encoder(inputs, training=training)
    qZ = self.latent(e,
                     training=training,
                     n_samples=n_samples,
                     mode=Statistic.DIST)
    d = self.decoder(qZ.sample(n_samples), training=training)
    pX = self.output_dist(d, mode=Statistic.DIST)

    # calculating the losses
    kl = self.latent.kl_divergence(analytic_kl=self.kl_analytic,
                                   n_samples=n_samples)
    llk = tf.expand_dims(pX.log_prob(inputs), -1)
    kl_weight = self.kl_weight
    elbo = llk - kl * kl_weight
    elbo = tf.reduce_logsumexp(elbo, axis=0)
    loss = tf.reduce_mean(-elbo)

    if training:
      self.add_loss(lambda: loss)

    # NOTE: add_metric should not be in control if-then-else
    self.add_metric(tf.reduce_mean(kl), aggregation='mean', name="KLqp")
    self.add_metric(tf.reduce_mean(llk), aggregation='mean', name="LLK")
    return pX, qZ
