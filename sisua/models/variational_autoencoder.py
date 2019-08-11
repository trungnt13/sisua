from __future__ import absolute_import, division, print_function

import tensorflow as tf

from odin.backend import Identity
from odin.bay import DistributionLayer, Statistic, kl_divergence, print_dist
from odin.bay.distribution_alias import parse_distribution
from sisua.models.base import SingleCellModel
from sisua.models.latents import get_latent
from sisua.models.networks import DenseNetwork


class VariationalAutoEncoder(SingleCellModel):
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
               linear_decoder=False,
               **kwargs):
    super(VariationalAutoEncoder, self).__init__(**kwargs)
    self.encoder = DenseNetwork(n_units=hdim,
                                n_layers=n_layers,
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
                                  n_layers=n_layers,
                                  activation='relu',
                                  batchnorm=batchnorm,
                                  input_dropout=zdrop,
                                  output_dropout=ddrop,
                                  seed=self.seed,
                                  name='Decoder')
    self.latent = get_latent(zdist)(units=zdim, name=self.name + 'Latent')
    self.xdist = parse_distribution(xdist)[0]

  @property
  def is_semi_supervised(self):
    return False

  def _apply_network(self, x, training, n_samples):
    if not isinstance(self.xdist, DistributionLayer):
      self.xdist = DistributionLayer(x.shape[1],
                                     posterior=self.xdist,
                                     use_bias=True,
                                     name='xdist')
    # applying encoding
    if 'training' in self.encoder._call_fn_args:
      e = self.encoder(x, training=training)
    else:
      e = self.encoder(x)

    # latent distribution
    qZ = self.latent(e,
                     training=training,
                     n_samples=n_samples,
                     mode=Statistic.DIST)

    # decoding the latent
    if 'training' in self.decoder._call_fn_args:
      d = self.decoder(qZ.sample(n_samples), training=training)
    else:
      d = self.decoder(qZ.sample(n_samples))

    # output distribution
    pX = self.xdist(d, mode=Statistic.DIST)
    return pX, qZ, e, d

  def call(self, inputs, training=None, n_samples=1):
    # check arguments
    if n_samples is None:
      n_samples = 1

    # applying the layers
    pX, qZ, e, d = self._apply_network(inputs, training, n_samples)

    # calculating the losses
    kl = self.latent.kl_divergence(analytic_kl=self.kl_analytic,
                                   n_samples=n_samples)
    llk = tf.expand_dims(pX.log_prob(inputs), -1)
    elbo = llk - kl * self.kl_weight
    elbo = tf.reduce_logsumexp(elbo, axis=0)
    loss = tf.reduce_mean(-elbo)

    if training:
      self.add_loss(lambda: loss)

    # NOTE: add_metric should not be in control if-then-else
    self.add_metric(tf.reduce_mean(kl), aggregation='mean', name="KLqp")
    self.add_metric(tf.reduce_mean(-llk), aggregation='mean', name="NLLK")
    return pX, qZ
