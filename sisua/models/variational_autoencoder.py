from __future__ import absolute_import, division, print_function

import inspect

import tensorflow as tf
from tensorflow.python.keras.layers import Layer

from odin.bay import Statistic, kl_divergence, print_dist
from odin.bay.distribution_alias import parse_distribution
from odin.networks import DistributionDense, Identity
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
    super(VariationalAutoEncoder, self).__init__(parameters=locals(), **kwargs)
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
    # multiple inputs could be provided
    if not isinstance(xdist, (tuple, list)):
      xdist = [xdist]
    self.xdist = [parse_distribution(i)[0] for i in xdist]
    self.dispersion = dispersion

  @property
  def is_semi_supervised(self):
    return len(self.xdist) > 1

  def _call(self, x, y, masks, training, n_samples):
    # initializing the output layers
    if not isinstance(self.xdist[0], Layer):
      dist = []
      for idx, (i, d) in enumerate(zip([x] + y, self.xdist)):
        if 'dispersion' in inspect.getfullargspec(d.__init__).args:
          post = d(i.shape[1], dispersion=self.dispersion)
        else:
          post = d(i.shape[1])
        dist.append(
            DistributionDense(i.shape[1],
                              posterior=post,
                              activation='linear',
                              use_bias=True,
                              name='Output%d' % idx))
      self.xdist = dist

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
    pX = [dist(d, mode=Statistic.DIST) for dist in self.xdist]

    # calculating the losses
    kl = self.latent.kl_divergence(analytic_kl=self.kl_analytic,
                                   n_samples=n_samples)
    llk_x = tf.expand_dims(pX[0].log_prob(x), -1)
    llk_y = 0
    for i, m, dist in zip(y, masks, pX[1:]):
      llk_y += tf.expand_dims(dist.log_prob(i), -1) * m

    elbo = llk_x + llk_y - kl * self.kl_weight
    elbo = tf.reduce_logsumexp(elbo, axis=0)
    loss = tf.reduce_mean(-elbo)

    if training:
      self.add_loss(lambda: loss)

    # NOTE: add_metric should not be in control if-then-else
    self.add_metric(tf.reduce_mean(kl), aggregation='mean', name="KLqp")
    self.add_metric(tf.reduce_mean(-llk_x), aggregation='mean', name="nllk_x")
    if self.is_semi_supervised:
      self.add_metric(tf.reduce_mean(-llk_y), aggregation='mean', name="nllk_y")
    return pX if self.is_semi_supervised else pX[0], qZ
