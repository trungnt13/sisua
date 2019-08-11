from __future__ import absolute_import, division, print_function

from typing import Iterable, List, Text, Union

import tensorflow as tf
from tensorflow.python.keras.layers import Layer

from odin.backend.layers import Parallel
from odin.bay import DistributionLayer, Statistic
from odin.bay.distribution_alias import parse_distribution
from odin.utils import as_tuple
from sisua.models.base import SingleCellModel
from sisua.models.variational_autoencoder import VariationalAutoEncoder


class MultiTaskVAE(VariationalAutoEncoder):
  """ Variational autoencoder with multitask learning extension
  """

  def __init__(self,
               dispersion: Text = 'gene-cell',
               xdist: Text = 'zinb',
               ydist: Union[Text, List[Text]] = 'nb',
               zdist: Text = 'normal',
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
    kw = dict(locals())
    del kw['self']
    del kw['kwargs']
    del kw['ydist']
    del kw['__class__']
    kw.update(kwargs)
    super(MultiTaskVAE, self).__init__(**kw)
    if not isinstance(ydist, (tuple, list)):
      ydist = [ydist]
    self.ydist = [parse_distribution(y)[0] for y in ydist]

  @property
  def is_semi_supervised(self):
    return True

  def call(self, inputs, training=None, n_samples=1):
    # check arguments
    if n_samples is None:
      n_samples = 1
    x, y, masks = self._to_semisupervised_inputs(inputs)
    pX, qZ, e, d = self._apply_network(x, training, n_samples)
    if not isinstance(self.ydist, Layer):
      self.ydist = as_tuple(self.ydist, N=len(y))
      self.ydist = Parallel([
          DistributionLayer(i.shape[1], posterior=d)
          for d, i in zip(self.ydist, y)
      ],
                            name='ydist')
    pY = self.ydist(d, mode=Statistic.DIST)

    # calculating the losses
    kl = self.latent.kl_divergence(analytic_kl=self.kl_analytic,
                                   n_samples=n_samples)
    llk_x = tf.expand_dims(pX.log_prob(x), -1)
    llk_y = 0
    for p, i, m in zip(pY, y, masks):
      llk_y += tf.expand_dims(p.log_prob(i), -1) * m
    elbo = llk_x + llk_y - kl * self.kl_weight
    elbo = tf.reduce_logsumexp(elbo, axis=0)
    loss = tf.reduce_mean(-elbo)
    if training:
      self.add_loss(lambda: loss)

    # NOTE: add_metric should not be in control if-then-else
    self.add_metric(tf.reduce_mean(kl), aggregation='mean', name="KLqp")
    self.add_metric(tf.reduce_mean(-llk_x), aggregation='mean', name="NLLK_X")
    self.add_metric(tf.reduce_mean(-llk_y), aggregation='mean', name="NLLK_Y")
    return (pX,) + pY, qZ


class MultiLatentVAE(VariationalAutoEncoder):
  # TODO
  pass
