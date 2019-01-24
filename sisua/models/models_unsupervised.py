# Basic unsupervised Autoencoder
from __future__ import print_function, division, absolute_import

import numpy as np
import tensorflow as tf

from odin import nnet as N, backend as K

from sisua.models.helpers import parse_loss_function, create_network

@N.Lambda
def unsupervised(X, T, C, mask, y, nsample, kwargs):
  """ Dropout out the latent as well """
  f_encoder, f_decoder = create_network(
      kwargs['hdim'], batchnorm=kwargs['batchnorm'],
      n_layers=kwargs['nlayer'],
      xdrop=kwargs['xdrop'], zdrop=kwargs['zdrop'],
      edrop=kwargs['edrop'], ddrop=kwargs['ddrop'],
      name='unsupervised')
  f_latent = N.Dense(num_units=kwargs['zdim'], b_init=None,
                     activation=K.linear, name="Bottleneck")
  f_rec = N.Dense(num_units=X.shape.as_list()[-1],
                  activation=K.linear)

  E = f_encoder(X)
  Z = f_latent(E)
  D = f_decoder(Z)
  # ====== create loss ====== #
  W, loss = parse_loss_function(true=T, logit=f_rec(D), mask=None,
                                loss_name=kwargs['rec_loss'])
  loss = tf.identity(loss, name='reconstruction_%s' % kwargs['rec_loss'].lower())
  # ====== return ====== #
  return {'W': W, 'Z': Z, 'loss': loss, 'metr': loss}